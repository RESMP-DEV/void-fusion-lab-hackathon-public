from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from best_of_n import (
    BestOfNError,
    main,
    manifest_input_paths,
    overlay,
    write_contact_sheet,
)
from pass2_compare import main as compare_pass2
from PIL import Image

from samurai.best_of_n import (
    CHAIN_MANIFEST_SCHEMA,
    FOREGROUND_WHITE,
    ROI_BLACK,
    plan_from_manifest,
    score_candidate,
    validate_manifest,
)
from samurai.media import decode_gray_frames, write_gray_video

FRAMES = 6
FPS = 12.0
TARGET = (slice(2, 5), slice(2, 5))
STALE_TARGET = (slice(5, 8), slice(5, 8))
EVIDENCE_PATH = Path(__file__).resolve().parents[2] / "evidence/submission-evidence.json"


def _flatten_sum(node: ast.expr) -> list[ast.expr]:
    """Split a summed objective expression into its terms, in source order."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_sum(node.left) + _flatten_sum(node.right)
    return [node]


def shipped_full_chain_objective_terms() -> list[tuple[float, str, bool]]:
    """Read the shipped full-chain objective weights straight out of the product code."""
    source = (Path(__file__).resolve().parents[2] / "src/samurai/best_of_n.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "full_chain_pass2_record"
    )
    assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "objective" for target in node.targets
        )
    )
    terms: list[tuple[float, str, bool]] = []
    for term in _flatten_sum(assignment.value):
        assert isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult)
        weight_node = term.left
        assert isinstance(weight_node, ast.Constant)
        weight_value = weight_node.value
        assert isinstance(weight_value, float)
        weight = weight_value
        value = term.right
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub):
            complement = value.left
            assert isinstance(complement, ast.Constant) and complement.value == 1.0
            metric_node = value.right
            assert isinstance(metric_node, ast.Subscript)
            key_node = metric_node.slice
            assert isinstance(key_node, ast.Constant)
            metric = key_node.value
            assert isinstance(metric, str)
            terms.append((weight, metric, True))
        else:
            assert isinstance(value, ast.Attribute) and value.attr == "objective"
            terms.append((weight, "pre_pass2_objective", False))
    return terms


def plate_frames() -> list[np.ndarray]:
    result = []
    for index in range(FRAMES):
        frame = np.full((8, 8), 20, dtype=np.uint8)
        frame[TARGET] = 20 + index
        result.append(frame)
    return result


def candidate_frames() -> list[np.ndarray]:
    result = []
    for index in range(FRAMES):
        frame = np.full((8, 8), 20, dtype=np.uint8)
        frame[TARGET] = 200 + index
        result.append(frame)
    return result


def target_mask_frames() -> list[np.ndarray]:
    result = []
    for _ in range(FRAMES):
        frame = np.full((8, 8), 255, dtype=np.uint8)
        frame[TARGET] = 0
        result.append(frame)
    return result


def inserted_mask_frames(region: tuple[slice, slice]) -> list[np.ndarray]:
    result = []
    for _ in range(FRAMES):
        frame = np.full((8, 8), 255, dtype=np.uint8)
        frame[region] = 0
        result.append(frame)
    return result


def cleanup_mask_frames(region: tuple[slice, slice]) -> list[np.ndarray]:
    result = []
    for _ in range(FRAMES):
        frame = np.full((8, 8), 255, dtype=np.uint8)
        rows = slice(region[0].start - 1, region[0].stop + 1)
        columns = slice(region[1].start - 1, region[1].stop + 1)
        frame[rows, columns] = 0
        frame[region] = 255
        result.append(frame)
    return result


def pass2_result_frames(*, refine_ring: bool) -> list[np.ndarray]:
    """Build a local Pass2 result that changes only the cleanup ring when asked."""
    result = candidate_frames()
    if refine_ring:
        ring = cleanup_mask_frames(TARGET)[0] < 128
        for frame in result:
            frame[ring] = 120
    return result


def rgb(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Expand the gray fixture frames into the RGB geometry the scorer expects."""
    return [np.repeat(frame[:, :, None], 3, axis=2) for frame in frames]


def write_video(frames: list[np.ndarray], path: Path) -> Path:
    write_gray_video(frames, path, fps=FPS)
    return path


@pytest.fixture
def campaign(tmp_path: Path) -> dict[str, Any]:
    """Build one plate, one target mask, and two candidates that share a video."""
    plate = write_video(plate_frames(), tmp_path / "plate.mp4")
    mask = write_video(target_mask_frames(), tmp_path / "mask.mp4")
    candidate = write_video(candidate_frames(), tmp_path / "candidate.mp4")
    inserted = write_video(inserted_mask_frames(TARGET), tmp_path / "inserted.mp4")
    cleanup = write_video(cleanup_mask_frames(TARGET), tmp_path / "cleanup.mp4")
    stale_inserted = write_video(
        inserted_mask_frames(STALE_TARGET), tmp_path / "stale-inserted.mp4"
    )
    stale_cleanup = write_video(cleanup_mask_frames(STALE_TARGET), tmp_path / "stale-cleanup.mp4")
    return {
        "root": tmp_path,
        "plate": plate,
        "mask": mask,
        "candidate": candidate,
        "inserted": inserted,
        "cleanup": cleanup,
        "stale_inserted": stale_inserted,
        "stale_cleanup": stale_cleanup,
    }


def chain_manifest(campaign: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": CHAIN_MANIFEST_SCHEMA,
        "plate": {"path": str(campaign["plate"])},
        "target_mask": {"path": str(campaign["mask"]), "polarity": "roi_black"},
        "background_prompt": "A plain gray scene",
        "thresholds": {"min_remask_agreement": 0.5},
        "candidates": [
            {
                "id": "good",
                "kind": "fitted_insertion",
                "insertion_video": {"path": str(campaign["candidate"])},
                "inserted_mask": {"path": str(campaign["inserted"])},
                "cleanup_mask": {"path": str(campaign["cleanup"])},
                "cleanup_ring_pixels": 96,
            },
            {
                "id": "stale",
                "kind": "fitted_insertion",
                "insertion_video": {"path": str(campaign["candidate"])},
                "inserted_mask": {"path": str(campaign["stale_inserted"])},
                "cleanup_mask": {"path": str(campaign["stale_cleanup"])},
            },
        ],
    }
    manifest.update(overrides)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def write_pass2_package(
    campaign: dict[str, Any], tmp_path: Path, name: str, *, refine_ring: bool
) -> dict[str, Any]:
    root = tmp_path / "pass2"
    root.mkdir(exist_ok=True)
    native_frames = pass2_result_frames(refine_ring=refine_ring)
    native = write_video(native_frames, root / f"{name}-native.mp4")
    retimed_frames = [
        np.clip(frame.astype(np.int16) + 1, 0, 255).astype(np.uint8) for frame in native_frames
    ]
    retimed = write_video(retimed_frames, root / f"{name}-retimed.mp4")
    result = root / f"{name}-result.mp4"
    shutil.copy2(campaign["candidate"], result)
    comparator = root / f"{name}-comparator.json"
    compare_pass2(
        [
            "--baseline",
            str(result),
            "--pass2",
            str(retimed),
            "--inserted-mask",
            str(campaign["inserted"]),
            "--cleanup-mask",
            str(campaign["cleanup"]),
            "--output",
            str(comparator),
        ]
    )

    def media(path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": 8,
            "height": 8,
            "frames": FRAMES,
            "fps": "12/1",
        }

    return {
        "native": media(native),
        "retimed": media(retimed),
        "result": media(result),
        "comparator": {
            "path": str(comparator),
            "sha256": hashlib.sha256(comparator.read_bytes()).hexdigest(),
        },
    }


def full_chain_manifest(
    campaign: dict[str, Any], packages: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    manifest = chain_manifest(campaign)
    manifest["require_full_chain"] = True
    manifest["candidates"] = [
        {
            "id": candidate_id,
            "kind": "fitted_insertion",
            "insertion_video": {"path": str(campaign["candidate"])},
            "inserted_mask": {"path": str(campaign["inserted"])},
            "cleanup_mask": {"path": str(campaign["cleanup"])},
            "pass2": packages[candidate_id],
        }
        for candidate_id in sorted(packages)
    ]
    return manifest


def test_chain_manifest_rejects_duplicate_unsafe_and_bad_polarity(campaign) -> None:
    manifest = chain_manifest(campaign)
    manifest["target_mask"]["polarity"] = "inverted"
    with pytest.raises(BestOfNError, match="polarity"):
        plan_from_manifest(manifest)

    manifest = chain_manifest(campaign)
    manifest["candidates"][1]["id"] = "good"
    with pytest.raises(BestOfNError, match="unique"):
        plan_from_manifest(manifest)

    manifest = chain_manifest(campaign)
    manifest["candidates"][1]["id"] = "../escape"
    with pytest.raises(BestOfNError, match="safe path slug"):
        plan_from_manifest(manifest)


def test_chain_manifest_requires_masks_and_validates_full_chain_keys(campaign) -> None:
    manifest = chain_manifest(campaign)
    del manifest["candidates"][0]["inserted_mask"]
    with pytest.raises(BestOfNError, match="inserted_mask"):
        plan_from_manifest(manifest)

    manifest = chain_manifest(campaign)
    del manifest["candidates"][0]["cleanup_mask"]
    with pytest.raises(BestOfNError, match="cleanup_mask"):
        plan_from_manifest(manifest)

    manifest = chain_manifest(campaign)
    manifest["candidates"][0]["pass2"] = {"status": "executed"}
    with pytest.raises(BestOfNError, match=r"pass2.*unsupported key"):
        plan_from_manifest(manifest)

    manifest = chain_manifest(campaign)
    manifest["require_full_chain"] = "yes"
    with pytest.raises(BestOfNError, match="require_full_chain"):
        plan_from_manifest(manifest)


def test_chain_manifest_rejects_unsupported_keys_and_digest_mismatch(campaign, tmp_path) -> None:
    manifest = chain_manifest(campaign)
    manifest["unexpected"] = 1
    with pytest.raises(BestOfNError, match="unsupported key"):
        plan_from_manifest(manifest)

    manifest = chain_manifest(campaign)
    manifest["plate"]["sha256"] = "0" * 64
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    with pytest.raises(BestOfNError, match="sha256 mismatch"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])


def test_remask_consistency_gate_eliminates_a_stale_mask(campaign, tmp_path) -> None:
    manifest_path = write_manifest(tmp_path / "manifest.json", chain_manifest(campaign))

    exit_code = main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-chain-report.json").read_text())
    candidates = {item["candidate_id"]: item for item in report["candidates"]}

    assert exit_code == 0
    assert candidates["good"]["pre_pass2"]["gates"]["remask_consistency"] is True
    assert candidates["good"]["pre_pass2"]["remask_agreement"] == pytest.approx(1.0)
    assert candidates["stale"]["pre_pass2"]["gates"]["remask_consistency"] is False
    assert candidates["stale"]["pre_pass2"]["remask_agreement"] == pytest.approx(0.0)
    assert "remask_consistency" in candidates["stale"]["failure_reason"]
    assert candidates["stale"]["stage"] == "eliminated"
    assert report["survivors"] == ["good"]
    assert "stale" not in [entry["candidate_id"] for entry in report["ranking_pre_pass2"]]


def test_chain_stages_one_pass2_sequence_per_survivor(campaign, tmp_path) -> None:
    manifest_path = write_manifest(tmp_path / "manifest.json", chain_manifest(campaign))

    exit_code = main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    sequence = tmp_path / "out/void-pass2-sequence/good"

    assert exit_code == 0
    assert (sequence / "input_video.mp4").is_file()
    assert (sequence / "quadmask_0.mp4").is_file()
    assert json.loads((sequence / "prompt.json").read_text()) == {"bg": "A plain gray scene"}
    assert (tmp_path / "out/void-pass2-sequence/stale").exists() is False

    frames = decode_gray_frames(
        sequence / "quadmask_0.mp4", tmp_path / "quadmask", expected_frames=FRAMES
    )
    assert set(np.unique(frames[0]).tolist()) == {255}
    assert set(np.unique(frames[1]).tolist()) == {127, 255}
    assert int(np.count_nonzero(frames[1] == 127)) == 16


def test_chain_report_records_ranking_pre_pass2_and_selected(campaign, tmp_path) -> None:
    manifest_path = write_manifest(tmp_path / "manifest.json", chain_manifest(campaign))

    exit_code = main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-chain-report.json").read_text())

    assert exit_code == 0
    assert report["schema"] == "void-fusion-lab.best-of-n.chain-report.v1"
    assert report["mode"] == "chain"
    assert report["target_mask"]["polarity"] == "roi_black"
    assert report["ranking_pre_pass2"] == [
        {"candidate_id": "good", "objective": report["candidates"][0]["pre_pass2"]["objective"]}
    ]
    assert report["selected"]["candidate_id"] == "good"
    assert report["selected"]["basis"] == "pre_pass2"
    assert report["ranking_full_chain"] == []
    assert report["selected"]["pass2_sequence"].endswith("void-pass2-sequence/good")
    assert report["candidates"][0]["artifacts"]["inserted_mask_source"] == "declared"
    assert report["candidates"][0]["artifacts"]["cleanup_ring_pixels_measured"] == 16 * FRAMES


def test_full_chain_records_rank_and_select_on_pass2_aware_basis(campaign, tmp_path) -> None:
    packages = {
        "best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False),
        "noisy": write_pass2_package(campaign, tmp_path, "noisy", refine_ring=True),
    }
    manifest_path = write_manifest(
        tmp_path / "full-chain.json", full_chain_manifest(campaign, packages)
    )

    exit_code = main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-chain-report.json").read_text())
    rankings = {entry["candidate_id"]: entry["objective"] for entry in report["ranking_full_chain"]}
    best = next(item for item in report["candidates"] if item["candidate_id"] == "best")

    assert exit_code == 0
    assert report["require_full_chain"] is True
    assert list(rankings) == ["best", "noisy"]
    assert rankings["best"] > rankings["noisy"]
    assert report["selected"]["candidate_id"] == "best"
    assert report["selected"]["basis"] == "full_chain"
    assert report["selected"]["artifacts"]["pass2_native"].endswith("best-native.mp4")
    assert report["selected"]["artifacts"]["pass2_retimed"].endswith("best-retimed.mp4")
    assert report["selected"]["artifacts"]["pass2_result"].endswith("best-result.mp4")
    assert report["selected"]["artifacts"]["pass2_comparator"].endswith("best-comparator.json")
    for key in ("native", "retimed", "result"):
        assert best["pass2"][key]["width"] == 8
        assert best["pass2"][key]["height"] == 8
        assert best["pass2"][key]["frames"] == FRAMES
        assert best["pass2"][key]["fps"] == "12/1"
    assert len(best["pass2"]["comparator"]["sha256"]) == 64


def test_manifest_inputs_protect_every_candidate_pass2_package(campaign, tmp_path) -> None:
    packages = {
        "best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False),
        "noisy": write_pass2_package(campaign, tmp_path, "noisy", refine_ring=True),
    }
    manifest = full_chain_manifest(campaign, packages)
    protected = manifest_input_paths(plan_from_manifest(manifest))

    for package in packages.values():
        for key in ("native", "retimed", "result", "comparator"):
            assert Path(package[key]["path"]) in protected


def test_full_chain_rejects_contradictory_embedded_comparator_cadence(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    comparator_path = Path(packages["best"]["comparator"]["path"])
    comparator = json.loads(comparator_path.read_text())
    comparator["frame_timestamps"]["baseline"]["frames"] = 99
    comparator_path.write_text(json.dumps(comparator, indent=2) + "\n")
    packages["best"]["comparator"]["sha256"] = hashlib.sha256(
        comparator_path.read_bytes()
    ).hexdigest()
    manifest_path = write_manifest(
        tmp_path / "cadence.json", full_chain_manifest(campaign, packages)
    )

    with pytest.raises(BestOfNError, match="timestamp frame counts"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])


def test_full_chain_rejects_embedded_mask_geometry_disagreement(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    comparator_path = Path(packages["best"]["comparator"]["path"])
    comparator = json.loads(comparator_path.read_text())
    comparator["frame_alignment"]["inserted_mask"]["width"] = 99
    comparator_path.write_text(json.dumps(comparator, indent=2) + "\n")
    packages["best"]["comparator"]["sha256"] = hashlib.sha256(
        comparator_path.read_bytes()
    ).hexdigest()
    manifest_path = write_manifest(
        tmp_path / "embedded-geometry.json", full_chain_manifest(campaign, packages)
    )

    with pytest.raises(BestOfNError, match="inserted_mask geometry"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])


def test_full_chain_binds_pass2_result_to_the_candidate_insertion(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    other = write_video(pass2_result_frames(refine_ring=True), tmp_path / "other-result.mp4")
    packages["best"]["result"]["path"] = str(other)
    packages["best"]["result"]["sha256"] = hashlib.sha256(other.read_bytes()).hexdigest()
    manifest_path = write_manifest(
        tmp_path / "wrong-candidate.json", full_chain_manifest(campaign, packages)
    )

    with pytest.raises(BestOfNError, match="does not bind to its insertion artifact"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])


def test_full_chain_rejects_a_changed_comparator_threshold(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    comparator_path = Path(packages["best"]["comparator"]["path"])
    comparator = json.loads(comparator_path.read_text())
    comparator["thresholds"]["changed_threshold"] = 11.0
    comparator_path.write_text(json.dumps(comparator, indent=2) + "\n")
    packages["best"]["comparator"]["sha256"] = hashlib.sha256(
        comparator_path.read_bytes()
    ).hexdigest()
    manifest_path = write_manifest(
        tmp_path / "threshold.json", full_chain_manifest(campaign, packages)
    )

    with pytest.raises(BestOfNError, match="changed_threshold does not match"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])


def test_full_chain_rejects_a_self_compared_pass2_result(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    packages["best"]["result"] = dict(packages["best"]["retimed"])
    manifest = full_chain_manifest(campaign, packages)

    with pytest.raises(BestOfNError, match="self-compared baseline"):
        plan_from_manifest(manifest)


def test_full_chain_rejects_malformed_embedded_alignment_fps(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    comparator_path = Path(packages["best"]["comparator"]["path"])
    comparator = json.loads(comparator_path.read_text())
    del comparator["frame_alignment"]["baseline"]["fps"]
    comparator_path.write_text(json.dumps(comparator, indent=2) + "\n")
    packages["best"]["comparator"]["sha256"] = hashlib.sha256(
        comparator_path.read_bytes()
    ).hexdigest()
    manifest_path = write_manifest(
        tmp_path / "malformed-fps.json", full_chain_manifest(campaign, packages)
    )

    with pytest.raises(BestOfNError, match="baseline fps must be numeric"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])


def test_invalid_rerun_demotes_the_previous_success_report(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    output = tmp_path / "out"
    manifest_path = write_manifest(tmp_path / "valid.json", full_chain_manifest(campaign, packages))
    main(["select", "--manifest", str(manifest_path), "--out", str(output)])
    canonical = output / "best-of-n-chain-report.json"
    assert canonical.is_file()

    packages["best"]["native"]["path"] = str(tmp_path / "absent.mp4")
    invalid_path = write_manifest(
        tmp_path / "invalid.json", full_chain_manifest(campaign, packages)
    )
    with pytest.raises(BestOfNError, match="does not name an existing file"):
        main(["select", "--manifest", str(invalid_path), "--out", str(output)])

    assert not canonical.is_file()
    assert list((output / "failed").glob("*best-of-n-chain-report.json"))


def test_zero_denominator_pass2_fps_fails_closed(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    packages["best"]["native"]["fps"] = "1/0"
    manifest = full_chain_manifest(campaign, packages)

    with pytest.raises(BestOfNError, match="finite positive rational"):
        plan_from_manifest(manifest)

    packages["best"]["native"]["fps"] = f"{10**400}/1"
    manifest = full_chain_manifest(campaign, packages)

    with pytest.raises(BestOfNError, match="finite positive rational"):
        plan_from_manifest(manifest)


def test_require_full_chain_fails_closed_when_a_survivor_has_no_pass2_record(
    campaign, tmp_path
) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    manifest = full_chain_manifest(campaign, packages)
    manifest["candidates"].append(
        {
            "id": "unexecuted",
            "kind": "fitted_insertion",
            "insertion_video": {"path": str(campaign["candidate"])},
            "inserted_mask": {"path": str(campaign["inserted"])},
            "cleanup_mask": {"path": str(campaign["cleanup"])},
        }
    )
    manifest_path = write_manifest(tmp_path / "missing.json", manifest)

    with pytest.raises(BestOfNError, match="unexecuted.*no pass2 record"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])

    assert not (tmp_path / "out/best-of-n-chain-report.json").exists()
    assert not (tmp_path / "out/void-pass2-sequence").exists()


def test_full_chain_rejects_a_stale_retimed_digest(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    replacement = write_video(pass2_result_frames(refine_ring=True), tmp_path / "replacement.mp4")
    Path(packages["best"]["retimed"]["path"]).write_bytes(replacement.read_bytes())
    manifest_path = write_manifest(tmp_path / "stale.json", full_chain_manifest(campaign, packages))

    with pytest.raises(BestOfNError, match="pass2.retimed sha256 mismatch"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])

    assert not (tmp_path / "out/best-of-n-chain-report.json").exists()


def test_full_chain_rejects_a_comparator_bound_to_another_result(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    comparator_path = Path(packages["best"]["comparator"]["path"])
    comparator = json.loads(comparator_path.read_text())
    comparator["inputs"]["baseline"]["sha256"] = "0" * 64
    comparator_path.write_text(json.dumps(comparator, indent=2) + "\n")
    packages["best"]["comparator"]["sha256"] = hashlib.sha256(
        comparator_path.read_bytes()
    ).hexdigest()
    manifest_path = write_manifest(
        tmp_path / "mismatch.json", full_chain_manifest(campaign, packages)
    )

    with pytest.raises(BestOfNError, match="baseline digest mismatch"):
        main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])


def test_full_chain_rejects_declared_geometry_disagreement(campaign, tmp_path) -> None:
    packages = {"best": write_pass2_package(campaign, tmp_path, "best", refine_ring=False)}
    packages["best"]["result"]["width"] = 9
    manifest_path = write_manifest(
        tmp_path / "geometry.json", full_chain_manifest(campaign, packages)
    )

    with pytest.raises(BestOfNError, match="must share width, height, and frame count"):
        plan_from_manifest(json.loads(manifest_path.read_text()))


def test_chain_report_is_deterministic(campaign, tmp_path) -> None:
    manifest_path = write_manifest(tmp_path / "manifest.json", chain_manifest(campaign))
    output = tmp_path / "out"

    main(["select", "--manifest", str(manifest_path), "--out", str(output)])
    first = json.loads((output / "best-of-n-chain-report.json").read_text())
    main(["select", "--manifest", str(manifest_path), "--out", str(output)])
    second = json.loads((output / "best-of-n-chain-report.json").read_text())

    first.pop("recorded_at")
    second.pop("recorded_at")
    assert first == second


def test_chain_run_without_a_survivor_exits_three(campaign, tmp_path) -> None:
    manifest = chain_manifest(campaign)
    manifest["thresholds"]["min_object_precision"] = 1.0
    manifest["candidates"][0]["insertion_video"] = {"path": str(campaign["plate"])}
    manifest["candidates"][1]["insertion_video"] = {"path": str(campaign["plate"])}
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)

    exit_code = main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-chain-report.json").read_text())

    assert exit_code == 3
    assert report["survivors"] == []
    assert report["selected"] is None
    assert report["ranking_pre_pass2"] == []
    assert not (tmp_path / "out/void-pass2-sequence").exists()


def test_legacy_manifest_runs_as_an_adapter_with_identical_scores(campaign, tmp_path) -> None:
    manifest = {
        "source_plate": str(campaign["plate"]),
        "mask_video": str(campaign["mask"]),
        "background_prompt": "A plain gray scene",
        "candidates": [{"id": "good", "artifact": str(campaign["candidate"])}],
    }
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)

    exit_code = main(["--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-report.json").read_text())
    expected = score_candidate(
        candidate_id="good",
        plate=rgb(plate_frames()),
        candidate=rgb(candidate_frames()),
        masks=target_mask_frames(),
    )

    assert exit_code == 0
    assert report["mode"] == "legacy-adapter"
    assert report["candidates"][0]["inserted_precision"] == pytest.approx(
        expected.inserted_precision
    )
    assert report["candidates"][0]["objective"] == pytest.approx(expected.objective)
    assert report["candidates"][0]["gates"] == expected.gates
    assert report["survivors"] == ["good"]
    assert report["golden"]["candidate_id"] == "good"
    assert not (tmp_path / "out/best-of-n-chain-report.json").exists()


def test_legacy_manifest_without_a_survivor_still_exits_three(campaign, tmp_path) -> None:
    manifest = {
        "source_plate": str(campaign["plate"]),
        "mask_video": str(campaign["mask"]),
        "background_prompt": "A plain gray scene",
        "candidates": [{"id": "noop", "artifact": str(campaign["plate"])}],
    }
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)

    exit_code = main(["--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-report.json").read_text())

    assert exit_code == 3
    assert report["golden"] is None
    assert report["survivors"] == []
    assert not (tmp_path / "out/golden-candidate.mp4").exists()
    assert not (tmp_path / "out/void-pass2-sequence").exists()


def test_legacy_and_chain_paths_measure_the_same_pre_pass2_score(campaign, tmp_path) -> None:
    legacy_manifest = write_manifest(
        tmp_path / "legacy.json",
        {
            "source_plate": str(campaign["plate"]),
            "mask_video": str(campaign["mask"]),
            "background_prompt": "A plain gray scene",
            "candidates": [{"id": "good", "artifact": str(campaign["candidate"])}],
        },
    )
    chain_path = write_manifest(tmp_path / "chain.json", chain_manifest(campaign))

    main(["--manifest", str(legacy_manifest), "--out", str(tmp_path / "legacy-out")])
    main(["select", "--manifest", str(chain_path), "--out", str(tmp_path / "chain-out")])
    legacy = json.loads((tmp_path / "legacy-out/best-of-n-report.json").read_text())
    chain = json.loads((tmp_path / "chain-out/best-of-n-chain-report.json").read_text())
    chain_candidate = next(item for item in chain["candidates"] if item["candidate_id"] == "good")

    assert legacy["candidates"][0]["objective"] == pytest.approx(
        chain_candidate["pre_pass2"]["objective"]
    )
    assert legacy["candidates"][0]["gates"] == {
        key: value
        for key, value in chain_candidate["pre_pass2"]["gates"].items()
        if key != "remask_consistency"
    }
    assert legacy["survivors"] == [entry["candidate_id"] for entry in chain["ranking_pre_pass2"]]


def test_shipped_example_matches_the_chain_schema(tmp_path) -> None:
    example = Path(__file__).resolve().parents[2] / "configs/best-of-n.chain-manifest.example.json"
    manifest = json.loads(example.read_text())

    assert manifest["schema"] == CHAIN_MANIFEST_SCHEMA
    assert manifest["target_mask"]["polarity"] in ("roi_black", "foreground_white")
    assert manifest["candidates"]
    for candidate in manifest["candidates"]:
        assert candidate["kind"] in ("fitted_insertion", "generated_replacement")
        for key in ("insertion_video", "inserted_mask", "cleanup_mask"):
            assert isinstance(candidate[key]["path"], str)
            assert candidate[key]["path"]
        for key in ("native", "retimed", "result", "comparator"):
            assert candidate["pass2"][key]["path"]

    placeholder = tmp_path / "artifact.bin"
    placeholder.write_bytes(b"placeholder")
    manifest["plate"]["path"] = str(placeholder)
    manifest["target_mask"]["path"] = str(placeholder)
    for candidate in manifest["candidates"]:
        for key in ("insertion_video", "inserted_mask", "cleanup_mask"):
            candidate[key]["path"] = str(placeholder)
        for key_index, key in enumerate(("native", "retimed", "result", "comparator")):
            candidate["pass2"][key]["path"] = str(placeholder)
            candidate["pass2"][key]["sha256"] = f"{key_index:064x}"

    plan = plan_from_manifest(manifest)

    assert plan.mode == "chain"
    assert plan.require_full_chain is True
    assert plan.target_polarity == manifest["target_mask"]["polarity"]
    assert len(plan.candidates) == len(manifest["candidates"])
    assert plan.thresholds.min_remask_agreement == 0.5


def test_stale_cleanup_preserves_manifest_inputs_in_the_output_directory(
    campaign, tmp_path
) -> None:
    """A campaign may write into the directory that holds the media it reads."""
    output = tmp_path / "out"
    output.mkdir()
    insertion = output / "candidate-smooth-04-insertion.mp4"
    inserted = output / "candidate-smooth-04-inserted-mask.mp4"
    cleanup = output / "candidate-smooth-04-cleanup-mask.mp4"
    shutil.copy2(campaign["candidate"], insertion)
    shutil.copy2(campaign["inserted"], inserted)
    shutil.copy2(campaign["cleanup"], cleanup)
    manifest = {
        "schema": CHAIN_MANIFEST_SCHEMA,
        "plate": {"path": str(campaign["plate"])},
        "target_mask": {"path": str(campaign["mask"]), "polarity": "roi_black"},
        "background_prompt": "A plain gray scene",
        "candidates": [
            {
                "id": "smooth-04",
                "kind": "fitted_insertion",
                "insertion_video": {"path": str(insertion)},
                "inserted_mask": {"path": str(inserted)},
                "cleanup_mask": {"path": str(cleanup)},
            }
        ],
    }
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    before = {path: path.read_bytes() for path in (insertion, inserted, cleanup)}

    exit_code = main(["select", "--manifest", str(manifest_path), "--out", str(output)])

    assert exit_code == 0
    for path, payload in before.items():
        assert path.is_file(), f"stale cleanup deleted the manifest input {path.name}"
        assert path.read_bytes() == payload, f"the campaign rewrote the input {path.name}"
    assert (output / "normalized/candidate-smooth-04-insertion.mp4").is_file()


def test_survivor_staging_refuses_to_overwrite_a_normalized_manifest_input(
    campaign, tmp_path
) -> None:
    output = tmp_path / "out"
    normalized = output / "normalized/candidate-smooth-04-insertion.mp4"
    normalized.parent.mkdir(parents=True)
    shutil.copy2(campaign["candidate"], normalized)
    inserted = output / "candidate-smooth-04-inserted-mask.mp4"
    cleanup = output / "candidate-smooth-04-cleanup-mask.mp4"
    shutil.copy2(campaign["inserted"], inserted)
    shutil.copy2(campaign["cleanup"], cleanup)
    manifest = {
        "schema": CHAIN_MANIFEST_SCHEMA,
        "plate": {"path": str(campaign["plate"])},
        "target_mask": {"path": str(campaign["mask"]), "polarity": "roi_black"},
        "background_prompt": "A plain gray scene",
        "candidates": [
            {
                "id": "smooth-04",
                "kind": "fitted_insertion",
                "insertion_video": {"path": str(normalized)},
                "inserted_mask": {"path": str(inserted)},
                "cleanup_mask": {"path": str(cleanup)},
            }
        ],
    }
    before = normalized.read_bytes()

    with pytest.raises(BestOfNError, match="would overwrite a declared manifest input"):
        main(
            [
                "select",
                "--manifest",
                str(write_manifest(tmp_path / "manifest.json", manifest)),
                "--out",
                str(output),
            ]
        )

    assert normalized.read_bytes() == before


def test_a_corrupt_declared_mask_exits_two_without_a_traceback(campaign, tmp_path) -> None:
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_text("not a video", encoding="utf-8")
    manifest = chain_manifest(campaign)
    manifest["candidates"][0]["inserted_mask"]["path"] = str(corrupt)
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/best_of_n.py"),
            "select",
            "--manifest",
            str(manifest_path),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Best-of-N selection failed:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_zero_max_background_change_is_scored_without_a_division_crash(campaign, tmp_path) -> None:
    """A zero cap demands zero background change and still scores every candidate."""
    manifest = chain_manifest(campaign)
    manifest["thresholds"]["max_background_change"] = 0.0
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)

    exit_code = main(["select", "--manifest", str(manifest_path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-chain-report.json").read_text())
    candidates = {item["candidate_id"]: item for item in report["candidates"]}

    assert exit_code == 0
    assert report["thresholds"]["max_background_change"] == 0.0
    assert candidates["good"]["pre_pass2"]["gates"]["background_preservation"] is True
    assert candidates["good"]["pre_pass2"]["background_change_ratio"] == pytest.approx(0.0)

    clean = score_candidate(
        candidate_id="clean",
        plate=rgb(plate_frames()),
        candidate=rgb(candidate_frames()),
        masks=target_mask_frames(),
        max_background_change=0.0,
    )
    scattered_frames = candidate_frames()
    for frame in scattered_frames:
        frame[0, 0] = 240
    scattered = score_candidate(
        candidate_id="scattered",
        plate=rgb(plate_frames()),
        candidate=rgb(scattered_frames),
        masks=target_mask_frames(),
        max_background_change=0.0,
    )

    assert clean.gates["background_preservation"] is True
    assert scattered.gates["background_preservation"] is False
    assert clean.objective == pytest.approx(1.0)
    assert scattered.objective < clean.objective


def test_legacy_manifest_keeps_the_original_background_change_bound(campaign) -> None:
    """The legacy lane keeps its own (0, 1] bound on max_background_change."""
    manifest = {
        "source_plate": str(campaign["plate"]),
        "mask_video": str(campaign["mask"]),
        "background_prompt": "A plain gray scene",
        "candidates": [{"id": "good", "artifact": str(campaign["candidate"])}],
    }

    accepted = validate_manifest({**manifest, "max_background_change": 1.0})
    assert accepted["max_background_change"] == 1.0
    with pytest.raises(BestOfNError, match="max_background_change"):
        validate_manifest({**manifest, "max_background_change": 2.0})
    with pytest.raises(BestOfNError, match="max_background_change"):
        validate_manifest({**manifest, "max_background_change": 0.0})


def test_contact_sheet_honors_the_declared_target_mask_polarity(tmp_path) -> None:
    """A foreground_white mask tints the removal region, not the background."""
    plate = [np.full((8, 8, 3), 20, dtype=np.uint8)]
    candidate = [np.full((8, 8, 3), 200, dtype=np.uint8)]
    masks = [np.full((8, 8), 255, dtype=np.uint8)]

    tinted = overlay(plate[0], masks[0], FOREGROUND_WHITE)
    untinted = overlay(plate[0], masks[0], ROI_BLACK)

    assert tinted[0, 0, 0] > 100 and tinted[0, 0, 1] < 60
    assert untinted[0, 0, 0] == 20 and untinted[0, 0, 1] == 20

    sheet = tmp_path / "contact.jpg"
    write_contact_sheet(
        plate=plate,
        candidate=candidate,
        masks=masks,
        output=sheet,
        selected_frames=[0],
        target_polarity=FOREGROUND_WHITE,
    )
    image = np.asarray(Image.open(sheet))

    assert image[60, 400, 1] < 60
    assert image[60, 100, 1] == 20


def test_the_shipped_chain_objective_weights_match_the_published_evidence_formula() -> None:
    """The published formula and the shipped implementation must state one objective."""
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    formula = payload["real_full_chain"]["full_chain_objective_formula"]
    published = [
        (term["weight"], term["metric"], term["complemented"]) for term in formula["terms"]
    ]

    assert shipped_full_chain_objective_terms() == published
    assert published == [
        (0.35, "pre_pass2_objective", False),
        (0.25, "object_preservation_changed_ratio", True),
        (0.20, "background_changed_ratio", True),
        (0.20, "temporal_pop_score", True),
    ]


def test_the_published_candidate_objectives_reproduce_the_recorded_ranking() -> None:
    """Every published candidate objective recomputes from the shipped formula."""
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    candidates = payload["real_full_chain"]["candidates"]
    terms = shipped_full_chain_objective_terms()

    recomputed: dict[str, float] = {}
    for candidate_id, candidate in candidates.items():
        total = 0.0
        for weight, metric, complemented in terms:
            if metric == "pre_pass2_objective":
                value = candidate["pre_pass2_objective"]
            else:
                value = candidate["metrics"][metric]
            if complemented:
                value = 1.0 - value
            total += weight * value
        recomputed[candidate_id] = total
        assert total == pytest.approx(candidate["full_chain_objective"], abs=1e-9), candidate_id

    ordered = tuple(sorted(recomputed, key=lambda candidate_id: -recomputed[candidate_id]))
    assert ordered == ("smooth-00", "smooth-06", "smooth-02", "smooth-04")

    selection = payload["real_full_chain"]["full_chain_selection"]
    assert selection["selected_candidate_id"] == ordered[0]
    assert [entry["candidate_id"] for entry in selection["ranking"]] == list(ordered)
    for entry in selection["ranking"]:
        assert entry["objective"] == pytest.approx(recomputed[entry["candidate_id"]], abs=1e-9)
