from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from contact_grounding import ContactGroundingScriptError
from owned_frame_contact import (
    OwnedFrameContactScriptError,
    main,
    parse_candidate_manifest,
)
from PIL import Image

from samurai.owned_frame_contact import (
    OwnedFrameContactError,
    contact_keypoint_payload,
    parse_ownership_report,
)

FRAME_COUNT = 96
OWNED_FRAMES = (0, 1, 7, 13)
FRAME_HEIGHT = 96
FRAME_WIDTH = 128
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PIXEL_POINTS = {
    "left_hip": (60.0, 40.0),
    "right_hip": (68.0, 40.0),
    "left_knee": (58.0, 60.0),
    "right_knee": (70.0, 60.0),
    "left_ankle": (56.0, 80.0),
    "right_ankle": (72.0, 80.0),
    "left_heel": (54.0, 82.0),
    "right_heel": (74.0, 82.0),
    "left_foot_index": (50.0, 84.0),
    "right_foot_index": (78.0, 84.0),
}


def ownership_report() -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for index in range(FRAME_COUNT):
        owned = index in OWNED_FRAMES
        landmarks = (
            {
                name: {"x": x / FRAME_WIDTH, "y": y / FRAME_HEIGHT, "z": 0.0, "visibility": 0.9}
                for name, (x, y) in PIXEL_POINTS.items()
            }
            if owned
            else {}
        )
        frames.append(
            {
                "frame_index": index,
                "owned_pose_ids": ["person"] if owned else [],
                "poses": [
                    {
                        "pose_id": "person",
                        "owned": owned,
                        "landmarks": landmarks,
                    }
                ],
            }
        )
    return {
        "schema": "lower-body-ownership-report-v1",
        "summary": {
            "frames": FRAME_COUNT,
            "poses": FRAME_COUNT,
            "owned_poses": len(OWNED_FRAMES),
            "owned_frames": len(OWNED_FRAMES),
            "frame_ownership_ratio": 0.041667,
        },
        "frames": frames,
        "non_claims": [],
    }


def object_mask(*, bottom: int = 84) -> np.ndarray:
    mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)
    mask[60 : bottom + 1, 56:72] = True
    return mask


def plate() -> np.ndarray:
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    ramp = np.clip(np.arange(FRAME_WIDTH, dtype=np.float64) * 1.5, 0, 255).astype(np.uint8)
    frame[:, :, :] = ramp[None, :, None]
    return frame


def write_frame_directory(
    directory: Path,
    *,
    frames: int,
    make_frame: np.ndarray,
) -> Path:
    directory.mkdir(parents=True)
    image = Image.fromarray(make_frame)
    for index in range(frames):
        image.save(directory / f"frame-{index:06d}.png")
    return directory


def write_inputs(directory: Path, *, short_mask: bool = False) -> tuple[Path, Path]:
    ownership = directory / "ownership.json"
    ownership.write_text(json.dumps(ownership_report()), encoding="utf-8")
    write_frame_directory(
        directory / "mask-a",
        frames=FRAME_COUNT,
        make_frame=np.where(object_mask(), 0, 255).astype(np.uint8),
    )
    write_frame_directory(
        directory / "mask-b",
        frames=FRAME_COUNT,
        make_frame=np.where(object_mask(bottom=70), 0, 255).astype(np.uint8),
    )
    write_frame_directory(directory / "plate", frames=FRAME_COUNT, make_frame=plate())
    candidates = directory / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema": "void-fusion-lab.owned-frame-contact.candidates.v1",
                "plate": "plate",
                "candidates": [
                    {"id": "a", "object_mask": "mask-a"},
                    {"id": "b", "object_mask": "mask-b"},
                ],
            }
            if not short_mask
            else {
                "schema": "void-fusion-lab.owned-frame-contact.candidates.v1",
                "plate": "plate",
                "candidates": [
                    {
                        "id": "short",
                        "object_mask": write_frame_directory(
                            directory / "mask-short",
                            frames=3,
                            make_frame=np.where(object_mask(), 0, 255).astype(np.uint8),
                        ).name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return ownership, candidates


def test_parse_ownership_report_retains_a_four_of_ninety_six_subset() -> None:
    selection = parse_ownership_report(ownership_report())

    assert selection.total_frames == 96
    assert tuple(frame.frame_index for frame in selection.owned_frames) == OWNED_FRAMES
    assert selection.owned_frame_ratio == pytest.approx(4 / 96)


def test_duplicate_pose_ids_fail_even_when_one_instance_is_owned() -> None:
    payload = ownership_report()
    payload["frames"][0]["poses"] = [
        {
            "pose_id": "person",
            "owned": False,
            "landmarks": {"right_ankle": {"x": 0.9, "y": 0.9}},
        },
        {
            "pose_id": "person",
            "owned": True,
            "landmarks": {
                name: {"x": x / FRAME_WIDTH, "y": y / FRAME_HEIGHT}
                for name, (x, y) in PIXEL_POINTS.items()
            },
        },
    ]

    with pytest.raises(OwnedFrameContactError, match="repeats pose id person"):
        parse_ownership_report(payload)


def test_pose_summary_counts_must_match_frame_records() -> None:
    payload = ownership_report()
    payload["summary"]["owned_poses"] = 999

    with pytest.raises(OwnedFrameContactError, match="owned poses but carries"):
        parse_ownership_report(payload)

    payload = ownership_report()
    payload["summary"]["poses"] = 999

    with pytest.raises(OwnedFrameContactError, match="poses but carries"):
        parse_ownership_report(payload)


def test_sparse_landmarks_map_to_reindexed_pixel_coordinates() -> None:
    payload = contact_keypoint_payload(
        parse_ownership_report(ownership_report()), width=FRAME_WIDTH, height=FRAME_HEIGHT
    )

    assert [row["frame"] for row in payload["frames"]] == [0, 1, 2, 3]
    assert payload["frames"][0]["points"]["right_ankle"] == [72.0, 80.0]


def test_ownership_summary_mismatch_fails_closed() -> None:
    payload = ownership_report()
    payload["summary"]["owned_frames"] = 5
    payload["summary"]["frame_ownership_ratio"] = 0.052083

    with pytest.raises(OwnedFrameContactError, match="owned frames but carries"):
        parse_ownership_report(payload)


def test_two_owned_poses_in_one_frame_are_ambiguous() -> None:
    payload = ownership_report()
    first = payload["frames"][0]
    first["poses"].append({**first["poses"][0], "pose_id": "second"})
    first["owned_pose_ids"] = ["person", "second"]

    with pytest.raises(OwnedFrameContactError, match="unambiguous pose"):
        parse_ownership_report(payload)


def test_an_ownership_frame_outside_the_declared_clip_fails_closed() -> None:
    payload = ownership_report()
    payload["frames"][-1]["frame_index"] = FRAME_COUNT

    with pytest.raises(OwnedFrameContactError, match="outside the 96 frame report"):
        parse_ownership_report(payload)


def test_candidate_manifest_resolves_a_shared_plate_and_candidate_masks(
    tmp_path: Path,
) -> None:
    _, manifest = write_inputs(tmp_path)

    shared_plate, candidates = parse_candidate_manifest(
        json.loads(manifest.read_text(encoding="utf-8")), manifest
    )

    assert shared_plate == tmp_path / "plate"
    assert [candidate.candidate_id for candidate in candidates] == ["a", "b"]
    assert candidates[0].plate == shared_plate
    assert candidates[0].object_mask == tmp_path / "mask-a"


def test_cli_measures_each_candidate_only_on_four_owned_frames(tmp_path: Path) -> None:
    ownership, candidates = write_inputs(tmp_path)
    output = tmp_path / "evidence"

    code = main(
        [
            "--ownership-report",
            str(ownership),
            "--candidates",
            str(candidates),
            "--out",
            str(output),
            "--fps",
            "24",
        ]
    )

    assert code == 0
    report = json.loads((output / "owned-frame-contact-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "void-fusion-lab.owned-frame-contact.report.v1"
    assert report["ownership"] == {
        "total_frames": 96,
        "owned_frames": list(OWNED_FRAMES),
        "owned_frame_ratio": pytest.approx(4 / 96),
        "continuous_ownership_claimed": False,
    }
    assert [candidate["candidate_id"] for candidate in report["candidates"]] == ["a", "b"]
    assert (output / "candidates/a/shadow-mask.mp4").is_file()
    assert (output / "candidates/b/shadow-mask.mp4").is_file()
    for candidate in report["candidates"]:
        assert [frame["source_frame"] for frame in candidate["frames"]] == list(OWNED_FRAMES)
        assert [frame["frame"] for frame in candidate["frames"]] == [0, 1, 2, 3]
        assert candidate["summary"]["measured_frames"] == 4
        assert candidate["summary"]["mask_frames"] == 4
        assert candidate["shadow_mask"]["not_void_quadmask"] is True
        assert all(len(frame["shadow_digest"]) == 64 for frame in candidate["frames"])
    grounded = report["candidates"][0]["frames"]
    floating = report["candidates"][1]["frames"]
    assert all(frame["contact_coverage"] > 0.9 for frame in grounded)
    assert all(frame["contact_coverage"] == 0 for frame in floating)
    assert all(frame["shadow"]["total_pixels"] > 0 for frame in grounded)


def test_missing_candidate_masks_fail_closed_without_a_partial_report(tmp_path: Path) -> None:
    ownership, candidates = write_inputs(tmp_path)
    manifest = json.loads(candidates.read_text(encoding="utf-8"))
    manifest["candidates"][0]["object_mask"] = "absent-mask"
    candidates.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "evidence"

    with pytest.raises(ContactGroundingScriptError, match="object mask does not exist"):
        main(
            [
                "--ownership-report",
                str(ownership),
                "--candidates",
                str(candidates),
                "--out",
                str(output),
                "--fps",
                "24",
            ]
        )

    assert not output.exists()


def test_a_short_candidate_mask_sequence_fails_closed(tmp_path: Path) -> None:
    ownership, candidates = write_inputs(tmp_path, short_mask=True)
    output = tmp_path / "evidence"

    with pytest.raises(OwnedFrameContactScriptError, match="missing masks fail closed"):
        main(
            [
                "--ownership-report",
                str(ownership),
                "--candidates",
                str(candidates),
                "--out",
                str(output),
                "--fps",
                "24",
            ]
        )

    assert not output.exists()


def test_the_public_evidence_records_the_owned_frame_contact_boundary() -> None:
    payload = json.loads(
        (REPOSITORY_ROOT / "evidence" / "submission-evidence.json").read_text(encoding="utf-8")
    )
    contact = payload["real_full_chain"]["owned_frame_contact"]

    # The published ownership numbers still match the implementation contract.
    assert contact["owned_frames"] == list(OWNED_FRAMES)
    assert contact["total_frames"] == FRAME_COUNT
    assert contact["owned_frame_ratio"] == pytest.approx(4 / 96)
    assert contact["continuous_ownership_claimed"] is False

    # The measured report stays bound by digest without redistributing it.
    assert re.fullmatch(r"[0-9a-f]{64}", contact["report_sha256"])

    # Coverage is reported for exactly the candidates the public export lists.
    candidates = payload["real_full_chain"]["candidates"]
    assert set(contact["candidate_mean_contact_coverage"]) == set(candidates)
    for coverage in contact["candidate_mean_contact_coverage"].values():
        assert 0.0 <= coverage <= 1.0

    # The sparse-ownership limitation survives as a published non-claim.
    assert any(
        "4/96 frames" in claim and "no continuous ownership" in claim
        for claim in payload["non_claims"]
    )
