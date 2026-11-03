from __future__ import annotations

import json

import numpy as np
import pytest
from best_of_n import BestOfNError, main, score_candidate, validate_manifest

from samurai.media import write_gray_video


def frames(value: int, *, background: int = 20) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for index in range(6):
        frame = np.full((8, 8, 3), background, dtype=np.uint8)
        frame[2:5, 2:5] = value + index
        result.append(frame)
    return result


def masks() -> list[np.ndarray]:
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[2:5, 2:5] = 0
    return [mask.copy() for _ in range(6)]


def test_selector_prefers_targeted_stable_replacement() -> None:
    plate = frames(20)
    good = frames(200)
    scattered = frames(200)
    for frame in scattered:
        frame[0, 0] = 240
    good_score = score_candidate(candidate_id="good", plate=plate, candidate=good, masks=masks())
    scattered_score = score_candidate(
        candidate_id="scattered", plate=plate, candidate=scattered, masks=masks()
    )

    assert good_score.inserted_precision == pytest.approx(1.0)
    assert good_score.background_change_ratio == 0.0
    assert all(good_score.gates.values())
    assert good_score.target_present_frames == len(plate)
    assert scattered_score.inserted_precision < good_score.inserted_precision
    assert good_score.objective > scattered_score.objective


def test_selector_fails_closed_on_empty_or_misaligned_geometry() -> None:
    plate = frames(20)
    with pytest.raises(BestOfNError, match="no changed pixels"):
        score_candidate(candidate_id="same", plate=plate, candidate=frames(20), masks=masks())
    bad_masks = [np.full((4, 4), 255, dtype=np.uint8) for _ in plate]
    with pytest.raises(BestOfNError, match="RGB/mask geometry mismatch"):
        score_candidate(candidate_id="bad", plate=plate, candidate=frames(200), masks=bad_masks)


def test_selector_rejects_temporal_disappearance() -> None:
    plate = frames(20)
    candidate = frames(200)
    candidate[3][:] = 20
    score = score_candidate(candidate_id="drop", plate=plate, candidate=candidate, masks=masks())

    assert score.gates["object_present_every_frame"] is False
    assert score.gates["temporal_stability"] is False


def test_campaign_reports_noop_candidate_and_keeps_valid_survivor(tmp_path) -> None:
    plate_frames = [np.full((8, 8), 20, dtype=np.uint8) for _ in range(6)]
    good_frames = [frame.copy() for frame in plate_frames]
    for frame in good_frames:
        frame[2:5, 2:5] = 220
    mask_frames = [np.full((8, 8), 255, dtype=np.uint8) for _ in range(6)]
    for frame in mask_frames:
        frame[2:5, 2:5] = 0
    plate = tmp_path / "plate.mp4"
    mask = tmp_path / "mask.mp4"
    good = tmp_path / "good.mp4"
    noop = tmp_path / "noop.mp4"
    encoded = (
        (plate, plate_frames),
        (mask, mask_frames),
        (good, good_frames),
        (noop, plate_frames),
    )
    for path, frames in encoded:
        write_gray_video(frames, path, fps=12.0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_plate": str(plate),
                "mask_video": str(mask),
                "background_prompt": "A plain gray scene",
                "candidates": [
                    {"id": "noop", "artifact": str(noop)},
                    {"id": "good", "artifact": str(good)},
                ],
            }
        )
        + "\n"
    )

    exit_code = main(["--manifest", str(manifest), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out/best-of-n-report.json").read_text())
    scores = {item["candidate_id"]: item for item in report["candidates"]}

    assert exit_code == 0
    assert scores["noop"]["failure_reason"] == "candidate 'noop' has no changed pixels"
    assert not any(scores["noop"]["gates"].values())
    assert report["golden"]["candidate_id"] == "good"

    strict = json.loads(manifest.read_text())
    strict["candidates"] = [{"id": "noop", "artifact": str(noop)}]
    manifest.write_text(json.dumps(strict) + "\n")
    exit_code = main(["--manifest", str(manifest), "--out", str(tmp_path / "out")])
    failed_report = json.loads((tmp_path / "out/best-of-n-report.json").read_text())

    assert exit_code == 3
    assert failed_report["golden"] is None
    assert failed_report["survivors"] == []
    assert not (tmp_path / "out/golden-candidate.mp4").exists()
    assert not (tmp_path / "out/void-pass2-sequence").exists()


def test_manifest_rejects_unsafe_ids_and_missing_prompt(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub")
    payload = {
        "source_plate": str(source),
        "mask_video": str(source),
        "background_prompt": "A plain gray scene",
        "candidates": [{"id": "../escape", "artifact": str(source)}],
    }
    with pytest.raises(BestOfNError, match="safe path slug"):
        validate_manifest(payload)
    payload.pop("background_prompt")
    payload["candidates"] = [{"id": "safe", "artifact": str(source)}]
    with pytest.raises(BestOfNError, match="background_prompt"):
        validate_manifest(payload)
