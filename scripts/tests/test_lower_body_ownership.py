"""Contracts for the lower-body ownership command-line interface."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from samurai.lower_body import MEDIAPIPE_POSE_LANDMARK_NAMES

SCRIPT = Path(__file__).resolve().parents[1] / "lower_body_ownership.py"
spec = importlib.util.spec_from_file_location("lower_body_ownership", SCRIPT)
assert spec is not None and spec.loader is not None
lower_body_ownership = importlib.util.module_from_spec(spec)
sys.modules["lower_body_ownership"] = lower_body_ownership
spec.loader.exec_module(lower_body_ownership)

BASE_COORDS = {
    "left_hip": (0.45, 0.55),
    "right_hip": (0.55, 0.55),
    "left_knee": (0.45, 0.70),
    "right_knee": (0.55, 0.70),
    "left_ankle": (0.45, 0.85),
    "right_ankle": (0.55, 0.85),
    "left_heel": (0.44, 0.88),
    "right_heel": (0.56, 0.88),
    "left_foot_index": (0.40, 0.90),
    "right_foot_index": (0.60, 0.90),
}


def pose_payload(
    *,
    drop: tuple[str, ...] = (),
    visibility: float = 0.9,
    dx: float = 0.0,
) -> list[dict[str, object]]:
    """Build a MediaPipe pose in the canonical 33-landmark order."""
    payload: list[dict[str, object]] = []
    for index, name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES):
        if name in drop:
            continue
        if index < 23:
            x, y = 0.5, 0.2
        else:
            x, y = BASE_COORDS[name]
        entry: dict[str, object] = {
            "x": x + dx,
            "y": y,
            "z": 0.0,
            "visibility": visibility,
        }
        if drop:
            entry["name"] = name
        payload.append(entry)
    return payload


def pose_document(*frames: list[dict[str, object]]) -> dict[str, object]:
    """Wrap pose payloads into a canonical pose document."""
    return {
        "schema": "mediapipe-pose-frames-v1",
        "frames": [
            {"frame_index": index, "poses": [{"landmarks": pose}]}
            for index, pose in enumerate(frames)
        ],
    }


def write_document(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class _Stdin:
    """A stdin stand-in that exposes the byte buffer the CLI reads."""

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)


def test_cli_emits_strict_json_and_owns_a_lower_body(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload()))

    exit_code = lower_body_ownership.main(["--input", str(document)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema"] == "lower-body-ownership-report-v1"
    assert payload["pose_source"]["label"] == "mediapipe-pose-landmarker"
    assert payload["pose_source"]["is_dwpose"] is False
    assert payload["pose_source"]["is_rtmpose"] is False
    assert payload["summary"]["owned_poses"] == 1
    assert payload["frames"][0]["poses"][0]["gates"]["chain_complete"] is True
    assert "NaN" not in captured.out
    assert "Infinity" not in captured.out


def test_cli_reports_the_input_digest_and_only_the_file_name(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload()))

    assert lower_body_ownership.main(["--input", str(document)]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["input"]["name"] == "poses.json"
    assert len(payload["input"]["sha256"]) == 64
    assert payload["input"]["bytes"] > 0
    assert str(tmp_path) not in captured.out


def test_cli_writes_the_report_file(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload()))
    output = tmp_path / "nested" / "report.json"

    assert lower_body_ownership.main(["--input", str(document), "--output", str(output)]) == 0

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema"] == "lower-body-ownership-report-v1"
    assert written == json.loads(capsys.readouterr().out)


def test_cli_reads_a_document_from_stdin(tmp_path: Path, capsys, monkeypatch) -> None:
    raw = json.dumps(pose_document(pose_payload())).encode("utf-8")
    monkeypatch.setattr(sys, "stdin", _Stdin(raw))

    exit_code = lower_body_ownership.main(["--input", "-"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["summary"]["owned_poses"] == 1


def test_cli_returns_three_when_no_frame_is_owned(tmp_path: Path, capsys) -> None:
    document = write_document(
        tmp_path / "poses.json",
        pose_document(pose_payload(drop=("left_foot_index",))),
    )

    exit_code = lower_body_ownership.main(["--input", str(document)])

    assert exit_code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["owned_poses"] == 0
    assert payload["frames"][0]["poses"][0]["owned"] is False
    assert payload["frames"][0]["poses"][0]["missing_required"] == ["left_foot_index"]


def test_cli_returns_two_for_a_document_that_is_not_json(tmp_path: Path, capsys) -> None:
    document = tmp_path / "poses.json"
    document.write_text("{not json", encoding="utf-8")

    exit_code = lower_body_ownership.main(["--input", str(document)])

    assert exit_code == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert capsys.readouterr().out == ""


def test_cli_returns_two_for_a_non_finite_json_constant(tmp_path: Path, capsys) -> None:
    document = tmp_path / "poses.json"
    document.write_text('{"frames": [{"poses": NaN}]}', encoding="utf-8")

    exit_code = lower_body_ownership.main(["--input", str(document)])

    assert exit_code == 2
    assert "non-finite constant" in capsys.readouterr().err


def test_cli_returns_two_for_a_missing_input_file(tmp_path: Path, capsys) -> None:
    exit_code = lower_body_ownership.main(["--input", str(tmp_path / "absent.json")])

    assert exit_code == 2
    assert "Lower-body ownership failed" in capsys.readouterr().err


def test_cli_returns_two_for_a_landmark_without_depth(tmp_path: Path, capsys) -> None:
    document: dict[str, Any] = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [
                    {
                        "landmarks": [
                            {
                                "name": "left_hip",
                                "x": 0.45,
                                "y": 0.55,
                                "visibility": 0.9,
                            }
                        ]
                    }
                ],
            }
        ]
    }
    path = write_document(tmp_path / "poses.json", document)

    exit_code = lower_body_ownership.main(["--input", str(path)])

    assert exit_code == 2
    assert "must carry x, y, and z" in capsys.readouterr().err


def test_cli_returns_two_for_duplicate_frame_indices(tmp_path: Path, capsys) -> None:
    document: dict[str, Any] = pose_document(pose_payload(), pose_payload())
    document["frames"][1]["frame_index"] = 0
    path = write_document(tmp_path / "poses.json", document)

    exit_code = lower_body_ownership.main(["--input", str(path)])

    assert exit_code == 2
    assert "frame indices must strictly increase" in capsys.readouterr().err


def test_cli_returns_two_for_an_unknown_required_chain(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload()))

    exit_code = lower_body_ownership.main(
        ["--input", str(document), "--required-chains", "left_leg,tail"]
    )

    assert exit_code == 2
    assert "unknown required chains" in capsys.readouterr().err


def test_cli_returns_two_for_an_impossible_threshold(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload()))

    exit_code = lower_body_ownership.main(["--input", str(document), "--min-visibility", "1.5"])

    assert exit_code == 2
    assert "min_visibility" in capsys.readouterr().err


def test_cli_threshold_flag_changes_the_ownership_verdict(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload(visibility=0.6)))

    exit_code = lower_body_ownership.main(["--input", str(document), "--min-visibility", "0.95"])

    assert exit_code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["thresholds"]["min_visibility"] == pytest.approx(0.95)
    assert payload["frames"][0]["poses"][0]["gates"]["visibility_ok"] is False


def test_cli_required_chain_flag_relaxes_the_gate(tmp_path: Path, capsys) -> None:
    document = write_document(
        tmp_path / "poses.json",
        pose_document(pose_payload(drop=("right_heel",))),
    )

    exit_code = lower_body_ownership.main(
        ["--input", str(document), "--required-chains", "left_leg,right_leg"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["thresholds"]["required_chains"] == ["left_leg", "right_leg"]


def test_cli_reports_a_temporal_jump_as_an_unowned_frame(tmp_path: Path, capsys) -> None:
    document = write_document(
        tmp_path / "poses.json",
        pose_document(pose_payload(), pose_payload(dx=0.9)),
    )

    exit_code = lower_body_ownership.main(["--input", str(document)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["owned_poses"] == 1
    assert payload["frames"][1]["poses"][0]["owned"] is False
    assert payload["frames"][1]["poses"][0]["gates"]["temporal_ok"] is False
    assert payload["frames"][1]["poses"][0]["jumps"]["pelvis_jump"] == pytest.approx(0.9)


def test_cli_output_is_deterministic(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload()))

    assert lower_body_ownership.main(["--input", str(document)]) == 0
    first = capsys.readouterr().out
    assert lower_body_ownership.main(["--input", str(document)]) == 0
    second = capsys.readouterr().out

    assert first == second


def test_cli_rejects_a_pose_document_that_names_a_non_mediapipe_landmark(
    tmp_path: Path, capsys
) -> None:
    document: dict[str, Any] = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [{"landmarks": [{"name": "left_thigh", "x": 0.5, "y": 0.5}]}],
            }
        ]
    }
    path = write_document(tmp_path / "poses.json", document)

    exit_code = lower_body_ownership.main(["--input", str(path)])

    assert exit_code == 2
    assert "unknown landmark" in capsys.readouterr().err


def test_cli_returns_two_when_the_report_cannot_be_written(tmp_path: Path, capsys) -> None:
    document = write_document(tmp_path / "poses.json", pose_document(pose_payload()))
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    exit_code = lower_body_ownership.main(
        ["--input", str(document), "--output", str(blocker / "report.json")]
    )

    assert exit_code == 2
    assert "Lower-body ownership failed" in capsys.readouterr().err
