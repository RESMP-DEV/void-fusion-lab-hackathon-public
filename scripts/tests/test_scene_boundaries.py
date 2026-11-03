from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scene_boundaries.py"
spec = importlib.util.spec_from_file_location("scene_boundaries", SCRIPT)
assert spec is not None and spec.loader is not None
scene_boundaries = importlib.util.module_from_spec(spec)
sys.modules["scene_boundaries"] = scene_boundaries
spec.loader.exec_module(scene_boundaries)


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="scene-boundary integration tests need local ffmpeg and ffprobe",
)


def write_synthetic_scenes(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=12:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=64x48:rate=12:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=64x48:rate=12:duration=1",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )


def read_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@requires_ffmpeg
def test_synthetic_three_scene_clip_makes_frame_accurate_restart_plan(tmp_path: Path) -> None:
    video = tmp_path / "scenes.mp4"
    output = tmp_path / "plan.json"
    write_synthetic_scenes(video)

    exit_code = scene_boundaries.main(
        [
            str(video),
            "--threshold",
            "0.10",
            "--min-scene-length",
            "4",
            "--context-frames",
            "3",
            "--output",
            str(output),
        ]
    )
    plan = read_plan(output)

    assert exit_code == 0
    assert plan["input"]["frame_count"] == 36
    assert plan["frame_validation"]["enumerated_frame_count"] == 36
    assert [cut["frame_index"] for cut in plan["cuts"]] == [12, 24]
    assert [cut["pts"] for cut in plan["cuts"]] == [12288, 24576]
    assert [segment["target"]["start_frame"] for segment in plan["segments"]] == [0, 12, 24]
    assert [segment["target"]["end_frame_exclusive"] for segment in plan["segments"]] == [
        12,
        24,
        36,
    ]
    assert all(
        segment["restart"]["frame_index"] == segment["target"]["start_frame"]
        for segment in plan["segments"]
    )
    assert plan["segments"][0]["context_after"] == {
        "start_frame": 12,
        "end_frame_exclusive": 15,
    }
    assert plan["segments"][1]["context_before"] == {
        "start_frame": 9,
        "end_frame_exclusive": 12,
    }
    assert plan["segments"][1]["generation_span"] == {
        "start_frame": 9,
        "end_frame_exclusive": 27,
    }
    assert plan["continuity_scorer"] == scene_boundaries.FALLBACK_SCORER_LABEL
    for cut in plan["cuts"]:
        score = cut["continuity"]
        assert score["scorer"] == scene_boundaries.FALLBACK_SCORER_LABEL
        assert score["semantic_claim"] is False
        assert math.isfinite(score["similarity"])
        assert 0.0 <= score["similarity"] <= 1.0


@requires_ffmpeg
def test_synthetic_clip_can_use_safe_argv_embedding_scorer(tmp_path: Path) -> None:
    video = tmp_path / "scenes.mp4"
    write_synthetic_scenes(video)
    wrapper = tmp_path / "embed.py"
    wrapper.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import sys
with open(sys.argv[1], "rb") as stream:
    value = hashlib.sha256(stream.read()).digest()[0]
print(json.dumps({"embedding": [value / 255.0, 0.5]}))
""",
        encoding="utf-8",
    )
    command = tmp_path / "embedding.json"
    command.write_text(
        json.dumps({"argv": [sys.executable, str(wrapper), scene_boundaries.FRAME_PLACEHOLDER]}),
        encoding="utf-8",
    )
    output = tmp_path / "plan.json"

    scene_boundaries.main(
        [
            str(video),
            "--threshold",
            "0.10",
            "--context-frames",
            "2",
            "--embedding-command-json",
            str(command),
            "--output",
            str(output),
        ]
    )
    plan = read_plan(output)

    assert plan["continuity_scorer"] == scene_boundaries.EMBEDDING_SCORER_LABEL
    assert len(plan["cuts"]) == 2
    for cut in plan["cuts"]:
        score = cut["continuity"]
        assert score["scorer"] == scene_boundaries.EMBEDDING_SCORER_LABEL
        assert score["command_sha256"] == scene_boundaries._sha256_file(command)
        assert score["command_contract"] == "argv-only-frame-path-placeholder"
        assert "command_argv_template" not in score
        assert str(wrapper) not in json.dumps(plan)
        assert score["embedding_dimensions"] == 2
        assert -1.0 <= score["similarity"] <= 1.0


def test_lavfi_scene_metadata_is_parsed_and_mapped_by_integer_pts() -> None:
    metadata_output = "\n".join(
        [
            "frame:0    pts:12288   pts_time:1",
            "lavfi.scene_score=0.443971",
            "",
            "frame:1    pts:24576   pts_time:2",
            "lavfi.scene_score=0.400000",
            "",
        ]
    )
    frames = [
        scene_boundaries.FrameIndex(12, 12288, 1.0),
        scene_boundaries.FrameIndex(24, 24576, 2.0),
    ]

    original = scene_boundaries.run_tool

    def fixed_run_tool(command: list[str], context: str, *, timeout: float | None = None) -> str:
        return metadata_output if command[0] == "ffmpeg" else "{}"

    scene_boundaries.run_tool = fixed_run_tool  # ty: ignore[unresolved-attribute]
    try:
        cuts = scene_boundaries.detect_candidate_cuts(
            Path("clip.mp4"), threshold=0.1, frame_indices=frames
        )
    finally:
        scene_boundaries.run_tool = original  # ty: ignore[unresolved-attribute]

    assert [(cut.frame_index, cut.pts, cut.scene_score) for cut in cuts] == [
        (12, 12288, 0.443971),
        (24, 24576, 0.4),
    ]


def test_minimum_scene_length_rejects_cuts_inside_target_spans() -> None:
    cuts = [
        scene_boundaries.CandidateCut(5, 50, 5.0, 0.9),
        scene_boundaries.CandidateCut(8, 80, 8.0, 0.9),
        scene_boundaries.CandidateCut(12, 120, 12.0, 0.9),
    ]

    accepted, rejected = scene_boundaries.enforce_minimum_scene_length(
        cuts,
        frame_count=15,
        minimum_frames=4,
    )

    assert [cut.frame_index for cut in accepted] == [5]
    assert [record["frame_index"] for record in rejected] == [8, 12]
    assert all(record["reason"] == "minimum-scene-length" for record in rejected)


def test_embedding_command_contract_is_argv_only_and_fails_closed(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps({"argv": ["/bin/echo", scene_boundaries.FRAME_PLACEHOLDER]}),
        encoding="utf-8",
    )
    invalid_placeholder = tmp_path / "invalid-placeholder.json"
    invalid_placeholder.write_text(
        json.dumps({"argv": ["/bin/echo", "{video_path}"]}),
        encoding="utf-8",
    )
    relative = tmp_path / "relative.json"
    relative.write_text(
        json.dumps({"argv": ["echo", scene_boundaries.FRAME_PLACEHOLDER]}),
        encoding="utf-8",
    )
    extra_key = tmp_path / "extra.json"
    extra_key.write_text(
        json.dumps(
            {
                "argv": ["/bin/echo", scene_boundaries.FRAME_PLACEHOLDER],
                "environment": {"SECRET": "value"},
            }
        ),
        encoding="utf-8",
    )

    assert scene_boundaries.load_embedding_command(valid).argv_template == (
        "/bin/echo",
        scene_boundaries.FRAME_PLACEHOLDER,
    )
    for path in (invalid_placeholder, relative, extra_key):
        with pytest.raises(scene_boundaries.SceneBoundaryError):
            scene_boundaries.load_embedding_command(path)


def test_plan_refuses_output_alias_before_probe_or_detection(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really video")

    def fail_probe(path: Path) -> None:
        raise AssertionError("probe must not run")

    original = scene_boundaries.probe_video
    scene_boundaries.probe_video = fail_probe  # ty: ignore[unresolved-attribute]
    try:
        with pytest.raises(SystemExit, match="must not overwrite"):
            scene_boundaries.main([str(video), "--output", str(video)])
    finally:
        scene_boundaries.probe_video = original  # ty: ignore[unresolved-attribute]
