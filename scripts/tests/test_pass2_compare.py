from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pass2_compare import main
from PIL import Image

from samurai.media import write_gray_video

SCRIPT = Path(__file__).resolve().parents[1] / "pass2_compare.py"
ROOT = Path(__file__).resolve().parents[2]
PYTHON_ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
}
SIZE = 16
FRAMES = 5
INSERTED = (slice(4, 8), slice(4, 8))
CLEANUP = (slice(2, 10), slice(2, 10))
FPS = "12"


def write_rgb_video(frames: list[np.ndarray], path: Path) -> None:
    staging = path.parent / f".{path.stem}-frames"
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(staging / f"frame-{index:06d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            FPS,
            "-i",
            str(staging / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def write_shifted_gray_video(frames: list[np.ndarray], path: Path, *, offset: float) -> None:
    """Write a gray mask video whose presentation timestamps are shifted."""
    staging = path.parent / f".{path.stem}-frames"
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(staging / f"frame-{index:06d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            FPS,
            "-i",
            str(staging / "frame-%06d.png"),
            "-vf",
            f"setpts=PTS+{offset}/TB",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def baseline_frames() -> list[np.ndarray]:
    frames = []
    for index in range(FRAMES):
        frame = np.full((SIZE, SIZE, 3), 20, dtype=np.uint8)
        frame[INSERTED] = 100
        frame[0, 0] = 40 + index
        frames.append(frame)
    return frames


def pass2_frames() -> list[np.ndarray]:
    frames = []
    for index in range(FRAMES):
        frame = np.full((SIZE, SIZE, 3), 20, dtype=np.uint8)
        frame[INSERTED] = 130
        frame[2, 2] = 60
        frame[0, 0] = 40 + index
        frames.append(frame)
    return frames


def mask_frames(region: tuple[slice, slice]) -> list[np.ndarray]:
    mask = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    mask[region] = 0
    return [mask.copy() for _ in range(FRAMES)]


def build_inputs(
    tmp_path: Path, *, pass2_size: int = SIZE, pass2_frames_count: int = FRAMES
) -> dict[str, Path]:
    baseline = tmp_path / "baseline.mp4"
    write_rgb_video(baseline_frames(), baseline)
    pass2 = tmp_path / "pass2.mp4"
    frames = pass2_frames()[:pass2_frames_count]
    if pass2_size != SIZE:
        frames = [
            np.pad(
                frame,
                ((0, pass2_size - SIZE), (0, pass2_size - SIZE), (0, 0)),
                mode="edge",
            )
            for frame in frames
        ]
    write_rgb_video(frames, pass2)
    inserted = tmp_path / "inserted.mp4"
    write_gray_video(mask_frames(INSERTED), inserted, fps=12.0)
    cleanup = tmp_path / "cleanup.mp4"
    write_gray_video(mask_frames(CLEANUP), cleanup, fps=12.0)
    return {
        "baseline": baseline,
        "pass2": pass2,
        "inserted_mask": inserted,
        "cleanup_mask": cleanup,
    }


def test_cli_reports_json_metrics_for_a_real_pair(tmp_path) -> None:
    inputs = build_inputs(tmp_path)
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["inserted_mask"]),
            "--cleanup-mask",
            str(inputs["cleanup_mask"]),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())

    assert exit_code == 0
    assert report["schema"] == "void-fusion-lab.pass2-compare.report.v3"
    assert report["frame_alignment"]["aligned"] is True
    assert report["frame_alignment"]["baseline"]["frames"] == FRAMES
    assert report["frame_timestamps"]["aligned"] is True
    assert report["frame_timestamps"]["pts_aligned"] is True
    assert report["frame_timestamps"]["frame_rate_character_aligned"] is True
    assert report["frame_timestamps"]["baseline"]["frames"] == FRAMES
    assert report["frame_timestamps"]["baseline"]["variable_frame_rate"] is False
    assert report["frame_timestamps"]["baseline"]["first_pts"] == "0/1"
    assert len(report["frame_timestamps"]["baseline"]["pts_digest"]) == 64
    assert (
        report["frame_timestamps"]["baseline"]["pts_digest"]
        == report["frame_timestamps"]["pass2"]["pts_digest"]
    )
    assert report["thresholds"]["cleanup_value"] == 127
    assert report["object_preservation"]["mae"] == pytest.approx(30.0, abs=1.0)
    assert report["object_preservation"]["changed_ratio"] == pytest.approx(1.0)
    assert report["object_preservation"]["psnr_db"] is not None
    assert report["cleanup_ring"]["mae"] == pytest.approx(40.0 / 48.0, abs=0.2)
    assert report["cleanup_ring"]["changed_ratio"] == pytest.approx(1.0 / 48.0, abs=0.01)
    assert report["cleanup_ring"]["frames_measured"] == FRAMES - 1
    assert report["cleanup_ring"]["pixels"] == 48 * (FRAMES - 1)
    assert report["background"]["mae"] > 0.0
    assert report["background"]["changed_ratio"] > 0.0
    assert report["temporal"]["frames"] == FRAMES
    assert report["temporal"]["frames_with_difference"] == FRAMES
    assert report["temporal"]["measured_frames"] == FRAMES - 1
    assert report["temporal"]["structural_lead_in"] == 1
    assert report["temporal"]["lead_in_changed_pixels"] == 17
    assert report["temporal"]["lead_in_violation"] is True
    assert report["temporal"]["no_changed_pixels"] is False
    assert report["temporal"]["exactly_identical"] is False
    assert report["non_claims"]
    assert all(len(record["sha256"]) == 64 for record in report["inputs"].values())


def test_cli_reports_the_first_frame_ring_as_background_for_the_remove_value(tmp_path) -> None:
    inputs = build_inputs(tmp_path)
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["inserted_mask"]),
            "--cleanup-mask",
            str(inputs["cleanup_mask"]),
            "--cleanup-value",
            "0",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())

    assert exit_code == 0
    assert report["thresholds"]["cleanup_value"] == 0
    assert report["cleanup_ring"]["frames_measured"] == FRAMES
    assert report["cleanup_ring"]["pixels"] == 48 * FRAMES
    assert report["background"]["mae"] == 0.0
    assert report["background"]["changed_ratio"] == 0.0


def steady_ring_cleanup_frames() -> list[np.ndarray]:
    """Differ from the baseline only inside the cleanup ring, and not on frame zero."""
    result = baseline_frames()
    for frame in result[1:]:
        frame[CLEANUP] = 220
        frame[INSERTED] = 100
    return result


def test_cli_scores_a_steady_affected_cleanup_as_stable(tmp_path) -> None:
    baseline = tmp_path / "steady-baseline.mp4"
    pass2 = tmp_path / "steady-pass2.mp4"
    write_rgb_video(baseline_frames(), baseline)
    write_rgb_video(steady_ring_cleanup_frames(), pass2)
    inserted = tmp_path / "steady-inserted.mp4"
    cleanup = tmp_path / "steady-cleanup.mp4"
    write_gray_video(mask_frames(INSERTED), inserted, fps=12.0)
    write_gray_video(mask_frames(CLEANUP), cleanup, fps=12.0)
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "--baseline",
            str(baseline),
            "--pass2",
            str(pass2),
            "--inserted-mask",
            str(inserted),
            "--cleanup-mask",
            str(cleanup),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())

    assert exit_code == 0
    assert report["thresholds"]["cleanup_value"] == 127
    assert report["cleanup_ring"]["frames_measured"] == FRAMES - 1
    assert report["cleanup_ring"]["changed_ratio"] == pytest.approx(1.0)
    temporal = report["temporal"]
    assert temporal["frames"] == FRAMES
    assert temporal["structural_lead_in"] == 1
    assert temporal["measured_frames"] == FRAMES - 1
    assert temporal["lead_in_changed_pixels"] == 0
    assert temporal["lead_in_violation"] is False
    assert temporal["frames_with_difference"] == FRAMES - 1
    assert temporal["per_frame_mass"] == [0] + [48] * (FRAMES - 1)
    assert temporal["mass_pop"] == 0.0
    assert temporal["centroid_pop"] == 0.0
    assert temporal["temporal_pop_score"] == 0.0


def test_cli_fails_closed_on_geometry_mismatch(tmp_path) -> None:
    inputs = build_inputs(tmp_path, pass2_size=20)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["inserted_mask"]),
            "--cleanup-mask",
            str(inputs["cleanup_mask"]),
        ],
        capture_output=True,
        text=True,
        env=PYTHON_ENV,
    )

    assert result.returncode == 2
    assert "geometry mismatch" in result.stderr


def test_cli_fails_closed_on_frame_count_mismatch(tmp_path) -> None:
    inputs = build_inputs(tmp_path, pass2_frames_count=3)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["inserted_mask"]),
            "--cleanup-mask",
            str(inputs["cleanup_mask"]),
        ],
        capture_output=True,
        text=True,
        env=PYTHON_ENV,
    )

    assert result.returncode == 2
    assert "frame count mismatch" in result.stderr


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the comparator script in a child process."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env=PYTHON_ENV,
    )


def test_cli_fails_closed_on_a_frame_rate_mismatch(tmp_path) -> None:
    inputs = build_inputs(tmp_path)
    write_gray_video(mask_frames(INSERTED), tmp_path / "inserted-10fps.mp4", fps=10.0)

    result = run_cli(
        "--baseline",
        str(inputs["baseline"]),
        "--pass2",
        str(inputs["pass2"]),
        "--inserted-mask",
        str(tmp_path / "inserted-10fps.mp4"),
        "--cleanup-mask",
        str(inputs["cleanup_mask"]),
    )

    assert result.returncode == 2
    assert "frame rate mismatch" in result.stderr


def test_cli_fails_closed_on_a_timestamp_mismatch(tmp_path) -> None:
    inputs = build_inputs(tmp_path)
    shifted = tmp_path / "cleanup-shifted.mp4"
    write_shifted_gray_video(mask_frames(CLEANUP), shifted, offset=0.5)

    result = run_cli(
        "--baseline",
        str(inputs["baseline"]),
        "--pass2",
        str(inputs["pass2"]),
        "--inserted-mask",
        str(inputs["inserted_mask"]),
        "--cleanup-mask",
        str(shifted),
    )

    assert result.returncode == 2
    assert "timestamp mismatch at frame 0" in result.stderr


def test_cli_rejects_a_non_finite_threshold(tmp_path) -> None:
    inputs = build_inputs(tmp_path)

    result = run_cli(
        "--baseline",
        str(inputs["baseline"]),
        "--pass2",
        str(inputs["pass2"]),
        "--inserted-mask",
        str(inputs["inserted_mask"]),
        "--cleanup-mask",
        str(inputs["cleanup_mask"]),
        "--changed-threshold",
        "nan",
    )

    assert result.returncode == 2
    assert "must be finite" in result.stderr


def test_input_digests_describe_the_bytes_present_when_analysis_started(
    tmp_path, monkeypatch
) -> None:
    import hashlib

    import pass2_compare

    inputs = build_inputs(tmp_path)
    output = tmp_path / "report.json"
    original = inputs["baseline"].read_bytes()

    real_compare = pass2_compare.compare_videos

    def replace_baseline_during_analysis(**arguments):
        comparison = real_compare(**arguments)
        inputs["baseline"].write_bytes(b"changed during analysis")
        return comparison

    monkeypatch.setattr(pass2_compare, "compare_videos", replace_baseline_during_analysis)
    pass2_compare.main(
        [
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["inserted_mask"]),
            "--cleanup-mask",
            str(inputs["cleanup_mask"]),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())

    assert report["inputs"]["baseline"]["sha256"] == hashlib.sha256(original).hexdigest()


def test_cli_reports_a_media_subprocess_failure_as_exit_two(tmp_path) -> None:
    inputs = build_inputs(tmp_path)
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video", encoding="utf-8")

    result = run_cli(
        "--baseline",
        str(broken),
        "--pass2",
        str(inputs["pass2"]),
        "--inserted-mask",
        str(inputs["inserted_mask"]),
        "--cleanup-mask",
        str(inputs["cleanup_mask"]),
    )

    assert result.returncode == 2
    assert "Pass 2 comparison failed" in result.stderr
    assert "Traceback" not in result.stderr
