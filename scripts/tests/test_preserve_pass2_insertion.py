"""Checks for enforcing the Pass-2 inserted-object keep contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from samurai.media import decode_gray_frames, iter_color_frames, probe_video, write_gray_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preserve_pass2_insertion import main, preserve_frame, write_yuv444_video  # noqa: E402

HEIGHT = 12
WIDTH = 16
FRAMES = 3
FPS = 24.0


def mask_frames() -> list[np.ndarray]:
    frames = []
    for index in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        frame[3:9, 4 + index : 10 + index] = 0
        frames.append(frame)
    return frames


def baseline_frames() -> list[np.ndarray]:
    frames = []
    for index, mask in enumerate(mask_frames()):
        frame = np.full((HEIGHT, WIDTH, 3), 70, dtype=np.uint8)
        frame[mask == 0] = (210, 160, 20)
        frame[4, 6 + index] = (255, 255, 255)
        frames.append(frame)
    return frames


def pass2_frames() -> list[np.ndarray]:
    frames = []
    for index, mask in enumerate(mask_frames()):
        frame = np.full((HEIGHT, WIDTH, 3), 130, dtype=np.uint8)
        frame[mask == 0] = (10, 20, 200)
        frame[:, : 4 + index] = (180, 80, 30)
        frames.append(frame)
    return frames


def write_inputs(tmp_path: Path) -> dict[str, Path]:
    masks = mask_frames()
    write_gray_video(masks, tmp_path / "mask.mp4", fps=FPS)
    write_yuv444_video(baseline_frames(), tmp_path / "baseline.mp4", fps=FPS)
    write_yuv444_video(pass2_frames(), tmp_path / "pass2.mp4", fps=FPS)
    return {
        "baseline": tmp_path / "baseline.mp4",
        "pass2": tmp_path / "pass2.mp4",
        "mask": tmp_path / "mask.mp4",
    }


def test_preserve_frame_keeps_object_and_pass2_surroundings() -> None:
    preserved = preserve_frame(baseline_frames()[0], pass2_frames()[0], mask_frames()[0])
    keep = mask_frames()[0] == 0

    assert np.array_equal(preserved[keep], baseline_frames()[0][keep])
    assert np.array_equal(preserved[~keep], pass2_frames()[0][~keep])


def test_cli_writes_yuv444_output_report_and_brand_gate(tmp_path: Path) -> None:
    inputs = write_inputs(tmp_path)
    output = tmp_path / "preserved.mp4"
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["mask"]),
            "--output",
            str(output),
            "--report",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    info = probe_video(output)
    with tempfile.TemporaryDirectory() as temporary:
        decode_root = Path(temporary)
        frames = list(iter_color_frames(output, decode_root / "frames"))
        masks = list(decode_gray_frames(inputs["mask"], decode_root / "masks"))
    for actual, baseline, result, mask in zip(
        frames, baseline_frames(), pass2_frames(), masks, strict=True
    ):
        keep = mask == 0
        assert np.max(np.abs(actual[keep].astype(int) - baseline[keep].astype(int))) <= 2
        assert np.max(np.abs(actual[~keep].astype(int) - result[~keep].astype(int))) <= 2

    assert exit_code == 0
    pix_fmt = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert pix_fmt == "yuv444p"
    assert info.frames == FRAMES
    assert info.fps == FPS
    assert report["summary"]["brand_gate"]["passed"] is True
    assert report["summary"]["min_brand_pixels"] >= 24


def test_cli_rejects_an_existing_output(tmp_path: Path) -> None:
    inputs = write_inputs(tmp_path)
    output = tmp_path / "preserved.mp4"
    output.write_bytes(b"occupied")

    exit_code = main(
        [
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["mask"]),
            "--output",
            str(output),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 2


def test_cli_rejects_output_and_report_aliasing(tmp_path: Path) -> None:
    inputs = write_inputs(tmp_path)
    output = tmp_path / "preserved.mp4"

    exit_code = main(
        [
            "--baseline",
            str(inputs["baseline"]),
            "--pass2",
            str(inputs["pass2"]),
            "--inserted-mask",
            str(inputs["mask"]),
            "--output",
            str(output),
            "--report",
            str(output),
        ]
    )

    assert exit_code == 2
    assert not output.exists()
