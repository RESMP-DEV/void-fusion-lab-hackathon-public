"""Lossless gray-video helpers shared by segmentation and VOID preparation."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


class MediaError(RuntimeError):
    """A video probe, decode, or encode operation failed."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)


def _ffmpeg_framerate(fps: float) -> str:
    fraction = Fraction(fps).limit_denominator(1001)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def probe_video(path: Path) -> VideoInfo:
    """Probe a video using ffprobe and return frame dimensions/count and FPS."""
    if not path.is_file():
        raise MediaError(f"video does not exist: {path}")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        stream = json.loads(result.stdout)["streams"][0]
        numerator, denominator = stream["r_frame_rate"].split("/", 1)
        info = VideoInfo(
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=float(numerator) / float(denominator),
            frames=int(stream["nb_read_frames"]),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MediaError(f"could not parse ffprobe output for {path}") from error
    if info.width <= 0 or info.height <= 0 or info.frames <= 0 or info.fps <= 0:
        raise MediaError(f"invalid video metadata for {path}: {info}")
    return info


def probe_frame_timestamps(path: Path) -> list[Fraction]:
    """Return exact frame presentation timestamps in seconds.

    A frame timestamp is the exact rational product of ``frame.pts`` and the
    stream time base (``numerator / denominator`` seconds per tick), so two CFR
    streams can be compared even when their containers use different time bases.
    """

    if not path.is_file():
        raise MediaError(f"video does not exist: {path}")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base",
            "-show_entries",
            "frame=pts",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        numerator_text, denominator_text = stream["time_base"].split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        if numerator <= 0 or denominator <= 0:
            raise ValueError("nonpositive time base")
        timestamps = [
            Fraction(int(frame["pts"]) * numerator, denominator) for frame in payload["frames"]
        ]
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ) as error:
        raise MediaError(f"could not parse frame timestamps for {path}") from error
    if not timestamps:
        raise MediaError(f"no frame timestamps decoded from {path}")
    return timestamps


def assert_video_alignment(
    videos: dict[str, Path],
    *,
    width: int,
    height: int,
    frames: int,
) -> None:
    """Fail closed unless every named video has identical geometry, FPS, and PTS."""

    if not videos:
        raise MediaError("no videos were supplied for alignment validation")
    reference_name = next(iter(videos))
    reference_info = probe_video(videos[reference_name])
    expected_fps = Fraction(reference_info.fps).limit_denominator(1001)
    reference_pts = probe_frame_timestamps(videos[reference_name])
    if len(reference_pts) != frames:
        raise MediaError(
            f"{reference_name} has {len(reference_pts)} timestamped frames, expected {frames}"
        )
    for name, path in videos.items():
        info = probe_video(path)
        if (info.width, info.height, info.frames) != (width, height, frames):
            raise MediaError(
                f"{name} geometry/frame count is {info.width}x{info.height}x{info.frames}, "
                f"expected {width}x{height}x{frames}"
            )
        if Fraction(info.fps).limit_denominator(1001) != expected_fps:
            raise MediaError(
                f"{name} frame rate {info.fps} differs from {reference_name} {reference_info.fps}"
            )
        pts = probe_frame_timestamps(path)
        if len(pts) != frames:
            raise MediaError(f"{name} has {len(pts)} timestamped frames, expected {frames}")
        if pts != reference_pts:
            first_difference = next(
                index
                for index, (actual, expected) in enumerate(zip(pts, reference_pts, strict=False))
                if actual != expected
            )
            raise MediaError(
                f"{name} PTS differs from {reference_name} first at frame {first_difference}: "
                f"{pts[first_difference]} != {reference_pts[first_difference]}"
            )


def iter_gray_frames(
    video: Path,
    destination: Path,
    *,
    expected_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """Decode a video to gray arrays lazily, one frame at a time.

    The frames are extracted with the same lossless ffmpeg invocation used by
    ``decode_gray_frames`` and then loaded one at a time, so a long clip costs
    one frame of memory instead of one frame per second of video.
    """
    destination.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "format=gray",
            str(destination / "frame-%06d.png"),
        ]
    )
    files = sorted(destination.glob("frame-*.png"))
    if expected_frames is not None and len(files) != expected_frames:
        raise MediaError(f"decoded {len(files)} frames from {video}, expected {expected_frames}")
    if not files:
        raise MediaError(f"no frames decoded from {video}")
    for path in files:
        with Image.open(path) as image:
            yield np.asarray(image.convert("L")).copy()


def decode_gray_frames(
    video: Path,
    destination: Path,
    *,
    expected_frames: int | None = None,
) -> list[np.ndarray]:
    """Decode a video to gray arrays through ffmpeg and Pillow."""
    return list(iter_gray_frames(video, destination, expected_frames=expected_frames))


def decode_first_gray_frame(video: Path, destination: Path) -> np.ndarray:
    """Decode only the first frame of a video into one grayscale array.

    The anchor gate reads frame zero of a mask video and nothing
    else. Decoding the whole clip would materialize every frame on
    disk and in memory for a single-frame read, so ffmpeg is told to
    stop after the first frame (``-frames:v 1``) and exactly one
    image file is written. A video that yields no frame raises
    MediaError: an unreadable anchor fails closed.
    """
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "frame-000000.png"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "format=gray",
            "-y",
            str(output),
        ]
    )
    if not output.is_file():
        raise MediaError(f"no frame decoded from {video}")
    with Image.open(output) as image:
        return np.asarray(image.convert("L"))


def extract_color_frames(video: Path, destination: Path) -> list[Path]:
    """Extract original-color frames as PNG files for expert-model adapters."""
    destination.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            str(destination / "frame-%06d.png"),
        ]
    )
    frames = sorted(destination.glob("frame-*.png"))
    if not frames:
        raise MediaError(f"no color frames extracted from {video}")
    return frames


def iter_color_frames(video: Path, destination: Path) -> Iterator[np.ndarray]:
    """Extract original-color frames and yield them lazily as RGB arrays."""
    for path in extract_color_frames(video, destination):
        with Image.open(path) as image:
            yield np.asarray(image.convert("RGB")).copy()


class GrayVideoWriter:
    """Incrementally write a lossless gray video without retaining every frame."""

    def __init__(self, output: Path, *, width: int, height: int, fps: float):
        if width <= 0 or height <= 0 or fps <= 0:
            raise MediaError(f"invalid writer dimensions/fps: {width}x{height}@{fps}")
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        self._temporary = TemporaryDirectory(prefix=".samurai-video-", dir=output.parent)
        self._frame_dir = Path(self._temporary.name)
        output.parent.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width):
            raise MediaError(f"frame has shape {frame.shape}, expected {(self.height, self.width)}")
        Image.fromarray(frame.astype(np.uint8), mode="L").save(
            self._frame_dir / f"frame-{self.frames_written:06d}.png"
        )
        self.frames_written += 1

    def close(self) -> None:
        if self.frames_written == 0:
            raise MediaError("cannot close an empty mask video")
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                _ffmpeg_framerate(self.fps),
                "-i",
                str(self._frame_dir / "frame-%06d.png"),
                "-c:v",
                "libx264",
                "-bf",
                "0",
                "-qp",
                "0",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "yuv444p",
                "-y",
                str(self.output),
            ]
        )
        self._temporary.cleanup()

    def __enter__(self) -> GrayVideoWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._temporary.cleanup()


def write_gray_video(frames: list[np.ndarray], output: Path, *, fps: float) -> None:
    """Write a lossless H.264 yuv444p gray video suitable for VOID masks."""
    if not frames:
        raise MediaError("cannot write an empty mask video")
    shape = frames[0].shape
    if len(shape) != 2:
        raise MediaError(f"mask frame must be two-dimensional, got shape {shape}")
    if any(frame.shape != shape for frame in frames):
        raise MediaError("mask frames have inconsistent dimensions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".samurai-video-", dir=output.parent) as temporary:
        frame_dir = Path(temporary)
        for index, frame in enumerate(frames):
            Image.fromarray(frame.astype(np.uint8), mode="L").save(
                frame_dir / f"frame-{index:06d}.png"
            )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                _ffmpeg_framerate(fps),
                "-i",
                str(frame_dir / "frame-%06d.png"),
                "-c:v",
                "libx264",
                "-bf",
                "0",
                "-qp",
                "0",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "yuv444p",
                "-y",
                str(output),
            ]
        )
