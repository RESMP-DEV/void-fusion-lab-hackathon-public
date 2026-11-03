#!/usr/bin/env python3
"""Build deterministic, frame-accurate scene-restart plans with FFmpeg.

The detector never asks a model what a scene is. It runs FFmpeg's numeric
LAVFI ``scene`` expression, maps each selected PTS back to a coded frame index
with ffprobe, and emits one contiguous, non-overlapping segment plan. A segment
restart begins at its target start frame; optional context ranges are separate
so a generator can look across a boundary without overwriting output frames.

Continuity is intentionally only an interface here. Without an embedding
command, each boundary is scored with a clearly labeled grayscale-MAE fallback.
With ``--embedding-command-json``, an operator-owned command is invoked through
an argv list (never a shell), with only a frame-image path substituted, a
sanitized environment, and a timeout. This script itself sends no model,
credential, or network request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DETECTOR_LABEL = "ffmpeg-lavfi-scene-select-v1"
FALLBACK_SCORER_LABEL = "deterministic-grayscale-mae-v1"
EMBEDDING_SCORER_LABEL = "safe-argv-embedding-cosine-v1"
SCHEMA_VERSION = 1
FRAME_PLACEHOLDER = "{frame_path}"
PLACEHOLDER_PATTERN = re.compile(r"\{[a-z0-9_]+\}")
MAX_EMBEDDING_COMMAND_ARGUMENTS = 64
MAX_EMBEDDING_COMMAND_LENGTH = 4096
MAX_EMBEDDING_COMMAND_JSON_BYTES = 1024 * 1024
MAX_EMBEDDING_DIMENSIONS = 1_000_000


class SceneBoundaryError(RuntimeError):
    """A scene-plan input or FFmpeg/ffprobe result is unusable."""


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    width: int
    height: int
    fps: float
    fps_text: str
    sha256: str


@dataclass(frozen=True)
class CandidateCut:
    frame_index: int
    pts: int
    pts_time: float
    scene_score: float


@dataclass(frozen=True)
class FrameIndex:
    index: int
    pts: int
    pts_time: float


@dataclass(frozen=True)
class EmbeddingCommand:
    argv_template: tuple[str, ...]
    sha256: str


def _tool_failure(error: subprocess.CalledProcessError) -> str:
    lines = (error.stderr or error.stdout or "").strip().splitlines()
    suffix = f"; {lines[-1]}" if lines else ""
    return f"exit {error.returncode}{suffix}"


def run_tool(command: list[str], context: str, *, timeout: float | None = None) -> str:
    """Run an external tool without a shell and surface a bounded failure."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        raise SceneBoundaryError(f"{context}: {_tool_failure(error)}") from error
    except FileNotFoundError as error:
        raise SceneBoundaryError(f"{context}: {command[0]} is not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise SceneBoundaryError(f"{context}: timed out after {error.timeout} seconds") from error
    return result.stdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_stream(path: Path) -> dict[str, Any]:
    output = run_tool(
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
        ],
        f"ffprobe failed to probe {path.name}",
    )
    try:
        streams = json.loads(output)["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SceneBoundaryError(
            f"ffprobe returned no parsable stream list for {path.name}"
        ) from error
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise SceneBoundaryError(f"ffprobe found no usable video stream in {path.name}")
    return streams[0]


def _required_stream_value(stream: dict[str, Any], key: str, path: Path) -> str:
    value = stream.get(key)
    if value is None or (isinstance(value, str) and value.strip() in ("", "N/A")):
        raise SceneBoundaryError(
            f"ffprobe reports {path.name} {key} as {value!r}; refusing to guess"
        )
    return str(value)


def probe_video(path: Path) -> VideoMetadata:
    """Probe exact coded dimensions, frame rate, and decoded frame count."""
    stream = _video_stream(path)
    try:
        width = int(_required_stream_value(stream, "width", path))
        height = int(_required_stream_value(stream, "height", path))
        frame_count = int(_required_stream_value(stream, "nb_read_frames", path))
        rate_text = _required_stream_value(stream, "r_frame_rate", path)
        numerator_text, separator, denominator_text = rate_text.partition("/")
        if not separator:
            raise ValueError(rate_text)
        numerator = float(numerator_text)
        denominator = float(denominator_text)
        fps = numerator / denominator
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise SceneBoundaryError(
            f"ffprobe returned unusable video metadata for {path.name}"
        ) from error

    if width <= 0 or height <= 0 or frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise SceneBoundaryError(
            f"ffprobe returned invalid video metadata for {path.name}: "
            f"{width}x{height}, {frame_count} frames, {rate_text!r}"
        )
    return VideoMetadata(
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        fps_text=rate_text,
        sha256=_sha256_file(path),
    )


def _frame_number(value: Any, key: str) -> int:
    if isinstance(value, bool) or value is None:
        raise SceneBoundaryError(f"ffprobe frame record has invalid {key}: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise SceneBoundaryError(f"ffprobe frame record has invalid {key}: {value!r}") from error


def _frame_seconds(value: Any, pts: int) -> float:
    if value is None or isinstance(value, bool):
        raise SceneBoundaryError(f"ffprobe frame {pts} has invalid pts_time: {value!r}")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise SceneBoundaryError(f"ffprobe frame {pts} has invalid pts_time: {value!r}") from error
    if not math.isfinite(seconds):
        raise SceneBoundaryError(f"ffprobe frame {pts} has non-finite pts_time: {value!r}")
    return seconds


def validate_frame_indices(path: Path, metadata: VideoMetadata) -> list[FrameIndex]:
    """Return every coded video frame's decode position, integer PTS, and time.

    ``nb_read_frames`` and this per-frame record must agree. The detector uses
    integer PTS, not a rounded wall-clock guess, to map LAVFI metadata to a
    frame index.
    """
    output = run_tool(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts,pts_time,media_type",
            "-of",
            "json",
            str(path),
        ],
        f"ffprobe failed to enumerate frames in {path.name}",
    )
    try:
        frames = json.loads(output)["frames"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SceneBoundaryError(
            f"ffprobe returned no parsable frame list for {path.name}"
        ) from error
    if not isinstance(frames, list):
        raise SceneBoundaryError(f"ffprobe returned a non-list frame record for {path.name}")

    records: list[FrameIndex] = []
    seen_pts: set[int] = set()
    previous_pts: int | None = None
    for position, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("media_type") != "video":
            raise SceneBoundaryError(
                f"ffprobe returned a non-video frame at decode position {position} in {path.name}"
            )
        pts = _frame_number(frame.get("pts"), "pts")
        if pts in seen_pts:
            raise SceneBoundaryError(f"ffprobe returned duplicate video PTS {pts} in {path.name}")
        if previous_pts is not None and pts < previous_pts:
            raise SceneBoundaryError(
                f"ffprobe returned decreasing video PTS {previous_pts}, then {pts} in {path.name}"
            )
        seen_pts.add(pts)
        previous_pts = pts
        records.append(
            FrameIndex(index=position, pts=pts, pts_time=_frame_seconds(frame.get("pts_time"), pts))
        )

    if len(records) != metadata.frame_count:
        raise SceneBoundaryError(
            f"ffprobe frame validation failed for {path.name}: stream reports "
            f"{metadata.frame_count} frames, frame enumeration returned {len(records)}"
        )
    return records


def scene_filter(threshold: float) -> str:
    """Build only the fixed LAVFI expression from a validated numeric threshold."""
    threshold_text = format(threshold, ".12g")
    return f"select='gt(scene\\,{threshold_text})',metadata=print:file=-"


def detect_candidate_cuts(
    path: Path,
    *,
    threshold: float,
    frame_indices: list[FrameIndex],
) -> list[CandidateCut]:
    """Run FFmpeg scene selection and map every selected PTS to a frame index."""
    output = run_tool(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-noautorotate",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            scene_filter(threshold),
            "-an",
            "-f",
            "null",
            "-",
        ],
        "ffmpeg failed to run LAVFI scene detection",
    )

    pts_to_frame = {frame.pts: frame for frame in frame_indices}
    header_pattern = re.compile(r"frame:\d+\s+pts:(-?\d+)(?:\s+.*)?")
    cuts: list[CandidateCut] = []
    current: dict[str, int | float] = {}

    def finish_block() -> None:
        if not current:
            return
        try:
            pts = int(current["pts"])
            score = float(current["lavfi.scene_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise SceneBoundaryError(
                f"ffmpeg returned an incomplete scene metadata block: {current!r}"
            ) from error
        frame = pts_to_frame.get(pts)
        if frame is None:
            raise SceneBoundaryError(
                f"ffmpeg scene PTS {pts} is absent from ffprobe frame validation"
            )
        if not math.isfinite(score):
            raise SceneBoundaryError(
                f"ffmpeg returned non-finite scene score at frame {frame.index}"
            )
        cuts.append(
            CandidateCut(
                frame_index=frame.index,
                pts=pts,
                pts_time=frame.pts_time,
                scene_score=score,
            )
        )

    for line in output.splitlines():
        if line.startswith("frame:"):
            finish_block()
            current = {}
            match = header_pattern.fullmatch(line)
            if match is None:
                raise SceneBoundaryError(
                    f"ffmpeg returned an invalid scene metadata header: {line!r}"
                )
            current["pts"] = int(match.group(1))
        elif line.startswith("lavfi.scene_score="):
            try:
                current["lavfi.scene_score"] = float(line.partition("=")[2])
            except ValueError as error:
                raise SceneBoundaryError(
                    f"ffmpeg returned an invalid scene score: {line!r}"
                ) from error
    finish_block()

    return sorted(cuts, key=lambda cut: (cut.frame_index, cut.pts, cut.scene_score))


def enforce_minimum_scene_length(
    candidates: list[CandidateCut],
    *,
    frame_count: int,
    minimum_frames: int,
) -> tuple[list[CandidateCut], list[dict[str, Any]]]:
    """Keep only cuts that leave a non-empty target span on both sides."""
    accepted: list[CandidateCut] = []
    rejected: list[dict[str, Any]] = []
    previous_boundary = 0
    for candidate in candidates:
        latest_start = frame_count - minimum_frames
        if (
            candidate.frame_index - previous_boundary < minimum_frames
            or frame_count - candidate.frame_index < minimum_frames
            or candidate.frame_index > latest_start
        ):
            rejected.append(
                {
                    "frame_index": candidate.frame_index,
                    "pts": candidate.pts,
                    "pts_time": candidate.pts_time,
                    "scene_score": candidate.scene_score,
                    "reason": "minimum-scene-length",
                }
            )
            continue
        accepted.append(candidate)
        previous_boundary = candidate.frame_index
    return accepted, rejected


def _exclusive_range(start: int, end: int) -> dict[str, int]:
    return {"start_frame": start, "end_frame_exclusive": end}


def build_plan(
    *,
    path: Path,
    metadata: VideoMetadata,
    cuts: list[CandidateCut],
    rejected_cuts: list[dict[str, Any]],
    threshold: float,
    minimum_scene_length: int,
    context_frames: int,
    continuity_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    boundaries = [0, *(cut.frame_index for cut in cuts), metadata.frame_count]
    segments: list[dict[str, Any]] = []
    segment_pairs = list(zip(boundaries, boundaries[1:], strict=False))
    for segment_index, (start, end) in enumerate(segment_pairs):
        before_start = (
            start
            if segment_index == 0
            else max(boundaries[segment_index - 1], start - context_frames)
        )
        after_end = (
            end
            if segment_index + 2 >= len(boundaries)
            else min(
                boundaries[segment_index + 2],
                end + context_frames,
            )
        )
        context_start = before_start
        context_end = after_end
        segments.append(
            {
                "index": segment_index,
                "target": _exclusive_range(start, end),
                "restart": {
                    "frame_index": start,
                    "reason": "generation-restart",
                },
                "context_before": _exclusive_range(before_start, start),
                "context_after": _exclusive_range(end, after_end),
                "generation_span": _exclusive_range(context_start, context_end),
            }
        )

    score_by_frame = {score["frame_index"]: score["continuity"] for score in continuity_scores}
    cut_records: list[dict[str, Any]] = []
    for boundary_index, cut in enumerate(cuts):
        left_segment_end = boundaries[boundary_index + 1]
        right_segment_end = boundaries[boundary_index + 2]
        cut_records.append(
            {
                "frame_index": cut.frame_index,
                "pts": cut.pts,
                "pts_time": cut.pts_time,
                "scene_score": cut.scene_score,
                "left_context": _exclusive_range(
                    max(boundaries[boundary_index], cut.frame_index - context_frames),
                    left_segment_end,
                ),
                "right_context": _exclusive_range(
                    cut.frame_index,
                    min(right_segment_end, cut.frame_index + context_frames),
                ),
                "continuity": score_by_frame[cut.frame_index],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "detector": {
            "label": DETECTOR_LABEL,
            "threshold": threshold,
            "minimum_scene_length_frames": minimum_scene_length,
            "context_frames": context_frames,
        },
        "input": {
            "sha256": metadata.sha256,
            "frame_count": metadata.frame_count,
            "coded_size": {"width": metadata.width, "height": metadata.height},
            "frame_rate": {"text": metadata.fps_text, "frames_per_second": metadata.fps},
            "name": path.name,
        },
        "frame_validation": {
            "tool": "ffprobe-frame-enumeration-v1",
            "stream_frame_count": metadata.frame_count,
            "enumerated_frame_count": metadata.frame_count,
            "pts_mapping": "integer-exact",
        },
        "cuts": cut_records,
        "rejected_cuts": rejected_cuts,
        "segments": segments,
        "plan_invariants": [
            "segments are contiguous and non-overlapping in frame-index order",
            "end_frame_exclusive is excluded and end equals the next start",
            "a generation restart begins at its segment target start frame",
            "context ranges stay outside their target range and inside adjacent segments",
        ],
    }


def _frame_select_filter(frame_indices: list[int]) -> str:
    expressions = [f"eq(n\\,{index})" for index in frame_indices]
    return f"select='{'+'.join(expressions)}'"


def extract_frames(
    video: Path,
    frame_indices: list[int],
    destination: Path,
    *,
    width: int,
    height: int,
) -> dict[int, Path]:
    """Decode exactly the requested coded-frame indices as PNG images."""
    if not frame_indices:
        return {}
    run_tool(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-noautorotate",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            _frame_select_filter(frame_indices),
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(len(frame_indices)),
            str(destination / "frame-%08d.png"),
        ],
        f"ffmpeg failed to extract validation frames {frame_indices}",
    )
    files = sorted(destination.glob("frame-*.png"))
    if len(files) != len(frame_indices):
        raise SceneBoundaryError(
            f"ffmpeg extracted {len(files)} frames for {frame_indices}; "
            f"expected {len(frame_indices)}"
        )
    for path in files:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise SceneBoundaryError(
                    f"decoded frame {path.name} has size {image.size}; expected {(width, height)}"
                )
    return dict(zip(frame_indices, files, strict=True))


def fallback_continuity_score(frame_index: int, left: Path, right: Path) -> dict[str, Any]:
    """Return the explicitly deterministic, non-semantic grayscale fallback."""
    with Image.open(left) as left_image, Image.open(right) as right_image:
        left_gray = np.asarray(left_image.convert("L"), dtype=np.float64)
        right_gray = np.asarray(right_image.convert("L"), dtype=np.float64)
    if left_gray.shape != right_gray.shape or left_gray.size == 0:
        raise SceneBoundaryError(
            f"fallback scorer received incompatible frames around {frame_index}"
        )
    total_error = math.fsum(
        abs(int(left) - int(right))
        for left, right in zip(left_gray.flat, right_gray.flat, strict=True)
    )
    mae = total_error / (left_gray.size * 255.0)
    return {
        "scorer": FALLBACK_SCORER_LABEL,
        "metric": "cosine-free normalized grayscale mean absolute error",
        "similarity": 1.0 - mae,
        "mae": mae,
        "semantic_claim": False,
    }


def load_embedding_command(path: Path) -> EmbeddingCommand:
    """Load a local argv embedding contract without invoking anything."""
    try:
        command_size = path.stat().st_size
    except OSError as error:
        raise SceneBoundaryError(f"embedding command file is unreadable: {path.name}") from error
    if command_size > MAX_EMBEDDING_COMMAND_JSON_BYTES:
        raise SceneBoundaryError(
            f"embedding command JSON exceeds {MAX_EMBEDDING_COMMAND_JSON_BYTES} bytes"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SceneBoundaryError(
            f"embedding command file is not valid UTF-8 JSON: {path.name}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {"argv"}:
        raise SceneBoundaryError(
            "embedding command JSON must contain exactly {'argv': [...]}; "
            f"got keys {sorted(payload) if isinstance(payload, dict) else type(payload).__name__}"
        )
    argv = payload["argv"]
    if (
        not isinstance(argv, list)
        or not 1 <= len(argv) <= MAX_EMBEDDING_COMMAND_ARGUMENTS
        or any(not isinstance(argument, str) for argument in argv)
        or any(not argument for argument in argv)
        or any("\x00" in argument for argument in argv)
        or any(len(argument) > MAX_EMBEDDING_COMMAND_LENGTH for argument in argv)
    ):
        raise SceneBoundaryError(
            f"embedding command argv must be 1-{MAX_EMBEDDING_COMMAND_ARGUMENTS} non-empty strings"
        )
    if not any(argument == FRAME_PLACEHOLDER for argument in argv):
        raise SceneBoundaryError(f"embedding command argv must include {FRAME_PLACEHOLDER}")
    for argument in argv:
        for match in PLACEHOLDER_PATTERN.findall(argument):
            if match != FRAME_PLACEHOLDER:
                raise SceneBoundaryError(
                    f"embedding command has unsupported placeholder {match}; "
                    f"only {FRAME_PLACEHOLDER} is allowed"
                )
    executable = Path(argv[0])
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise SceneBoundaryError(
            f"embedding executable must be an existing absolute executable path, got {argv[0]!r}"
        )
    return EmbeddingCommand(
        argv_template=tuple(argv),
        sha256=_sha256_file(path),
    )


def _parse_embedding(stdout: str, command: str) -> list[float]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise SceneBoundaryError(f"embedding command returned invalid JSON: {command}") from error
    raw = payload.get("embedding") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw or len(raw) > MAX_EMBEDDING_DIMENSIONS:
        raise SceneBoundaryError(
            f"embedding command must return a non-empty JSON vector of at most "
            f"{MAX_EMBEDDING_DIMENSIONS} numbers: {command}"
        )
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SceneBoundaryError(f"embedding command returned a non-numeric value: {command}")
        number = float(value)
        if not math.isfinite(number):
            raise SceneBoundaryError(f"embedding command returned a non-finite value: {command}")
        values.append(number)
    return values


def run_embedding_command(
    command: EmbeddingCommand,
    frame_path: Path,
    *,
    timeout: float,
) -> list[float]:
    """Execute only the checked argv template with one frame path substituted."""
    argv = [
        argument.replace(FRAME_PLACEHOLDER, str(frame_path)) for argument in command.argv_template
    ]
    environment = {
        "LC_ALL": "C",
        "TZ": "UTC",
        "PATH": os.defpath,
    }
    try:
        result = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip().splitlines()
        suffix = f"; {detail[-1]}" if detail else ""
        raise SceneBoundaryError(
            f"embedding command failed with exit {error.returncode}{suffix}"
        ) from error
    except FileNotFoundError as error:
        raise SceneBoundaryError("embedding executable disappeared before launch") from error
    except subprocess.TimeoutExpired as error:
        raise SceneBoundaryError(f"embedding command timed out after {timeout} seconds") from error
    return _parse_embedding(result.stdout, command.argv_template[0])


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise SceneBoundaryError("embedding dimensions differ across boundary frames")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise SceneBoundaryError("embedding command returned a zero vector")
    similarity = math.fsum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, similarity))


def score_boundaries(
    cuts: list[CandidateCut],
    *,
    video: Path,
    metadata: VideoMetadata,
    work_directory: Path,
    embedding_command: EmbeddingCommand | None,
    embedding_timeout: float,
) -> list[dict[str, Any]]:
    """Score adjacent frames at every accepted cut with the selected scorer."""
    requested_indices = sorted(
        {index for cut in cuts for index in (cut.frame_index - 1, cut.frame_index)}
    )
    frame_directory = work_directory / "boundary-frames"
    frame_directory.mkdir()
    frame_paths = extract_frames(
        video,
        requested_indices,
        frame_directory,
        width=metadata.width,
        height=metadata.height,
    )
    embeddings: dict[int, list[float]] = {}
    scores: list[dict[str, Any]] = []
    for cut in cuts:
        left_index = cut.frame_index - 1
        right_index = cut.frame_index
        if embedding_command is None:
            continuity = fallback_continuity_score(
                cut.frame_index,
                frame_paths[left_index],
                frame_paths[right_index],
            )
        else:
            for index in (left_index, right_index):
                if index not in embeddings:
                    embeddings[index] = run_embedding_command(
                        embedding_command,
                        frame_paths[index],
                        timeout=embedding_timeout,
                    )
            continuity = {
                "scorer": EMBEDDING_SCORER_LABEL,
                "metric": "deterministic cosine similarity",
                "similarity": _cosine_similarity(embeddings[left_index], embeddings[right_index]),
                "embedding_dimensions": len(embeddings[left_index]),
                "command_sha256": embedding_command.sha256,
                "command_contract": "argv-only-frame-path-placeholder",
                "semantic_claim": False,
            }
        scores.append({"frame_index": cut.frame_index, "continuity": continuity})
    return scores


def _write_json(destination: Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination is None:
        sys.stdout.write(text)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min-scene-length", type=int, default=1)
    parser.add_argument("--context-frames", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--embedding-command-json", type=Path)
    parser.add_argument("--embedding-timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("scene threshold must be a finite value between 0 and 1")
    if args.min_scene_length < 1:
        raise SystemExit("minimum scene length must be at least one frame")
    if args.context_frames < 0:
        raise SystemExit("context frame count cannot be negative")
    if not math.isfinite(args.embedding_timeout) or args.embedding_timeout <= 0:
        raise SystemExit("embedding timeout must be a positive finite number")
    if not args.video.is_file():
        raise SystemExit(f"input video does not exist: {args.video}")
    if args.output is not None and args.video.resolve() == args.output.resolve():
        raise SystemExit("output plan must not overwrite the input video")
    if (
        args.output is not None
        and args.embedding_command_json is not None
        and args.output.resolve() == args.embedding_command_json.resolve()
    ):
        raise SystemExit("output plan must not overwrite the embedding command file")

    embedding_command = (
        load_embedding_command(args.embedding_command_json)
        if args.embedding_command_json is not None
        else None
    )
    metadata = probe_video(args.video)
    frame_indices = validate_frame_indices(args.video, metadata)
    candidates = detect_candidate_cuts(
        args.video,
        threshold=args.threshold,
        frame_indices=frame_indices,
    )
    cuts, rejected = enforce_minimum_scene_length(
        candidates,
        frame_count=metadata.frame_count,
        minimum_frames=args.min_scene_length,
    )

    with tempfile.TemporaryDirectory(prefix="scene-boundaries-") as temporary:
        scores = score_boundaries(
            cuts,
            video=args.video,
            metadata=metadata,
            work_directory=Path(temporary),
            embedding_command=embedding_command,
            embedding_timeout=args.embedding_timeout,
        )
    plan = build_plan(
        path=args.video,
        metadata=metadata,
        cuts=cuts,
        rejected_cuts=rejected,
        threshold=args.threshold,
        minimum_scene_length=args.min_scene_length,
        context_frames=args.context_frames,
        continuity_scores=scores,
    )
    if embedding_command is not None:
        plan["continuity_scorer"] = EMBEDDING_SCORER_LABEL
    else:
        plan["continuity_scorer"] = FALLBACK_SCORER_LABEL
    _write_json(args.output, plan)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SceneBoundaryError as error:
        print(f"scene-boundaries: {error}", file=sys.stderr)
        raise SystemExit(1) from error
