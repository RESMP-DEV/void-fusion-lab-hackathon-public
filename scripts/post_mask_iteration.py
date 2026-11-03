#!/usr/bin/env python3
"""Run a bounded, deterministic post-mask replacement improvement loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from samurai.insertion import (
    InsertionError,
    IterationScore,
    aggregate_score,
    contiguous_poses,
    optimize_transform,
    score_alpha,
    spatial_objective,
    transform_alpha,
    transform_anchor_poses,
)
from samurai.media import (
    MediaError,
    assert_video_alignment,
    decode_gray_frames,
    iter_gray_frames,
    probe_video,
    write_gray_video,
)
from samurai.quadmask import build_insertion_sequence

DEFAULT_SMOOTHING_PASSES = (0, 2, 4, 6)
PoseRowLike = dict[str, Any]
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OPTIMIZATION_FRAMES = (0, 26, 48, 80)
DEFAULT_TRANSLATIONS = tuple(
    (float(dx), float(dy)) for dx in range(-48, 49, 6) for dy in range(-48, 49, 6)
)
DEFAULT_SCALES = (0.94, 0.97, 1.0, 1.04, 1.07, 1.10)
DERIVATIVE_FRAME_COUNT = 85
DERIVATIVE_WIDTH = 672
DERIVATIVE_HEIGHT = 384


class IterationError(RuntimeError):
    """The post-mask iteration could not satisfy its input contract."""


def parse_int_list(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected nonnegative comma-separated integers")
    return result


def transform_boundary_hit(transform: Any) -> dict[str, bool]:
    """Report whether each selected transform coordinate reaches a grid edge."""
    dx_values = [item[0] for item in DEFAULT_TRANSLATIONS]
    dy_values = [item[1] for item in DEFAULT_TRANSLATIONS]
    dimensions = {
        "translation_x": transform.dx in (min(dx_values), max(dx_values)),
        "translation_y": transform.dy in (min(dy_values), max(dy_values)),
        "scale": transform.scale in (min(DEFAULT_SCALES), max(DEFAULT_SCALES)),
    }
    return {**dimensions, "any": any(dimensions.values())}


def validate_background_prompt(prompt: str | None, *, dry_run: bool) -> None:
    """Reject empty and whitespace-only prompts before any media side effects."""
    if dry_run:
        return
    if prompt is None or not prompt.strip():
        raise IterationError(
            "--background-prompt must contain non-whitespace text unless --dry-run is set"
        )


def validate_clean_plate_pane(source_record: dict[str, Any] | None, source: Path) -> None:
    """Require an explicit, digest-pinned declaration for the right source pane."""
    pane = source_record.get("right_pane") if isinstance(source_record, dict) else None
    if not isinstance(pane, dict) or pane.get("role") != "void_pass1_clean_plate":
        raise IterationError(
            "warp_before_after.right_pane.role must declare void_pass1_clean_plate"
        )
    digest = pane.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or not all(character in "0123456789abcdef" for character in digest)
    ):
        raise IterationError("warp_before_after.right_pane.sha256 must be a lowercase SHA-256")
    if sha256(source) != digest:
        raise IterationError("warp_before_after.right_pane.sha256 does not match the source video")


def resolve_staged_frame_count(requested: int | None, *, source_count: int) -> int:
    """Resolve an explicit VOID Pass 2 staging count to the supported 4n+1 cadence."""
    count = source_count if requested is None else requested
    if count < 1 or count > source_count:
        raise IterationError(f"staged Pass 2 frame count must be in 1..{source_count}, got {count}")
    if count % 4 != 1:
        raise IterationError(
            f"staged Pass 2 frame count must be 4n+1 for the pinned temporal lane, got {count}"
        )
    return count


def run_tool(command: list[str], *, context: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise IterationError(f"{context} exit {completed.returncode}{suffix}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IterationError(f"invalid JSON document {path}: {error}") from error


def package_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise IterationError(f"{label} path must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise IterationError(f"{label} path must be relative to the evidence root: {value}")
    return path


def render_identity_matches(
    render_manifest: Any,
    *,
    blender: Path,
    blender_sha256: str,
    pose_path: Path,
    script_sha256: str,
    frame_count: int,
    samples: int,
) -> bool:
    required = {
        "blender_executable",
        "blender_executable_sha256",
        "engine",
        "frames",
        "height",
        "pose_sha256",
        "samples",
        "script_sha256",
        "width",
    }
    if not isinstance(render_manifest, dict) or not required.issubset(render_manifest):
        return False
    expected = {
        "blender_executable": str(blender.resolve()),
        "blender_executable_sha256": blender_sha256,
        "engine": "CYCLES",
        "frames": frame_count,
        "height": OUTPUT_HEIGHT,
        "pose_sha256": sha256(pose_path),
        "samples": samples,
        "script_sha256": script_sha256,
        "width": OUTPUT_WIDTH,
    }
    return all(render_manifest.get(key) == value for key, value in expected.items())


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def load_alpha(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("RGBA", "LA", "PA"):
            raise IterationError(f"expected an alpha-channel render: {path} (mode {image.mode})")
        alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    if not np.any(alpha < 255):
        raise IterationError(f"render has no transparent pixels: {path}")
    return alpha.copy()


def extract_source_side(
    source: Path,
    output: Path,
    *,
    frame_count: int,
    x_offset: int,
    label: str,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    source_info = probe_video(source)
    if source_info.width != OUTPUT_WIDTH * 2 or source_info.height != OUTPUT_HEIGHT:
        raise IterationError(
            f"source geometry must be {OUTPUT_WIDTH * 2}x{OUTPUT_HEIGHT}, "
            f"got {source_info.width}x{source_info.height}"
        )
    run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter:v",
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:{x_offset}:0",
            "-frames:v",
            str(frame_count),
            str(output / "frame-%06d.png"),
            "-y",
        ],
        context=f"{label} decode",
    )
    frames = sorted(output.glob("frame-*.png"))
    if len(frames) != frame_count:
        raise IterationError(f"decoded {len(frames)} {label} frames, expected {frame_count}")
    return frames


def resolve_target_masks(
    masks: list[np.ndarray],
    *,
    declared_polarity: str,
) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
    """Resolve either input polarity to white target masks and record the decision."""
    if not masks:
        raise IterationError("cannot resolve polarity for an empty mask sequence")
    black_pixels = sum(int(np.count_nonzero(mask < 128)) for mask in masks)
    white_pixels = sum(int(np.count_nonzero(mask >= 128)) for mask in masks)
    if black_pixels == 0 or white_pixels == 0:
        raise IterationError("mask sequence must contain both foreground and background pixels")
    if declared_polarity == "auto":
        raise IterationError(
            "mask polarity cannot be inferred safely; declare roi_black or foreground_white"
        )
    polarity = declared_polarity
    if polarity == "roi_black":
        targets = [np.where(mask < 128, 255, 0).astype(np.uint8) for mask in masks]
    else:
        targets = [np.where(mask >= 128, 255, 0).astype(np.uint8) for mask in masks]
    target_pixels = sum(int(np.count_nonzero(mask == 255)) for mask in targets)
    return targets, {
        "requested": declared_polarity,
        "resolved": polarity,
        "black_pixels": black_pixels,
        "white_pixels": white_pixels,
        "target_pixels": target_pixels,
    }


def verify_package(evidence_root: Path, manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise IterationError("package manifest must be a JSON object")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise IterationError("package manifest has no files record")
    for rel, record in files.items():
        relative = package_relative_path(rel, "package file")
        path = evidence_root / relative
        if not path.is_file():
            raise IterationError(f"package file is absent: {relative}")
        if not isinstance(record, dict) or "bytes" not in record or "sha256" not in record:
            raise IterationError(f"package file record lacks bytes/sha256: {relative}")
        size = record["bytes"]
        digest = record["sha256"]
        if isinstance(size, bool) or not isinstance(size, int):
            raise IterationError(f"package file byte count must be an integer: {relative}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not digest.islower()
            or not all(character in "0123456789abcdef" for character in digest)
        ):
            raise IterationError(f"package file digest must be a lowercase SHA-256: {relative}")
        if path.stat().st_size != size:
            raise IterationError(f"package file size mismatch: {relative}")
        if sha256(path) != digest:
            raise IterationError(f"package file digest mismatch: {relative}")


def pose_payload(rows: list[PoseRowLike], *, frame_count: int) -> dict[str, object]:
    return {
        "width": OUTPUT_WIDTH,
        "height": OUTPUT_HEIGHT,
        "frame_count": frame_count,
        "frames": rows,
    }


def render_candidate(
    blender: Path,
    blender_script: Path,
    poses: Path,
    output: Path,
    *,
    samples: int,
) -> None:
    if output.exists():
        raise IterationError(f"candidate render output already exists: {output}")
    output.mkdir(parents=True)
    try:
        run_tool(
            [
                str(blender),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(blender_script),
                "--",
                "--poses",
                str(poses),
                "--out",
                str(output),
                "--samples",
                str(samples),
                "--renderer",
                str(blender),
            ],
            context="Blender candidate render",
        )
    except BaseException:
        shutil.rmtree(output)
        raise


def load_candidate_alphas(directory: Path, *, frame_count: int) -> list[np.ndarray]:
    files = sorted(directory.glob("alpha-*.png"))
    if len(files) != frame_count:
        raise IterationError(f"candidate rendered {len(files)} frames, expected {frame_count}")
    return [load_alpha(path) for path in files]


def composite_frames(
    clean_plate_frames: list[Path],
    alpha_directory: Path,
    output: Path,
    *,
    frame_count: int,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("frame-*.png"):
        stale.unlink()
    alpha_files = sorted(alpha_directory.glob("alpha-*.png"))
    if len(alpha_files) != frame_count:
        raise IterationError(f"cannot composite {len(alpha_files)} alpha frames")
    composite_files: list[Path] = []
    composite_inputs = zip(clean_plate_frames, alpha_files, strict=True)
    for index, (source_path, alpha_path) in enumerate(composite_inputs):
        source = load_rgba(source_path)[:, :, :3].astype(np.float64)
        rendered = load_rgba(alpha_path)
        alpha = rendered[:, :, 3].astype(np.float64) / 255.0
        composite = source * (1.0 - alpha[..., None]) + rendered[:, :, :3] * alpha[..., None]
        output_path = output / f"frame-{index:06d}.png"
        Image.fromarray(np.clip(np.rint(composite), 0, 255).astype(np.uint8), mode="RGB").save(
            output_path
        )
        composite_files.append(output_path)
    return composite_files


def insertion_masks(
    alphas: list[np.ndarray],
    *,
    cleanup_radius: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create ROI-black inserted-object and cleanup-ring masks from render alpha."""
    if cleanup_radius < 1:
        raise IterationError("--cleanup-radius must be a positive integer")
    inserted: list[np.ndarray] = []
    cleanup: list[np.ndarray] = []
    for alpha in alphas:
        object_white = (alpha > 0).astype(np.uint8) * 255
        kernel_size = 2 * cleanup_radius + 1
        dilated_white = np.asarray(
            Image.fromarray(object_white).filter(ImageFilter.MaxFilter(kernel_size)),
            dtype=np.uint8,
        )
        ring_white = np.where((dilated_white == 255) & (object_white == 0), 255, 0).astype(np.uint8)
        inserted.append(np.where(object_white == 255, 0, 255).astype(np.uint8))
        cleanup.append(np.where(ring_white == 255, 0, 255).astype(np.uint8))
    return inserted, cleanup


def write_composite_video(frames: list[Path], output: Path, *, fps: float) -> None:
    first = frames[0]
    directory_frames = sorted(first.parent.glob("frame-*.png"))
    if directory_frames != frames:
        raise IterationError("composite directory does not match the supplied frame list")
    run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(first.parent / "frame-%06d.png"),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
            "-y",
        ],
        context="composite video encode",
    )


def write_staged_color_video(source: Path, output: Path, *, frame_count: int) -> None:
    """Write an exact stream-copy prefix for a staged VOID Pass 2 input."""
    run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "copy",
            "-an",
            str(output),
            "-y",
        ],
        context="staged Pass 2 input trim",
    )
    if probe_video(output).frames != frame_count:
        raise IterationError(f"staged Pass 2 input does not contain {frame_count} frames")


def ffmpeg_framerate(fps: float) -> str:
    """Return the canonical FFmpeg rate used by reproducible derivatives."""

    fraction = Fraction(fps).limit_denominator(1001)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def validate_derivative_contract(
    *, frame_count: int, width: int, height: int, source_count: int
) -> None:
    if frame_count < 1 or frame_count > source_count:
        raise IterationError(
            f"derivative frame count must be in 1..{source_count}, got {frame_count}"
        )
    if frame_count % 4 != 1:
        raise IterationError(
            f"derivative frame count must be 4n+1 for the pinned temporal lane, got {frame_count}"
        )
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise IterationError(
            f"derivative geometry must have even dimensions of at least 2x2, got {width}x{height}"
        )


def write_color_derivative(
    source: Path,
    output: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    """Write the deterministic color derivative consumed by the VOID execution lane."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setpts=N/FRAME_RATE/TB",
        "-r",
        ffmpeg_framerate(fps),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-bf",
        "0",
        "-crf",
        "12",
        "-preset",
        "veryslow",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-an",
        str(output),
        "-y",
    ]
    run_tool(command, context="execution color derivative")
    info = probe_video(output)
    if (info.width, info.height, info.frames) != (width, height, frame_count):
        raise IterationError(
            f"execution color derivative is {info.width}x{info.height}x{info.frames}, "
            f"expected {width}x{height}x{frame_count}"
        )
    return command


def verify_binary_mask_video(
    source: Path,
    *,
    expected_frames: int,
    label: str,
) -> None:
    """Decode a mask video and require every pixel to remain exactly 0 or 255."""

    with TemporaryDirectory(prefix=".post-mask-binary-", dir=source.parent) as temporary:
        decoded = 0
        for frame in iter_gray_frames(source, Path(temporary)):
            if not np.isin(frame, (0, 255)).all():
                raise IterationError(f"{label} contains non-binary pixels")
            decoded += 1
        if decoded != expected_frames:
            raise IterationError(f"{label} decoded {decoded} frames, expected {expected_frames}")


def write_binary_mask_derivative(
    source: Path,
    output: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    label: str,
) -> list[str]:
    """Write a lossless derivative while preserving only mask values 0 and 255."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-vf",
        f"scale={width}:{height}:flags=neighbor,format=gray,setpts=N/FRAME_RATE/TB",
        "-r",
        ffmpeg_framerate(fps),
        "-fps_mode",
        "cfr",
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
        "-threads",
        "1",
        "-an",
        str(output),
        "-y",
    ]
    run_tool(command, context=f"{label} execution mask derivative")
    info = probe_video(output)
    if (info.width, info.height, info.frames) != (width, height, frame_count):
        raise IterationError(
            f"{label} execution mask derivative is {info.width}x{info.height}x{info.frames}, "
            f"expected {width}x{height}x{frame_count}"
        )
    verify_binary_mask_video(
        output,
        expected_frames=frame_count,
        label=f"{label} execution mask derivative",
    )
    return command


def output_record(path: Path) -> dict[str, int | str]:
    """Record an output's exact size and digest beside its runtime path."""

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_execution_derivative(
    source: Path,
    inserted_mask: Path,
    cleanup_mask: Path,
    output_dir: Path,
    *,
    background_prompt: str,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    source_count: int,
) -> dict[str, Any]:
    """Create the deterministic 85-frame 672x384 VOID execution sequence."""

    validate_derivative_contract(
        frame_count=frame_count,
        width=width,
        height=height,
        source_count=source_count,
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    input_video = output_dir / "input_video.mp4"
    inserted_derivative = output_dir / "inserted-mask.mp4"
    cleanup_derivative = output_dir / "cleanup-mask.mp4"
    commands = {
        "color": write_color_derivative(
            source,
            input_video,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
        ),
        "inserted_mask": write_binary_mask_derivative(
            inserted_mask,
            inserted_derivative,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            label="inserted",
        ),
        "cleanup_mask": write_binary_mask_derivative(
            cleanup_mask,
            cleanup_derivative,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            label="cleanup",
        ),
    }
    sequence = build_insertion_sequence(
        input_video=input_video,
        inserted_mask=inserted_derivative,
        cleanup_mask=cleanup_derivative,
        background_prompt=background_prompt,
        sequence_dir=output_dir / "void-pass2-sequence",
    )
    staged_input = sequence.sequence_dir / "input_video.mp4"
    quadmask = sequence.sequence_dir / "quadmask_0.mp4"
    assert_video_alignment(
        {
            "input_video": staged_input,
            "inserted_mask": inserted_derivative,
            "cleanup_mask": cleanup_derivative,
            "quadmask": quadmask,
        },
        width=width,
        height=height,
        frames=frame_count,
    )
    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "mask_scaling": "ffmpeg neighbor scaling plus lossless gray decode validation",
        "commands": commands,
        "outputs": {
            "input_video": input_video,
            "inserted_mask": inserted_derivative,
            "cleanup_mask": cleanup_derivative,
            "quadmask": quadmask,
            "prompt": sequence.sequence_dir / "prompt.json",
        },
        "sequence_dir": str(sequence.sequence_dir),
    }


def export_candidate_chain(
    *,
    candidate: IterationScore,
    alpha_directory: Path,
    alphas: list[np.ndarray],
    clean_plate_frames: list[Path],
    masks: list[np.ndarray],
    output_root: Path,
    background_prompt: str,
    cleanup_radius: int,
    staged_frame_count: int,
    source_fps: float,
    source_frame_count: int,
    derivative_frame_count: int = DERIVATIVE_FRAME_COUNT,
    derivative_width: int = DERIVATIVE_WIDTH,
    derivative_height: int = DERIVATIVE_HEIGHT,
) -> dict[str, Any]:
    """Export one fitted candidate's complete staged insertion and execution chain."""

    source_height, source_width = alphas[0].shape
    chain_dir = output_root / "candidate-chains" / candidate.name
    if chain_dir.exists():
        shutil.rmtree(chain_dir)
    chain_dir.mkdir(parents=True)
    composite_dir = chain_dir / "composite-frames"
    composite_files = composite_frames(
        clean_plate_frames,
        alpha_directory,
        composite_dir,
        frame_count=len(alphas),
    )
    video = chain_dir / "diegetic-insertion.mp4"
    write_composite_video(composite_files, video, fps=source_fps)
    inserted_masks, cleanup_masks = insertion_masks(alphas, cleanup_radius=cleanup_radius)
    inserted_mask_video = chain_dir / "inserted-mask.mp4"
    cleanup_mask_video = chain_dir / "cleanup-mask.mp4"
    write_gray_video(inserted_masks, inserted_mask_video, fps=source_fps)
    write_gray_video(cleanup_masks, cleanup_mask_video, fps=source_fps)
    verify_binary_mask_video(
        inserted_mask_video,
        expected_frames=source_frame_count,
        label=f"{candidate.name} inserted mask",
    )
    verify_binary_mask_video(
        cleanup_mask_video,
        expected_frames=source_frame_count,
        label=f"{candidate.name} cleanup mask",
    )

    staged_input = chain_dir / "void-pass2-input.mp4"
    staged_inserted = chain_dir / "staged-inserted-mask.mp4"
    staged_cleanup = chain_dir / "staged-cleanup-mask.mp4"
    write_staged_color_video(video, staged_input, frame_count=staged_frame_count)
    write_gray_video(inserted_masks[:staged_frame_count], staged_inserted, fps=source_fps)
    write_gray_video(cleanup_masks[:staged_frame_count], staged_cleanup, fps=source_fps)
    sequence = build_insertion_sequence(
        input_video=staged_input,
        inserted_mask=staged_inserted,
        cleanup_mask=staged_cleanup,
        background_prompt=background_prompt,
        sequence_dir=chain_dir / "void-pass2-sequence",
    )
    assert_video_alignment(
        {
            "input_video": sequence.sequence_dir / "input_video.mp4",
            "inserted_mask": staged_inserted,
            "cleanup_mask": staged_cleanup,
            "quadmask": sequence.sequence_dir / "quadmask_0.mp4",
        },
        width=source_width,
        height=source_height,
        frames=staged_frame_count,
    )
    derivative = write_execution_derivative(
        video,
        inserted_mask_video,
        cleanup_mask_video,
        chain_dir / "execution-derivative",
        background_prompt=background_prompt,
        frame_count=derivative_frame_count,
        width=derivative_width,
        height=derivative_height,
        fps=source_fps,
        source_count=source_frame_count,
    )
    metadata = {
        "schema": "void-fusion-lab.post-mask-candidate-chain.v1",
        "candidate": asdict(candidate),
        "source": {
            "frames": source_frame_count,
            "width": source_width,
            "height": source_height,
            "fps": source_fps,
        },
        "staged_pass2": {
            "frames": staged_frame_count,
            "sequence_dir": str(sequence.sequence_dir),
        },
        "execution_derivative": {
            **derivative,
            "outputs": {name: output_record(path) for name, path in derivative["outputs"].items()},
        },
        "outputs": {
            "diegetic_insertion": output_record(video),
            "inserted_mask": output_record(inserted_mask_video),
            "cleanup_mask": output_record(cleanup_mask_video),
            "void_pass2_input": output_record(staged_input),
            "staged_inserted_mask": output_record(staged_inserted),
            "staged_cleanup_mask": output_record(staged_cleanup),
            "void_pass2_quadmask": output_record(sequence.sequence_dir / "quadmask_0.mp4"),
        },
        "non_claims": [
            "This export is deterministic media staging and does not run VOID Pass 2 or a model.",
            "Neighbor scaling preserves binary ROI masks but does not establish mask truth.",
        ],
    }
    metadata_path = chain_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "name": candidate.name,
        "chain_dir": chain_dir,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def write_contact_sheet(
    source_frames: list[Path],
    masks: list[np.ndarray],
    alpha_directory: Path,
    composite_directory: Path,
    output: Path,
    *,
    selected_frames: list[int],
) -> None:
    panels: list[np.ndarray] = []
    for frame in selected_frames:
        source = np.asarray(Image.open(source_frames[frame]).convert("RGB").resize((320, 180)))
        mask_resized = np.asarray(
            Image.fromarray(masks[frame]).resize((320, 180), Image.Resampling.NEAREST)
        )
        mask_rgb = np.repeat((mask_resized == 255)[..., None], 3, axis=2)
        mask_overlay = source.copy()
        mask_overlay[mask_rgb[..., 0]] = np.round(
            source[mask_rgb[..., 0]] * 0.55 + np.array((255, 0, 255)) * 0.45
        )
        mask_panel = np.asarray(Image.fromarray(mask_overlay.astype(np.uint8)).resize((320, 180)))
        alpha_path = alpha_directory / f"alpha-{frame:06d}.png"
        alpha = load_alpha(alpha_path)
        alpha_panel = np.repeat(alpha[..., None], 3, axis=2)
        alpha_panel = np.asarray(Image.fromarray(alpha_panel).resize((320, 180)))
        composite = np.asarray(
            Image.open(composite_directory / f"frame-{frame:06d}.png")
            .convert("RGB")
            .resize((320, 180))
        )
        row = np.concatenate((source, mask_panel, alpha_panel, composite), axis=1)
        label = np.zeros((30, row.shape[1], 3), dtype=np.uint8)
        label_image = Image.fromarray(label)
        ImageDraw.Draw(label_image).text((12, 8), f"frame {frame}", fill=(255, 255, 255))
        panels.append(np.concatenate((np.asarray(label_image), row), axis=0))
    sheet = np.concatenate(panels, axis=0)
    Image.fromarray(sheet).save(output, quality=92)


def frame_metrics(
    alphas: list[np.ndarray], masks: list[np.ndarray]
) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for index, (alpha, mask) in enumerate(zip(alphas, masks, strict=True)):
        score = score_alpha(alpha, mask)
        result.append(
            {"frame": index, **asdict(score), "spatial_objective": spatial_objective(score)}
        )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--evidence-root", type=Path, required=True)
    result.add_argument(
        "--package",
        type=Path,
        default=Path("build/post-mask-replacement"),
    )
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--blender", type=Path)
    result.add_argument(
        "--blender-script",
        type=Path,
        default=Path(__file__).parent / "post_mask/blender_proxy.py",
    )
    result.add_argument("--samples", type=int, default=24)
    result.add_argument(
        "--mask-polarity",
        choices=("roi_black", "foreground_white"),
        required=True,
        help="input SAM-mask polarity; output masks are always canonical ROI-black",
    )
    result.add_argument(
        "--cleanup-radius",
        type=int,
        default=7,
        help="dilation radius in pixels; 7 produces a 7-pixel cleanup ring",
    )
    result.add_argument(
        "--background-prompt",
        help="VOID background prompt for the staged Pass 2 sequence",
    )
    result.add_argument(
        "--pass2-frame-count",
        type=int,
        help="explicit 4n+1 frame count for the staged VOID Pass 2 inputs",
    )
    result.add_argument(
        "--smoothing-passes", type=parse_int_list, default=list(DEFAULT_SMOOTHING_PASSES)
    )
    result.add_argument(
        "--reuse-renders",
        action="store_true",
        help="reuse complete candidate alpha directories instead of rerendering them",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="write the search plan only; execution-only Pass 2 staging checks are deferred",
    )
    return result


def report_non_claims(transform_hits_boundary: bool) -> list[str]:
    """Return non-claims that match this run rather than a prior campaign."""
    claims = [
        "The ensemble mask is loose; spatial metrics do not establish silhouette ground truth.",
        "The proxy is model-free pneumatic geometry, not learned image-to-3D reconstruction.",
        "The paired right pane was declared as a retained VOID Pass 1 clean plate and its "
        "whole-source digest was validated before compositing.",
        "The staged Pass 2 inputs use an explicit 4n+1 prefix; this is not evidence of a "
        "new VOID inference run.",
    ]
    if transform_hits_boundary:
        claims.append(
            "Vertical translation and scale lie on the declared transform-search boundary; "
            "the fit is not extrapolated beyond that grid."
        )
    else:
        claims.append(
            "The selected transform remains inside the declared search grid; no transform "
            "extrapolation is claimed."
        )
    claims.extend(
        [
            "The insertion-aware sequence is staged, but no VOID Pass 2, SAM call, "
            "Wan/H3 pass, paid API, or model inference ran.",
            "Bounded temporal smoothing preserves anchor landmarks but does not prove pose truth.",
        ]
    )
    return claims


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.samples <= 0:
        raise IterationError("--samples must be positive")
    if args.cleanup_radius <= 0:
        raise IterationError("--cleanup-radius must be a positive integer")
    validate_background_prompt(args.background_prompt, dry_run=args.dry_run)
    if not args.dry_run and args.blender is None:
        raise IterationError("--blender is required unless --dry-run is set")
    blender_path = args.blender.resolve() if args.blender is not None else None
    blender_script = args.blender_script.resolve()
    blender_sha = sha256(blender_path) if blender_path is not None else ""
    renderer_script_sha = sha256(blender_script)

    args.out.mkdir(parents=True, exist_ok=True)
    package = args.evidence_root / args.package
    manifest_path = package / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise IterationError("package manifest must be a JSON object")
    verify_package(args.evidence_root, manifest)
    selected_frames_raw = manifest.get("selected_frames")
    if not isinstance(selected_frames_raw, list):
        raise IterationError("package manifest lacks selected_frames list")
    if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in selected_frames_raw):
        raise IterationError("package selected_frames must contain integers")
    selected_frames = selected_frames_raw
    anchor_payload = read_json(package / "records/pose-series-selected.json")
    if not isinstance(anchor_payload, dict):
        raise IterationError("pose-series-selected.json must be a JSON object")
    anchor_rows_raw = anchor_payload.get("frames")
    if not isinstance(anchor_rows_raw, list):
        raise IterationError("pose-series-selected.json lacks frames list")
    anchor_rows = anchor_rows_raw
    try:
        anchor_frame_values = [int(row["frame"]) for row in anchor_rows]
    except (KeyError, TypeError, ValueError) as error:
        raise IterationError("pose-series frame records must carry integer frames") from error
    if anchor_frame_values != selected_frames:
        raise IterationError("pose-series frames disagree with manifest selected_frames")

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise IterationError("package manifest lacks sources object")
    source_record = sources.get("warp_before_after")
    mask_record = sources.get("ensemble_mask")
    if (
        not isinstance(source_record, dict)
        or not isinstance(mask_record, dict)
        or "path" not in source_record
        or "path" not in mask_record
    ):
        raise IterationError("package manifest lacks source video records")
    source_rel = package_relative_path(source_record["path"], "source video")
    mask_rel = package_relative_path(mask_record["path"], "ensemble mask")
    source = args.evidence_root / source_rel
    mask_video = args.evidence_root / mask_rel
    validate_clean_plate_pane(source_record, source)
    source_info = probe_video(source)
    mask_info = probe_video(mask_video)
    if source_info.frames != mask_info.frames:
        raise IterationError("source and mask frame counts differ")
    if mask_info.width != OUTPUT_WIDTH or mask_info.height != OUTPUT_HEIGHT:
        raise IterationError(
            f"mask geometry must be {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}, "
            f"got {mask_info.width}x{mask_info.height}"
        )
    frame_count = source_info.frames
    if len(selected_frames) != len(set(selected_frames)) or any(
        frame < 0 or frame >= frame_count for frame in selected_frames
    ):
        raise IterationError(
            f"selected_frames must be unique and inside 0..{frame_count - 1}: {selected_frames}"
        )

    source_frames = extract_source_side(
        source,
        args.out / "source-original",
        frame_count=frame_count,
        x_offset=0,
        label="source-original",
    )
    clean_plate_frames = extract_source_side(
        source,
        args.out / "void-pass1-clean-plate",
        frame_count=frame_count,
        x_offset=OUTPUT_WIDTH,
        label="void-pass1-clean-plate",
    )
    raw_masks = decode_gray_frames(
        mask_video, args.out / "input-masks", expected_frames=frame_count
    )
    masks, mask_polarity = resolve_target_masks(
        raw_masks,
        declared_polarity=args.mask_polarity,
    )
    anchor_alphas = [
        load_alpha(package / f"alpha/render-{frame:03d}.png") for frame in selected_frames
    ]
    anchor_masks = [masks[frame] for frame in selected_frames]
    for alpha, mask in zip(anchor_alphas, anchor_masks, strict=True):
        if alpha.shape != mask.shape:
            raise IterationError(f"anchor geometry mismatch: {alpha.shape} != {mask.shape}")

    missing_optimization = [frame for frame in OPTIMIZATION_FRAMES if frame not in selected_frames]
    if missing_optimization:
        raise IterationError(
            f"optimization frames absent from selected_frames: {missing_optimization}"
        )
    optimization_indices = [selected_frames.index(frame) for frame in OPTIMIZATION_FRAMES]
    transform, optimization_scores = optimize_transform(
        [anchor_alphas[index] for index in optimization_indices],
        [anchor_masks[index] for index in optimization_indices],
        translations=DEFAULT_TRANSLATIONS,
        scales=DEFAULT_SCALES,
    )
    transformed_anchor_alphas = [transform_alpha(alpha, transform) for alpha in anchor_alphas]
    transformed_pairs = zip(transformed_anchor_alphas, anchor_masks, strict=True)
    transformed_anchor_scores = [score_alpha(alpha, mask) for alpha, mask in transformed_pairs]
    transformed_anchor_rows = transform_anchor_poses(anchor_rows, transform)

    plan = {
        "schema": "void-fusion-lab.post-mask-iteration.plan.v1",
        "transform_search": {
            "translations": [list(item) for item in DEFAULT_TRANSLATIONS],
            "scales": list(DEFAULT_SCALES),
            "best": asdict(transform),
            "optimization_frames": list(OPTIMIZATION_FRAMES),
            "boundary_hit": transform_boundary_hit(transform),
        },
        "smoothing_passes": args.smoothing_passes,
        "mask_polarity": mask_polarity,
        "cleanup_radius": args.cleanup_radius,
        "anchor_validation": [asdict(score) for score in transformed_anchor_scores],
    }
    (args.out / "iteration-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({"status": "planned", **plan}, indent=2))
        return 0

    staged_frame_count = resolve_staged_frame_count(
        args.pass2_frame_count, source_count=frame_count
    )
    background_prompt = args.background_prompt
    if background_prompt is None:
        raise IterationError("--background-prompt is required unless --dry-run is set")
    candidates: list[IterationScore] = []
    candidate_directories: dict[str, Path] = {}
    candidate_frame_scores: dict[str, list[dict[str, float | int]]] = {}
    candidate_chains: dict[str, dict[str, Any]] = {}
    for passes in args.smoothing_passes:
        name = f"smooth-{passes:02d}"
        contiguous = contiguous_poses(
            transformed_anchor_rows,
            frame_count=frame_count,
            smoothing_passes=passes,
        )
        pose_path = args.out / f"{name}-poses.json"
        pose_path.write_text(
            json.dumps(pose_payload(contiguous, frame_count=frame_count), indent=2) + "\n"
        )
        render_dir = args.out / f"candidate-{name}"
        alphas: list[np.ndarray] = []
        if args.reuse_renders and render_dir.is_dir():
            try:
                render_manifest = json.loads((render_dir / "render-manifest.json").read_text())
                if blender_path is None or not render_identity_matches(
                    render_manifest,
                    blender=blender_path,
                    blender_sha256=blender_sha,
                    pose_path=pose_path,
                    script_sha256=renderer_script_sha,
                    frame_count=frame_count,
                    samples=args.samples,
                ):
                    raise IterationError("candidate render identity mismatch")
                alphas = load_candidate_alphas(render_dir, frame_count=frame_count)
            except IterationError:
                alphas = []
                shutil.rmtree(render_dir)
            except (OSError, json.JSONDecodeError):
                alphas = []
                shutil.rmtree(render_dir)
        if not alphas:
            if blender_path is None:
                raise IterationError("--blender is required unless --dry-run is set")
            render_candidate(
                blender_path,
                blender_script,
                pose_path,
                render_dir,
                samples=args.samples,
            )
            alphas = load_candidate_alphas(render_dir, frame_count=frame_count)
        scores = [score_alpha(alpha, mask) for alpha, mask in zip(alphas, masks, strict=True)]
        candidate = aggregate_score(
            name=name,
            transform=transform,
            smoothing_passes=passes,
            frame_scores=scores,
            alphas=alphas,
            expected_anchor_rows=transformed_anchor_rows,
            final_pose_rows=contiguous,
        )
        candidates.append(candidate)
        candidate_directories[name] = render_dir
        candidate_frame_scores[name] = frame_metrics(alphas, masks)
        candidate_chains[name] = export_candidate_chain(
            candidate=candidate,
            alpha_directory=render_dir,
            alphas=alphas,
            clean_plate_frames=clean_plate_frames,
            masks=masks,
            output_root=args.out,
            background_prompt=background_prompt,
            cleanup_radius=args.cleanup_radius,
            staged_frame_count=staged_frame_count,
            source_fps=source_info.fps,
            source_frame_count=frame_count,
        )

    best = max(candidates, key=lambda item: item.objective)
    best_alpha_dir = candidate_directories[best.name]
    best_chain = candidate_chains[best.name]
    composite_dir = best_chain["chain_dir"] / "composite-frames"
    contact_sheet = args.out / "best-contact-sheet.jpg"
    write_contact_sheet(
        source_frames,
        masks,
        best_alpha_dir,
        composite_dir,
        contact_sheet,
        selected_frames=selected_frames,
    )

    try:
        repository = Path(__file__).resolve().parents[1]
        source_revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        source_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"], text=True
            ).strip()
        )
    except (subprocess.CalledProcessError, OSError) as error:
        raise IterationError("git metadata unavailable in the script repository") from error

    best_metadata = best_chain["metadata"]
    report = {
        "schema": "void-fusion-lab.post-mask-iteration.report.v3",
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "inputs": {
            "evidence_master_manifest": str(manifest_path),
            "source_frames": frame_count,
            "staged_pass2_frames": staged_frame_count,
            "source_fps": source_info.fps,
            "source_sha256": sha256(source),
            "mask_sha256": sha256(mask_video),
            "mask_polarity": mask_polarity,
            "execution_derivative": {
                "frame_count": DERIVATIVE_FRAME_COUNT,
                "width": DERIVATIVE_WIDTH,
                "height": DERIVATIVE_HEIGHT,
            },
        },
        "plan": plan,
        "candidates": [asdict(item) for item in candidates],
        "best": asdict(best),
        "best_frame_metrics": candidate_frame_scores[best.name],
        "candidate_chains": [
            {
                "name": chain["name"],
                "chain_dir": str(chain["chain_dir"]),
                "metadata": chain["metadata"],
            }
            for chain in candidate_chains.values()
        ],
        "outputs": {
            "best_candidate_chain": str(best_chain["chain_dir"]),
            "video": best_metadata["outputs"]["diegetic_insertion"],
            "original_source_directory": str(source_frames[0].parent),
            "void_pass1_clean_plate_directory": str(clean_plate_frames[0].parent),
            "inserted_mask": best_metadata["outputs"]["inserted_mask"],
            "cleanup_mask": best_metadata["outputs"]["cleanup_mask"],
            "void_pass2_sequence": best_metadata["staged_pass2"]["sequence_dir"],
            "void_pass2_sequence_frames": staged_frame_count,
            "void_pass2_input": best_metadata["outputs"]["void_pass2_input"],
            "staged_inserted_mask": best_metadata["outputs"]["staged_inserted_mask"],
            "staged_cleanup_mask": best_metadata["outputs"]["staged_cleanup_mask"],
            "contact_sheet": str(contact_sheet),
            "contact_sheet_sha256": sha256(contact_sheet),
            "alpha_directory": str(best_alpha_dir),
            "composite_directory": str(composite_dir),
        },
        "non_claims": report_non_claims(plan["transform_search"]["boundary_hit"]["any"]),
    }
    (args.out / "iteration-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "best": asdict(best),
                "report": str(args.out / "iteration-report.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        IterationError,
        InsertionError,
        MediaError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        print(f"post-mask iteration failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
