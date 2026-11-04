#!/usr/bin/env python3
"""Assemble a labeled multi-shot advertising reel and its comparison master.

The assembler consumes one manifest that names, for every shot of an
advertising spot, the five videos of the canonical chain: the original scene,
the mask overlay of the selected object, the VOID Pass 1 clean plate, the
fitted visible insertion over that plate, and the insertion-aware VOID
Pass 2 result.  It writes two artifacts: a sequential, labeled reel of the
ad-ready result, and a labeled comparison master whose rows place every stage
of every shot side by side, which is the no-Pass-2 versus Pass-2 comparison of
the whole spot.

It fails closed on a missing stage, on a geometry, frame count, frame rate, or
timestamp mismatch between the stages of a shot or between shots, and on an
insertion stage that is nearly identical to the Pass 1 clean plate, because a
no-op insertion cannot be presented as a diegetic replacement.  It measures and
reports; it makes no quality verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from samurai.media import MediaError, extract_color_frames, probe_frame_timestamps, probe_video

MANIFEST_SCHEMA = "void-fusion-lab.advertising-reel.manifest.v1"
REPORT_SCHEMA = "void-fusion-lab.advertising-reel.report.v1"

STAGES: tuple[str, ...] = (
    "original",
    "mask_overlay",
    "pass1_clean_plate",
    "inserted",
    "pass2",
)

STAGE_CAPTIONS: dict[str, str] = {
    "original": "ORIGINAL",
    "mask_overlay": "MASK OVERLAY (SELECTION)",
    "pass1_clean_plate": "VOID PASS 1 CLEAN PLATE",
    "inserted": "FITTED VISIBLE INSERTION (NO PASS 2)",
    "pass2": "VOID PASS 2 RESULT",
}

DEFAULT_CHANGED_THRESHOLD = 10.0
DEFAULT_MIN_INSERTION_CHANGED_RATIO = 0.001
FRAME_TEMPLATE = "frame-{index:06d}.png"
FRAME_PATTERN = "frame-%06d.png"

LABEL_BACKGROUND = (18, 18, 18)
LABEL_FOREGROUND = (245, 245, 245)

NON_CLAIMS: tuple[str, ...] = (
    "The reel and the comparison master are assembly artifacts. They prove stage presence, "
    "geometry, frame cadence, and that the insertion differs from the Pass 1 clean plate. "
    "They claim no artistic quality, physical plausibility, or diegetic correctness.",
    "Stage alignment is bound to decoded pixel dimensions, container frame counts, frame "
    "rates, and presentation timestamps. It does not prove that the tracked object occupies "
    "the same semantic region in every stage.",
    "The insertion check proves only that the inserted stage differs from the clean plate "
    "above a pixel-change threshold. It does not prove the inserted asset is a learned "
    "image-to-3D generation, that it is correctly fitted, or that it is temporally stable.",
    "No VOID Pass 2 quality metric is computed here. The Pass 2 comparator remains the "
    "authority for Pass 2 change, cleanup-ring, and temporal measurements.",
    "This assembler runs no model and no CUDA lane. It consumes already-produced stage "
    "videos and does not verify how they were produced.",
    "Labels are rendered with the Pillow default font, so label pixels are not guaranteed "
    "to be identical across Pillow versions.",
)


class AdvertisingReelError(RuntimeError):
    """A manifest, stage, or geometry validation failed."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fps_fraction(fps: float) -> Fraction:
    """Return the exact rational frame rate of a probed stream."""
    return Fraction(fps).limit_denominator(1001)


def fps_text(fps: Fraction) -> str:
    """Return the ffmpeg frame-rate text for an exact rational frame rate."""
    if fps.denominator == 1:
        return str(fps.numerator)
    return f"{fps.numerator}/{fps.denominator}"


@dataclass(frozen=True)
class ShotPlan:
    """One shot of the advertising spot and the five stage videos it names."""

    shot_id: str
    label: str
    stages: dict[str, Path]


@dataclass(frozen=True)
class StageProbe:
    """The probed geometry, frame count, frame rate, and timestamps of one stage."""

    path: Path
    width: int
    height: int
    frames: int
    fps: Fraction
    timestamps: tuple[Fraction, ...]


def plan_shots(manifest_path: Path, payload: Any) -> list[ShotPlan]:
    """Validate the manifest and return one validated shot plan per shot."""
    if not isinstance(payload, dict):
        raise AdvertisingReelError("manifest must be a JSON object")
    schema = payload.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise AdvertisingReelError(
            f"unsupported manifest schema {schema!r}: expected {MANIFEST_SCHEMA}"
        )
    shots = payload.get("shots")
    if not isinstance(shots, list) or not shots:
        raise AdvertisingReelError("manifest must declare a nonempty shots list")
    plans: list[ShotPlan] = []
    seen: set[str] = set()
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise AdvertisingReelError(f"shot {index} must be a JSON object")
        shot_id = shot.get("id")
        if not isinstance(shot_id, str) or not shot_id.strip():
            raise AdvertisingReelError(f"shot {index} must declare a nonempty id")
        if shot_id in seen:
            raise AdvertisingReelError(f"shot id {shot_id!r} is declared more than once")
        seen.add(shot_id)
        label = shot.get("label")
        if not isinstance(label, str) or not label.strip():
            raise AdvertisingReelError(f"shot {shot_id} must declare a nonempty label")
        stages = shot.get("stages")
        if not isinstance(stages, dict):
            raise AdvertisingReelError(f"shot {shot_id} must declare a stages object")
        unknown = sorted(set(stages) - set(STAGES))
        if unknown:
            raise AdvertisingReelError(f"shot {shot_id} declares unknown stages: {unknown}")
        resolved: dict[str, Path] = {}
        for stage in STAGES:
            value = stages.get(stage)
            if value is None:
                raise AdvertisingReelError(f"shot {shot_id} is missing stage {stage}")
            if not isinstance(value, str) or not value.strip():
                raise AdvertisingReelError(f"shot {shot_id} stage {stage} must name a video file")
            path = Path(value)
            if not path.is_absolute():
                path = manifest_path.parent / path
            if not path.is_file():
                raise AdvertisingReelError(f"shot {shot_id} stage {stage} does not exist: {path}")
            resolved[stage] = path
        plans.append(ShotPlan(shot_id=shot_id, label=label, stages=resolved))
    return plans


def probe_stage(path: Path) -> StageProbe:
    """Probe one stage video and bind its exact frame timeline."""
    try:
        info = probe_video(path)
        timestamps = tuple(probe_frame_timestamps(path))
    except (MediaError, OSError, subprocess.SubprocessError) as error:
        raise AdvertisingReelError(f"could not probe stage video {path}: {error}") from error
    if len(timestamps) != info.frames:
        raise AdvertisingReelError(
            f"{path} has {len(timestamps)} timestamped frames but reports {info.frames} frames"
        )
    return StageProbe(
        path=path,
        width=info.width,
        height=info.height,
        frames=info.frames,
        fps=fps_fraction(info.fps),
        timestamps=timestamps,
    )


def validate_shot_alignment(shot_id: str, probes: dict[str, StageProbe]) -> None:
    """Fail closed unless every stage of one shot shares geometry, cadence, and timing."""
    reference = probes[STAGES[0]]
    for stage in STAGES[1:]:
        probe = probes[stage]
        if (probe.width, probe.height) != (reference.width, reference.height):
            raise AdvertisingReelError(
                f"shot {shot_id} stage {stage} geometry mismatch: "
                f"{probe.width}x{probe.height}, expected {reference.width}x{reference.height}"
            )
        if probe.frames != reference.frames:
            raise AdvertisingReelError(
                f"shot {shot_id} stage {stage} frame count mismatch: "
                f"{probe.frames} frames, expected {reference.frames}"
            )
        if probe.fps != reference.fps:
            raise AdvertisingReelError(
                f"shot {shot_id} stage {stage} frame rate mismatch: "
                f"{fps_text(probe.fps)}, expected {fps_text(reference.fps)}"
            )
        if probe.timestamps != reference.timestamps:
            first = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(probe.timestamps, reference.timestamps, strict=False)
                )
                if actual != expected
            )
            raise AdvertisingReelError(
                f"shot {shot_id} stage {stage} timestamp mismatch at frame {first}: "
                f"{probe.timestamps[first]} != {reference.timestamps[first]}"
            )


def validate_spot_alignment(
    shots: list[ShotPlan], probes: dict[str, dict[str, StageProbe]]
) -> None:
    """Fail closed unless every shot of the spot shares one canvas and frame rate."""
    reference_shot = shots[0]
    reference = probes[reference_shot.shot_id][STAGES[0]]
    for shot in shots[1:]:
        probe = probes[shot.shot_id][STAGES[0]]
        if (probe.width, probe.height) != (reference.width, reference.height):
            raise AdvertisingReelError(
                f"shot {shot.shot_id} geometry mismatch against shot {reference_shot.shot_id}: "
                f"{probe.width}x{probe.height}, expected {reference.width}x{reference.height}"
            )
        if probe.fps != reference.fps:
            raise AdvertisingReelError(
                f"shot {shot.shot_id} frame rate mismatch against shot {reference_shot.shot_id}: "
                f"{fps_text(probe.fps)}, expected {fps_text(reference.fps)}"
            )


def extract_stage_frames(stage: str, path: Path, destination: Path) -> list[Path]:
    """Decode one stage video into its per-frame PNG files."""
    try:
        return extract_color_frames(path, destination)
    except (MediaError, OSError, subprocess.SubprocessError) as error:
        raise AdvertisingReelError(
            f"could not decode stage {stage} video {path}: {error}"
        ) from error


def measure_insertion(
    plate_frames: list[Path],
    inserted_frames: list[Path],
    *,
    changed_threshold: float,
) -> dict[str, Any]:
    """Measure how far the inserted stage departs from the Pass 1 clean plate."""
    if len(plate_frames) != len(inserted_frames):
        raise AdvertisingReelError(
            f"insertion stages decoded {len(plate_frames)} and {len(inserted_frames)} frames"
        )
    total_changed = 0
    total_pixels = 0
    absolute_sum = 0.0
    max_frame_changed_ratio = 0.0
    frames_with_change = 0
    exactly_identical = True
    for plate_path, inserted_path in zip(plate_frames, inserted_frames, strict=True):
        with Image.open(plate_path) as handle:
            plate = np.asarray(handle.convert("RGB"), dtype=np.int16)
        with Image.open(inserted_path) as handle:
            inserted = np.asarray(handle.convert("RGB"), dtype=np.int16)
        difference = np.abs(inserted - plate)
        changed = difference.mean(axis=2) > changed_threshold
        changed_pixels = int(changed.sum())
        pixels = int(changed.size)
        if pixels == 0:
            raise AdvertisingReelError(
                f"insertion stage frame {plate_path.name} decoded zero pixels"
            )
        total_changed += changed_pixels
        total_pixels += pixels
        absolute_sum += float(difference.mean())
        max_frame_changed_ratio = max(max_frame_changed_ratio, changed_pixels / pixels)
        if changed_pixels:
            frames_with_change += 1
        if not np.array_equal(plate, inserted):
            exactly_identical = False
    frames = len(plate_frames)
    return {
        "changed_threshold": changed_threshold,
        "frames": frames,
        "mean_absolute_difference": absolute_sum / frames if frames else 0.0,
        "mean_changed_ratio": total_changed / total_pixels if total_pixels else 0.0,
        "max_frame_changed_ratio": max_frame_changed_ratio,
        "frames_with_change": frames_with_change,
        "exactly_identical": exactly_identical,
    }


def validate_insertion(
    shot_id: str,
    metrics: dict[str, Any],
    *,
    min_changed_ratio: float,
) -> None:
    """Fail closed when the insertion stage is nearly identical to the clean plate."""
    if metrics["exactly_identical"]:
        raise AdvertisingReelError(
            f"shot {shot_id} insertion stage is nearly identical to the pass 1 clean plate: "
            "the inserted and clean-plate frames are exactly equal"
        )
    if metrics["max_frame_changed_ratio"] < min_changed_ratio:
        raise AdvertisingReelError(
            f"shot {shot_id} insertion stage is nearly identical to the pass 1 clean plate: "
            f"the largest changed-pixel ratio is {metrics['max_frame_changed_ratio']:.6f}, "
            f"below the required {min_changed_ratio:.6f}"
        )


def label_font(size: int) -> ImageFont.BaseImageFont:
    """Return the bundled scalable Pillow font, falling back to its bitmap default."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - only reachable on Pillow older than 10.1
        return ImageFont.load_default()


def band_height(height: int, *, divisor: int, minimum: int) -> int:
    """Return a label band height that never covers more than half of the frame."""
    return max(1, min(height // 2, max(minimum, height // divisor)))


def text_height(text: str, font: ImageFont.BaseImageFont) -> int:
    """Return the rendered height of one text line."""
    box = font.getbbox(text)
    if box is None:
        return 1
    return int(box[3] - box[1])


def draw_band(
    image: Image.Image,
    text: str,
    height: int,
    font: ImageFont.BaseImageFont,
) -> Image.Image:
    """Draw a filled caption band with left-aligned text across the bottom of one frame."""
    draw = ImageDraw.Draw(image)
    top = image.height - height
    draw.rectangle([(0, top), (image.width, image.height)], fill=LABEL_BACKGROUND)
    baseline = top + max(0, (height - text_height(text, font)) // 2)
    draw.text((6, baseline), text, font=font, fill=LABEL_FOREGROUND)
    return image


def text_baseline(top: int, height: int, font: ImageFont.BaseImageFont, text: str) -> int:
    """Return the y offset that centers one text line inside a band."""
    return top + max(0, (height - text_height(text, font)) // 2)


def pad_even(image: Image.Image) -> Image.Image:
    """Pad a frame by at most one column and row so that yuv420p can encode it."""
    extra_width = image.width % 2
    extra_height = image.height % 2
    if not extra_width and not extra_height:
        return image
    padded = Image.new(
        "RGB",
        (image.width + extra_width, image.height + extra_height),
        (0, 0, 0),
    )
    padded.paste(image, (0, 0))
    return padded


def build_reel_frames(
    shots: list[ShotPlan],
    frames_by_shot: dict[str, dict[str, list[Path]]],
    reel_stage: str,
    destination: Path,
) -> tuple[int, int, int]:
    """Write the sequential labeled reel frames and return their count, width, and height."""
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    width = 0
    height = 0
    for index, shot in enumerate(shots):
        frames = frames_by_shot[shot.shot_id][reel_stage]
        with Image.open(frames[0]) as handle:
            width, height = handle.size
        band = band_height(height, divisor=10, minimum=24)
        font = label_font(max(8, band - 8))
        text = f"SHOT {index + 1:02d} | {shot.label}"
        for source in frames:
            with Image.open(source) as handle:
                frame = handle.convert("RGB")
            frame = pad_even(frame)
            draw_band(frame, text, band, font)
            frame.save(destination / FRAME_TEMPLATE.format(index=written))
            written += 1
    return written, width + (width % 2), height + (height % 2)


def build_master_frames(
    shots: list[ShotPlan],
    frames_by_shot: dict[str, dict[str, list[Path]]],
    destination: Path,
) -> tuple[int, int, int]:
    """Write the labeled comparison master frames and return their count, width, and height."""
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    width = 0
    height = 0
    for index, shot in enumerate(shots):
        stage_frames = frames_by_shot[shot.shot_id]
        with Image.open(stage_frames[STAGES[0]][0]) as handle:
            cell_width, cell_height = handle.size
        shot_band = band_height(cell_height, divisor=12, minimum=20)
        caption = band_height(cell_height, divisor=16, minimum=14)
        row_height = shot_band + cell_height + caption
        master_width = cell_width * len(STAGES)
        width = master_width
        height = row_height
        shot_font = label_font(max(8, shot_band - 6))
        caption_font = label_font(max(8, caption - 4))
        shot_text = f"SHOT {index + 1:02d} | {shot.label}"
        for frame_index in range(len(stage_frames[STAGES[0]])):
            row = Image.new("RGB", (master_width, row_height), LABEL_BACKGROUND)
            draw = ImageDraw.Draw(row)
            draw.text(
                (6, text_baseline(0, shot_band, shot_font, shot_text)),
                shot_text,
                font=shot_font,
                fill=LABEL_FOREGROUND,
            )
            for column, stage in enumerate(STAGES):
                with Image.open(stage_frames[stage][frame_index]) as handle:
                    cell = handle.convert("RGB")
                left = column * cell_width
                row.paste(cell, (left, shot_band))
                caption_top = shot_band + cell_height
                caption_text = STAGE_CAPTIONS[stage]
                draw.rectangle(
                    [(left, caption_top), (left + cell_width, row_height)],
                    fill=LABEL_BACKGROUND,
                )
                draw.text(
                    (left + 6, text_baseline(caption_top, caption, caption_font, caption_text)),
                    STAGE_CAPTIONS[stage],
                    font=caption_font,
                    fill=LABEL_FOREGROUND,
                )
            row = pad_even(row)
            row.save(destination / FRAME_TEMPLATE.format(index=written))
            written += 1
    return written, width + (width % 2), height + (height % 2)


def encode_frames(frames: Path, output: Path, fps: Fraction, *, lossless: bool) -> None:
    """Encode one PNG frame sequence into an H.264 video."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        fps_text(fps),
        "-i",
        str(frames / FRAME_PATTERN),
        "-r",
        fps_text(fps),
        "-c:v",
        "libx264",
    ]
    if lossless:
        command += ["-qp", "0", "-bf", "0", "-pix_fmt", "yuv444p", "-preset", "veryslow"]
    else:
        command += ["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "medium"]
    command += ["-y", str(output)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise AdvertisingReelError(
            f"ffmpeg failed to encode {output}: {error.stderr.strip()}"
        ) from error


def scratch_directory(parent: Path) -> TemporaryDirectory[str]:
    """Create a scratch directory beside the output, falling back to the system temp."""
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(prefix=".advertising-reel-", dir=parent)
    except OSError:
        return TemporaryDirectory(prefix=".advertising-reel-")


def build_report(
    manifest_path: Path,
    shots: list[ShotPlan],
    probes: dict[str, dict[str, StageProbe]],
    insertions: dict[str, dict[str, Any]],
    reel: dict[str, Any],
    master: dict[str, Any],
    thresholds: dict[str, Any],
    reel_stage: str,
) -> dict[str, Any]:
    """Assemble the JSON report for one completed assembly."""
    return {
        "schema": REPORT_SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "stages": list(STAGES),
        "stage_captions": dict(STAGE_CAPTIONS),
        "reel_stage": reel_stage,
        "thresholds": thresholds,
        "shots": [
            {
                "id": shot.shot_id,
                "label": shot.label,
                "stages": {
                    stage: {
                        "path": str(shot.stages[stage]),
                        "sha256": sha256(shot.stages[stage]),
                        "width": probes[shot.shot_id][stage].width,
                        "height": probes[shot.shot_id][stage].height,
                        "frames": probes[shot.shot_id][stage].frames,
                        "fps": fps_text(probes[shot.shot_id][stage].fps),
                    }
                    for stage in STAGES
                },
                "insertion_check": insertions[shot.shot_id],
            }
            for shot in shots
        ],
        "reel": reel,
        "comparison_master": master,
        "non_claims": list(NON_CLAIMS),
    }


def build_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce the report to its scalar facts for terminal output."""
    return {
        "reel": report["reel"],
        "comparison_master": report["comparison_master"],
        "shots": [
            {
                "id": shot["id"],
                "label": shot["label"],
                "max_frame_changed_ratio": shot["insertion_check"]["max_frame_changed_ratio"],
                "exactly_identical": shot["insertion_check"]["exactly_identical"],
            }
            for shot in report["shots"]
        ],
        "non_claims": report["non_claims"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    result.add_argument("--manifest", type=Path, required=True, help="advertising spot manifest")
    result.add_argument("--reel", type=Path, help="labeled sequential reel output path")
    result.add_argument(
        "--master",
        type=Path,
        help="labeled comparison master output path",
    )
    result.add_argument("--report", type=Path, help="optional JSON report path")
    result.add_argument(
        "--reel-stage",
        choices=STAGES,
        default="pass2",
        help="stage shown in the sequential reel",
    )
    result.add_argument(
        "--changed-threshold",
        type=float,
        default=DEFAULT_CHANGED_THRESHOLD,
        help="mean absolute per-channel difference above which a pixel counts as changed",
    )
    result.add_argument(
        "--min-insertion-changed-ratio",
        type=float,
        default=DEFAULT_MIN_INSERTION_CHANGED_RATIO,
        help=(
            "smallest changed-pixel ratio in any one frame that still counts as a real "
            "insertion over the Pass 1 clean plate"
        ),
    )
    result.add_argument(
        "--workdir", type=Path, help="optional scratch directory for decoded frames"
    )
    result.add_argument(
        "--lossless",
        action="store_true",
        help="encode lossless yuv444p instead of the delivery yuv420p rendition",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not math.isfinite(args.changed_threshold) or args.changed_threshold < 0:
        raise AdvertisingReelError("--changed-threshold must be a finite nonnegative number")
    ratio = args.min_insertion_changed_ratio
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise AdvertisingReelError("--min-insertion-changed-ratio must be a finite ratio in [0, 1]")
    manifest_path = args.manifest
    if not manifest_path.is_file():
        raise AdvertisingReelError(f"manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdvertisingReelError(f"manifest is not valid JSON: {error}") from error
    shots = plan_shots(manifest_path, payload)
    reel_path = args.reel or manifest_path.parent / f"{manifest_path.stem}-reel.mp4"
    master_path = (
        args.master or manifest_path.parent / f"{manifest_path.stem}-comparison-master.mp4"
    )
    if reel_path.resolve() == master_path.resolve():
        raise AdvertisingReelError("--reel and --master must name different files")
    probes: dict[str, dict[str, StageProbe]] = {}
    for shot in shots:
        shot_probes = {stage: probe_stage(shot.stages[stage]) for stage in STAGES}
        validate_shot_alignment(shot.shot_id, shot_probes)
        probes[shot.shot_id] = shot_probes
    validate_spot_alignment(shots, probes)
    fps = probes[shots[0].shot_id][STAGES[0]].fps
    insertions: dict[str, dict[str, Any]] = {}
    with scratch_directory(args.workdir or reel_path.parent) as scratch:
        root = Path(scratch)
        frames_by_shot: dict[str, dict[str, list[Path]]] = {}
        for index, shot in enumerate(shots):
            shot_root = root / f"shot-{index:02d}"
            shot_frames = {
                stage: extract_stage_frames(stage, shot.stages[stage], shot_root / stage)
                for stage in STAGES
            }
            for stage in STAGES:
                expected_frames = probes[shot.shot_id][stage].frames
                if len(shot_frames[stage]) != expected_frames:
                    raise AdvertisingReelError(
                        f"shot {shot.shot_id} stage {stage} decoded "
                        f"{len(shot_frames[stage])} frames, expected {expected_frames}"
                    )
            frames_by_shot[shot.shot_id] = shot_frames
            metrics = measure_insertion(
                shot_frames["pass1_clean_plate"],
                shot_frames["inserted"],
                changed_threshold=args.changed_threshold,
            )
            validate_insertion(
                shot.shot_id,
                metrics,
                min_changed_ratio=args.min_insertion_changed_ratio,
            )
            insertions[shot.shot_id] = metrics
        reel_frames, reel_width, reel_height = build_reel_frames(
            shots,
            frames_by_shot,
            args.reel_stage,
            root / "reel",
        )
        master_frames, master_width, master_height = build_master_frames(
            shots,
            frames_by_shot,
            root / "master",
        )
        encode_frames(root / "reel", reel_path, fps, lossless=args.lossless)
        encode_frames(root / "master", master_path, fps, lossless=args.lossless)
    reel = {
        "path": str(reel_path),
        "sha256": sha256(reel_path),
        "width": reel_width,
        "height": reel_height,
        "frames": reel_frames,
        "fps": fps_text(fps),
        "stage": args.reel_stage,
        "lossless": bool(args.lossless),
        "shots": len(shots),
    }
    master = {
        "path": str(master_path),
        "sha256": sha256(master_path),
        "width": master_width,
        "height": master_height,
        "frames": master_frames,
        "fps": fps_text(fps),
        "columns": list(STAGES),
        "rows": len(shots),
        "lossless": bool(args.lossless),
    }
    report = build_report(
        manifest_path,
        shots,
        probes,
        insertions,
        reel,
        master,
        {
            "changed_threshold": args.changed_threshold,
            "min_insertion_changed_ratio": args.min_insertion_changed_ratio,
        },
        args.reel_stage,
    )
    document = json.dumps(report, indent=2, allow_nan=False)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(document + "\n", encoding="utf-8")
        print(json.dumps(build_summary(report), indent=2, allow_nan=False))
    else:
        print(document)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AdvertisingReelError,
        MediaError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Advertising reel assembly failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
