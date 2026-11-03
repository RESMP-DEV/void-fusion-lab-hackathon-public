#!/usr/bin/env python3
"""Measure deterministic contact and shadow constraints for a fitted insertion.

The script reads per-frame lower-body keypoints, the inserted object's
region-of-interest mask, and optionally the untouched plate, then writes one
JSON report and one lossless gray shadow mask video.  It calls no model, opens
no network connection, and invokes no renderer: the outputs are measurements
and a suggested mask, never a quality verdict.

Mask convention follows the repository: the region of interest is black (0)
and the rest is white (255).  The shadow mask uses the same convention, so a
shadow pixel is 0 and a keep pixel is 255.  That is the ROI-black convention
only: the shadow mask is not a VOID quadmask, carries none of the
0/63/127/255 quadmask values, and must not be read by the VOID lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from samurai.contact_grounding import (
    NON_CLAIMS,
    ContactGroundingError,
    GroundingSettings,
    ground_contact,
    mask_digest,
    parse_keypoint_frames,
    summarize,
)
from samurai.media import (
    GrayVideoWriter,
    MediaError,
    decode_gray_frames,
    probe_video,
)

SCHEMA = "void-fusion-lab.contact-grounding.report.v1"
EXIT_INPUT_ERROR = 2
FRAME_NAME = re.compile(r"frame-(\d{6})\.png")


class ContactGroundingScriptError(RuntimeError):
    """A script input could not be read or is inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(files: Sequence[Path]) -> str:
    """Digest a frame directory deterministically.

    A directory has no single file digest, so the report records a null file
    digest and this directory digest instead. The digest covers each frame's
    name and bytes in frame order, so it is stable across runs and machines
    while still binding the exact frames the run measured.
    """
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(file)))
        digest.update(b"\0")
    return digest.hexdigest()


def input_digests(path: Path) -> dict[str, str | None]:
    """Return the digest fields for one input path.

    A file carries its own SHA-256. A frame directory carries a null file
    digest plus a directory digest, so an empty string can never be mistaken
    for a recorded digest.
    """
    if path.is_file():
        return {"sha256": sha256(path), "directory_sha256": None}
    if path.is_dir():
        return {"sha256": None, "directory_sha256": directory_sha256(frame_directory_files(path))}
    return {"sha256": None, "directory_sha256": None}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContactGroundingScriptError(f"invalid JSON document {path}: {error}") from error


def load_object_masks(path: Path, *, invert: bool) -> tuple[list[np.ndarray], float]:
    """Load the inserted-object ROI mask as boolean frames plus the clip fps."""
    if path.is_dir():
        files = frame_directory_files(path)
        frames: list[np.ndarray] = []
        for file in files:
            with Image.open(file) as image:
                frames.append(np.asarray(image.convert("L")))
        return [_to_roi(frame, invert=invert) for frame in frames], 0.0
    if not path.is_file():
        raise ContactGroundingScriptError(f"object mask does not exist: {path}")
    info = probe_video(path)
    with TemporaryDirectory(prefix=".samurai-contact-mask-") as temporary:
        decoded = decode_gray_frames(path, Path(temporary), expected_frames=info.frames)
    return [_to_roi(frame, invert=invert) for frame in decoded], info.fps


def _to_roi(frame: np.ndarray, *, invert: bool) -> np.ndarray:
    if invert:
        return frame >= 128
    return frame < 128


def frame_directory_files(path: Path) -> list[Path]:
    """Return a frame directory's files, enforcing contiguous frame numbering.

    The directory must hold a contiguous run of frame-NNNNNN.png files with no
    gap and no stray image.  The run may start at any index: this repository's
    own extractors and stock ffmpeg image2 output start at frame-000001.png,
    while hand-prepared dumps often start at frame-000000.png.  Frames are
    indexed by list position, so a gap would silently renumber the frames, a
    keyed frame could be measured against the wrong image, and a stray image
    would be dropped without a record.
    """
    files = sorted(path.glob("*.png"))
    if not files:
        raise ContactGroundingScriptError(f"no PNG frames found in {path}")
    numbered: list[tuple[int, Path]] = []
    for file in files:
        match = FRAME_NAME.fullmatch(file.name)
        if match is None:
            raise ContactGroundingScriptError(
                f"frame directory {path} holds {file.name}, which is not a frame-NNNNNN.png name"
            )
        numbered.append((int(match.group(1)), file))
    numbers = sorted(index for index, _ in numbered)
    first = numbers[0]
    if numbers != list(range(first, first + len(numbered))):
        last = first + len(numbered) - 1
        raise ContactGroundingScriptError(
            f"frame directory {path} is not contiguous: expected frame-{first:06d}.png "
            f"through frame-{last:06d}.png"
        )
    return [file for _, file in sorted(numbered)]


def extract_selected_color_frames(video: Path, destination: Path, wanted: list[int]) -> list[Path]:
    """Extract only the requested frame indices as color PNGs."""
    expression = "+".join(f"eq(n\\,{index})" for index in sorted(wanted))
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select={expression}",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            str(destination / "selected-%06d.png"),
        ],
        check=True,
    )
    files = sorted(destination.glob("selected-*.png"))
    if len(files) != len(set(wanted)):
        raise ContactGroundingScriptError(
            f"decoded {len(files)} selected plate frames, expected {len(set(wanted))}"
        )
    return files


def read_selected_plate_frames(
    files: list[Path], wanted: list[int], total_frames: int
) -> list[np.ndarray | None]:
    """Map selectively decoded PNGs back to their source frame indices."""
    frames: list[np.ndarray | None] = [None] * total_frames
    for index, file in zip(sorted(set(wanted)), files, strict=True):
        with Image.open(file) as image:
            frames[index] = np.asarray(image.convert("RGB")).copy()
    return frames


def load_plate_frames(path: Path, wanted: list[int]) -> list[np.ndarray | None]:
    """Load only the plate frames the keypoints reference.

    The plate is either a video or a directory of PNG frames; a directory keeps
    the tool usable when no encoder is available.

    A video is extracted into a temporary directory, so its frames are read
    while that directory still exists: the extracted images are gone once the
    temporary directory is removed.
    """
    if path.is_dir():
        files = frame_directory_files(path)
        return read_plate_frames(files, wanted)
    elif path.is_file():
        info = probe_video(path)
        with TemporaryDirectory(prefix=".samurai-contact-plate-") as temporary:
            files = extract_selected_color_frames(path, Path(temporary), wanted)
            return read_selected_plate_frames(files, wanted, info.frames)
    else:
        raise ContactGroundingScriptError(f"plate does not exist: {path}")


def read_plate_frames(files: list[Path], wanted: list[int]) -> list[np.ndarray | None]:
    """Read the requested plate frames; the caller owns the files' lifetime."""
    frames: list[np.ndarray | None] = [None] * len(files)
    for index in sorted(set(wanted)):
        if not 0 <= index < len(files):
            raise ContactGroundingScriptError(f"plate frame {index} is out of range")
        with Image.open(files[index]) as image:
            frames[index] = np.asarray(image.convert("RGB")).copy()
    return frames


def build_settings(args: argparse.Namespace) -> GroundingSettings:
    return GroundingSettings(
        contact_thickness_ratio=args.contact_thickness_ratio,
        contact_spread_ratio=args.contact_spread_ratio,
        shadow_length_ratio=args.shadow_length_ratio,
        shadow_spread_ratio=args.shadow_spread_ratio,
        vertical_tolerance_ratio=args.vertical_tolerance_ratio,
        max_foot_lift_ratio=args.max_foot_lift_ratio,
        min_plate_samples=args.min_plate_samples,
        clip_shadow_to_ground=args.clip_shadow_to_ground,
        min_light_confidence=args.min_light_confidence,
        contact_line_min_baseline_ratio=args.contact_line_min_baseline_ratio,
    )


def build_report(
    *,
    args: argparse.Namespace,
    keypoints: Any,
    mask_frame_count: int,
    fps: float,
    plate_frame_count: int | None,
    measurements: list[Any],
    shadow_digests: dict[int, str],
    settings: GroundingSettings,
    mask_info: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize(measurements, mask_frames=mask_frame_count)
    return {
        "schema": SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "keypoints": {
                "path": str(args.keypoints),
                **input_digests(args.keypoints),
                "frames": len(keypoints),
            },
            "object_mask": {
                "path": str(args.object_mask),
                **input_digests(args.object_mask),
                "frames": mask_frame_count,
                "width": mask_info["width"],
                "height": mask_info["height"],
                "fps": fps,
                "roi_convention": "bright" if args.mask_invert else "dark",
            },
            "plate": (
                {
                    "path": str(args.plate),
                    **input_digests(args.plate),
                    "frames": plate_frame_count,
                }
                if args.plate is not None
                else None
            ),
        },
        "settings": asdict(settings),
        "frames": [
            {
                **asdict(measurement),
                "shadow_digest": shadow_digests[measurement.frame],
            }
            for measurement in measurements
        ],
        "summary": summary,
        "outputs": {
            "shadow_mask": {
                "path": str(args.out / "shadow-mask.mp4"),
                "sha256": sha256(args.out / "shadow-mask.mp4"),
                "mask_family": "roi-black",
                "convention": (
                    "ROI-black: shadow pixels are 0, keep pixels are 255. This is "
                    "not a VOID quadmask and carries none of the 0/63/127/255 values."
                ),
            }
        },
        "non_claims": list(NON_CLAIMS),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--keypoints", type=Path, required=True)
    result.add_argument("--object-mask", type=Path, required=True)
    result.add_argument("--plate", type=Path)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--fps", type=float)
    result.add_argument("--mask-invert", action="store_true")
    result.add_argument("--contact-thickness-ratio", type=float, default=0.06)
    result.add_argument("--contact-spread-ratio", type=float, default=0.05)
    result.add_argument("--shadow-length-ratio", type=float, default=0.6)
    result.add_argument("--shadow-spread-ratio", type=float, default=1.0)
    result.add_argument("--vertical-tolerance-ratio", type=float, default=0.03)
    result.add_argument("--max-foot-lift-ratio", type=float, default=0.12)
    result.add_argument("--min-plate-samples", type=int, default=64)
    result.add_argument("--min-light-confidence", type=float, default=0.5)
    result.add_argument(
        "--contact-line-min-baseline-ratio",
        type=float,
        default=0.05,
    )
    result.add_argument("--clip-shadow-to-ground", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.keypoints.is_file():
        raise ContactGroundingScriptError(f"keypoints file does not exist: {args.keypoints}")
    keypoints = parse_keypoint_frames(read_json(args.keypoints))
    masks, fps = load_object_masks(args.object_mask, invert=args.mask_invert)
    if not masks:
        raise ContactGroundingScriptError("the object mask sequence is empty")
    height, width = masks[0].shape
    if args.fps is not None:
        fps = args.fps
    if not math.isfinite(fps):
        raise ContactGroundingScriptError("the clip frame rate must be finite")
    if fps <= 0.0:
        raise ContactGroundingScriptError("the clip frame rate is unknown; pass --fps")
    plate_frames: list[np.ndarray | None] | None = None
    if args.plate is not None:
        plate_frames = load_plate_frames(args.plate, [item.frame for item in keypoints])
        if len(plate_frames) != len(masks):
            raise ContactGroundingScriptError(
                f"the plate has {len(plate_frames)} frames but the object mask has {len(masks)}"
            )
        for index, frame in enumerate(plate_frames):
            if frame is not None and frame.shape[:2] != (height, width):
                raise ContactGroundingScriptError(
                    f"plate frame {index} is {frame.shape[:2]} but the object mask is "
                    f"{(height, width)}"
                )
    settings = build_settings(args)
    mask_frame_count = len(masks)
    measurements, shadows = ground_contact(
        keypoints, masks, plate_frames=plate_frames, settings=settings
    )
    plate_frame_count = len(plate_frames) if plate_frames is not None else None
    del masks
    del plate_frames
    args.out.mkdir(parents=True, exist_ok=True)
    shadow_digests: dict[int, str] = {}
    with GrayVideoWriter(
        args.out / "shadow-mask.mp4", width=width, height=height, fps=fps
    ) as writer:
        for index, shadow in enumerate(shadows):
            writer.write(np.where(shadow, 0, 255).astype(np.uint8))
            shadow_digests[index] = mask_digest(shadow)
    del shadows
    report = build_report(
        args=args,
        keypoints=keypoints,
        mask_frame_count=mask_frame_count,
        fps=fps,
        plate_frame_count=plate_frame_count,
        measurements=measurements,
        shadow_digests=shadow_digests,
        settings=settings,
        mask_info={"width": width, "height": height},
    )
    (args.out / "contact-grounding-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frames": report["summary"]["measured_frames"],
                "mean_contact_coverage": report["summary"]["mean_contact_coverage"],
                "frames_within_vertical_tolerance": report["summary"][
                    "frames_within_vertical_tolerance"
                ],
                "shadow_pixels_total": report["summary"]["shadow_pixels_total"],
                "report": str(args.out / "contact-grounding-report.json"),
            },
            indent=2,
        )
    )
    return 0


def _tool_failure_detail(error: subprocess.CalledProcessError) -> str:
    """Summarize a failed tool run: the exit code and the last stderr line."""
    lines = (error.stderr or "").strip().splitlines()
    suffix = f"; {lines[-1]}" if lines else ""
    return f"exit {error.returncode}{suffix}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ContactGroundingError,
        ContactGroundingScriptError,
        MediaError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(
                f"Contact grounding failed: {_tool_failure_detail(error)}",
                file=sys.stderr,
            )
        else:
            print(f"Contact grounding failed: {error}", file=sys.stderr)
        raise SystemExit(EXIT_INPUT_ERROR) from error
