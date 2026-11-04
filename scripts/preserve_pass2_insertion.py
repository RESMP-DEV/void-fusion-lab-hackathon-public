#!/usr/bin/env python3
"""Enforce the inserted-object keep contract after VOID Pass 2.

The insertion quadmask marks the fitted 3D object as KEEP.  When diffusion
recolors that object, this lane copies the pre-Pass-2 Blender render back into
the ROI-black inserted region and keeps the Pass-2 result everywhere else.  The
cleanup ring and background therefore remain model-refined while the advertised
object remains the persisted true-3D asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from advertising_tire_asset import ffmpeg_framerate
from PIL import Image

from samurai.media import (
    MediaError,
    assert_video_alignment,
    iter_color_frames,
    probe_video,
)

SCHEMA = "void-fusion-lab.pass2-insertion-preserve.v1"
EXIT_INPUT_ERROR = 2
MIN_BRAND_PIXELS = 24


class PreserveInsertionError(RuntimeError):
    """An input or media contract for insertion preservation failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"revision": revision, "dirty": bool(status)}


def reject_path_aliases(paths: dict[str, Path]) -> None:
    """Fail closed when two preservation roles resolve to one path."""
    resolved: dict[Path, str] = {}
    for label, path in paths.items():
        key = path.expanduser().resolve(strict=False)
        if key in resolved:
            raise PreserveInsertionError(
                f"--{label.replace('_', '-')} and --{resolved[key].replace('_', '-')} "
                "must name different files"
            )
        resolved[key] = label


def brand_pixels(frame: np.ndarray, inserted: np.ndarray) -> int:
    """Count yellow advertising-band pixels inside one ROI-black object mask."""
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    yellow = (red >= 120) & (green >= 90) & ((red - blue) >= 45)
    return int(np.count_nonzero(yellow & (inserted == 0)))


def preserve_frame(baseline: np.ndarray, pass2: np.ndarray, inserted: np.ndarray) -> np.ndarray:
    """Return one frame with the baseline object and Pass-2 surroundings."""
    keep = inserted == 0
    return np.where(keep[..., None], baseline, pass2).astype(np.uint8)


def write_yuv444_video(frames: list[np.ndarray], output: Path, *, fps: float) -> None:
    """Encode RGB frames as constant-framerate H.264 yuv444p."""
    if not frames or output.exists():
        raise PreserveInsertionError("preservation requires new frames and an absent output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".preserve-insertion-") as temporary:
        frame_root = Path(temporary)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_root / f"frame-{index:06d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                ffmpeg_framerate(fps),
                "-i",
                str(frame_root / "frame-%06d.png"),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-bf",
                "0",
                "-qp",
                "0",
                "-pix_fmt",
                "yuv444p",
                "-an",
                str(output),
                "-y",
            ],
            check=True,
        )


def input_record(path: Path, label: str) -> dict[str, Any]:
    info = probe_video(path)
    return {
        "path": str(path),
        "label": label,
        "sha256": sha256(path),
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frames": info.frames,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    reject_path_aliases(
        {
            "baseline": args.baseline,
            "pass2": args.pass2,
            "inserted_mask": args.inserted_mask,
            "output": args.output,
            "report": args.report,
        }
    )
    for label, path in (
        ("baseline", args.baseline),
        ("pass2", args.pass2),
        ("inserted mask", args.inserted_mask),
    ):
        if not path.is_file():
            raise PreserveInsertionError(f"the {label} does not exist: {path}")
    if args.output.exists():
        raise PreserveInsertionError(f"the output already exists: {args.output}")

    reference = probe_video(args.baseline)
    assert_video_alignment(
        {
            "baseline": args.baseline,
            "pass2": args.pass2,
            "inserted_mask": args.inserted_mask,
        },
        width=reference.width,
        height=reference.height,
        frames=reference.frames,
    )
    preserved: list[np.ndarray] = []
    object_pixels = 0
    pass2_pixels = 0
    brand_counts: list[int] = []
    with TemporaryDirectory(prefix=".preserve-insertion-media-") as temporary:
        root = Path(temporary)
        baselines = iter_color_frames(args.baseline, root / "baseline")
        results = iter_color_frames(args.pass2, root / "pass2")
        masks = iter_color_frames(args.inserted_mask, root / "mask")
        for baseline, result, mask_rgb in zip(baselines, results, masks, strict=True):
            inserted = mask_rgb[:, :, 0]
            values = np.unique(inserted)
            if not np.isin(values, (0, 255)).all() or not np.any(inserted == 0):
                raise PreserveInsertionError(
                    f"inserted mask frame {len(preserved)} is not nonempty ROI-black binary"
                )
            frame = preserve_frame(baseline, result, inserted)
            keep = inserted == 0
            object_pixels += int(np.count_nonzero(keep))
            pass2_pixels += int(np.count_nonzero(~keep))
            brand_counts.append(brand_pixels(frame, inserted))
            preserved.append(frame)

    write_yuv444_video(preserved, args.output, fps=reference.fps)
    output_info = probe_video(args.output)
    if (output_info.width, output_info.height, output_info.frames) != (
        reference.width,
        reference.height,
        reference.frames,
    ) or output_info.fps != reference.fps:
        raise PreserveInsertionError(f"the preserved output geometry changed: {output_info}")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "scope": (
            "Copied the pre-Pass-2 Blender object inside the ROI-black keep "
            "region and retained the VOID Pass-2 cleanup ring and background."
        ),
        "inputs": {
            "baseline": input_record(args.baseline, "pre-Pass-2 true-3D composite"),
            "pass2": input_record(args.pass2, "retimed VOID Pass-2 result"),
            "inserted_mask": input_record(args.inserted_mask, "ROI-black inserted object"),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "width": output_info.width,
            "height": output_info.height,
            "fps": output_info.fps,
            "frames": output_info.frames,
            "pix_fmt": "yuv444p",
            "codec": "h264",
            "qp": 0,
        },
        "summary": {
            "kept_object_pixels": object_pixels,
            "pass2_source_pixels": pass2_pixels,
            "min_brand_pixels": min(brand_counts),
            "frames_with_brand_band": sum(value >= MIN_BRAND_PIXELS for value in brand_counts),
            "brand_gate": {
                "min_brand_pixels_required": MIN_BRAND_PIXELS,
                "passed": min(brand_counts) >= MIN_BRAND_PIXELS
                and sum(value >= MIN_BRAND_PIXELS for value in brand_counts) == reference.frames,
            },
        },
        "non_claims": [
            "This deterministic keep-contract composite does not run a model.",
            "Pass-2 execution and cleanup-ring measurements remain bound to the "
            "supplied retimed result and comparator report.",
            "Brand-pixel visibility is a color-region gate, not OCR or human readability proof.",
            "QP0 H.264 yuv444p is codec-lossless; RGB-to-YUV color conversion can "
            "still introduce small integer rounding on RGB decode.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pass2", type=Path, required=True)
    parser.add_argument("--inserted-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (PreserveInsertionError, MediaError, OSError, subprocess.CalledProcessError) as error:
        print(f"insertion preservation failed: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": report["output"]["sha256"],
                "min_brand_pixels": report["summary"]["min_brand_pixels"],
                "brand_gate_passed": report["summary"]["brand_gate"]["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
