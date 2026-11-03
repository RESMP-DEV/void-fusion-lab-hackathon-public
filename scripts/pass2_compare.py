#!/usr/bin/env python3
"""Compare a no-Pass2 baseline insert against a VOID Pass 2 result.

The comparator reports numbers only: how much Pass 2 changed the inserted
object, the cleanup ring around it, and the untouched background, plus the
the temporal stability of those changes.  It fails closed on geometry,
frame-count, frame-rate, or timestamp mismatch, and it makes no quality
verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from samurai.media import MediaError
from samurai.pass2_metrics import (
    DEFAULT_CHANGED_THRESHOLD,
    DEFAULT_FPS_TOLERANCE,
    NON_CLAIMS,
    FrameTimestamps,
    Pass2Comparison,
    Pass2MetricsError,
    compare_videos,
    fraction_text,
)
from samurai.quadmask import AFFECTED, REMOVE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_record(timestamps: FrameTimestamps) -> dict[str, Any]:
    """Reduce one timestamp probe to a JSON-safe summary and digest."""
    return {
        "name": timestamps.name,
        "frames": timestamps.count,
        "first_pts": fraction_text(timestamps.pts[0]),
        "last_pts": fraction_text(timestamps.pts[-1]),
        "cfr_delta": (
            None if timestamps.cfr_delta is None else fraction_text(timestamps.cfr_delta)
        ),
        "variable_frame_rate": timestamps.variable_frame_rate,
        "pts_digest": timestamps.digest,
    }


def build_report(comparison: Pass2Comparison, inputs: dict[str, Any]) -> dict[str, Any]:
    """Assemble the JSON report for one completed comparison."""
    timestamps = comparison.timestamps
    return {
        "schema": "void-fusion-lab.pass2-compare.report.v3",
        "recorded_at": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "thresholds": {
            "changed_threshold": comparison.changed_threshold,
            "fps_tolerance": comparison.frame_alignment.fps_tolerance,
            "cleanup_value": comparison.cleanup_value,
        },
        "frame_alignment": asdict(comparison.frame_alignment),
        "frame_timestamps": {
            "aligned": timestamps.aligned,
            "pts_aligned": timestamps.pts_aligned,
            "frame_rate_character_aligned": timestamps.frame_rate_character_aligned,
            "baseline": timestamp_record(timestamps.baseline),
            "pass2": timestamp_record(timestamps.pass2),
            "inserted_mask": timestamp_record(timestamps.inserted_mask),
            "cleanup_mask": timestamp_record(timestamps.cleanup_mask),
        },
        "object_preservation": asdict(comparison.object_preservation),
        "cleanup_ring": asdict(comparison.cleanup_ring),
        "background": asdict(comparison.background),
        "temporal": asdict(comparison.temporal),
        "non_claims": list(NON_CLAIMS),
    }


def build_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce the report to its scalar metrics for terminal output."""
    preservation = report["object_preservation"]
    ring = report["cleanup_ring"]
    background = report["background"]
    temporal = report["temporal"]
    return {
        "object_preservation": {
            "mae": preservation["mae"],
            "psnr_db": preservation["psnr_db"],
            "changed_ratio": preservation["changed_ratio"],
        },
        "cleanup_ring": {
            "mae": ring["mae"],
            "changed_ratio": ring["changed_ratio"],
            "changed_pixels": ring["changed_pixels"],
        },
        "background": {
            "mae": background["mae"],
            "changed_ratio": background["changed_ratio"],
        },
        "temporal": {
            "temporal_pop_score": temporal["temporal_pop_score"],
            "centroid_pop": temporal["centroid_pop"],
            "mass_pop": temporal["mass_pop"],
            "frames_with_difference": temporal["frames_with_difference"],
            "no_changed_pixels": temporal["no_changed_pixels"],
            "exactly_identical": temporal["exactly_identical"],
            "measured_frames": temporal["measured_frames"],
            "structural_lead_in": temporal["structural_lead_in"],
            "lead_in_changed_pixels": temporal["lead_in_changed_pixels"],
            "lead_in_violation": temporal["lead_in_violation"],
        },
        "aligned": report["frame_alignment"]["aligned"] and report["frame_timestamps"]["aligned"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="inserted video before VOID Pass 2",
    )
    result.add_argument(
        "--pass2",
        type=Path,
        required=True,
        help="video after VOID Pass 2",
    )
    result.add_argument(
        "--inserted-mask",
        type=Path,
        required=True,
        help="ROI-black mask video of the inserted object",
    )
    result.add_argument(
        "--cleanup-mask",
        type=Path,
        required=True,
        help="ROI-black mask video of the cleanup ring",
    )
    result.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path; the report is printed when this is omitted",
    )
    result.add_argument(
        "--changed-threshold",
        type=float,
        default=DEFAULT_CHANGED_THRESHOLD,
        help="mean absolute per-channel difference above which a pixel counts as changed",
    )
    result.add_argument(
        "--fps-tolerance",
        type=float,
        default=DEFAULT_FPS_TOLERANCE,
        help="relative frame-rate tolerance allowed across the compared inputs",
    )
    result.add_argument(
        "--cleanup-value",
        type=int,
        choices=(REMOVE, AFFECTED),
        default=AFFECTED,
        help=(
            "VOID quadmask value of the cleanup ring: "
            f"{AFFECTED} affected or {REMOVE} remove; with {AFFECTED} VOID clears the "
            "affected-only ring on frame zero"
        ),
    )
    result.add_argument(
        "--workdir",
        type=Path,
        help="optional working directory for decoded frames",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name, value in (
        ("--changed-threshold", args.changed_threshold),
        ("--fps-tolerance", args.fps_tolerance),
    ):
        if not math.isfinite(value):
            raise Pass2MetricsError(f"{name} must be finite, got {value}")
    if args.changed_threshold < 0:
        raise Pass2MetricsError("--changed-threshold must be nonnegative")
    if args.fps_tolerance < 0:
        raise Pass2MetricsError("--fps-tolerance must be nonnegative")
    inputs = {
        "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline)},
        "pass2": {"path": str(args.pass2), "sha256": sha256(args.pass2)},
        "inserted_mask": {"path": str(args.inserted_mask), "sha256": sha256(args.inserted_mask)},
        "cleanup_mask": {"path": str(args.cleanup_mask), "sha256": sha256(args.cleanup_mask)},
    }
    comparison = compare_videos(
        baseline=args.baseline,
        pass2=args.pass2,
        inserted_mask=args.inserted_mask,
        cleanup_mask=args.cleanup_mask,
        changed_threshold=args.changed_threshold,
        fps_tolerance=args.fps_tolerance,
        cleanup_value=args.cleanup_value,
        workdir=args.workdir,
    )
    report = build_report(comparison, inputs)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(json.dumps(build_summary(report), indent=2, allow_nan=False))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        Pass2MetricsError,
        MediaError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Pass 2 comparison failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
