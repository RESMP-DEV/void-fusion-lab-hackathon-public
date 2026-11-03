#!/usr/bin/env python3
"""Measure per-frame lower-body ownership from MediaPipe pose JSON.

The pose source is MediaPipe Pose Landmarker output: normalized ``x``/``y``,
relative ``z``, and ``visibility`` per landmark. It is not DWPose and not
RTMPose. The script reads one JSON document, applies the fail-closed
ownership gates from ``samurai.lower_body``, and writes one strict JSON
report. It sends no model, credential, or network request, and it embeds no
absolute path in the report.

Run it from the repository root with the project environment::

    uv run python scripts/lower_body_ownership.py \
        --input <pose-document.json> --output <report.json>

The script is executable, so ``./scripts/lower_body_ownership.py`` works too
when the ``samurai`` package is importable from the active environment.

Exit codes: 0 the report holds at least one owned lower body, 2 the input
document or a threshold is unusable, 3 the report holds no owned lower body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from samurai.lower_body import (
    DEFAULT_REQUIRED_CHAINS,
    LOWER_BODY_CHAINS,
    LowerBodyError,
    OwnershipThresholds,
    build_report,
    report_payload,
)

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_NO_OWNER = 3


def _reject_constant(name: str) -> float:
    raise LowerBodyError(f"pose document holds the non-finite constant {name}")


def parse_document(raw: bytes) -> Any:
    """Decode a strict JSON pose document; non-finite constants are refused."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LowerBodyError(f"pose document is not valid UTF-8: {error}") from error
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise LowerBodyError(f"pose document is not valid JSON: {error}") from error


def _read_bytes(path: Path) -> bytes:
    if path.name == "-":
        return sys.stdin.buffer.read()
    return path.read_bytes()


def _parse_chains(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise LowerBodyError("required chains must name at least one chain")
    unknown = [name for name in names if name not in LOWER_BODY_CHAINS]
    if unknown:
        known = ", ".join(sorted(LOWER_BODY_CHAINS))
        listed = ", ".join(sorted(unknown))
        raise LowerBodyError(f"unknown required chains: {listed}; known chains: {known}")
    return names


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--input",
        type=Path,
        required=True,
        help="MediaPipe pose JSON document, or - to read stdin",
    )
    result.add_argument("--output", type=Path, help="optional path for the JSON report")
    result.add_argument("--min-visibility", type=float, default=0.5)
    result.add_argument("--max-abs-z", type=float, default=1.0)
    result.add_argument("--min-segment-length", type=float, default=0.01)
    result.add_argument("--max-segment-length", type=float, default=1.5)
    result.add_argument("--max-pelvis-jump", type=float, default=0.2)
    result.add_argument("--max-landmark-jump", type=float, default=0.35)
    result.add_argument(
        "--required-chains",
        default=",".join(DEFAULT_REQUIRED_CHAINS),
        help="comma separated chain names that a pose must supply",
    )
    result.add_argument(
        "--require-previous-frame",
        action="store_true",
        help="refuse ownership on the first frame a pose appears in",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Emit one strict JSON ownership report for one pose document."""
    args = parser().parse_args(argv)
    try:
        raw = _read_bytes(args.input)
        document = parse_document(raw)
        thresholds = OwnershipThresholds(
            min_visibility=args.min_visibility,
            max_abs_z=args.max_abs_z,
            min_segment_length=args.min_segment_length,
            max_segment_length=args.max_segment_length,
            max_pelvis_jump=args.max_pelvis_jump,
            max_landmark_jump=args.max_landmark_jump,
            required_chains=_parse_chains(args.required_chains),
            require_previous_frame=args.require_previous_frame,
        )
        report = build_report(document, thresholds)
        payload = report_payload(report)
        payload["input"] = {
            "name": args.input.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
    except (LowerBodyError, OSError, ValueError) as error:
        print(f"Lower-body ownership failed: {error}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    sys.stdout.write(text)
    return EXIT_OK if report.owned_pose_count else EXIT_NO_OWNER


if __name__ == "__main__":
    raise SystemExit(main())
