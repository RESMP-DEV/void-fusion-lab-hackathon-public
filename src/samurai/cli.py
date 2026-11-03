"""Command-line surface for SAM 3.1 segmentation and VOID execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .calibration import BetaSmoothing, CalibrationSettings, calibrate_config
from .ensemble import load_config as load_ensemble_config
from .ensemble import run_ensemble
from .quadmask import build_insertion_sequence, build_sequence
from .sam31 import SegmentationPrompt, segment_video
from .void import execute, load_config


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _point(value: str) -> tuple[float, float]:
    parts = value.split(",", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be X,Y")
    try:
        point = (float(parts[0]), float(parts[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError("point coordinates must be numeric") from error
    if point[0] < 0 or point[1] < 0:
        raise argparse.ArgumentTypeError("point coordinates must be nonnegative")
    return point


def _box(value: str) -> tuple[float, float, float, float]:
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be X1,Y1,X2,Y2")
    try:
        box = tuple(float(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("box coordinates must be numeric") from error
    if box[0] >= box[2] or box[1] >= box[3]:
        raise argparse.ArgumentTypeError("box must have positive width and height")
    return box[0], box[1], box[2], box[3]


def _doctor(args: argparse.Namespace) -> int:
    sam3_spec = importlib.util.find_spec("sam3")
    torch_spec = importlib.util.find_spec("torch")
    cuda_available = False
    torch_version = None
    sam3_importable = False
    sam3_import_error = None
    if torch_spec is not None:
        torch = importlib.import_module("torch")

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    if sam3_spec is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
                sam3 = importlib.import_module("sam3")

            sam3_importable = hasattr(sam3, "build_sam3_predictor")
        except Exception as error:
            sam3_import_error = f"{type(error).__name__}: {error}"
    checkpoint = os.environ.get("SAMURAI_SAM31_CHECKPOINT")
    void_root = os.environ.get("SAMURAI_VOID_ROOT")
    aimlapi_key = os.environ.get("AIMLAPI_KEY")
    checks = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "torch_installed": torch_spec is not None,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "sam3_installed": sam3_spec is not None,
        "sam3_importable": sam3_importable,
        "sam3_import_error": sam3_import_error,
        "sam31_checkpoint_configured": bool(checkpoint),
        "sam31_checkpoint_exists": bool(checkpoint and Path(checkpoint).is_file()),
        "void_root_configured": bool(void_root),
        "void_reference_exists": bool(
            void_root and (Path(void_root) / "inference/cogvideox_fun/predict_v2v.py").is_file()
        ),
        "aimlapi_key_configured": bool(aimlapi_key),
    }
    _emit({"checks": checks, "strict": args.strict})
    required = (
        ["ffmpeg", "ffprobe"]
        + (
            ["torch_installed", "cuda_available", "sam3_installed", "sam3_importable"]
            if args.strict
            else []
        )
        + (["sam31_checkpoint_exists"] if args.strict and checkpoint else [])
        + (["void_reference_exists"] if args.strict and void_root else [])
    )
    return 1 if any(not checks[name] for name in required) else 0


def _segment(args: argparse.Namespace) -> int:
    # Build and validate the prompt BEFORE importing torch/sam3 and BEFORE
    # building the predictor: an invalid combination must fail closed
    # without paying the model load, and must not be masked by an
    # import/CUDA error on a CUDA-less host. main() maps ValueError to 2.
    prompt = SegmentationPrompt(
        frame_index=args.frame_index,
        text=args.text,
        points=tuple(args.point or ()),
        point_labels=tuple(args.point_label or ()),
        boxes_xyxy=tuple(args.box or ()),
        object_id=args.object_id,
        collective=args.collective,
    )
    prompt.validate()
    try:
        torch = importlib.import_module("torch")
        model_builder = importlib.import_module("sam3.model_builder")
        build_sam3_predictor = model_builder.build_sam3_predictor
    except ImportError as error:
        raise SystemExit(
            f"SAM 3.1 extra is not installed; run 'uv sync --extra sam31': {error}"
        ) from error
    if not torch.cuda.is_available():
        raise SystemExit("SAM 3.1's public predictor requires an available CUDA device")
    predictor = build_sam3_predictor(
        checkpoint_path=args.checkpoint,
        version="sam3.1",
        compile=args.compile,
        use_fa3=args.use_fa3,
        max_num_objects=args.max_objects,
    )
    if args.scores_output:
        result = segment_video(predictor, args.video, prompt, args.output, args.scores_output)
    else:
        result = segment_video(predictor, args.video, prompt, args.output)
    _emit(result)
    return 0


def _quadmask(args: argparse.Namespace) -> int:
    result = build_sequence(
        input_video=args.input_video,
        primary_mask=args.primary_mask,
        affected_mask=args.affected_mask,
        background_prompt=args.background_prompt,
        sequence_dir=args.sequence_dir,
    )
    _emit(result.__dict__)
    return 0


def _insert_quadmask(args: argparse.Namespace) -> int:
    result = build_insertion_sequence(
        input_video=args.input_video,
        inserted_mask=args.inserted_mask,
        cleanup_mask=args.cleanup_mask,
        background_prompt=args.background_prompt,
        sequence_dir=args.sequence_dir,
        cleanup_value=args.cleanup_value,
    )
    _emit(result.__dict__)
    return 0


def _ensemble(args: argparse.Namespace) -> int:
    config = load_ensemble_config(args.config)
    result = run_ensemble(config, args.output, args.report)
    _emit(result)
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    settings = CalibrationSettings(
        positive_prior=BetaSmoothing(args.positive_prior_alpha, args.positive_prior_beta),
        sensitivity=BetaSmoothing(args.sensitivity_alpha, args.sensitivity_beta),
        false_positive_rate=BetaSmoothing(
            args.false_positive_rate_alpha, args.false_positive_rate_beta
        ),
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        max_samples=args.max_samples,
        max_batch_pixels=args.max_batch_pixels,
        seed=args.seed,
    )
    report = calibrate_config(
        args.config,
        args.config_output,
        args.report,
        settings,
        latent_output=args.latent_output,
    )
    _emit(report)
    return 0


def _void(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    results = execute(config, dry_run=args.dry_run)
    _emit({"results": results})
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="samurai")
    subparsers = result.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="report local execution readiness")
    doctor.add_argument("--strict", action="store_true", help="fail on optional lanes")
    doctor.set_defaults(handler=_doctor)

    segment = subparsers.add_parser("segment", help="track objects with SAM 3.1")
    segment.add_argument("video", type=Path, help="input MP4 or JPEG directory")
    segment.add_argument("--text", help="open-vocabulary text prompt")
    segment.add_argument("--point", type=_point, action="append", help="absolute X,Y point")
    segment.add_argument(
        "--point-label", type=int, choices=(0, 1), action="append", help="point label"
    )
    segment.add_argument("--box", type=_box, action="append", help="absolute X1,Y1,X2,Y2 box")
    segment.add_argument("--frame-index", type=int, default=0)
    segment.add_argument(
        "--object-id",
        type=int,
        help=(
            "stable SAM 3.1 tracker object ID; required with point prompts and valid only with them"
        ),
    )
    segment.add_argument(
        "--collective",
        action="store_true",
        help="declare the union over all detections as the intended collective referent",
    )
    segment.add_argument("--output", type=Path, required=True)
    segment.add_argument(
        "--scores-output",
        type=Path,
        help="optional .npy probability support tensor for expert fusion",
    )
    segment.add_argument("--checkpoint", help="local sam3.1_multiplex.pt path")
    segment.add_argument("--compile", action="store_true")
    segment.add_argument("--use-fa3", action="store_true")
    segment.add_argument("--max-objects", type=int, default=16)
    segment.set_defaults(handler=_segment)

    quadmask = subparsers.add_parser("quadmask", help="build a VOID sequence directory")
    quadmask.add_argument("--input-video", type=Path, required=True)
    quadmask.add_argument("--primary-mask", type=Path, required=True)
    quadmask.add_argument("--affected-mask", type=Path, required=True)
    quadmask.add_argument("--background-prompt", required=True)
    quadmask.add_argument("--sequence-dir", type=Path, required=True)
    quadmask.set_defaults(handler=_quadmask)

    insert_quadmask = subparsers.add_parser(
        "insert-quadmask", help="build a VOID sequence directory for the insertion lane"
    )
    insert_quadmask.add_argument("--input-video", type=Path, required=True)
    insert_quadmask.add_argument("--inserted-mask", type=Path, required=True)
    insert_quadmask.add_argument("--cleanup-mask", type=Path, required=True)
    insert_quadmask.add_argument("--background-prompt", required=True)
    insert_quadmask.add_argument("--sequence-dir", type=Path, required=True)
    insert_quadmask.add_argument(
        "--cleanup-value",
        type=int,
        choices=(0, 127),
        default=127,
        help="quadmask value of the cleanup ring: 127 affected or 0 remove",
    )
    insert_quadmask.set_defaults(handler=_insert_quadmask)

    ensemble = subparsers.add_parser(
        "ensemble", help="fuse multi-model mask tensors into one mask video"
    )
    ensemble.add_argument("--config", type=Path, required=True)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--report", type=Path, required=True)
    ensemble.set_defaults(handler=_ensemble)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="estimate label-free expert confidence weights with bounded Bernoulli EM",
    )
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--config-output", type=Path, required=True)
    calibrate.add_argument("--report", type=Path, required=True)
    calibrate.add_argument("--positive-prior-alpha", type=float, default=1.0)
    calibrate.add_argument("--positive-prior-beta", type=float, default=1.0)
    calibrate.add_argument("--sensitivity-alpha", type=float, default=1.0)
    calibrate.add_argument("--sensitivity-beta", type=float, default=1.0)
    calibrate.add_argument("--false-positive-rate-alpha", type=float, default=1.0)
    calibrate.add_argument("--false-positive-rate-beta", type=float, default=1.0)
    calibrate.add_argument(
        "--latent-output",
        type=Path,
        help="optional .npy path for the estimated latent mask probability field",
    )
    calibrate.add_argument("--tolerance", type=float, default=1e-6)
    calibrate.add_argument("--max-iterations", type=int, default=100)
    calibrate.add_argument("--max-samples", type=int, default=200_000)
    calibrate.add_argument("--max-batch-pixels", type=int, default=8_000_000)
    calibrate.add_argument("--seed", type=int, default=0)
    calibrate.set_defaults(handler=_calibrate)

    void = subparsers.add_parser("void", help="execute Netflix VOID passes")
    void.add_argument("--config", type=Path, required=True)
    void.add_argument("--dry-run", action="store_true")
    void.set_defaults(handler=_void)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"samurai: {error}", file=sys.stderr)
        return 2
