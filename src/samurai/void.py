"""Validation and bounded execution of Netflix's VOID reference commands."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class VoidError(RuntimeError):
    """VOID configuration or execution state is invalid."""


@dataclass(frozen=True)
class VoidConfig:
    void_root: Path
    data_rootdir: Path
    output_dir: Path
    base_model: Path
    pass1_checkpoint: Path
    pass2_checkpoint: Path | None
    sequences: tuple[str, ...]
    sample_size: tuple[int, int] = (384, 672)
    max_video_length: int = 197
    temporal_window_size: int = 85
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    pass2_guidance_scale: float = 6.0
    seed: int = 42
    warped_noise_cache_dir: Path | None = None
    gpu_memory_mode: str = "model_cpu_offload_and_qfloat8"


def load_config(path: Path) -> VoidConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoidError(f"cannot load config {path}: {error}") from error
    required = {
        "void_root",
        "data_rootdir",
        "output_dir",
        "base_model",
        "pass1_checkpoint",
        "sequences",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise VoidError(f"missing VOID config fields: {', '.join(missing)}")
    if not raw["sequences"]:
        raise VoidError("at least one sequence is required")
    sample_size = raw.get("sample_size", [384, 672])
    if len(sample_size) != 2:
        raise VoidError("sample_size must contain height and width")
    return VoidConfig(
        void_root=Path(raw["void_root"]).expanduser(),
        data_rootdir=Path(raw["data_rootdir"]).expanduser(),
        output_dir=Path(raw["output_dir"]).expanduser(),
        base_model=Path(raw["base_model"]).expanduser(),
        pass1_checkpoint=Path(raw["pass1_checkpoint"]).expanduser(),
        pass2_checkpoint=(
            Path(raw["pass2_checkpoint"]).expanduser() if raw.get("pass2_checkpoint") else None
        ),
        sequences=tuple(raw["sequences"]),
        sample_size=(int(sample_size[0]), int(sample_size[1])),
        max_video_length=int(raw.get("max_video_length", 197)),
        temporal_window_size=int(raw.get("temporal_window_size", 85)),
        num_inference_steps=int(raw.get("num_inference_steps", 50)),
        guidance_scale=float(raw.get("guidance_scale", 1.0)),
        pass2_guidance_scale=float(raw.get("pass2_guidance_scale", 6.0)),
        seed=int(raw.get("seed", 42)),
        warped_noise_cache_dir=(
            Path(raw["warped_noise_cache_dir"]).expanduser()
            if raw.get("warped_noise_cache_dir")
            else None
        ),
        gpu_memory_mode=raw.get("gpu_memory_mode", "model_cpu_offload_and_qfloat8"),
    )


def validate(config: VoidConfig) -> list[str]:
    errors: list[str] = []
    required_paths = {
        "VOID inference script": config.void_root / "inference/cogvideox_fun/predict_v2v.py",
        "VOID quadmask config": config.void_root / "config/quadmask_cogvideox.py",
        "data root": config.data_rootdir,
        "base model": config.base_model,
        "Pass 1 checkpoint": config.pass1_checkpoint,
    }
    if config.pass2_checkpoint is not None:
        required_paths["Pass 2 checkpoint"] = config.pass2_checkpoint
    for label, path in required_paths.items():
        if not path.exists():
            errors.append(f"{label} does not exist: {path}")
    for name in config.sequences:
        sequence_dir = config.data_rootdir / name
        for filename in ("input_video.mp4", "quadmask_0.mp4", "prompt.json"):
            if not (sequence_dir / filename).is_file():
                errors.append(f"sequence {name} is missing {filename}")
        prompt_path = sequence_dir / "prompt.json"
        if prompt_path.is_file():
            try:
                prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
                if not prompt.get("bg", "").strip():
                    errors.append(f"sequence {name} prompt.json has no nonempty bg")
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"sequence {name} prompt.json is unreadable: {error}")
    return errors


def pass1_command(config: VoidConfig) -> list[str]:
    height, width = config.sample_size
    return [
        "python",
        str(config.void_root / "inference/cogvideox_fun/predict_v2v.py"),
        "--config",
        str(config.void_root / "config/quadmask_cogvideox.py"),
        f"--config.data.data_rootdir={config.data_rootdir}",
        f"--config.experiment.run_seqs={','.join(config.sequences)}",
        f"--config.experiment.save_path={config.output_dir}",
        f"--config.video_model.model_name={config.base_model}",
        f"--config.video_model.transformer_path={config.pass1_checkpoint}",
        f"--config.data.sample_size={height}x{width}",
        f"--config.data.max_video_length={config.max_video_length}",
        f"--config.video_model.temporal_window_size={config.temporal_window_size}",
        f"--config.video_model.num_inference_steps={config.num_inference_steps}",
        f"--config.video_model.guidance_scale={config.guidance_scale}",
        f"--config.system.gpu_memory_mode={config.gpu_memory_mode}",
    ]


def pass2_command(config: VoidConfig) -> list[str]:
    if config.pass2_checkpoint is None:
        raise VoidError("Pass 2 requires pass2_checkpoint")
    height, width = config.sample_size
    cache_dir = config.warped_noise_cache_dir or config.output_dir.parent / "pass1-warped-noise"
    return [
        "python",
        str(config.void_root / "inference/cogvideox_fun/inference_with_pass1_warped_noise.py"),
        "--video_names",
        *config.sequences,
        "--data_rootdir",
        str(config.data_rootdir),
        "--pass1_dir",
        str(config.output_dir),
        "--output_dir",
        str(config.output_dir.parent / f"{config.output_dir.name}-pass2"),
        "--model_name",
        str(config.base_model),
        "--model_checkpoint",
        str(config.pass2_checkpoint),
        "--max_video_length",
        str(config.max_video_length),
        "--temporal_window_size",
        str(config.temporal_window_size),
        "--height",
        str(height),
        "--width",
        str(width),
        "--seed",
        str(config.seed),
        "--guidance_scale",
        str(config.pass2_guidance_scale),
        "--num_inference_steps",
        str(config.num_inference_steps),
        "--warped_noise_cache_dir",
        str(cache_dir),
        "--use_quadmask",
    ]


def _record(command: list[str], status: str) -> dict[str, object]:
    result: dict[str, object] = {"command": command, "status": status, "returncode": 0}
    return result


def execute(config: VoidConfig, *, dry_run: bool = False) -> list[dict[str, object]]:
    errors = validate(config)
    if errors:
        raise VoidError("; ".join(errors))
    if shutil.which("ffmpeg") is None:
        raise VoidError("ffmpeg is required by the VOID reference implementation")
    command = pass1_command(config)
    if dry_run:
        results: list[dict[str, object]] = []
        results.append(_record(command, "dry_run"))
        if config.pass2_checkpoint is not None:
            results.append(_record(pass2_command(config), "dry_run"))
        return results
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise VoidError(f"VOID Pass 1 exited with {completed.returncode}")
    results = [_record(command, "completed")]
    if config.pass2_checkpoint is not None:
        second = pass2_command(config)
        completed = subprocess.run(second, check=False)
        if completed.returncode != 0:
            raise VoidError(f"VOID Pass 2 exited with {completed.returncode}")
        results.append(_record(second, "completed"))
    return results
