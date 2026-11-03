from __future__ import annotations

import json

from samurai.void import load_config, pass1_command, pass2_command, validate


def make_config(tmp_path):
    root = tmp_path / "void"
    sequence = tmp_path / "data" / "sample"
    output = tmp_path / "output"
    checkpoint = tmp_path / "void.safetensors"
    for path in (
        root / "inference/cogvideox_fun/predict_v2v.py",
        root / "config/quadmask_cogvideox.py",
        tmp_path / "base",
        checkpoint,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    sequence.mkdir(parents=True, exist_ok=True)
    (sequence / "input_video.mp4").touch()
    (sequence / "quadmask_0.mp4").touch()
    (sequence / "prompt.json").write_text(json.dumps({"bg": "empty bench"}))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "void_root": str(root),
                "data_rootdir": str(tmp_path / "data"),
                "output_dir": str(output),
                "base_model": str(tmp_path / "base"),
                "pass1_checkpoint": str(checkpoint),
                "pass2_checkpoint": str(tmp_path / "pass2.safetensors"),
                "sequences": ["sample"],
            }
        )
    )
    (tmp_path / "pass2.safetensors").touch()
    return load_config(config_path)


def test_validate_and_commands(tmp_path):
    config = make_config(tmp_path)
    assert validate(config) == []
    first = pass1_command(config)
    second = pass2_command(config)
    assert "--config.data.data_rootdir=" + str(config.data_rootdir) in first
    assert "--video_names" in second and "sample" in second
    assert "--model_checkpoint" in second


def test_validate_reports_missing_sequence_inputs(tmp_path):
    config = make_config(tmp_path)
    (config.data_rootdir / "sample/quadmask_0.mp4").unlink()
    errors = validate(config)
    assert any("missing quadmask_0.mp4" in error for error in errors)
