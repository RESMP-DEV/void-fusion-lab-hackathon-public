from __future__ import annotations

import json
from argparse import ArgumentTypeError

import numpy as np
import pytest

from samurai.cli import _box, _point, main, parser
from samurai.media import write_gray_video


def test_point_and_box_parsers():
    assert _point("12.5,4") == (12.5, 4.0)
    assert _box("1,2,3,5") == (1.0, 2.0, 3.0, 5.0)
    with pytest.raises(ArgumentTypeError):
        _box("3,2,1,5")


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0
    assert "samurai" in capsys.readouterr().out


def test_segment_target_point_flag_is_removed():
    # R4 conformance: --target-point is deleted, with no replacement flag.
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "segment",
                "video.mp4",
                "--text",
                "person",
                "--target-point",
                "12.5,4",
                "--output",
                "mask.mp4",
            ]
        )


def test_segment_collective_flag_parses():
    args = parser().parse_args(
        ["segment", "video.mp4", "--text", "person", "--collective", "--output", "mask.mp4"]
    )
    assert args.collective is True


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        # Point prompt with no --object-id (and no --box).
        (
            ["--point", "1,1", "--point-label", "1"],
            "point prompts require object_id",
        ),
        (["--text", "person"], "no prompt lock"),
        (
            ["--text", "person", "--point", "1,1", "--point-label", "1", "--object-id", "7"],
            "point prompts cannot be combined with text or box prompts",
        ),
        (
            [
                "--point",
                "1,1",
                "--point-label",
                "1",
                "--box",
                "0,0,2,2",
                "--object-id",
                "7",
            ],
            "point prompts cannot be combined with text or box prompts",
        ),
        (["--text", "person", "--object-id", "7"], "object_id is valid only with point prompts"),
        (
            [
                "--point",
                "1,1",
                "--point-label",
                "1",
                "--object-id",
                "7",
                "--collective",
            ],
            "collective cannot be combined with point or box prompts",
        ),
    ],
)
def test_segment_invalid_combos_fail_before_predictor_build(extra, message, capsys):
    # Validation runs before the torch/sam3 imports and the predictor
    # build, so the error is the validation error (exit 2 via main()),
    # never an import or CUDA error.
    code = main(["segment", "video.mp4", *extra, "--output", "mask.mp4"])
    assert code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "CUDA" not in err
    assert "not installed" not in err


def test_cli_doctor_reports_model_lanes_without_secret_values(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "aimlapi_key_configured" in report["checks"]
    assert "AIMLAPI_KEY" not in capsys.readouterr().out


def test_cli_ensemble_writes_mask_and_report(tmp_path, capsys):
    video = tmp_path / "input.mp4"
    write_gray_video(
        [np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)],
        video,
        fps=12,
    )
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    np.save(second, np.array([np.full((4, 6), -4.0), np.full((4, 6), 4.0)]))
    config = tmp_path / "ensemble.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "first", "path": str(first)},
                    {"name": "second", "path": str(second)},
                ],
                "fusion": {"mode": "top_k", "top_k_ratio": 0.25, "batch_size": 1},
            }
        )
    )
    output = tmp_path / "mask.mp4"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "ensemble",
                "--config",
                str(config),
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert json.loads(report.read_text())["frames"] == 2
    assert "frames" in json.loads(capsys.readouterr().out)
