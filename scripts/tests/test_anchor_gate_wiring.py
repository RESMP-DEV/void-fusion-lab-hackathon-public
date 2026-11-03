"""run_ensemble wiring for the anchor gate (CODE-MAP section 2.4).

A seeded tensor expert declares its frame-0 source mask through the
per-expert ``anchor_source`` config field. The gate runs inside
run_ensemble after the sources are built and before any fusion or
output: a trip raises EnsembleError and writes no output video and no
run report; a pass records the AnchorGateReport in the result dict
under ``anchor_gate`` (and therefore in the run report JSON the CLI
writes); an expert with no anchor_source keeps the old behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from samurai import media
from samurai.ensemble import EnsembleError, load_config, load_frame0_mask, run_ensemble
from samurai.media import decode_first_gray_frame, write_gray_video

HEIGHT, WIDTH = 4, 6
SOURCE_PIXELS = 20


def _source_mask() -> np.ndarray:
    source = np.zeros((HEIGHT, WIDTH), dtype=bool)
    source.flat[:SOURCE_PIXELS] = True
    return source


def _seeded_tensor(missed: int, extra: int = 0) -> np.ndarray:
    """Two-frame logits tensor whose frame 0 misses ``missed`` source px.

    The seed values pin the contract transform: +0.25 is under the
    raw 0.5 cut, but sigmoid(0.25) = 0.56 holds the pixel after
    sigmoid(logit/temperature); -0.25 drops under both rules. A gate
    that thresholds the raw logits fails the pass tests below.
    """
    frame0 = np.full((HEIGHT, WIDTH), -0.25)
    frame0.flat[: SOURCE_PIXELS - missed] = 0.25
    if extra:
        frame0.flat[SOURCE_PIXELS : SOURCE_PIXELS + extra] = 0.25
    frame1 = np.full((HEIGHT, WIDTH), 0.25)
    return np.stack([frame0, frame1])


def _write_run_inputs(tmp_path, seeded: np.ndarray, anchor_name: str | None):
    video = tmp_path / "input.mp4"
    write_gray_video(
        [np.full((HEIGHT, WIDTH), value, dtype=np.uint8) for value in (0, 255)],
        video,
        fps=12,
    )
    first = tmp_path / "seeded.npy"
    second = tmp_path / "other.npy"
    np.save(first, seeded)
    np.save(second, np.stack([np.full((HEIGHT, WIDTH), -4.0), np.full((HEIGHT, WIDTH), 4.0)]))
    seeded_expert: dict = {"name": "seeded", "path": str(first), "weight": 1.0}
    if anchor_name is not None:
        # A bare filename: load_config resolves it relative to the
        # config file directory (tmp_path here).
        seeded_expert["anchor_source"] = anchor_name
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    seeded_expert,
                    {"name": "other", "path": str(second), "weight": 1.0},
                ],
                "fusion": {
                    "mode": "top_k",
                    "top_k_ratio": 0.25,
                    "dynamic_weights": "static",
                    "batch_size": 1,
                },
            }
        )
    )
    return config_path


def test_anchor_gate_trip_writes_no_output_or_report(tmp_path):
    np.save(tmp_path / "anchor.npy", _source_mask())
    # Frame 0 holds 10 of 20 source px: 10 missed px = 50% > 5% limit.
    config_path = _write_run_inputs(tmp_path, _seeded_tensor(missed=10), "anchor.npy")
    output = tmp_path / "mask.mp4"
    report = tmp_path / "report.json"
    with pytest.raises(EnsembleError, match="anchor gate refused"):
        run_ensemble(load_config(config_path), output, report)
    assert not output.exists()
    assert not report.exists()


def test_anchor_gate_pass_records_exact_report(tmp_path):
    np.save(tmp_path / "anchor.npy", _source_mask())
    # Frame 0 misses 1 of 20 source px (exactly 5.00%, at the limit)
    # and holds 1 extra px outside the source.
    config_path = _write_run_inputs(tmp_path, _seeded_tensor(missed=1, extra=1), "anchor.npy")
    config = load_config(config_path)
    assert config.experts[0].anchor_source_path == tmp_path / "anchor.npy"
    assert config.experts[1].anchor_source_path is None
    result = run_ensemble(config, tmp_path / "mask.mp4", tmp_path / "report.json")
    expected = [
        {
            "expert_name": "seeded",
            "source_pixels": 20,
            "expert_pixels": 20,
            "missed_pixels": 1,
            "missed_fraction": pytest.approx(0.05),
            "extra_pixels": 1,
            "max_missed_fraction": 0.05,
        }
    ]
    assert result["anchor_gate"] == expected
    saved_report = json.loads((tmp_path / "report.json").read_text())
    assert saved_report["anchor_gate"] == result["anchor_gate"]
    assert (tmp_path / "mask.mp4").is_file()


def test_expert_without_anchor_source_is_unchanged(tmp_path):
    config_path = _write_run_inputs(tmp_path, _seeded_tensor(missed=10), None)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["anchor_gate"] == []
    assert result["frames"] == 2
    assert (tmp_path / "mask.mp4").is_file()
    saved_report = json.loads((tmp_path / "report.json").read_text())
    assert saved_report["anchor_gate"] == []


def test_anchor_gate_pass_with_png_source(tmp_path):
    image = np.where(_source_mask(), 255, 0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(tmp_path / "anchor.png")
    config_path = _write_run_inputs(tmp_path, _seeded_tensor(missed=0), "anchor.png")
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["anchor_gate"][0]["source_pixels"] == 20
    assert result["anchor_gate"][0]["missed_pixels"] == 0
    assert result["anchor_gate"][0]["missed_fraction"] == 0.0


def test_load_frame0_mask_formats_and_failures(tmp_path):
    source = _source_mask()
    # 3D .npy takes frame 0; a different frame 1 must not leak in.
    tensor_path = tmp_path / "anchor3d.npy"
    np.save(tensor_path, np.stack([source, np.zeros_like(source)]))
    assert np.array_equal(load_frame0_mask(tensor_path), source)
    # 2D .npy is used as-is.
    flat_path = tmp_path / "anchor2d.npy"
    np.save(flat_path, source)
    assert np.array_equal(load_frame0_mask(flat_path), source)
    # A mask video contributes frame 0, binarized over 127.
    video_path = tmp_path / "anchor.mp4"
    write_gray_video(
        [np.where(source, 255, 0).astype(np.uint8), np.zeros((HEIGHT, WIDTH), dtype=np.uint8)],
        video_path,
        fps=12,
    )
    assert np.array_equal(load_frame0_mask(video_path), source)
    # Unsupported suffix and missing file both fail closed.
    bad_path = tmp_path / "anchor.txt"
    bad_path.write_text("not a mask")
    with pytest.raises(EnsembleError, match="unsupported anchor source"):
        load_frame0_mask(bad_path)
    with pytest.raises(EnsembleError, match="not found"):
        load_frame0_mask(tmp_path / "missing.npy")
    # Unreadable .npy content fails closed.
    corrupt_path = tmp_path / "corrupt.npy"
    corrupt_path.write_bytes(b"not an npy file")
    with pytest.raises(EnsembleError, match="cannot read anchor source"):
        load_frame0_mask(corrupt_path)
    # An undecodable video anchor fails closed too: the single-frame
    # decode cannot produce a frame, so the gate refuses the anchor.
    broken_video = tmp_path / "broken.mp4"
    broken_video.write_bytes(b"not a video")
    with pytest.raises(EnsembleError, match="cannot read anchor source"):
        load_frame0_mask(broken_video)


def test_video_anchor_source_decodes_frame_zero_only(tmp_path):
    """Round-2 finding 3: a video anchor source contributes frame
    zero, and the decode materializes that one frame only. The whole
    clip is never extracted and held in memory for a single-frame
    read."""
    source = _source_mask()
    video_path = tmp_path / "anchor.mp4"
    write_gray_video(
        [
            np.where(source, 255, 0).astype(np.uint8),
            np.zeros((HEIGHT, WIDTH), dtype=np.uint8),
            np.full((HEIGHT, WIDTH), 255, dtype=np.uint8),
        ],
        video_path,
        fps=12,
    )
    destination = tmp_path / "decoded"
    frame = decode_first_gray_frame(video_path, destination)
    assert np.array_equal(frame > 127, source)
    # Exactly one image file was written for a three-frame clip.
    assert len(list(destination.glob("*.png"))) == 1


def test_load_frame0_mask_reads_a_video_anchor_with_one_frame_decode(tmp_path, monkeypatch):
    """Round-2 finding 3: the anchor read of a video issues exactly
    one ffmpeg decode, and that decode is bounded to one frame, so
    the run never extracts every frame of the clip."""
    source = _source_mask()
    video_path = tmp_path / "anchor.mp4"
    write_gray_video(
        [
            np.where(source, 255, 0).astype(np.uint8),
            np.zeros((HEIGHT, WIDTH), dtype=np.uint8),
            np.full((HEIGHT, WIDTH), 255, dtype=np.uint8),
        ],
        video_path,
        fps=12,
    )
    commands: list[list[str]] = []
    original_run = media._run

    def recording_run(command: list[str], *, cwd=None):
        commands.append(list(command))
        return original_run(command, cwd=cwd)

    monkeypatch.setattr(media, "_run", recording_run)
    mask = load_frame0_mask(video_path)
    assert np.array_equal(mask, source)
    assert len(commands) == 1
    assert "-frames:v" in commands[0]
    assert commands[0][commands[0].index("-frames:v") + 1] == "1"
