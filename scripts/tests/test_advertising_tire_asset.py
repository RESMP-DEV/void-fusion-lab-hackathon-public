from __future__ import annotations

import json
import subprocess
from pathlib import Path

import advertising_tire_asset as asset
import numpy as np
import pytest
from advertising_tire_asset import (
    SCHEMA,
    TireFit,
    _font_advance,
    build_inserted_cleanup_masks,
    composite_tile,
    fit_ellipse,
    main,
    measure_visibility,
    render_tire_tile,
    smooth_fit_series,
)
from PIL import Image, ImageFont

from samurai.media import decode_gray_frames, probe_video, write_gray_video

FRAME_COUNT = 12
FRAME_HEIGHT = 120
FRAME_WIDTH = 160
FPS = 12.0


def ellipse_mask(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    """Return the ROI of one synthetic tire ellipse."""
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    normalized = ((xs + 0.5 - center_x) / radius_x) ** 2 + ((ys + 0.5 - center_y) / radius_y) ** 2
    return normalized <= 1.0


def scene_frame(index: int) -> np.ndarray:
    """Return one synthetic road frame with a dark tire on it."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(40, 150, FRAME_WIDTH, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(60, 170, FRAME_WIDTH, dtype=np.uint8)[None, :]
    frame[:, :, 2] = 70
    center_x = 40.0 + 6.0 * index
    center_y = 60.0
    frame[ellipse_mask(center_x, center_y, 26.0, 20.0)] = (18, 18, 22)
    return frame


def plate_frame(index: int) -> np.ndarray:
    """Return the clean plate: the same scene without the tire."""
    frame = scene_frame(index)
    center_x = 40.0 + 6.0 * index
    frame[ellipse_mask(center_x, 60.0, 26.0, 20.0)] = (90, 92, 96)
    return frame


def write_color_video(path: Path, frames: list[np.ndarray]) -> None:
    """Encode RGB frames into one H.264 video."""
    directory = path.parent / f"{path.stem}-frames"
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame, mode="RGB").save(directory / f"frame-{index:06d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(directory / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-y",
            str(path),
        ],
        check=True,
    )


def build_inputs(directory: Path, *, bright: bool = False) -> dict[str, Path]:
    """Write the synthetic video, primary mask, and clean plate."""
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "scene.mp4"
    plate = directory / "plate.mp4"
    mask = directory / "primary-mask.mp4"
    write_color_video(video, [scene_frame(index) for index in range(FRAME_COUNT)])
    write_color_video(plate, [plate_frame(index) for index in range(FRAME_COUNT)])
    masks = []
    for index in range(FRAME_COUNT):
        roi = ellipse_mask(40.0 + 6.0 * index, 60.0, 26.0, 20.0)
        masks.append(np.where(roi, 255 if bright else 0, 0 if bright else 255).astype(np.uint8))
    write_gray_video(masks, mask, fps=FPS)
    return {"video": video, "plate": plate, "mask": mask}


def run_cli(inputs: dict[str, Path], output: Path, *, extra: list[str] | None = None) -> int:
    """Run the lane CLI over the synthetic inputs."""
    return main(
        [
            "--input",
            str(inputs["video"]),
            "--primary-mask",
            str(inputs["mask"]),
            "--plate",
            str(inputs["plate"]),
            "--output",
            str(output / "advertised.mp4"),
            "--inserted-mask",
            str(output / "inserted-mask.mp4"),
            "--cleanup-mask",
            str(output / "cleanup-mask.mp4"),
            "--contact-sheet",
            str(output / "contact-sheet.png"),
            "--report",
            str(output / "report.json"),
            *(extra or []),
        ]
    )


def test_fit_ellipse_recovers_center_radii_and_orientation():
    mask = ellipse_mask(40.0, 60.0, 26.0, 20.0)
    fit = fit_ellipse(mask)

    assert fit is not None
    assert fit.center_x == pytest.approx(40.0, abs=0.5)
    assert fit.center_y == pytest.approx(60.0, abs=0.5)
    assert fit.radius_x == pytest.approx(26.0, rel=0.08)
    assert fit.radius_y == pytest.approx(20.0, rel=0.08)
    assert fit.area == int(mask.sum())


def test_fit_ellipse_returns_none_for_an_empty_region():
    assert fit_ellipse(np.zeros((10, 10), dtype=bool)) is None


def test_fit_ellipse_ignores_a_disconnected_stray_region():
    mask = ellipse_mask(80.0, 60.0, 26.0, 20.0)
    mask[10:25, 10:25] = True

    fit = fit_ellipse(mask)

    assert fit is not None
    assert fit.center_x == pytest.approx(80.0, abs=1.0)
    assert fit.center_y == pytest.approx(60.0, abs=1.0)


def test_smoothing_holds_a_missing_frame_and_keeps_the_series():
    first = fit_ellipse(ellipse_mask(40.0, 60.0, 26.0, 20.0))
    second = fit_ellipse(ellipse_mask(50.0, 60.0, 26.0, 20.0))
    assert first is not None
    assert second is not None

    smoothed = smooth_fit_series([first, None, second], window=3)

    assert len(smoothed) == 3
    assert smoothed[1].held is True
    assert smoothed[0].held is False
    expected = (first.center_x + second.center_x) / 2.0
    assert smoothed[1].center_x == pytest.approx(expected, abs=2.0)
    assert smoothed[0].center_x == pytest.approx(first.center_x, abs=1.5)


def test_gap_frames_hold_the_nearest_fitted_pose():
    before = TireFit(
        center_x=0.0, center_y=0.0, radius_x=10.0, radius_y=8.0, angle_degrees=0.0, area=100
    )
    after = TireFit(
        center_x=30.0, center_y=0.0, radius_x=10.0, radius_y=8.0, angle_degrees=0.0, area=100
    )

    smoothed = smooth_fit_series([before, None, None, after], window=1)

    assert smoothed[1].center_x == pytest.approx(before.center_x)
    assert smoothed[1].held is True
    assert smoothed[2].center_x == pytest.approx(after.center_x)
    assert smoothed[2].held is True


def test_orientation_smoothing_handles_the_angle_wrap():
    positive = TireFit(
        center_x=0.0, center_y=0.0, radius_x=10.0, radius_y=8.0, angle_degrees=89.0, area=100
    )
    negative = TireFit(
        center_x=0.0, center_y=0.0, radius_x=10.0, radius_y=8.0, angle_degrees=-89.0, area=100
    )

    smoothed = smooth_fit_series([positive, negative], window=2)

    assert smoothed[0].angle_degrees == pytest.approx(90.0, abs=1.0)
    assert smoothed[1].angle_degrees == pytest.approx(90.0, abs=1.0)


def test_font_advance_supports_pillow_bitmap_fallback():
    font = ImageFont.load_default()

    advance = _font_advance(font, "M")

    assert advance > 0.0


def test_run_tool_preserves_the_first_ffmpeg_diagnostic(monkeypatch):
    completed = subprocess.CompletedProcess(
        ["ffmpeg"], returncode=2, stderr="primary failure\nsecondary detail", stdout=""
    )
    monkeypatch.setattr(asset.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(asset.AdvertisingTireAssetError, match="primary failure"):
        asset.run_tool(["ffmpeg"], context="composite video encode")


def test_render_tile_alpha_covers_the_fitted_ellipse():
    fit = TireFit(
        center_x=80.0, center_y=60.0, radius_x=30.0, radius_y=22.0, angle_degrees=0.0, area=1
    )
    tile = render_tire_tile(fit, brand_text="MICHELIN")

    rgb = tile.rgb
    alpha = tile.alpha
    origin = tile.origin
    assert rgb.shape[2] == 3
    assert alpha.shape == rgb.shape[:2]
    assert alpha.max() == pytest.approx(1.0)
    assert alpha.min() == pytest.approx(0.0)
    assert tile.lettering is True
    center_x = alpha.shape[1] // 2
    center_y = alpha.shape[0] // 2
    assert alpha[center_y, center_x] == pytest.approx(1.0)
    assert alpha[0, 0] == pytest.approx(0.0)
    assert origin[0] + center_x == pytest.approx(80.0, abs=2.0)
    assert origin[1] + center_y == pytest.approx(60.0, abs=2.0)


def test_inserted_and_cleanup_masks_are_roi_black_and_disjoint():
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[10:30, 10:30] = 1.0

    inserted, cleanup = build_inserted_cleanup_masks(alpha, cleanup_radius=3)

    assert set(np.unique(inserted).tolist()) <= {0, 255}
    assert set(np.unique(cleanup).tolist()) <= {0, 255}
    assert inserted[20, 20] == 0
    assert cleanup[20, 20] == 255
    assert inserted.shape == (46, 46)
    assert cleanup.shape == (46, 46)
    assert inserted[23, 23] == 0
    assert cleanup[23, 23] == 255
    assert cleanup[9, 23] == 255
    assert cleanup[10, 23] == 0
    assert cleanup[35, 23] == 0
    assert cleanup[36, 23] == 255
    assert cleanup[0, 0] == 255
    assert np.count_nonzero(cleanup == 0) > 0


def test_composite_tile_clips_a_negative_origin_on_both_sides():
    frame = np.zeros((5, 6, 3), dtype=np.uint8)
    tile_rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
    tile_alpha = np.ones((4, 4), dtype=np.float32)

    composite = composite_tile(frame, tile_rgb, tile_alpha, (-1, -1))

    assert composite.shape == frame.shape
    expected = np.zeros_like(frame)
    expected[0:3, 0:3] = 255
    assert np.array_equal(composite, expected)


def test_visibility_measures_the_changed_footprint_and_brand_band():
    base = np.full((40, 40, 3), 120, dtype=np.uint8)
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[10:30, 10:30] = 1.0
    composite = base.copy()
    composite[10:30, 10:30] = (247, 200, 0)

    metrics = measure_visibility(base, composite, alpha, (0, 0))

    assert metrics["footprint_pixels"] == 400
    assert metrics["changed_fraction"] == pytest.approx(1.0)
    assert metrics["brand_pixels"] == 400


def read_color_frames(path: Path) -> list[np.ndarray]:
    """Decode a color video into RGB frames."""
    directory = path.parent / f"{path.stem}-decoded"
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            str(directory / "frame-%06d.png"),
            "-y",
        ],
        check=True,
    )
    frames = []
    for file in sorted(directory.glob("frame-*.png")):
        with Image.open(file) as image:
            frames.append(np.asarray(image.convert("RGB")).copy())
    return frames


def brand_band_pixels(frame: np.ndarray) -> int:
    """Count the yellow advertising band pixels in one frame."""
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    yellow = (red >= 120) & (green >= 90) & ((red - blue) >= 45)
    return int(np.count_nonzero(yellow))


def test_cli_renders_a_visible_tire_and_writes_every_artifact(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    assert run_cli(inputs, output) == 0

    info = probe_video(output / "advertised.mp4")
    assert (info.width, info.height, info.frames) == (FRAME_WIDTH, FRAME_HEIGHT, FRAME_COUNT)

    inserted = decode_gray_frames(output / "inserted-mask.mp4", tmp_path / "inserted")
    cleanup = decode_gray_frames(output / "cleanup-mask.mp4", tmp_path / "cleanup")
    assert len(inserted) == FRAME_COUNT
    assert len(cleanup) == FRAME_COUNT
    for frame in inserted:
        assert set(np.unique(frame).tolist()) <= {0, 255}
        assert np.count_nonzero(frame == 0) > 0
    for frame in cleanup:
        assert set(np.unique(frame).tolist()) <= {0, 255}
        assert np.count_nonzero(frame == 0) > 0
    overlap = np.count_nonzero((inserted[0] == 0) & (cleanup[0] == 0))
    assert overlap == 0

    frames = read_color_frames(output / "advertised.mp4")
    assert len(frames) == FRAME_COUNT
    band_counts = [brand_band_pixels(frame) for frame in frames]
    assert min(band_counts) > 0

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA
    assert report["summary"]["frames"] == FRAME_COUNT
    assert report["summary"]["mean_changed_fraction"] > 0.5
    assert report["summary"]["frames_with_brand_band"] == FRAME_COUNT
    assert report["summary"]["frames_with_lettering"] == FRAME_COUNT
    assert report["summary"]["visibility_gate"]["passed"] is True
    assert report["settings"]["composite_base"] == "plate"
    assert len(report["fit"]) == FRAME_COUNT
    centers = [record["center_x"] for record in report["fit"]]
    steps = [after - before for before, after in zip(centers, centers[1:], strict=False)]
    assert all(step == pytest.approx(6.0, abs=0.5) for step in steps[1:-1])
    assert steps[0] == pytest.approx(3.0, abs=0.5)
    assert steps[-1] == pytest.approx(3.0, abs=0.5)
    assert centers[0] == pytest.approx(40.0, abs=3.0)
    assert centers[-1] == pytest.approx(40.0 + 6.0 * (FRAME_COUNT - 1), abs=4.0)
    assert report["non_claims"]

    with Image.open(output / "contact-sheet.png") as sheet:
        assert sheet.width == 4 * 320
        assert sheet.height == 4 * (22 + 180)


def test_cli_accepts_a_bright_roi_mask(tmp_path):
    inputs = build_inputs(tmp_path / "inputs", bright=True)
    output = tmp_path / "out"

    assert run_cli(inputs, output, extra=["--mask-invert"]) == 0

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["inputs"]["primary_mask"]["roi_convention"] == "bright"
    assert report["summary"]["frames_with_brand_band"] == FRAME_COUNT


def test_cli_composites_over_the_source_without_a_plate(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    exit_code = main(
        [
            "--input",
            str(inputs["video"]),
            "--primary-mask",
            str(inputs["mask"]),
            "--output",
            str(output / "advertised.mp4"),
            "--inserted-mask",
            str(output / "inserted-mask.mp4"),
            "--cleanup-mask",
            str(output / "cleanup-mask.mp4"),
            "--contact-sheet",
            str(output / "contact-sheet.png"),
            "--report",
            str(output / "report.json"),
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["settings"]["composite_base"] == "source"
    assert report["inputs"]["plate"] is None


def test_cli_fails_closed_when_the_mask_geometry_differs(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    short_mask = tmp_path / "short-mask.mp4"
    write_gray_video(
        [np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8) for _ in range(3)],
        short_mask,
        fps=FPS,
    )
    inputs = {**inputs, "mask": short_mask}
    output = tmp_path / "out"

    assert run_cli(inputs, output) == 2
    assert not (output / "advertised.mp4").exists()


def test_cli_fails_closed_when_mask_cadence_differs(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    fast_video = tmp_path / "fast-scene.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(inputs["video"]),
            "-vf",
            "setpts=N/(24*TB),fps=24",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-y",
            str(fast_video),
        ],
        check=True,
    )
    inputs = {**inputs, "video": fast_video}
    output = tmp_path / "out"

    assert run_cli(inputs, output) == 2
    assert not (output / "advertised.mp4").exists()


def test_cli_fails_closed_when_no_frame_holds_a_tire(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    empty_mask = tmp_path / "empty-mask.mp4"
    write_gray_video(
        [np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8) for _ in range(FRAME_COUNT)],
        empty_mask,
        fps=FPS,
    )
    inputs = {**inputs, "mask": empty_mask}
    output = tmp_path / "out"

    assert run_cli(inputs, output) == 2
    assert not (output / "advertised.mp4").exists()


def test_cli_fails_closed_on_a_missing_input(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    inputs = {**inputs, "video": tmp_path / "missing.mp4"}
    output = tmp_path / "out"

    assert run_cli(inputs, output) == 2


def test_cli_rejects_an_invalid_cleanup_radius(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    assert run_cli(inputs, output, extra=["--cleanup-radius", "0"]) == 2
    assert not (output / "advertised.mp4").exists()


def test_cli_rejects_an_invalid_smoothing_window(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    assert run_cli(inputs, output, extra=["--smoothing-window", "0"]) == 2
    assert not (output / "advertised.mp4").exists()


def test_cli_rejects_identical_mask_output_paths(tmp_path):
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    exit_code = main(
        [
            "--input",
            str(inputs["video"]),
            "--primary-mask",
            str(inputs["mask"]),
            "--plate",
            str(inputs["plate"]),
            "--output",
            str(output / "advertised.mp4"),
            "--inserted-mask",
            str(output / "same-mask.mp4"),
            "--cleanup-mask",
            str(output / "same-mask.mp4"),
            "--contact-sheet",
            str(output / "contact-sheet.png"),
        ]
    )

    assert exit_code == 2
    assert not (output / "advertised.mp4").exists()


def test_cli_returns_input_error_for_filesystem_failures(tmp_path, monkeypatch):
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"
    arguments = [
        "--input",
        str(inputs["video"]),
        "--primary-mask",
        str(inputs["mask"]),
        "--output",
        str(output / "advertised.mp4"),
        "--inserted-mask",
        str(output / "inserted-mask.mp4"),
        "--cleanup-mask",
        str(output / "cleanup-mask.mp4"),
        "--contact-sheet",
        str(output / "contact-sheet.png"),
    ]

    def unavailable(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise FileNotFoundError("output filesystem unavailable")

    monkeypatch.setattr(asset, "run_lane", unavailable)

    assert main(arguments) == 2
