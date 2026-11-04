"""Checks for the true-3D advertising tire lane.

The lane is exercised with a synthetic road clip, a ROI-black tire mask, and a
clean plate.  Blender itself is replaced by a fake renderer that draws the
wheel silhouette from the pose JSON the lane writes, so the checks cover the
compiler, the ellipse-to-wheel pose mapping, the composite, and the receipt
without a Blender installation.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import advertising_tire_3d as lane
import numpy as np
import pytest
from PIL import Image, ImageDraw

from samurai.media import decode_gray_frames, probe_video, write_gray_video

FRAME_COUNT = 6
FRAME_HEIGHT = 120
FRAME_WIDTH = 160
FPS = 12.0
RENDER_WIDTH = 672
RENDER_HEIGHT = 384
METERS_PER_PIXEL = lane.METERS_PER_PIXEL
BLENDER_SCRIPT = Path(__file__).resolve().parents[1] / "blender_tire_render.py"
PROFILE = lane.load_wheel_profile(BLENDER_SCRIPT)


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
    frame[ellipse_mask(40.0 + 8.0 * index, 60.0, 26.0, 18.0)] = (18, 18, 22)
    return frame


def plate_frame(index: int) -> np.ndarray:
    """Return the clean plate: the same scene without the tire."""
    frame = scene_frame(index)
    frame[ellipse_mask(40.0 + 8.0 * index, 60.0, 26.0, 18.0)] = (90, 92, 96)
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


def fitted_ellipse(index: int) -> Any:
    """Return the fitted tire ellipse of one frame."""
    mask = ellipse_mask(40.0 + 8.0 * index, 60.0, 26.0, 18.0)
    fit = lane.fit_ellipse(mask)
    assert fit is not None
    return fit


def fitted_ellipse_mask(index: int, *, scale: float = 1.0) -> np.ndarray:
    """Return the fitted tire ellipse of one frame, optionally shrunk."""
    fit = fitted_ellipse(index)
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    normalized = ((xs + 0.5 - fit.center_x) / (fit.radius_x * scale)) ** 2 + (
        (ys + 0.5 - fit.center_y) / (fit.radius_y * scale)
    ) ** 2
    return normalized <= 1.0


def build_inputs(directory: Path) -> dict[str, Path]:
    """Write the synthetic video, ROI-black primary mask, and clean plate."""
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "scene.mp4"
    plate = directory / "plate.mp4"
    mask = directory / "primary-mask.mp4"
    write_color_video(video, [scene_frame(index) for index in range(FRAME_COUNT)])
    write_color_video(plate, [plate_frame(index) for index in range(FRAME_COUNT)])
    masks = [
        np.where(ellipse_mask(40.0 + 8.0 * index, 60.0, 26.0, 18.0), 0, 255).astype(np.uint8)
        for index in range(FRAME_COUNT)
    ]
    write_gray_video(masks, mask, fps=FPS)
    return {"video": video, "plate": plate, "mask": mask}


def fake_blender(
    monkeypatch: pytest.MonkeyPatch,
    *,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
    frames: int = FRAME_COUNT,
) -> list[dict[str, Any]]:
    """Replace the Blender invocation with a renderer driven by the pose JSON.

    The fake draws the wheel silhouette at the pose position, so a composite
    that lands on the tire proves that the pose mapping reached the renderer.
    """
    poses_seen: list[dict[str, Any]] = []

    def run(
        blender: Path,
        script: Path,
        poses: Path,
        output: Path,
        *,
        samples: int,
        device: str,
        blend: Path | None,
        glb: Path | None,
    ) -> None:
        payload = json.loads(poses.read_text(encoding="utf-8"))
        poses_seen.append(payload)
        output.mkdir(parents=True, exist_ok=True)
        for pose in payload["poses"]:
            radius_x = pose["scale"] / METERS_PER_PIXEL
            radius_y = radius_x * (pose["radius_y"] / pose["radius_x"])
            center_x = pose["position"][0] / METERS_PER_PIXEL + width / 2.0
            center_y = height / 2.0 - pose["position"][1] / METERS_PER_PIXEL
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                (
                    center_x - radius_x,
                    center_y - radius_y,
                    center_x + radius_x,
                    center_y + radius_y,
                ),
                fill=(250, 200, 0, 255),
            )
            image.save(output / f"frame-{pose['frame']:06d}.png")
        manifest = {
            "schema": "void-fusion-lab.blender-tire-render.v1",
            "profile": payload["profile"],
            "blender_version": "4.2.0 fake",
            "blender_executable": str(blender),
            "blender_executable_sha256": lane.sha256(blender),
            "script_sha256": lane.sha256(script),
            "pose_sha256": lane.sha256(poses),
            "engine": "CYCLES",
            "device": device,
            "samples": samples,
            "width": width,
            "height": height,
            "frames": frames,
            "counts": {"mesh_objects": 38, "mesh_vertices": 4096, "material_count": 7},
            "blend": {"path": str(blend) if blend else None, "sha256": "0" * 64},
            "glb": {"path": str(glb) if glb else None, "sha256": "0" * 64},
        }
        (output / "render-manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    monkeypatch.setattr(lane, "run_blender", run)
    return poses_seen


def test_compile_target_accepts_a_bounded_tire_description():
    """A plain tire description must compile into the default wheel palette."""
    spec = lane.compile_target("a black tire with a yellow band")
    assert spec.target == "tire"
    assert spec.tread_color == lane.COLOR_WORDS["black"]
    assert spec.band_color == lane.COLOR_WORDS["yellow"]
    assert spec.label is None
    assert spec.tokens[:4] == ("a", "black", "tire", "with")


def test_compile_target_reads_the_band_color_and_the_label():
    """A band color, a wheel synonym, and a label must all compile."""
    spec = lane.compile_target("the front wheel with a red band labeled AXION")
    assert spec.tread_color == lane.COLOR_WORDS["black"]
    assert spec.band_color == lane.COLOR_WORDS["red"]
    assert spec.label == "AXION"
    assert lane.compile_target("a tire branded axion").label == "AXION"
    assert lane.compile_target("a tire labeled A").label == "A"
    assert lane.compile_target("a tire labeled THE").label == "THE"


def test_compile_target_accepts_a_natural_conjoined_label():
    spec = lane.compile_target("a black tire with a yellow band and MICHELIN label")

    assert spec.target == "tire"
    assert spec.label == "MICHELIN"
    assert spec.tread_color == lane.COLOR_WORDS["black"]
    assert spec.band_color == lane.COLOR_WORDS["yellow"]


def test_compile_target_does_not_treat_a_color_label_as_a_brand():
    spec = lane.compile_target("a black tire with a yellow label")

    assert spec.label is None


def test_compile_target_rejects_a_description_without_a_tire():
    """A description that names no tire must fail closed."""
    with pytest.raises(lane.AdvertisingTire3DError, match="unsupported target"):
        lane.compile_target("a red sports car")


def test_compile_target_rejects_an_unsupported_token():
    """An unknown word must fail closed instead of being ignored."""
    with pytest.raises(lane.AdvertisingTire3DError, match="unsupported token"):
        lane.compile_target("a tire with flaming skulls")


def test_compile_target_rejects_an_empty_or_oversized_description():
    """The description must be present and bounded."""
    with pytest.raises(lane.AdvertisingTire3DError, match="empty"):
        lane.compile_target("   ")
    with pytest.raises(lane.AdvertisingTire3DError, match="bound"):
        lane.compile_target("a tire " + "very " * 100 + "black")


def test_compile_target_rejects_two_tread_colors():
    """An ambiguous color specification must fail closed."""
    with pytest.raises(lane.AdvertisingTire3DError, match="two tread colors"):
        lane.compile_target("a red blue tire")


def test_compile_target_rejects_a_label_that_is_not_a_short_word():
    """A label must be a short alphanumeric token."""
    with pytest.raises(lane.AdvertisingTire3DError, match="label"):
        lane.compile_target("a tire labeled")


def test_the_wheel_profile_is_read_from_the_render_script():
    """The pose solver must use the profile the render script actually builds."""
    assert PROFILE.major_radius == pytest.approx(0.78)
    assert PROFILE.minor_radius == pytest.approx(0.22)
    assert PROFILE.tread_half_depth == pytest.approx(0.17)
    assert PROFILE.outer_radius == pytest.approx(1.0)
    record = lane.profile_record(PROFILE)
    assert record == {
        "major_radius": 0.78,
        "minor_radius": 0.22,
        "tread_half_depth": 0.17,
        "outer_radius": 1.0,
    }


def test_the_wheel_profile_is_parsed_without_executing_the_render_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A supplied render script must not gain arbitrary Python execution."""
    script = tmp_path / "profile-only.py"
    script.write_text(
        'print("side effect")\n'
        "TIRE_MAJOR_RADIUS = 0.78\n"
        "TIRE_MINOR_RADIUS = 0.22\n"
        "TREAD_DEPTH = 0.34\n",
        encoding="utf-8",
    )

    profile = lane.load_wheel_profile(script)

    assert profile.outer_radius == pytest.approx(1.0)
    assert capsys.readouterr().out == ""


def test_the_cleanup_radius_must_fit_inside_the_render_canvas(tmp_path: Path):
    """A radius that empties the deterministic crop must fail before rendering."""
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    exit_code = run_cli(
        inputs,
        output,
        extra=["--cleanup-radius", "60", "--render-width", "120", "--render-height", "120"],
    )

    assert exit_code == lane.EXIT_INPUT_ERROR
    assert not (output / "inserted.mp4").exists()


def test_output_and_report_paths_cannot_alias(tmp_path: Path):
    """A JSON receipt must never replace the successful composite path."""
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"

    exit_code = run_cli(inputs, output, extra=["--report", str(output / "composite.mp4")])

    assert exit_code == lane.EXIT_INPUT_ERROR
    assert not (output / "composite.mp4").exists()


def test_the_render_manifest_binds_the_requested_device():
    """The accepted manifest must prove the requested CPU or GPU device."""
    manifest = {
        "blender_executable_sha256": "a" * 64,
        "blender_version": "fake",
        "counts": {"mesh_objects": 1},
        "device": "GPU",
        "engine": "CYCLES",
        "frames": 1,
        "height": 384,
        "pose_sha256": "b" * 64,
        "profile": lane.profile_record(PROFILE),
        "samples": 1,
        "script_sha256": "c" * 64,
        "width": 672,
    }

    with pytest.raises(lane.AdvertisingTire3DError, match="device"):
        lane.verify_render_manifest(
            manifest,
            blender_sha256="a" * 64,
            script_sha256="c" * 64,
            pose_sha256="b" * 64,
            samples=1,
            width=672,
            height=384,
            frames=1,
            profile=PROFILE,
            device="CPU",
        )


def test_the_solved_tilt_makes_the_modelled_silhouette_match_the_ellipse():
    """The tilt solve must return the tilt whose silhouette matches the ratio."""
    assert lane.projected_minor_ratio(PROFILE, 0.0) == pytest.approx(1.0)
    assert lane.solve_tilt_degrees(PROFILE, 1.0) == pytest.approx(0.0)
    for ratio in (0.45, 0.6, 0.8, 0.95):
        tilt = lane.solve_tilt_degrees(PROFILE, ratio)
        assert lane.projected_minor_ratio(PROFILE, tilt) == pytest.approx(ratio, abs=1e-6)
    assert lane.solve_tilt_degrees(PROFILE, 0.05) == pytest.approx(lane.MAX_TILT_DEGREES)


def test_fit_wheel_poses_maps_the_ellipse_onto_a_three_dimensional_wheel():
    """The fitted ellipse must become a wheel position, scale, tilt, and axis."""
    fits = [fit for fit in [lane.fit_ellipse(ellipse_mask(40.0, 60.0, 26.0, 18.0))] if fit]
    assert fits
    poses = lane.fit_wheel_poses(
        fits,
        profile=PROFILE,
        source_width=FRAME_WIDTH,
        source_height=FRAME_HEIGHT,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
        meters_per_pixel=METERS_PER_PIXEL,
    )
    assert len(poses) == 1
    pose = poses[0]
    fit = fits[0]
    left, top, right, bottom = lane.content_box(
        FRAME_WIDTH, FRAME_HEIGHT, RENDER_WIDTH, RENDER_HEIGHT
    )
    factor_x = (right - left) / FRAME_WIDTH
    factor_y = (bottom - top) / FRAME_HEIGHT
    assert pose.position[0] == pytest.approx(
        (left + fit.center_x * factor_x - RENDER_WIDTH / 2.0) * METERS_PER_PIXEL
    )
    assert pose.position[1] == pytest.approx(
        (RENDER_HEIGHT / 2.0 - (top + fit.center_y * factor_y)) * METERS_PER_PIXEL
    )
    assert pose.scale == pytest.approx(fit.radius_x * factor_x * METERS_PER_PIXEL)
    ratio = fit.radius_y / fit.radius_x
    assert pose.tilt_degrees == pytest.approx(lane.solve_tilt_degrees(PROFILE, ratio))
    assert lane.projected_minor_ratio(PROFILE, pose.tilt_degrees) == pytest.approx(ratio, abs=1e-6)
    assert pose.axis[2] == pytest.approx(math.cos(math.radians(pose.tilt_degrees)))
    assert math.hypot(pose.axis[0], pose.axis[1]) == pytest.approx(
        math.sin(math.radians(pose.tilt_degrees))
    )
    assert pose.held is False


def test_the_wheel_rotation_matrix_is_orthonormal_and_reproduces_the_euler_angles():
    """The pose rotation must be a proper rotation that the Euler angles rebuild."""
    fits = [fit for fit in [lane.fit_ellipse(ellipse_mask(40.0, 60.0, 26.0, 18.0))] if fit]
    pose = lane.fit_wheel_poses(
        fits,
        profile=PROFILE,
        source_width=FRAME_WIDTH,
        source_height=FRAME_HEIGHT,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
        meters_per_pixel=METERS_PER_PIXEL,
    )[0]
    matrix = np.array(pose.rotation_matrix, dtype=np.float64)
    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    rebuilt = lane.matrix_from_euler(*pose.rotation_euler_degrees)
    assert rebuilt == pytest.approx(matrix, abs=1e-9)


def test_the_spin_advances_with_the_fitted_centre_motion():
    """A wheel that moves forward must spin backwards about its own axis."""
    fits = [
        fit
        for fit in (
            lane.fit_ellipse(ellipse_mask(40.0 + 8.0 * index, 60.0, 26.0, 18.0))
            for index in range(3)
        )
        if fit
    ]
    assert len(fits) == 3
    poses = lane.fit_wheel_poses(
        fits,
        profile=PROFILE,
        source_width=FRAME_WIDTH,
        source_height=FRAME_HEIGHT,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
        meters_per_pixel=METERS_PER_PIXEL,
    )
    assert poses[0].spin_degrees == pytest.approx(0.0)
    travel = fits[1].center_x - fits[0].center_x
    expected = -math.degrees(travel / fits[1].radius_x)
    assert poses[1].spin_degrees == pytest.approx(expected)
    second = (fits[2].center_x - fits[1].center_x) / fits[2].radius_x
    assert poses[2].spin_degrees == pytest.approx(expected - math.degrees(second))


def run_cli(
    inputs: dict[str, Path],
    output: Path,
    *,
    target: str = "a black tire with a yellow band",
    extra: list[str] | None = None,
) -> int:
    """Run the lane CLI over the synthetic inputs."""
    output.mkdir(parents=True, exist_ok=True)
    blender = output / "blender"
    blender.write_bytes(b"blender-placeholder")
    command = [
        "--input",
        str(inputs["video"]),
        "--primary-mask",
        str(inputs["mask"]),
        "--plate",
        str(inputs["plate"]),
        "--target",
        target,
        "--output",
        str(output / "composite.mp4"),
        "--inserted-mask",
        str(output / "inserted.mp4"),
        "--cleanup-mask",
        str(output / "cleanup.mp4"),
        "--poses",
        str(output / "poses.json"),
        "--blender",
        str(blender),
        "--report",
        str(output / "receipt.json"),
    ]
    if extra is not None:
        command += extra
    return lane.main(command)


def test_the_lane_composites_the_render_and_writes_the_masks_and_the_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The full lane must write the composite, both masks, the poses, and the receipt."""
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"
    poses_seen = fake_blender(monkeypatch)
    exit_code = run_cli(inputs, output)
    assert exit_code == 0
    composite = probe_video(output / "composite.mp4")
    assert (composite.width, composite.height, composite.frames) == (
        FRAME_WIDTH,
        FRAME_HEIGHT,
        FRAME_COUNT,
    )
    inserted = decode_gray_frames(output / "inserted.mp4", tmp_path / "decoded-inserted")
    cleanup = decode_gray_frames(output / "cleanup.mp4", tmp_path / "decoded-cleanup")
    assert len(inserted) == FRAME_COUNT
    for index, frame in enumerate(inserted):
        assert set(np.unique(frame).tolist()) <= {0, 255}
        assert np.all(frame[fitted_ellipse_mask(index, scale=0.8)] == 0)
        assert frame[0, 0] == 255
    for index, frame in enumerate(cleanup):
        assert set(np.unique(frame).tolist()) <= {0, 255}
        assert np.all(frame[fitted_ellipse_mask(index, scale=0.8)] == 255)
        assert frame[0, 0] == 255
    poses = json.loads((output / "poses.json").read_text(encoding="utf-8"))
    assert poses["schema"] == lane.POSE_SCHEMA
    assert poses["width"] == RENDER_WIDTH
    assert poses["height"] == RENDER_HEIGHT
    assert len(poses["poses"]) == FRAME_COUNT
    assert poses["target"]["label"] is None
    assert poses_seen[0]["poses"][0]["frame"] == 0
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == lane.SCHEMA
    assert receipt["target"]["compiled"]["band_color"] == list(lane.COLOR_WORDS["yellow"])
    assert receipt["summary"]["frames"] == FRAME_COUNT
    assert receipt["summary"]["frames_with_inserted_pixels"] == FRAME_COUNT
    assert receipt["summary"]["mean_footprint_pixels"] > 0.0
    assert receipt["render"]["engine"] == "CYCLES"
    assert receipt["render"]["counts"]["mesh_objects"] == 38
    assert receipt["outputs"]["composite"]["sha256"] == lane.sha256(output / "composite.mp4")
    assert receipt["outputs"]["inserted_mask"]["mask_family"] == "roi-black"
    assert receipt["outputs"]["cleanup_mask"]["mask_family"] == "roi-black"
    assert receipt["non_claims"]


def test_the_lane_fails_closed_on_an_unsupported_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """An unsupported target must stop the lane before any render runs."""
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"
    fake_blender(monkeypatch)
    exit_code = run_cli(inputs, output, target="a red sports car")
    assert exit_code == lane.EXIT_INPUT_ERROR
    assert "unsupported target" in capsys.readouterr().err
    assert not (output / "composite.mp4").exists()


def test_the_lane_rejects_a_render_manifest_that_does_not_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A manifest whose digest does not match the pose file must fail closed."""
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "out"
    fake_blender(monkeypatch)

    def run(
        blender: Path,
        script: Path,
        poses: Path,
        render_output: Path,
        *,
        samples: int,
        device: str,
        blend: Path | None,
        glb: Path | None,
    ) -> None:
        render_output.mkdir(parents=True, exist_ok=True)
        (render_output / "render-manifest.json").write_text(
            json.dumps(
                {
                    "blender_version": "4.2.0 fake",
                    "blender_executable_sha256": "0" * 64,
                    "script_sha256": lane.sha256(script),
                    "pose_sha256": lane.sha256(poses),
                    "engine": "CYCLES",
                    "samples": samples,
                    "width": RENDER_WIDTH,
                    "height": RENDER_HEIGHT,
                    "frames": FRAME_COUNT,
                    "counts": {"mesh_objects": 1},
                    "profile": lane.profile_record(PROFILE),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(lane, "run_blender", run)
    exit_code = run_cli(inputs, output)
    assert exit_code == lane.EXIT_INPUT_ERROR
    assert not (output / "composite.mp4").exists()
