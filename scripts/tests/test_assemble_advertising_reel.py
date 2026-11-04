from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from assemble_advertising_reel import STAGES, band_height, measure_insertion
from PIL import Image

from samurai.media import extract_color_frames

SCRIPT = Path(__file__).resolve().parents[1] / "assemble_advertising_reel.py"
ROOT = Path(__file__).resolve().parents[2]
PYTHON_ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
}
MANIFEST_SCHEMA = "void-fusion-lab.advertising-reel.manifest.v1"
REPORT_SCHEMA = "void-fusion-lab.advertising-reel.report.v1"
WIDTH = 64
HEIGHT = 48
FRAMES = 4
FPS = 12
BACKGROUND = 20


def write_rgb_video(frames: list[np.ndarray], path: Path, *, fps: int = FPS) -> None:
    """Write one lossless color video from a list of RGB arrays."""
    staging = path.parent / f".{path.stem}-frames"
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(staging / f"frame-{index:06d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(staging / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def base_frame() -> np.ndarray:
    """Return one empty background frame."""
    return np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)


def original_frames() -> list[np.ndarray]:
    """Return frames that show the selected object on the empty background."""
    frames = []
    for _ in range(FRAMES):
        frame = base_frame()
        frame[10:24, 16:32] = 180
        frames.append(frame)
    return frames


def mask_overlay_frames() -> list[np.ndarray]:
    """Return frames that show the selection mask halo over the background."""
    frames = []
    for _ in range(FRAMES):
        frame = base_frame()
        frame[10:24, 16:32] = 220
        frame[8:26, 14:34] = 90
        frames.append(frame)
    return frames


def clean_plate_frames() -> list[np.ndarray]:
    """Return the Pass 1 clean plate frames with the object removed."""
    return [base_frame() for _ in range(FRAMES)]


def inserted_frames() -> list[np.ndarray]:
    """Return frames that carry a real fitted insertion over the clean plate."""
    frames = []
    for _ in range(FRAMES):
        frame = base_frame()
        frame[12:22, 18:30] = 200
        frames.append(frame)
    return frames


def pass2_frames() -> list[np.ndarray]:
    """Return Pass 2 frames that keep the insertion and refine its cleanup ring."""
    frames = []
    for _ in range(FRAMES):
        frame = base_frame()
        frame[12:22, 18:30] = 190
        frame[10:24, 16:32] = 60
        frames.append(frame)
    return frames


def fit_frames(frames: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    """Pad or crop frames to one canvas size."""
    width, height = size
    result = []
    for frame in frames:
        padded = frame
        if padded.shape[1] < width:
            padded = np.pad(padded, ((0, 0), (0, width - padded.shape[1]), (0, 0)), mode="edge")
        if padded.shape[0] < height:
            padded = np.pad(padded, ((0, height - padded.shape[0]), (0, 0), (0, 0)), mode="edge")
        result.append(padded[:height, :width])
    return result


def build_shot(
    tmp_path: Path,
    name: str,
    *,
    size: tuple[int, int] = (WIDTH, HEIGHT),
    frames: int = FRAMES,
    fps: int = FPS,
    inserted: list[np.ndarray] | None = None,
) -> dict[str, Path]:
    """Write the five stage videos of one shot and return their paths."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    stages = {
        "original": original_frames(),
        "mask_overlay": mask_overlay_frames(),
        "pass1_clean_plate": clean_plate_frames(),
        "inserted": inserted_frames() if inserted is None else inserted,
        "pass2": pass2_frames(),
    }
    paths: dict[str, Path] = {}
    for stage, stage_frames in stages.items():
        path = root / f"{stage}.mp4"
        write_rgb_video(fit_frames(stage_frames, size)[:frames], path, fps=fps)
        paths[stage] = path
    return paths


def write_manifest(
    path: Path,
    shots: list[tuple[str, str, dict[str, Path]]],
    *,
    schema: str = MANIFEST_SCHEMA,
) -> Path:
    """Write one advertising spot manifest."""
    payload = {
        "schema": schema,
        "title": "VOID Fusion Ad Spot",
        "shots": [
            {
                "id": shot_id,
                "label": label,
                "stages": {stage: str(stage_path) for stage, stage_path in stages.items()},
            }
            for shot_id, label, stages in shots
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the assembler script in a child process."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env=PYTHON_ENV,
    )


def decode_frames(video: Path, destination: Path) -> list[np.ndarray]:
    """Decode one video into its RGB frames."""
    return [
        np.asarray(Image.open(path).convert("RGB")).copy()
        for path in extract_color_frames(video, destination)
    ]


def build_two_shot_spot(tmp_path: Path) -> tuple[Path, list[dict[str, Path]]]:
    """Build a two-shot advertising spot and its manifest."""
    shot_one = build_shot(tmp_path, "shot-one")
    shot_two = build_shot(tmp_path, "shot-two")
    manifest = write_manifest(
        tmp_path / "spot.json",
        [
            ("shot-01", "WIDE ESTABLISHING", shot_one),
            ("shot-02", "PRODUCT CLOSE UP", shot_two),
        ],
    )
    return manifest, [shot_one, shot_two]


def assert_close(left: np.ndarray, right: np.ndarray, tolerance: int = 4) -> None:
    """Assert that two decoded frames agree within a lossy-codec tolerance."""
    difference = np.abs(left.astype(int) - right.astype(int))
    assert difference.max() <= tolerance, difference.max()


def test_cli_assembles_the_reel_and_the_comparison_master(tmp_path) -> None:
    manifest, _ = build_two_shot_spot(tmp_path)
    reel = tmp_path / "reel.mp4"
    master = tmp_path / "master.mp4"
    report_path = tmp_path / "report.json"

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(reel),
        "--master",
        str(master),
        "--report",
        str(report_path),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["schema"] == REPORT_SCHEMA
    assert report["stages"] == list(STAGES)
    assert [shot["id"] for shot in report["shots"]] == ["shot-01", "shot-02"]
    assert [shot["label"] for shot in report["shots"]] == [
        "WIDE ESTABLISHING",
        "PRODUCT CLOSE UP",
    ]
    assert report["reel"]["frames"] == FRAMES * 2
    assert report["comparison_master"]["frames"] == FRAMES * 2
    assert report["reel"]["fps"] == "12"
    assert report["reel"]["width"] == WIDTH
    assert report["reel"]["height"] == HEIGHT
    assert report["reel"]["stage"] == "pass2"
    assert report["comparison_master"]["columns"] == list(STAGES)
    assert report["comparison_master"]["rows"] == 2
    assert report["comparison_master"]["width"] == WIDTH * len(STAGES)
    assert report["non_claims"]
    assert reel.is_file()
    assert master.is_file()
    for shot in report["shots"]:
        assert set(shot["stages"]) == set(STAGES)
        for stage in STAGES:
            record = shot["stages"][stage]
            assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
            assert record["frames"] == FRAMES
            assert record["width"] == WIDTH
            assert record["height"] == HEIGHT
        assert shot["insertion_check"]["exactly_identical"] is False
        assert shot["insertion_check"]["max_frame_changed_ratio"] > 0.0
    assert report["reel"]["sha256"] == hashlib.sha256(reel.read_bytes()).hexdigest()
    assert report["comparison_master"]["sha256"] == hashlib.sha256(master.read_bytes()).hexdigest()
    summary = json.loads(result.stdout)
    assert summary["reel"]["path"] == str(reel)
    assert summary["non_claims"]


def test_cli_writes_a_labeled_reel_in_shot_order(tmp_path) -> None:
    manifest, shots = build_two_shot_spot(tmp_path)
    reel = tmp_path / "reel.mp4"

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(reel),
        "--master",
        str(tmp_path / "master.mp4"),
        "--lossless",
    )

    assert result.returncode == 0, result.stderr
    frames = decode_frames(reel, tmp_path / "decoded-reel")
    assert len(frames) == FRAMES * 2
    band = band_height(HEIGHT, divisor=10, minimum=24)
    body = HEIGHT - band
    for index, shot in enumerate(shots):
        source = decode_frames(shot["pass2"], tmp_path / f"source-{index}")[0]
        frame = frames[index * FRAMES]
        assert_close(frame[:body], source[:body])
        assert frame[body:].max() > 100


def test_cli_places_every_stage_of_every_shot_in_the_comparison_master(tmp_path) -> None:
    manifest, shots = build_two_shot_spot(tmp_path)
    master = tmp_path / "master.mp4"

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(master),
        "--lossless",
    )

    assert result.returncode == 0, result.stderr
    frames = decode_frames(master, tmp_path / "decoded-master")
    assert len(frames) == FRAMES * 2
    shot_band = band_height(HEIGHT, divisor=12, minimum=20)
    for index, shot in enumerate(shots):
        frame = frames[index * FRAMES]
        assert frame[:shot_band].max() > 100
        for column, stage in enumerate(STAGES):
            source = decode_frames(shot[stage], tmp_path / f"cell-{index}-{stage}")[0]
            left = column * WIDTH
            cell = frame[shot_band : shot_band + HEIGHT, left : left + WIDTH]
            assert_close(cell, source)
            assert frame[shot_band + HEIGHT :, left : left + WIDTH].max() > 100


def test_cli_can_build_the_reel_from_another_stage(tmp_path) -> None:
    manifest, shots = build_two_shot_spot(tmp_path)
    reel = tmp_path / "reel.mp4"

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(reel),
        "--master",
        str(tmp_path / "master.mp4"),
        "--reel-stage",
        "inserted",
        "--lossless",
    )

    assert result.returncode == 0, result.stderr
    frames = decode_frames(reel, tmp_path / "decoded-reel")
    band = band_height(HEIGHT, divisor=10, minimum=24)
    body = HEIGHT - band
    source = decode_frames(shots[0]["inserted"], tmp_path / "source")[0]
    assert_close(frames[0][:body], source[:body])


def test_cli_fails_closed_on_a_missing_stage(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "shots": [
            {
                "id": "shot-01",
                "label": "WIDE ESTABLISHING",
                "stages": {
                    stage: str(stage_path) for stage, stage_path in shot.items() if stage != "pass2"
                },
            }
        ],
    }
    manifest = tmp_path / "spot.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reel = tmp_path / "reel.mp4"

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(reel),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "missing stage" in result.stderr
    assert "pass2" in result.stderr
    assert not reel.exists()


def test_cli_fails_closed_on_a_missing_stage_file(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    shot["inserted"] = tmp_path / "shot-one" / "absent.mp4"
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert "inserted" in result.stderr


def test_cli_fails_closed_on_a_geometry_mismatch(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    odd = tmp_path / "shot-one" / "pass2-odd.mp4"
    write_rgb_video(fit_frames(pass2_frames(), (WIDTH + 16, HEIGHT + 8)), odd)
    shot["pass2"] = odd
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "geometry mismatch" in result.stderr
    assert "pass2" in result.stderr


def test_cli_fails_closed_on_a_frame_count_mismatch(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    short = tmp_path / "shot-one" / "pass2-short.mp4"
    write_rgb_video(fit_frames(pass2_frames(), (WIDTH, HEIGHT))[:2], short)
    shot["pass2"] = short
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "frame count mismatch" in result.stderr


def test_cli_fails_closed_on_a_frame_rate_mismatch(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    slow = tmp_path / "shot-one" / "pass2-slow.mp4"
    write_rgb_video(fit_frames(pass2_frames(), (WIDTH, HEIGHT)), slow, fps=10)
    shot["pass2"] = slow
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "frame rate mismatch" in result.stderr


def test_cli_fails_closed_on_a_geometry_mismatch_between_shots(tmp_path) -> None:
    shot_one = build_shot(tmp_path, "shot-one")
    shot_two = build_shot(tmp_path, "shot-two", size=(WIDTH + 16, HEIGHT + 8))
    manifest = write_manifest(
        tmp_path / "spot.json",
        [
            ("shot-01", "WIDE ESTABLISHING", shot_one),
            ("shot-02", "PRODUCT CLOSE UP", shot_two),
        ],
    )

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "geometry mismatch" in result.stderr


def test_cli_fails_closed_when_the_insertion_matches_the_clean_plate(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one", inserted=clean_plate_frames())
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "nearly identical" in result.stderr
    assert not (tmp_path / "reel.mp4").exists()


def test_cli_fails_closed_when_the_insertion_is_below_the_change_threshold(tmp_path) -> None:
    faint = [
        np.clip(frame.astype(int) + 2, 0, 255).astype(np.uint8) for frame in clean_plate_frames()
    ]
    shot = build_shot(tmp_path, "shot-one", inserted=faint)
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "nearly identical" in result.stderr


def test_insertion_change_threshold_uses_the_documented_channel_mean(tmp_path) -> None:
    plate = tmp_path / "plate.png"
    inserted = tmp_path / "inserted.png"
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(plate)
    Image.fromarray(np.array([[[30, 0, 0]]], dtype=np.uint8)).save(inserted)

    metrics = measure_insertion([plate], [inserted], changed_threshold=20.0)

    assert metrics["mean_absolute_difference"] == pytest.approx(10.0)
    assert metrics["mean_changed_ratio"] == 0.0


def test_cli_fails_closed_on_an_unsupported_manifest_schema(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    manifest = write_manifest(
        tmp_path / "spot.json",
        [("shot-01", "WIDE", shot)],
        schema="void-fusion-lab.something-else.v1",
    )

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "unsupported manifest schema" in result.stderr


def test_cli_fails_closed_on_a_duplicate_shot_id(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    manifest = write_manifest(
        tmp_path / "spot.json",
        [("shot-01", "WIDE", shot), ("shot-01", "CLOSE", shot)],
    )

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "declared more than once" in result.stderr


def test_cli_rejects_a_non_finite_change_threshold(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
        "--changed-threshold",
        "nan",
    )

    assert result.returncode == 2
    assert "must be a finite nonnegative number" in result.stderr


def test_cli_rejects_an_out_of_range_insertion_ratio(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
        "--min-insertion-changed-ratio",
        "1.5",
    )

    assert result.returncode == 2
    assert "must be a finite ratio in [0, 1]" in result.stderr


def test_cli_prints_the_full_report_when_no_report_path_is_given(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == REPORT_SCHEMA
    assert report["shots"][0]["id"] == "shot-01"


def test_cli_fails_closed_on_an_unreadable_stage_file(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    broken = tmp_path / "shot-one" / "broken.mp4"
    broken.write_text("not a video", encoding="utf-8")
    shot["pass2"] = broken
    manifest = write_manifest(tmp_path / "spot.json", [("shot-01", "WIDE", shot)])

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 2
    assert "could not probe stage video" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_resolves_stage_paths_relative_to_the_manifest(tmp_path) -> None:
    shot = build_shot(tmp_path, "shot-one")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "shots": [
            {
                "id": "shot-01",
                "label": "WIDE ESTABLISHING",
                "stages": {
                    stage: str(stage_path.relative_to(tmp_path))
                    for stage, stage_path in shot.items()
                },
            }
        ],
    }
    manifest = tmp_path / "spot.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = run_cli(
        "--manifest",
        str(manifest),
        "--reel",
        str(tmp_path / "reel.mp4"),
        "--master",
        str(tmp_path / "master.mp4"),
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "reel.mp4").is_file()
    assert (tmp_path / "master.mp4").is_file()
