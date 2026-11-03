from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from post_mask_iteration import (
    DERIVATIVE_FRAME_COUNT,
    DERIVATIVE_HEIGHT,
    DERIVATIVE_WIDTH,
    IterationError,
    assert_video_alignment,
    composite_frames,
    export_candidate_chain,
    insertion_masks,
    report_non_claims,
    resolve_staged_frame_count,
    resolve_target_masks,
    validate_background_prompt,
    validate_clean_plate_pane,
    write_execution_derivative,
)

from samurai.insertion import IterationScore, SpatialTransform
from samurai.media import decode_gray_frames, probe_video, write_gray_video
from samurai.quadmask import KEEP


def test_foreground_white_sam_mask_becomes_white_target_and_recorded_polarity() -> None:
    foreground_white = np.zeros((4, 4), dtype=np.uint8)
    foreground_white[1:3, 1:3] = 255

    targets, record = resolve_target_masks([foreground_white], declared_polarity="foreground_white")

    assert np.array_equal(targets[0], foreground_white)
    assert record["requested"] == "foreground_white"
    assert record["resolved"] == "foreground_white"
    assert record["target_pixels"] == 4


def test_auto_mask_polarity_fails_closed() -> None:
    foreground_white = np.zeros((4, 4), dtype=np.uint8)
    foreground_white[1:3, 1:3] = 255

    with pytest.raises(IterationError, match="mask polarity cannot be inferred"):
        resolve_target_masks([foreground_white], declared_polarity="auto")


def test_canonical_roi_black_mask_becomes_white_target_for_scoring() -> None:
    raw = np.full((4, 4), 255, dtype=np.uint8)
    raw[1:3, 1:3] = 0

    targets, record = resolve_target_masks([raw], declared_polarity="roi_black")

    assert np.count_nonzero(targets[0] == 255) == 4
    assert record["requested"] == "roi_black"
    assert record["resolved"] == "roi_black"
    assert record["target_pixels"] == 4


def test_composite_uses_clean_plate_and_full_render_alpha(tmp_path: Path) -> None:
    plate = np.full((4, 4, 3), 20, dtype=np.uint8)
    plate[0, 0] = (10, 20, 30)
    Image.fromarray(plate, mode="RGB").save(tmp_path / "plate-000000.png")
    rendered = np.zeros((4, 4, 4), dtype=np.uint8)
    rendered[1:3, 1:3] = (200, 40, 60, 255)
    Image.fromarray(rendered, mode="RGBA").save(tmp_path / "alpha-000000.png")

    outputs = composite_frames(
        [tmp_path / "plate-000000.png"],
        tmp_path,
        tmp_path / "composite",
        frame_count=1,
    )

    result = np.asarray(Image.open(outputs[0]).convert("RGB"))
    assert tuple(result[0, 0]) == (10, 20, 30)
    assert tuple(result[1, 1]) == (200, 40, 60)


def test_insertion_masks_are_roi_black_and_ring_excludes_object() -> None:
    alpha = np.zeros((7, 7), dtype=np.uint8)
    alpha[2:5, 2:5] = 255

    inserted, cleanup = insertion_masks([alpha], cleanup_radius=3)

    assert np.count_nonzero(inserted[0] == 0) == 9
    assert np.count_nonzero(cleanup[0] == 0) == 40
    assert np.all(inserted[0][cleanup[0] == 0] == 255)


def test_insertion_mask_covers_every_nonzero_render_alpha_pixel() -> None:
    alpha = np.zeros((1, 11), dtype=np.uint8)
    alpha[0, 2:5] = 255
    alpha[0, 9] = 5

    inserted, cleanup = insertion_masks([alpha], cleanup_radius=3)

    assert inserted[0][0, 9] == 0
    assert cleanup[0][0, 9] == 255


def test_whitespace_only_background_prompt_fails_before_media_work() -> None:
    with pytest.raises(IterationError, match="non-whitespace"):
        validate_background_prompt("   ", dry_run=False)


def test_real_background_prompt_passes_preflight() -> None:
    validate_background_prompt("an empty warehouse background", dry_run=False)


def test_report_non_claims_match_the_current_transform_search_result() -> None:
    boundary = report_non_claims(True)
    interior = report_non_claims(False)

    assert any("lie on the declared transform-search boundary" in item for item in boundary)
    assert not any("lie on the declared transform-search boundary" in item for item in interior)
    assert any("remains inside the declared search grid" in item for item in interior)
    assert all("predates pane role/digest validation" not in item for item in boundary + interior)
    assert any("digest was validated" in item for item in boundary + interior)


def test_staged_pass2_frame_count_accepts_the_supported_cadence() -> None:
    assert resolve_staged_frame_count(85, source_count=96) == 85


@pytest.mark.parametrize("count", [0, 97, 96, 84, 86])
def test_staged_pass2_frame_count_rejects_unsupported_or_out_of_range_counts(count: int) -> None:
    with pytest.raises(IterationError, match="staged Pass 2 frame count"):
        resolve_staged_frame_count(count, source_count=96)


def test_clean_plate_pane_requires_the_declared_role_and_matching_digest(tmp_path: Path) -> None:
    source = tmp_path / "paired-source.mp4"
    source.write_bytes(b"paired source bytes")
    digest = hashlib.sha256(b"paired source bytes").hexdigest()

    validate_clean_plate_pane(
        {"right_pane": {"role": "void_pass1_clean_plate", "sha256": digest}}, source
    )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({}, "right_pane"),
        ({"right_pane": {"role": "other", "sha256": "0" * 64}}, "right_pane"),
        ({"right_pane": {"role": "void_pass1_clean_plate", "sha256": "0" * 64}}, "match"),
    ],
)
def test_clean_plate_pane_fails_closed_without_role_or_digest_match(
    tmp_path: Path, record: dict, message: str
) -> None:
    source = tmp_path / "paired-source.mp4"
    source.write_bytes(b"different bytes")

    with pytest.raises(IterationError, match=message):
        validate_clean_plate_pane(record, source)


def make_color_video(path: Path, frame_count: int, width: int = 8, height: int = 4) -> None:
    frame_dir = path.parent / f"{path.stem}-frames"
    frame_dir.mkdir()
    for index in range(frame_count):
        array = np.full((height, width, 3), (20 + index, 40, 60), dtype=np.uint8)
        from PIL import Image

        Image.fromarray(array, mode="RGB").save(frame_dir / f"frame-{index:06d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "12",
            "-i",
            str(frame_dir / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def make_candidate(name: str, passes: int) -> IterationScore:
    return IterationScore(
        name=name,
        transform=SpatialTransform(),
        smoothing_passes=passes,
        mean_iou=0.0,
        mean_precision=0.0,
        mean_mask_coverage=0.0,
        spatial_objective=0.0,
        temporal_centroid_pop=0.0,
        temporal_mass_pop=0.0,
        temporal_pop_score=0.0,
        anchor_landmark_rmse=0.0,
        objective=float(passes),
    )


def test_export_candidate_chain_exports_every_fitted_candidate_with_derivative_metadata(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "chains"
    source_mask = np.full((4, 8), 255, dtype=np.uint8)
    source_mask[1:3, 3:5] = 0
    chains = []
    for name, passes in (("smooth-00", 0), ("smooth-02", 2)):
        candidate = make_candidate(name, passes)
        alpha_dir = tmp_path / name
        alpha_dir.mkdir()
        alpha = np.zeros((4, 8, 4), dtype=np.uint8)
        alpha[1:3, 3:5] = (200, 40, 60, 255)
        Image.fromarray(alpha, mode="RGBA").save(alpha_dir / "alpha-000000.png")
        plate = tmp_path / f"{name}-plate-000000.png"
        Image.fromarray(np.full((4, 8, 3), 30, dtype=np.uint8), mode="RGB").save(plate)
        chains.append(
            export_candidate_chain(
                candidate=candidate,
                alpha_directory=alpha_dir,
                alphas=[alpha[:, :, 3]],
                clean_plate_frames=[plate],
                masks=[source_mask],
                output_root=output_root,
                background_prompt="an empty room",
                cleanup_radius=1,
                staged_frame_count=1,
                source_fps=12.0,
                source_frame_count=1,
                derivative_frame_count=1,
                derivative_width=4,
                derivative_height=2,
            )
        )

    assert [chain["name"] for chain in chains] == ["smooth-00", "smooth-02"]
    for chain in chains:
        metadata = json.loads((chain["chain_dir"] / "metadata.json").read_text())
        assert metadata["candidate"]["name"] == chain["name"]
        derivative = metadata["execution_derivative"]
        assert derivative["frame_count"] == 1
        assert derivative["width"] == 4
        assert derivative["height"] == 2
        assert any(
            token == "scale=4:2:flags=neighbor,format=gray,setpts=N/FRAME_RATE/TB"
            for token in derivative["commands"]["inserted_mask"]
        )
        assert any(
            token == "scale=4:2:flags=neighbor,format=gray,setpts=N/FRAME_RATE/TB"
            for token in derivative["commands"]["cleanup_mask"]
        )
        for record in derivative["outputs"].values():
            assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
        inserted = decode_gray_frames(
            chain["chain_dir"] / "inserted-mask.mp4", tmp_path / f"{chain['name']}-inserted"
        )
        cleanup = decode_gray_frames(
            chain["chain_dir"] / "cleanup-mask.mp4", tmp_path / f"{chain['name']}-cleanup"
        )
        assert all(np.isin(frame, (0, 255)).all() for frame in inserted + cleanup)


def test_execution_derivative_is_cadence_aligned_binary_and_reproducible(
    tmp_path: Path,
) -> None:
    assert DERIVATIVE_FRAME_COUNT == 85
    assert (DERIVATIVE_WIDTH, DERIVATIVE_HEIGHT) == (672, 384)
    source = tmp_path / "source.mp4"
    make_color_video(source, 6)
    inserted_source = tmp_path / "inserted-source.mp4"
    cleanup_source = tmp_path / "cleanup-source.mp4"
    inserted_frames = []
    cleanup_frames = []
    for index in range(6):
        inserted = np.full((4, 8), 255, dtype=np.uint8)
        cleanup = np.full((4, 8), 255, dtype=np.uint8)
        inserted[1:3, 2 + index % 3 : 3 + index % 3] = 0
        cleanup[1:3, 4:6] = 0
        inserted_frames.append(inserted)
        cleanup_frames.append(cleanup)
    write_gray_video(inserted_frames, inserted_source, fps=12.0)
    write_gray_video(cleanup_frames, cleanup_source, fps=12.0)

    derivative = write_execution_derivative(
        source,
        inserted_source,
        cleanup_source,
        tmp_path / "derivative",
        background_prompt="an empty room",
        frame_count=5,
        width=6,
        height=4,
        fps=12.0,
        source_count=6,
    )

    assert derivative["frame_count"] == 5
    assert derivative["frame_count"] % 4 == 1
    assert any("flags=neighbor" in token for token in derivative["commands"]["inserted_mask"])
    assert any("flags=neighbor" in token for token in derivative["commands"]["cleanup_mask"])
    output_paths = list(derivative["outputs"].values())
    for path in output_paths[:3]:
        info = probe_video(path)
        assert (info.width, info.height, info.frames) == (6, 4, 5)
    assert_video_alignment(
        {str(index): path for index, path in enumerate(output_paths[:3])},
        width=6,
        height=4,
        frames=5,
    )
    for name in ("inserted_mask", "cleanup_mask"):
        frames = decode_gray_frames(derivative["outputs"][name], tmp_path / f"{name}-decoded")
        assert all(np.isin(frame, (0, 255)).all() for frame in frames)
    quadmask_frames = decode_gray_frames(
        derivative["outputs"]["quadmask"], tmp_path / "quadmask-decoded"
    )
    assert np.all(quadmask_frames[0] == KEEP)

    original_digests = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in derivative["outputs"].items()
    }
    repeat = write_execution_derivative(
        source,
        inserted_source,
        cleanup_source,
        tmp_path / "derivative-repeat",
        background_prompt="an empty room",
        frame_count=5,
        width=6,
        height=4,
        fps=12.0,
        source_count=6,
    )
    assert original_digests == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in repeat["outputs"].items()
    }


def test_execution_derivative_pts_alignment_survives_varying_color_and_static_mask(
    tmp_path: Path,
) -> None:
    source = tmp_path / "varying-source.mp4"
    make_color_video(source, 13)
    inserted_source = tmp_path / "varying-inserted.mp4"
    cleanup_source = tmp_path / "static-cleanup.mp4"
    inserted_frames = []
    cleanup_frames = []
    for index in range(13):
        inserted = np.full((4, 8), 255, dtype=np.uint8)
        inserted[1:3, 2 + index % 4 : 3 + index % 4] = 0
        inserted_frames.append(inserted)
        cleanup_frames.append(np.full((4, 8), 255, dtype=np.uint8))
    write_gray_video(inserted_frames, inserted_source, fps=12.0)
    write_gray_video(cleanup_frames, cleanup_source, fps=12.0)

    derivative = write_execution_derivative(
        source,
        inserted_source,
        cleanup_source,
        tmp_path / "derivative",
        background_prompt="an empty room",
        frame_count=13,
        width=6,
        height=4,
        fps=12.0,
        source_count=13,
    )

    assert_video_alignment(
        {
            name: derivative["outputs"][name]
            for name in (
                "input_video",
                "inserted_mask",
                "cleanup_mask",
                "quadmask",
            )
        },
        width=6,
        height=4,
        frames=13,
    )
    for name in ("color", "inserted_mask", "cleanup_mask"):
        command = derivative["commands"][name]
        assert command[command.index("-bf") + 1] == "0"
