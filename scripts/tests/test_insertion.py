from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from samurai.insertion import (
    LANDMARK_NAMES,
    InsertionError,
    SpatialTransform,
    aggregate_score,
    contiguous_poses,
    optimize_transform,
    score_alpha,
    spatial_objective,
    transform_alpha,
    transform_anchor_poses,
)

RELAXATION_WEIGHT = 0.35


def anchor(frame: int, x: float) -> dict[str, Any]:
    return {
        "frame": frame,
        "points": {name: [x + frame, 20.0 + frame] for name in LANDMARK_NAMES},
    }


def test_contiguous_poses_interpolate_and_hold_missing_channels() -> None:
    first = anchor(0, 10.0)
    last = anchor(4, 30.0)
    last["points"]["left_wrist"] = None
    series = contiguous_poses([first, last], frame_count=5, smoothing_passes=0)

    assert len(series) == 5
    assert series[2]["points"]["nose"] == [22.0, 22.0]
    assert series[3]["points"]["nose"] == [28.0, 23.0]
    assert series[1]["points"]["left_wrist"] == [10.0, 20.0]
    assert series[3]["points"]["left_wrist"] == [10.0, 20.0]


def test_smoothing_preserves_anchor_landmarks_and_reduces_interior_jerk() -> None:
    first = anchor(0, 0.0)
    middle = anchor(1, 20.0)
    last = anchor(2, 0.0)
    rough = contiguous_poses([first, last], frame_count=3, smoothing_passes=0)
    smoothed = contiguous_poses([first, middle, last], frame_count=3, smoothing_passes=1)

    assert rough[1]["points"]["nose"] == [1.0, 21.0]
    assert smoothed[1]["points"]["nose"] == middle["points"]["nose"]
    assert smoothed[0]["points"] == first["points"]
    assert smoothed[2]["points"] == last["points"]


def test_catmull_rom_interpolation_goes_beyond_piecewise_linear() -> None:
    rows = [anchor(frame, value) for frame, value in ((0, 0.0), (1, 0.0), (3, 20.0), (4, 10.0))]
    series = contiguous_poses(rows, frame_count=5, smoothing_passes=0)

    assert series[1]["points"]["nose"] == [1.0, 21.0]
    assert series[3]["points"]["nose"] == [23.0, 23.0]
    assert series[2]["points"]["nose"][0] != 10.0


def test_relaxation_uses_the_original_neighbor_series() -> None:
    rows = [anchor(frame, value) for frame, value in ((0, 0.0), (1, 0.0), (3, 20.0), (4, 10.0))]
    base = contiguous_poses(rows, frame_count=5, smoothing_passes=0)
    smoothed = contiguous_poses(rows, frame_count=5, smoothing_passes=1)
    anchors = {0, 1, 3, 4}

    for frame in range(5):
        if frame in anchors:
            assert smoothed[frame]["points"] == base[frame]["points"]
            continue
        for name, current_value in base[frame]["points"].items():
            assert current_value is not None
            previous_value = base[frame - 1]["points"][name]
            following_value = base[frame + 1]["points"][name]
            neighbors = [previous_value, following_value]
            assert neighbors == [value for value in neighbors if value is not None]
            current = np.asarray(current_value, dtype=np.float64)
            neighbor_mean = np.mean(
                [np.asarray(value, dtype=np.float64) for value in neighbors], axis=0
            )
            expected = (1.0 - RELAXATION_WEIGHT) * current + RELAXATION_WEIGHT * neighbor_mean
            assert smoothed[frame]["points"][name] == pytest.approx(expected.tolist())


def test_invalid_pose_and_smoothing_inputs_fail_closed() -> None:
    with pytest.raises(Exception, match="frame_count must be positive"):
        contiguous_poses([anchor(0, 1.0)], frame_count=0, smoothing_passes=0)
    with pytest.raises(Exception, match="smoothing_passes must be nonnegative"):
        contiguous_poses([anchor(0, 1.0)], frame_count=1, smoothing_passes=-1)
    bad = anchor(0, 1.0)
    bad["points"]["nose"] = [float("nan"), 1.0]
    with pytest.raises(Exception, match="invalid landmark nose"):
        contiguous_poses([bad], frame_count=1, smoothing_passes=0)
    nonnumeric = anchor(0, 1.0)
    nonnumeric["points"]["nose"] = ["x", 1.0]
    with pytest.raises(InsertionError, match="invalid landmark nose"):
        contiguous_poses([nonnumeric], frame_count=1, smoothing_passes=0)


def test_score_alpha_distinguishes_precision_coverage_and_iou() -> None:
    alpha = np.zeros((4, 8), dtype=np.uint8)
    mask = np.zeros((4, 8), dtype=np.uint8)
    alpha[1:3, 1:4] = 255
    mask[1:3, 2:6] = 255
    score = score_alpha(alpha, mask)

    assert score.intersection == 4
    assert score.render_pixels == 6
    assert score.mask_pixels == 8
    assert score.outside_mask == 2
    assert score.iou == pytest.approx(4 / 10)
    assert score.precision == pytest.approx(4 / 6)
    assert score.mask_coverage == pytest.approx(4 / 8)
    assert spatial_objective(score) == pytest.approx(
        0.55 * score.precision + 0.35 * score.iou + 0.10 * score.mask_coverage
    )


def test_transform_alpha_and_anchor_poses_use_one_contract() -> None:
    transform = SpatialTransform(dx=2.0, dy=1.0, scale=1.0)
    alpha = np.zeros((5, 5), dtype=np.uint8)
    alpha[2, 2] = 255
    transformed = transform_alpha(alpha, transform)
    assert transformed[3, 4] == 255

    row = anchor(0, 3.0)
    result = transform_anchor_poses([row], transform)[0]
    assert result["points"]["nose"] == [5.0, 21.0]


def test_optimize_transform_finds_masked_translation() -> None:
    alpha = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    alpha[3, 3] = 255
    mask[4, 4] = 255
    transform, scores = optimize_transform(
        [alpha],
        [mask],
        translations=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        scales=[1.0],
    )
    assert (transform.dx, transform.dy, transform.scale) == (1.0, 1.0, 1.0)
    assert scores[0].intersection > 0


def test_optimize_transform_skips_fully_translated_empty_candidates() -> None:
    alpha = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    alpha[0, 0] = 255
    mask[7, 7] = 255
    transform, scores = optimize_transform(
        [alpha],
        [mask],
        translations=[(-10.0, -10.0), (7.0, 7.0)],
        scales=[1.0],
    )

    assert (transform.dx, transform.dy, transform.scale) == (7.0, 7.0, 1.0)
    assert scores[0].intersection == 1


def test_aggregate_score_penalizes_temporal_delta_and_anchor_error() -> None:
    alpha_a = np.zeros((6, 6), dtype=np.uint8)
    alpha_a[1, 1] = 255
    alpha_b = np.zeros((6, 6), dtype=np.uint8)
    alpha_b[1, 2] = 255
    alpha_b[2, 2] = 255
    alpha_c = np.zeros((6, 6), dtype=np.uint8)
    alpha_c[1, 4] = 255
    mask = np.full((6, 6), 255, dtype=np.uint8)
    original = anchor(0, 1.0)
    transformed_rows = transform_anchor_poses([original], SpatialTransform(dx=2.0, dy=0.0))
    final_rows = transform_anchor_poses([original], SpatialTransform(dx=2.0, dy=1.0))
    score = aggregate_score(
        name="candidate",
        transform=SpatialTransform(dx=2.0, dy=0.0),
        smoothing_passes=0,
        frame_scores=[score_alpha(alpha, mask) for alpha in (alpha_a, alpha_b, alpha_c)],
        alphas=[alpha_a, alpha_b, alpha_c],
        expected_anchor_rows=transformed_rows,
        final_pose_rows=final_rows,
    )
    assert score.temporal_centroid_pop > 0
    assert score.temporal_mass_pop > 0
    assert score.temporal_pop_score > 0
    assert score.anchor_landmark_rmse == pytest.approx(np.sqrt(0.5))
    assert score.objective < score.spatial_objective


def test_aggregate_score_rejects_duplicate_expected_anchor_frames() -> None:
    alpha_a = np.zeros((6, 6), dtype=np.uint8)
    alpha_a[1, 1] = 255
    alpha_b = np.zeros((6, 6), dtype=np.uint8)
    alpha_b[1, 2] = 255
    alpha_c = np.zeros((6, 6), dtype=np.uint8)
    alpha_c[1, 4] = 255
    mask = np.full((6, 6), 255, dtype=np.uint8)
    original = anchor(0, 1.0)
    final_rows = transform_anchor_poses([original], SpatialTransform())
    with pytest.raises(InsertionError, match="duplicate expected anchor frame: 0"):
        aggregate_score(
            name="candidate",
            transform=SpatialTransform(),
            smoothing_passes=0,
            frame_scores=[score_alpha(alpha, mask) for alpha in (alpha_a, alpha_b, alpha_c)],
            alphas=[alpha_a, alpha_b, alpha_c],
            expected_anchor_rows=[original, dict(original)],
            final_pose_rows=final_rows,
        )
