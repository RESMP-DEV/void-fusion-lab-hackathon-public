from __future__ import annotations

import math
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from samurai.media import VideoInfo, write_gray_video
from samurai.pass2_metrics import (
    DEFAULT_CHANGED_THRESHOLD,
    FrameTimestamps,
    Pass2MetricsError,
    check_alignment,
    check_timestamp_alignment,
    compare_sequences,
    fraction_text,
    probe_frame_timestamps,
    region_metrics,
    roi_from_gray,
    temporal_stability,
)
from samurai.quadmask import AFFECTED, REMOVE

SIZE = 8
INSERTED = (slice(2, 5), slice(2, 5))
CLEANUP = (slice(1, 6), slice(1, 6))
FRAME_COUNT = 5


def frames(value: int = 20) -> list[np.ndarray]:
    return [np.full((SIZE, SIZE, 3), value, dtype=np.uint8) for _ in range(FRAME_COUNT)]


def inserted_masks() -> list[np.ndarray]:
    mask = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    mask[INSERTED] = 0
    return [mask.copy() for _ in range(FRAME_COUNT)]


def cleanup_masks() -> list[np.ndarray]:
    mask = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    mask[CLEANUP] = 0
    return [mask.copy() for _ in range(FRAME_COUNT)]


def with_block(
    frames_: list[np.ndarray], value: int, region: tuple[slice, slice]
) -> list[np.ndarray]:
    result = [frame.copy() for frame in frames_]
    for frame in result:
        frame[region] = value
    return result


def with_ring_change(frames_: list[np.ndarray], value: int) -> list[np.ndarray]:
    """Change only the cleanup ring, leaving the inserted ROI untouched."""
    result = with_block(frames_, value, CLEANUP)
    for frame in result:
        frame[INSERTED] = 20
    return result


def test_identical_sequences_measure_zero_error_and_no_temporal_difference() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=frames(),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
    )

    assert comparison.object_preservation.mae == 0.0
    assert comparison.object_preservation.mse == 0.0
    assert comparison.object_preservation.psnr_db is None
    assert comparison.object_preservation.changed_ratio == 0.0
    assert comparison.cleanup_ring.mae == 0.0
    assert comparison.background.mae == 0.0
    assert comparison.temporal.no_changed_pixels is True
    assert comparison.temporal.exactly_identical is True
    assert comparison.temporal.temporal_pop_score == 0.0
    assert comparison.temporal.mass_pop == 0.0
    assert comparison.temporal.centroid_pop == 0.0
    assert comparison.temporal.per_frame_mass == [0] * FRAME_COUNT


def test_object_change_is_measured_only_inside_the_inserted_roi() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=with_block(frames(), 220, INSERTED),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
    )

    assert comparison.object_preservation.mae == pytest.approx(200.0)
    assert comparison.object_preservation.mse == pytest.approx(200.0**2)
    assert comparison.object_preservation.psnr_db == pytest.approx(
        10.0 * math.log10(255.0**2 / 200.0**2)
    )
    assert comparison.object_preservation.changed_ratio == pytest.approx(1.0)
    assert comparison.object_preservation.pixels == 9 * FRAME_COUNT
    assert comparison.cleanup_ring.mae == 0.0
    assert comparison.background.mae == 0.0


def test_cleanup_ring_excludes_the_inserted_roi() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=with_ring_change(frames(), 220),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=REMOVE,
    )

    assert comparison.cleanup_ring.pixels == 16 * FRAME_COUNT
    assert comparison.cleanup_ring.mae == pytest.approx(200.0)
    assert comparison.cleanup_ring.changed_ratio == pytest.approx(1.0)
    assert comparison.object_preservation.mae == 0.0
    assert comparison.background.mae == 0.0


def test_background_region_covers_every_pixel_outside_both_rois() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=with_block(frames(), 220, (slice(0, 1), slice(0, 1))),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=REMOVE,
    )

    assert comparison.background.pixels == 39 * FRAME_COUNT
    assert comparison.background.mae == pytest.approx(200.0 / 39)
    assert comparison.background.changed_ratio == pytest.approx(1.0 / 39)
    assert (
        comparison.object_preservation.pixels
        + comparison.cleanup_ring.pixels
        + comparison.background.pixels
        == SIZE * SIZE * FRAME_COUNT
    )


def test_changed_ratio_follows_the_threshold() -> None:
    quiet = compare_sequences(
        baseline=frames(),
        pass2=with_block(frames(), 25, INSERTED),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
    )
    loud = compare_sequences(
        baseline=frames(),
        pass2=with_block(frames(), 25, INSERTED),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        changed_threshold=2.0,
    )

    assert quiet.object_preservation.changed_ratio == 0.0
    assert quiet.object_preservation.mae == pytest.approx(5.0)
    assert loud.object_preservation.changed_ratio == pytest.approx(1.0)


def test_temporal_stability_penalizes_a_flickering_difference() -> None:
    steady = with_block(frames(), 220, INSERTED)
    comparison = compare_sequences(
        baseline=frames(),
        pass2=steady,
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
    )

    flickering = frames()
    for index, frame in enumerate(flickering):
        region = INSERTED if index % 2 == 0 else (slice(0, 3), slice(0, 3))
        frame[region] = 220
    flicker = compare_sequences(
        baseline=frames(),
        pass2=flickering,
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
    )

    assert comparison.temporal.temporal_pop_score == 0.0
    assert comparison.temporal.centroid_pop == 0.0
    assert comparison.temporal.mass_pop == 0.0
    assert flicker.temporal.centroid_pop > 0.0
    assert flicker.temporal.temporal_pop_score > comparison.temporal.temporal_pop_score


def test_temporal_stability_penalizes_a_missing_difference() -> None:
    partial = frames()
    partial[2][INSERTED] = 220
    comparison = compare_sequences(
        baseline=frames(),
        pass2=partial,
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
    )

    assert comparison.temporal.frames_with_difference == 1
    assert comparison.temporal.mass_pop == 1.0
    assert comparison.temporal.centroid_pop == 1.0
    assert comparison.temporal.temporal_pop_score == 1.0
    assert comparison.temporal.no_changed_pixels is False
    assert comparison.temporal.exactly_identical is False


def test_temporal_stability_scores_an_interior_disappearance_as_a_maximal_pop() -> None:
    maps = [np.zeros((1, 4), dtype=np.float64) for _ in range(96)]
    for map_frame in maps:
        map_frame[0, :] = 64.0
    maps[50][0, :] = 0.0

    stability = temporal_stability(maps, changed_threshold=10.0)

    assert stability.per_frame_mass == [4] * 50 + [0] + [4] * 45
    assert stability.frames_with_difference == 95
    assert stability.mass_pop == 1.0
    assert stability.centroid_pop == 1.0
    assert stability.temporal_pop_score == 1.0


def test_temporal_stability_scores_an_alternating_single_pixel_as_a_maximal_pop() -> None:
    maps = [np.zeros((1, 4), dtype=np.float64) for _ in range(96)]
    for index, map_frame in enumerate(maps):
        if index % 2 == 0:
            map_frame[0, 0] = 64.0

    stability = temporal_stability(maps, changed_threshold=10.0)

    assert stability.per_frame_mass == [1 if index % 2 == 0 else 0 for index in range(96)]
    assert stability.frames_with_difference == 48
    assert stability.mass_pop == 1.0
    assert stability.centroid_pop == 1.0
    assert stability.temporal_pop_score == 1.0


def test_temporal_stability_does_not_dilate_an_isolated_centroid_pop() -> None:
    maps = [np.zeros((1, 100), dtype=np.float64) for _ in range(96)]
    for map_frame in maps:
        map_frame[0, 0] = 1.0
    maps[50][0, 0] = 0.0
    maps[50][0, 50] = 1.0

    stability = temporal_stability(maps, changed_threshold=0.5)

    assert stability.frames_with_difference == len(maps)
    assert stability.centroid_pop == 1.0
    assert stability.temporal_pop_score == 0.5


def ring_change_after_frame_zero(value: int) -> list[np.ndarray]:
    """Change the cleanup ring only on frames 1..N, leaving frame zero untouched."""
    changed = with_ring_change(frames(), value)
    changed[0][CLEANUP] = 20
    return changed


def test_structural_lead_in_keeps_a_steady_cleanup_out_of_the_dropout_score() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=ring_change_after_frame_zero(220),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=AFFECTED,
    )

    assert comparison.temporal.structural_lead_in == 1
    assert comparison.temporal.measured_frames == FRAME_COUNT - 1
    assert comparison.temporal.lead_in_changed_pixels == 0
    assert comparison.temporal.lead_in_violation is False
    assert comparison.temporal.frames_with_difference == FRAME_COUNT - 1
    assert comparison.temporal.per_frame_mass == [0] + [16] * (FRAME_COUNT - 1)
    assert comparison.temporal.mass_pop == 0.0
    assert comparison.temporal.centroid_pop == 0.0
    assert comparison.temporal.temporal_pop_score == 0.0


def test_structural_lead_in_reports_a_changed_frame_zero_without_scoring_a_pop() -> None:
    only_first_frame = frames()
    only_first_frame[0][CLEANUP] = 220

    comparison = compare_sequences(
        baseline=frames(),
        pass2=only_first_frame,
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=AFFECTED,
    )

    assert comparison.temporal.structural_lead_in == 1
    assert comparison.temporal.measured_frames == FRAME_COUNT - 1
    assert comparison.temporal.lead_in_changed_pixels == 25
    assert comparison.temporal.lead_in_violation is True
    assert comparison.temporal.frames_with_difference == 1
    assert comparison.temporal.exactly_identical is False
    assert comparison.temporal.mass_pop == 0.0
    assert comparison.temporal.centroid_pop == 0.0
    assert comparison.temporal.temporal_pop_score == 0.0


def test_remove_cleanup_has_no_structural_lead_in() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=with_ring_change(frames(), 220),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=REMOVE,
    )

    assert comparison.temporal.structural_lead_in == 0
    assert comparison.temporal.measured_frames == FRAME_COUNT
    assert comparison.temporal.lead_in_changed_pixels == 0
    assert comparison.temporal.per_frame_mass == [16] * FRAME_COUNT
    assert comparison.temporal.mass_pop == 0.0
    assert comparison.temporal.centroid_pop == 0.0
    assert comparison.temporal.temporal_pop_score == 0.0


def test_remove_cleanup_scores_an_unchanged_frame_zero_as_a_dropout() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=ring_change_after_frame_zero(220),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=REMOVE,
    )

    assert comparison.temporal.structural_lead_in == 0
    assert comparison.temporal.measured_frames == FRAME_COUNT
    assert comparison.temporal.per_frame_mass == [0] + [16] * (FRAME_COUNT - 1)
    assert comparison.temporal.mass_pop == 1.0
    assert comparison.temporal.centroid_pop == 1.0
    assert comparison.temporal.temporal_pop_score == 1.0


def test_temporal_stability_requires_three_measured_frames_after_the_lead_in() -> None:
    with pytest.raises(Pass2MetricsError, match="at least 3 measured frames"):
        temporal_stability(
            [np.zeros((4, 4), dtype=np.float64) for _ in range(3)],
            changed_threshold=DEFAULT_CHANGED_THRESHOLD,
            structural_lead_in=1,
        )


def test_temporal_stability_does_not_saturate_on_one_pixel_boundary_jitter() -> None:
    maps = [np.zeros((1, 4), dtype=np.float64) for _ in range(96)]
    for map_frame in maps:
        map_frame[0, 0] = 64.0
    for index, map_frame in enumerate(maps):
        if index % 2 == 0:
            map_frame[0, 1] = 64.0

    stability = temporal_stability(maps, changed_threshold=10.0)

    assert stability.per_frame_mass == [2 if index % 2 == 0 else 1 for index in range(96)]
    assert stability.mass_pop == 0.5
    assert stability.temporal_pop_score <= 0.5


def test_temporal_mass_scale_has_a_two_pixel_noise_floor() -> None:
    maps = [np.zeros((1, 4), dtype=np.float64) for _ in range(96)]
    for map_frame in maps:
        map_frame[0, 0] = 1.0
    for map_frame in maps[40:]:
        map_frame[0, 1] = 1.0

    stability = temporal_stability(maps, changed_threshold=0.5)

    assert stability.per_frame_mass == [1] * 40 + [2] * 56
    assert stability.mass_pop == 0.25


def test_compare_sequences_consumes_lazy_iterables_without_materializing_the_clip() -> None:
    def lazy(items: list[np.ndarray]) -> Iterator[np.ndarray]:
        yield from items

    list_based = compare_sequences(
        baseline=frames(),
        pass2=ring_change_after_frame_zero(220),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=AFFECTED,
    )
    stream_based = compare_sequences(
        baseline=lazy(frames()),
        pass2=lazy(ring_change_after_frame_zero(220)),
        inserted_masks=lazy(inserted_masks()),
        cleanup_masks=lazy(cleanup_masks()),
        cleanup_value=AFFECTED,
    )

    assert stream_based == list_based


def test_lazy_sequences_with_different_lengths_fail_closed() -> None:
    def lazy(items: list[np.ndarray]) -> Iterator[np.ndarray]:
        yield from items

    with pytest.raises(Pass2MetricsError, match="equal, nonempty"):
        compare_sequences(
            baseline=lazy(frames()),
            pass2=lazy(frames()[:-1]),
            inserted_masks=lazy(inserted_masks()),
            cleanup_masks=lazy(cleanup_masks()),
        )


def test_empty_roi_fails_closed_instead_of_reporting_a_number() -> None:
    blank = [np.full((SIZE, SIZE), 255, dtype=np.uint8) for _ in range(FRAME_COUNT)]
    with pytest.raises(Pass2MetricsError, match="empty in every frame"):
        compare_sequences(
            baseline=frames(),
            pass2=with_block(frames(), 220, INSERTED),
            inserted_masks=blank,
            cleanup_masks=cleanup_masks(),
        )


def test_frame_count_and_geometry_mismatches_fail_closed() -> None:
    with pytest.raises(Pass2MetricsError, match="equal, nonempty"):
        compare_sequences(
            baseline=frames(),
            pass2=frames()[:-1],
            inserted_masks=inserted_masks(),
            cleanup_masks=cleanup_masks(),
        )
    with pytest.raises(Pass2MetricsError, match="does not match video"):
        compare_sequences(
            baseline=frames(),
            pass2=frames(),
            inserted_masks=[np.full((4, 4), 0, dtype=np.uint8) for _ in range(FRAME_COUNT)],
            cleanup_masks=cleanup_masks(),
        )
    with pytest.raises(Pass2MetricsError, match="geometry mismatch"):
        compare_sequences(
            baseline=frames(),
            pass2=[frame[:4, :4].copy() for frame in frames()],
            inserted_masks=inserted_masks(),
            cleanup_masks=cleanup_masks(),
        )


def test_region_metric_rejects_a_mismatched_region_shape() -> None:
    with pytest.raises(Pass2MetricsError, match="does not match region"):
        region_metrics(
            "broken",
            [np.zeros((4, 4), dtype=np.float64)],
            [np.zeros((4, 4), dtype=np.float64)],
            [np.ones((8, 8), dtype=bool)],
            changed_threshold=DEFAULT_CHANGED_THRESHOLD,
        )


def test_roi_selection_uses_the_black_region_convention() -> None:
    frame = np.array([[0, 127, 128, 255]], dtype=np.uint8)

    assert roi_from_gray(frame).tolist() == [[True, True, False, False]]


def test_temporal_stability_requires_at_least_one_frame() -> None:
    with pytest.raises(Pass2MetricsError, match="at least one frame"):
        temporal_stability([], changed_threshold=DEFAULT_CHANGED_THRESHOLD)


def info(
    width: int = 16, height: int = 8, frames: int = FRAME_COUNT, fps: float = 12.0
) -> VideoInfo:
    return VideoInfo(width=width, height=height, fps=fps, frames=frames)


def test_alignment_fails_closed_on_geometry_frame_and_rate_mismatch() -> None:
    with pytest.raises(Pass2MetricsError, match="geometry mismatch"):
        check_alignment(baseline=info(), pass2=info(width=8), inserted=info(), cleanup=info())
    with pytest.raises(Pass2MetricsError, match="frame count mismatch"):
        check_alignment(baseline=info(), pass2=info(frames=3), inserted=info(), cleanup=info())
    with pytest.raises(Pass2MetricsError, match="frame rate mismatch"):
        check_alignment(baseline=info(), pass2=info(fps=24.0), inserted=info(), cleanup=info())


def test_alignment_reports_matched_inputs() -> None:
    alignment = check_alignment(baseline=info(), pass2=info(), inserted=info(), cleanup=info())

    assert alignment.aligned is True
    assert alignment.dimensions_aligned is True
    assert alignment.frames_aligned is True
    assert alignment.fps_aligned is True
    assert alignment.fps_delta == 0.0
    assert alignment.baseline.frames == FRAME_COUNT


def test_affected_cleanup_value_reports_the_first_frame_ring_as_background() -> None:
    affected = compare_sequences(
        baseline=frames(),
        pass2=with_ring_change(frames(), 220),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=AFFECTED,
    )
    removed = compare_sequences(
        baseline=frames(),
        pass2=with_ring_change(frames(), 220),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
        cleanup_value=REMOVE,
    )

    assert affected.cleanup_ring.pixels == 16 * (FRAME_COUNT - 1)
    assert affected.cleanup_ring.frames_measured == FRAME_COUNT - 1
    assert affected.cleanup_ring.frames_total == FRAME_COUNT
    assert affected.background.pixels == removed.background.pixels + 16
    assert affected.background.mae > 0.0
    assert removed.background.mae == 0.0
    assert (
        affected.object_preservation.pixels
        + affected.cleanup_ring.pixels
        + affected.background.pixels
        == SIZE * SIZE * FRAME_COUNT
    )


def test_cleanup_value_must_be_remove_or_affected() -> None:
    with pytest.raises(Pass2MetricsError, match="cleanup_value must be"):
        compare_sequences(
            baseline=frames(),
            pass2=frames(),
            inserted_masks=inserted_masks(),
            cleanup_masks=cleanup_masks(),
            cleanup_value=63,
        )


def test_difference_below_the_threshold_is_not_exactly_identical() -> None:
    comparison = compare_sequences(
        baseline=frames(),
        pass2=with_block(frames(), 25, INSERTED),
        inserted_masks=inserted_masks(),
        cleanup_masks=cleanup_masks(),
    )

    assert comparison.temporal.no_changed_pixels is True
    assert comparison.temporal.exactly_identical is False
    assert comparison.object_preservation.mae == pytest.approx(5.0)


def test_non_finite_thresholds_are_rejected() -> None:
    with pytest.raises(Pass2MetricsError, match="changed_threshold must be finite"):
        compare_sequences(
            baseline=frames(),
            pass2=frames(),
            inserted_masks=inserted_masks(),
            cleanup_masks=cleanup_masks(),
            changed_threshold=math.nan,
        )
    with pytest.raises(Pass2MetricsError, match="changed_threshold must be finite"):
        region_metrics(
            "broken",
            [np.zeros((4, 4), dtype=np.float64)],
            [np.zeros((4, 4), dtype=np.float64)],
            [np.ones((4, 4), dtype=bool)],
            changed_threshold=math.inf,
        )
    with pytest.raises(Pass2MetricsError, match="fps_tolerance must be finite"):
        check_alignment(
            baseline=info(),
            pass2=info(),
            inserted=info(),
            cleanup=info(),
            fps_tolerance=math.inf,
        )


def timestamps(name: str, pts: tuple[Fraction, ...], *, variable: bool = False) -> FrameTimestamps:
    """Build a timestamp record for one comparison input."""
    deltas = [pts[index + 1] - pts[index] for index in range(len(pts) - 1)]
    return FrameTimestamps(
        name=name,
        count=len(pts),
        pts=pts,
        variable_frame_rate=variable,
        cfr_delta=None if variable or not deltas else deltas[0],
        digest="0" * 64,
    )


def test_timestamp_alignment_accepts_one_shared_timeline() -> None:
    pts = (Fraction(0), Fraction(1, 12), Fraction(1, 6))
    alignment = check_timestamp_alignment(
        baseline=timestamps("baseline", pts),
        pass2=timestamps("pass2", pts),
        inserted=timestamps("inserted_mask", pts),
        cleanup=timestamps("cleanup_mask", pts),
    )

    assert alignment.aligned is True
    assert alignment.pts_aligned is True
    assert alignment.frame_rate_character_aligned is True
    assert fraction_text(Fraction(1, 12)) == "1/12"


def test_timestamp_alignment_fails_closed_on_a_per_frame_mismatch() -> None:
    with pytest.raises(Pass2MetricsError, match="timestamp mismatch at frame 1"):
        check_timestamp_alignment(
            baseline=timestamps("baseline", (Fraction(0), Fraction(1, 12))),
            pass2=timestamps("pass2", (Fraction(0), Fraction(1, 12) + Fraction(1, 1000))),
            inserted=timestamps("inserted_mask", (Fraction(0), Fraction(1, 12))),
            cleanup=timestamps("cleanup_mask", (Fraction(0), Fraction(1, 12))),
        )


def test_timestamp_alignment_fails_closed_on_a_frame_rate_character_mismatch() -> None:
    constant = (Fraction(0), Fraction(1, 12), Fraction(1, 6))
    variable = (Fraction(0), Fraction(1, 12), Fraction(1, 5))
    with pytest.raises(Pass2MetricsError, match="frame rate character mismatch"):
        check_timestamp_alignment(
            baseline=timestamps("baseline", constant),
            pass2=timestamps("pass2", variable, variable=True),
            inserted=timestamps("inserted_mask", constant),
            cleanup=timestamps("cleanup_mask", constant),
        )


def test_timestamp_alignment_fails_closed_on_a_timestamp_count_mismatch() -> None:
    with pytest.raises(Pass2MetricsError, match="timestamp count mismatch"):
        check_timestamp_alignment(
            baseline=timestamps("baseline", (Fraction(0), Fraction(1, 12))),
            pass2=timestamps("pass2", (Fraction(0),)),
            inserted=timestamps("inserted_mask", (Fraction(0), Fraction(1, 12))),
            cleanup=timestamps("cleanup_mask", (Fraction(0), Fraction(1, 12))),
        )


def test_probe_frame_timestamps_reads_exact_rational_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "mask.mp4"
    write_gray_video(
        [np.full((SIZE, SIZE), 255, dtype=np.uint8) for _ in range(FRAME_COUNT)],
        video,
        fps=12.0,
    )

    probed = probe_frame_timestamps(video, name="mask")

    assert probed.count == FRAME_COUNT
    assert probed.variable_frame_rate is False
    assert probed.cfr_delta == Fraction(1, 12)
    assert probed.pts[0] == 0
    assert len(probed.digest) == 64


def test_probe_frame_timestamps_fails_closed_on_a_non_video(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video", encoding="utf-8")

    with pytest.raises(Pass2MetricsError, match="ffprobe failed"):
        probe_frame_timestamps(broken, name="broken")
