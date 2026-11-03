"""Bounded post-mask insertion iteration contracts.

This module owns the numerical part of a repeatable replacement loop:
contiguous landmark interpolation, bounded temporal relaxation, alpha/mask
scoring, and a small spatial-transform search.  It deliberately does not own
Blender, MediaPipe, VOID, provider APIs, or media rights.  Those remain
optional execution lanes around these deterministic contracts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

LANDMARK_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

PosePoint = list[float] | None
PosePoints = dict[str, PosePoint]
PoseRow = dict[str, Any]


class InsertionError(RuntimeError):
    """A post-mask iteration input or score is invalid."""


@dataclass(frozen=True)
class SpatialTransform:
    """A two-dimensional similarity transform in image pixels."""

    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.dx) or not np.isfinite(self.dy):
            raise InsertionError(f"translation must be finite: {self}")
        if not np.isfinite(self.scale) or not 0.5 <= self.scale <= 2.0:
            raise InsertionError(f"scale must be finite and in [0.5, 2.0]: {self}")

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.scale + self.dx, y * self.scale + self.dy)


@dataclass(frozen=True)
class AlphaScore:
    """Spatial agreement between one raw render alpha and target mask."""

    intersection: int
    render_pixels: int
    mask_pixels: int
    outside_mask: int
    iou: float
    precision: float
    mask_coverage: float


@dataclass(frozen=True)
class IterationScore:
    """Aggregate score for one bounded render candidate."""

    name: str
    transform: SpatialTransform
    smoothing_passes: int
    mean_iou: float
    mean_precision: float
    mean_mask_coverage: float
    spatial_objective: float
    temporal_centroid_pop: float
    temporal_mass_pop: float
    temporal_pop_score: float
    anchor_landmark_rmse: float
    objective: float


def _validate_points(points: dict[str, list[float] | None]) -> None:
    missing = set(LANDMARK_NAMES) - set(points)
    if missing:
        raise InsertionError(f"pose lacks landmarks: {sorted(missing)}")
    for name, value in points.items():
        if value is None:
            continue
        if len(value) != 2 or not all(np.isfinite(item) for item in value):
            raise InsertionError(f"invalid landmark {name}: {value!r}")


def _row_frame(row: dict[str, object]) -> int:
    value = row.get("frame")
    if isinstance(value, bool) or not isinstance(value, int):
        raise InsertionError(f"pose frame must be an integer, got {value!r}")
    return value


def _row_points(row: dict[str, object]) -> PosePoints:
    value = row.get("points")
    if not isinstance(value, dict):
        raise InsertionError(f"pose points must be an object, got {value!r}")
    result: PosePoints = {}
    for name, item in value.items():
        if name not in LANDMARK_NAMES:
            continue
        if item is None:
            result[name] = None
        elif isinstance(item, list) and len(item) == 2:
            try:
                result[name] = [float(item[0]), float(item[1])]
            except (TypeError, ValueError) as error:
                raise InsertionError(f"invalid landmark {name}: {item!r}") from error
        else:
            raise InsertionError(f"invalid landmark {name}: {item!r}")
    _validate_points(result)
    return result


def contiguous_poses(
    anchor_rows: Iterable[PoseRow],
    *,
    frame_count: int,
    smoothing_passes: int,
) -> list[PoseRow]:
    """Interpolate sparse anchor poses into one contiguous frame-indexed series.

    Missing channels interpolate independently with a nonuniform Catmull-Rom
    Hermite spline between the nearest prior and following observed anchor.
    Boundary gaps hold the nearest observed value.  This preserves every anchor
    while avoiding the velocity discontinuities of piecewise-linear poses.
    A bounded relaxation then smooths only non-anchor frames while keeping all
    supplied anchor values fixed, so smoothing cannot silently rewrite an input.
    """
    rows = list(anchor_rows)
    if not rows:
        raise InsertionError("contiguous pose series requires at least one anchor")
    if frame_count <= 0:
        raise InsertionError(f"frame_count must be positive, got {frame_count}")
    if smoothing_passes < 0:
        raise InsertionError(f"smoothing_passes must be nonnegative, got {smoothing_passes}")

    anchors: dict[int, dict[str, list[float] | None]] = {}
    for row in rows:
        frame = _row_frame(row)
        if frame in anchors:
            raise InsertionError(f"duplicate pose anchor frame: {frame}")
        if not 0 <= frame < frame_count:
            raise InsertionError(f"anchor frame {frame} outside 0..{frame_count - 1}")
        points = _row_points(row)
        anchors[frame] = points
    frames = sorted(anchors)

    series: list[PoseRow] = []
    for frame in range(frame_count):
        points: dict[str, list[float] | None] = {}
        for name in LANDMARK_NAMES:
            if frame in anchors:
                anchor_point = anchors[frame][name]
                if anchor_point is not None:
                    points[name] = list(anchor_point)
                    continue
            prior = max(
                (
                    anchor
                    for anchor in frames
                    if anchor < frame and anchors[anchor][name] is not None
                ),
                default=None,
            )
            following = min(
                (
                    anchor
                    for anchor in frames
                    if anchor > frame and anchors[anchor][name] is not None
                ),
                default=None,
            )
            if prior is not None and following is not None:
                left_value = anchors[prior][name]
                right_value = anchors[following][name]
                if left_value is None or right_value is None:
                    raise InsertionError("selected anchor unexpectedly lacked a finite landmark")
                left = np.asarray(left_value, dtype=np.float64)
                right = np.asarray(right_value, dtype=np.float64)
                duration = float(following - prior)
                amount = (frame - prior) / duration
                prior_index = frames.index(prior)
                following_index = frames.index(following)
                before = frames[prior_index - 1] if prior_index > 0 else None
                after = frames[following_index + 1] if following_index + 1 < len(frames) else None
                if before is not None and anchors[before][name] is not None:
                    before_value = np.asarray(anchors[before][name], dtype=np.float64)
                    left_tangent = (right - before_value) / (following - before)
                else:
                    left_tangent = (right - left) / duration
                if after is not None and anchors[after][name] is not None:
                    after_value = np.asarray(anchors[after][name], dtype=np.float64)
                    right_tangent = (after_value - left) / (after - prior)
                else:
                    right_tangent = (right - left) / duration
                amount_sq = amount * amount
                amount_cube = amount_sq * amount
                value = (
                    (2.0 * amount_cube - 3.0 * amount_sq + 1.0) * left
                    + (amount_cube - 2.0 * amount_sq + amount) * duration * left_tangent
                    + (-2.0 * amount_cube + 3.0 * amount_sq) * right
                    + (amount_cube - amount_sq) * duration * right_tangent
                )
                points[name] = [round(float(value[0]), 6), round(float(value[1]), 6)]
            elif prior is not None or following is not None:
                source = prior if prior is not None else following
                if source is None:
                    raise InsertionError("boundary hold did not select an anchor")
                boundary_value = anchors[source][name]
                if boundary_value is None:
                    raise InsertionError("boundary hold selected a missing landmark")
                points[name] = list(boundary_value)
            else:
                points[name] = None
        series.append({"frame": frame, "points": points})

    anchor_set = set(anchors)
    for _ in range(smoothing_passes):
        relaxed = [{**row, "points": dict(row["points"])} for row in series]
        for frame in range(frame_count):
            if frame in anchor_set:
                continue
            current_points = series[frame]["points"]
            for name in LANDMARK_NAMES:
                if current_points[name] is None:
                    continue
                previous = series[frame - 1]["points"][name] if frame > 0 else None
                following = series[frame + 1]["points"][name] if frame + 1 < frame_count else None
                neighbors = [item for item in (previous, following) if item is not None]
                if not neighbors:
                    continue
                current = np.asarray(current_points[name], dtype=np.float64)
                neighbor_mean = np.mean(
                    [np.asarray(item, dtype=np.float64) for item in neighbors], axis=0
                )
                value = 0.65 * current + 0.35 * neighbor_mean
                relaxed[frame]["points"][name] = [
                    round(float(value[0]), 6),
                    round(float(value[1]), 6),
                ]
        series = relaxed
    return series


def transform_anchor_poses(
    rows: Iterable[PoseRow],
    transform: SpatialTransform,
) -> list[PoseRow]:
    """Apply a spatial transform to every finite landmark in sparse anchors."""
    result: list[PoseRow] = []
    for row in rows:
        points = _row_points(row)
        transformed: dict[str, list[float] | None] = {}
        for name, value in points.items():
            if value is None:
                transformed[name] = None
                continue
            transformed[name] = list(transform.apply(float(value[0]), float(value[1])))
        result.append(dict(row, points=transformed))
    return result


def transform_alpha(alpha: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    """Transform an RGBA render alpha with a finite, invertible similarity."""
    if alpha.ndim != 2 or alpha.shape[0] <= 0 or alpha.shape[1] <= 0:
        raise InsertionError(f"alpha must be nonempty and two-dimensional, got {alpha.shape}")
    inverse_scale = 1.0 / transform.scale
    matrix = (
        inverse_scale,
        0.0,
        -transform.dx * inverse_scale,
        0.0,
        inverse_scale,
        -transform.dy * inverse_scale,
    )
    image = Image.fromarray(alpha.astype(np.uint8), mode="L")
    warped = image.transform(
        image.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BILINEAR,
        fill=0,
    )
    return np.asarray(warped, dtype=np.uint8)


def score_alpha(alpha: np.ndarray, mask: np.ndarray, *, threshold: int = 10) -> AlphaScore:
    """Compute mask agreement without treating a gated subset as IoU."""
    if alpha.shape != mask.shape:
        raise InsertionError(f"alpha/mask shape mismatch: {alpha.shape} != {mask.shape}")
    render = alpha > threshold
    target = mask > threshold
    intersection = int(np.count_nonzero(render & target))
    render_pixels = int(np.count_nonzero(render))
    mask_pixels = int(np.count_nonzero(target))
    union = int(np.count_nonzero(render | target))
    if render_pixels == 0 or mask_pixels == 0 or union == 0:
        raise InsertionError("cannot score an empty render or target mask")
    return AlphaScore(
        intersection=intersection,
        render_pixels=render_pixels,
        mask_pixels=mask_pixels,
        outside_mask=int(np.count_nonzero(render & ~target)),
        iou=intersection / union,
        precision=intersection / render_pixels,
        mask_coverage=intersection / mask_pixels,
    )


def spatial_objective(score: AlphaScore) -> float:
    """Prioritize in-render precision while still rewarding target coverage.

    The paired ensemble mask is known to be loose.  IoU alone therefore rewards
    an oversized blob.  Precision receives the largest weight, IoU mediates
    shape overlap, and bounded coverage discourages a tiny conservative render.
    """
    return 0.55 * score.precision + 0.35 * score.iou + 0.10 * score.mask_coverage


def optimize_transform(
    alphas: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    translations: Iterable[tuple[float, float]],
    scales: Iterable[float],
) -> tuple[SpatialTransform, list[AlphaScore]]:
    """Exhaustively search a bounded transform grid and maximize spatial score."""
    if len(alphas) != len(masks) or not alphas:
        raise InsertionError("transform search requires equal, nonempty alpha/mask lists")
    scale_items = list(scales)
    translation_items = list(translations)
    if not scale_items or not translation_items:
        raise InsertionError("transform search requires scales and translations")
    for alpha, mask in zip(alphas, masks, strict=True):
        if np.count_nonzero(alpha > 10) == 0:
            raise InsertionError("transform search requires nonempty source alphas")
        if np.count_nonzero(mask > 10) == 0:
            raise InsertionError("transform search requires nonempty target masks")
    best: tuple[float, SpatialTransform, list[AlphaScore]] | None = None
    for scale in scale_items:
        for dx, dy in translation_items:
            transform = SpatialTransform(dx=dx, dy=dy, scale=scale)
            transformed_alphas = [transform_alpha(alpha, transform) for alpha in alphas]
            if any(np.count_nonzero(alpha > 10) == 0 for alpha in transformed_alphas):
                continue
            scores = [
                score_alpha(alpha, mask)
                for alpha, mask in zip(transformed_alphas, masks, strict=True)
            ]
            objective = float(np.mean([spatial_objective(score) for score in scores]))
            if best is None or objective > best[0]:
                best = (objective, transform, scores)
    if best is None:
        raise InsertionError("transform search produced no candidate")
    return best[1], best[2]


def aggregate_score(
    *,
    name: str,
    transform: SpatialTransform,
    smoothing_passes: int,
    frame_scores: list[AlphaScore],
    alphas: list[np.ndarray],
    expected_anchor_rows: list[PoseRow],
    final_pose_rows: list[PoseRow],
) -> IterationScore:
    """Combine spatial agreement, temporal stability, and anchor preservation."""
    if len(frame_scores) != len(alphas) or not frame_scores:
        raise InsertionError("aggregate scoring requires equal, nonempty frame scores/alphas")
    if not expected_anchor_rows:
        raise InsertionError("anchor preservation requires at least one anchor")
    final_rows: dict[int, PosePoints] = {}
    for row in final_pose_rows:
        frame = _row_frame(row)
        if frame in final_rows:
            raise InsertionError(f"duplicate final pose frame: {frame}")
        final_rows[frame] = _row_points(row)
    mean_iou = float(np.mean([score.iou for score in frame_scores]))
    mean_precision = float(np.mean([score.precision for score in frame_scores]))
    mean_coverage = float(np.mean([score.mask_coverage for score in frame_scores]))
    spatial = 0.55 * mean_precision + 0.35 * mean_iou + 0.10 * mean_coverage

    centroids: list[np.ndarray] = []
    masses: list[float] = []
    for alpha in alphas:
        coordinates = np.argwhere(alpha > 10)
        if len(coordinates) == 0:
            raise InsertionError("temporal scoring requires nonempty alpha on every frame")
        centroids.append(coordinates.mean(axis=0))
        masses.append(float(np.count_nonzero(alpha > 10)))
    centroid_steps = [
        float(np.linalg.norm(centroids[index] - centroids[index - 1]))
        for index in range(1, len(centroids))
    ]
    mass_steps = [abs(masses[index] - masses[index - 1]) for index in range(1, len(masses))]
    centroid_scale = max(float(np.median(centroid_steps)), 1.0) if centroid_steps else 1.0
    mass_scale = max(float(np.median(mass_steps)), 1.0) if mass_steps else 1.0
    centroid_pops: list[float] = []
    mass_pops: list[float] = []
    for index in range(1, len(centroids) - 1):
        second = centroids[index + 1] - 2.0 * centroids[index] + centroids[index - 1]
        centroid_pops.append(min(1.0, float(np.linalg.norm(second)) / centroid_scale))
        mass_second = masses[index + 1] - 2.0 * masses[index] + masses[index - 1]
        mass_pops.append(min(1.0, abs(float(mass_second)) / mass_scale))
    centroid_pop = float(np.mean(centroid_pops)) if centroid_pops else 0.0
    mass_pop = float(np.mean(mass_pops)) if mass_pops else 0.0
    temporal_pop = 0.5 * centroid_pop + 0.5 * mass_pop

    squared_errors: list[float] = []
    expected_frames: set[int] = set()
    for expected_row in expected_anchor_rows:
        frame = _row_frame(expected_row)
        if frame in expected_frames:
            raise InsertionError(f"duplicate expected anchor frame: {frame}")
        expected_frames.add(frame)
        expected_points = _row_points(expected_row)
        final_points = final_rows.get(frame)
        if final_points is None:
            raise InsertionError(f"anchor frame {frame} is absent from the final pose series")
        for landmark_name in LANDMARK_NAMES:
            target = expected_points[landmark_name]
            actual = final_points[landmark_name]
            if target is None or actual is None:
                continue
            squared_errors.extend(
                (float(actual[index]) - float(target[index])) ** 2 for index in range(2)
            )
    anchor_rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0

    # A second-difference pop is penalized directly; ordinary first-frame
    # motion is not.  This lets a moving contiguous figure remain in motion.
    objective = spatial - 0.25 * temporal_pop - 0.002 * anchor_rmse
    return IterationScore(
        name=name,
        transform=transform,
        smoothing_passes=smoothing_passes,
        mean_iou=mean_iou,
        mean_precision=mean_precision,
        mean_mask_coverage=mean_coverage,
        spatial_objective=spatial,
        temporal_centroid_pop=centroid_pop,
        temporal_mass_pop=mass_pop,
        temporal_pop_score=temporal_pop,
        anchor_landmark_rmse=anchor_rmse,
        objective=objective,
    )
