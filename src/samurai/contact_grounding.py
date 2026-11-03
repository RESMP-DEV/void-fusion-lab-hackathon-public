"""Deterministic contact and shadow constraints for a fitted 3D insertion.

A fitted insertion only reads as part of the scene when its base meets the
ground the scene already establishes and when the shadow beneath it agrees
with the scene's lighting.  This module measures those two relationships
from inputs the insertion lane already owns: per-frame lower-body
keypoints, the inserted object's region-of-interest mask, and the untouched
plate (the scene without the inserted object).

Every result is a deterministic image-space measurement:

* the support polygon is the convex hull of the person's ground-bearing foot
  footprints;
* the contact line is the least-squares line through the ground-bearing foot
  samples and stands in for the ground plane in image space;
* contact coverage reports how much of the inserted object's ground-contact
  band falls inside that support polygon;
* vertical alignment compares the object's lowest edge against the contact
  line under the object's centre;
* the light direction is a least-squares luminance-plane fit over the plate
  region in front of the object, so the brighter side of the plate names the
  light azimuth;
* the suggested contact/cast shadow mask is a deterministic tapered wedge from
  the object's base, pointing away from that light, plus a contact band that
  hugs the base.

The suggested shadow mask follows the repository's ROI-black convention: a
shadow pixel is 0 and a keep pixel is 255.  It is not a VOID quadmask and
carries none of the 0/63/127/255 quadmask values, so the VOID lane must not
read it.

Invalid geometry fails closed: ``ContactGroundingError`` is raised instead of
returning a plausible-looking number.  No renderer is invoked, no model is
called, and no quality verdict is produced.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

NON_CLAIMS = (
    "These are image-space geometric and photometric measurements of a fitted "
    "insertion; they are not a quality verdict and not an acceptance decision.",
    "The support polygon and contact line are a 2D projection proxy built from "
    "lower-body keypoints; they do not reconstruct a metric 3D ground plane.",
    "The plate-luminance light direction is a first-order gradient estimate; it "
    "does not resolve penumbra, softness, occlusion, or material response.",
    "The shadow length is a fixed width ratio, not an elevation-derived quantity.",
    "The suggested shadow mask is geometry only; no renderer was invoked and no "
    "composited frame was produced, graded, or judged.",
    "The suggested shadow mask uses the ROI-black convention (0 shadow, 255 keep); "
    "it is not a VOID quadmask and must not be consumed by the VOID lane.",
    "A plate window whose luminance varies by at most one code unit is reported "
    "as flat; that is a measurement of the plate, not a claim that the scene is "
    "evenly lit.",
)

FOOT_LANDMARKS = (
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
BODY_LANDMARKS = ("left_hip", "right_hip", "left_knee", "right_knee")

_LANDMARK_ALIASES: dict[str, str] = {
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_heel": "left_heel",
    "right_heel": "right_heel",
    "left_foot_index": "left_foot_index",
    "right_foot_index": "right_foot_index",
    "left_foot": "left_foot_index",
    "right_foot": "right_foot_index",
    "left_toe": "left_foot_index",
    "right_toe": "right_foot_index",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
}


class ContactGroundingError(RuntimeError):
    """A contact-grounding input or derived geometry is invalid."""


@dataclass(frozen=True)
class GroundingSettings:
    """Scale-relative knobs for the contact and shadow geometry."""

    foot_width_ratio: float = 0.25
    foot_depth_ratio: float = 0.08
    contact_band_ratio: float = 0.02
    max_foot_lift_ratio: float = 0.12
    object_band_ratio: float = 0.05
    vertical_tolerance_ratio: float = 0.03
    contact_thickness_ratio: float = 0.06
    contact_spread_ratio: float = 0.05
    shadow_length_ratio: float = 0.6
    shadow_spread_ratio: float = 1.0
    min_plate_samples: int = 64
    clip_shadow_to_ground: bool = False
    flat_luminance_range: float = 1.0
    min_light_confidence: float = 0.5
    contact_line_min_baseline_ratio: float = 0.05


@dataclass(frozen=True)
class KeypointFrame:
    """One frame of normalized lower-body keypoints in image pixels."""

    frame: int
    points: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class SupportPolygon:
    """The person's ground-bearing footprint and the contact line under it."""

    points: tuple[tuple[float, float], ...]
    area_px: float
    ground_y: float
    contact_slope: float
    contact_intercept: float
    foot_samples: int
    ground_samples: int
    planted_foot_ratio: float
    side_ground_levels: tuple[tuple[str, float], ...]
    leg_span_px: float
    contact_line_conditioned: bool

    def ground_y_at(self, x: float) -> float:
        """Return the contact line's y coordinate at a given x."""
        return _finite(
            self.contact_slope * x + self.contact_intercept,
            f"the contact line height at x={x}",
        )


@dataclass(frozen=True)
class LightEstimate:
    """A first-order plate-luminance light direction in image space."""

    source: str
    direction_x: float
    direction_y: float
    azimuth_degrees: float
    gradient_x: float
    gradient_y: float
    confidence: float
    samples: int
    window: tuple[int, int, int, int]


@dataclass(frozen=True)
class ShadowGeometry:
    """The suggested contact and cast shadow for one frame."""

    direction_x: float
    direction_y: float
    length_px: float
    half_width_px: float
    contact_pixels: int
    cast_pixels: int
    total_pixels: int
    above_contact_line_pixels: int


@dataclass(frozen=True)
class FrameGrounding:
    """Every contact measurement for one frame."""

    frame: int
    support: SupportPolygon
    object_bbox: tuple[int, int, int, int]
    object_bottom_y: float
    object_centroid_x: float
    object_height_px: int
    object_width_px: int
    footprint_pixels: int
    contact_pixels: int
    contact_coverage: float
    ground_y_at_object: float
    vertical_offset_px: float
    vertical_tolerance_px: float
    vertical_aligned: bool
    light: LightEstimate
    shadow: ShadowGeometry


def _landmark_name(name: str) -> str | None:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _LANDMARK_ALIASES.get(key)


def _point_pair(value: Any, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise ContactGroundingError(f"{label} must provide x and y")
        raw: Any = [value["x"], value["y"]]
    elif isinstance(value, list | tuple):
        raw = list(value)
    else:
        raise ContactGroundingError(f"{label} must be a two-element [x, y] pair")
    if len(raw) != 2:
        raise ContactGroundingError(f"{label} must be a two-element [x, y] pair")
    try:
        x = float(raw[0])
        y = float(raw[1])
    except (TypeError, ValueError, OverflowError) as error:
        raise ContactGroundingError(f"{label} coordinates must be numeric") from error
    if not math.isfinite(x) or not math.isfinite(y):
        raise ContactGroundingError(f"{label} coordinates must be finite")
    return (x, y)


def parse_keypoint_frames(payload: Any) -> list[KeypointFrame]:
    """Validate and normalize per-frame lower-body keypoints.

    The payload is either a list of rows or an object with a ``frames`` list.
    Each row names an integer ``frame`` and a ``points`` object mapping
    landmark names to ``[x, y]`` pairs or ``null``.  Unknown landmark names
    are ignored so a full-body pose file can be reused unchanged.
    """
    if isinstance(payload, dict):
        rows = payload.get("frames")
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ContactGroundingError("keypoints must contain a nonempty frames list")
    frames: list[KeypointFrame] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ContactGroundingError("each keypoint row must be an object")
        index = row.get("frame")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ContactGroundingError(f"keypoint frame must be a nonnegative integer: {index!r}")
        if index in seen:
            raise ContactGroundingError(f"duplicate keypoint frame: {index}")
        seen.add(index)
        raw_points = row.get("points")
        if not isinstance(raw_points, dict):
            raise ContactGroundingError(f"keypoint frame {index} must provide a points object")
        points: dict[str, tuple[float, float]] = {}
        for name, value in raw_points.items():
            if not isinstance(name, str):
                raise ContactGroundingError(
                    f"keypoint frame {index} has a non-string landmark name"
                )
            canonical = _landmark_name(name)
            if canonical is None:
                continue
            pair = _point_pair(value, f"keypoint frame {index} landmark {name}")
            if pair is not None:
                points[canonical] = pair
        if not any(name in FOOT_LANDMARKS for name in points):
            raise ContactGroundingError(
                f"keypoint frame {index} has no ground-bearing foot landmark"
            )
        frames.append(KeypointFrame(frame=index, points=points))
    frames.sort(key=lambda item: item.frame)
    return frames


def _cross(
    origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]
) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the convex hull of points in counter-clockwise order."""
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Return the absolute shoelace area of a closed polygon."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def rasterize_polygon(points: Sequence[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon into a boolean image using pixel-center sampling.

    Pixel ``(row, column)`` covers ``[column, column + 1) x [row, row + 1)``
    and is sampled at its centre, so the polygon's edges fall on the same
    continuous frame the keypoints use.
    """
    height, width = shape
    result = np.zeros((height, width), dtype=bool)
    if len(points) < 3:
        return result
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = max(0, int(math.floor(min(xs))))
    x_max = min(width - 1, int(math.ceil(max(xs))))
    y_min = max(0, int(math.floor(min(ys))))
    y_max = min(height - 1, int(math.ceil(max(ys))))
    count = len(points)
    for row in range(y_min, y_max + 1):
        y = row + 0.5
        crossings: list[float] = []
        for index in range(count):
            x0, y0 = points[index]
            x1, y1 = points[(index + 1) % count]
            if (y0 <= y < y1) or (y1 <= y < y0):
                ratio = (y - y0) / (y1 - y0)
                crossings.append(x0 + ratio * (x1 - x0))
        crossings.sort()
        for pair in range(0, len(crossings) - 1, 2):
            first = int(math.ceil(crossings[pair] - 0.5))
            last = int(math.floor(crossings[pair + 1] - 0.5))
            first = max(first, x_min)
            last = min(last, x_max)
            if last >= first:
                result[row, first : last + 1] = True
    return result


def validate_settings(settings: GroundingSettings) -> GroundingSettings:
    """Reject settings that would produce meaningless geometry."""
    ratios = {
        "foot_width_ratio": settings.foot_width_ratio,
        "foot_depth_ratio": settings.foot_depth_ratio,
        "contact_band_ratio": settings.contact_band_ratio,
        "object_band_ratio": settings.object_band_ratio,
        "vertical_tolerance_ratio": settings.vertical_tolerance_ratio,
        "contact_thickness_ratio": settings.contact_thickness_ratio,
        "contact_spread_ratio": settings.contact_spread_ratio,
        "shadow_length_ratio": settings.shadow_length_ratio,
        "shadow_spread_ratio": settings.shadow_spread_ratio,
        "max_foot_lift_ratio": settings.max_foot_lift_ratio,
        "flat_luminance_range": settings.flat_luminance_range,
        "min_light_confidence": settings.min_light_confidence,
        "contact_line_min_baseline_ratio": settings.contact_line_min_baseline_ratio,
    }
    for name, value in ratios.items():
        if not math.isfinite(value) or value < 0.0:
            raise ContactGroundingError(f"{name} must be finite and nonnegative, got {value}")
    if settings.min_plate_samples <= 0:
        raise ContactGroundingError(
            f"min_plate_samples must be positive, got {settings.min_plate_samples}"
        )
    if not 0.0 <= settings.min_light_confidence <= 1.0:
        raise ContactGroundingError(
            f"min_light_confidence must be between 0 and 1, got {settings.min_light_confidence}"
        )
    return settings


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _finite(value: float, label: str) -> float:
    """Return the value, or fail closed when it is not finite.

    Every derived number that leaves this module is checked here so an
    overflowed or non-finite intermediate cannot reach a report as if it were
    a measurement.
    """
    if not math.isfinite(value):
        raise ContactGroundingError(f"{label} must be finite, got {value!r}")
    return float(value)


def support_polygon(frame: KeypointFrame, settings: GroundingSettings) -> SupportPolygon:
    """Derive the ground-bearing support polygon and contact line for a frame.

    Each foot side carries its own ground level, so a person whose feet sit
    at different depths in the frame produces a tilted contact line instead of
    one flattened sample.  A side whose ground level rises more than
    ``max_foot_lift_ratio`` above the nearest side is treated as lifted: it
    leaves the support polygon and the line fit, and the ratio of planted feet
    records it.  A side's ground-bearing samples are the ones within
    ``contact_band_ratio`` of that side's lowest foot landmark; each planted
    side becomes a rectangular footprint, and the support polygon is the
    convex hull of those footprints.
    """
    foot_points = [(name, x, y) for name, (x, y) in frame.points.items() if name in FOOT_LANDMARKS]
    if not foot_points:
        raise ContactGroundingError(f"frame {frame.frame} has no ground-bearing foot landmark")
    body_samples = [(name, y) for name, (_, y) in frame.points.items() if name in BODY_LANDMARKS]
    hips = [y for name, y in body_samples if name.endswith("hip")]
    knees = [y for name, y in body_samples if name.endswith("knee")]
    if hips:
        body_y = sum(hips) / len(hips)
    elif knees:
        body_y = sum(knees) / len(knees)
    else:
        raise ContactGroundingError(
            f"frame {frame.frame} lacks a hip or knee landmark for body scale"
        )
    sides: dict[str, list[tuple[str, float, float]]] = {}
    for name, x, y in foot_points:
        sides.setdefault(name.split("_", 1)[0], []).append((name, x, y))
    side_levels = {side: max(y for _, _, y in samples) for side, samples in sides.items()}
    ground_y = max(side_levels.values())
    leg_span = ground_y - body_y
    if leg_span <= 0.0:
        raise ContactGroundingError(f"frame {frame.frame} has a degenerate body scale")
    _finite(leg_span, f"frame {frame.frame} leg span")
    band = max(1.0, settings.contact_band_ratio * leg_span)
    lift_limit = settings.max_foot_lift_ratio * leg_span
    foot_width = _clamp(settings.foot_width_ratio * leg_span, 2.0, leg_span)
    foot_depth = _clamp(settings.foot_depth_ratio * leg_span, 1.0, leg_span)
    _finite(band, f"frame {frame.frame} contact band")
    _finite(lift_limit, f"frame {frame.frame} foot lift limit")
    _finite(foot_width, f"frame {frame.frame} foot width")
    _finite(foot_depth, f"frame {frame.frame} foot depth")
    corners: list[tuple[float, float]] = []
    line_points: list[tuple[float, float]] = []
    ground_samples = 0
    for side in sorted(sides):
        samples = sides[side]
        level = side_levels[side]
        if ground_y - level > lift_limit:
            continue
        planted = [(name, x, y) for name, x, y in samples if y >= level - band]
        if not planted:
            continue
        ground_samples += len(planted)
        planted_xs = [x for _, x, _ in planted]
        line_points.append((sum(planted_xs) / len(planted_xs), level))
        left = min(planted_xs) - foot_width / 2.0
        right = max(planted_xs) + foot_width / 2.0
        _finite(left, f"frame {frame.frame} footprint left edge")
        _finite(right, f"frame {frame.frame} footprint right edge")
        # The footprint straddles its own contact level by half a foot depth on
        # each side, so a base pixel row that sits on the line falls inside it.
        corners.extend(
            [
                (left, level - foot_depth / 2.0),
                (right, level - foot_depth / 2.0),
                (left, level + foot_depth / 2.0),
                (right, level + foot_depth / 2.0),
            ]
        )
    if not corners:
        raise ContactGroundingError(f"frame {frame.frame} has no foot footprint to support")
    hull = convex_hull(corners)
    area = polygon_area(hull)
    _finite(area, f"frame {frame.frame} support polygon area")
    if area <= 0.0:
        raise ContactGroundingError(f"frame {frame.frame} produced a zero-area support polygon")
    line_xs = [x for x, _ in line_points]
    line_ys = [y for _, y in line_points]
    spread = max(line_xs) - min(line_xs)
    minimum_baseline = max(
        1e-6,
        settings.contact_line_min_baseline_ratio * leg_span,
    )
    if len(line_xs) >= 2 and spread > minimum_baseline:
        mean_x = sum(line_xs) / len(line_xs)
        mean_y = sum(line_ys) / len(line_ys)
        try:
            denominator = sum((x - mean_x) ** 2 for x in line_xs)
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in line_points)
            slope = _finite(numerator / denominator, f"frame {frame.frame} contact-line slope")
            intercept = _finite(
                mean_y - slope * mean_x, f"frame {frame.frame} contact-line intercept"
            )
        except OverflowError as error:
            raise ContactGroundingError(
                f"frame {frame.frame} contact-line fit overflowed"
            ) from error
    else:
        # A near-coincident horizontal projection cannot identify tilt. Use the
        # lowest planted level rather than extrapolating a huge slope.
        slope = 0.0
        intercept = _finite(ground_y, f"frame {frame.frame} contact-line intercept")
    return SupportPolygon(
        points=tuple(hull),
        area_px=area,
        ground_y=ground_y,
        contact_slope=slope,
        contact_intercept=intercept,
        foot_samples=len(foot_points),
        ground_samples=ground_samples,
        planted_foot_ratio=len(line_points) / len(sides),
        side_ground_levels=tuple(sorted(side_levels.items())),
        leg_span_px=leg_span,
        contact_line_conditioned=len(line_xs) >= 2 and spread > minimum_baseline,
    )


def _object_geometry(
    mask: np.ndarray, settings: GroundingSettings
) -> tuple[tuple[int, int, int, int], int, float, float, int, int, np.ndarray, int]:
    """Return the inserted object's bounding box, footprint, and extents."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ContactGroundingError("the inserted-object mask has no pixels")
    top_row = int(rows[0])
    bottom_row = int(rows[-1])
    left_column = int(columns[0])
    right_column = int(columns[-1])
    height = bottom_row - top_row + 1
    width = right_column - left_column + 1
    band_rows = max(1, int(round(settings.object_band_ratio * height)))
    first_band_row = max(0, bottom_row - band_rows + 1)
    footprint = np.zeros_like(mask)
    footprint[first_band_row:, :] = mask[first_band_row:, :]
    footprint_pixels = int(np.count_nonzero(footprint))
    if footprint_pixels == 0:
        raise ContactGroundingError("the inserted-object mask has no ground-contact footprint")
    footprint_columns = np.nonzero(footprint.any(axis=0))[0]
    # The footprint's column mean, not the midpoint of its extreme columns: a
    # footprint with a notch or two separate feet has its mass off centre, and
    # the extremes would place the object somewhere it does not touch ground.
    column_counts = footprint.sum(axis=0, dtype=np.int64)
    weights = column_counts[footprint_columns].astype(np.float64)
    column_centers = footprint_columns.astype(np.float64) + 0.5
    centroid_x = _finite(
        float(np.sum(column_centers * weights) / np.sum(weights)),
        "the inserted-object footprint column mean",
    )
    bbox = (left_column, top_row, right_column + 1, bottom_row + 1)
    return (
        bbox,
        bottom_row,
        float(bottom_row + 1),
        centroid_x,
        height,
        width,
        footprint,
        footprint_pixels,
    )


def _plate_luminance(plate: np.ndarray) -> np.ndarray:
    """Return the plate's luminance as float64."""
    if plate.ndim == 3:
        if plate.shape[2] < 3:
            raise ContactGroundingError("plate frame must carry three color channels")
        return (
            0.299 * plate[:, :, 0].astype(np.float64)
            + 0.587 * plate[:, :, 1].astype(np.float64)
            + 0.114 * plate[:, :, 2].astype(np.float64)
        )
    if plate.ndim == 2:
        return plate.astype(np.float64)
    raise ContactGroundingError("plate frame must be a two- or three-dimensional array")


def _code_unit_scale(plate: np.ndarray) -> float:
    """Return how many 8-bit code units one unit of the plate's scale holds.

    An integer plate carries its own full-scale range; a float plate is
    expected to hold 0..255 code units already.
    """
    if np.issubdtype(plate.dtype, np.integer):
        return 255.0 / float(np.iinfo(plate.dtype).max)
    return 1.0


def _safe_rms(values: np.ndarray) -> float:
    """Return the root-mean-square of ``values`` without squaring overflow."""
    if values.size == 0:
        return 0.0
    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        return 0.0
    return peak * float(np.sqrt(np.mean((values / peak) ** 2)))


def estimate_light_direction(
    plate: np.ndarray,
    *,
    object_mask: np.ndarray,
    support: SupportPolygon,
    centroid_x: float,
    object_width: int,
    object_height: int,
    settings: GroundingSettings,
) -> LightEstimate:
    """Estimate the light direction from a luminance fit over the near-side plate.

    The sampled window is the floor in front of the person, bounded by the
    contact line and widened around the inserted object's footprint.  Pixels
    covered by the inserted object are excluded because the plate may still
    carry the content the object will cover.  The fitted luminance plane points
    toward the brighter side of the plate, which names the light azimuth.

    A window whose luminance spans at most ``flat_luminance_range`` 8-bit code
    units carries no usable gradient: quantization noise at that scale would
    otherwise become a confident light direction and a full-length cast
    shadow.  Such a window is reported as ``plate-luminance-flat`` with a
    zero direction and zero confidence, so no cast shadow is derived from it.
    """
    if plate.shape[:2] != object_mask.shape:
        raise ContactGroundingError(
            f"plate frame {plate.shape[:2]} does not match the object mask {object_mask.shape}"
        )
    luminance = _plate_luminance(plate)
    code_unit_scale = _code_unit_scale(plate)
    height, width = object_mask.shape
    margin = max(4.0, 0.25 * object_width)
    band = max(8.0, float(object_height))
    ground_y = support.ground_y_at(centroid_x)
    x_min = max(0, int(math.floor(centroid_x - object_width / 2.0 - margin)))
    x_max = min(width - 1, int(math.ceil(centroid_x + object_width / 2.0 + margin)))
    y_min = max(0, int(math.floor(ground_y)))
    y_max = min(height - 1, int(math.ceil(ground_y + band)))
    if x_max < x_min or y_max < y_min:
        raise ContactGroundingError("the plate sampling window is empty")
    row_centers = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
    column_centers = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
    grid_x = np.broadcast_to(column_centers[None, :], (row_centers.size, column_centers.size))
    grid_y = np.broadcast_to(row_centers[:, None], (row_centers.size, column_centers.size))
    near_side = grid_y >= support.contact_slope * grid_x + support.contact_intercept - 1.0
    keep = near_side & ~object_mask[y_min : y_max + 1, x_min : x_max + 1]
    xs = grid_x[keep]
    ys = grid_y[keep]
    values = luminance[y_min : y_max + 1, x_min : x_max + 1][keep]
    if values.size < settings.min_plate_samples:
        raise ContactGroundingError(
            f"the plate sampling window holds {values.size} usable pixels, "
            f"fewer than the required {settings.min_plate_samples}"
        )
    if not bool(np.all(np.isfinite(values))):
        raise ContactGroundingError("the plate sampling window holds nonfinite luminance")
    with np.errstate(over="ignore"):
        span = _finite(float(values.max() - values.min()), "the plate luminance range")
    if span * code_unit_scale <= settings.flat_luminance_range:
        return LightEstimate(
            source="plate-luminance-flat",
            direction_x=0.0,
            direction_y=0.0,
            azimuth_degrees=0.0,
            gradient_x=0.0,
            gradient_y=0.0,
            confidence=0.0,
            samples=int(values.size),
            window=(x_min, y_min, x_max, y_max),
        )
    span_x = max(1.0, float(x_max - x_min))
    span_y = max(1.0, float(y_max - y_min))
    reference_x = 0.5 * (x_min + x_max)
    reference_y = 0.5 * (y_min + y_max)
    design = np.column_stack(
        [np.ones(xs.size), (xs - reference_x) / span_x, (ys - reference_y) / span_y]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    if not bool(np.all(np.isfinite(coefficients))):
        raise ContactGroundingError("the plate luminance fit produced nonfinite coefficients")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = values - design @ coefficients
    residual_rms = _safe_rms(residual)
    gradient_x = _finite(float(coefficients[1]) / span_x, "the plate luminance gradient x")
    gradient_y = _finite(float(coefficients[2]) / span_y, "the plate luminance gradient y")
    magnitude = math.hypot(gradient_x, gradient_y)
    radius = 0.5 * math.hypot(span_x, span_y)
    signal = _finite(magnitude * radius / max(residual_rms, 1e-6), "the plate luminance fit signal")
    confidence = _clamp(signal / (1.0 + signal), 0.0, 1.0)
    _finite(confidence, "the plate luminance confidence")
    if magnitude > 0.0:
        direction_x = gradient_x / magnitude
        direction_y = gradient_y / magnitude
        azimuth = math.degrees(math.atan2(gradient_y, gradient_x))
        source = "plate-luminance"
    else:
        direction_x = 0.0
        direction_y = 0.0
        azimuth = 0.0
        source = "plate-luminance-flat"
    return LightEstimate(
        source=source,
        direction_x=direction_x,
        direction_y=direction_y,
        azimuth_degrees=azimuth,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        confidence=confidence,
        samples=int(values.size),
        window=(x_min, y_min, x_max, y_max),
    )


def _dilate_columns(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask horizontally without filling disconnected bases."""
    result = np.zeros_like(mask)
    for offset in range(-radius, radius + 1):
        if offset < 0:
            result[:, :offset] |= mask[:, -offset:]
        elif offset > 0:
            result[:, offset:] |= mask[:, :-offset]
        else:
            result |= mask
    return result


def suggested_shadow_mask(
    object_mask: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    bottom_row: int,
    bottom_y: float,
    centroid_x: float,
    object_width: int,
    object_height: int,
    support: SupportPolygon,
    light: LightEstimate | None,
    settings: GroundingSettings,
    footprint: np.ndarray,
) -> tuple[np.ndarray, ShadowGeometry]:
    """Build the suggested contact and cast shadow for one frame.

    The contact band hugs the inserted object's base.  The cast wedge starts
    at that base and extends away from the estimated light, tapering to a
    point.  Both are clipped away from the inserted object itself, which
    occludes its own shadow.
    """
    height, width = object_mask.shape
    thickness = max(1.0, settings.contact_thickness_ratio * object_height)
    spread = max(1.0, settings.contact_spread_ratio * object_width)
    _finite(thickness, "the contact band thickness")
    _finite(spread, "the contact band spread")
    first_row = max(0, int(math.floor(bottom_row - thickness)) + 1)
    base = footprint.copy()
    base[:first_row] = False
    if not base.any():
        raise ContactGroundingError("the inserted-object mask has no bottom-band footprint")
    contact = _dilate_columns(base, max(1, int(math.ceil(spread))))

    if (
        light is None
        or light.source != "plate-luminance"
        or light.confidence < settings.min_light_confidence
    ):
        direction_x, direction_y = 0.0, 1.0
        length = 0.0
    else:
        direction_x = -light.direction_x
        direction_y = -light.direction_y
        length = settings.shadow_length_ratio * object_width
    _finite(direction_x, "the shadow direction x")
    _finite(direction_y, "the shadow direction y")
    _finite(length, "the shadow length")
    half_width = 0.5 * settings.shadow_spread_ratio * object_width
    _finite(half_width, "the shadow half width")
    cast = np.zeros_like(object_mask)
    if length > 0.0:
        normal_x, normal_y = -direction_y, direction_x
        extent_x = abs(direction_x) * length + abs(normal_x) * half_width
        extent_y = abs(direction_y) * length + abs(normal_y) * half_width
        _finite(extent_x, "the shadow horizontal extent")
        _finite(extent_y, "the shadow vertical extent")
        x_min = max(0, int(math.floor(centroid_x - extent_x)))
        x_max = min(width - 1, int(math.ceil(centroid_x + extent_x)))
        y_min = max(0, int(math.floor(bottom_y - extent_y)))
        y_max = min(height - 1, int(math.ceil(bottom_y + extent_y)))
        if x_max >= x_min and y_max >= y_min:
            columns = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
            rows = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
            grid_x, grid_y = np.meshgrid(columns, rows)
            offset_x = grid_x - centroid_x
            offset_y = grid_y - bottom_y
            along = offset_x * direction_x + offset_y * direction_y
            across = offset_x * normal_x + offset_y * normal_y
            taper = 1.0 - np.clip(along, 0.0, length) / length
            cast[y_min : y_max + 1, x_min : x_max + 1] = (
                (along >= 0.0) & (along <= length) & (np.abs(across) <= half_width * taper)
            )

    shadow = (contact | cast) & ~object_mask
    columns = np.arange(width, dtype=np.float64) + 0.5
    line = support.contact_slope * columns + support.contact_intercept
    above = shadow & ((np.arange(height, dtype=np.float64)[:, None] + 0.5) < line[None, :] - 1.0)
    above_pixels = int(np.count_nonzero(above))
    if settings.clip_shadow_to_ground:
        shadow = shadow & ~above
    contact_pixels = int(np.count_nonzero(contact & shadow))
    cast_pixels = int(np.count_nonzero(cast & shadow))
    total_pixels = int(np.count_nonzero(shadow))
    return shadow, ShadowGeometry(
        direction_x=direction_x,
        direction_y=direction_y,
        length_px=length,
        half_width_px=half_width,
        contact_pixels=contact_pixels,
        cast_pixels=cast_pixels,
        total_pixels=total_pixels,
        above_contact_line_pixels=above_pixels,
    )


NO_PLATE_LIGHT = LightEstimate(
    source="no-plate",
    direction_x=0.0,
    direction_y=0.0,
    azimuth_degrees=0.0,
    gradient_x=0.0,
    gradient_y=0.0,
    confidence=0.0,
    samples=0,
    window=(0, 0, 0, 0),
)


def frame_grounding(
    frame: KeypointFrame,
    object_mask: np.ndarray,
    *,
    plate: np.ndarray | None = None,
    settings: GroundingSettings | None = None,
) -> tuple[FrameGrounding, np.ndarray]:
    """Measure one frame and return its metrics with the suggested shadow mask."""
    resolved = validate_settings(settings or GroundingSettings())
    if object_mask.ndim != 2:
        raise ContactGroundingError("the inserted-object mask must be a two-dimensional array")
    if object_mask.dtype != np.bool_:
        raise ContactGroundingError(
            f"the inserted-object mask must be a boolean array, got {object_mask.dtype}"
        )
    support = support_polygon(frame, resolved)
    (
        bbox,
        bottom_row,
        bottom_y,
        centroid_x,
        object_height,
        object_width,
        footprint,
        footprint_pixels,
    ) = _object_geometry(object_mask, resolved)
    frame_shape = (int(object_mask.shape[0]), int(object_mask.shape[1]))
    support_region = rasterize_polygon(support.points, frame_shape)
    contact_pixels = int(np.count_nonzero(footprint & support_region))
    coverage = contact_pixels / footprint_pixels
    _finite(coverage, "the contact coverage")
    ground_y_at_object = support.ground_y_at(centroid_x)
    vertical_offset = bottom_y - ground_y_at_object
    _finite(vertical_offset, "the vertical offset")
    tolerance = max(2.0, resolved.vertical_tolerance_ratio * support.leg_span_px)
    _finite(tolerance, "the vertical tolerance")
    light = (
        estimate_light_direction(
            plate,
            object_mask=object_mask,
            support=support,
            centroid_x=centroid_x,
            object_width=object_width,
            object_height=object_height,
            settings=resolved,
        )
        if plate is not None
        else NO_PLATE_LIGHT
    )
    shadow, shadow_geometry = suggested_shadow_mask(
        object_mask,
        bbox=bbox,
        bottom_row=bottom_row,
        bottom_y=bottom_y,
        centroid_x=centroid_x,
        object_width=object_width,
        object_height=object_height,
        support=support,
        light=light,
        settings=resolved,
        footprint=footprint,
    )
    grounding = FrameGrounding(
        frame=frame.frame,
        support=support,
        object_bbox=bbox,
        object_bottom_y=bottom_y,
        object_centroid_x=centroid_x,
        object_height_px=object_height,
        object_width_px=object_width,
        footprint_pixels=footprint_pixels,
        contact_pixels=contact_pixels,
        contact_coverage=coverage,
        ground_y_at_object=ground_y_at_object,
        vertical_offset_px=vertical_offset,
        vertical_tolerance_px=tolerance,
        vertical_aligned=abs(vertical_offset) <= tolerance,
        light=light,
        shadow=shadow_geometry,
    )
    return grounding, shadow


def ground_contact(
    keypoints: Sequence[KeypointFrame],
    object_masks: Sequence[np.ndarray],
    *,
    plate_frames: Sequence[np.ndarray | None] | None = None,
    settings: GroundingSettings | None = None,
) -> tuple[list[FrameGrounding], list[np.ndarray]]:
    """Measure every keyed frame against its inserted-object mask.

    The returned shadow list has one entry per mask frame; frames without
    keypoints keep an empty (all-keep) mask so the mask sequence stays aligned
    with the source clip.
    """
    resolved = validate_settings(settings or GroundingSettings())
    if not object_masks:
        raise ContactGroundingError("the inserted-object mask sequence is empty")
    shape = object_masks[0].shape
    for index, mask in enumerate(object_masks):
        if mask.shape != shape:
            raise ContactGroundingError(
                f"object mask {index} has shape {mask.shape}, expected {shape}"
            )
    if plate_frames is not None and len(plate_frames) != len(object_masks):
        raise ContactGroundingError(
            f"the plate has {len(plate_frames)} frames but the object mask has {len(object_masks)}"
        )
    if not keypoints:
        raise ContactGroundingError("no keypoint frames were supplied")
    measured: list[FrameGrounding] = []
    shadows: list[np.ndarray] = [np.zeros(shape, dtype=bool) for _ in object_masks]
    for frame in keypoints:
        if not 0 <= frame.frame < len(object_masks):
            raise ContactGroundingError(
                f"keypoint frame {frame.frame} is outside the "
                f"{len(object_masks)} frame mask sequence; valid frames are 0 through "
                f"{len(object_masks) - 1}"
            )
        plate = plate_frames[frame.frame] if plate_frames is not None else None
        if plate is None and plate_frames is not None:
            raise ContactGroundingError(f"no plate frame was supplied for frame {frame.frame}")
        grounding, shadow = frame_grounding(
            frame, object_masks[frame.frame], plate=plate, settings=resolved
        )
        measured.append(grounding)
        shadows[frame.frame] = shadow
    return measured, shadows


def summarize(measurements: Sequence[FrameGrounding], *, mask_frames: int) -> dict[str, Any]:
    """Aggregate per-frame measurements into bounded numeric observations.

    The observations are geometry, not verdicts: nothing here scores quality
    or decides whether an insertion is acceptable.
    """
    if not measurements:
        raise ContactGroundingError("cannot summarize an empty measurement set")
    coverages = [item.contact_coverage for item in measurements]
    offsets = [item.vertical_offset_px for item in measurements]
    areas = [item.support.area_px for item in measurements]
    ratios = [item.support.planted_foot_ratio for item in measurements]
    shadow_totals = [item.shadow.total_pixels for item in measurements]
    aligned = [item for item in measurements if item.vertical_aligned]
    observations: list[str] = []
    omitted = 0
    for item in measurements:
        if item.vertical_aligned and item.contact_coverage >= 0.5:
            continue
        if len(observations) < 10:
            observations.append(
                f"frame {item.frame}: object base is {item.vertical_offset_px:+.2f} px "
                f"from the contact line (tolerance {item.vertical_tolerance_px:.2f} px) and "
                f"{item.contact_coverage:.3f} of its ground band lies in the support polygon"
            )
        else:
            omitted += 1
    if omitted:
        observations.append(
            f"{omitted} further frames fall outside the vertical tolerance or the "
            f"support polygon; their individual deviations are not listed"
        )
    return {
        "measured_frames": len(measurements),
        "mask_frames": mask_frames,
        "frames_without_keypoints": mask_frames - len(measurements),
        "mean_contact_coverage": float(sum(coverages) / len(coverages)),
        "min_contact_coverage": float(min(coverages)),
        "max_contact_coverage": float(max(coverages)),
        "mean_vertical_offset_px": float(sum(offsets) / len(offsets)),
        "max_abs_vertical_offset_px": float(max(abs(value) for value in offsets)),
        "frames_within_vertical_tolerance": len(aligned),
        "mean_support_area_px": float(sum(areas) / len(areas)),
        "mean_planted_foot_ratio": float(sum(ratios) / len(ratios)),
        "shadow_pixels_total": int(sum(shadow_totals)),
        "frames_with_shadow": sum(1 for value in shadow_totals if value > 0),
        "observations": observations,
    }


def mask_digest(mask: np.ndarray) -> str:
    """SHA-256 of a boolean mask's bytes in C order."""
    array = np.ascontiguousarray(mask, dtype=bool)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()
