"""Per-frame lower-body ownership measured from MediaPipe pose JSON.

The pose source here is MediaPipe Pose Landmarker output: one pose per
person, 33 landmarks in a fixed order, normalized ``x`` and ``y`` in image
units, a relative ``z`` scaled roughly like ``x``, and a ``visibility`` score.
This module is not DWPose and not RTMPose; those keypoint schemas differ and
are not accepted here.

The module owns deterministic geometry and gate contracts only. It calls no
model, no network endpoint, and no credential, and it reads no path from the
environment.

Ownership is fail closed. A pose instance owns its lower body in a frame only
when every required chain is complete, every required landmark is visible
enough and shallow enough, every required segment has a plausible length, and
the instance did not jump since the previous frame it appeared in. A failed
gate withholds ownership for that frame and records the reason.

The input is fail closed too. A landmark must carry x, y, and z, so a report
never presents an absent depth as a measured zero. Frame indices must strictly
increase, so no two frames share an index and every temporal jump names the
frame it was measured against. A cross-frame displacement converts each
landmark with the vertical scale of the frame it came from, because every
frame normalizes y to its own height.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

POSE_SOURCE_LABEL = "mediapipe-pose-landmarker"
POSE_LANDMARK_MODEL = "mediapipe_pose_33"
IS_DWPOSE = False
IS_RTMPOSE = False
REPORT_SCHEMA = "lower-body-ownership-report-v1"
REPORT_GENERATOR = "samurai.lower_body"
ROUNDING_DIGITS = 6

MEDIAPIPE_POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

LANDMARK_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES)
}


@dataclass(frozen=True)
class Chain:
    """An ordered landmark chain that a pose must supply to claim ownership."""

    name: str
    landmarks: tuple[str, ...]


@dataclass(frozen=True)
class SegmentSpec:
    """A measured bone or foot span between two landmark names."""

    name: str
    start: str
    end: str


LOWER_BODY_CHAINS: dict[str, Chain] = {
    "left_leg": Chain("left_leg", ("left_hip", "left_knee", "left_ankle")),
    "right_leg": Chain("right_leg", ("right_hip", "right_knee", "right_ankle")),
    "left_foot": Chain("left_foot", ("left_ankle", "left_heel", "left_foot_index")),
    "right_foot": Chain("right_foot", ("right_ankle", "right_heel", "right_foot_index")),
}

LOWER_BODY_SEGMENTS: tuple[SegmentSpec, ...] = (
    SegmentSpec("left_thigh", "left_hip", "left_knee"),
    SegmentSpec("right_thigh", "right_hip", "right_knee"),
    SegmentSpec("left_shin", "left_knee", "left_ankle"),
    SegmentSpec("right_shin", "right_knee", "right_ankle"),
    SegmentSpec("left_foot", "left_ankle", "left_foot_index"),
    SegmentSpec("right_foot", "right_ankle", "right_foot_index"),
    SegmentSpec("left_heel_span", "left_heel", "left_foot_index"),
    SegmentSpec("right_heel_span", "right_heel", "right_foot_index"),
)

CHAIN_SEGMENTS: dict[str, tuple[str, ...]] = {
    "left_leg": ("left_thigh", "left_shin"),
    "right_leg": ("right_thigh", "right_shin"),
    "left_foot": ("left_foot", "left_heel_span"),
    "right_foot": ("right_foot", "right_heel_span"),
}

LOWER_BODY_LANDMARK_NAMES: tuple[str, ...] = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

DEFAULT_REQUIRED_CHAINS: tuple[str, ...] = ("left_leg", "right_leg", "left_foot", "right_foot")


class LowerBodyError(RuntimeError):
    """A pose document, threshold, or report request is unusable."""


@dataclass(frozen=True)
class Landmark:
    """One MediaPipe landmark with its visibility and relative depth."""

    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class Segment:
    """A measured bone or foot span in normalized-width metric space."""

    name: str
    start: str
    end: str
    vector: tuple[float, float, float]
    length: float
    min_visibility: float
    z_delta: float


@dataclass(frozen=True)
class Pose:
    """One pose instance in one frame, keyed by MediaPipe landmark name."""

    pose_id: str
    landmarks: dict[str, Landmark]


@dataclass(frozen=True)
class PoseFrame:
    """One frame of pose instances with optional source image geometry."""

    frame_index: int
    poses: tuple[Pose, ...]
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class LowerBodyGeometry:
    """Lower-body landmarks and spans derived from one pose instance."""

    frame_index: int
    pose_id: str
    pelvis: Landmark | None
    landmarks: dict[str, Landmark]
    segments: dict[str, Segment]
    missing_landmarks: tuple[str, ...]
    y_scale: float
    aspect_corrected: bool


@dataclass(frozen=True)
class JumpMeasurement:
    """Displacement between a pose instance and its previous appearance."""

    basis: str
    previous_frame_index: int | None = None
    previous_owned: bool | None = None
    pelvis_jump: float | None = None
    max_landmark_jump: float | None = None
    max_jump_landmark: str | None = None
    y_scale: float = 1.0
    previous_y_scale: float | None = None


@dataclass(frozen=True)
class GateOutcome:
    """The gate verdicts and measured evidence behind one ownership decision."""

    gates: dict[str, bool]
    reasons: tuple[str, ...]
    missing_required: tuple[str, ...]
    min_required_visibility: float
    max_required_abs_z: float

    @property
    def owned(self) -> bool:
        return all(self.gates.values())


@dataclass(frozen=True)
class OwnershipDecision:
    """The fail-closed ownership verdict for one pose instance in one frame."""

    frame_index: int
    pose_id: str
    owned: bool
    gates: dict[str, bool]
    reasons: tuple[str, ...]
    missing_required: tuple[str, ...]
    min_required_visibility: float
    max_required_abs_z: float
    geometry: LowerBodyGeometry
    jumps: JumpMeasurement


@dataclass(frozen=True)
class FrameOwnership:
    """Every pose instance evaluated in one frame."""

    frame_index: int
    poses: tuple[OwnershipDecision, ...]

    @property
    def owned_pose_ids(self) -> tuple[str, ...]:
        return tuple(decision.pose_id for decision in self.poses if decision.owned)


@dataclass(frozen=True)
class OwnershipReport:
    """The per-frame ownership verdicts for one parsed pose document."""

    frames: tuple[FrameOwnership, ...]
    thresholds: OwnershipThresholds

    @property
    def pose_count(self) -> int:
        return sum(len(frame.poses) for frame in self.frames)

    @property
    def owned_pose_count(self) -> int:
        return sum(1 for frame in self.frames for decision in frame.poses if decision.owned)

    @property
    def owned_frame_count(self) -> int:
        return sum(1 for frame in self.frames if frame.owned_pose_ids)


@dataclass(frozen=True)
class OwnershipThresholds:
    """Numeric gates that decide whether a pose instance owns its lower body."""

    min_visibility: float = 0.5
    max_abs_z: float = 1.0
    min_segment_length: float = 0.01
    max_segment_length: float = 1.5
    max_pelvis_jump: float = 0.2
    max_landmark_jump: float = 0.35
    required_chains: tuple[str, ...] = DEFAULT_REQUIRED_CHAINS
    require_previous_frame: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("min_visibility", self.min_visibility),
            ("max_abs_z", self.max_abs_z),
            ("min_segment_length", self.min_segment_length),
            ("max_segment_length", self.max_segment_length),
            ("max_pelvis_jump", self.max_pelvis_jump),
            ("max_landmark_jump", self.max_landmark_jump),
        ):
            if not math.isfinite(value):
                raise LowerBodyError(f"{name} must be finite")
        if not 0.0 <= self.min_visibility <= 1.0:
            raise LowerBodyError("min_visibility must be within [0, 1]")
        if self.max_abs_z <= 0.0:
            raise LowerBodyError("max_abs_z must be positive")
        if self.min_segment_length < 0.0:
            raise LowerBodyError("min_segment_length must not be negative")
        if self.max_segment_length <= self.min_segment_length:
            raise LowerBodyError("max_segment_length must exceed min_segment_length")
        if self.max_pelvis_jump <= 0.0:
            raise LowerBodyError("max_pelvis_jump must be positive")
        if self.max_landmark_jump <= 0.0:
            raise LowerBodyError("max_landmark_jump must be positive")
        if not self.required_chains:
            raise LowerBodyError("at least one required chain must be named")
        unknown = [name for name in self.required_chains if name not in LOWER_BODY_CHAINS]
        if unknown:
            raise LowerBodyError("unknown required chains: " + ", ".join(sorted(unknown)))

    @property
    def required_landmarks(self) -> tuple[str, ...]:
        """Landmark names required by the selected chains, in stable order."""
        required: list[str] = []
        for chain_name in self.required_chains:
            for name in LOWER_BODY_CHAINS[chain_name].landmarks:
                if name not in required:
                    required.append(name)
        return tuple(required)

    @property
    def required_segments(self) -> tuple[str, ...]:
        """Segment names required by the selected chains, in stable order."""
        required: list[str] = []
        for chain_name in self.required_chains:
            for name in CHAIN_SEGMENTS[chain_name]:
                if name not in required:
                    required.append(name)
        return tuple(required)


def _as_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LowerBodyError(f"{context} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _as_sequence(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise LowerBodyError(f"{context} must be a JSON array")
    return list(value)


def _as_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LowerBodyError(f"{context} must be a JSON number")
    try:
        number = float(value)
    except OverflowError as error:
        raise LowerBodyError(f"{context} is outside the finite float range") from error
    if not math.isfinite(number):
        raise LowerBodyError(f"{context} must be finite")
    return number


def _optional_number(value: object, default: float, context: str) -> float:
    if value is None:
        return default
    return _as_number(value, context)


def _optional_dimension(value: object, default: int | None, context: str) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise LowerBodyError(f"{context} must be an integer")
    if value <= 0:
        raise LowerBodyError(f"{context} must be positive")
    try:
        if not math.isfinite(float(value)):
            raise LowerBodyError(f"{context} is outside the finite float range")
    except OverflowError as error:
        raise LowerBodyError(f"{context} is outside the finite float range") from error
    return value


def _parse_landmark(payload: object, name: str) -> Landmark:
    mapping = _as_mapping(payload, f"landmark {name}")
    if "x" not in mapping or "y" not in mapping or "z" not in mapping:
        raise LowerBodyError(f"landmark {name} must carry x, y, and z")
    visibility = _optional_number(mapping.get("visibility"), 0.0, f"landmark {name} visibility")
    if not 0.0 <= visibility <= 1.0:
        raise LowerBodyError(f"landmark {name} visibility must be within [0, 1]")
    return Landmark(
        x=_as_number(mapping["x"], f"landmark {name} x"),
        y=_as_number(mapping["y"], f"landmark {name} y"),
        z=_as_number(mapping["z"], f"landmark {name} z"),
        visibility=visibility,
    )


def _parse_landmark_list(payload: object, context: str) -> dict[str, Landmark]:
    entries = _as_sequence(payload, context)
    named = any(isinstance(entry, dict) and "name" in entry for entry in entries)
    landmarks: dict[str, Landmark] = {}
    for position, entry in enumerate(entries):
        if named:
            mapping = _as_mapping(entry, f"{context} entry {position}")
            raw_name = mapping.get("name")
            if not isinstance(raw_name, str):
                raise LowerBodyError(f"{context} entry {position} must name its landmark")
            name = raw_name
            if name not in LANDMARK_INDEX:
                raise LowerBodyError(f"{context} entry {position} names unknown landmark {name!r}")
        else:
            if position >= len(MEDIAPIPE_POSE_LANDMARK_NAMES):
                raise LowerBodyError(
                    f"{context} holds more than {len(MEDIAPIPE_POSE_LANDMARK_NAMES)} landmarks"
                )
            name = MEDIAPIPE_POSE_LANDMARK_NAMES[position]
        if name in landmarks:
            raise LowerBodyError(f"{context} repeats landmark {name}")
        landmarks[name] = _parse_landmark(entry, name)
    return landmarks


def _parse_pose(payload: object, context: str, default_id: str) -> Pose:
    pose_id = default_id
    landmarks_payload: object
    if isinstance(payload, dict):
        mapping = _as_mapping(payload, context)
        for key in ("pose_id", "id"):
            raw = mapping.get(key)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, str)):
                raise LowerBodyError(f"{context} {key} must be a string or an integer")
            pose_id = str(raw)
            break
        if "landmarks" in mapping:
            landmarks_payload = mapping["landmarks"]
        elif "pose_landmarks" in mapping:
            landmarks_payload = mapping["pose_landmarks"]
        else:
            raise LowerBodyError(f"{context} must carry landmarks or pose_landmarks")
    elif isinstance(payload, list):
        landmarks_payload = payload
    else:
        raise LowerBodyError(f"{context} must be a JSON object or array")
    landmarks = _parse_landmark_list(landmarks_payload, f"{context} landmarks")
    return Pose(pose_id=pose_id, landmarks=landmarks)


def _looks_like_landmark(entry: object) -> bool:
    return isinstance(entry, dict) and "x" in entry and "y" in entry


def _parse_frame(
    payload: object,
    position: int,
    default_width: int | None,
    default_height: int | None,
) -> PoseFrame:
    frame_index = position
    width = default_width
    height = default_height
    poses_payload: object
    if isinstance(payload, dict):
        mapping = _as_mapping(payload, f"frame {position}")
        raw_index = mapping.get("frame_index")
        if raw_index is not None:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise LowerBodyError(f"frame {position} frame_index must be an integer")
            frame_index = raw_index
        width = _optional_dimension(mapping.get("width"), width, f"frame {position} width")
        height = _optional_dimension(mapping.get("height"), height, f"frame {position} height")
        if "poses" in mapping:
            poses_payload = _as_sequence(mapping["poses"], f"frame {position} poses")
        elif "landmarks" in mapping:
            poses_payload = [mapping["landmarks"]]
        elif "pose_landmarks" in mapping:
            poses_payload = [mapping["pose_landmarks"]]
        else:
            raise LowerBodyError(f"frame {position} must carry poses or landmarks")
    elif isinstance(payload, list):
        if payload and all(_looks_like_landmark(entry) for entry in payload):
            poses_payload = [payload]
        else:
            poses_payload = payload
    else:
        raise LowerBodyError(f"frame {position} must be a JSON object or array")
    poses: list[Pose] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(poses_payload):
        pose = _parse_pose(entry, f"frame {position} pose {index}", str(index))
        if pose.pose_id in seen_ids:
            raise LowerBodyError(f"frame {position} repeats pose id {pose.pose_id}")
        seen_ids.add(pose.pose_id)
        poses.append(pose)
    return PoseFrame(
        frame_index=frame_index,
        poses=tuple(poses),
        width=width,
        height=height,
    )


def parse_pose_frames(document: object) -> tuple[PoseFrame, ...]:
    """Parse a MediaPipe pose document into per-frame pose instances.

    The canonical shape is ``{"frames": [{"poses": [{"landmarks": [...]}]}]}``.
    A frame may also carry a single bare pose, and a pose may be a bare list of
    landmarks in MediaPipe index order. ``pose_world_landmarks`` is rejected
    on purpose: it carries metres, not normalized image units. Frame indices
    must strictly increase; a repeated or reversed index is refused because a
    consumer would merge two distinct frames under one key.
    """
    width: int | None = None
    height: int | None = None
    frames_payload: object
    if isinstance(document, dict):
        mapping = _as_mapping(document, "pose document")
        width = _optional_dimension(mapping.get("width"), None, "document width")
        height = _optional_dimension(mapping.get("height"), None, "document height")
        if "frames" in mapping:
            frames_payload = _as_sequence(mapping["frames"], "document frames")
        elif "poses" in mapping:
            frames_payload = [mapping["poses"]]
        else:
            raise LowerBodyError("pose document must carry frames")
    elif isinstance(document, list):
        frames_payload = document
    else:
        raise LowerBodyError("pose document must be a JSON object or array")
    frames = [
        _parse_frame(entry, position, width, height)
        for position, entry in enumerate(_as_sequence(frames_payload, "document frames"))
    ]
    if not frames:
        raise LowerBodyError("pose document carries no frame")
    _require_increasing_frame_indices(frames)
    return tuple(frames)


def _require_increasing_frame_indices(frames: list[PoseFrame]) -> None:
    """Refuse duplicate or non-increasing frame indices."""
    previous: int | None = None
    for position, frame in enumerate(frames):
        if previous is not None and frame.frame_index <= previous:
            raise LowerBodyError(
                f"frame {position} declares frame_index {frame.frame_index} after "
                f"{previous}; frame indices must strictly increase"
            )
        previous = frame.frame_index


def _metric_y_scale(width: int | None, height: int | None) -> tuple[float, bool]:
    """Return the y scale that puts y on the same foot as x and z."""
    if width is not None and height is not None:
        try:
            scale = height / width
        except OverflowError as error:
            raise LowerBodyError(
                "frame height/width ratio is outside the finite float range"
            ) from error
        if not math.isfinite(scale):
            raise LowerBodyError("frame height/width ratio must be finite")
        return scale, True
    return 1.0, False


def _distance(
    first: Landmark,
    second: Landmark,
    first_y_scale: float,
    second_y_scale: float,
) -> float:
    """Measure a displacement with each landmark in its own frame's metric space.

    Every frame normalizes y to its own height, so a cross-frame delta has to
    scale each side with the scale of the frame that landmark came from.
    """
    dx = second.x - first.x
    dy = second.y * second_y_scale - first.y * first_y_scale
    dz = second.z - first.z
    distance = math.hypot(dx, dy, dz)
    if not math.isfinite(distance):
        raise LowerBodyError("landmark displacement is outside the finite float range")
    return distance


def extract_lower_body(frame: PoseFrame, pose: Pose) -> LowerBodyGeometry:
    """Derive the lower-body landmarks, spans, and pelvis of one pose instance."""
    y_scale, aspect_corrected = _metric_y_scale(frame.width, frame.height)
    landmarks = {
        name: pose.landmarks[name] for name in LOWER_BODY_LANDMARK_NAMES if name in pose.landmarks
    }
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")
    pelvis: Landmark | None = None
    if left_hip is not None and right_hip is not None:
        pelvis = Landmark(
            x=(left_hip.x + right_hip.x) / 2.0,
            y=(left_hip.y + right_hip.y) / 2.0,
            z=(left_hip.z + right_hip.z) / 2.0,
            visibility=min(left_hip.visibility, right_hip.visibility),
        )
    segments: dict[str, Segment] = {}
    for spec in LOWER_BODY_SEGMENTS:
        start = landmarks.get(spec.start)
        end = landmarks.get(spec.end)
        if start is None or end is None:
            continue
        vector = (
            end.x - start.x,
            (end.y - start.y) * y_scale,
            end.z - start.z,
        )
        length = math.hypot(*vector)
        if not math.isfinite(length):
            raise LowerBodyError(f"segment {spec.name} length is outside the finite float range")
        segments[spec.name] = Segment(
            name=spec.name,
            start=spec.start,
            end=spec.end,
            vector=vector,
            length=length,
            min_visibility=min(start.visibility, end.visibility),
            z_delta=abs(vector[2]),
        )
    missing = tuple(name for name in LOWER_BODY_LANDMARK_NAMES if name not in landmarks)
    return LowerBodyGeometry(
        frame_index=frame.frame_index,
        pose_id=pose.pose_id,
        pelvis=pelvis,
        landmarks=landmarks,
        segments=segments,
        missing_landmarks=missing,
        y_scale=y_scale,
        aspect_corrected=aspect_corrected,
    )


def _measure_jumps(
    geometry: LowerBodyGeometry,
    previous: tuple[int, LowerBodyGeometry, bool] | None,
) -> JumpMeasurement:
    """Measure the displacement since the previous frame of the same pose."""
    if previous is None:
        return JumpMeasurement(basis="bootstrap", y_scale=geometry.y_scale)
    previous_index, previous_geometry, previous_owned = previous
    pelvis_jump: float | None = None
    if geometry.pelvis is not None and previous_geometry.pelvis is not None:
        pelvis_jump = _distance(
            previous_geometry.pelvis,
            geometry.pelvis,
            previous_geometry.y_scale,
            geometry.y_scale,
        )
    max_jump: float | None = None
    max_name: str | None = None
    for name in LOWER_BODY_LANDMARK_NAMES:
        current = geometry.landmarks.get(name)
        earlier = previous_geometry.landmarks.get(name)
        if current is None or earlier is None:
            continue
        distance = _distance(
            earlier,
            current,
            previous_geometry.y_scale,
            geometry.y_scale,
        )
        if max_jump is None or distance > max_jump:
            max_jump = distance
            max_name = name
    return JumpMeasurement(
        basis="previous_frame",
        previous_frame_index=previous_index,
        previous_owned=previous_owned,
        pelvis_jump=pelvis_jump,
        max_landmark_jump=max_jump,
        max_jump_landmark=max_name,
        y_scale=geometry.y_scale,
        previous_y_scale=previous_geometry.y_scale,
    )


def _evaluate_gates(
    geometry: LowerBodyGeometry,
    thresholds: OwnershipThresholds,
    jumps: JumpMeasurement,
) -> GateOutcome:
    """Apply every ownership gate; any failed gate withholds ownership."""
    reasons: list[str] = []
    present = [name for name in thresholds.required_landmarks if name in geometry.landmarks]
    missing = tuple(
        name for name in thresholds.required_landmarks if name not in geometry.landmarks
    )

    chain_complete = not missing
    if missing:
        reasons.append("incomplete required chain landmarks: " + ", ".join(missing))

    visibilities = [geometry.landmarks[name].visibility for name in present]
    min_visibility = min(visibilities) if visibilities else 0.0
    visibility_ok = bool(present) and min_visibility >= thresholds.min_visibility
    if present and not visibility_ok:
        reasons.append(
            f"minimum required visibility {min_visibility:.4f} is below "
            f"{thresholds.min_visibility:.4f}"
        )

    depths = [abs(geometry.landmarks[name].z) for name in present]
    max_abs_z = max(depths) if depths else 0.0
    depth_ok = bool(present) and max_abs_z <= thresholds.max_abs_z
    if present and not depth_ok:
        reasons.append(f"maximum required |z| {max_abs_z:.4f} exceeds {thresholds.max_abs_z:.4f}")

    segment_problems: list[str] = []
    for name in thresholds.required_segments:
        segment = geometry.segments.get(name)
        if segment is None:
            segment_problems.append(f"{name} is not measurable")
        elif segment.length < thresholds.min_segment_length:
            segment_problems.append(
                f"{name} length {segment.length:.4f} is below {thresholds.min_segment_length:.4f}"
            )
        elif segment.length > thresholds.max_segment_length:
            segment_problems.append(
                f"{name} length {segment.length:.4f} exceeds {thresholds.max_segment_length:.4f}"
            )
    geometry_ok = not segment_problems
    reasons.extend(segment_problems)

    temporal_ok = True
    if jumps.basis == "bootstrap":
        temporal_ok = not thresholds.require_previous_frame
        if not temporal_ok:
            reasons.append("no previous frame supports temporal continuity")
    else:
        pelvis_jump = jumps.pelvis_jump
        if pelvis_jump is not None and pelvis_jump > thresholds.max_pelvis_jump:
            temporal_ok = False
            reasons.append(
                f"pelvis jump {pelvis_jump:.4f} exceeds {thresholds.max_pelvis_jump:.4f}"
            )
        landmark_jump = jumps.max_landmark_jump
        if pelvis_jump is None and landmark_jump is None:
            temporal_ok = False
            reasons.append("previous frame shares no lower-body landmark")
        if landmark_jump is not None and landmark_jump > thresholds.max_landmark_jump:
            temporal_ok = False
            reasons.append(
                f"landmark jump {landmark_jump:.4f} on {jumps.max_jump_landmark} "
                f"exceeds {thresholds.max_landmark_jump:.4f}"
            )

    gates = {
        "chain_complete": chain_complete,
        "visibility_ok": visibility_ok,
        "depth_ok": depth_ok,
        "geometry_ok": geometry_ok,
        "temporal_ok": temporal_ok,
    }
    return GateOutcome(
        gates=gates,
        reasons=tuple(reasons),
        missing_required=missing,
        min_required_visibility=min_visibility,
        max_required_abs_z=max_abs_z,
    )


def evaluate_ownership(
    frames: Sequence[PoseFrame],
    thresholds: OwnershipThresholds | None = None,
) -> OwnershipReport:
    """Score every pose instance of every frame against the ownership gates."""
    limits = thresholds or OwnershipThresholds()
    previous: dict[str, tuple[int, LowerBodyGeometry, bool]] = {}
    frame_records: list[FrameOwnership] = []
    for frame in frames:
        decisions: list[OwnershipDecision] = []
        for pose in frame.poses:
            geometry = extract_lower_body(frame, pose)
            jumps = _measure_jumps(geometry, previous.get(pose.pose_id))
            outcome = _evaluate_gates(geometry, limits, jumps)
            decisions.append(
                OwnershipDecision(
                    frame_index=frame.frame_index,
                    pose_id=pose.pose_id,
                    owned=outcome.owned,
                    gates=outcome.gates,
                    reasons=outcome.reasons,
                    missing_required=outcome.missing_required,
                    min_required_visibility=outcome.min_required_visibility,
                    max_required_abs_z=outcome.max_required_abs_z,
                    geometry=geometry,
                    jumps=jumps,
                )
            )
        for pose, decision in zip(frame.poses, decisions, strict=True):
            previous[pose.pose_id] = (
                frame.frame_index,
                decision.geometry,
                decision.owned,
            )
        frame_records.append(FrameOwnership(frame_index=frame.frame_index, poses=tuple(decisions)))
    return OwnershipReport(frames=tuple(frame_records), thresholds=limits)


def build_report(
    document: object,
    thresholds: OwnershipThresholds | None = None,
) -> OwnershipReport:
    """Parse a pose document and score it in one step."""
    return evaluate_ownership(parse_pose_frames(document), thresholds)


NON_CLAIMS: tuple[str, ...] = (
    "Pose source is MediaPipe Pose Landmarker output only; DWPose and RTMPose "
    "keypoint schemas are not accepted.",
    "Ownership is a geometric gate decision over pose landmarks, not a "
    "segmentation, mask, or tracking claim.",
    "Pose identity across frames uses the exporter pose id or the per-frame pose "
    "index; no appearance-based re-identification is performed.",
    "Lengths are normalized image-width units; the y axis is aspect-corrected "
    "only when the document supplies a width and a height.",
    "MediaPipe z is a relative depth estimate in x-scale units, not a metric distance.",
    "Jump thresholds are per frame, not per second; no frame rate is inferred.",
    "Landmark depth is required input; a landmark without z is refused, so an "
    "absent depth is never reported as a measured zero.",
)


def _round(value: float) -> float:
    rounded = round(float(value), ROUNDING_DIGITS)
    return rounded + 0.0


def _landmark_payload(landmark: Landmark) -> dict[str, float]:
    return {
        "x": _round(landmark.x),
        "y": _round(landmark.y),
        "z": _round(landmark.z),
        "visibility": _round(landmark.visibility),
    }


def _optional_round(value: float | None) -> float | None:
    return None if value is None else _round(value)


def report_payload(report: OwnershipReport) -> dict[str, Any]:
    """Build the strict JSON payload for one ownership report."""
    frames: list[dict[str, Any]] = []
    for frame in report.frames:
        poses: list[dict[str, Any]] = []
        for decision in frame.poses:
            geometry = decision.geometry
            poses.append(
                {
                    "pose_id": decision.pose_id,
                    "owned": decision.owned,
                    "gates": dict(sorted(decision.gates.items())),
                    "reasons": list(decision.reasons),
                    "missing_required": list(decision.missing_required),
                    "min_required_visibility": _round(decision.min_required_visibility),
                    "max_required_abs_z": _round(decision.max_required_abs_z),
                    "pelvis": (
                        None if geometry.pelvis is None else _landmark_payload(geometry.pelvis)
                    ),
                    "landmarks": {
                        name: _landmark_payload(landmark)
                        for name, landmark in sorted(geometry.landmarks.items())
                    },
                    "segments": {
                        name: {
                            "start": segment.start,
                            "end": segment.end,
                            "vector": [_round(component) for component in segment.vector],
                            "length": _round(segment.length),
                            "min_visibility": _round(segment.min_visibility),
                            "z_delta": _round(segment.z_delta),
                        }
                        for name, segment in sorted(geometry.segments.items())
                    },
                    "jumps": {
                        "basis": decision.jumps.basis,
                        "previous_frame_index": decision.jumps.previous_frame_index,
                        "previous_owned": decision.jumps.previous_owned,
                        "pelvis_jump": _optional_round(decision.jumps.pelvis_jump),
                        "max_landmark_jump": _optional_round(decision.jumps.max_landmark_jump),
                        "max_jump_landmark": decision.jumps.max_jump_landmark,
                        "y_scale": _round(decision.jumps.y_scale),
                        "previous_y_scale": _optional_round(decision.jumps.previous_y_scale),
                    },
                    "aspect_corrected": geometry.aspect_corrected,
                }
            )
        frames.append(
            {
                "frame_index": frame.frame_index,
                "owned_pose_ids": list(frame.owned_pose_ids),
                "poses": poses,
            }
        )
    thresholds = report.thresholds
    frame_count = len(report.frames)
    owned_frames = report.owned_frame_count
    return {
        "schema": REPORT_SCHEMA,
        "generator": REPORT_GENERATOR,
        "pose_source": {
            "label": POSE_SOURCE_LABEL,
            "landmark_model": POSE_LANDMARK_MODEL,
            "is_dwpose": IS_DWPOSE,
            "is_rtmpose": IS_RTMPOSE,
            "metric_space": (
                "normalized_width_units"
                if all(
                    pose.geometry.aspect_corrected
                    for frame in report.frames
                    for pose in frame.poses
                )
                else "mixed_normalized_units"
            ),
        },
        "thresholds": {
            "min_visibility": _round(thresholds.min_visibility),
            "max_abs_z": _round(thresholds.max_abs_z),
            "min_segment_length": _round(thresholds.min_segment_length),
            "max_segment_length": _round(thresholds.max_segment_length),
            "max_pelvis_jump": _round(thresholds.max_pelvis_jump),
            "max_landmark_jump": _round(thresholds.max_landmark_jump),
            "required_chains": list(thresholds.required_chains),
            "require_previous_frame": thresholds.require_previous_frame,
        },
        "summary": {
            "frames": frame_count,
            "poses": report.pose_count,
            "owned_poses": report.owned_pose_count,
            "owned_frames": owned_frames,
            "frame_ownership_ratio": _round(owned_frames / frame_count) if frame_count else 0.0,
        },
        "frames": frames,
        "non_claims": list(NON_CLAIMS),
    }
