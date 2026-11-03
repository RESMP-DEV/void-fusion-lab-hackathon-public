"""Admit only owned lower-body frames into contact grounding.

Lower-body ownership is intentionally sparse and per frame.  It is not a
tracking claim and must never be expanded into one.  This module validates a
generated lower-body ownership report, retains only frames with exactly one
owned pose instance, and converts those normalized MediaPipe landmarks into
pixel keypoints for the existing contact-grounding kernel.

The adapter is fail closed.  A malformed ownership summary, inconsistent owned
IDs, a duplicate frame index, non-finite coordinates, or two owned pose
instances in one frame is refused instead of being resolved silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from samurai.contact_grounding import FOOT_LANDMARKS
from samurai.lower_body import LOWER_BODY_LANDMARK_NAMES

OWNERSHIP_REPORT_SCHEMA = "lower-body-ownership-report-v1"


class OwnedFrameContactError(RuntimeError):
    """An ownership report cannot be turned into contact-grounding input."""


@dataclass(frozen=True)
class OwnedFrame:
    """One frame with exactly one owned lower-body pose instance."""

    frame_index: int
    pose_id: str
    normalized_points: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class OwnershipSelection:
    """The sparse owned subset retained from one ownership report."""

    total_frames: int
    owned_frames: tuple[OwnedFrame, ...]

    @property
    def owned_frame_ratio(self) -> float:
        return len(self.owned_frames) / self.total_frames


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OwnedFrameContactError(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise OwnedFrameContactError(f"{label} must be a JSON array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OwnedFrameContactError(f"{label} must be an integer at least {minimum}")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OwnedFrameContactError(f"{label} must be a nonempty string")
    return value


def _coordinate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OwnedFrameContactError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise OwnedFrameContactError(f"{label} is outside the finite float range") from error
    if not math.isfinite(number):
        raise OwnedFrameContactError(f"{label} must be finite")
    return number


def _owned_points(payload: Any, frame_index: int, pose_id: str) -> dict[str, tuple[float, float]]:
    raw_points = _mapping(payload, f"owned frame {frame_index} pose {pose_id} landmarks")
    points: dict[str, tuple[float, float]] = {}
    for name, raw_point in raw_points.items():
        if not isinstance(name, str) or name not in LOWER_BODY_LANDMARK_NAMES:
            continue
        point = _mapping(raw_point, f"owned frame {frame_index} landmark {name}")
        if "x" not in point or "y" not in point:
            raise OwnedFrameContactError(
                f"owned frame {frame_index} landmark {name} must provide x and y"
            )
        points[name] = (
            _coordinate(point["x"], f"owned frame {frame_index} landmark {name} x"),
            _coordinate(point["y"], f"owned frame {frame_index} landmark {name} y"),
        )
    if not any(name in FOOT_LANDMARKS for name in points):
        raise OwnedFrameContactError(
            f"owned frame {frame_index} pose {pose_id} has no ground-bearing foot landmark"
        )
    return points


def parse_ownership_report(payload: Any) -> OwnershipSelection:
    """Validate one ownership report and retain only its owned frames."""
    report = _mapping(payload, "ownership report")
    if report.get("schema") != OWNERSHIP_REPORT_SCHEMA:
        raise OwnedFrameContactError(f"ownership report schema must be {OWNERSHIP_REPORT_SCHEMA!r}")
    raw_summary = _mapping(report.get("summary"), "ownership report summary")
    total_frames = _integer(raw_summary.get("frames"), "ownership summary frames", minimum=1)
    declared_poses = _integer(raw_summary.get("poses"), "ownership summary poses")
    declared_owned_poses = _integer(raw_summary.get("owned_poses"), "ownership summary owned_poses")
    declared_owned = _integer(raw_summary.get("owned_frames"), "ownership summary owned_frames")
    if declared_owned > total_frames:
        raise OwnedFrameContactError(
            "ownership summary claims more owned frames than source frames"
        )
    ratio = _coordinate(
        raw_summary.get("frame_ownership_ratio"), "ownership summary frame_ownership_ratio"
    )
    if not 0.0 <= ratio <= 1.0:
        raise OwnedFrameContactError("ownership summary frame_ownership_ratio must be in [0, 1]")
    expected_ratio = declared_owned / total_frames
    if abs(expected_ratio - ratio) > 0.5e-6:
        raise OwnedFrameContactError(
            "ownership summary frame_ownership_ratio does not match its frame counts"
        )

    raw_frames = _list(report.get("frames"), "ownership report frames")
    if len(raw_frames) != total_frames:
        raise OwnedFrameContactError(
            f"ownership report declares {total_frames} frames but carries {len(raw_frames)}"
        )
    owned: list[OwnedFrame] = []
    seen_frames: set[int] = set()
    total_pose_count = 0
    owned_pose_count = 0
    for position, raw_frame in enumerate(raw_frames):
        frame = _mapping(raw_frame, f"ownership frame {position}")
        frame_index = _integer(frame.get("frame_index"), f"ownership frame {position} frame_index")
        if frame_index >= total_frames:
            raise OwnedFrameContactError(
                f"ownership frame {frame_index} is outside the {total_frames} frame report"
            )
        if frame_index in seen_frames:
            raise OwnedFrameContactError(f"ownership report repeats frame index {frame_index}")
        seen_frames.add(frame_index)
        raw_poses = _list(frame.get("poses"), f"ownership frame {frame_index} poses")
        owned_ids: list[str] = []
        owned_pose_records: list[dict[str, Any]] = []
        seen_pose_ids: set[str] = set()
        for pose_position, raw_pose in enumerate(raw_poses):
            pose = _mapping(raw_pose, f"ownership frame {frame_index} pose {pose_position}")
            pose_id = _identifier(pose.get("pose_id"), f"ownership frame {frame_index} pose_id")
            if pose_id in seen_pose_ids:
                raise OwnedFrameContactError(
                    f"ownership frame {frame_index} repeats pose id {pose_id}"
                )
            seen_pose_ids.add(pose_id)
            is_owned = pose.get("owned")
            if not isinstance(is_owned, bool):
                raise OwnedFrameContactError(
                    f"ownership frame {frame_index} pose {pose_id} owned must be boolean"
                )
            total_pose_count += 1
            if is_owned:
                owned_ids.append(pose_id)
                owned_pose_records.append(pose)
                owned_pose_count += 1
        declared_ids = [
            _identifier(value, f"ownership frame {frame_index} owned_pose_ids entry")
            for value in _list(
                frame.get("owned_pose_ids"), f"ownership frame {frame_index} owned_pose_ids"
            )
        ]
        if declared_ids != owned_ids:
            raise OwnedFrameContactError(
                f"ownership frame {frame_index} owned_pose_ids does not match its owned poses"
            )
        if not owned_ids:
            continue
        if len(owned_ids) != 1:
            raise OwnedFrameContactError(
                f"owned frame {frame_index} has {len(owned_ids)} owned poses; "
                "candidate contact grounding requires an unambiguous pose"
            )
        raw_landmarks = owned_pose_records[0].get("landmarks")
        if not isinstance(raw_landmarks, dict):
            raise OwnedFrameContactError(
                f"owned frame {frame_index} pose {owned_ids[0]} carries no landmark object"
            )
        owned.append(
            OwnedFrame(
                frame_index=frame_index,
                pose_id=owned_ids[0],
                normalized_points=_owned_points(raw_landmarks, frame_index, owned_ids[0]),
            )
        )
    if total_pose_count != declared_poses:
        raise OwnedFrameContactError(
            f"ownership summary declares {declared_poses} poses but carries {total_pose_count}"
        )
    if owned_pose_count != declared_owned_poses:
        raise OwnedFrameContactError(
            f"ownership summary declares {declared_owned_poses} owned poses "
            f"but carries {owned_pose_count}"
        )
    if len(owned) != declared_owned:
        raise OwnedFrameContactError(
            f"ownership summary declares {declared_owned} owned frames but carries {len(owned)}"
        )
    return OwnershipSelection(total_frames=total_frames, owned_frames=tuple(owned))


def contact_keypoint_payload(
    selection: OwnershipSelection, *, width: int, height: int
) -> dict[str, Any]:
    """Convert owned normalized landmarks to reindexed pixel keypoints.

    Contact grounding measures a dense mask list, so the sparse owned subset is
    reindexed from zero.  The original source frame index remains on the
    integration report and must not be inferred from this payload.
    """
    if width <= 0 or height <= 0:
        raise OwnedFrameContactError("candidate mask dimensions must be positive")
    rows: list[dict[str, Any]] = []
    for local_index, owned in enumerate(selection.owned_frames):
        points = {name: [x * width, y * height] for name, (x, y) in owned.normalized_points.items()}
        rows.append({"frame": local_index, "points": points})
    if not rows:
        raise OwnedFrameContactError("no owned frames are available for contact grounding")
    return {"frames": rows}
