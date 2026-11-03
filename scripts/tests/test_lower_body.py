"""Contracts for per-frame lower-body ownership from MediaPipe pose JSON."""

from __future__ import annotations

import math
from typing import Any

import pytest

from samurai.lower_body import (
    LOWER_BODY_CHAINS,
    MEDIAPIPE_POSE_LANDMARK_NAMES,
    LowerBodyError,
    OwnershipThresholds,
    build_report,
    evaluate_ownership,
    extract_lower_body,
    parse_pose_frames,
    report_payload,
)

BASE_COORDS = {
    "left_hip": (0.45, 0.55),
    "right_hip": (0.55, 0.55),
    "left_knee": (0.45, 0.70),
    "right_knee": (0.55, 0.70),
    "left_ankle": (0.45, 0.85),
    "right_ankle": (0.55, 0.85),
    "left_heel": (0.44, 0.88),
    "right_heel": (0.56, 0.88),
    "left_foot_index": (0.40, 0.90),
    "right_foot_index": (0.60, 0.90),
}


def pose_payload(
    *,
    drop: tuple[str, ...] = (),
    visibility: float = 0.9,
    z: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    override: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, object]]:
    """Build a MediaPipe pose in the canonical 33-landmark order."""
    overrides = override or {}
    payload: list[dict[str, object]] = []
    for index, name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES):
        if name in drop:
            continue
        if name in overrides:
            x, y = overrides[name]
        elif index < 23:
            x, y = 0.5, 0.2
        else:
            x, y = BASE_COORDS[name]
        entry: dict[str, object] = {
            "x": x + dx,
            "y": y + dy,
            "z": z,
            "visibility": visibility,
        }
        if drop:
            entry["name"] = name
        payload.append(entry)
    return payload


def pose_document(*frames: list[dict[str, object]]) -> dict[str, object]:
    """Wrap pose payloads into a canonical pose document."""
    return {
        "schema": "mediapipe-pose-frames-v1",
        "frames": [
            {"frame_index": index, "poses": [{"landmarks": pose}]}
            for index, pose in enumerate(frames)
        ],
    }


def scaled_pose_document(
    frames: tuple[list[dict[str, object]], ...],
    dimensions: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    """Wrap pose payloads in frames that each carry their own image geometry."""
    assert len(frames) == len(dimensions)
    return {
        "frames": [
            {
                "frame_index": index,
                "width": width,
                "height": height,
                "poses": [{"landmarks": pose}],
            }
            for index, ((width, height), pose) in enumerate(zip(dimensions, frames, strict=True))
        ],
    }


def test_valid_chains_yield_an_owned_lower_body() -> None:
    report = build_report(pose_document(pose_payload()))

    assert report.owned_pose_count == 1
    assert report.owned_frame_count == 1
    decision = report.frames[0].poses[0]
    assert decision.owned is True
    assert decision.gates == {
        "chain_complete": True,
        "visibility_ok": True,
        "depth_ok": True,
        "geometry_ok": True,
        "temporal_ok": True,
    }
    assert decision.reasons == ()


def test_pelvis_midpoint_and_bone_spans_are_derived() -> None:
    frames = parse_pose_frames(pose_document(pose_payload()))
    geometry = extract_lower_body(frames[0], frames[0].poses[0])

    assert geometry.pelvis is not None
    assert geometry.pelvis.x == pytest.approx(0.50)
    assert geometry.pelvis.y == pytest.approx(0.55)
    assert geometry.pelvis.visibility == pytest.approx(0.9)

    thigh = geometry.segments["left_thigh"]
    assert thigh.start == "left_hip"
    assert thigh.end == "left_knee"
    assert thigh.vector == pytest.approx((0.0, 0.15, 0.0))
    assert thigh.length == pytest.approx(0.15)

    shin = geometry.segments["left_shin"]
    assert shin.length == pytest.approx(0.15)

    foot = geometry.segments["left_foot"]
    assert foot.start == "left_ankle"
    assert foot.end == "left_foot_index"
    assert foot.length == pytest.approx(math.sqrt(0.05**2 + 0.05**2))

    heel_span = geometry.segments["left_heel_span"]
    assert heel_span.length == pytest.approx(math.sqrt(0.04**2 + 0.02**2))


def test_visibility_and_depth_are_preserved_through_the_report() -> None:
    report = build_report(pose_document(pose_payload(visibility=0.42, z=-0.31)))
    payload = report_payload(report)
    pose = payload["frames"][0]["poses"][0]

    assert pose["landmarks"]["left_knee"]["visibility"] == pytest.approx(0.42)
    assert pose["landmarks"]["left_knee"]["z"] == pytest.approx(-0.31)
    assert pose["segments"]["left_thigh"]["min_visibility"] == pytest.approx(0.42)
    assert pose["segments"]["left_thigh"]["z_delta"] == pytest.approx(0.0)
    assert pose["max_required_abs_z"] == pytest.approx(0.31)


def test_missing_foot_fails_the_chain_closed() -> None:
    report = build_report(pose_document(pose_payload(drop=("left_foot_index",))))
    decision = report.frames[0].poses[0]

    assert decision.owned is False
    assert decision.gates["chain_complete"] is False
    assert decision.gates["geometry_ok"] is False
    assert decision.missing_required == ("left_foot_index",)
    assert any("left_foot_index" in reason for reason in decision.reasons)
    assert report.owned_pose_count == 0


def test_missing_knee_breaks_the_leg_chain_and_the_segment() -> None:
    report = build_report(pose_document(pose_payload(drop=("left_knee",))))
    decision = report.frames[0].poses[0]

    assert decision.owned is False
    assert decision.gates["chain_complete"] is False
    assert "left_knee" in decision.missing_required
    assert "left_thigh is not measurable" in decision.reasons
    assert "left_shin is not measurable" in decision.reasons


def test_low_visibility_fails_the_visibility_gate() -> None:
    report = build_report(pose_document(pose_payload(visibility=0.2)))
    decision = report.frames[0].poses[0]

    assert decision.owned is False
    assert decision.gates["visibility_ok"] is False
    assert decision.gates["chain_complete"] is True
    assert decision.min_required_visibility == pytest.approx(0.2)
    assert any("visibility" in reason for reason in decision.reasons)


def test_deep_landmark_fails_the_depth_gate() -> None:
    report = build_report(pose_document(pose_payload(z=2.5)))
    decision = report.frames[0].poses[0]

    assert decision.owned is False
    assert decision.gates["depth_ok"] is False
    assert decision.max_required_abs_z == pytest.approx(2.5)


def test_degenerate_segment_fails_the_geometry_gate() -> None:
    collapsed = {"left_knee": BASE_COORDS["left_hip"]}
    report = build_report(pose_document(pose_payload(override=collapsed)))
    decision = report.frames[0].poses[0]

    assert decision.owned is False
    assert decision.gates["geometry_ok"] is False
    assert any("left_thigh length" in reason for reason in decision.reasons)


def test_oversized_segment_fails_the_geometry_gate() -> None:
    report = build_report(
        pose_document(pose_payload()),
        OwnershipThresholds(max_segment_length=0.05),
    )
    decision = report.frames[0].poses[0]

    assert decision.owned is False
    assert decision.gates["geometry_ok"] is False


def test_temporal_jump_fails_the_temporal_gate() -> None:
    report = build_report(pose_document(pose_payload(), pose_payload(dx=0.9)))
    second = report.frames[1].poses[0]

    assert report.frames[0].poses[0].owned is True
    assert second.owned is False
    assert second.gates["temporal_ok"] is False
    assert second.jumps.basis == "previous_frame"
    assert second.jumps.previous_frame_index == 0
    assert second.jumps.pelvis_jump == pytest.approx(0.9)
    assert second.jumps.max_landmark_jump == pytest.approx(0.9)
    # Every lower-body landmark shifts by the same amount, so the tie resolves
    # to the first landmark in the stable lower-body order.
    assert second.jumps.max_jump_landmark == "left_hip"


def test_small_motion_keeps_the_temporal_gate_open() -> None:
    report = build_report(pose_document(pose_payload(), pose_payload(dx=0.01)))

    assert report.frames[1].poses[0].owned is True
    assert report.frames[1].poses[0].jumps.pelvis_jump == pytest.approx(0.01)


def test_cross_frame_jump_uses_each_frames_own_vertical_scale() -> None:
    document = scaled_pose_document(
        (pose_payload(), pose_payload()),
        ((640, 480), (640, 960)),
    )

    report = build_report(document)
    second = report.frames[1].poses[0]

    assert second.jumps.y_scale == pytest.approx(1.5)
    assert second.jumps.previous_y_scale == pytest.approx(0.75)
    assert second.jumps.pelvis_jump == pytest.approx(0.55 * (1.5 - 0.75))
    assert second.jumps.max_landmark_jump == pytest.approx(0.90 * (1.5 - 0.75))
    assert second.jumps.max_jump_landmark == "left_foot_index"
    assert second.gates["temporal_ok"] is False
    assert second.owned is False
    assert any("landmark jump" in reason for reason in second.reasons)


def test_cross_frame_jump_distance_is_symmetric_across_opposite_scale_changes() -> None:
    document = scaled_pose_document(
        (pose_payload(), pose_payload()),
        ((640, 960), (640, 480)),
    )

    report = build_report(document)
    second = report.frames[1].poses[0]

    assert second.jumps.y_scale == pytest.approx(0.75)
    assert second.jumps.previous_y_scale == pytest.approx(1.5)
    assert second.jumps.pelvis_jump == pytest.approx(0.55 * (1.5 - 0.75))
    assert second.jumps.max_landmark_jump == pytest.approx(0.90 * (1.5 - 0.75))
    assert second.gates["temporal_ok"] is False


def test_jump_distance_is_unchanged_when_both_frames_share_one_scale() -> None:
    document = scaled_pose_document(
        (pose_payload(), pose_payload(dx=0.01)),
        ((640, 480), (640, 480)),
    )

    report = build_report(document)
    second = report.frames[1].poses[0]

    assert second.jumps.pelvis_jump == pytest.approx(0.01)
    assert second.jumps.max_landmark_jump == pytest.approx(0.01)
    assert second.gates["temporal_ok"] is True
    assert second.owned is True


def test_first_frame_reports_a_bootstrap_temporal_basis() -> None:
    report = build_report(pose_document(pose_payload()))
    jumps = report.frames[0].poses[0].jumps

    assert jumps.basis == "bootstrap"
    assert jumps.previous_frame_index is None
    assert jumps.pelvis_jump is None
    assert report.frames[0].poses[0].gates["temporal_ok"] is True


def test_require_previous_frame_refuses_the_bootstrap_frame() -> None:
    report = build_report(
        pose_document(pose_payload(), pose_payload()),
        OwnershipThresholds(require_previous_frame=True),
    )

    assert report.frames[0].poses[0].owned is False
    assert report.frames[0].poses[0].gates["temporal_ok"] is False
    assert report.frames[1].poses[0].owned is True


def test_required_chains_select_which_chains_gate_ownership() -> None:
    thresholds = OwnershipThresholds(required_chains=("left_leg",))

    assert thresholds.required_landmarks == ("left_hip", "left_knee", "left_ankle")
    assert thresholds.required_segments == ("left_thigh", "left_shin")

    report = build_report(
        pose_document(pose_payload(drop=("right_foot_index",))),
        thresholds,
    )
    decision = report.frames[0].poses[0]

    assert decision.owned is True
    assert decision.missing_required == ()


def test_aspect_correction_scales_the_vertical_axis() -> None:
    document = pose_document(pose_payload())
    document["width"] = 640
    document["height"] = 480

    frames = parse_pose_frames(document)
    geometry = extract_lower_body(frames[0], frames[0].poses[0])

    assert geometry.aspect_corrected is True
    assert geometry.y_scale == pytest.approx(0.75)
    assert geometry.segments["left_thigh"].vector[1] == pytest.approx(0.15 * 0.75)
    assert geometry.segments["left_thigh"].length == pytest.approx(0.15 * 0.75)


def test_missing_frame_geometry_keeps_the_raw_vertical_scale() -> None:
    frames = parse_pose_frames(pose_document(pose_payload()))
    geometry = extract_lower_body(frames[0], frames[0].poses[0])

    assert geometry.aspect_corrected is False
    assert geometry.y_scale == pytest.approx(1.0)
    assert geometry.segments["left_thigh"].length == pytest.approx(0.15)


def test_multiple_pose_instances_keep_their_own_identity() -> None:
    document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [
                    {"pose_id": "a", "landmarks": pose_payload()},
                    {"pose_id": "b", "landmarks": pose_payload(dx=0.2)},
                ],
            }
        ]
    }
    report = build_report(document)

    assert report.pose_count == 2
    assert report.frames[0].owned_pose_ids == ("a", "b")


def test_unknown_landmark_name_is_refused() -> None:
    document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [{"landmarks": [{"name": "left_thigh", "x": 0.5, "y": 0.5}]}],
            }
        ]
    }

    with pytest.raises(LowerBodyError, match="unknown landmark"):
        parse_pose_frames(document)


def test_pose_world_landmarks_are_refused() -> None:
    document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [{"pose_world_landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}]}],
            }
        ]
    }

    with pytest.raises(LowerBodyError, match="landmarks or pose_landmarks"):
        parse_pose_frames(document)


def test_duplicate_pose_id_in_one_frame_is_refused() -> None:
    document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [
                    {"pose_id": "a", "landmarks": pose_payload()},
                    {"pose_id": "a", "landmarks": pose_payload()},
                ],
            }
        ]
    }

    with pytest.raises(LowerBodyError, match="repeats pose id"):
        parse_pose_frames(document)


def test_visibility_outside_the_unit_interval_is_refused() -> None:
    document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [
                    {
                        "landmarks": [
                            {
                                "name": "left_hip",
                                "x": 0.5,
                                "y": 0.5,
                                "z": 0.0,
                                "visibility": 1.4,
                            }
                        ]
                    }
                ],
            }
        ]
    }

    with pytest.raises(LowerBodyError, match="visibility must be within"):
        parse_pose_frames(document)


def test_a_landmark_without_depth_is_refused_fail_closed() -> None:
    document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [
                    {
                        "landmarks": [
                            {
                                "name": "left_hip",
                                "x": 0.5,
                                "y": 0.5,
                                "visibility": 0.9,
                            }
                        ]
                    }
                ],
            }
        ]
    }

    with pytest.raises(LowerBodyError, match="must carry x, y, and z"):
        parse_pose_frames(document)


def test_duplicate_frame_index_is_refused() -> None:
    document: dict[str, Any] = pose_document(pose_payload(), pose_payload())
    document["frames"][1]["frame_index"] = 0

    with pytest.raises(LowerBodyError, match="frame indices must strictly increase"):
        parse_pose_frames(document)


def test_non_increasing_frame_index_is_refused() -> None:
    document: dict[str, Any] = pose_document(pose_payload(), pose_payload())
    document["frames"][0]["frame_index"] = 7
    document["frames"][1]["frame_index"] = 5

    with pytest.raises(LowerBodyError, match="frame indices must strictly increase"):
        parse_pose_frames(document)


def test_a_document_without_a_frame_is_refused() -> None:
    with pytest.raises(LowerBodyError, match="carries no frame"):
        parse_pose_frames({"frames": []})


def test_a_frame_without_a_pose_is_retained_as_unowned() -> None:
    frames = parse_pose_frames({"frames": [{"frame_index": 0, "poses": []}]})

    assert len(frames) == 1
    assert frames[0].poses == ()


def test_thresholds_refuse_unknown_chains_and_impossible_bounds() -> None:
    with pytest.raises(LowerBodyError, match="unknown required chains"):
        OwnershipThresholds(required_chains=("tail",))
    with pytest.raises(LowerBodyError, match="max_segment_length"):
        OwnershipThresholds(min_segment_length=0.5, max_segment_length=0.2)
    with pytest.raises(LowerBodyError, match="min_visibility"):
        OwnershipThresholds(min_visibility=1.5)
    with pytest.raises(LowerBodyError, match="at least one required chain"):
        OwnershipThresholds(required_chains=())


def test_every_declared_chain_has_landmarks_and_segments() -> None:
    assert set(LOWER_BODY_CHAINS) == {"left_leg", "right_leg", "left_foot", "right_foot"}
    for chain in LOWER_BODY_CHAINS.values():
        assert len(chain.landmarks) == 3
        for name in chain.landmarks:
            assert name in MEDIAPIPE_POSE_LANDMARK_NAMES


def test_unbounded_integers_and_derived_geometry_fail_closed() -> None:
    huge_integer = 10**400
    number_document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [
                    {
                        "landmarks": [
                            {
                                "name": "left_hip",
                                "x": huge_integer,
                                "y": 0.5,
                                "z": 0.0,
                                "visibility": 0.9,
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(LowerBodyError, match="outside the finite float range"):
        parse_pose_frames(number_document)

    dimension_document = pose_document(pose_payload())
    dimension_document["width"] = 1
    dimension_document["height"] = huge_integer
    with pytest.raises(LowerBodyError, match="outside the finite float range"):
        parse_pose_frames(dimension_document)

    overflow = {
        "left_hip": (-1e308, 0.55),
        "left_knee": (1e308, 0.70),
    }
    with pytest.raises(LowerBodyError, match="outside the finite float range"):
        build_report(pose_document(pose_payload(override=overflow)))


def test_reappearance_without_a_shared_lower_body_landmark_fails_temporally() -> None:
    document = {
        "frames": [
            {
                "frame_index": 0,
                "poses": [
                    {
                        "pose_id": "a",
                        "landmarks": [
                            {"name": "nose", "x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.9}
                        ],
                    }
                ],
            },
            {
                "frame_index": 1,
                "poses": [{"pose_id": "a", "landmarks": pose_payload()}],
            },
        ]
    }

    report = build_report(document, OwnershipThresholds(require_previous_frame=True))
    second = report.frames[1].poses[0]

    assert second.gates["temporal_ok"] is False
    assert second.owned is False
    assert "previous frame shares no lower-body landmark" in second.reasons


def test_report_metric_space_distinguishes_unscaled_vertical_units() -> None:
    mixed = report_payload(build_report(pose_document(pose_payload())))
    width_units_document = pose_document(pose_payload())
    width_units_document["width"] = 640
    width_units_document["height"] = 480
    width_units = report_payload(build_report(width_units_document))

    assert mixed["pose_source"]["metric_space"] == "mixed_normalized_units"
    assert width_units["pose_source"]["metric_space"] == "normalized_width_units"


def test_report_payload_labels_the_mediapipe_source_only() -> None:
    payload = report_payload(build_report(pose_document(pose_payload())))

    assert payload["schema"] == "lower-body-ownership-report-v1"
    assert payload["pose_source"]["label"] == "mediapipe-pose-landmarker"
    assert payload["pose_source"]["is_dwpose"] is False
    assert payload["pose_source"]["is_rtmpose"] is False
    assert payload["summary"]["owned_poses"] == 1
    assert payload["non_claims"]


def test_report_payload_is_finite_and_rounded() -> None:
    payload = report_payload(build_report(pose_document(pose_payload())))
    pose = payload["frames"][0]["poses"][0]

    assert pose["segments"]["left_thigh"]["length"] == pytest.approx(0.15)
    assert math.isfinite(pose["pelvis"]["x"])
    assert math.isfinite(pose["jumps"]["pelvis_jump"] or 0.0)


def test_evaluate_ownership_accepts_parsed_frames() -> None:
    frames = parse_pose_frames(pose_document(pose_payload()))
    report = evaluate_ownership(frames)

    assert report.owned_pose_count == 1
    assert report.thresholds.required_chains == (
        "left_leg",
        "right_leg",
        "left_foot",
        "right_foot",
    )
