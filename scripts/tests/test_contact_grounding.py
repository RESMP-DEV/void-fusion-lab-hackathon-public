from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import contact_grounding as contact_grounding_script
import numpy as np
import pytest
from contact_grounding import (
    ContactGroundingScriptError,
    directory_sha256,
    frame_directory_files,
    load_plate_frames,
    main,
)
from PIL import Image

from samurai.contact_grounding import (
    ContactGroundingError,
    GroundingSettings,
    KeypointFrame,
    ground_contact,
    parse_keypoint_frames,
    summarize,
    support_polygon,
)

FRAME_HEIGHT = 96
FRAME_WIDTH = 128
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def keypoints(frame: int = 0, *, ground_y: float = 80.0) -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame": frame,
                "points": {
                    "left_hip": [60.0, 40.0],
                    "right_hip": [68.0, 40.0],
                    "left_knee": [58.0, 60.0],
                    "right_knee": [70.0, 60.0],
                    "left_ankle": [56.0, ground_y],
                    "right_ankle": [72.0, ground_y],
                },
            }
        ]
    }


def object_mask(*, bottom: int = 80, top: int = 60, left: int = 56, right: int = 72) -> np.ndarray:
    mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)
    mask[top : bottom + 1, left:right] = True
    return mask


def plate(*, bright_to_the_right: bool = True) -> np.ndarray:
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    columns = np.arange(FRAME_WIDTH, dtype=np.float64)
    if bright_to_the_right:
        value = np.clip(columns * 1.5, 0, 255)
    else:
        value = np.clip((FRAME_WIDTH - 1 - columns) * 1.5, 0, 255)
    ramp = value.astype(np.uint8)
    frame[:, :, 0] = ramp[None, :]
    frame[:, :, 1] = ramp[None, :]
    frame[:, :, 2] = ramp[None, :]
    return frame


def test_support_polygon_spans_the_feet_and_the_contact_line() -> None:
    frames = parse_keypoint_frames(keypoints())

    support = support_polygon(frames[0], GroundingSettings())

    assert support.area_px > 0.0
    assert len(support.points) >= 3
    assert support.ground_y == pytest.approx(80.0)
    assert support.contact_slope == pytest.approx(0.0, abs=1e-9)
    assert support.planted_foot_ratio == pytest.approx(1.0)
    assert support.ground_y_at(64.0) == pytest.approx(80.0)


def test_support_polygon_reports_a_raised_foot_as_not_ground_bearing() -> None:
    payload = keypoints()
    payload["frames"][0]["points"]["left_ankle"] = [56.0, 64.0]

    support = support_polygon(parse_keypoint_frames(payload)[0], GroundingSettings())

    assert support.ground_samples == 1
    assert support.planted_foot_ratio == pytest.approx(0.5)
    assert support.area_px > 0.0


def test_feet_at_different_depths_produce_a_tilted_contact_line() -> None:
    payload = {
        "frames": [
            {
                "frame": 0,
                "points": {
                    "left_hip": [60.0, 100.0],
                    "right_hip": [68.0, 100.0],
                    "left_knee": [58.0, 150.0],
                    "right_knee": [70.0, 150.0],
                    "left_ankle": [56.0, 200.0],
                    "right_ankle": [72.0, 192.0],
                },
            }
        ]
    }

    support = support_polygon(parse_keypoint_frames(payload)[0], GroundingSettings())

    assert support.planted_foot_ratio == pytest.approx(1.0)
    assert support.contact_slope == pytest.approx(-0.5, abs=1e-6)
    assert support.ground_y_at(64.0) == pytest.approx(196.0, abs=1e-6)
    assert support.side_ground_levels == (("left", 200.0), ("right", 192.0))


def test_grounded_object_reports_full_contact_coverage() -> None:
    measured, shadows = ground_contact(parse_keypoint_frames(keypoints()), [object_mask()])

    frame = measured[0]
    assert frame.contact_coverage == pytest.approx(1.0)
    assert frame.contact_pixels == frame.footprint_pixels
    assert frame.vertical_aligned
    assert frame.vertical_offset_px == pytest.approx(1.0, abs=1e-6)
    assert frame.shadow.total_pixels > 0
    assert shadows[0].any()


def test_floating_object_reports_no_contact_and_a_vertical_offset() -> None:
    measured, _ = ground_contact(parse_keypoint_frames(keypoints()), [object_mask(bottom=70)])

    frame = measured[0]
    assert frame.contact_coverage == pytest.approx(0.0)
    assert frame.vertical_offset_px == pytest.approx(-9.0, abs=1e-6)
    assert not frame.vertical_aligned


def test_object_outside_the_support_polygon_reports_no_contact() -> None:
    measured, _ = ground_contact(
        parse_keypoint_frames(keypoints()), [object_mask(left=100, right=116)]
    )

    assert measured[0].contact_coverage == pytest.approx(0.0)
    assert measured[0].contact_pixels == 0


def test_light_direction_follows_the_plate_gradient() -> None:
    measured, _ = ground_contact(
        parse_keypoint_frames(keypoints()),
        [object_mask()],
        plate_frames=[plate(bright_to_the_right=True)],
    )

    light = measured[0].light
    assert light.source == "plate-luminance"
    assert light.direction_x == pytest.approx(1.0, abs=0.05)
    assert light.direction_y == pytest.approx(0.0, abs=0.05)
    assert light.confidence > 0.5
    assert light.samples > 0


def test_shadow_points_away_from_the_estimated_light() -> None:
    wide = object_mask(left=40, right=80)
    right_lit, right_shadow = ground_contact(
        parse_keypoint_frames(keypoints()),
        [wide],
        plate_frames=[plate(bright_to_the_right=True)],
    )
    left_lit, left_shadow = ground_contact(
        parse_keypoint_frames(keypoints()),
        [wide],
        plate_frames=[plate(bright_to_the_right=False)],
    )

    assert right_lit[0].shadow.direction_x < -0.9
    assert left_lit[0].shadow.direction_x > 0.9
    right_centroid = np.nonzero(right_shadow[0])[1].mean()
    left_centroid = np.nonzero(left_shadow[0])[1].mean()
    assert right_centroid < 60
    assert left_centroid > 60


def test_shadow_mask_never_covers_the_inserted_object() -> None:
    mask = object_mask()

    _, shadows = ground_contact(parse_keypoint_frames(keypoints()), [mask], plate_frames=[plate()])

    assert not np.any(shadows[0] & mask)


def test_missing_plate_leaves_the_light_estimate_unmeasured() -> None:
    measured, shadows = ground_contact(parse_keypoint_frames(keypoints()), [object_mask()])

    assert measured[0].light.source == "no-plate"
    assert measured[0].light.confidence == pytest.approx(0.0)
    assert measured[0].shadow.length_px == pytest.approx(0.0)
    assert measured[0].shadow.contact_pixels > 0
    assert shadows[0].any()


def test_keypoint_parsing_rejects_invalid_payloads() -> None:
    with pytest.raises(ContactGroundingError, match="nonempty frames list"):
        parse_keypoint_frames({"frames": []})
    with pytest.raises(ContactGroundingError, match="duplicate keypoint frame"):
        parse_keypoint_frames({"frames": [keypoints(0)["frames"][0], keypoints(0)["frames"][0]]})
    with pytest.raises(ContactGroundingError, match="nonnegative integer"):
        parse_keypoint_frames({"frames": [{"frame": "0", "points": {}}]})
    with pytest.raises(ContactGroundingError, match="no ground-bearing foot landmark"):
        parse_keypoint_frames({"frames": [{"frame": 0, "points": {"nose": [1.0, 2.0]}}]})
    with pytest.raises(ContactGroundingError, match="must be finite"):
        parse_keypoint_frames(
            {
                "frames": [
                    {
                        "frame": 0,
                        "points": {"left_ankle": [float("nan"), 2.0]},
                    }
                ]
            }
        )


def test_keypoint_parsing_ignores_unknown_landmarks() -> None:
    payload = keypoints()
    payload["frames"][0]["points"]["nose"] = [10.0, 10.0]
    payload["frames"][0]["points"]["left_heel"] = [54.0, 80.0]

    frames = parse_keypoint_frames(payload)

    assert len(frames) == 1
    assert "nose" not in frames[0].points
    assert frames[0].points["left_heel"] == (54.0, 80.0)


def test_missing_body_scale_landmarks_fail_closed() -> None:
    payload = {
        "frames": [
            {
                "frame": 0,
                "points": {"left_ankle": [56.0, 80.0], "right_ankle": [72.0, 80.0]},
            }
        ]
    }

    with pytest.raises(ContactGroundingError, match="hip or knee landmark"):
        ground_contact(parse_keypoint_frames(payload), [object_mask()])


def test_ground_contact_fails_closed_on_invalid_geometry() -> None:
    frames = parse_keypoint_frames(keypoints())
    with pytest.raises(ContactGroundingError, match="mask sequence is empty"):
        ground_contact(frames, [])
    with pytest.raises(ContactGroundingError, match="has no pixels"):
        ground_contact(frames, [np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)])
    with pytest.raises(ContactGroundingError, match="has shape"):
        ground_contact(frames, [object_mask(), np.zeros((10, 10), dtype=bool)])
    with pytest.raises(ContactGroundingError, match="outside the"):
        ground_contact(parse_keypoint_frames(keypoints(4)), [object_mask()])
    with pytest.raises(ContactGroundingError, match="outside the"):
        ground_contact(
            [KeypointFrame(frame=-1, points=parse_keypoint_frames(keypoints())[0].points)],
            [object_mask()],
        )
    with pytest.raises(ContactGroundingError, match="does not match the object mask"):
        ground_contact(
            frames,
            [object_mask()],
            plate_frames=[np.zeros((10, 10, 3), dtype=np.uint8)],
        )
    with pytest.raises(ContactGroundingError, match="usable pixels"):
        ground_contact(
            frames,
            [object_mask()],
            plate_frames=[plate()],
            settings=GroundingSettings(min_plate_samples=100000),
        )


def test_summary_reports_geometry_without_a_quality_verdict() -> None:
    measured, _ = ground_contact(parse_keypoint_frames(keypoints()), [object_mask()])

    summary = summarize(measured, mask_frames=1)

    assert summary["measured_frames"] == 1
    assert summary["frames_without_keypoints"] == 0
    assert summary["mean_contact_coverage"] == pytest.approx(1.0)
    forbidden = ("pass", "fail", "score", "quality", "verdict", "accept")
    assert not [key for key in summary if any(word in key.lower() for word in forbidden)]


def write_mask_directory(directory: Path, frames: int = 2) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        gray = np.where(object_mask(), 0, 255).astype(np.uint8)
        Image.fromarray(gray).save(directory / f"frame-{index:06d}.png")
    return directory


def write_plate_directory(directory: Path, frames: int = 2) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        Image.fromarray(plate()).save(directory / f"frame-{index:06d}.png")
    return directory


def test_cli_writes_a_report_and_a_shadow_mask(tmp_path: Path) -> None:
    mask_directory = write_mask_directory(tmp_path / "mask")
    plate_directory = write_plate_directory(tmp_path / "plate")
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(
        json.dumps(
            {
                "frames": [
                    keypoints(0)["frames"][0],
                    keypoints(1)["frames"][0],
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    code = main(
        [
            "--keypoints",
            str(keypoint_path),
            "--object-mask",
            str(mask_directory),
            "--plate",
            str(plate_directory),
            "--out",
            str(output),
            "--fps",
            "24",
        ]
    )

    assert code == 0
    report = json.loads((output / "contact-grounding-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "void-fusion-lab.contact-grounding.report.v1"
    assert report["summary"]["measured_frames"] == 2
    assert report["summary"]["mean_contact_coverage"] == pytest.approx(1.0)
    assert report["non_claims"]
    assert len(report["frames"]) == 2
    assert report["outputs"]["shadow_mask"]["sha256"]
    assert (output / "shadow-mask.mp4").is_file()


def test_cli_fails_closed_on_a_missing_keypoint_file(tmp_path: Path) -> None:
    with pytest.raises(ContactGroundingScriptError, match="keypoints file does not exist"):
        main(
            [
                "--keypoints",
                str(tmp_path / "absent.json"),
                "--object-mask",
                str(write_mask_directory(tmp_path / "mask")),
                "--out",
                str(tmp_path / "out"),
                "--fps",
                "24",
            ]
        )


def test_directory_inputs_record_a_deterministic_directory_digest(tmp_path: Path) -> None:
    mask_directory = write_mask_directory(tmp_path / "mask")
    identical_mask_directory = write_mask_directory(tmp_path / "mask-identical")
    changed_mask_directory = write_mask_directory(tmp_path / "mask-changed")
    changed = np.asarray(Image.open(changed_mask_directory / "frame-000000.png")).copy()
    changed[0, 0] = 1 if changed[0, 0] == 0 else 0
    Image.fromarray(changed).save(changed_mask_directory / "frame-000000.png")
    plate_directory = write_plate_directory(tmp_path / "plate")
    identical_plate_directory = write_plate_directory(tmp_path / "plate-identical")
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")

    def run(mask_directory_path: Path) -> dict[str, Any]:
        output = tmp_path / f"out-{mask_directory_path.name}"
        code = main(
            [
                "--keypoints",
                str(keypoint_path),
                "--object-mask",
                str(mask_directory_path),
                "--plate",
                str(plate_directory),
                "--out",
                str(output),
                "--fps",
                "24",
            ]
        )
        assert code == 0
        return json.loads((output / "contact-grounding-report.json").read_text(encoding="utf-8"))

    report = run(mask_directory)

    for name in ("object_mask", "plate"):
        recorded = report["inputs"][name]
        assert recorded["sha256"] is None
        assert re.fullmatch(r"[0-9a-f]{64}", recorded["directory_sha256"] or "")
    assert report["inputs"]["object_mask"]["directory_sha256"] == directory_sha256(
        sorted(identical_mask_directory.glob("*.png"))
    )
    assert report["inputs"]["plate"]["directory_sha256"] == directory_sha256(
        sorted(identical_plate_directory.glob("*.png"))
    )
    changed_report = run(changed_mask_directory)
    assert (
        changed_report["inputs"]["object_mask"]["directory_sha256"]
        != report["inputs"]["object_mask"]["directory_sha256"]
    )


def test_a_file_input_still_records_its_own_digest(tmp_path: Path) -> None:
    mask_directory = write_mask_directory(tmp_path / "mask")
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")
    output = tmp_path / "out"

    code = main(
        [
            "--keypoints",
            str(keypoint_path),
            "--object-mask",
            str(mask_directory),
            "--out",
            str(output),
            "--fps",
            "24",
        ]
    )

    assert code == 0
    report = json.loads((output / "contact-grounding-report.json").read_text(encoding="utf-8"))
    assert (
        report["inputs"]["keypoints"]["sha256"]
        == hashlib.sha256(keypoint_path.read_bytes()).hexdigest()
    )
    assert report["inputs"]["keypoints"]["directory_sha256"] is None
    assert report["inputs"]["object_mask"]["sha256"] is None


def test_a_failed_tool_run_exits_two_without_a_traceback(tmp_path: Path) -> None:
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")
    corrupt_video = tmp_path / "corrupt-object-mask.mp4"
    corrupt_video.write_bytes(b"this is not a video")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "contact_grounding.py"),
            "--keypoints",
            str(keypoint_path),
            "--object-mask",
            str(corrupt_video),
            "--out",
            str(tmp_path / "out"),
            "--fps",
            "24",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Contact grounding failed: exit" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_fails_closed_on_a_plate_shape_mismatch(tmp_path: Path) -> None:
    plate_directory = tmp_path / "plate"
    plate_directory.mkdir()
    for index in range(2):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
            plate_directory / f"frame-{index:06d}.png"
        )
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")

    with pytest.raises(ContactGroundingScriptError, match="but the object mask is"):
        main(
            [
                "--keypoints",
                str(keypoint_path),
                "--object-mask",
                str(write_mask_directory(tmp_path / "mask")),
                "--plate",
                str(plate_directory),
                "--out",
                str(tmp_path / "out"),
                "--fps",
                "24",
            ]
        )


def test_cli_round_trips_a_mask_video(tmp_path: Path) -> None:
    from samurai.media import decode_gray_frames, write_gray_video

    mask_video = tmp_path / "object-mask.mp4"
    write_gray_video(
        [np.where(object_mask(), 0, 255).astype(np.uint8) for _ in range(3)], mask_video, fps=24.0
    )
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(
        json.dumps(
            {
                "frames": [
                    keypoints(0)["frames"][0],
                    keypoints(1)["frames"][0],
                    keypoints(2)["frames"][0],
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    code = main(
        [
            "--keypoints",
            str(keypoint_path),
            "--object-mask",
            str(mask_video),
            "--out",
            str(output),
        ]
    )

    assert code == 0
    report = json.loads((output / "contact-grounding-report.json").read_text(encoding="utf-8"))
    assert report["inputs"]["object_mask"]["fps"] == pytest.approx(24.0)
    assert report["summary"]["measured_frames"] == 3
    with TemporaryDirectory(prefix=".samurai-contact-test-") as temporary:
        decoded = decode_gray_frames(output / "shadow-mask.mp4", Path(temporary))
    assert len(decoded) == 3
    assert set(np.unique(decoded[0])) <= {0, 255}
    assert np.any(decoded[0] == 0)


def test_invalid_settings_fail_closed() -> None:
    frames = parse_keypoint_frames(keypoints())

    with pytest.raises(ContactGroundingError, match="finite and nonnegative"):
        ground_contact(
            frames, [object_mask()], settings=GroundingSettings(shadow_length_ratio=-1.0)
        )
    with pytest.raises(ContactGroundingError, match="min_plate_samples must be positive"):
        ground_contact(frames, [object_mask()], settings=GroundingSettings(min_plate_samples=0))


def test_a_non_boolean_object_mask_fails_closed() -> None:
    frames = parse_keypoint_frames(keypoints())

    with pytest.raises(ContactGroundingError, match="must be a boolean array"):
        ground_contact(frames, [np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)])


def test_plate_video_frames_are_read_before_the_temporary_directory_is_removed(
    tmp_path: Path,
) -> None:
    from samurai.media import write_gray_video

    video = tmp_path / "plate.mp4"
    write_gray_video([np.full((48, 64), 200, dtype=np.uint8) for _ in range(3)], video, fps=24.0)

    frames = load_plate_frames(video, [0, 2])

    assert len(frames) == 3
    assert frames[1] is None
    for index in (0, 2):
        frame = frames[index]
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        assert float(frame.mean()) == pytest.approx(200.0, abs=3.0)


def test_plate_video_extracts_only_requested_frames(tmp_path, monkeypatch) -> None:
    from samurai.media import write_gray_video

    video = tmp_path / "plate.mp4"
    write_gray_video([np.full((48, 64), 200, dtype=np.uint8) for _ in range(3)], video, fps=24.0)
    filters: list[str] = []

    original_run = subprocess.run

    def run_with_select_probe(command: list[str], *args: Any, **kwargs: Any):
        if command[0] == "ffmpeg" and "-vf" in command:
            filters.append(command[command.index("-vf") + 1])
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_select_probe)

    frames = contact_grounding_script.load_plate_frames(video, [0, 2])

    assert filters == ["select=eq(n\\,0)+eq(n\\,2)"]
    assert [index for index, frame in enumerate(frames) if frame is not None] == [0, 2]


def test_a_huge_json_integer_coordinate_fails_closed() -> None:
    payload = keypoints(0)
    payload["frames"][0]["points"]["left_ankle"][0] = 10**400

    with pytest.raises(ContactGroundingError, match="coordinates must be numeric"):
        parse_keypoint_frames(payload)


def test_a_near_coincident_contact_line_falls_back_to_the_lowest_level() -> None:
    document = keypoints(0)
    document["frames"][0]["points"]["left_ankle"] = [56.0, 80.0]
    document["frames"][0]["points"]["right_ankle"] = [56.000002, 80.0]
    frames = parse_keypoint_frames(document)

    support = support_polygon(frames[0], GroundingSettings())

    assert support.contact_line_conditioned is False
    assert support.contact_slope == 0.0
    assert support.contact_intercept == pytest.approx(80.0)
    assert abs(support.ground_y_at(66.0) - 80.0) < 1e-9


def test_separated_object_bases_do_not_receive_shadow_in_the_gap() -> None:
    mask = object_mask()
    mask[76:81, 60:68] = False
    frames = parse_keypoint_frames(keypoints(0))

    measurements, shadows = ground_contact(frames, [mask])

    assert not shadows[0][79, 60:68].any()


def test_a_gapped_frame_directory_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "gap"
    directory.mkdir()
    for index in (0, 2):
        Image.fromarray(np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)).save(
            directory / f"frame-{index:06d}.png"
        )
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")

    with pytest.raises(ContactGroundingScriptError, match="not contiguous"):
        main(
            [
                "--keypoints",
                str(keypoint_path),
                "--object-mask",
                str(directory),
                "--out",
                str(tmp_path / "out"),
                "--fps",
                "24",
            ]
        )


def test_a_one_based_frame_directory_is_accepted_in_list_order(tmp_path: Path) -> None:
    directory = tmp_path / "one-based"
    directory.mkdir()
    for index in (1, 2, 3):
        Image.fromarray(np.full((FRAME_HEIGHT, FRAME_WIDTH), 128, dtype=np.uint8)).save(
            directory / f"frame-{index:06d}.png"
        )

    files = frame_directory_files(directory)

    assert [path.name for path in files] == [
        "frame-000001.png",
        "frame-000002.png",
        "frame-000003.png",
    ]


def test_a_stray_image_in_a_frame_directory_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "stray"
    directory.mkdir()
    gray = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    Image.fromarray(gray).save(directory / "frame-000000.png")
    Image.fromarray(gray).save(directory / "extra.png")
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")

    with pytest.raises(ContactGroundingScriptError, match="frame-NNNNNN.png"):
        main(
            [
                "--keypoints",
                str(keypoint_path),
                "--object-mask",
                str(directory),
                "--out",
                str(tmp_path / "out"),
                "--fps",
                "24",
            ]
        )


def test_an_overflowing_contact_line_fit_fails_closed() -> None:
    points = {
        "left_hip": [20.0, 60.0],
        "right_hip": [38.0, 60.0],
        "left_knee": [22.0, 80.0],
        "right_knee": [36.0, 80.0],
        "left_ankle": [30.0, 100.0],
        "right_ankle": [40.0, 100.0],
        "left_heel": [1e200, 104.0],
        "right_heel": [42.0, 104.0],
    }
    frames = parse_keypoint_frames({"frames": [{"frame": 0, "points": points}]})

    with pytest.raises(ContactGroundingError, match="contact-line fit overflowed"):
        support_polygon(frames[0], GroundingSettings())


def test_a_nonfinite_frame_rate_fails_closed(tmp_path: Path) -> None:
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")

    for value in ("nan", "inf"):
        with pytest.raises(ContactGroundingScriptError, match="frame rate must be finite"):
            main(
                [
                    "--keypoints",
                    str(keypoint_path),
                    "--object-mask",
                    str(write_mask_directory(tmp_path / "mask", frames=1)),
                    "--out",
                    str(tmp_path / "out"),
                    "--fps",
                    value,
                ]
            )


def test_overflowing_geometry_fails_closed() -> None:
    payload = keypoints()
    payload["frames"][0]["points"]["left_hip"] = [60.0, -1e308]
    payload["frames"][0]["points"]["right_hip"] = [68.0, -1e308]

    with pytest.raises(ContactGroundingError, match="must be finite"):
        ground_contact(parse_keypoint_frames(payload), [object_mask()])


def test_a_nonfinite_plate_luminance_fails_closed() -> None:
    for value in (float("nan"), float("inf")):
        plate_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float64)
        plate_frame[80:, 50:90, 0] = value

        with pytest.raises(ContactGroundingError, match="nonfinite luminance"):
            ground_contact(
                parse_keypoint_frames(keypoints()),
                [object_mask()],
                plate_frames=[plate_frame],
            )


def test_an_overflowing_plate_luminance_range_fails_closed() -> None:
    plate_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 1e308, dtype=np.float64)
    plate_frame[80:88, 50:90, :] = -1e308

    with pytest.raises(ContactGroundingError, match="must be finite"):
        ground_contact(
            parse_keypoint_frames(keypoints()),
            [object_mask()],
            plate_frames=[plate_frame],
        )


def test_a_one_code_unit_plate_gradient_is_flat_and_derives_no_cast_shadow() -> None:
    step = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    step[:, 64:, :] = 1

    measured, _ = ground_contact(
        parse_keypoint_frames(keypoints()),
        [object_mask()],
        plate_frames=[step],
    )

    light = measured[0].light
    assert light.source == "plate-luminance-flat"
    assert light.direction_x == 0.0
    assert light.direction_y == 0.0
    assert light.azimuth_degrees == 0.0
    assert light.confidence == 0.0
    assert measured[0].shadow.length_px == 0.0
    assert measured[0].shadow.cast_pixels == 0
    assert measured[0].shadow.contact_pixels > 0


def test_a_two_code_unit_plate_gradient_still_names_a_light_direction() -> None:
    step = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    step[:, 64:, :] = 2

    measured, _ = ground_contact(
        parse_keypoint_frames(keypoints()),
        [object_mask()],
        plate_frames=[step],
    )

    assert measured[0].light.source == "plate-luminance"
    assert measured[0].light.confidence > 0.0
    assert measured[0].shadow.cast_pixels > 0


def test_a_low_confidence_luminance_outlier_derives_no_cast_shadow() -> None:
    noisy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    noisy[84, 64, :] = 12

    measured, _ = ground_contact(
        parse_keypoint_frames(keypoints()),
        [object_mask()],
        plate_frames=[noisy],
    )

    assert measured[0].light.source == "plate-luminance"
    assert 0.0 < measured[0].light.confidence < GroundingSettings().min_light_confidence
    assert measured[0].shadow.length_px == 0.0
    assert measured[0].shadow.cast_pixels == 0
    assert measured[0].shadow.total_pixels == measured[0].shadow.contact_pixels


def test_the_minimum_light_confidence_is_validated() -> None:
    with pytest.raises(ContactGroundingError, match="min_light_confidence"):
        ground_contact(
            parse_keypoint_frames(keypoints()),
            [object_mask()],
            plate_frames=[plate()],
            settings=GroundingSettings(min_light_confidence=1.5),
        )


def test_the_footprint_centroid_is_the_column_mean_not_the_extrema_midpoint() -> None:
    skewed = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)
    skewed[0:81, 56:72] = True
    skewed[77:80, 57:72] = False

    measured, _ = ground_contact(parse_keypoint_frames(keypoints()), [skewed])

    # The occupied columns span 56..71, so the midpoint of the extreme columns
    # is 64.0 while the footprint's pixel-weighted column mean is 62.8158.
    assert measured[0].object_centroid_x == pytest.approx(62.81578947, abs=1e-6)
    solid, _ = ground_contact(parse_keypoint_frames(keypoints()), [object_mask()])
    assert solid[0].object_centroid_x == pytest.approx(64.0)


def test_omitted_frames_do_not_claim_a_shared_deviation_pattern() -> None:
    frames: list[Any] = []
    for index in range(12):
        frames.extend(parse_keypoint_frames(keypoints(index)))
    masks = [object_mask(bottom=70) for _ in range(12)]

    measured, _ = ground_contact(frames, masks)

    summary = summarize(measured, mask_frames=12)
    text = " ".join(summary["observations"])
    assert "same deviation pattern" not in text
    assert "2 further frames fall outside the vertical tolerance" in text


def test_clipping_to_the_ground_recomputes_all_shadow_counts() -> None:
    frames = parse_keypoint_frames(keypoints())
    floating_mask = [object_mask(bottom=70)]

    unclipped, _ = ground_contact(frames, floating_mask)
    clipped, _ = ground_contact(
        frames,
        floating_mask,
        settings=GroundingSettings(clip_shadow_to_ground=True),
    )

    assert unclipped[0].shadow.contact_pixels > 0
    assert clipped[0].shadow.above_contact_line_pixels > 0
    assert clipped[0].shadow.contact_pixels == 0
    assert clipped[0].shadow.cast_pixels == 0
    assert clipped[0].shadow.total_pixels == 0


def test_the_shadow_mask_is_roi_black_and_not_a_void_quadmask(tmp_path: Path) -> None:
    from samurai.media import decode_gray_frames

    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(json.dumps(keypoints(0)), encoding="utf-8")
    output = tmp_path / "out"

    code = main(
        [
            "--keypoints",
            str(keypoint_path),
            "--object-mask",
            str(write_mask_directory(tmp_path / "mask", frames=1)),
            "--out",
            str(output),
            "--fps",
            "24",
        ]
    )

    assert code == 0
    report = json.loads((output / "contact-grounding-report.json").read_text(encoding="utf-8"))
    shadow = report["outputs"]["shadow_mask"]
    assert shadow["mask_family"] == "roi-black"
    assert "quadmask" in shadow["convention"].lower()
    assert any("not a VOID quadmask" in claim for claim in report["non_claims"])
    with TemporaryDirectory(prefix=".samurai-contact-test-") as temporary:
        decoded = decode_gray_frames(output / "shadow-mask.mp4", Path(temporary))
    values = set(np.unique(decoded[0]).tolist())
    assert values <= {0, 255}
    assert 63 not in values
    assert 127 not in values


def test_the_cli_streams_gray_frames_and_never_builds_a_whole_clip_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_writer = contact_grounding_script.GrayVideoWriter
    written: list[tuple[int, ...]] = []

    class RecordingWriter(real_writer):
        def write(self, frame: np.ndarray) -> None:
            written.append(frame.shape)
            super().write(frame)

    monkeypatch.setattr(contact_grounding_script, "GrayVideoWriter", RecordingWriter)
    monkeypatch.setattr(
        contact_grounding_script,
        "write_gray_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("whole-clip writer used")),
        raising=False,
    )
    keypoint_path = tmp_path / "keypoints.json"
    keypoint_path.write_text(
        json.dumps(
            {
                "frames": [
                    keypoints(0)["frames"][0],
                    keypoints(1)["frames"][0],
                ]
            }
        ),
        encoding="utf-8",
    )

    code = contact_grounding_script.main(
        [
            "--keypoints",
            str(keypoint_path),
            "--object-mask",
            str(write_mask_directory(tmp_path / "mask", frames=2)),
            "--out",
            str(tmp_path / "out"),
            "--fps",
            "24",
        ]
    )

    assert code == 0
    assert written == [(FRAME_HEIGHT, FRAME_WIDTH), (FRAME_HEIGHT, FRAME_WIDTH)]
    assert (tmp_path / "out" / "shadow-mask.mp4").is_file()


def test_the_public_evidence_pins_the_contact_grounding_boundary() -> None:
    """Bind public contact evidence to its exact measured ownership scope."""
    payload = json.loads(
        (REPOSITORY_ROOT / "evidence" / "historical-full-chain-evidence.json").read_text(
            encoding="utf-8"
        )
    )

    # The public record pins the executed lane by revision and binds the
    # immutable outputs by SHA-256.
    execution = payload["real_full_chain"]["execution"]
    assert re.fullmatch(r"[0-9a-f]{40}", execution["void_revision"])
    contact = payload["real_full_chain"]["owned_frame_contact"]
    assert re.fullmatch(r"[0-9a-f]{64}", contact["report_sha256"])

    # The contact-grounding kernel that produced the published coverage is
    # public, so its recorded markers stay checkable in this checkout.
    source = (REPOSITORY_ROOT / "src" / "samurai" / "contact_grounding.py").read_text(
        encoding="utf-8"
    )
    assert "flat_luminance_range" in source
    assert "footprint's column mean" in source

    # The evidence, rather than the current product architecture narrative,
    # remains the authoritative record for this historical experiment.
    assert contact["owned_frames"] == [0, 1, 7, 13]
    assert contact["total_frames"] == 96
    assert contact["continuous_ownership_claimed"] is False

    # The public record carries no machine path. The token is
    # assembled here so this guard does not itself put a home-absolute path
    # into the published tree.
    assert ("/Us" + "ers/") not in json.dumps(payload)
