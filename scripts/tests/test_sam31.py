from __future__ import annotations

import numpy as np
import pytest

from samurai.cli import parser as cli_parser
from samurai.media import MediaError, VideoInfo, decode_gray_frames, write_gray_video
from samurai.sam31 import SegmentationPrompt, segment_video


class FakePredictor:
    def __init__(self):
        self.requests = []
        self.stream_object_id = 0

    def handle_request(self, request):
        self.requests.append(request)
        if request.get("type") == "add_prompt" and request.get("obj_id") is not None:
            self.stream_object_id = int(request["obj_id"])
        return {"session_id": "session"}

    def handle_stream_request(self, request):
        self.requests.append(request)
        for index in range(2):
            yield {
                "frame_index": index,
                "outputs": {
                    "out_binary_masks": np.zeros((1, 4, 6), dtype=bool),
                    "out_obj_ids": np.array([self.stream_object_id]),
                    "out_probs": np.array([0.75], dtype=np.float32),
                },
            }


def test_prompt_normalizes_absolute_coordinates():
    # R8 rework (W3): the old construction combined points with a box and
    # set no object_id, which the ported validate() rules now reject.
    # The normalization intent is preserved with two legal prompts: a
    # point prompt that carries object_id, and a box-only prompt.
    video = VideoInfo(width=100, height=200, fps=30, frames=3)
    prompt = SegmentationPrompt(
        frame_index=3,
        points=((10, 20),),
        point_labels=(1,),
        object_id=7,
    )
    request = prompt.request(video)
    assert request["points"] == [[0.1, 0.1]]
    assert request["point_labels"] == [1]
    assert request["obj_id"] == 7
    box_prompt = SegmentationPrompt(
        frame_index=3,
        boxes_xyxy=((10, 20, 30, 60),),
    )
    box_request = box_prompt.request(video)
    assert box_request["bounding_boxes"] == [[0.1, 0.1, 0.2, 0.2]]
    assert box_request["bounding_box_labels"] == [True]


def test_prompt_passes_object_id_for_point_refinement():
    prompt = SegmentationPrompt(
        frame_index=3,
        points=((10, 20),),
        point_labels=(1,),
        object_id=7,
    )
    video = VideoInfo(width=100, height=20, fps=30, frames=4)
    request = prompt.request(video)
    assert request["obj_id"] == 7
    assert request["frame_index"] == 3


def test_point_prompt_without_box_or_object_id_fails_closed():
    prompt = SegmentationPrompt(
        frame_index=0,
        points=((10, 20),),
        point_labels=(1,),
    )
    with pytest.raises(ValueError, match="point prompts require object_id"):
        prompt.validate()


def test_segment_video_forwards_object_id_without_caller_masks(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakePredictor()
    prompt = SegmentationPrompt(
        frame_index=0,
        points=((3, 2),),
        point_labels=(1,),
        object_id=7,
    )
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")

    add_prompt = predictor.requests[1]
    assert add_prompt["type"] == "add_prompt"
    assert add_prompt["obj_id"] == 7
    assert predictor.requests[2]["type"] == "propagate_in_video"
    assert result["object_id"] == 7
    assert result["referent"] == {
        "frame_index": 0,
        "text": None,
        "points": ((3, 2),),
        "point_labels": (1,),
        "boxes_xyxy": (),
        "object_id": 7,
        "collective": False,
    }


def test_segment_video_result_carries_referent_for_text_prompt(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakePredictor()
    prompt = SegmentationPrompt(frame_index=0, text="cup", collective=True)
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["object_id"] is None
    assert result["referent"] == {
        "frame_index": 0,
        "text": "cup",
        "points": (),
        "point_labels": (),
        "boxes_xyxy": (),
        "object_id": None,
        "collective": True,
    }


def test_segment_video_writes_all_frames_and_closes_session(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakePredictor()
    prompt = SegmentationPrompt(frame_index=0, text="cup", collective=True)
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["frames"] == 2
    assert result["model_version"] == "sam3.1"
    assert (tmp_path / "mask.mp4").is_file()
    assert predictor.requests[-1]["type"] == "close_session"


def test_segment_video_can_emit_probability_support_tensor(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakePredictor()
    prompt = SegmentationPrompt(frame_index=0, text="cup", collective=True)
    result = segment_video(
        predictor,
        video,
        prompt,
        tmp_path / "mask.mp4",
        tmp_path / "scores.npy",
    )
    scores = np.load(tmp_path / "scores.npy")
    assert scores.shape == (2, 4, 6)
    assert scores.dtype == np.float32
    assert result["scores_semantics"].startswith("per-pixel object probability")


# --- Referent lock: the prompt is the lock (R4 conformance, REAUDIT-R4) ---
#
# The lock key is the prompt itself: points + a stable object_id, or a
# box. There is no post-hoc selection over SAM's detections and no
# containment test. Score never selects: the score trap below is the
# clip C record in miniature (bystander 0.902344, subject 0.626465);
# the prompt names object_id 20, so the subject is kept.


def _two_person_masks():
    """(2, 4, 6) bool stack: index 0 bystander (right), index 1 subject (left)."""
    masks = np.zeros((2, 4, 6), dtype=bool)
    masks[0, :, 3:6] = True  # bystander, 12 px
    masks[1, :, 0:2] = True  # subject, 8 px; contains point (x=1, y=1)
    return masks


class FakeReferentPredictor:
    """Two detections at seeding; the higher score is the bystander."""

    def __init__(self, prompt_masks=None):
        self.requests = []
        self.prompt_masks = _two_person_masks() if prompt_masks is None else prompt_masks

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "add_prompt":
            count = int(self.prompt_masks.shape[0])
            return {
                "session_id": "session",
                "outputs": {
                    "out_binary_masks": self.prompt_masks,
                    "out_obj_ids": np.array([10, 20][:count]),
                    "out_probs": np.array([0.902344, 0.626465][:count], dtype=np.float32),
                },
            }
        return {"session_id": "session"}

    def handle_stream_request(self, request):
        self.requests.append(request)
        for index in range(2):
            yield {
                "frame_index": index,
                "outputs": {
                    "out_binary_masks": _two_person_masks(),
                    "out_obj_ids": np.array([10, 20]),
                    "out_probs": np.array([0.902344, 0.626465], dtype=np.float32),
                },
            }


def _points_prompt(point=(1, 1)):
    return SegmentationPrompt(
        frame_index=0,
        points=(point,),
        point_labels=(1,),
        object_id=20,
    )


def test_segment_video_referent_lock_keeps_only_subject(tmp_path):
    # Run-level score trap: the bystander scores 0.902344 and the
    # subject scores 0.626465. The prompt locks object_id 20 (the
    # subject); with no selection code present, the kept mask and the
    # score tensor carry the subject only.
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeReferentPredictor()
    prompt = _points_prompt()
    result = segment_video(
        predictor,
        video,
        prompt,
        tmp_path / "mask.mp4",
        tmp_path / "scores.npy",
    )
    assert result["referent_selection"] is None
    assert result["object_id"] == 20
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    assert len(frames) == 2
    for frame in frames:
        assert np.array_equal(frame > 0, _two_person_masks()[1])
    scores = np.load(tmp_path / "scores.npy")
    assert scores.shape == (2, 4, 6)
    assert not scores[:, :, 3:6].any()  # bystander support never enters the tensor
    assert np.allclose(scores[:, :, 0:2], 0.626465, atol=1e-6)


def test_segment_video_referent_gap_point_proceeds_and_keeps_object_id(tmp_path):
    # Conformance flip (REAUDIT-R4 D2): the point sits in the gap
    # column (2, 1), contained by no detection. No containment test
    # exists, so the run proceeds and keeps object_id 20's mask.
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeReferentPredictor()
    prompt = _points_prompt(point=(2, 1))
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["object_id"] == 20
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    assert len(frames) == 2
    for frame in frames:
        assert np.array_equal(frame > 0, _two_person_masks()[1])


def test_segment_video_referent_double_containment_proceeds_and_keeps_object_id(tmp_path):
    # Conformance flip (REAUDIT-R4 D2): both seeding masks contain the
    # point. No containment test exists, so the run proceeds and keeps
    # object_id 20's mask only.
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    masks = _two_person_masks()
    masks[0, :, 0:2] = True  # bystander seeding mask now also holds the point
    predictor = FakeReferentPredictor(prompt_masks=masks)
    prompt = _points_prompt()
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["object_id"] == 20
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    assert len(frames) == 2
    for frame in frames:
        assert np.array_equal(frame > 0, _two_person_masks()[1])


def test_segment_video_text_without_lock_or_collective_fails_closed(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeReferentPredictor()
    prompt = SegmentationPrompt(frame_index=0, text="person")
    with pytest.raises(ValueError, match="no prompt lock"):
        segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert not (tmp_path / "mask.mp4").exists()


def test_segment_video_text_with_box_proceeds_as_locked(tmp_path):
    # REAUDIT-R4 D3: a box is a strong spatial lock, so text+box with
    # collective=False proceeds instead of failing closed.
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeReferentPredictor()
    prompt = SegmentationPrompt(frame_index=0, text="person", boxes_xyxy=((0, 0, 2, 4),))
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["referent"]["boxes_xyxy"] == ((0, 0, 2, 4),)
    assert result["object_id"] == 20
    assert result["referent_selection"] == {
        "kind": "box",
        "object_id": 20,
        "candidate_object_ids": [10, 20],
        "selected_index": 1,
        "box_overlap_pixels": [0, 8],
        "detection_areas": [12, 8],
        "box_containment": [0.0, 1.0],
    }
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    for frame in frames:
        assert np.array_equal(frame > 0, _two_person_masks()[1])


def test_box_lock_ranks_containment_over_raw_crossing_area(tmp_path):
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, :, 1:8] = True  # larger crossing bystander: 24 pixels in the box
    masks[1, :4, 0:2] = True  # fully contained subject: 8 pixels in the box

    class CrossingBystanderPredictor(FakeReferentPredictor):
        def handle_stream_request(self, request):
            self.requests.append(request)
            for index in range(2):
                yield {
                    "frame_index": index,
                    "outputs": {
                        "out_binary_masks": masks,
                        "out_obj_ids": np.array([10, 20]),
                        "out_probs": np.array([0.9, 0.6], dtype=np.float32),
                    },
                }

    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((8, 8), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = CrossingBystanderPredictor(prompt_masks=masks)
    prompt = SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, 4, 8),))
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["object_id"] == 20
    assert result["referent_selection"]["box_overlap_pixels"] == [24, 8]
    assert result["referent_selection"]["box_containment"][0] < 0.5
    assert result["referent_selection"]["box_containment"][1] == 1.0


def test_segment_video_box_lock_fails_when_no_seeded_detection_overlaps(tmp_path):
    video = _two_frame_video(tmp_path)
    predictor = FakeReferentPredictor()
    prompt = SegmentationPrompt(frame_index=0, boxes_xyxy=((2, 0, 3, 4),))
    with pytest.raises(MediaError, match="box overlaps no seeded detection"):
        segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert not (tmp_path / "mask.mp4").exists()


def test_segment_video_box_lock_fails_on_ambiguous_seeded_detections(tmp_path):
    video = _two_frame_video(tmp_path)
    equal_containment_masks = np.zeros((2, 4, 6), dtype=bool)
    equal_containment_masks[0, :4, 0:2] = True
    equal_containment_masks[1, :4, 3:5] = True
    predictor = FakeReferentPredictor(prompt_masks=equal_containment_masks)
    prompt = SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, 5, 4),))
    with pytest.raises(MediaError, match="box cannot separate seeded detections"):
        segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert not (tmp_path / "mask.mp4").exists()


def test_segment_video_empty_box_seeding_fails_closed(tmp_path):
    class EmptySeedingPredictor(FakeReferentPredictor):
        def handle_request(self, request):
            self.requests.append(request)
            if request["type"] == "add_prompt":
                return {
                    "session_id": "session",
                    "outputs": {
                        "out_binary_masks": np.zeros((0, 4, 6), dtype=bool),
                        "out_obj_ids": np.array([], dtype=np.int64),
                        "out_probs": np.array([], dtype=np.float32),
                    },
                }
            return {"session_id": "session"}

    video = _two_frame_video(tmp_path)
    predictor = EmptySeedingPredictor()
    prompt = SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, 2, 4),))
    with pytest.raises(MediaError, match="prompt response seeded no detection"):
        segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert not (tmp_path / "mask.mp4").exists()


def test_single_low_containment_box_seed_fails_closed(tmp_path):
    masks = np.zeros((1, 4, 6), dtype=bool)
    masks[0, :, :] = True  # four of 24 pixels lie inside the box
    predictor = FakeReferentPredictor(prompt_masks=masks)
    prompt = SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, 1, 4),))
    with pytest.raises(MediaError, match="less than half contained"):
        segment_video(predictor, _two_frame_video(tmp_path), prompt, tmp_path / "mask.mp4")
    assert not (tmp_path / "mask.mp4").exists()


def test_segment_video_box_only_and_points_only_proceed(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    box_predictor = FakeReferentPredictor()
    box_prompt = SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, 2, 4),))
    box_result = segment_video(box_predictor, video, box_prompt, tmp_path / "box.mp4")
    assert box_result["referent"]["boxes_xyxy"] == ((0, 0, 2, 4),)
    points_predictor = FakeReferentPredictor()
    points_result = segment_video(
        points_predictor, video, _points_prompt(), tmp_path / "points.mp4"
    )
    assert points_result["object_id"] == 20


# --- Set A new-contract tests (W2, R5+R12+R13), re-keyed by REAUDIT-R4 ---


def test_cli_target_point_removed_and_prompt_fields_parse():
    # --target-point is removed: it is now an unrecognized argument.
    with pytest.raises(SystemExit):
        cli_parser().parse_args(
            [
                "segment",
                "video.mp4",
                "--text",
                "person",
                "--target-point",
                "1,2",
                "--output",
                "mask.mp4",
            ]
        )
    args = cli_parser().parse_args(
        [
            "segment",
            "video.mp4",
            "--point",
            "1,1",
            "--point-label",
            "1",
            "--object-id",
            "20",
            "--output",
            "mask.mp4",
        ]
    )
    assert args.point == [(1.0, 1.0)]
    assert args.point_label == [1]
    assert args.object_id == 20
    prompt = SegmentationPrompt(
        frame_index=args.frame_index,
        points=tuple(args.point or ()),
        point_labels=tuple(args.point_label or ()),
        object_id=args.object_id,
    )
    assert prompt.points == ((1.0, 1.0),)
    assert prompt.point_labels == (1,)
    assert prompt.object_id == 20
    box_args = cli_parser().parse_args(
        ["segment", "video.mp4", "--box", "0,0,2,4", "--output", "mask.mp4"]
    )
    assert box_args.box == [(0.0, 0.0, 2.0, 4.0)]
    args = cli_parser().parse_args(
        ["segment", "video.mp4", "--text", "person", "--collective", "--output", "mask.mp4"]
    )
    assert args.collective is True


def test_segment_video_collective_text_prompt_unions_and_marks_collective(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeReferentPredictor()
    prompt = SegmentationPrompt(frame_index=0, text="person", collective=True)
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["referent_selection"] is None
    assert result["referent"]["collective"] is True
    assert "target_point_px" not in result["referent"]
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    expected = _two_person_masks()[0] | _two_person_masks()[1]
    for frame in frames:
        assert np.array_equal(frame > 0, expected)


def test_segment_video_referent_carries_prompt_lock_after_locked_run(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeReferentPredictor()
    prompt = _points_prompt()
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["referent"]["points"] == ((1, 1),)
    assert result["referent"]["object_id"] == 20
    assert "target_point_px" not in result["referent"]
    assert result["referent_selection"] is None


# --- Set B tests (W3, R9+R8) ---
#
# R9: while the referent lock is active, identity is proven by object ID
# only. A frame that carries masks but no usable IDs fails closed; it
# never falls back to the kept array position.
# R8: SegmentationPrompt.validate() carries the three combination rules
# (messages copied verbatim from the upstream validate()).


class FakeReorderedReferentPredictor(FakeReferentPredictor):
    """Stream frames return the two objects in swapped order, with IDs."""

    def handle_stream_request(self, request):
        self.requests.append(request)
        swapped = _two_person_masks()[::-1]
        for index in range(2):
            yield {
                "frame_index": index,
                "outputs": {
                    "out_binary_masks": swapped,
                    "out_obj_ids": np.array([20, 10]),
                    "out_probs": np.array([0.626465, 0.902344], dtype=np.float32),
                },
            }


class FakeNoFrameIdsReferentPredictor(FakeReferentPredictor):
    """Seeding carries IDs; stream frames carry masks but no object IDs."""

    def handle_stream_request(self, request):
        self.requests.append(request)
        for index in range(2):
            yield {
                "frame_index": index,
                "outputs": {
                    "out_binary_masks": _two_person_masks(),
                    "out_probs": np.array([0.902344, 0.626465], dtype=np.float32),
                },
            }


class FakeNoSeedingIdsReferentPredictor(FakeReferentPredictor):
    """Seeding response carries no object IDs; stream frames carry IDs."""

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "add_prompt":
            count = int(self.prompt_masks.shape[0])
            return {
                "session_id": "session",
                "outputs": {
                    "out_binary_masks": self.prompt_masks,
                    "out_probs": np.array([0.902344, 0.626465][:count], dtype=np.float32),
                },
            }
        return {"session_id": "session"}


def test_segment_video_referent_lock_follows_object_id_on_reorder(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeReorderedReferentPredictor()
    prompt = _points_prompt()
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["referent_selection"] is None
    assert result["object_id"] == 20
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    assert len(frames) == 2
    for frame in frames:
        # The subject sits at position 0 in the reordered frames. The
        # kept mask follows the prompt's object_id 20, not the array
        # position it held at seeding (position 1 in these frames is
        # the bystander).
        assert np.array_equal(frame > 0, _two_person_masks()[1])


def test_segment_video_referent_lock_no_frame_ids_fails_closed(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeNoFrameIdsReferentPredictor()
    prompt = _points_prompt()
    with pytest.raises(MediaError, match="stream response has no out_obj_ids"):
        segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert not (tmp_path / "mask.mp4").exists()
    assert predictor.requests[-1]["type"] == "close_session"


def test_segment_video_referent_lock_no_seeding_ids_proceeds(tmp_path):
    # REAUDIT-R4 replacement: under prompt-is-lock the kept ID comes
    # from the prompt, not the seeding response, so a seeding response
    # without out_obj_ids no longer blocks the run. The stream frames
    # carry IDs and the run follows object_id 20. (The fail-closed R9
    # proof survives in the no-frame-IDs test above.)
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeNoSeedingIdsReferentPredictor()
    prompt = _points_prompt()
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["object_id"] == 20
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    assert len(frames) == 2
    for frame in frames:
        assert np.array_equal(frame > 0, _two_person_masks()[1])


def test_segment_video_without_lock_unions_frames_without_object_ids(tmp_path):
    # No lock is active (collective text prompt, the SET A union
    # contract): frames that carry no object IDs still union as before.
    # The R9 raise fires only on the locked path.
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakeNoFrameIdsReferentPredictor()
    prompt = SegmentationPrompt(frame_index=0, text="person", collective=True)
    result = segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert result["referent_selection"] is None
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    expected = _two_person_masks()[0] | _two_person_masks()[1]
    for frame in frames:
        assert np.array_equal(frame > 0, expected)


def test_prompt_points_require_object_id():
    prompt = SegmentationPrompt(
        frame_index=0,
        points=((1, 1),),
        point_labels=(1,),
    )
    with pytest.raises(ValueError, match="point prompts require object_id"):
        prompt.validate()


def test_prompt_points_cannot_combine_with_text_or_box():
    text_prompt = SegmentationPrompt(
        frame_index=0,
        text="person",
        points=((1, 1),),
        point_labels=(1,),
        object_id=7,
    )
    with pytest.raises(
        ValueError, match="point prompts cannot be combined with text or box prompts"
    ):
        text_prompt.validate()
    box_prompt = SegmentationPrompt(
        frame_index=0,
        points=((1, 1),),
        point_labels=(1,),
        boxes_xyxy=((0, 0, 2, 2),),
        object_id=7,
    )
    with pytest.raises(
        ValueError, match="point prompts cannot be combined with text or box prompts"
    ):
        box_prompt.validate()


def test_prompt_object_id_only_with_points():
    text_prompt = SegmentationPrompt(frame_index=0, text="person", object_id=7)
    with pytest.raises(ValueError, match="object_id is valid only with point prompts"):
        text_prompt.validate()
    box_prompt = SegmentationPrompt(
        frame_index=0,
        boxes_xyxy=((0, 0, 2, 2),),
        object_id=7,
    )
    with pytest.raises(ValueError, match="object_id is valid only with point prompts"):
        box_prompt.validate()


def test_prompt_box_lock_requires_exactly_one_box():
    prompt = SegmentationPrompt(
        frame_index=0,
        boxes_xyxy=((0, 0, 2, 4), (3, 0, 6, 4)),
    )
    with pytest.raises(ValueError, match="box prompts require exactly one box"):
        prompt.validate()


def test_prompt_coordinates_must_be_finite():
    point = SegmentationPrompt(
        frame_index=0,
        points=((float("nan"), 1),),
        point_labels=(1,),
        object_id=7,
    )
    with pytest.raises(ValueError, match="point coordinates must be finite"):
        point.validate()
    box = SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, float("inf"), 4),))
    with pytest.raises(ValueError, match="box coordinates must be finite"):
        box.validate()


def test_prompt_coordinates_must_fit_the_destination_frame(tmp_path):
    video = _two_frame_video(tmp_path)
    predictor = FakePredictor()
    prompt = SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, 7, 4),))
    with pytest.raises(ValueError, match="box must fit inside the 6x4 frame"):
        segment_video(predictor, video, prompt, tmp_path / "mask.mp4")
    assert predictor.requests == []


def test_prompt_legal_combinations_pass_validation():
    # Points with object_id (and nothing else) is the legal point form.
    SegmentationPrompt(
        frame_index=0,
        points=((1, 1),),
        point_labels=(1,),
        object_id=7,
    ).validate()
    # Box-only and text-only forms stay legal.
    SegmentationPrompt(frame_index=0, boxes_xyxy=((0, 0, 2, 2),)).validate()
    SegmentationPrompt(frame_index=0, text="person", collective=True).validate()
    # Text+box is legal: the box is the spatial lock (REAUDIT-R4 D3).
    SegmentationPrompt(frame_index=0, text="person", boxes_xyxy=((0, 0, 2, 2),)).validate()


class StreamResponsePredictor:
    def __init__(self, frame_response):
        self.frame_response = frame_response
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        return {"session_id": "session"}

    def handle_stream_request(self, request):
        self.requests.append(request)
        for index in range(2):
            yield self.frame_response(index)


def _two_frame_video(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    return video


def _frame_with_two_persons(index, obj_ids=None, probs=None):
    return {
        "frame_index": index,
        "outputs": {
            "out_binary_masks": _two_person_masks(),
            "out_obj_ids": np.array([10, 20] if obj_ids is None else obj_ids),
            "out_probs": np.array(
                [0.902344, 0.626465] if probs is None else probs, dtype=np.float32
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_response", "message"),
    [
        ({"outputs": {"out_binary_masks": _two_person_masks()}}, "no frame_index"),
        ({"frame_index": 0}, "no outputs"),
        ({"frame_index": 0, "outputs": None}, "no outputs"),
    ],
)
def test_stream_response_contract_failures_are_named(tmp_path, bad_response, message):
    predictor = StreamResponsePredictor(lambda _index: bad_response)
    with pytest.raises(MediaError, match=message):
        segment_video(
            predictor, _two_frame_video(tmp_path), _points_prompt(), tmp_path / "mask.mp4"
        )
    assert not (tmp_path / "mask.mp4").exists()


@pytest.mark.parametrize(
    ("obj_ids", "probs", "message"),
    [
        ([10], [0.9, 0.6], "out_obj_ids length"),
        ([10, 20], [0.9], "out_probs length"),
    ],
)
def test_stream_array_length_mismatches_fail_closed(tmp_path, obj_ids, probs, message):
    predictor = StreamResponsePredictor(
        lambda index: _frame_with_two_persons(index, obj_ids=obj_ids, probs=probs)
    )
    with pytest.raises(MediaError, match=message):
        segment_video(
            predictor,
            _two_frame_video(tmp_path),
            _points_prompt(),
            tmp_path / "mask.mp4",
            tmp_path / "scores.npy",
        )


def test_locked_referent_may_be_absent_after_it_is_observed(tmp_path):
    def frame_response(index):
        if index == 0:
            return _frame_with_two_persons(index)
        return {
            "frame_index": index,
            "outputs": {
                "out_binary_masks": np.zeros((0, 4, 6), dtype=bool),
                "out_obj_ids": np.array([], dtype=np.int64),
                "out_probs": np.array([], dtype=np.float32),
            },
        }

    predictor = StreamResponsePredictor(frame_response)
    result = segment_video(
        predictor, _two_frame_video(tmp_path), _points_prompt(), tmp_path / "mask.mp4"
    )
    frames = decode_gray_frames(tmp_path / "mask.mp4", tmp_path / "decoded")
    assert np.array_equal(frames[0] > 0, _two_person_masks()[1])
    assert not frames[1].any()
    assert result["object_counts"] == [0, 2]


def test_referent_never_located_in_stream_fails_instead_of_writing_all_black(tmp_path):
    predictor = StreamResponsePredictor(
        lambda index: _frame_with_two_persons(index, obj_ids=[30, 40])
    )
    with pytest.raises(MediaError, match="never returned the selected referent id"):
        segment_video(
            predictor, _two_frame_video(tmp_path), _points_prompt(), tmp_path / "mask.mp4"
        )
    assert not (tmp_path / "mask.mp4").exists()
