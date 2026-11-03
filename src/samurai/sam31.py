"""SAM 3.1 request adapter and mask-video generation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .media import MediaError, VideoInfo, probe_video, write_gray_video

BOX_AMBIGUITY_MARGIN = 0.05


@dataclass(frozen=True)
class SegmentationPrompt:
    """An absolute-coordinate SAM 3.1 prompt on one frame."""

    frame_index: int
    text: str | None = None
    points: tuple[tuple[float, float], ...] = ()
    point_labels: tuple[int, ...] = ()
    boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    object_id: int | None = None
    collective: bool = False

    def validate(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if not self.text and not self.points and not self.boxes_xyxy:
            raise ValueError("provide a text prompt, point prompt, or box prompt")
        if len(self.points) != len(self.point_labels):
            raise ValueError("every point needs one label")
        if self.points and self.object_id is None:
            raise ValueError("point prompts require object_id")
        if self.object_id is not None and not self.points:
            raise ValueError("object_id is valid only with point prompts")
        if not self.points and not self.boxes_xyxy and not self.collective:
            raise ValueError(
                "no prompt lock: provide points with object_id, a box, or declare collective"
            )
        if self.boxes_xyxy and len(self.boxes_xyxy) != 1:
            raise ValueError("box prompts require exactly one box")
        if self.collective and (self.points or self.boxes_xyxy):
            raise ValueError("collective cannot be combined with point or box prompts")
        if self.points and (self.text or self.boxes_xyxy):
            raise ValueError("point prompts cannot be combined with text or box prompts")
        if self.points and any(label not in (0, 1) for label in self.point_labels):
            raise ValueError("point labels must be 0 or 1")
        if any(x < 0 or y < 0 for x, y in self.points):
            raise ValueError("point coordinates must be nonnegative")
        if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in self.points):
            raise ValueError("point coordinates must be finite")
        if any(x2 <= x1 or y2 <= y1 for x1, y1, x2, y2 in self.boxes_xyxy):
            raise ValueError("boxes must use x1,y1,x2,y2 with positive size")
        if any(x1 < 0 or y1 < 0 for x1, y1, _x2, _y2 in self.boxes_xyxy):
            raise ValueError("box coordinates must be nonnegative")
        if any(
            not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            for x1, y1, x2, y2 in self.boxes_xyxy
        ):
            raise ValueError("box coordinates must be finite")
        if self.object_id is not None and self.object_id < 0:
            raise ValueError("object_id must be nonnegative")

    def validate_for_video(self, video: VideoInfo) -> None:
        """Validate absolute coordinates against the one video being segmented."""
        self.validate()
        if any(x > video.width or y > video.height for x, y in self.points):
            raise ValueError(
                f"point coordinates must fit inside the {video.width}x{video.height} frame"
            )
        if any(x2 > video.width or y2 > video.height for _x1, _y1, x2, y2 in self.boxes_xyxy):
            raise ValueError(f"box must fit inside the {video.width}x{video.height} frame")

    def request(self, video: VideoInfo) -> dict[str, Any]:
        self.validate()
        request: dict[str, Any] = {"type": "add_prompt", "frame_index": self.frame_index}
        if self.object_id is not None:
            request["obj_id"] = self.object_id
        if self.text:
            request["text"] = self.text
        if self.points:
            request["points"] = [[x / video.width, y / video.height] for x, y in self.points]
            request["point_labels"] = list(self.point_labels)
        if self.boxes_xyxy:
            request["bounding_boxes"] = [
                [
                    x1 / video.width,
                    y1 / video.height,
                    (x2 - x1) / video.width,
                    (y2 - y1) / video.height,
                ]
                for x1, y1, x2, y2 in self.boxes_xyxy
            ]
            request["bounding_box_labels"] = [True] * len(self.boxes_xyxy)
        return request


def _outputs_of(response: Any) -> dict[str, Any] | None:
    """Outputs mapping of one predictor response, or None when it is absent."""
    if not isinstance(response, dict):
        return None
    outputs = response.get("outputs")
    return outputs if isinstance(outputs, dict) else None


def _detection_stack(outputs: dict[str, Any]) -> np.ndarray:
    """Boolean detection stack of one response, failing closed when malformed.

    A well-formed stack with zero detections is a valid frame state:
    every object is hidden in that frame. It is not a defect, so it is
    kept and the frame resolves to an empty target frame. A stack that
    is not 3D, or that cannot be read as masks, still fails closed.
    """
    masks = outputs.get("out_binary_masks")
    if masks is None:
        raise MediaError("SAM 3.1 response has no out_binary_masks")
    try:
        stack = np.asarray(masks, dtype=bool)
    except (TypeError, ValueError) as error:
        raise MediaError(f"SAM 3.1 returned an unreadable mask stack: {error}") from error
    if stack.ndim != 3:
        raise MediaError(f"SAM 3.1 returned invalid mask shape {stack.shape}")
    return stack


def _object_ids(outputs: dict[str, Any], n_detections: int) -> list[int] | None:
    """Object ids of one response as ints, or None when the response has none.

    A present id list is validated against the number of masks before it
    is used as a position: a shorter or longer list would index the
    wrong mask or raise an out-of-range index.
    """
    raw = outputs.get("out_obj_ids")
    if raw is None:
        return None
    try:
        ids = [int(value) for value in np.asarray(raw).reshape(-1).tolist()]
    except (TypeError, ValueError) as error:
        raise MediaError(f"SAM 3.1 out_obj_ids are not a list of object ids: {error}") from error
    if len(ids) != n_detections:
        raise MediaError(
            f"SAM 3.1 out_obj_ids length {len(ids)} does not match {n_detections} masks"
        )
    return ids


def _probabilities(outputs: dict[str, Any], n_detections: int) -> np.ndarray | None:
    """Per-object probabilities of one response, or None when it has none."""
    raw = outputs.get("out_probs")
    if raw is None:
        return None
    try:
        values = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as error:
        raise MediaError(f"SAM 3.1 out_probs are not a list of numbers: {error}") from error
    if values.shape[0] != n_detections:
        raise MediaError(
            f"SAM 3.1 out_probs length {values.shape[0]} does not match {n_detections} masks"
        )
    return values


def _frame_index_of(response: dict[str, Any]) -> int:
    """Frame index of one stream response, failing closed when it is absent."""
    raw = response.get("frame_index")
    if raw is None:
        raise MediaError("SAM 3.1 stream response has no frame_index")
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise MediaError(f"SAM 3.1 stream response has an invalid frame_index: {raw!r}") from error


def _locked_object_id_from_boxes(
    prompt_response: Any, prompt: SegmentationPrompt, info: VideoInfo
) -> dict[str, Any]:
    """Lock and record the seeded detection separated by the prompt box."""
    outputs = _outputs_of(prompt_response)
    if outputs is None:
        raise MediaError("box prompt lock failed closed: prompt response has no outputs")
    masks = _detection_stack(outputs)
    ids = _object_ids(outputs, int(masks.shape[0]))
    if ids is None:
        raise MediaError(
            "box prompt lock failed closed: prompt response has no out_obj_ids, "
            "so the selected object cannot be tracked by identity"
        )
    box_mask = np.zeros((info.height, info.width), dtype=bool)
    for x1, y1, x2, y2 in prompt.boxes_xyxy:
        left = min(max(int(round(x1)), 0), info.width)
        right = min(max(int(round(x2)), 0), info.width)
        top = min(max(int(round(y1)), 0), info.height)
        bottom = min(max(int(round(y2)), 0), info.height)
        box_mask[top:bottom, left:right] = True
    overlaps = [int(np.count_nonzero(masks[index] & box_mask)) for index in range(len(ids))]
    areas = [int(np.count_nonzero(masks[index])) for index in range(len(ids))]
    containment = [
        overlaps[index] / areas[index] if areas[index] else 0.0 for index in range(len(ids))
    ]
    order = sorted(range(len(ids)), key=lambda index: (-containment[index], index))
    if not order:
        raise MediaError("box prompt lock failed closed: prompt response seeded no detection")
    best = order[0]
    if overlaps[best] == 0:
        raise MediaError("box prompt lock failed closed: box overlaps no seeded detection")
    if containment[best] < 0.5:
        raise MediaError(
            "box prompt lock failed closed: best seeded detection is less than half "
            f"contained by the box ({containment[best]:.3f})"
        )
    if len(order) > 1:
        second = order[1]
        if containment[best] - containment[second] <= BOX_AMBIGUITY_MARGIN * containment[best]:
            raise MediaError(
                "box prompt lock failed closed: box cannot separate seeded detections "
                f"{ids[best]} and {ids[second]} "
                f"(containment {containment[best]:.3f} vs {containment[second]:.3f})"
            )
    return {
        "kind": "box",
        "object_id": ids[best],
        "candidate_object_ids": ids,
        "selected_index": best,
        "box_overlap_pixels": overlaps,
        "detection_areas": areas,
        "box_containment": containment,
    }


def _kept_position(outputs: dict[str, Any], kept_obj_id: int | None) -> int | None:
    """Position of the kept object in one frame's outputs, or None if absent.

    The prompt-frame array position is never reused as a per-frame
    position: detection order is not stable across frames, so the same
    index can hold a different object. A stream response without object
    ids cannot establish identity at all and is defective, not an empty
    referent frame. A present id list that does not contain the kept id
    means the referent is absent on that frame.
    """
    n_detections = int(_detection_stack(outputs).shape[0])
    ids = _object_ids(outputs, n_detections)
    if ids is None:
        raise MediaError(
            "SAM 3.1 stream response has no out_obj_ids; the selected referent "
            "cannot be tracked by identity"
        )
    if kept_obj_id is None:
        return None
    return ids.index(kept_obj_id) if kept_obj_id in ids else None


def _mask_from_outputs(
    outputs: dict[str, Any], keep_position: int | None = None
) -> tuple[np.ndarray, int]:
    masks = _detection_stack(outputs)
    total = int(masks.shape[0])
    if total == 0:
        # Every object is hidden on this frame: the target frame is empty
        # and the frame carries no detection.
        height, width = int(masks.shape[1]), int(masks.shape[2])
        return np.zeros((height, width), dtype=np.uint8), 0
    if keep_position is not None:
        masks = masks[[keep_position]]
    return np.any(masks, axis=0).astype(np.uint8) * 255, total


def _score_from_outputs(outputs: dict[str, Any], keep_position: int | None = None) -> np.ndarray:
    masks = _detection_stack(outputs)
    object_probabilities = _probabilities(outputs, int(masks.shape[0]))
    if object_probabilities is None:
        object_probabilities = np.ones(masks.shape[0], dtype=np.float32)
    if keep_position is not None:
        masks = masks[[keep_position]]
        object_probabilities = object_probabilities[[keep_position]]
    if masks.shape[0] == 0:
        # No detection on this frame: the probability support is empty.
        return np.zeros(masks.shape[1:], dtype=np.float32)
    return np.max(masks.astype(np.float32) * object_probabilities[:, None, None], axis=0)


def segment_video(
    predictor: Any,
    video: Path,
    prompt: SegmentationPrompt,
    output: Path,
    scores_output: Path | None = None,
) -> dict[str, Any]:
    """Run one SAM 3.1 prompt through forward propagation and write a black mask video.

    The prompt is the lock. A point prompt with a stable object ID binds the
    stream to that ID on every frame. A box is the spatial lock. Collective is
    the explicit declaration that a union over all tracked detections is
    intended. Box prompts use one deterministic local containment check over the
    seeding response; detection scores never select a referent.
    """
    prompt.validate()
    if not video.exists():
        raise FileNotFoundError(video)
    info = probe_video(video)
    prompt.validate_for_video(info)
    session = predictor.handle_request({"type": "start_session", "resource_path": str(video)})
    session_id = session["session_id"]
    frame_masks: dict[int, np.ndarray] = {}
    frame_scores: dict[int, np.ndarray] | None = {} if scores_output else None
    object_counts: set[int] = set()
    referent_selection: dict[str, Any] | None = None
    locked_obj_id = prompt.object_id if prompt.points else None
    located_referent = False
    try:
        request = prompt.request(info)
        request["session_id"] = session_id
        prompt_response = predictor.handle_request(request)
        if prompt.boxes_xyxy:
            referent_selection = _locked_object_id_from_boxes(prompt_response, prompt, info)
            locked_obj_id = int(referent_selection["object_id"])
        stream = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
        }
        for response in predictor.handle_stream_request(stream):
            outputs = _outputs_of(response)
            if outputs is None:
                raise MediaError("SAM 3.1 stream response has no outputs")
            frame_index = _frame_index_of(response)
            if locked_obj_id is not None:
                keep_position = _kept_position(outputs, locked_obj_id)
                if keep_position is not None:
                    located_referent = True
                if keep_position is None:
                    # The kept object is absent on this frame: the frame is
                    # empty for the selected referent. No other object is
                    # admitted and no re-selection happens. A frame where
                    # every object is hidden carries no detection at all and
                    # resolves to the same empty target frame and zero
                    # score instead of aborting the run.
                    mask = np.zeros((info.height, info.width), dtype=np.uint8)
                    _, count = _mask_from_outputs(outputs)
                    score = (
                        np.zeros((info.height, info.width), dtype=np.float32)
                        if frame_scores is not None
                        else None
                    )
                else:
                    mask, count = _mask_from_outputs(outputs, keep_position=keep_position)
                    score = (
                        _score_from_outputs(outputs, keep_position=keep_position)
                        if frame_scores is not None
                        else None
                    )
            else:
                mask, count = _mask_from_outputs(outputs)
                score = _score_from_outputs(outputs) if frame_scores is not None else None
            if frame_scores is not None:
                assert score is not None
                frame_scores[frame_index] = score
            if frame_index in frame_masks:
                raise MediaError(f"duplicate SAM 3.1 output frame {frame_index}")
            frame_masks[frame_index] = mask
            object_counts.add(count)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    if locked_obj_id is not None and not located_referent:
        raise MediaError(
            "SAM 3.1 never returned the selected referent id in a stream response; "
            "an all-black mask would not be target tracking"
        )
    if set(frame_masks) != set(range(info.frames)):
        missing = sorted(set(range(info.frames)) - set(frame_masks))
        raise MediaError(f"SAM 3.1 did not return every frame; first missing={missing[:8]}")
    ordered = [frame_masks[index] for index in range(info.frames)]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_gray_video(ordered, output, fps=info.fps)
    if scores_output is not None and frame_scores is not None:
        scores = np.stack([frame_scores[index] for index in range(info.frames)])
        scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(scores_output, scores.astype(np.float32))
    return {
        "video": str(video),
        "output": str(output),
        "referent": asdict(prompt),
        "referent_selection": referent_selection,
        "frames": info.frames,
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "object_counts": sorted(object_counts),
        "object_id": locked_obj_id,
        "model_version": "sam3.1",
        "scores_output": str(scores_output) if scores_output else None,
        "scores_semantics": (
            (
                (
                    "per-pixel object probability support for the selected tracked "
                    "object only (the prompt-carried object ID is the lock); binary "
                    "mask multiplied by SAM object probability"
                    if prompt.points
                    else "per-pixel object probability support for the selected tracked "
                    "object only (the box-derived seeded object ID is the lock); binary "
                    "mask multiplied by SAM object probability"
                )
                if locked_obj_id is not None
                else (
                    "per-pixel object probability support; binary mask "
                    "multiplied by SAM object probability"
                )
            )
            if scores_output
            else None
        ),
    }
