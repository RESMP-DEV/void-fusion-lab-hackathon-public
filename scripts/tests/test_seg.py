"""Tests for scripts/sam31_pipe/seg.py (item #4 ruling, hackathon #12).

The ruling: a frame mask is the union of the components associated with
the selected target object ONLY; the referent is carried in and the
association is recorded; defective or ambiguous frames fail closed and
never emit a silent all-zero mask.

Unit tests import seg.py by path (it is a pipe script, not a package).
CLI tests run the real seg.py bytes in a subprocess with only its two
external dependencies stubbed (torch CUDA probe, SAM predictor), the
same technique as the L2 verification lane.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SEG_PATH = Path(__file__).resolve().parents[1] / "sam31_pipe" / "seg.py"
H, W = 6, 8
SHAPE = (H, W)


def _load_seg():
    spec = importlib.util.spec_from_file_location("seg_under_test", SEG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seg = _load_seg()


def _rect(stack, idx, r0, r1, c0, c1):
    stack[idx, r0:r1, c0:c1] = True
    return stack


def _two_object_stack():
    """Object id 1: two split blobs (4+4 px). Object id 2: one blob (20 px)."""
    stack = np.zeros((2, H, W), dtype=bool)
    _rect(stack, 0, 0, 1, 0, 4)  # target blob A, 4 px
    _rect(stack, 0, 5, 6, 4, 8)  # target blob B, 4 px (same slice, split)
    _rect(stack, 1, 1, 5, 0, 5)  # bystander, 20 px
    return stack


# --- (a) union contains only the target's components ---------------------


def test_union_contains_only_target_components():
    stack = _two_object_stack()
    mask, record = seg.resolve_frame(stack, [1, 2], SHAPE, 0, target_obj_ids=[1])
    assert np.array_equal(mask, stack[0])
    assert int(mask.sum()) == 8
    # The bystander's 20 px are nowhere in the union.
    assert not mask[1:5, 0:5].any()
    assert record["retained_obj_ids"] == [1]
    assert record["dropped_obj_ids"] == [2]
    assert record["retained_area"] == 8
    assert record["dropped_area"] == 20
    assert record["association"] == "obj_id"
    assert record["status"] == "ok"


def test_union_of_split_components_via_overlap_excludes_bystander():
    prev = np.zeros(SHAPE, dtype=bool)
    prev[0:2, 0:4] = True
    prev[4:6, 4:8] = True
    stack = np.zeros((3, H, W), dtype=bool)
    _rect(stack, 0, 0, 2, 0, 4)  # target part 1: fully overlaps prev
    _rect(stack, 1, 4, 6, 4, 8)  # target part 2: fully overlaps prev
    _rect(stack, 2, 2, 4, 5, 8)  # bystander: no overlap with prev
    mask, record = seg.resolve_frame(stack, None, SHAPE, 3, target_prev_mask=prev)
    assert np.array_equal(mask, stack[0] | stack[1])
    assert not mask[2:4, 5:8].any()
    assert record["retained_components"] == [0, 1]
    assert record["dropped_components"] == [2]
    assert record["association"] == "overlap"


def test_lock_with_box_selects_the_overlapping_object():
    stack = _two_object_stack()
    box_mask = seg.box_mask_from([0.0, 0.0, 0.5, 0.2], W, H)  # covers blob A only
    lock = seg.lock_target([1, 2], stack, box_mask, SHAPE, 0)
    assert lock["target_obj_ids"] == [1]


# --- (b) defective frames fail closed ------------------------------------


def test_defective_missing_mask_fails_closed():
    with pytest.raises(seg.SegDefectError):
        seg.resolve_frame(None, [1], SHAPE, 1, target_obj_ids=[1])


def test_defective_wrong_shape_fails_closed():
    bad = np.ones((1, 4, 5), dtype=bool)  # 20 px at the wrong shape
    with pytest.raises(seg.SegDefectError):
        seg.resolve_frame(bad, [1], SHAPE, 2, target_obj_ids=[1])


def test_defective_empty_stack_fails_closed():
    empty = np.zeros((0, H, W), dtype=bool)
    with pytest.raises(seg.SegDefectError):
        seg.resolve_frame(empty, [], SHAPE, 0, target_obj_ids=[1])


def test_defective_obj_id_length_mismatch_fails_closed():
    stack = _two_object_stack()
    with pytest.raises(seg.SegDefectError):
        seg.resolve_frame(stack, [1], SHAPE, 0, target_obj_ids=[1])


# --- (c) ambiguous frames fail closed ------------------------------------


def test_ambiguous_prompt_lock_without_anchor_fails_closed():
    stack = _two_object_stack()
    with pytest.raises(seg.SegAmbiguityError):
        seg.lock_target([1, 2], stack, None, SHAPE, 0)


def test_ambiguous_box_tie_fails_closed():
    stack = np.zeros((2, H, W), dtype=bool)
    _rect(stack, 0, 0, 2, 0, 4)
    _rect(stack, 1, 0, 2, 4, 8)
    box_mask = np.ones(SHAPE, dtype=bool)  # overlaps both candidates equally
    with pytest.raises(seg.SegAmbiguityError):
        seg.lock_target([1, 2], stack, box_mask, SHAPE, 0)


def test_ambiguous_multi_component_frame_without_association_fails_closed():
    stack = _two_object_stack()
    with pytest.raises(seg.SegAmbiguityError):
        seg.resolve_frame(stack, None, SHAPE, 0)


def test_frame_without_obj_ids_cannot_match_id_locked_target():
    stack = _two_object_stack()
    with pytest.raises(seg.SegAmbiguityError):
        seg.resolve_frame(stack, None, SHAPE, 0, target_obj_ids=[1])


# --- explicit recorded states, not silent zeros ---------------------------


def test_true_empty_is_an_explicit_recorded_state():
    empty = np.zeros((1, H, W), dtype=bool)
    mask, record = seg.resolve_frame(empty, [1], SHAPE, 0, target_obj_ids=[1])
    assert not mask.any()
    assert record["status"] == "empty"


def test_target_absent_is_an_explicit_recorded_state():
    stack = _two_object_stack()
    mask, record = seg.resolve_frame(stack, [1, 2], SHAPE, 0, target_obj_ids=[9])
    assert not mask.any()
    assert record["status"] == "target_absent"
    assert record["dropped_obj_ids"] == [1, 2]
    assert record["dropped_area"] == 28


# --- CLI end-to-end with stubbed predictor --------------------------------

_TORCH_STUB = """
class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def get_device_name(_index):
        # Real torch raises on a host without a CUDA device, so the
        # guarded probe must never reach this call.
        raise RuntimeError("CUDA device name probed without CUDA")


cuda = _Cuda()
"""

_COMPAT_STUB = """
import json
import os

import numpy as np

DEFAULT_MODEL = "sam3.1"


class _Predictor:
    def __init__(self, scenario):
        self._scenario = scenario

    def handle_request(self, req):
        kind = req["type"]
        if kind == "start_session":
            return {"session_id": "stub-session"}
        if kind == "add_prompt":
            prompt = self._scenario.get("prompt_response", "default")
            if prompt != "default":
                return prompt
            outputs = {"out_obj_ids": self._scenario["prompt_obj_ids"]}
            if "prompt_masks" in self._scenario:
                outputs["out_binary_masks"] = np.asarray(
                    self._scenario["prompt_masks"], dtype=bool
                )
            return {"outputs": outputs}
        return {}

    def handle_stream_request(self, _req):
        if self._scenario.get("stream_error"):
            raise RuntimeError(self._scenario["stream_error"])
        for key in sorted(self._scenario["frames"], key=int):
            entry = self._scenario["frames"][key]
            if entry is None:
                yield {"frame_index": int(key), "outputs": {"out_binary_masks": None}}
                continue
            kind = entry.get("kind", "normal")
            if kind == "outputs-none":
                yield {"frame_index": int(key), "outputs": None}
                continue
            if kind == "no-outputs":
                yield {"frame_index": int(key)}
                continue
            if kind == "null-response":
                yield None
                continue
            masks = np.asarray(entry["masks"], dtype=bool)
            yield {
                "frame_index": int(key),
                "outputs": {
                    "out_binary_masks": masks,
                    "out_obj_ids": entry["obj_ids"],
                },
            }


def build_predictor(version=None):
    with open(os.environ["SEG_TEST_SCENARIO"]) as f:
        return _Predictor(json.load(f))
"""


def _mask_list(rects):
    stack = np.zeros((len(rects), H, W), dtype=bool)
    for i, (r0, r1, c0, c1) in enumerate(rects):
        stack[i, r0:r1, c0:c1] = True
    return stack.tolist()


def _run_cli(tmp_path, scenario, n_frames=3, extra_files=(), env_extra=None, extra_args=()):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(SEG_PATH, run_dir / "seg.py")
    (run_dir / "torch.py").write_text(_TORCH_STUB)
    (run_dir / "sam3_compat.py").write_text(_COMPAT_STUB)
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(n_frames):
        Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(frames / f"{i:05d}.png")
    for name in extra_files:
        (frames / name).write_text("not an image")
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario))
    masks = tmp_path / "masks"
    env = dict(os.environ)
    env.update(
        {
            "PROMPT_FRAME": "0",
            "PROMPT": "person",
            "SEG_TEST_SCENARIO": str(scenario_path),
        }
    )
    env.pop("BOX", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(run_dir / "seg.py"), str(frames), str(masks), *extra_args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return proc, masks


def _png_pixels(path):
    return int((np.asarray(Image.open(path)) > 0).sum())


def test_cli_union_excludes_bystander_and_records_association(tmp_path):
    target = (0, 2, 0, 4)  # 8 px
    bystander = (2, 6, 0, 5)  # 20 px
    prompt_frame = {"obj_ids": [1], "masks": _mask_list([target])}
    later = {"obj_ids": [1, 2], "masks": _mask_list([target, bystander])}
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": prompt_frame, "1": later, "2": later},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for i in range(3):
        assert _png_pixels(masks / f"{i:05d}.png") == 8
    report = json.loads((masks / "seg_report.json").read_text())
    assert report["referent"]["prompt"] == "person"
    assert report["targets"][0]["target_obj_ids"] == [1]
    assert report["defects"] == []
    later_rec = report["frames"][1]
    assert later_rec["retained_obj_ids"] == [1]
    assert later_rec["dropped_obj_ids"] == [2]
    assert later_rec["dropped_area"] == 20


def test_cli_defective_frame_fails_closed_no_silent_zero_mask(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {"prompt_obj_ids": [1], "frames": {"0": good, "1": None, "2": good}}
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    # No PNG at all for the defective frame: a consumer cannot mistake a
    # fabricated all-zero mask for a real one.
    assert not (masks / "00001.png").exists()
    assert _png_pixels(masks / "00000.png") == 8
    assert _png_pixels(masks / "00002.png") == 8
    report = json.loads((masks / "seg_report.json").read_text())
    assert [d["frame"] for d in report["defects"]] == [1]
    assert report["defects"][0]["kind"] == "defective"
    statuses = {rec["frame"]: rec["status"] for rec in report["frames"]}
    assert statuses[1] == "defective"


def test_cli_ambiguous_prompt_fails_closed(tmp_path):
    frame = {
        "obj_ids": [1, 2],
        "masks": _mask_list([(0, 2, 0, 4), (2, 6, 0, 5)]),
    }
    scenario = {"prompt_obj_ids": [1, 2], "frames": {"0": frame, "1": frame, "2": frame}}
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not list(masks.glob("*.png"))
    report = json.loads((masks / "seg_report.json").read_text())
    assert report["defects"]
    assert {d["kind"] for d in report["defects"]} == {"ambiguous"}


# --- T23: BOX parsing -------------------------------------------------------


def test_parse_box_valid_returns_four_floats():
    assert seg.parse_box("0.1,0.2,0.3,0.4") == [0.1, 0.2, 0.3, 0.4]


def test_parse_box_unset_or_empty_is_none():
    assert seg.parse_box(None) is None
    assert seg.parse_box("") is None


@pytest.mark.parametrize("bad", ["1,2", "1,2,3,4,5", "a,b,c,d", "nan,0,1,1", "inf,0,1,1"])
def test_parse_box_malformed_fails_closed_naming_box_form(bad):
    with pytest.raises(SystemExit, match="BOX must be 'x,y,w,h'"):
        seg.parse_box(bad)


def test_cli_malformed_box_fails_closed(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {"prompt_obj_ids": [1], "frames": {"0": good, "1": good, "2": good}}
    proc, masks = _run_cli(tmp_path, scenario, env_extra={"BOX": "1,2"})
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "BOX must be" in proc.stderr
    assert not list(masks.glob("*.png"))


# --- T22: frame listing filters to image files ------------------------------


def test_cli_non_image_files_in_frames_dir_are_ignored(tmp_path):
    target = (0, 2, 0, 4)
    frame = {"obj_ids": [1], "masks": _mask_list([target])}
    scenario = {"prompt_obj_ids": [1], "frames": {"0": frame, "1": frame, "2": frame}}
    proc, masks = _run_cli(tmp_path, scenario, extra_files=("notes.txt", "README", ".DS_Store"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for i in range(3):
        assert _png_pixels(masks / f"{i:05d}.png") == 8


def test_cli_no_image_frames_fails_closed(tmp_path):
    scenario = {"prompt_obj_ids": [1], "frames": {}}
    proc, masks = _run_cli(tmp_path, scenario, n_frames=0, extra_files=("notes.txt",))
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "no image frames" in proc.stderr
    assert not list(masks.glob("*.png"))


# --- SAM referent-tracking review repairs (2026-10-30): predictor response contracts ---


class _FakeCuda:
    def __init__(self, available):
        self._available = available
        self.probed = False

    def is_available(self):
        return self._available

    def get_device_name(self, _index):
        self.probed = True
        return "fake-gpu"


class _FakeTorch:
    def __init__(self, available):
        self.cuda = _FakeCuda(available)


def test_cuda_label_skips_device_probe_without_cuda():
    torch_stub = _FakeTorch(False)
    assert seg.cuda_device_label(torch_stub) == (False, "none")
    assert torch_stub.cuda.probed is False


def test_cuda_label_reports_device_name_when_cuda_available():
    torch_stub = _FakeTorch(True)
    assert seg.cuda_device_label(torch_stub) == (True, "fake-gpu")
    assert torch_stub.cuda.probed is True


def test_cli_runs_without_cuda_device_probe(tmp_path):
    # The stubbed torch raises when the device name is probed without CUDA.
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {"prompt_obj_ids": [1], "frames": {"0": good, "1": good, "2": good}}
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cuda False none" in proc.stdout
    assert _png_pixels(masks / "00000.png") == 8


def test_cli_missing_prompt_response_fails_closed(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {
        "prompt_response": None,
        "prompt_obj_ids": [1],
        "frames": {"0": good, "1": good, "2": good},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not list(masks.glob("*.png"))
    report = json.loads((masks / "seg_report.json").read_text())
    assert [d["frame"] for d in report["defects"]] == [0, 1, 2]
    assert {d["kind"] for d in report["defects"]} == {"defective"}


def test_cli_prompt_response_without_outputs_fails_closed(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {
        "prompt_response": {"outputs": None},
        "prompt_obj_ids": [1],
        "frames": {"0": good, "1": good, "2": good},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not list(masks.glob("*.png"))
    report = json.loads((masks / "seg_report.json").read_text())
    assert {d["kind"] for d in report["defects"]} == {"defective"}


def test_cli_stream_response_without_outputs_fails_closed(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": good, "1": {"kind": "no-outputs"}, "2": good},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (masks / "00001.png").exists()
    report = json.loads((masks / "seg_report.json").read_text())
    assert [d["frame"] for d in report["defects"]] == [1]
    assert report["defects"][0]["kind"] == "defective"


def test_cli_stream_response_with_none_outputs_fails_closed(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": good, "1": {"kind": "outputs-none"}, "2": good},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (masks / "00001.png").exists()
    report = json.loads((masks / "seg_report.json").read_text())
    assert [d["frame"] for d in report["defects"]] == [1]


def test_cli_null_stream_response_fails_closed(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": good, "1": {"kind": "null-response"}, "2": good},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (masks / "00001.png").exists()
    report = json.loads((masks / "seg_report.json").read_text())
    assert [d["frame"] for d in report["defects"]] == [1]


def test_cli_obj_ids_length_mismatch_fails_closed(tmp_path):
    prompt_frame = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    mismatched = {
        "obj_ids": [1],
        "masks": _mask_list([(0, 2, 0, 4), (2, 6, 0, 5)]),
    }
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": prompt_frame, "1": mismatched, "2": prompt_frame},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (masks / "00001.png").exists()
    report = json.loads((masks / "seg_report.json").read_text())
    assert [d["frame"] for d in report["defects"]] == [1]
    assert report["defects"][0]["kind"] == "defective"


# --- SAM referent-tracking round 2 review repairs (2026-10-30): association evidence ---
#
# Uniqueness is never association evidence. A sole remaining component
# joins the target only through positive overlap with the reference
# target mask or through an identity lock; otherwise the target-only
# output is empty or the frame fails closed.


def test_sole_component_without_overlap_evidence_is_not_associated():
    # The target left the frame and one other object remains: the
    # target-only output is empty, and the remaining object is dropped.
    stack = np.zeros((1, H, W), dtype=bool)
    _rect(stack, 0, 0, 2, 0, 4)  # bystander only
    prev = np.zeros(SHAPE, dtype=bool)
    prev[4:6, 4:8] = True  # the reference target mask, elsewhere in the frame
    mask, record = seg.resolve_frame(stack, None, SHAPE, 7, target_prev_mask=prev)
    assert not mask.any()
    assert record["status"] == "target_absent"
    assert record["retained_components"] == []
    assert record["dropped_components"] == [0]
    assert record["association"] == "overlap"


def test_sole_component_with_overlap_evidence_is_associated():
    stack = np.zeros((1, H, W), dtype=bool)
    _rect(stack, 0, 0, 2, 0, 4)
    prev = np.zeros(SHAPE, dtype=bool)
    prev[0:2, 0:4] = True  # the same pixels: full overlap
    mask, record = seg.resolve_frame(stack, None, SHAPE, 7, target_prev_mask=prev)
    assert np.array_equal(mask, stack[0])
    assert record["status"] == "ok"
    assert record["retained_components"] == [0]
    assert record["association"] == "overlap"


def test_resolve_chunk_uses_the_nearest_reference_on_the_propagation_side(monkeypatch):
    prompt = np.zeros(SHAPE, dtype=bool)
    prompt[0:2, 0:4] = True
    before = np.zeros(SHAPE, dtype=bool)
    before[2:4, 0:4] = True
    later = np.zeros(SHAPE, dtype=bool)
    references = []

    def fake_resolve_frame(_raw, _ids, _shape, frame_index, **kwargs):
        mask = before if frame_index == 0 else prompt
        references.append((frame_index, kwargs.get("target_prev_mask")))
        return mask, {"retained_area": 1, "association": "overlap", "status": "ok"}

    monkeypatch.setattr(seg, "resolve_frame", fake_resolve_frame)
    seg.resolve_chunk(
        {0: (before, None), 1: (prompt, None), 2: (later, None)},
        3,
        1,
        {"target_index": 0},
        SHAPE,
    )
    frame2_reference = references[-1][1]
    assert frame2_reference is prompt


def test_sole_component_without_any_association_evidence_fails_closed():
    stack = np.zeros((1, H, W), dtype=bool)
    _rect(stack, 0, 0, 2, 0, 4)
    with pytest.raises(seg.SegAmbiguityError):
        seg.resolve_frame(stack, None, SHAPE, 7)


def test_identity_lock_still_associates_a_sole_component():
    # An object id lock is positive association evidence and is
    # unaffected by the uniqueness rule.
    stack = np.zeros((1, H, W), dtype=bool)
    _rect(stack, 0, 0, 2, 0, 4)
    mask, record = seg.resolve_frame(stack, [1], SHAPE, 7, target_obj_ids=[1])
    assert np.array_equal(mask, stack[0])
    assert record["association"] == "obj_id"
    assert record["status"] == "ok"


def test_cli_frame_with_only_a_bystander_left_is_empty_for_the_target(tmp_path):
    target = (0, 2, 0, 4)  # 8 px
    bystander = (2, 6, 0, 5)  # 20 px
    prompt_frame = {"obj_ids": [1], "masks": _mask_list([target])}
    # The target left the frame: only the bystander object id remains.
    later = {"obj_ids": [2], "masks": _mask_list([bystander])}
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": prompt_frame, "1": later, "2": later},
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _png_pixels(masks / "00000.png") == 8
    assert _png_pixels(masks / "00001.png") == 0
    assert _png_pixels(masks / "00002.png") == 0
    report = json.loads((masks / "seg_report.json").read_text())
    assert report["frames"][1]["status"] == "target_absent"
    assert report["frames"][1]["retained_obj_ids"] == []
    assert report["frames"][1]["dropped_obj_ids"] == [2]
    assert report["defects"] == []


def test_validate_chunk_rejects_nonpositive_values():
    assert seg.validate_chunk(None, 10) == 10
    assert seg.validate_chunk(0, 10) == 10
    assert seg.validate_chunk(4, 10) == 4
    for value in (-1, -10):
        with pytest.raises(SystemExit, match="--chunk must be a positive integer"):
            seg.validate_chunk(value, 10)


def test_prepare_mask_dir_rejects_stale_artifacts(tmp_path):
    empty = tmp_path / "empty"
    seg.prepare_mask_dir(empty)
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "00000.png").write_bytes(b"old mask")
    with pytest.raises(SystemExit, match="masks directory is not empty"):
        seg.prepare_mask_dir(stale)


# --- SAM referent-tracking round-5 review repairs: chunk ownership and propagation errors ---


def test_cli_chunk_mode_preserves_an_unrelated_sibling_directory(tmp_path):
    target = (0, 2, 0, 4)
    frame = {"obj_ids": [1], "masks": _mask_list([target])}
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": frame, "1": frame, "2": frame},
    }
    sibling = tmp_path / "masks_chunk"
    sibling.write_text("unrelated user data", encoding="utf-8")
    proc, masks = _run_cli(tmp_path, scenario, extra_args=("--chunk", "1"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sibling.read_text(encoding="utf-8") == "unrelated user data"
    assert not list(masks.glob(".sam31-chunk-*"))


def test_cli_locks_prompt_ids_and_masks_from_the_same_response(tmp_path):
    prompt_masks = _two_object_stack()
    stream_masks = np.stack([prompt_masks[1], prompt_masks[0]])
    stream = {
        "obj_ids": [2, 1],
        "masks": stream_masks.tolist(),
    }
    scenario = {
        "prompt_obj_ids": [1, 2],
        "prompt_masks": prompt_masks.tolist(),
        "frames": {str(index): stream for index in range(3)},
    }
    proc, masks = _run_cli(tmp_path, scenario, env_extra={"BOX": "0,0,0.5,0.2"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((masks / "seg_report.json").read_text())
    assert report["targets"][0]["target_obj_ids"] == [1]
    assert all(record["retained_obj_ids"] == [1] for record in report["frames"])


def test_cli_never_observed_locked_target_fails_closed(tmp_path):
    target = _two_object_stack()[0]
    stream = {"obj_ids": [1], "masks": np.stack([target]).tolist()}
    scenario = {
        "prompt_obj_ids": [99],
        "prompt_masks": np.stack([target]).tolist(),
        "frames": {str(index): stream for index in range(3)},
    }
    proc, masks = _run_cli(tmp_path, scenario, env_extra={"BOX": "0,0,0.5,0.2"})
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not list(masks.glob("*.png"))
    report = json.loads((masks / "seg_report.json").read_text())
    assert report["defects"]
    assert {record["status"] for record in report["frames"]} == {"defective"}


def test_cli_real_propagation_runtime_error_is_not_retried_as_an_empty_prompt(tmp_path):
    good = {"obj_ids": [1], "masks": _mask_list([(0, 2, 0, 4)])}
    scenario = {
        "prompt_obj_ids": [1],
        "frames": {"0": good, "1": good, "2": good},
        "stream_error": "CUDA out of memory",
    }
    proc, masks = _run_cli(tmp_path, scenario)
    assert proc.returncode != 0
    assert "CUDA out of memory" in proc.stderr
    assert not list(masks.glob("*.png"))
