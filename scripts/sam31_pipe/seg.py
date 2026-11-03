# ruff: noqa: UP031, E501
# Formatting note: the original percent-style formatting is kept verbatim;
# converting it is a follow-up cleanup.
# This runner needs the SAM 3.1 execution lane (torch + sam3) and does not
# run in the base environment.
#
# Referent ruling (2026-10-28): the mask for a frame is the union of the components
# associated with the selected target object ONLY, never the union of
# every detected object and never the single largest object. The target
# referent (PROMPT text + BOX anchor + prompt frame) is carried into
# this script and the locked target is recorded in seg_report.json.
# Association uses the predictor object ids (trajectory identity) when
# present; without ids it uses overlap against the nearest resolved
# target mask. Uniqueness is never association evidence: a frame whose
# only remaining component is some other object resolves to an empty
# target frame, and a frame with no association evidence at all fails
# closed. Defective frames (missing mask, empty stack, wrong shape,
# malformed output) and ambiguous frames (the component-to-target
# association cannot be decided) fail closed: no PNG is written for the
# frame, the failure is recorded in seg_report.json, and the run exits
# with code 2. A genuine target-absent frame is an explicit recorded
# state ("target_absent" / "empty" in seg_report.json), never a silent
# all-zero success mask.
"""SAM 3.x text(+box)-prompted video segmentation -> one binary mask png per frame.
usage: seg.py <frames_dir> <masks_dir> [--model sam3.1|sam3] [--chunk N]
env: PROMPT (default person), PROMPT_FRAME (default 30, clamped; per chunk = chunk middle when --chunk), BOX x,y,w,h (optional, normalized)
--chunk N: segment long clips in N-frame chunks, each a fresh session (bounds GPU memory).
Always writes seg_report.json into the masks dir: referent, locked target,
per-frame component association (retained/dropped ids and areas), defects.
Exit code 2 when any frame is defective or ambiguous (no PNG for it)."""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

# A component is associated with the target by overlap when at least this
# fraction of the component's own pixels overlap the reference target mask.
OVERLAP_MIN = 0.5
# An overlap score this close to OVERLAP_MIN is too close to call.
AMBIGUITY_MARGIN = 0.05


class SegError(Exception):
    """Base class for fail-closed segmentation failures."""


class SegDefectError(SegError):
    """The predictor output for a frame is defective (missing/malformed)."""


class SegAmbiguityError(SegError):
    """The component-to-target association for a frame is ambiguous."""


class EmptyPromptError(RuntimeError):
    """A prompt frame produced no object; another prompt frame may be tried."""


def cuda_device_label(torch_module):
    """Return (available, device_name) without probing an absent CUDA device.

    torch.cuda.get_device_name(0) raises on a host without a CUDA
    device, so the name is only read when is_available() is true.
    """
    available = bool(torch_module.cuda.is_available())
    if not available:
        return False, "none"
    return True, torch_module.cuda.get_device_name(0)


def outputs_mapping(response):
    """Outputs mapping of one predictor response, or None when it is absent."""
    if not isinstance(response, dict):
        return None
    outputs = response.get("outputs")
    return outputs if isinstance(outputs, dict) else None


def object_id_list(raw):
    """Object ids of one response as ints, or None when absent or unreadable."""
    if raw is None:
        return None
    try:
        return [int(value) for value in np.asarray(raw).reshape(-1).tolist()]
    except (TypeError, ValueError):
        return None


def normalize_stack(raw, expected_shape, frame_index=None):
    """Return raw masks as a bool stack (n, H, W) or raise SegDefectError."""
    where = "frame %s: " % frame_index if frame_index is not None else ""
    if raw is None:
        raise SegDefectError(where + "missing mask (predictor returned no mask)")
    arr = np.asarray(raw)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise SegDefectError(where + "malformed mask ndim=%d" % arr.ndim)
    if arr.shape[0] == 0:
        raise SegDefectError(where + "empty mask stack (0 objects)")
    if tuple(arr.shape[1:]) != tuple(expected_shape):
        raise SegDefectError(
            where
            + "shape mismatch got=%s expected=%s" % (tuple(arr.shape[1:]), tuple(expected_shape))
        )
    return arr.astype(bool)


def parse_box(box):
    """Parse a BOX value 'x,y,w,h' into exactly 4 finite floats.

    Returns None when BOX is unset/empty. Any other malformed value
    fails closed with a SystemExit naming the BOX form, instead of an
    unhandled ValueError/IndexError downstream in box_mask_from().
    """
    if not box:
        return None
    try:
        values = [float(v) for v in box.split(",")]
    except ValueError:
        raise SystemExit("BOX must be 'x,y,w,h' (4 finite numbers), got %r" % box) from None
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise SystemExit("BOX must be 'x,y,w,h' (4 finite numbers), got %r" % box)
    return values


def box_mask_from(box, width, height):
    """Boolean (H, W) mask of a normalized [x, y, w, h] box."""
    x0 = min(max(int(round(box[0] * width)), 0), width)
    y0 = min(max(int(round(box[1] * height)), 0), height)
    x1 = min(max(int(round((box[0] + box[2]) * width)), 0), width)
    y1 = min(max(int(round((box[1] + box[3]) * height)), 0), height)
    mask = np.zeros((height, width), bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def lock_target(obj_ids, masks, box_mask, expected_shape, frame_index=None):
    """Lock the selected target object on the prompt frame.

    Returns a record with target_obj_ids (list or None), target_index
    (int or None) and the association mode. Raises SegAmbiguityError when
    several candidate objects exist and the referent anchor (BOX) cannot
    select exactly one of them, and SegDefectError on malformed input.
    With neither ids nor a prompt-frame stack the lock stays open
    (association "overlap_chain") and frames resolve by overlap against
    the nearest resolved target mask instead.
    """
    where = "prompt frame %s: " % frame_index if frame_index is not None else ""
    ids = [int(x) for x in obj_ids] if obj_ids is not None else None
    if masks is None:
        if ids is None:
            return {
                "target_obj_ids": None,
                "target_index": None,
                "association": "overlap_chain",
                "n_candidates": 0,
                "areas": [],
                "box_overlaps": None,
            }
        if len(ids) == 1:
            return {
                "target_obj_ids": ids,
                "target_index": None,
                "association": "obj_id",
                "n_candidates": 1,
                "areas": [],
                "box_overlaps": None,
            }
        raise SegAmbiguityError(
            where + "%d candidate objects and no prompt-frame masks to apply the anchor" % len(ids)
        )
    stack = normalize_stack(masks, expected_shape, frame_index)
    n = int(stack.shape[0])
    if ids is not None and len(ids) != n:
        raise SegDefectError(
            where + "out_obj_ids length %d does not match %d masks" % (len(ids), n)
        )
    areas = [int(x) for x in stack.reshape(n, -1).sum(1)]
    if n == 1:
        return {
            "target_obj_ids": ids,
            "target_index": 0 if ids is None else None,
            "association": "obj_id" if ids is not None else "prompt_index",
            "n_candidates": 1,
            "areas": areas,
            "box_overlaps": None,
        }
    if box_mask is None:
        raise SegAmbiguityError(
            where
            + "%d candidate objects for one referent and no BOX anchor to select the target" % n
        )
    overlaps = [int((stack[i] & box_mask).sum()) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-overlaps[i], i))
    best, second = order[0], order[1]
    if overlaps[best] == 0:
        raise SegAmbiguityError(where + "BOX anchor overlaps no candidate object")
    if overlaps[best] - overlaps[second] <= AMBIGUITY_MARGIN * overlaps[best]:
        raise SegAmbiguityError(
            where
            + "BOX anchor cannot separate candidates %d and %d (overlaps %d vs %d)"
            % (best, second, overlaps[best], overlaps[second])
        )
    return {
        "target_obj_ids": [ids[best]] if ids is not None else None,
        "target_index": best if ids is None else None,
        "association": "obj_id+box" if ids is not None else "prompt_index+box",
        "n_candidates": n,
        "areas": areas,
        "box_overlaps": overlaps,
    }


def resolve_frame(
    raw,
    obj_ids,
    expected_shape,
    frame_index=None,
    target_obj_ids=None,
    target_index=None,
    target_prev_mask=None,
):
    """Resolve one frame to the union of its target-associated components.

    Returns (mask, record): mask is the (H, W) bool union of the retained
    components only; record carries component count, retained/dropped
    indices, object ids and areas. Raises SegDefectError for defective
    output and SegAmbiguityError when association cannot be decided.
    A frame whose detections are all unassociated resolves to an all-zero
    mask with status "target_absent": an explicit recorded state. A sole
    remaining component is associated only through positive overlap with
    the reference target mask, never because it is the only component.
    """
    stack = normalize_stack(raw, expected_shape, frame_index)
    n = int(stack.shape[0])
    ids = [int(x) for x in obj_ids] if obj_ids is not None else None
    if ids is not None and len(ids) != n:
        raise SegDefectError(
            "frame %s: out_obj_ids length %d does not match %d masks" % (frame_index, len(ids), n)
        )
    areas = [int(x) for x in stack.reshape(n, -1).sum(1)]
    if target_obj_ids is not None:
        if ids is None:
            raise SegAmbiguityError(
                "frame %s: target is locked by object id but the frame has no out_obj_ids"
                % frame_index
            )
        wanted = {int(x) for x in target_obj_ids}
        retained = [i for i in range(n) if ids[i] in wanted]
        association = "obj_id"
    elif target_index is not None:
        if not 0 <= int(target_index) < n:
            raise SegDefectError(
                "frame %s: locked target index %s out of range for %d components"
                % (frame_index, target_index, n)
            )
        retained = [int(target_index)]
        association = "prompt_index"
    else:
        # No identity lock on this frame, so the target can only be
        # associated through positive overlap with the nearest resolved
        # target mask. Being the sole remaining component is not
        # evidence: a frame whose only component is some other object
        # resolves to an empty target frame, not to that object.
        if target_prev_mask is None:
            raise SegAmbiguityError(
                "frame %s: %d components and no target association is available" % (frame_index, n)
            )
        prev = np.asarray(target_prev_mask, dtype=bool)
        if prev.shape != tuple(expected_shape):
            raise SegDefectError(
                "frame %s: reference target mask shape %s, expected %s"
                % (frame_index, tuple(prev.shape), tuple(expected_shape))
            )
        scores = []
        for i in range(n):
            if areas[i] == 0:
                scores.append(0.0)
            else:
                scores.append(float((stack[i] & prev).sum()) / float(areas[i]))
        for i, score in enumerate(scores):
            if 0 < abs(score - OVERLAP_MIN) <= AMBIGUITY_MARGIN:
                raise SegAmbiguityError(
                    "frame %s: component %d overlap %.3f is too close to the %.2f "
                    "association threshold to call" % (frame_index, i, score, OVERLAP_MIN)
                )
        retained = [i for i in range(n) if scores[i] >= OVERLAP_MIN]
        association = "overlap"
    dropped = [i for i in range(n) if i not in set(retained)]
    if retained:
        mask = np.any(stack[retained], axis=0)
    else:
        mask = np.zeros(tuple(expected_shape), bool)
    retained_area = int(mask.sum())
    if not retained:
        status = "target_absent"
    elif retained_area == 0:
        status = "empty"
    else:
        status = "ok"
    record = {
        "frame": frame_index,
        "status": status,
        "n_components": n,
        "areas": areas,
        "retained_components": retained,
        "retained_obj_ids": [ids[i] for i in retained] if ids is not None else None,
        "retained_area": retained_area,
        "dropped_components": dropped,
        "dropped_obj_ids": [ids[i] for i in dropped] if ids is not None else None,
        "dropped_area": int(sum(areas[i] for i in dropped)),
        "association": association,
    }
    return mask, record


def resolve_chunk(got, n_frames, prompt_frame, lock, expected_shape, chunk_start=0):
    """Resolve every frame of one chunk against the locked target.

    got maps local frame index -> (raw masks or None, object ids or None).
    Returns (resolved, errors): resolved maps local index -> (mask, record),
    errors maps local index -> SegError. Frames are resolved outward from
    the prompt frame so the overlap chain always has the nearest resolved
    target mask on the current side of propagation as its reference.
    """
    resolved = {}
    errors = {}
    target_ids = lock.get("target_obj_ids")
    target_index = lock.get("target_index")
    reference_masks: dict[int, np.ndarray] = {}
    order = sorted(range(n_frames), key=lambda j: (abs(j - prompt_frame), j))
    for j in order:
        global_index = chunk_start + j
        entry = got.get(j)
        if entry is None:
            errors[j] = SegDefectError("frame %d: missing mask (no stream entry)" % global_index)
            continue
        raw, ids = entry
        try:
            if target_ids is not None:
                mask, record = resolve_frame(
                    raw, ids, expected_shape, global_index, target_obj_ids=target_ids
                )
            elif j == prompt_frame and target_index is not None:
                mask, record = resolve_frame(
                    raw, ids, expected_shape, global_index, target_index=target_index
                )
            else:
                direction = j - prompt_frame
                if direction > 0:
                    directional_references = [
                        resolved_index
                        for resolved_index in reference_masks
                        if resolved_index >= prompt_frame
                    ]
                elif direction < 0:
                    directional_references = [
                        resolved_index
                        for resolved_index in reference_masks
                        if resolved_index <= prompt_frame
                    ]
                else:
                    directional_references = list(reference_masks)
                nearest = min(
                    directional_references or reference_masks,
                    key=lambda resolved_index: (abs(resolved_index - j), resolved_index),
                    default=None,
                )
                mask, record = resolve_frame(
                    raw,
                    ids,
                    expected_shape,
                    global_index,
                    target_prev_mask=reference_masks.get(nearest),
                )
        except SegError as error:
            errors[j] = error
            continue
        record["frame"] = global_index
        resolved[j] = (mask, record)
        if record["retained_area"] > 0:
            reference_masks[j] = mask
    return resolved, errors


def validate_chunk(value, frame_count):
    """Return a positive chunk size, rejecting zero and negative values."""
    chunk = frame_count if value is None or value == 0 else int(value)
    if chunk <= 0:
        raise SystemExit("--chunk must be a positive integer, got %r" % value)
    return chunk


def prepare_mask_dir(path):
    """Create an empty mask directory, refusing stale mask artifacts."""
    os.makedirs(path, exist_ok=True)
    with os.scandir(path) as entries:
        stale = next(iter(entries), None)
    if stale is not None:
        raise SystemExit(
            "masks directory is not empty; remove or choose a new directory so a "
            "defective frame cannot inherit a stale PNG: %s" % path
        )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("frames")
    ap.add_argument("masks")
    ap.add_argument(
        "--model",
        default=None,
        help="SAM version (default: SAM_MODEL env or sam3.1)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=int(os.environ.get("SEG_CHUNK", "0")),
        help="frames per session, 0 = whole clip (default 0)",
    )
    return ap.parse_args(argv)


def main(argv=None):
    import torch  # ty: ignore[unresolved-import]

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sam3_compat import DEFAULT_MODEL, build_predictor

    a = parse_args(argv)
    if a.model is None:
        a.model = os.environ.get("SAM_MODEL", DEFAULT_MODEL)
    fd, md = a.frames, a.masks
    prepare_mask_dir(md)
    cuda_available, cuda_device = cuda_device_label(torch)
    print("cuda", cuda_available, cuda_device, flush=True)
    p = build_predictor(version=a.model)
    names = sorted(n for n in os.listdir(fd) if n.lower().endswith((".png", ".jpg", ".jpeg")))
    if not names:
        raise SystemExit("no image frames (.png/.jpg/.jpeg) found in %s" % fd)
    N = len(names)
    W, H = Image.open(os.path.join(fd, names[0])).size
    box = os.environ.get("BOX")
    box_values = parse_box(box)
    referent = {
        "prompt": os.environ.get("PROMPT", "person"),
        "box": box,
        "model": a.model,
    }
    last_box = None
    C = validate_chunk(a.chunk, N)
    cnt = 0
    first_empty = None
    defects = []
    frame_records = []
    target_records = []
    for c0 in range(0, N, C):
        sub = names[c0 : c0 + C]
        n = len(sub)
        if C == N:
            src = fd
        else:
            src = tempfile.mkdtemp(prefix=".sam31-chunk-", dir=md)
            for j, nm in enumerate(sub):
                os.symlink(os.path.abspath(os.path.join(fd, nm)), os.path.join(src, "%05d.jpg" % j))
        sid = p.handle_request({"type": "start_session", "resource_path": src})["session_id"]
        cands = (
            [min(int(os.environ.get("PROMPT_FRAME", "30")), n - 1)]
            if C == N
            else [n // 2, n // 4, 3 * n // 4] + list(range(0, n, 25))
        )
        got = {}
        malformed = 0
        used = None
        prompt_resp = None
        prompt_pf = None
        prompt_bx = None
        tries = [(pf, box_values) for pf in dict.fromkeys(cands)]
        if last_box:
            tries.append((0, last_box))  # re-seed from the previous chunk's last good mask box
        for pf, bx in tries:  # chunk may open on a shot with nobody in it: try other prompt frames
            req = {
                "type": "add_prompt",
                "session_id": sid,
                "frame_index": pf,
                "text": referent["prompt"],
                "output_prob_thresh": 0.3,
            }
            if bx:
                req.update(bounding_boxes=[bx], bounding_box_labels=[1])
            try:
                r = p.handle_request(req)
                prompt_outputs = outputs_mapping(r)
                prompt_ids = object_id_list(
                    prompt_outputs.get("out_obj_ids") if prompt_outputs else None
                )
                if not prompt_ids:
                    # A missing, None, or malformed add_prompt response is an
                    # empty response: this prompt frame holds no object.
                    raise EmptyPromptError("no object on prompt frame")
                for out in p.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": sid,
                        "propagation_direction": "both",
                    }
                ):
                    if not isinstance(out, dict) or out.get("frame_index") is None:
                        # A malformed stream entry carries no frame index, so
                        # it cannot be attributed to a frame. It is dropped and
                        # that frame keeps no entry, which resolve_chunk
                        # records as a defective frame.
                        malformed += 1
                        continue
                    try:
                        fi = int(out["frame_index"])
                    except (TypeError, ValueError):
                        malformed += 1
                        continue
                    o = outputs_mapping(out)
                    if o is None and "outputs" not in out:
                        o = out  # flat SAM response
                    if o is None:
                        got[fi] = (None, None)
                        continue
                    m = o.get("out_binary_masks")
                    ids = object_id_list(o.get("out_obj_ids"))
                    if m is None:
                        got[fi] = (None, ids)
                        continue
                    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
                    got[fi] = (m, ids)
                used = "%d%s" % (
                    pf,
                    " (re-seeded from last good box)" if bx == last_box and bx is not None else "",
                )
                prompt_resp = r
                prompt_pf = pf
                prompt_bx = bx
                break
            except EmptyPromptError:
                got = {}
                malformed = 0
                p.handle_request({"type": "reset_session", "session_id": sid})
        print(
            "chunk %d: prompt frame %s"
            % (c0, used if used is not None else "none found (masks empty)"),
            flush=True,
        )
        if malformed:
            print(
                "chunk %d: %d malformed stream entry(ies) ignored (no frame index)"
                % (c0, malformed),
                flush=True,
            )
        try:
            p.handle_request({"type": "close_session", "session_id": sid})
        except Exception as e:
            print("close_session:", e, flush=True)
        if used is None:
            # No prompt frame produced an object: every frame of the chunk
            # is defective, not silently empty.
            for j in range(n):
                detail = "no object on any prompt frame of the chunk"
                defects.append({"frame": c0 + j, "kind": "defective", "detail": detail})
                frame_records.append({"frame": c0 + j, "status": "defective", "detail": detail})
                print("seg defect frame %d: %s (no mask written)" % (c0 + j, detail), flush=True)
            if C != N:
                shutil.rmtree(src)
            continue
        assert prompt_pf is not None  # set together with `used` above
        prompt_outputs = outputs_mapping(prompt_resp) or {}
        prompt_ids = object_id_list(prompt_outputs.get("out_obj_ids"))
        prompt_masks = prompt_outputs.get("out_binary_masks")
        if prompt_masks is not None and hasattr(prompt_masks, "detach"):
            prompt_masks = prompt_masks.detach().cpu().numpy()
        elif prompt_masks is not None:
            prompt_masks = np.asarray(prompt_masks)
        box_mask = box_mask_from(prompt_bx, W, H) if prompt_bx else None
        try:
            lock = lock_target(prompt_ids, prompt_masks, box_mask, (H, W), c0 + prompt_pf)
        except SegError as error:
            kind = "ambiguous" if isinstance(error, SegAmbiguityError) else "defective"
            for j in range(n):
                defects.append({"frame": c0 + j, "kind": kind, "detail": str(error)})
                frame_records.append({"frame": c0 + j, "status": kind, "detail": str(error)})
            print("seg %s: %s (chunk %d writes no masks)" % (kind, error, c0), flush=True)
            if C != N:
                shutil.rmtree(src)
            continue
        target_records.append(
            {
                "chunk_start": c0,
                "prompt_frame": c0 + prompt_pf,
                "target_obj_ids": lock["target_obj_ids"],
                "target_index": lock["target_index"],
                "association": lock["association"],
                "n_candidates": lock["n_candidates"],
                "areas": lock["areas"],
                "box_overlaps": lock["box_overlaps"],
            }
        )
        resolved, errors = resolve_chunk(got, n, prompt_pf, lock, (H, W), c0)
        if not any(record["retained_area"] > 0 for _mask, record in resolved.values()):
            detail = "locked target was never observed in the propagation stream"
            for j in range(n):
                defects.append({"frame": c0 + j, "kind": "defective", "detail": detail})
                frame_records.append({"frame": c0 + j, "status": "defective", "detail": detail})
            print("seg defect chunk %d: %s (no mask written)" % (c0, detail), flush=True)
            if C != N:
                shutil.rmtree(src)
            continue
        for j in range(n):
            if j in errors:
                error = errors[j]
                kind = "ambiguous" if isinstance(error, SegAmbiguityError) else "defective"
                defects.append({"frame": c0 + j, "kind": kind, "detail": str(error)})
                frame_records.append({"frame": c0 + j, "status": kind, "detail": str(error)})
                print(
                    "seg %s frame %d: %s (no mask written)" % (kind, c0 + j, error),
                    flush=True,
                )
                continue
            m, record = resolved[j]
            frame_records.append(record)
            if record["dropped_area"]:
                print(
                    "frame %d: dropped %d unassociated component(s), %d px"
                    % (c0 + j, len(record["dropped_components"]), record["dropped_area"]),
                    flush=True,
                )
            if m.any():
                cnt += 1
                ys, xs = np.nonzero(m)
                last_box = [
                    xs.min() / W,
                    ys.min() / H,
                    (xs.max() - xs.min()) / W,
                    (ys.max() - ys.min()) / H,
                ]
            elif first_empty is None:
                first_empty = c0 + j
            Image.fromarray((m > 0).astype(np.uint8) * 255).save(
                os.path.join(md, "%05d.png" % (c0 + j))
            )
        if C != N:
            shutil.rmtree(src)
            print("chunk %d-%d done" % (c0, c0 + n - 1), flush=True)
    frame_records.sort(key=lambda rec: rec["frame"])
    report = {
        "referent": referent,
        "targets": target_records,
        "frames": frame_records,
        "defects": defects,
    }
    with open(os.path.join(md, "seg_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    if defects:
        print(
            "SAM masks STOP: %d defective/ambiguous frame(s): %s"
            % (len(defects), [(d["frame"], d["kind"]) for d in defects]),
            flush=True,
        )
        sys.exit(2)
    print(
        "SAM masks nonempty %d/%d model=%s first_empty_frame=%s" % (cnt, N, a.model, first_empty),
        flush=True,
    )


if __name__ == "__main__":
    main()
