#!/usr/bin/env python3
"""Ground candidate insertions only on owned lower-body frames.

The command consumes a generated lower-body ownership report plus a strict
candidate manifest.  It selects only frames with one owned pose, requires a
mask and plate frame for the complete source frame count, reindexes that sparse
owned subset for the deterministic contact-grounding kernel, and writes one
ROI-black shadow-mask clip and one per-frame evidence record per candidate.
It makes no model or network call and claims no continuous ownership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import contact_grounding as contact_script
import numpy as np

from samurai.contact_grounding import (
    ContactGroundingError,
    mask_digest,
    parse_keypoint_frames,
    summarize,
)
from samurai.media import GrayVideoWriter, MediaError
from samurai.owned_frame_contact import (
    OWNERSHIP_REPORT_SCHEMA,
    OwnedFrameContactError,
    OwnershipSelection,
    contact_keypoint_payload,
    parse_ownership_report,
)

REPORT_SCHEMA = "void-fusion-lab.owned-frame-contact.report.v1"
CANDIDATE_MANIFEST_SCHEMA = "void-fusion-lab.owned-frame-contact.candidates.v1"
SAFE_CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
EXIT_INPUT_ERROR = 2

NON_CLAIMS = (
    "Sparse lower-body ownership is not continuous tracking, interpolation, or pose truth.",
    "Only frames with exactly one owned lower-body pose instance were measured; "
    "unowned frames are omitted rather than inferred.",
    "Contact and shadow values are image-space measurements, not a quality verdict "
    "or acceptance decision.",
    "The output shadow mask is ROI-black and is not a VOID quadmask.",
    "The plate-luminance direction is a first-order estimate, not a full lighting solution.",
    "No renderer, model, SAM, VOID, credential, or network call ran.",
)


class OwnedFrameContactScriptError(RuntimeError):
    """A candidate manifest or media input is unusable."""


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate's mask and effective plate inputs."""

    candidate_id: str
    object_mask: Path
    plate: Path
    object_mask_source: str
    plate_source: str


def reject_json_constant(name: str) -> None:
    """Reject JSON's non-finite constants before they can enter geometry."""
    raise OwnedFrameContactScriptError(f"JSON input holds the non-finite constant {name}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OwnedFrameContactScriptError(f"invalid JSON document {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path, source: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "source": source,
        **contact_script.input_digests(path),
    }


def manifest_path(manifest: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OwnedFrameContactScriptError(f"{label} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return path


def parse_candidate_manifest(
    payload: Any, manifest: Path
) -> tuple[Path | None, tuple[CandidateSpec, ...]]:
    """Validate the strict manifest and resolve paths relative to that manifest."""
    document = payload
    if not isinstance(document, dict):
        raise OwnedFrameContactScriptError("candidate manifest must be a JSON object")
    if document.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise OwnedFrameContactScriptError(
            f"candidate manifest schema must be {CANDIDATE_MANIFEST_SCHEMA!r}"
        )
    raw_shared_plate = document.get("plate")
    shared_plate = (
        manifest_path(manifest, raw_shared_plate, "shared plate")
        if raw_shared_plate is not None
        else None
    )
    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise OwnedFrameContactScriptError("candidates must be a nonempty JSON array")
    candidates: list[CandidateSpec] = []
    seen: set[str] = set()
    for position, raw_candidate in enumerate(raw_candidates):
        label = f"candidates[{position}]"
        if not isinstance(raw_candidate, dict):
            raise OwnedFrameContactScriptError(f"{label} must be a JSON object")
        candidate_id = raw_candidate.get("id")
        if (
            not isinstance(candidate_id, str)
            or SAFE_CANDIDATE_ID.fullmatch(candidate_id) is None
            or candidate_id in seen
        ):
            raise OwnedFrameContactScriptError(
                f"{label} id must be unique and match {SAFE_CANDIDATE_ID.pattern!r}"
            )
        seen.add(candidate_id)
        mask_source = raw_candidate.get("object_mask")
        object_mask = manifest_path(manifest, mask_source, f"{label} object_mask")
        raw_plate = raw_candidate.get("plate")
        plate_source = "plate" if raw_plate is None else str(raw_plate)
        plate = (
            shared_plate
            if raw_plate is None
            else manifest_path(manifest, raw_plate, f"{label} plate")
        )
        if plate is None:
            raise OwnedFrameContactScriptError(
                f"{label} has no plate and the manifest declares no shared plate"
            )
        candidates.append(
            CandidateSpec(
                candidate_id=candidate_id,
                object_mask=object_mask,
                plate=plate,
                object_mask_source=str(mask_source),
                plate_source=str(plate_source),
            )
        )
    return shared_plate, tuple(candidates)


def source_state() -> dict[str, Any]:
    """Record the current Git head and dirty state without publishing a path."""
    repository = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}
    return {"revision": head, "dirty": bool(status.strip())}


def build_settings(args: argparse.Namespace) -> Any:
    return contact_script.build_settings(args)


def selected_frames(frames: list[np.ndarray | None], indices: list[int]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for index in indices:
        frame = frames[index]
        if frame is None:
            raise OwnedFrameContactScriptError(f"plate frame {index} is missing")
        result.append(frame)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ownership-report", type=Path, required=True)
    result.add_argument("--candidates", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--fps", type=float)
    result.add_argument("--mask-invert", action="store_true")
    result.add_argument("--contact-thickness-ratio", type=float, default=0.06)
    result.add_argument("--contact-spread-ratio", type=float, default=0.05)
    result.add_argument("--shadow-length-ratio", type=float, default=0.6)
    result.add_argument("--shadow-spread-ratio", type=float, default=1.0)
    result.add_argument("--vertical-tolerance-ratio", type=float, default=0.03)
    result.add_argument("--max-foot-lift-ratio", type=float, default=0.12)
    result.add_argument("--min-plate-samples", type=int, default=64)
    result.add_argument("--min-light-confidence", type=float, default=0.5)
    result.add_argument("--contact-line-min-baseline-ratio", type=float, default=0.05)
    result.add_argument("--clip-shadow-to-ground", action="store_true")
    return result


def _tool_failure_detail(error: subprocess.CalledProcessError) -> str:
    lines = (error.stderr or "").strip().splitlines()
    suffix = f"; {lines[-1]}" if lines else ""
    return f"exit {error.returncode}{suffix}"


def _ground_candidates(
    *,
    args: argparse.Namespace,
    selection: OwnershipSelection,
    shared_plate: Path | None,
    candidates: tuple[CandidateSpec, ...],
    manifest: Path,
    output: Path,
) -> tuple[dict[str, Any], float]:
    source_indices = [frame.frame_index for frame in selection.owned_frames]
    shared_frames = (
        contact_script.load_plate_frames(shared_plate, source_indices) if shared_plate else None
    )
    shared_info = input_record(shared_plate, "plate") if shared_plate is not None else None
    if shared_frames is not None and len(shared_frames) != selection.total_frames:
        raise OwnedFrameContactScriptError(
            f"the shared plate has {len(shared_frames)} frames but the ownership report has "
            f"{selection.total_frames}"
        )
    shape: tuple[int, int] | None = None
    fps: float | None = args.fps
    candidate_reports: list[dict[str, Any]] = []
    settings = build_settings(args)
    for spec in candidates:
        masks, declared_fps = contact_script.load_object_masks(
            spec.object_mask, invert=args.mask_invert
        )
        if len(masks) != selection.total_frames:
            raise OwnedFrameContactScriptError(
                f"candidate {spec.candidate_id} mask has {len(masks)} frames but the ownership "
                f"report has {selection.total_frames}; missing masks fail closed"
            )
        mask_height, mask_width = masks[0].shape[:2]
        current_shape = (mask_height, mask_width)
        if shape is None:
            shape = current_shape
            height, width = shape
            keypoints = parse_keypoint_frames(
                contact_keypoint_payload(selection, width=width, height=height)
            )
        elif current_shape != shape:
            raise OwnedFrameContactScriptError(
                f"candidate {spec.candidate_id} mask is {current_shape} but an earlier candidate "
                f"is {shape}"
            )
        if fps is None:
            fps = declared_fps
        elif math.isclose(declared_fps, fps, rel_tol=0.0, abs_tol=0.0) is False and (
            math.isfinite(declared_fps) and declared_fps > 0.0
        ):
            raise OwnedFrameContactScriptError(
                f"candidate {spec.candidate_id} declares fps {declared_fps}, expected {fps}"
            )
        if fps is None or not math.isfinite(fps) or fps <= 0.0:
            raise OwnedFrameContactScriptError(
                "the owned-frame clip frame rate is unknown; pass --fps"
            )
        assert fps is not None
        if shared_frames is not None and shared_info is not None:
            plate_frames = shared_frames
            plate_info = shared_info
        else:
            plate_frames = contact_script.load_plate_frames(spec.plate, source_indices)
            plate_info = input_record(spec.plate, spec.plate_source)
        if len(plate_frames) != selection.total_frames:
            raise OwnedFrameContactScriptError(
                f"candidate {spec.candidate_id} plate has {len(plate_frames)} frames but the "
                f"ownership report has {selection.total_frames}"
            )
        selected_masks = [masks[index] for index in source_indices]
        selected_plates = selected_frames(plate_frames, source_indices)
        if any(frame.shape[:2] != shape for frame in selected_plates):
            raise OwnedFrameContactScriptError(
                f"candidate {spec.candidate_id} plate geometry does not match its object mask"
            )
        measurements, shadows = contact_script.ground_contact(
            keypoints, selected_masks, plate_frames=selected_plates, settings=settings
        )
        candidate_directory = output / "candidates" / spec.candidate_id
        candidate_directory.mkdir(parents=True, exist_ok=True)
        shadow_path = candidate_directory / "shadow-mask.mp4"
        frame_records: list[dict[str, Any]] = []
        with GrayVideoWriter(shadow_path, width=shape[1], height=shape[0], fps=fps) as writer:
            for local_index, (measurement, shadow) in enumerate(
                zip(measurements, shadows, strict=True)
            ):
                writer.write(np.where(shadow, 0, 255).astype(np.uint8))
                frame_records.append(
                    {
                        **asdict(measurement),
                        "source_frame": source_indices[local_index],
                        "owned_pose_id": selection.owned_frames[local_index].pose_id,
                        "shadow_digest": mask_digest(shadow),
                    }
                )
        candidate_reports.append(
            {
                "candidate_id": spec.candidate_id,
                "inputs": {
                    "object_mask": {
                        **input_record(spec.object_mask, spec.object_mask_source),
                        "frames": len(masks),
                        "width": shape[1],
                        "height": shape[0],
                        "roi_convention": "bright" if args.mask_invert else "dark",
                    },
                    "plate": {**plate_info, "frames": len(plate_frames)},
                },
                "settings": asdict(settings),
                "frames": frame_records,
                "summary": summarize(measurements, mask_frames=len(selected_masks)),
                "shadow_mask": {
                    "path": f"candidates/{spec.candidate_id}/shadow-mask.mp4",
                    "sha256": sha256(shadow_path),
                    "mask_family": "roi-black",
                    "not_void_quadmask": True,
                },
            }
        )
    if fps is None:
        raise OwnedFrameContactScriptError("the owned-frame clip frame rate is unknown; pass --fps")
    resolved_fps = float(fps)
    return (
        {
            "schema": REPORT_SCHEMA,
            "recorded_at": datetime.now(UTC).isoformat(),
            **source_state(),
            "inputs": {
                "ownership_report": {
                    "path": str(args.ownership_report),
                    "sha256": sha256(args.ownership_report),
                    "schema": OWNERSHIP_REPORT_SCHEMA,
                },
                "candidate_manifest": {
                    "path": str(manifest),
                    "sha256": sha256(manifest),
                    "schema": CANDIDATE_MANIFEST_SCHEMA,
                },
                "shared_plate": shared_info,
            },
            "ownership": {
                "total_frames": selection.total_frames,
                "owned_frames": source_indices,
                "owned_frame_ratio": selection.owned_frame_ratio,
                "continuous_ownership_claimed": False,
            },
            "fps": resolved_fps,
            "candidates": candidate_reports,
            "non_claims": list(NON_CLAIMS),
        },
        resolved_fps,
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.ownership_report.is_file():
        raise OwnedFrameContactScriptError(
            f"ownership report does not exist: {args.ownership_report}"
        )
    if not args.candidates.is_file():
        raise OwnedFrameContactScriptError(f"candidate manifest does not exist: {args.candidates}")
    selection = parse_ownership_report(read_json(args.ownership_report))
    if not selection.owned_frames:
        raise OwnedFrameContactScriptError("the ownership report contains no owned frames")
    shared_plate, candidates = parse_candidate_manifest(read_json(args.candidates), args.candidates)
    if args.out.exists():
        raise OwnedFrameContactScriptError(f"output directory already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.out.name}-", dir=args.out.parent) as temporary:
        report, fps = _ground_candidates(
            args=args,
            selection=selection,
            shared_plate=shared_plate,
            candidates=candidates,
            manifest=args.candidates,
            output=Path(temporary),
        )
        report_path = Path(temporary) / "owned-frame-contact-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(args.out)
    first = report["candidates"][0]
    print(
        json.dumps(
            {
                "owned_frames": report["ownership"]["owned_frames"],
                "owned_frame_ratio": report["ownership"]["owned_frame_ratio"],
                "candidates": len(report["candidates"]),
                "measured_frames": first["summary"]["measured_frames"],
                "fps": fps,
                "report": str(args.out / "owned-frame-contact-report.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ContactGroundingError,
        contact_script.ContactGroundingScriptError,
        OwnedFrameContactError,
        OwnedFrameContactScriptError,
        MediaError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(
                f"Owned-frame contact failed: {_tool_failure_detail(error)}",
                file=sys.stderr,
            )
        else:
            print(f"Owned-frame contact failed: {error}", file=sys.stderr)
        raise SystemExit(EXIT_INPUT_ERROR) from error
