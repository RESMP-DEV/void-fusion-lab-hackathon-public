#!/usr/bin/env python3
"""Select a golden generated replacement from a bounded Best-of-N campaign.

Two manifest shapes run through this one engine. The legacy flat manifest
keeps its own six gates, derives its masks from the plate difference, and
writes ``best-of-n-report.json`` exactly as it always has. The chain manifest
``void-fusion-lab.best-of-n.chain-manifest.v1`` binds every candidate to its
own inserted and cleanup masks, adds the ``remask_consistency`` gate, stages
one Pass 2 sequence per pre-Pass2 survivor, and writes
``best-of-n-chain-report.json``.

With ``require_full_chain``, every pre-Pass2 survivor must also supply strict
native, retimed, result, and comparator records. The selector verifies their
digests and geometry, ranks with the Pass2-aware objective, and selects on the
``full_chain`` basis. No model call is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from samurai.best_of_n import (
    EXIT_NO_SURVIVOR,
    MASK_SOURCE_DECLARED,
    MASK_SOURCE_DERIVED,
    MODE_LEGACY_ADAPTER,
    ROI_BLACK,
    BestOfNError,
    CandidateSpec,
    ChainCandidateScore,
    ChainPlan,
    Pass2MediaArtifact,
    build_chain_report,
    chain_score,
    changed_frames,
    full_chain_pass2_record,
    plan_from_manifest,
    score_candidate,
    target_region,
    validate_manifest,
)
from samurai.media import MediaError, _ffmpeg_framerate, probe_video, write_gray_video
from samurai.quadmask import build_insertion_sequence

# The stable import contract of this script. The canonical contracts are
# implemented in samurai.best_of_n and re-exported here unchanged.
__all__ = [
    "BestOfNError",
    "main",
    "score_candidate",
    "select",
    "validate_manifest",
]

SELECTED_FRAMES = (0, 16, 26, 48, 52, 60, 80, 95)
LEGACY_REPORT_NAME = "best-of-n-report.json"
CHAIN_REPORT_NAME = "best-of-n-chain-report.json"
NORMALIZED_DIR_NAME = "normalized"
SEQUENCE_DIR_NAME = "void-pass2-sequence"


def run_tool(command: list[str], *, context: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise BestOfNError(f"{context} exit {completed.returncode}{suffix}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BestOfNError(f"invalid JSON document {path}: {error}") from error


def decode_rgb(
    path: Path, output: Path, *, frame_count: int, width: int, height: int
) -> list[np.ndarray]:
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("frame-*.png"):
        stale.unlink()
    run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            str(frame_count),
            "-vf",
            f"scale={width}:{height}",
            str(output / "frame-%06d.png"),
            "-y",
        ],
        context=f"RGB decode {path}",
    )
    frames = sorted(output.glob("frame-*.png"))
    if len(frames) != frame_count:
        raise BestOfNError(f"decoded {len(frames)} frames from {path}, expected {frame_count}")
    result: list[np.ndarray] = []
    for frame in frames:
        with Image.open(frame) as image:
            result.append(np.asarray(image.convert("RGB")).copy())
    return result


def decode_gray(
    path: Path, output: Path, *, frame_count: int, width: int, height: int
) -> list[np.ndarray]:
    frames = decode_rgb(path, output, frame_count=frame_count, width=width, height=height)
    return [np.asarray(Image.fromarray(frame).convert("L")) for frame in frames]


def overlay(frame: np.ndarray, mask: np.ndarray, polarity: str = ROI_BLACK) -> np.ndarray:
    """Tint the removal region of one mask under its declared polarity."""
    target = target_region(mask, polarity)
    result = frame.copy()
    result[target] = np.round(result[target] * 0.55 + np.array((255, 0, 255)) * 0.45)
    return result.astype(np.uint8)


def write_contact_sheet(
    *,
    plate: list[np.ndarray],
    candidate: list[np.ndarray],
    masks: list[np.ndarray],
    output: Path,
    selected_frames: list[int],
    target_polarity: str = ROI_BLACK,
) -> None:
    rows: list[np.ndarray] = []
    for frame in selected_frames:
        panels = [
            Image.fromarray(plate[frame]).resize((320, 180)),
            Image.fromarray(overlay(plate[frame], masks[frame], target_polarity)).resize(
                (320, 180)
            ),
            Image.fromarray(candidate[frame]).resize((320, 180)),
        ]
        row = np.concatenate([np.asarray(item) for item in panels], axis=1)
        label = Image.new("RGB", (row.shape[1], 30), (0, 0, 0))
        ImageDraw.Draw(label).text((12, 8), f"frame {frame}", fill=(255, 255, 255))
        rows.append(np.concatenate((np.asarray(label), row), axis=0))
    Image.fromarray(np.concatenate(rows, axis=0)).save(output, quality=92)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size=size))) > 128


def write_golden_masks(
    *,
    plate: list[np.ndarray],
    candidate: list[np.ndarray],
    masks: list[np.ndarray],
    output: Path,
    changed_threshold: float,
    cleanup_radius: int,
    fps: float,
    target_polarity: str = ROI_BLACK,
) -> tuple[Path, Path]:
    inserted_path = output / "golden-inserted-mask.mp4"
    cleanup_path = output / "golden-cleanup-mask.mp4"
    inserted_frames: list[np.ndarray] = []
    cleanup_frames: list[np.ndarray] = []
    for source, generated, mask in zip(plate, candidate, masks, strict=True):
        difference = np.abs(generated.astype(np.float64) - source.astype(np.float64)).mean(axis=2)
        target = target_region(mask, target_polarity)
        inserted = (difference > changed_threshold) & target
        cleanup = dilate(inserted, cleanup_radius) & ~inserted
        inserted_frames.append(np.where(inserted, 0, 255).astype(np.uint8))
        cleanup_frames.append(np.where(cleanup, 0, 255).astype(np.uint8))
    write_gray_video(inserted_frames, inserted_path, fps=fps)
    write_gray_video(cleanup_frames, cleanup_path, fps=fps)
    return inserted_path, cleanup_path


def write_rgb_video(frames: list[np.ndarray], output: Path, *, fps: float) -> None:
    if not frames or fps <= 0:
        raise BestOfNError("RGB video output requires nonempty frames and positive fps")
    with TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(staging / f"frame-{index:06d}.png")
        run_tool(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                _ffmpeg_framerate(fps),
                "-i",
                str(staging / "frame-%06d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(output),
                "-y",
            ],
            context=f"RGB encode {output}",
        )


def demote_stale_success_reports(output: Path) -> None:
    """Move old success reports aside before a new campaign is validated."""
    for name in (LEGACY_REPORT_NAME, CHAIN_REPORT_NAME):
        report = output / name
        if not report.is_file():
            continue
        failed = output / "failed"
        failed.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        report.replace(failed / f"{stamp}-{name}")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_state() -> tuple[str, bool]:
    """Return the source revision and whether the working tree is dirty."""
    root = repository_root()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown", False
    return revision, dirty


def verify_digest(path: Path, declared: str | None, label: str) -> None:
    """Fail closed when a declared artifact digest does not match the file."""
    if declared is None:
        return
    measured = sha256(path)
    if measured != declared:
        raise BestOfNError(f"{label} sha256 mismatch: declared {declared}, measured {measured}")


def manifest_input_paths(plan: ChainPlan) -> set[Path]:
    """Return every manifest-referenced media path that the campaign reads."""
    paths: set[Path] = {plan.plate, plan.target_mask}
    for spec in plan.candidates:
        paths.add(spec.insertion_video)
        if spec.inserted_mask is not None:
            paths.add(spec.inserted_mask)
        if spec.cleanup_mask is not None:
            paths.add(spec.cleanup_mask)
        if spec.pass2 is not None:
            paths.update(
                {
                    spec.pass2.native.path,
                    spec.pass2.retimed.path,
                    spec.pass2.result.path,
                    spec.pass2.comparator.path,
                }
            )
    return paths


def clear_stale_outputs(output: Path, protected_paths: Iterable[Path] = ()) -> None:
    """Remove this tool's stale outputs without deleting campaign inputs."""
    protected = {path.resolve() for path in protected_paths}
    for stale_name in (
        "golden-candidate.mp4",
        "golden-inserted-mask.mp4",
        "golden-cleanup-mask.mp4",
        LEGACY_REPORT_NAME,
        CHAIN_REPORT_NAME,
    ):
        stale = output / stale_name
        if stale.is_file() and stale.resolve() not in protected:
            stale.unlink()
    for pattern in ("contact-*.jpg",):
        for stale in output.glob(pattern):
            if stale.is_file() and stale.resolve() not in protected:
                stale.unlink()
    normalized_dir = output / NORMALIZED_DIR_NAME
    if normalized_dir.is_dir():
        for stale in normalized_dir.glob("candidate-*-insertion.mp4"):
            if stale.is_file() and stale.resolve() not in protected:
                stale.unlink()
        try:
            normalized_dir.rmdir()
        except OSError:
            pass
    stale_sequence = output / SEQUENCE_DIR_NAME
    if stale_sequence.is_dir():
        shutil.rmtree(stale_sequence)


def _derived_inserted_mask(
    plate: list[np.ndarray],
    candidate: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    changed_threshold: float,
    target_polarity: str,
) -> list[np.ndarray]:
    """Derive the inserted mask from the plate difference, as the legacy lane does."""
    changed = changed_frames(plate, candidate, changed_threshold=changed_threshold)
    target = [target_region(frame, target_polarity) for frame in masks]
    return [
        changed_frame & target_frame
        for changed_frame, target_frame in zip(changed, target, strict=True)
    ]


def _declared_masks(spec: CandidateSpec, *, plate_info: Any) -> tuple[Path, Path]:
    """Return the declared mask paths after checking them against the plate."""
    if spec.inserted_mask is None or spec.cleanup_mask is None:
        raise BestOfNError(f"candidate {spec.candidate_id} declares no inserted or cleanup mask")
    for label, path in (("inserted_mask", spec.inserted_mask), ("cleanup_mask", spec.cleanup_mask)):
        info = probe_video(path)
        if info.frames != plate_info.frames:
            raise BestOfNError(
                f"candidate {spec.candidate_id} {label} has {info.frames} frames, "
                f"expected {plate_info.frames}"
            )
        if (info.width, info.height) != (plate_info.width, plate_info.height):
            raise BestOfNError(
                f"candidate {spec.candidate_id} {label} is {info.width}x{info.height}, "
                f"expected {plate_info.width}x{plate_info.height}"
            )
    return spec.inserted_mask, spec.cleanup_mask


def _ring_pixels(inserted: list[np.ndarray], cleanup: list[np.ndarray]) -> int:
    """Count the cleanup-ring pixels that sit outside the inserted region."""
    total = 0
    for inserted_frame, cleanup_frame in zip(inserted, cleanup, strict=True):
        total += int(np.count_nonzero(cleanup_frame & ~inserted_frame))
    return total


def _verify_pass2_media(artifact: Pass2MediaArtifact, *, candidate_id: str, label: str) -> None:
    """Probe one Pass 2 media artifact against its declared digest and geometry."""
    info = probe_video(artifact.path)
    verify_digest(artifact.path, artifact.sha256, f"candidate {candidate_id} {label}")
    geometry = (info.width, info.height, info.frames)
    declared = (artifact.width, artifact.height, artifact.frames)
    if geometry != declared:
        raise BestOfNError(
            f"candidate {candidate_id} {label} geometry mismatch: declared {declared}, "
            f"probed {geometry}"
        )
    declared_fps = float(Fraction(artifact.fps))
    if not math.isclose(info.fps, declared_fps, rel_tol=1e-6, abs_tol=1e-9):
        raise BestOfNError(
            f"candidate {candidate_id} {label} fps mismatch: declared {artifact.fps}, "
            f"probed {info.fps}"
        )


def _validated_pass2_records(
    plan: ChainPlan,
    scores: list[ChainCandidateScore],
    artifacts: dict[str, dict[str, Any]],
    *,
    plate_info: Any,
) -> dict[str, dict[str, Any]]:
    """Verify and score Pass 2 packages without executing any model."""
    records: dict[str, dict[str, Any]] = {}
    score_by_id = {item.candidate_id: item for item in scores}
    expected_fps_tolerance: float | None = None
    for spec in plan.candidates:
        if spec.pass2 is None:
            if plan.require_full_chain and score_by_id[spec.candidate_id].survivor:
                raise BestOfNError(
                    f"candidate {spec.candidate_id} is a pre-Pass2 survivor but has no "
                    "pass2 record; require_full_chain cannot rank it"
                )
            continue
        candidate_id = spec.candidate_id
        _verify_pass2_media(spec.pass2.native, candidate_id=candidate_id, label="pass2.native")
        _verify_pass2_media(spec.pass2.retimed, candidate_id=candidate_id, label="pass2.retimed")
        _verify_pass2_media(spec.pass2.result, candidate_id=candidate_id, label="pass2.result")
        verify_digest(
            spec.pass2.comparator.path,
            spec.pass2.comparator.sha256,
            f"candidate {candidate_id} pass2.comparator",
        )
        comparator_report = read_json(spec.pass2.comparator.path)
        candidate_artifacts = artifacts[candidate_id]
        insertion = candidate_artifacts.get("insertion_video")
        inserted = candidate_artifacts.get("inserted_mask")
        cleanup = candidate_artifacts.get("cleanup_mask")
        if not (
            isinstance(insertion, dict) and isinstance(inserted, dict) and isinstance(cleanup, dict)
        ):
            raise BestOfNError(
                f"candidate {candidate_id} needs measured insertion and mask digests for Pass 2"
            )
        if spec.pass2.result.sha256 != str(insertion["sha256"]):
            raise BestOfNError(
                f"candidate {candidate_id} pass2.result does not bind to its insertion artifact"
            )
        thresholds = comparator_report.get("thresholds")
        if not isinstance(thresholds, dict) or isinstance(thresholds.get("fps_tolerance"), bool):
            raise BestOfNError(f"candidate {candidate_id} pass2 comparator thresholds are invalid")
        fps_tolerance = float(thresholds["fps_tolerance"])
        if not math.isfinite(fps_tolerance) or fps_tolerance < 0.0:
            raise BestOfNError(
                f"candidate {candidate_id} pass2 comparator fps_tolerance is invalid"
            )
        if expected_fps_tolerance is None:
            expected_fps_tolerance = fps_tolerance
        elif fps_tolerance != expected_fps_tolerance:
            raise BestOfNError("pass2 comparator fps_tolerance differs between candidates")
        records[candidate_id] = full_chain_pass2_record(
            pre_pass2=score_by_id[candidate_id].pre_pass2,
            pass2=spec.pass2,
            comparator_report=comparator_report,
            inserted_mask_sha256=str(inserted["sha256"]),
            cleanup_mask_sha256=str(cleanup["sha256"]),
            candidate_result_sha256=str(insertion["sha256"]),
            expected_changed_threshold=plan.thresholds.changed_threshold,
            expected_fps_tolerance=expected_fps_tolerance,
            expected_width=plate_info.width,
            expected_height=plate_info.height,
            expected_frames=plate_info.frames,
            expected_fps=plate_info.fps,
        )
    return records


def execute_campaign(plan: ChainPlan, output: Path) -> dict[str, Any]:
    """Measure every candidate, stage the Pass 2 inputs, and write the report."""
    legacy = plan.mode == MODE_LEGACY_ADAPTER
    plate_path = plan.plate
    mask_path = plan.target_mask
    plate_info = probe_video(plate_path)
    mask_info = probe_video(mask_path)
    if plate_info.width <= 0 or plate_info.height <= 0:
        raise BestOfNError("invalid source geometry")
    if plate_info.frames != mask_info.frames:
        raise BestOfNError("source plate and mask frame counts differ")
    verify_digest(plate_path, plan.plate_sha256, "plate")
    verify_digest(mask_path, plan.target_mask_sha256, "target_mask")
    output.mkdir(parents=True, exist_ok=True)
    campaign_inputs = manifest_input_paths(plan)
    clear_stale_outputs(output, campaign_inputs)
    missing_inputs = sorted(str(path) for path in campaign_inputs if not path.is_file())
    if missing_inputs:
        raise BestOfNError(f"manifest inputs are missing after stale cleanup: {missing_inputs}")
    work = output / "decode"
    plate = decode_rgb(
        plate_path,
        work / "plate",
        frame_count=plate_info.frames,
        width=plate_info.width,
        height=plate_info.height,
    )
    masks = decode_gray(
        mask_path,
        work / "masks",
        frame_count=mask_info.frames,
        width=plate_info.width,
        height=plate_info.height,
    )
    thresholds = plan.thresholds
    scores: list[ChainCandidateScore] = []
    candidate_frames: dict[str, list[np.ndarray]] = {}
    declared_masks: dict[str, tuple[Path, Path]] = {}
    legacy_artifacts: dict[str, dict[str, Any]] = {}
    chain_artifacts: dict[str, dict[str, Any]] = {}
    contact_sheets: dict[str, str] = {}
    selected_frames = [frame for frame in SELECTED_FRAMES if frame < plate_info.frames]
    for spec in plan.candidates:
        candidate_id = spec.candidate_id
        artifact = spec.insertion_video
        artifact_info = probe_video(artifact)
        if artifact_info.frames < plate_info.frames:
            raise BestOfNError(
                f"candidate {candidate_id} has {artifact_info.frames} frames, "
                f"expected at least {plate_info.frames}"
            )
        verify_digest(
            artifact, spec.insertion_video_sha256, f"candidate {candidate_id} insertion_video"
        )
        frames = decode_rgb(
            artifact,
            work / f"candidate-{candidate_id}",
            frame_count=plate_info.frames,
            width=plate_info.width,
            height=plate_info.height,
        )
        candidate_frames[candidate_id] = frames
        if legacy:
            inserted = _derived_inserted_mask(
                plate,
                frames,
                masks,
                changed_threshold=thresholds.changed_threshold,
                target_polarity=plan.target_polarity,
            )
            cleanup: list[np.ndarray] = []
            mask_source = MASK_SOURCE_DERIVED
            inserted_path: Path | None = None
            cleanup_path: Path | None = None
            ring_pixels = 0
        else:
            inserted_path, cleanup_path = _declared_masks(spec, plate_info=plate_info)
            verify_digest(
                inserted_path, spec.inserted_mask_sha256, f"candidate {candidate_id} inserted_mask"
            )
            verify_digest(
                cleanup_path, spec.cleanup_mask_sha256, f"candidate {candidate_id} cleanup_mask"
            )
            inserted_raw = decode_gray(
                inserted_path,
                work / f"inserted-{candidate_id}",
                frame_count=plate_info.frames,
                width=plate_info.width,
                height=plate_info.height,
            )
            cleanup_raw = decode_gray(
                cleanup_path,
                work / f"cleanup-{candidate_id}",
                frame_count=plate_info.frames,
                width=plate_info.width,
                height=plate_info.height,
            )
            inserted = [frame < 128 for frame in inserted_raw]
            cleanup = [frame < 128 for frame in cleanup_raw]
            declared_masks[candidate_id] = (inserted_path, cleanup_path)
            mask_source = MASK_SOURCE_DECLARED
            ring_pixels = _ring_pixels(inserted, cleanup)
        score = chain_score(
            candidate_id=candidate_id,
            kind=spec.kind,
            plate=plate,
            candidate=frames,
            masks=masks,
            inserted=inserted,
            thresholds=thresholds,
            target_polarity=plan.target_polarity,
            inserted_mask_source=mask_source,
        )
        scores.append(score)
        sheet = output / f"contact-{candidate_id}.jpg"
        write_contact_sheet(
            plate=plate,
            candidate=frames,
            masks=masks,
            output=sheet,
            selected_frames=selected_frames,
            target_polarity=plan.target_polarity,
        )
        contact_sheets[candidate_id] = str(sheet)
        artifact_record = {
            "path": str(artifact),
            "sha256": sha256(artifact),
            "probe": asdict(artifact_info),
        }
        legacy_artifacts[candidate_id] = artifact_record
        chain_artifacts[candidate_id] = {
            "insertion_video": artifact_record,
            "inserted_mask": (
                {"path": str(inserted_path), "sha256": sha256(inserted_path)}
                if inserted_path is not None
                else None
            ),
            "cleanup_mask": (
                {"path": str(cleanup_path), "sha256": sha256(cleanup_path)}
                if cleanup_path is not None
                else None
            ),
            "inserted_mask_source": mask_source,
            "cleanup_ring_pixels_declared": spec.cleanup_ring_pixels,
            "cleanup_ring_pixels_measured": ring_pixels,
            "fit": spec.fit,
        }
    if legacy:
        return _write_legacy_report(
            plan,
            output,
            plate=plate,
            masks=masks,
            plate_info=plate_info,
            mask_info=mask_info,
            scores=scores,
            candidate_frames=candidate_frames,
            contact_sheets=contact_sheets,
            artifacts=legacy_artifacts,
        )
    pass2_records = _validated_pass2_records(plan, scores, chain_artifacts, plate_info=plate_info)
    return _write_chain_report(
        plan,
        output,
        plate_info=plate_info,
        mask_info=mask_info,
        scores=scores,
        candidate_frames=candidate_frames,
        declared_masks=declared_masks,
        contact_sheets=contact_sheets,
        artifacts=chain_artifacts,
        pass2_records=pass2_records,
    )


def _write_legacy_report(
    plan: ChainPlan,
    output: Path,
    *,
    plate: list[np.ndarray],
    masks: list[np.ndarray],
    plate_info: Any,
    mask_info: Any,
    scores: list[ChainCandidateScore],
    candidate_frames: dict[str, list[np.ndarray]],
    contact_sheets: dict[str, str],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Write the legacy report, whose scores and survivor order are unchanged."""
    thresholds = plan.thresholds
    survivors = [item for item in scores if all(item.pre_pass2.gates.values())]
    ranked = sorted(survivors, key=lambda item: (-item.pre_pass2.objective, item.candidate_id))
    golden_mask_records: dict[str, str] = {}
    if ranked:
        golden = ranked[0]
        inserted_mask, cleanup_mask = write_golden_masks(
            plate=plate,
            candidate=candidate_frames[golden.candidate_id],
            masks=masks,
            output=output,
            changed_threshold=thresholds.changed_threshold,
            cleanup_radius=plan.cleanup_radius,
            fps=plate_info.fps,
            target_polarity=plan.target_polarity,
        )
        golden_video = output / "golden-candidate.mp4"
        write_rgb_video(candidate_frames[golden.candidate_id], golden_video, fps=plate_info.fps)
        golden_sequence = output / SEQUENCE_DIR_NAME
        build_insertion_sequence(
            input_video=golden_video,
            inserted_mask=inserted_mask,
            cleanup_mask=cleanup_mask,
            background_prompt=plan.background_prompt,
            sequence_dir=golden_sequence,
        )
        golden_mask_records = {
            "normalized_video": str(golden_video),
            "inserted": str(inserted_mask),
            "cleanup": str(cleanup_mask),
            "sequence": str(golden_sequence),
        }
    report: dict[str, Any] = {
        "schema": "void-fusion-lab.best-of-n.report.v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "mode": MODE_LEGACY_ADAPTER,
        "source": {
            "plate": str(plan.plate),
            "plate_sha256": sha256(plan.plate),
            "plate_probe": asdict(plate_info),
            "mask": str(plan.target_mask),
            "mask_sha256": sha256(plan.target_mask),
            "mask_probe": asdict(mask_info),
        },
        "thresholds": {
            "changed_threshold": thresholds.changed_threshold,
            "min_object_coverage": thresholds.min_object_coverage,
            "min_object_precision": thresholds.min_object_precision,
            "max_background_change": thresholds.max_background_change,
            "max_temporal_pop": thresholds.max_temporal_pop,
        },
        "candidates": [asdict(item.pre_pass2) for item in scores],
        "survivors": [item.candidate_id for item in ranked],
        "golden": asdict(ranked[0].pre_pass2) if ranked else None,
        "golden_masks_and_sequence": golden_mask_records,
        "contact_sheets": contact_sheets,
        "artifacts": artifacts,
        "non_claims": [
            "This selector measures change placement, background preservation, and temporal "
            "stability; it does not prove object identity or artistic quality.",
            "Multimodal review is a separate mandatory audit and is not substituted by the "
            "deterministic objective.",
            "No VOID Pass 2 execution is claimed by selection alone.",
        ],
    }
    (output / LEGACY_REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _write_chain_report(
    plan: ChainPlan,
    output: Path,
    *,
    plate_info: Any,
    mask_info: Any,
    scores: list[ChainCandidateScore],
    candidate_frames: dict[str, list[np.ndarray]],
    declared_masks: dict[str, tuple[Path, Path]],
    contact_sheets: dict[str, str],
    artifacts: dict[str, dict[str, Any]],
    pass2_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Stage one Pass 2 sequence per survivor and write the chain report."""
    sequences: dict[str, str] = {}
    for item in scores:
        if not item.survivor:
            continue
        candidate_id = item.candidate_id
        normalized_dir = output / NORMALIZED_DIR_NAME
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized = normalized_dir / f"candidate-{candidate_id}-insertion.mp4"
        protected_inputs = {path.resolve() for path in manifest_input_paths(plan)}
        if normalized.resolve() in protected_inputs:
            raise BestOfNError(f"staging {normalized} would overwrite a declared manifest input")
        write_rgb_video(candidate_frames[candidate_id], normalized, fps=plate_info.fps)
        inserted_path, cleanup_path = declared_masks[candidate_id]
        sequence_dir = output / SEQUENCE_DIR_NAME / candidate_id
        build_insertion_sequence(
            input_video=normalized,
            inserted_mask=inserted_path,
            cleanup_mask=cleanup_path,
            background_prompt=plan.background_prompt,
            sequence_dir=sequence_dir,
        )
        sequences[candidate_id] = str(sequence_dir)
    revision, dirty = git_state()
    report = build_chain_report(
        plan=plan,
        plate={
            "path": str(plan.plate),
            "sha256": sha256(plan.plate),
            "probe": asdict(plate_info),
        },
        target_mask={
            "path": str(plan.target_mask),
            "sha256": sha256(plan.target_mask),
            "polarity": plan.target_polarity,
            "probe": asdict(mask_info),
        },
        scores=scores,
        artifacts=artifacts,
        contact_sheets=contact_sheets,
        sequences=sequences,
        pass2_records=pass2_records,
        revision=revision,
        dirty=dirty,
        recorded_at=datetime.now(UTC).isoformat(),
    )
    (output / CHAIN_REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    return report


def select(manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run one campaign and return its report."""
    plan = plan_from_manifest(manifest)
    return execute_campaign(plan, output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


def select_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="best_of_n.py select", description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["select"]:
        args = select_parser().parse_args(arguments[1:])
    else:
        args = parser().parse_args(arguments)
    demote_stale_success_reports(args.out)
    plan = plan_from_manifest(read_json(args.manifest))
    report = execute_campaign(plan, args.out)
    if plan.mode == MODE_LEGACY_ADAPTER:
        print(json.dumps({"golden": report["golden"], "survivors": report["survivors"]}, indent=2))
        return 0 if report["golden"] is not None else EXIT_NO_SURVIVOR
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "survivors": report["survivors"],
                "ranking_pre_pass2": report["ranking_pre_pass2"],
                "ranking_full_chain": report["ranking_full_chain"],
            },
            indent=2,
        )
    )
    return 0 if report["selected"] is not None else EXIT_NO_SURVIVOR


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        BestOfNError,
        MediaError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Best-of-N selection failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
