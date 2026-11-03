"""Canonical deterministic contracts for the Best-of-N chain selector.

This module owns the manifest schemas, the gate numerics, the ranking, and the
report shapes for the full-chain Best-of-N campaign. It performs no media
decode, no subprocess, and no model call: the media lane lives in
``scripts/best_of_n.py``, which imports these contracts unchanged.

Two manifest shapes are accepted:

``void-fusion-lab.best-of-n.chain-manifest.v1``
    One shared plate, one shared target mask with an explicit polarity, and N
    candidates that each carry an insertion video plus an inserted mask and a
    cleanup mask. This shape binds every candidate to its own mask, so the
    ``remask_consistency`` gate can reject a stale or mismatched mask.

the legacy flat manifest
    ``source_plate``, ``mask_video``, ``background_prompt``, flat numeric
    thresholds, and ``candidates[].artifact``. It is adapted into the same
    internal plan, keeps its own six gates unchanged, and derives its masks
    from the plate difference exactly as it always has.

When ``require_full_chain`` is true, every pre-Pass2 survivor must also carry a
validated Pass 2 package: the native VOID result, its source-cadence retiming,
the no-Pass2 comparator baseline derivative, and the strict comparator report.
The selector then ranks complete chains rather than stopping at insertion.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

CHAIN_MANIFEST_SCHEMA = "void-fusion-lab.best-of-n.chain-manifest.v1"
CHAIN_REPORT_SCHEMA = "void-fusion-lab.best-of-n.chain-report.v1"
PASS2_COMPARATOR_REPORT_SCHEMA = "void-fusion-lab.pass2-compare.report.v3"
LEGACY_REPORT_SCHEMA = "void-fusion-lab.best-of-n.report.v1"

DEFAULT_CHANGED_THRESHOLD = 10.0
DEFAULT_MIN_COVERAGE = 0.02
DEFAULT_MIN_PRECISION = 0.50
DEFAULT_MAX_BACKGROUND_CHANGE = 0.02
DEFAULT_MAX_TEMPORAL_POP = 0.90
DEFAULT_MIN_REMASK_AGREEMENT = 0.50

EXIT_NO_SURVIVOR = 3

SAFE_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
RATIONAL_FPS_PATTERN = re.compile(r"[0-9]+(?:/[0-9]+)?\Z")

ROI_BLACK = "roi_black"
FOREGROUND_WHITE = "foreground_white"
POLARITIES = (ROI_BLACK, FOREGROUND_WHITE)

FITTED_INSERTION = "fitted_insertion"
GENERATED_REPLACEMENT = "generated_replacement"
CANDIDATE_KINDS = (FITTED_INSERTION, GENERATED_REPLACEMENT)

MODE_CHAIN = "chain"
MODE_LEGACY_ADAPTER = "legacy-adapter"

MASK_SOURCE_DECLARED = "declared"
MASK_SOURCE_DERIVED = "derived_from_plate_difference"


class BestOfNError(RuntimeError):
    """A Best-of-N input or candidate is invalid."""


@dataclass(frozen=True)
class CandidateScore:
    """The pre-Pass2 measurement of one candidate against the shared plate."""

    candidate_id: str
    frames: int
    target_present_frames: int
    changed_pixels: int
    target_pixels: int
    background_pixels: int
    inserted_precision: float
    inserted_coverage: float
    inserted_iou: float
    background_change_ratio: float
    background_mae: float
    temporal_centroid_pop: float
    temporal_mass_pop: float
    temporal_pop_score: float
    objective: float
    gates: dict[str, bool]
    failure_reason: str = ""


@dataclass(frozen=True)
class ChainThresholds:
    """Every numeric bound the selector applies, recorded in every report."""

    changed_threshold: float = DEFAULT_CHANGED_THRESHOLD
    min_object_coverage: float = DEFAULT_MIN_COVERAGE
    min_object_precision: float = DEFAULT_MIN_PRECISION
    max_background_change: float = DEFAULT_MAX_BACKGROUND_CHANGE
    max_temporal_pop: float = DEFAULT_MAX_TEMPORAL_POP
    min_remask_agreement: float = DEFAULT_MIN_REMASK_AGREEMENT

    def as_record(self) -> dict[str, float]:
        return {
            "changed_threshold": self.changed_threshold,
            "min_object_coverage": self.min_object_coverage,
            "min_object_precision": self.min_object_precision,
            "max_background_change": self.max_background_change,
            "max_temporal_pop": self.max_temporal_pop,
            "min_remask_agreement": self.min_remask_agreement,
        }


@dataclass(frozen=True)
class Pass2MediaArtifact:
    """A strict Pass 2 media record with digest and complete geometry."""

    path: Path
    sha256: str
    width: int
    height: int
    frames: int
    fps: str
    pixel_format: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": self.fps,
            **({"pixel_format": self.pixel_format} if self.pixel_format else {}),
        }


@dataclass(frozen=True)
class Pass2ComparatorArtifact:
    """The strict comparator report bound to one candidate package."""

    path: Path
    sha256: str

    def record(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Pass2Record:
    """All externally produced artifacts needed to rank one complete chain."""

    native: Pass2MediaArtifact
    retimed: Pass2MediaArtifact
    result: Pass2MediaArtifact
    comparator: Pass2ComparatorArtifact


@dataclass(frozen=True)
class CandidateSpec:
    """One normalized campaign candidate and the artifacts it declares."""

    candidate_id: str
    kind: str
    insertion_video: Path
    insertion_video_sha256: str | None
    inserted_mask: Path | None
    inserted_mask_sha256: str | None
    cleanup_mask: Path | None
    cleanup_mask_sha256: str | None
    cleanup_ring_pixels: int | None = None
    fit: dict[str, Any] | None = None
    pass2: Pass2Record | None = None


@dataclass(frozen=True)
class ChainPlan:
    """A manifest normalized into the single internal campaign plan."""

    mode: str
    plate: Path
    plate_sha256: str | None
    target_mask: Path
    target_mask_sha256: str | None
    target_polarity: str
    background_prompt: str
    thresholds: ChainThresholds
    candidates: tuple[CandidateSpec, ...]
    cleanup_radius: int = 7
    require_full_chain: bool = False


@dataclass(frozen=True)
class ChainCandidateScore:
    """A candidate score plus the mask-binding evidence the chain adds."""

    candidate_id: str
    kind: str
    pre_pass2: CandidateScore
    remask_agreement: float
    insertion_target_overlap: float
    remask_consistent: bool
    inserted_mask_source: str
    failure_reason: str = ""

    @property
    def gates(self) -> dict[str, bool]:
        return {**self.pre_pass2.gates, "remask_consistency": self.remask_consistent}

    @property
    def survivor(self) -> bool:
        return all(self.pre_pass2.gates.values()) and self.remask_consistent

    def pre_pass2_record(self) -> dict[str, Any]:
        record = asdict(self.pre_pass2)
        record["gates"] = self.gates
        record["remask_agreement"] = self.remask_agreement
        record["insertion_target_overlap"] = self.insertion_target_overlap
        return record


def target_region(mask: np.ndarray, polarity: str) -> np.ndarray:
    """Return the removal region of one mask under an explicit polarity."""
    if polarity == ROI_BLACK:
        return mask < 128
    if polarity == FOREGROUND_WHITE:
        return mask >= 128
    raise BestOfNError(f"unsupported mask polarity: {polarity!r}")


def changed_frames(
    plate: list[np.ndarray], candidate: list[np.ndarray], *, changed_threshold: float
) -> list[np.ndarray]:
    """Return the per-frame changed-pixel masks between plate and candidate."""
    frames: list[np.ndarray] = []
    for source, generated in zip(plate, candidate, strict=True):
        difference = np.abs(generated.astype(np.float64) - source.astype(np.float64)).mean(axis=2)
        frames.append(difference > changed_threshold)
    return frames


def temporal_pops(changed: list[np.ndarray]) -> tuple[float, float, float]:
    """Measure centroid, mass, and combined temporal discontinuity."""
    centroids: list[np.ndarray] = []
    masses: list[float] = []
    for frame in changed:
        coordinates = np.argwhere(frame)
        if len(coordinates) == 0:
            raise BestOfNError("temporal scoring requires changed pixels on every frame")
        centroids.append(coordinates.mean(axis=0))
        masses.append(float(len(coordinates)))
    centroid_steps = [
        float(np.linalg.norm(centroids[index] - centroids[index - 1]))
        for index in range(1, len(centroids))
    ]
    mass_steps = [abs(masses[index] - masses[index - 1]) for index in range(1, len(masses))]
    centroid_scale = max(float(np.median(centroid_steps)), 1.0) if centroid_steps else 1.0
    mass_scale = max(float(np.median(mass_steps)), 1.0) if mass_steps else 1.0
    centroid_pops = [
        min(
            1.0,
            float(
                np.linalg.norm(centroids[index + 1] - 2.0 * centroids[index] + centroids[index - 1])
            )
            / centroid_scale,
        )
        for index in range(1, len(centroids) - 1)
    ]
    mass_pops = [
        min(1.0, abs(masses[index + 1] - 2.0 * masses[index] + masses[index - 1]) / mass_scale)
        for index in range(1, len(masses) - 1)
    ]
    centroid_pop = float(np.mean(centroid_pops)) if centroid_pops else 0.0
    mass_pop = float(np.mean(mass_pops)) if mass_pops else 0.0
    return centroid_pop, mass_pop, 0.5 * centroid_pop + 0.5 * mass_pop


def score_candidate(
    *,
    candidate_id: str,
    plate: list[np.ndarray],
    candidate: list[np.ndarray],
    masks: list[np.ndarray],
    changed_threshold: float = DEFAULT_CHANGED_THRESHOLD,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_precision: float = DEFAULT_MIN_PRECISION,
    max_background_change: float = DEFAULT_MAX_BACKGROUND_CHANGE,
    max_temporal_pop: float = DEFAULT_MAX_TEMPORAL_POP,
    target_polarity: str = ROI_BLACK,
) -> CandidateScore:
    """Measure one candidate against the shared plate and target mask."""
    if not plate or len(plate) != len(candidate) or len(plate) != len(masks):
        raise BestOfNError("candidate scoring requires equal, nonempty plate/candidate/mask lists")
    first_shape = plate[0].shape
    if any(frame.shape != first_shape for frame in plate + candidate):
        raise BestOfNError("plate/candidate RGB geometry mismatch")
    first_mask_shape = masks[0].shape
    if len(first_mask_shape) != 2 or any(frame.shape != first_mask_shape for frame in masks):
        raise BestOfNError("mask geometry mismatch")
    if first_shape[:2] != first_mask_shape:
        raise BestOfNError(f"RGB/mask geometry mismatch: {first_shape[:2]} != {first_mask_shape}")

    changed: list[np.ndarray] = []
    changed_pixels = 0
    target_pixels = 0
    background_pixels = 0
    changed_in_target = 0
    changed_in_background = 0
    target_present_frames = 0
    background_error_sum = 0.0
    for source, generated, mask in zip(plate, candidate, masks, strict=True):
        difference = np.abs(generated.astype(np.float64) - source.astype(np.float64)).mean(axis=2)
        frame_changed = difference > changed_threshold
        target = target_region(mask, target_polarity)
        background = ~target
        changed.append(frame_changed)
        changed_pixels += int(np.count_nonzero(frame_changed))
        target_pixels += int(np.count_nonzero(target))
        background_pixels += int(np.count_nonzero(background))
        changed_in_target += int(np.count_nonzero(frame_changed & target))
        changed_in_background += int(np.count_nonzero(frame_changed & background))
        target_present_frames += int(np.any(frame_changed & target))
        background_error_sum += float(difference[background].mean()) if np.any(background) else 0.0
    if target_pixels == 0 or background_pixels == 0:
        raise BestOfNError("mask must contain both target and background pixels")
    if changed_pixels == 0:
        raise BestOfNError(f"candidate {candidate_id!r} has no changed pixels")
    if max_background_change < 0.0:
        raise BestOfNError("max_background_change must be nonnegative")

    precision = changed_in_target / changed_pixels
    coverage = changed_in_target / target_pixels
    union = changed_pixels + target_pixels - changed_in_target
    iou = changed_in_target / union if union else 0.0
    background_ratio = changed_in_background / background_pixels
    background_mae = background_error_sum / len(plate)
    try:
        centroid_pop, mass_pop, temporal_pop = temporal_pops(changed)
    except BestOfNError as error:
        if "changed pixels on every frame" not in str(error):
            raise
        centroid_pop = mass_pop = temporal_pop = 1.0
    if max_background_change == 0.0:
        background_objective = 1.0 if background_ratio == 0.0 else 0.0
    else:
        background_objective = 1.0 - min(1.0, background_ratio / max_background_change)
    objective = (
        0.35 * precision
        + 0.30 * coverage
        + 0.20 * background_objective
        + 0.15 * (1.0 - temporal_pop)
    )
    gates = {
        "object_present_every_frame": all(np.any(frame) for frame in changed),
        "object_target_every_frame": target_present_frames == len(plate),
        "object_coverage": coverage >= min_coverage,
        "object_placement": precision >= min_precision,
        "background_preservation": background_ratio <= max_background_change,
        "temporal_stability": temporal_pop <= max_temporal_pop,
    }
    return CandidateScore(
        candidate_id=candidate_id,
        frames=len(plate),
        target_present_frames=target_present_frames,
        changed_pixels=changed_pixels,
        target_pixels=target_pixels,
        background_pixels=background_pixels,
        inserted_precision=precision,
        inserted_coverage=coverage,
        inserted_iou=iou,
        background_change_ratio=background_ratio,
        background_mae=background_mae,
        temporal_centroid_pop=centroid_pop,
        temporal_mass_pop=mass_pop,
        temporal_pop_score=temporal_pop,
        objective=objective,
        gates=gates,
    )


def failed_candidate_score(candidate_id: str, *, frames: int, reason: str) -> CandidateScore:
    """Return the closed-out score of a candidate that could not be measured."""
    gates = {
        "object_present_every_frame": False,
        "object_target_every_frame": False,
        "object_coverage": False,
        "object_placement": False,
        "background_preservation": False,
        "temporal_stability": False,
    }
    return CandidateScore(
        candidate_id=candidate_id,
        frames=frames,
        target_present_frames=0,
        changed_pixels=0,
        target_pixels=0,
        background_pixels=0,
        inserted_precision=0.0,
        inserted_coverage=0.0,
        inserted_iou=0.0,
        background_change_ratio=1.0,
        background_mae=1_000_000_000.0,
        temporal_centroid_pop=1.0,
        temporal_mass_pop=1.0,
        temporal_pop_score=1.0,
        objective=0.0,
        gates=gates,
        failure_reason=reason,
    )


def remask_metrics(
    *,
    changed: list[np.ndarray],
    inserted: list[np.ndarray],
    target: list[np.ndarray],
) -> tuple[float, float]:
    """Return the mask-binding agreement and the insertion overlap.

    ``remask_agreement`` is the share of the inserted-mask pixels that this
    candidate actually changed; it falls to zero when the mask carries no
    region of interest. ``insertion_target_overlap`` is the share of the
    removal region that the inserted mask claims.
    """
    matched = 0
    inserted_count = 0
    overlap_count = 0
    target_count = 0
    for changed_frame, inserted_frame, target_frame in zip(changed, inserted, target, strict=True):
        matched += int(np.count_nonzero(changed_frame & inserted_frame))
        inserted_count += int(np.count_nonzero(inserted_frame))
        overlap_count += int(np.count_nonzero(inserted_frame & target_frame))
        target_count += int(np.count_nonzero(target_frame))
    agreement = matched / inserted_count if inserted_count else 0.0
    overlap = overlap_count / target_count if target_count else 0.0
    return agreement, overlap


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BestOfNError(f"{label} must be a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BestOfNError(f"{label} must be a nonempty string")
    return value


def _reject_unknown(
    payload: dict[str, Any], allowed: set[str], label: str, deferred: dict[str, str]
) -> None:
    unknown = sorted(set(payload) - allowed)
    if not unknown:
        return
    for key in unknown:
        if key in deferred:
            raise BestOfNError(f"{label} key {key!r} is not supported: {deferred[key]}")
    raise BestOfNError(f"{label} has unsupported key(s): {', '.join(unknown)}")


def _optional_sha256(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise BestOfNError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _required_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise BestOfNError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _artifact(value: Any, label: str) -> dict[str, Any]:
    """Validate one artifact record and return its normalized fields."""
    record = _require_mapping(value, label)
    _reject_unknown(record, {"path", "sha256", "frames", "fps"}, label, {})
    path = _require_string(record.get("path"), f"{label}.path")
    if not Path(path).is_file():
        raise BestOfNError(f"{label}.path does not name an existing file: {path}")
    digest = _optional_sha256(record.get("sha256"), f"{label}.sha256")
    frames = record.get("frames")
    if frames is not None:
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
            raise BestOfNError(f"{label}.frames must be a positive integer")
    fps = record.get("fps")
    if fps is not None and not isinstance(fps, str):
        raise BestOfNError(f"{label}.fps must be a string")
    return {"path": path, "sha256": digest}


def _validate_pass2_media(value: Any, label: str) -> Pass2MediaArtifact:
    """Validate one mandatory Pass 2 media artifact without probing it."""
    record = _require_mapping(value, label)
    _reject_unknown(
        record,
        {"path", "sha256", "width", "height", "frames", "fps", "pixel_format"},
        label,
        {},
    )
    path = _require_string(record.get("path"), f"{label}.path")
    if not Path(path).is_file():
        raise BestOfNError(f"{label}.path does not name an existing file: {path}")
    digest = _required_sha256(record.get("sha256"), f"{label}.sha256")
    geometry: dict[str, int] = {}
    for key in ("width", "height", "frames"):
        raw = record.get(key)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
            raise BestOfNError(f"{label}.{key} must be a positive integer")
        geometry[key] = raw
    fps = _require_string(record.get("fps"), f"{label}.fps")
    if RATIONAL_FPS_PATTERN.fullmatch(fps) is None:
        raise BestOfNError(f"{label}.fps must be a positive rational such as 12/1")
    try:
        rational_fps = Fraction(fps)
        fps_value = float(rational_fps)
    except (ZeroDivisionError, OverflowError) as error:
        raise BestOfNError(f"{label}.fps must be a finite positive rational") from error
    if fps_value <= 0.0 or not np.isfinite(fps_value):
        raise BestOfNError(f"{label}.fps must be positive and finite")
    pixel_format = record.get("pixel_format")
    if pixel_format is not None:
        _require_string(pixel_format, f"{label}.pixel_format")
    return Pass2MediaArtifact(
        path=Path(path),
        sha256=digest,
        width=geometry["width"],
        height=geometry["height"],
        frames=geometry["frames"],
        fps=fps,
        pixel_format=pixel_format,
    )


def _validate_pass2(value: Any, label: str) -> Pass2Record:
    """Validate a complete Pass 2 package and its internally consistent geometry."""
    record = _require_mapping(value, label)
    _reject_unknown(record, {"native", "retimed", "result", "comparator"}, label, {})
    for key in ("native", "retimed", "result", "comparator"):
        if key not in record:
            raise BestOfNError(f"{label}.{key} is required for a full-chain candidate")
    native = _validate_pass2_media(record["native"], f"{label}.native")
    retimed = _validate_pass2_media(record["retimed"], f"{label}.retimed")
    result = _validate_pass2_media(record["result"], f"{label}.result")
    comparator_record = _require_mapping(record["comparator"], f"{label}.comparator")
    _reject_unknown(comparator_record, {"path", "sha256"}, f"{label}.comparator", {})
    comparator_path = _require_string(comparator_record.get("path"), f"{label}.comparator.path")
    if not Path(comparator_path).is_file():
        raise BestOfNError(
            f"{label}.comparator.path does not name an existing file: {comparator_path}"
        )
    comparator = Pass2ComparatorArtifact(
        path=Path(comparator_path),
        sha256=_required_sha256(comparator_record.get("sha256"), f"{label}.comparator.sha256"),
    )

    if retimed.sha256 == result.sha256:
        raise BestOfNError(
            f"{label}.retimed and {label}.result must not be identical; "
            "a self-compared baseline is not a completed Pass 2 chain"
        )
    for name, expected in (("retimed", retimed), ("result", result)):
        if (native.width, native.height, native.frames) != (
            expected.width,
            expected.height,
            expected.frames,
        ):
            raise BestOfNError(
                f"{label}.native and {label}.{name} must share width, height, and frame count"
            )
    if (retimed.width, retimed.height, retimed.frames, retimed.fps) != (
        result.width,
        result.height,
        result.frames,
        result.fps,
    ):
        raise BestOfNError(
            f"{label}.retimed and {label}.result must share exact geometry, frames, and fps"
        )
    return Pass2Record(native=native, retimed=retimed, result=result, comparator=comparator)


def _validate_fit(value: Any, label: str) -> dict[str, Any]:
    """Validate the declared fit record, which this slice only passes through."""
    fit = _require_mapping(value, label)
    allowed = {"asset", "pose_series", "pose_sha256", "transform", "smoothing_passes", "lighting"}
    _reject_unknown(fit, allowed, label, {})
    for key in ("asset", "pose_series"):
        if fit.get(key) is not None:
            _require_string(fit[key], f"{label}.{key}")
    _optional_sha256(fit.get("pose_sha256"), f"{label}.pose_sha256")
    passes = fit.get("smoothing_passes")
    if passes is not None:
        if not isinstance(passes, int) or isinstance(passes, bool) or passes < 0:
            raise BestOfNError(f"{label}.smoothing_passes must be a nonnegative integer")
    transform = fit.get("transform")
    if transform is not None:
        record = _require_mapping(transform, f"{label}.transform")
        _reject_unknown(record, {"dx", "dy", "scale"}, f"{label}.transform", {})
        for key in ("dx", "dy", "scale"):
            if record.get(key) is not None and not isinstance(record[key], (int, float)):
                raise BestOfNError(f"{label}.transform.{key} must be a number")
        scale = record.get("scale")
        if scale is not None and not 0.5 <= float(scale) <= 2.0:
            raise BestOfNError(f"{label}.transform.scale must lie between 0.5 and 2.0")
    lighting = fit.get("lighting")
    if lighting is not None:
        record = _require_mapping(lighting, f"{label}.lighting")
        _reject_unknown(
            record,
            {"key_energy", "key_size", "key_location", "key_rotation_deg"},
            f"{label}.lighting",
            {},
        )
        for key in ("key_energy", "key_size"):
            if record.get(key) is not None and not isinstance(record[key], (int, float)):
                raise BestOfNError(f"{label}.lighting.{key} must be a number")
        for key in ("key_location", "key_rotation_deg"):
            vector = record.get(key)
            if vector is None:
                continue
            if not isinstance(vector, list) or len(vector) != 3:
                raise BestOfNError(f"{label}.lighting.{key} must be a list of three numbers")
            if any(not isinstance(item, (int, float)) for item in vector):
                raise BestOfNError(f"{label}.lighting.{key} must be a list of three numbers")
    return fit


def _validate_thresholds(value: Any) -> ChainThresholds:
    """Validate the chain threshold block against its declared bounds."""
    record = _require_mapping(value, "thresholds")
    allowed = {
        "changed_threshold",
        "min_object_coverage",
        "min_object_precision",
        "max_background_change",
        "max_temporal_pop",
        "min_remask_agreement",
    }
    _reject_unknown(record, allowed, "thresholds", {})
    bounds: dict[str, tuple[float, float | None]] = {
        "changed_threshold": (0.0, None),
        "min_object_coverage": (0.0, 1.0),
        "min_object_precision": (0.0, 1.0),
        "max_background_change": (0.0, None),
        "max_temporal_pop": (0.0, 1.0),
        "min_remask_agreement": (0.0, 1.0),
    }
    values: dict[str, float] = {}
    for name, (lower, upper) in bounds.items():
        if name not in record or record[name] is None:
            continue
        raw = record[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise BestOfNError(f"thresholds.{name} must be a number")
        number = float(raw)
        if not np.isfinite(number) or number < lower or (upper is not None and number > upper):
            raise BestOfNError(
                f"thresholds.{name} must be finite and inside {(lower, upper)}: {number}"
            )
        values[name] = number
    return ChainThresholds(**values)


def validate_chain_manifest(payload: Any) -> ChainPlan:
    """Validate a chain manifest and return the normalized campaign plan."""
    document = _require_mapping(payload, "chain manifest")
    schema = document.get("schema")
    if schema != CHAIN_MANIFEST_SCHEMA:
        raise BestOfNError(f"chain manifest schema must be {CHAIN_MANIFEST_SCHEMA}: {schema!r}")
    allowed = {
        "schema",
        "require_full_chain",
        "plate",
        "target_mask",
        "background_prompt",
        "thresholds",
        "candidates",
        "_note",
        "$comment",
    }
    _reject_unknown(document, allowed, "chain manifest", {})
    for key in ("_note", "$comment"):
        if key in document:
            _require_string(document[key], key)

    plate = _artifact(document.get("plate"), "plate")
    mask_record = _require_mapping(document.get("target_mask"), "target_mask")
    _reject_unknown(mask_record, {"path", "sha256", "polarity"}, "target_mask", {})
    mask_path = _require_string(mask_record.get("path"), "target_mask.path")
    if not Path(mask_path).is_file():
        raise BestOfNError(f"target_mask.path does not name an existing file: {mask_path}")
    mask_digest = _optional_sha256(mask_record.get("sha256"), "target_mask.sha256")
    polarity = mask_record.get("polarity")
    if polarity not in POLARITIES:
        raise BestOfNError(f"target_mask.polarity must be one of {list(POLARITIES)}: {polarity!r}")

    background_prompt = _require_string(document.get("background_prompt"), "background_prompt")
    require_full_chain = document.get("require_full_chain", False)
    if not isinstance(require_full_chain, bool):
        raise BestOfNError("require_full_chain must be true or false")
    thresholds = (
        _validate_thresholds(document["thresholds"])
        if document.get("thresholds") is not None
        else ChainThresholds()
    )

    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise BestOfNError("candidates must be a nonempty list")
    specs: list[CandidateSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        label = f"candidates[{index}]"
        record = _require_mapping(raw, label)
        _reject_unknown(
            record,
            {
                "id",
                "kind",
                "insertion_video",
                "inserted_mask",
                "cleanup_mask",
                "cleanup_ring_pixels",
                "fit",
                "pass2",
            },
            label,
            {},
        )
        candidate_id = record.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise BestOfNError(f"candidate id must be unique and nonempty: {candidate_id!r}")
        if candidate_id in seen:
            raise BestOfNError(f"candidate id must be unique and nonempty: {candidate_id!r}")
        if SAFE_CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise BestOfNError(f"candidate id must be a safe path slug: {candidate_id!r}")
        seen.add(candidate_id)
        kind = record.get("kind")
        if kind not in CANDIDATE_KINDS:
            raise BestOfNError(f"{label}.kind must be one of {list(CANDIDATE_KINDS)}: {kind!r}")
        insertion = _artifact(record.get("insertion_video"), f"{label}.insertion_video")
        inserted = _artifact(record.get("inserted_mask"), f"{label}.inserted_mask")
        cleanup = _artifact(record.get("cleanup_mask"), f"{label}.cleanup_mask")
        ring = record.get("cleanup_ring_pixels")
        if ring is not None:
            if not isinstance(ring, int) or isinstance(ring, bool) or ring < 0:
                raise BestOfNError(f"{label}.cleanup_ring_pixels must be a nonnegative integer")
        fit = (
            _validate_fit(record["fit"], f"{label}.fit") if record.get("fit") is not None else None
        )
        pass2 = (
            _validate_pass2(record["pass2"], f"{label}.pass2")
            if record.get("pass2") is not None
            else None
        )
        specs.append(
            CandidateSpec(
                candidate_id=candidate_id,
                kind=str(kind),
                insertion_video=Path(insertion["path"]),
                insertion_video_sha256=insertion["sha256"],
                inserted_mask=Path(inserted["path"]),
                inserted_mask_sha256=inserted["sha256"],
                cleanup_mask=Path(cleanup["path"]),
                cleanup_mask_sha256=cleanup["sha256"],
                cleanup_ring_pixels=ring,
                fit=fit,
                pass2=pass2,
            )
        )
    return ChainPlan(
        mode=MODE_CHAIN,
        plate=Path(plate["path"]),
        plate_sha256=plate["sha256"],
        target_mask=Path(mask_path),
        target_mask_sha256=mask_digest,
        target_polarity=str(polarity),
        background_prompt=background_prompt,
        thresholds=thresholds,
        candidates=tuple(specs),
        require_full_chain=require_full_chain,
    )


def validate_manifest(payload: Any) -> dict[str, Any]:
    """Validate the legacy flat manifest and return it unchanged."""
    if not isinstance(payload, dict):
        raise BestOfNError("manifest must be a JSON object")
    source = payload.get("source_plate")
    mask = payload.get("mask_video")
    candidates = payload.get("candidates")
    if not isinstance(source, str) or not Path(source).is_file():
        raise BestOfNError("source_plate must name an existing file")
    if not isinstance(mask, str) or not Path(mask).is_file():
        raise BestOfNError("mask_video must name an existing file")
    if not isinstance(candidates, list) or not candidates:
        raise BestOfNError("candidates must be a nonempty list")
    background_prompt = payload.get("background_prompt")
    if not isinstance(background_prompt, str) or not background_prompt.strip():
        raise BestOfNError("background_prompt must be a nonempty string")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise BestOfNError("each candidate must be an object")
        candidate_id = candidate.get("id")
        artifact = candidate.get("artifact")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise BestOfNError(f"candidate id must be unique and nonempty: {candidate_id!r}")
        if SAFE_CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise BestOfNError(f"candidate id must be a safe path slug: {candidate_id!r}")
        seen.add(candidate_id)
        if not isinstance(artifact, str) or not Path(artifact).is_file():
            raise BestOfNError(f"candidate {candidate_id} artifact is absent")
    try:
        numeric_bounds = {
            "changed_threshold": (
                float(payload.get("changed_threshold", DEFAULT_CHANGED_THRESHOLD)),
                (0.0, None),
            ),
            "min_object_coverage": (
                float(payload.get("min_object_coverage", DEFAULT_MIN_COVERAGE)),
                (0.0, 1.0),
            ),
            "min_object_precision": (
                float(payload.get("min_object_precision", DEFAULT_MIN_PRECISION)),
                (0.0, 1.0),
            ),
            "max_background_change": (
                float(payload.get("max_background_change", DEFAULT_MAX_BACKGROUND_CHANGE)),
                (0.0, 1.0),
            ),
            "max_temporal_pop": (
                float(payload.get("max_temporal_pop", DEFAULT_MAX_TEMPORAL_POP)),
                (0.0, 1.0),
            ),
        }
        int(payload.get("cleanup_radius", 7))
    except (TypeError, ValueError, OverflowError) as error:
        raise BestOfNError(f"manifest numeric fields must be valid numbers: {error}") from error
    for name, (value, bounds) in numeric_bounds.items():
        lower, upper = bounds
        if not np.isfinite(value) or value <= lower or (upper is not None and value > upper):
            raise BestOfNError(f"{name} must be finite and inside {bounds}: {value}")
    cleanup_radius = int(payload.get("cleanup_radius", 7))
    if cleanup_radius < 0:
        raise BestOfNError(f"cleanup_radius must be nonnegative: {cleanup_radius}")
    return payload


def _legacy_plan(document: dict[str, Any]) -> ChainPlan:
    """Adapt the legacy flat manifest into the shared campaign plan."""
    validate_manifest(document)
    thresholds = ChainThresholds(
        changed_threshold=float(document.get("changed_threshold", DEFAULT_CHANGED_THRESHOLD)),
        min_object_coverage=float(document.get("min_object_coverage", DEFAULT_MIN_COVERAGE)),
        min_object_precision=float(document.get("min_object_precision", DEFAULT_MIN_PRECISION)),
        max_background_change=float(
            document.get("max_background_change", DEFAULT_MAX_BACKGROUND_CHANGE)
        ),
        max_temporal_pop=float(document.get("max_temporal_pop", DEFAULT_MAX_TEMPORAL_POP)),
    )
    specs = tuple(
        CandidateSpec(
            candidate_id=str(candidate["id"]),
            kind=GENERATED_REPLACEMENT,
            insertion_video=Path(str(candidate["artifact"])),
            insertion_video_sha256=None,
            inserted_mask=None,
            inserted_mask_sha256=None,
            cleanup_mask=None,
            cleanup_mask_sha256=None,
            cleanup_ring_pixels=None,
            fit=None,
        )
        for candidate in document["candidates"]
    )
    return ChainPlan(
        mode=MODE_LEGACY_ADAPTER,
        plate=Path(str(document["source_plate"])),
        plate_sha256=None,
        target_mask=Path(str(document["mask_video"])),
        target_mask_sha256=None,
        target_polarity=ROI_BLACK,
        background_prompt=str(document["background_prompt"]),
        thresholds=thresholds,
        candidates=specs,
        cleanup_radius=int(document.get("cleanup_radius", 7)),
    )


def plan_from_manifest(payload: Any) -> ChainPlan:
    """Normalize a legacy or chain manifest into the one internal plan."""
    document = _require_mapping(payload, "manifest")
    schema = document.get("schema")
    if schema is None:
        return _legacy_plan(document)
    if schema == CHAIN_MANIFEST_SCHEMA:
        return validate_chain_manifest(document)
    raise BestOfNError(
        f"unsupported manifest schema {schema!r}: expected {CHAIN_MANIFEST_SCHEMA} "
        "or a legacy manifest without a schema key"
    )


def chain_score(
    *,
    candidate_id: str,
    kind: str,
    plate: list[np.ndarray],
    candidate: list[np.ndarray],
    masks: list[np.ndarray],
    inserted: list[np.ndarray],
    thresholds: ChainThresholds,
    target_polarity: str,
    inserted_mask_source: str,
) -> ChainCandidateScore:
    """Score one chain candidate and bind its declared mask to its pixels.

    ``inserted`` holds one boolean region per frame, true where the inserted
    object is, exactly as the VOID inserted-mask convention reads it.
    """
    try:
        score = score_candidate(
            candidate_id=candidate_id,
            plate=plate,
            candidate=candidate,
            masks=masks,
            changed_threshold=thresholds.changed_threshold,
            min_coverage=thresholds.min_object_coverage,
            min_precision=thresholds.min_object_precision,
            max_background_change=thresholds.max_background_change,
            max_temporal_pop=thresholds.max_temporal_pop,
            target_polarity=target_polarity,
        )
    except BestOfNError as error:
        failed = failed_candidate_score(candidate_id, frames=len(plate), reason=str(error))
        return ChainCandidateScore(
            candidate_id=candidate_id,
            kind=kind,
            pre_pass2=failed,
            remask_agreement=0.0,
            insertion_target_overlap=0.0,
            remask_consistent=False,
            inserted_mask_source=inserted_mask_source,
            failure_reason=str(error),
        )
    changed = changed_frames(plate, candidate, changed_threshold=thresholds.changed_threshold)
    target = [target_region(frame, target_polarity) for frame in masks]
    agreement, overlap = remask_metrics(changed=changed, inserted=inserted, target=target)
    consistent = agreement >= thresholds.min_remask_agreement
    reason = ""
    if not consistent:
        reason = (
            f"remask_consistency failed: remask_agreement {agreement:.4f} "
            f"< {thresholds.min_remask_agreement:.4f}"
        )
    return ChainCandidateScore(
        candidate_id=candidate_id,
        kind=kind,
        pre_pass2=score,
        remask_agreement=agreement,
        insertion_target_overlap=overlap,
        remask_consistent=consistent,
        inserted_mask_source=inserted_mask_source,
        failure_reason=reason,
    )


def _metric_number(report: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = report
    label = ".".join(path)
    for index, component in enumerate(path):
        container_label = ".".join(path[:index])
        mapping = _require_mapping(value, container_label)
        if component not in mapping:
            raise BestOfNError(f"pass2 comparator {label}.{component} is required")
        value = mapping[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BestOfNError(f"pass2 comparator {label} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise BestOfNError(f"pass2 comparator {label} must be finite")
    return number


def _validate_comparator_inputs(
    report: dict[str, Any],
    *,
    pass2: Pass2Record,
    inserted_mask_sha256: str,
    cleanup_mask_sha256: str,
    candidate_result_sha256: str,
) -> None:
    expected = {
        "baseline": pass2.result.sha256,
        "pass2": pass2.retimed.sha256,
        "inserted_mask": inserted_mask_sha256,
        "cleanup_mask": cleanup_mask_sha256,
    }
    if pass2.result.sha256 != candidate_result_sha256:
        raise BestOfNError("pass2.result digest does not bind to the candidate insertion artifact")
    inputs = _require_mapping(report.get("inputs"), "pass2 comparator inputs")
    for name, digest in expected.items():
        record = _require_mapping(inputs.get(name), f"pass2 comparator inputs.{name}")
        measured = _required_sha256(record.get("sha256"), f"pass2 comparator inputs.{name}.sha256")
        if measured != digest:
            raise BestOfNError(
                f"pass2 comparator {name} digest mismatch: expected {digest}, got {measured}"
            )


def validate_pass2_comparison(
    report: Any,
    *,
    pass2: Pass2Record,
    inserted_mask_sha256: str,
    cleanup_mask_sha256: str,
    candidate_result_sha256: str,
    expected_changed_threshold: float,
    expected_fps_tolerance: float | None,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    """Validate a comparator report and return only its ranking-safe metrics."""
    document = _require_mapping(report, "pass2 comparator report")
    schema = document.get("schema")
    if schema != PASS2_COMPARATOR_REPORT_SCHEMA:
        raise BestOfNError(
            f"pass2 comparator schema must be {PASS2_COMPARATOR_REPORT_SCHEMA}: {schema!r}"
        )
    _require_mapping(document.get("thresholds"), "pass2 thresholds")
    changed_threshold = _metric_number(document, ("thresholds", "changed_threshold"))
    if changed_threshold != expected_changed_threshold:
        raise BestOfNError(
            "pass2 comparator changed_threshold does not match the campaign threshold"
        )
    cleanup_value = _metric_number(document, ("thresholds", "cleanup_value"))
    if cleanup_value != 127:
        raise BestOfNError("pass2 comparator cleanup_value must be AFFECTED (127)")
    fps_tolerance = _metric_number(document, ("thresholds", "fps_tolerance"))
    if expected_fps_tolerance is not None and fps_tolerance != expected_fps_tolerance:
        raise BestOfNError("pass2 comparator fps_tolerance differs between candidates")
    _validate_comparator_inputs(
        document,
        pass2=pass2,
        inserted_mask_sha256=inserted_mask_sha256,
        cleanup_mask_sha256=cleanup_mask_sha256,
        candidate_result_sha256=candidate_result_sha256,
    )
    alignment = _require_mapping(document.get("frame_alignment"), "pass2 frame_alignment")
    timestamps = _require_mapping(document.get("frame_timestamps"), "pass2 frame_timestamps")
    for record, label in (
        (alignment, "pass2 frame_alignment"),
        (timestamps, "pass2 frame_timestamps"),
    ):
        if record.get("aligned") is not True:
            raise BestOfNError(f"{label}.aligned must be true")
    if timestamps.get("pts_aligned") is not True:
        raise BestOfNError("pass2 frame_timestamps.pts_aligned must be true")
    if timestamps.get("frame_rate_character_aligned") is not True:
        raise BestOfNError("pass2 frame_timestamps.frame_rate_character_aligned must be true")

    alignment_records = {
        name: _require_mapping(alignment.get(name), f"pass2 frame_alignment.{name}")
        for name in ("baseline", "pass2", "inserted_mask", "cleanup_mask")
    }
    expected_geometry = (expected_width, expected_height, expected_frames)
    for name, record in alignment_records.items():
        geometry = (record.get("width"), record.get("height"), record.get("frames"))
        if geometry != expected_geometry:
            raise BestOfNError(
                f"pass2 comparator {name} geometry {geometry} does not match the campaign "
                f"geometry {expected_geometry}"
            )
        raw_fps = record.get("fps")
        if isinstance(raw_fps, bool) or not isinstance(raw_fps, (int, float)):
            raise BestOfNError(f"pass2 comparator {name} fps must be numeric")
        record_fps = float(raw_fps)
        if not np.isfinite(record_fps):
            raise BestOfNError(f"pass2 comparator {name} fps must be finite")
        if not np.isclose(record_fps, expected_fps, rtol=0.0, atol=1e-9):
            raise BestOfNError(f"pass2 comparator {name} fps does not match the campaign cadence")

    timestamp_records = {
        name: _require_mapping(timestamps.get(name), f"pass2 frame_timestamps.{name}")
        for name in ("baseline", "pass2", "inserted_mask", "cleanup_mask")
    }
    timestamp_digests = {record.get("pts_digest") for record in timestamp_records.values()}
    timestamp_frames = {record.get("frames") for record in timestamp_records.values()}
    timestamp_characters = {
        record.get("variable_frame_rate") for record in timestamp_records.values()
    }
    if len(timestamp_digests) != 1 or None in timestamp_digests:
        raise BestOfNError("pass2 comparator inputs do not share one exact PTS digest")
    if timestamp_frames != {expected_frames}:
        raise BestOfNError("pass2 comparator timestamp frame counts do not match the campaign")
    if len(timestamp_characters) != 1 or None in timestamp_characters:
        raise BestOfNError("pass2 comparator inputs disagree on frame-rate character")

    embedded_baseline = alignment.get("baseline")
    embedded_pass2 = alignment.get("pass2")
    for embedded, expected, label in (
        (embedded_baseline, pass2.result, "baseline"),
        (embedded_pass2, pass2.retimed, "pass2"),
    ):
        record = _require_mapping(embedded, f"pass2 frame_alignment.{label}")
        embedded_geometry = (
            record.get("width"),
            record.get("height"),
            record.get("frames"),
        )
        if embedded_geometry != (expected.width, expected.height, expected.frames):
            raise BestOfNError(
                f"pass2 comparator {label} geometry does not match its declared artifact"
            )

    changed = {
        name: _metric_number(document, (name, "changed_ratio"))
        for name in ("object_preservation", "cleanup_ring", "background")
    }
    temporal_pop = _metric_number(document, ("temporal", "temporal_pop_score"))
    for name, value in {**changed, "temporal": temporal_pop}.items():
        if not 0.0 <= value <= 1.0:
            raise BestOfNError(f"pass2 comparator {name} ratio must lie in [0, 1]")
    for name in ("object_preservation", "cleanup_ring", "background"):
        for field in ("mae", "mse", "max_abs_difference"):
            if _metric_number(document, (name, field)) < 0.0:
                raise BestOfNError(f"pass2 comparator {name}.{field} must be nonnegative")
    return {
        "object_preservation_changed_ratio": changed["object_preservation"],
        "cleanup_ring_changed_ratio": changed["cleanup_ring"],
        "background_changed_ratio": changed["background"],
        "temporal_pop_score": temporal_pop,
    }


def full_chain_pass2_record(
    *,
    pre_pass2: CandidateScore,
    pass2: Pass2Record,
    comparator_report: dict[str, Any],
    inserted_mask_sha256: str,
    cleanup_mask_sha256: str,
    candidate_result_sha256: str,
    expected_changed_threshold: float,
    expected_fps_tolerance: float | None,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    """Bind one comparator report to its artifacts and compute its chain objective."""
    metrics = validate_pass2_comparison(
        comparator_report,
        pass2=pass2,
        inserted_mask_sha256=inserted_mask_sha256,
        cleanup_mask_sha256=cleanup_mask_sha256,
        candidate_result_sha256=candidate_result_sha256,
        expected_changed_threshold=expected_changed_threshold,
        expected_fps_tolerance=expected_fps_tolerance,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_frames=expected_frames,
        expected_fps=expected_fps,
    )
    objective = (
        0.35 * pre_pass2.objective
        + 0.25 * (1.0 - metrics["object_preservation_changed_ratio"])
        + 0.20 * (1.0 - metrics["background_changed_ratio"])
        + 0.20 * (1.0 - metrics["temporal_pop_score"])
    )
    return {
        "native": pass2.native.record(),
        "retimed": pass2.retimed.record(),
        "result": pass2.result.record(),
        "comparator": {
            **pass2.comparator.record(),
            "schema": PASS2_COMPARATOR_REPORT_SCHEMA,
        },
        "metrics": metrics,
        "objective": objective,
    }


def rank_pre_pass2(scores: list[ChainCandidateScore]) -> list[dict[str, Any]]:
    """Order the pre-Pass2 survivors by objective, then by candidate id."""
    survivors = [item for item in scores if item.survivor]
    ordered = sorted(survivors, key=lambda item: (-item.pre_pass2.objective, item.candidate_id))
    return [
        {"candidate_id": item.candidate_id, "objective": item.pre_pass2.objective}
        for item in ordered
    ]


def rank_full_chain(
    scores: list[ChainCandidateScore], pass2_records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Order complete chains by the Pass2-aware objective, then candidate id."""
    complete = [item for item in scores if item.survivor and item.candidate_id in pass2_records]
    ordered = sorted(
        complete,
        key=lambda item: (
            -float(pass2_records[item.candidate_id]["objective"]),
            item.candidate_id,
        ),
    )
    return [
        {
            "candidate_id": item.candidate_id,
            "objective": pass2_records[item.candidate_id]["objective"],
        }
        for item in ordered
    ]


CHAIN_NON_CLAIMS = [
    "This selector measures change placement, background preservation, temporal "
    "stability, and mask binding; it does not prove object identity or artistic quality.",
    "Pass 2 records prove only that externally produced artifacts are digest-bound and "
    "geometry-aligned; they do not re-run or independently prove VOID execution.",
    "The full-chain objective is a deterministic preservation/stability score. Cleanup-ring "
    "change is reported but not treated as monotonic artistic quality.",
    "The remask_consistency gate binds a declared mask to the pixels of its own "
    "candidate; it does not prove the mask is the semantically correct region.",
    "A staged Pass 2 sequence is a prepared input, not a produced result.",
]


def build_chain_report(
    *,
    plan: ChainPlan,
    plate: dict[str, Any],
    target_mask: dict[str, Any],
    scores: list[ChainCandidateScore],
    artifacts: dict[str, dict[str, Any]],
    contact_sheets: dict[str, str],
    sequences: dict[str, str],
    pass2_records: dict[str, dict[str, Any]],
    revision: str,
    dirty: bool,
    recorded_at: str,
) -> dict[str, Any]:
    """Assemble the chain report from the measured candidate records."""
    ranking = rank_pre_pass2(scores)
    full_chain_ranking = rank_full_chain(scores, pass2_records) if plan.require_full_chain else []
    survivors = [entry["candidate_id"] for entry in ranking]
    candidates: list[dict[str, Any]] = []
    for item in scores:
        candidates.append(
            {
                "candidate_id": item.candidate_id,
                "kind": item.kind,
                "stage": "pre_pass2" if item.survivor else "eliminated",
                "pre_pass2": item.pre_pass2_record(),
                "pass2": pass2_records.get(item.candidate_id),
                "artifacts": artifacts.get(item.candidate_id, {}),
                "contact_sheet": contact_sheets.get(item.candidate_id),
                "pass2_sequence": sequences.get(item.candidate_id),
                "failure_reason": item.failure_reason,
            }
        )
    selected: dict[str, Any] | None = None
    selected_ranking = full_chain_ranking if plan.require_full_chain else ranking
    if selected_ranking:
        selected_id = str(selected_ranking[0]["candidate_id"])
        selected_artifacts = artifacts.get(selected_id, {})
        selected_pass2 = pass2_records.get(selected_id, {})
        selected = {
            "candidate_id": selected_id,
            "basis": "full_chain" if plan.require_full_chain else "pre_pass2",
            "artifacts": {
                "insertion_video": selected_artifacts.get("insertion_video", {}).get("path"),
                "inserted_mask": selected_artifacts.get("inserted_mask", {}).get("path"),
                "cleanup_mask": selected_artifacts.get("cleanup_mask", {}).get("path"),
                "pass2_native": selected_pass2.get("native", {}).get("path"),
                "pass2_retimed": selected_pass2.get("retimed", {}).get("path"),
                "pass2_result": selected_pass2.get("result", {}).get("path"),
                "pass2_comparator": selected_pass2.get("comparator", {}).get("path"),
            },
            "pass2_sequence": sequences.get(selected_id),
        }
    return {
        "schema": CHAIN_REPORT_SCHEMA,
        "recorded_at": recorded_at,
        "source_revision": revision,
        "source_dirty": dirty,
        "mode": plan.mode,
        "plate": plate,
        "target_mask": target_mask,
        "background_prompt": plan.background_prompt,
        "require_full_chain": plan.require_full_chain,
        "thresholds": plan.thresholds.as_record(),
        "candidates": candidates,
        "survivors": survivors,
        "ranking_pre_pass2": ranking,
        "ranking_full_chain": full_chain_ranking,
        "selected": selected,
        "non_claims": list(CHAIN_NON_CLAIMS),
    }
