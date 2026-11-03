"""Weighted spatial expert fusion for mask-coordinate tensors."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .expert_persist import (
    DEFAULT_ARTIFACT_VERSION,
    MANIFEST_NAME,
    ExpertPersistError,
    manifest_lock,
    persist_expert_tensors,
    verify_manifest,
)
from .media import GrayVideoWriter, decode_first_gray_frame, extract_color_frames, probe_video


class EnsembleError(RuntimeError):
    """Expert configuration, output, or tensor alignment is invalid."""


CommandRunner = Callable[[list[str], dict[str, Any]], dict[str, Any]]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    weight: float
    temperature: float
    kind: str = "tensor"
    path: Path | None = None
    representation: str = "logits"
    command: tuple[str, ...] = ()
    prompt: str | None = None
    normalized_coordinates: bool = True
    boundary_sigma: float = 0.0
    anchor_source_path: Path | None = None

    def validate(self) -> None:
        if not self.name:
            raise EnsembleError("expert name is required")
        if self.weight < 0:
            raise EnsembleError(f"expert {self.name}: weight must be nonnegative")
        if self.temperature <= 0:
            raise EnsembleError(f"expert {self.name}: temperature must be positive")
        if self.kind not in {"tensor", "command"}:
            raise EnsembleError(f"expert {self.name}: unknown kind {self.kind}")
        if self.kind == "tensor":
            if self.path is None:
                raise EnsembleError(f"expert {self.name}: tensor path is required")
            if self.representation not in {"logits", "probabilities"}:
                raise EnsembleError(f"expert {self.name}: invalid representation")
        if self.kind == "command":
            if not self.command:
                raise EnsembleError(f"expert {self.name}: command is required")
            if self.boundary_sigma < 0:
                raise EnsembleError(f"expert {self.name}: boundary_sigma is negative")


@dataclass(frozen=True)
class FusionSpec:
    mode: str = "top_k"
    threshold: float = 0.5
    top_k_ratio: float = 0.02
    min_expert_support: int = 1
    support_threshold: float = 0.5
    dynamic_weights: str = "entropy"
    batch_size: int = 8
    max_batch_pixels: int = 32_000_000

    def validate(self) -> None:
        if self.mode not in {"threshold", "top_k"}:
            raise EnsembleError(f"unknown fusion mode: {self.mode}")
        if not 0 < self.threshold < 1:
            raise EnsembleError("threshold must be between zero and one")
        if self.top_k_ratio <= 0:
            raise EnsembleError("top_k_ratio must be positive")
        if self.min_expert_support < 1:
            raise EnsembleError("min_expert_support must be at least one")
        if not 0 < self.support_threshold < 1:
            raise EnsembleError("support_threshold must be between zero and one")
        if self.dynamic_weights not in {"static", "entropy"}:
            raise EnsembleError(f"unknown dynamic_weights mode: {self.dynamic_weights}")
        if self.batch_size < 1 or self.max_batch_pixels < 1:
            raise EnsembleError("batch size and pixel budget must be positive")


EXPERT_PERSIST_MODES = frozenset({"off", "opt-in", "required"})

# D4 storage dtype: float16, two bytes per element per expert. It is
# the dtype the tensor-persistence spec fixes for stored expert tensors
# and half the size of the float64 buffer it replaces.
PERSISTED_DTYPE = np.float16


def _to_persisted_dtype(values: np.ndarray, expert_name: str) -> np.ndarray:
    """Cast consumed expert values to the persisted float16 storage dtype.

    A value outside the float16 range would be stored as infinity, so
    the cast is checked: the run refuses instead of persisting a
    corrupted tensor.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        cast = np.asarray(values, dtype=PERSISTED_DTYPE)
    if not np.isfinite(cast).all():
        raise EnsembleError(
            f"expert {expert_name!r}: values exceed the float16 storage range; "
            "the run refuses to persist them"
        )
    return cast


@dataclass(frozen=True)
class EnsembleConfig:
    input_video: Path
    experts: tuple[ExpertSpec, ...]
    fusion: FusionSpec
    calibration: dict[str, Any] = field(default_factory=dict)
    # D4 (gate D4): when set, every expert tensor the fusion
    # consumes -- tensor experts and Muse command rasterizations
    # alike -- is persisted as a versioned .npy artifact with a
    # SHA-256 manifest in this directory. The artifacts are stored in
    # float16 (PERSISTED_DTYPE). None keeps the
    # in-memory-only behaviour, which the decision rejects for a
    # rerunnable fusion/calibration run: governed run configs
    # (see configs/) set this field, and run_ensemble states the
    # persistence choice in every run report either way.
    expert_persist_dir: Path | None = None
    # D4 feature flag (gate D4; ruling dated 2026-10-28): one value
    # selects how strongly a run insists on the persistence above.
    # "off" never persists and allocates no expert buffers. "opt-in"
    # (the default) keeps the behaviour above: persistence runs only
    # when expert_persist_dir is set. "required" makes the governed
    # path always persist: run_ensemble refuses, fail-closed, before
    # any frame work when no target is set or the target cannot be
    # written. The flag changes persistence only, never fusion.
    expert_persist_mode: str = "opt-in"
    expert_artifact_version: str = DEFAULT_ARTIFACT_VERSION
    # D1 (gate D1): the declared weight source for this run. None
    # keeps the legacy behaviour: the config's per-expert weight
    # values apply and the run is not governed. A governed run
    # declares a calibration_record source; resolve_weight_source
    # validates it at run start and the record weights replace the
    # per-expert weight values. A hand_baseline source declares the
    # 0.40/0.35/0.25 comparison map and is never governed.
    weight_source: dict[str, Any] | None = None


def _parse_weight_source(raw: Any) -> dict[str, Any] | None:
    """Normalize the config's declared weight source (gate D1).

    Shape validation only: the calibration record itself is loaded
    and checked by resolve_weight_source at run start. Returns None
    when the config declares no source.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise EnsembleError("weight_source must be an object")
    kind = raw.get("type")
    if kind == "calibration_record":
        record_path = raw.get("path")
        sha256 = raw.get("sha256")
        if not isinstance(record_path, str) or not record_path:
            raise EnsembleError("weight_source calibration_record requires a path")
        if not isinstance(sha256, str) or not sha256:
            raise EnsembleError("weight_source calibration_record requires a sha256")
        return {"type": kind, "path": record_path, "sha256": sha256.lower()}
    if kind == "hand_baseline":
        weights = raw.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise EnsembleError("weight_source hand_baseline requires a weights map")
        parsed: dict[str, float] = {}
        for name, value in weights.items():
            try:
                parsed[str(name)] = float(value)
            except (TypeError, ValueError) as error:
                raise EnsembleError(
                    f"weight_source hand_baseline weight for {name!r} is not a number"
                ) from error
        return {"type": kind, "weights": parsed}
    raise EnsembleError(f"unknown weight_source type: {kind!r}")


def load_config(path: Path) -> EnsembleConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EnsembleError(f"cannot load ensemble config {path}: {error}") from error
    required = {"input_video", "experts", "fusion"}
    missing = sorted(required - raw.keys())
    if missing:
        raise EnsembleError(f"missing ensemble fields: {', '.join(missing)}")
    raw_experts = raw["experts"]
    if not isinstance(raw_experts, list) or len(raw_experts) < 2:
        raise EnsembleError("at least two experts are required for triangulation")
    experts: list[ExpertSpec] = []
    for item in raw_experts:
        kind = item.get("kind", "tensor")
        anchor_raw = item.get("anchor_source")
        if anchor_raw is None:
            anchor_source_path = None
        elif not isinstance(anchor_raw, str) or not anchor_raw:
            raise EnsembleError(
                f"expert {item.get('name')}: anchor_source must be a non-empty string"
            )
        else:
            anchor_expanded = Path(anchor_raw).expanduser()
            anchor_source_path = (
                anchor_expanded if anchor_expanded.is_absolute() else path.parent / anchor_expanded
            )
        expert = ExpertSpec(
            name=item["name"],
            weight=float(item.get("weight", 1.0)),
            temperature=float(item.get("temperature", 1.0)),
            kind=kind,
            path=Path(item["path"]).expanduser() if item.get("path") else None,
            representation=(
                "probabilities" if kind == "command" else item.get("representation", "logits")
            ),
            command=tuple(item.get("command", [])),
            prompt=item.get("prompt"),
            normalized_coordinates=bool(item.get("normalized_coordinates", True)),
            boundary_sigma=float(item.get("boundary_sigma", 0.0)),
            anchor_source_path=anchor_source_path,
        )
        expert.validate()
        experts.append(expert)
    if len({expert.name for expert in experts}) != len(experts):
        raise EnsembleError("expert names must be unique")
    if sum(expert.weight for expert in experts) <= 0:
        raise EnsembleError("total expert weight must be positive")
    raw_fusion = raw["fusion"]
    calibration = raw.get("calibration", {})
    if not isinstance(calibration, dict):
        raise EnsembleError("ensemble calibration must be an object")
    weight_source = _parse_weight_source(raw.get("weight_source"))
    fusion = FusionSpec(
        mode=raw_fusion.get("mode", "top_k"),
        threshold=float(raw_fusion.get("threshold", 0.5)),
        top_k_ratio=float(raw_fusion.get("top_k_ratio", 0.02)),
        min_expert_support=int(raw_fusion.get("min_expert_support", 1)),
        support_threshold=float(raw_fusion.get("support_threshold", 0.5)),
        dynamic_weights=raw_fusion.get("dynamic_weights", "entropy"),
        batch_size=int(raw_fusion.get("batch_size", 8)),
        max_batch_pixels=int(raw_fusion.get("max_batch_pixels", 32_000_000)),
    )
    fusion.validate()
    persist_mode = str(raw.get("expert_persist_mode", "opt-in"))
    if persist_mode not in EXPERT_PERSIST_MODES:
        raise EnsembleError(
            f"unknown expert_persist_mode: {persist_mode!r} "
            f"(expected one of {sorted(EXPERT_PERSIST_MODES)})"
        )
    if fusion.min_expert_support > len(experts):
        raise EnsembleError("min_expert_support exceeds expert count")
    return EnsembleConfig(
        input_video=Path(raw["input_video"]).expanduser(),
        experts=tuple(experts),
        fusion=fusion,
        calibration=calibration,
        weight_source=weight_source,
        expert_persist_dir=(
            Path(raw["expert_persist_dir"]).expanduser() if raw.get("expert_persist_dir") else None
        ),
        expert_persist_mode=persist_mode,
        expert_artifact_version=str(raw.get("expert_artifact_version", DEFAULT_ARTIFACT_VERSION)),
    )


# D1 (gate D1): measured D3 weights govern a run only through a
# declared, digest-pinned calibration record. The record carries no
# digest of itself, so the run config pins the SHA-256 of the record
# file bytes and the loader recomputes it. Every check fails closed:
# a refused source raises EnsembleError naming the failed check, the
# record path, and the declared digest, and the run writes no mask
# and no report. There is never a fallback to the hand weights or to
# config weight values: a failed gate is a refusal, not a fallback.
# The governed label follows the source alone (plan question 1,
# default): a run whose calibration record passes every check is
# governed; a hand-baseline run never is.
# Two of those checks bind the record to the bytes the run actually
# consumes: the D4 manifest digest (check 6) and, per expert, the
# calibrated expert name with the exact artifact file, version, and
# digest (check 7). A command expert is refused by check 6: the
# values it consumes are produced during the run, so no pre-run
# artifact binds them.
SUPPORTED_CALIBRATION_RECORD_VERSIONS = frozenset({1})
_WEIGHT_SUM_TOLERANCE = 1e-6
HAND_BASELINE_NON_CLAIM = "comparison baseline only, not the governed rule"


@dataclass(frozen=True)
class ResolvedWeightSource:
    """The weight source a run actually applies, after all checks."""

    kind: str
    governed: bool
    weights: dict[str, float]
    report: dict[str, Any]


def _check_weight_map(
    weights: dict[str, Any], expert_names: set[str], context: str
) -> dict[str, float]:
    """Weight-source checks 4-5: exact expert set, valid distribution."""
    record_names = set(weights)
    if record_names != expert_names:
        missing = sorted(expert_names - record_names)
        extra = sorted(record_names - expert_names)
        raise EnsembleError(
            "weight source refused at check expert_set: weight experts do not "
            f"equal the run experts (missing: {missing}, extra: {extra}) {context}"
        )
    parsed: dict[str, float] = {}
    for name, value in weights.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise EnsembleError(
                "weight source refused at check weight_distribution: weight for "
                f"expert {name!r} is not a number {context}"
            ) from error
        if not math.isfinite(number) or number < 0:
            raise EnsembleError(
                "weight source refused at check weight_distribution: weight for "
                f"expert {name!r} must be finite and nonnegative, got {number} {context}"
            )
        parsed[str(name)] = number
    total = sum(parsed.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise EnsembleError(
            "weight source refused at check weight_distribution: weights sum to "
            f"{total}, not 1 within {_WEIGHT_SUM_TOLERANCE} {context}"
        )
    return parsed


def _file_sha256(path: Path) -> str:
    """SHA-256 of one file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_consumed_manifest(
    config: EnsembleConfig, context: str
) -> tuple[str, dict[str, dict[str, str]]]:
    """Verify the D4 manifest and every consumed tensor.

    A governed run refuses to consume a tensor set it cannot prove.
    The D4 manifest beside each tensor must exist and must verify
    (every artifact it lists is present with a matching SHA-256, shape,
    and dtype), and the manifest must list each consumed tensor with a
    digest that matches the file the run reads. The manifest record
    for a consumed file must name the same expert the run config
    consumes it for, so one expert cannot be pointed at another
    expert's artifact. All tensor experts must name the same manifest.

    A command expert consumes values the run itself produces, so no
    pre-run D4 artifact binds them: a governed run refuses a panel
    that contains one instead of skipping the expert.

    Returns the manifest digest, which binds the run to the
    calibration record's inputs, and the per-expert artifact bindings
    (file name, artifact version, artifact digest) the record must
    match.
    """
    directories: dict[Path, list[tuple[ExpertSpec, Path]]] = {}
    for spec in config.experts:
        if spec.kind != "tensor" or spec.path is None:
            continue
        directories.setdefault(spec.path.parent, []).append((spec, spec.path))
    if not directories:
        raise EnsembleError(
            "weight source refused at check manifest: the run consumes no "
            f"persisted tensor set to cross-check {context}"
        )
    command_experts = sorted(spec.name for spec in config.experts if spec.kind == "command")
    if command_experts:
        raise EnsembleError(
            "weight source refused at check command_expert_binding: expert(s) "
            f"{', '.join(command_experts)} are command experts; their consumed "
            "values have no pre-run D4 artifact binding, so a governed run "
            "cannot prove which bytes it fuses "
            f"{context}"
        )
    digests: set[str] = set()
    bindings: dict[str, dict[str, str]] = {}
    for directory, specs in directories.items():
        first_expert = specs[0][0]
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise EnsembleError(
                "weight source refused at check manifest: expert "
                f"{first_expert.name!r} has no D4 manifest beside its tensor "
                f"({manifest_path}) {context}"
            )
        try:
            manifest = verify_manifest(directory)
        except ExpertPersistError as error:
            raise EnsembleError(
                "weight source refused at check manifest: the D4 manifest "
                f"{manifest_path} does not verify: {error} {context}"
            ) from error
        records = {str(item.get("file")): item for item in manifest["artifacts"]}
        for spec, tensor_path in specs:
            record = records.get(tensor_path.name)
            if record is None:
                raise EnsembleError(
                    "weight source refused at check manifest: expert "
                    f"{spec.name!r} tensor {tensor_path} is not listed in the "
                    f"D4 manifest {manifest_path} {context}"
                )
            manifest_expert = str(record.get("expert"))
            if manifest_expert != spec.name:
                raise EnsembleError(
                    "weight source refused at check manifest: expert "
                    f"{spec.name!r} consumes artifact {tensor_path.name}, which "
                    f"the D4 manifest {manifest_path} records as expert "
                    f"{manifest_expert!r} {context}"
                )
            actual = _file_sha256(tensor_path)
            if actual != record.get("sha256"):
                raise EnsembleError(
                    "weight source refused at check manifest: expert "
                    f"{spec.name!r} tensor {tensor_path} hashes to {actual}, not "
                    f"the manifest digest {record.get('sha256')} {context}"
                )
            bindings[spec.name] = {
                "file": tensor_path.name,
                "version": str(record.get("version")),
                "sha256": actual,
            }
        digests.add(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    if len(digests) != 1:
        raise EnsembleError(
            "weight source refused at check manifest: the run's tensor experts "
            f"name different D4 manifests {sorted(digests)} {context}"
        )
    return digests.pop(), bindings


def _verify_record_artifact_bindings(
    record: dict[str, Any],
    bindings: dict[str, dict[str, str]],
    context: str,
) -> None:
    """Check 7: bind every run expert to the artifact it was calibrated on.

    The record must name, for each run expert, the calibrated expert
    name and the exact D4 artifact the calibration consumed: the
    artifact file, the artifact version, and the artifact digest. A
    record that pins only the manifest digest cannot prove which
    bytes the weights were fitted on, so it is refused rather than
    trusted. The record may not calibrate an expert the run does not
    consume either: the two panels must be the same panel.
    """
    inputs = record.get("inputs")
    artifacts = inputs.get("artifacts") if isinstance(inputs, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise EnsembleError(
            "weight source refused at check artifact_binding: the record lists no "
            "D4 artifact bindings, so the run cannot bind its experts to the "
            f"artifacts the calibration consumed {context}"
        )
    record_version = inputs.get("artifact_version") if isinstance(inputs, dict) else None
    if record_version is None:
        raise EnsembleError(
            "weight source refused at check artifact_binding: the record names no "
            "D4 artifact version, so the run cannot bind its experts to the "
            f"artifact version the calibration consumed {context}"
        )
    by_expert: dict[str, list[dict[str, Any]]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not item.get("expert"):
            raise EnsembleError(
                "weight source refused at check artifact_binding: a record artifact "
                f"entry names no calibrated expert {context}"
            )
        by_expert.setdefault(str(item["expert"]), []).append(item)
    for name, binding in sorted(bindings.items()):
        entries = by_expert.get(name, [])
        if len(entries) != 1:
            raise EnsembleError(
                "weight source refused at check artifact_binding: run expert "
                f"{name!r} has {len(entries)} calibrated artifact entries; exactly "
                f"one is required {context}"
            )
        entry = entries[0]
        file_name = str(entry.get("file"))
        if file_name != binding["file"]:
            raise EnsembleError(
                "weight source refused at check artifact_binding: run expert "
                f"{name!r} consumes artifact {binding['file']!r}, but the record "
                f"calibrated {file_name!r} {context}"
            )
        version = str(entry.get("version"))
        if version != binding["version"]:
            raise EnsembleError(
                "weight source refused at check artifact_binding: run expert "
                f"{name!r} consumes artifact {binding['file']!r} at version "
                f"{binding['version']!r}, but the record calibrated version "
                f"{version!r} {context}"
            )
        digest = str(entry.get("sha256"))
        if digest != binding["sha256"]:
            raise EnsembleError(
                "weight source refused at check artifact_binding: run expert "
                f"{name!r} consumes artifact {binding['file']!r} with digest "
                f"{binding['sha256']}, but the record calibrated digest {digest} "
                f"{context}"
            )
        if str(record_version) != version:
            raise EnsembleError(
                "weight source refused at check artifact_binding: the record's "
                f"artifact version {str(record_version)!r} does not match expert "
                f"{name!r}'s calibrated version {version!r} {context}"
            )
    extra = sorted(set(by_expert) - set(bindings))
    if extra:
        raise EnsembleError(
            "weight source refused at check artifact_binding: the record calibrates "
            f"expert(s) {', '.join(extra)} that this run does not consume {context}"
        )


def resolve_weight_source(config: EnsembleConfig) -> ResolvedWeightSource | None:
    """Resolve the config's declared weight source (gate D1).

    Returns None when the config declares no source: the config
    weight values apply and the run is not governed. A hand_baseline
    source applies its declared map and is never governed. A
    calibration_record source passes eight fail-closed checks, in
    order: (1) the record file exists; (2) its bytes hash to the
    pinned digest; (3) it parses as a D3 record with a supported
    record version and status recorded; (4) its weight experts equal
    the run experts exactly; (5) its weights are finite, nonnegative,
    and sum to 1; (6) its inputs.manifest_sha256 equals the D4
    manifest digest of the tensors this run consumes, and no expert in
    the panel is a command expert (a command expert's consumed values
    have no pre-run artifact binding); (7) the record binds every run
    expert to the calibrated expert name and the exact D4 artifact
    file, version, and digest it was fitted on; (8) its recorded
    representations and temperatures equal the run config's for each
    expert. Any refusal raises EnsembleError before any output is
    written.
    """
    declared = config.weight_source
    if declared is None:
        return None
    expert_names = {spec.name for spec in config.experts}
    if declared["type"] == "hand_baseline":
        weights = _check_weight_map(declared["weights"], expert_names, "(source: hand_baseline)")
        report = {
            "type": "hand_baseline",
            "weights": weights,
            "governed": False,
            "non_claim": HAND_BASELINE_NON_CLAIM,
        }
        return ResolvedWeightSource(
            kind="hand_baseline", governed=False, weights=weights, report=report
        )
    if declared["type"] != "calibration_record":
        raise EnsembleError(f"unknown weight_source type: {declared['type']!r}")
    record_path = Path(str(declared["path"])).expanduser()
    declared_sha256 = str(declared["sha256"])
    context = f"(record path: {record_path}, declared sha256: {declared_sha256})"
    # Check 1: the record file exists.
    if not record_path.is_file():
        raise EnsembleError(f"weight source refused at check record_exists: no such file {context}")
    payload = record_path.read_bytes()
    # Check 2: the file bytes hash to the pinned digest.
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != declared_sha256:
        raise EnsembleError(
            "weight source refused at check record_digest: the record file "
            f"hashes to {actual_sha256} {context}"
        )
    # Check 3: the record parses as a supported, successful D3 record.
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as error:
        raise EnsembleError(
            f"weight source refused at check record_shape: invalid JSON: {error} {context}"
        ) from error
    if not isinstance(record, dict) or record.get("gate") != "D3":
        raise EnsembleError(
            f"weight source refused at check record_shape: not a D3 record {context}"
        )
    if record.get("record_version") not in SUPPORTED_CALIBRATION_RECORD_VERSIONS:
        raise EnsembleError(
            "weight source refused at check record_shape: unsupported "
            f"record_version {record.get('record_version')!r} {context}"
        )
    if record.get("status") != "recorded":
        raise EnsembleError(
            "weight source refused at check record_shape: status is "
            f"{record.get('status')!r}, not 'recorded' {context}"
        )
    record_weights = record.get("weights")
    if not isinstance(record_weights, dict):
        raise EnsembleError(
            f"weight source refused at check record_shape: weights is not a map {context}"
        )
    # Checks 4-5: exact expert set and a valid weight distribution.
    weights = _check_weight_map(record_weights, expert_names, context)
    # Check 6: the record was fitted on the tensors this run consumes.
    # The D4 manifest and every consumed tensor are verified first, so
    # the binding is proved against the bytes on disk.
    manifest_sha256, bindings = _verify_consumed_manifest(config, context)
    inputs = record.get("inputs")
    recorded_manifest = inputs.get("manifest_sha256") if isinstance(inputs, dict) else None
    if recorded_manifest != manifest_sha256:
        raise EnsembleError(
            "weight source refused at check manifest: record inputs.manifest_sha256 "
            f"{recorded_manifest!r} != the run tensors' D4 manifest "
            f"{manifest_sha256} {context}"
        )
    # Check 7: every run expert binds to the artifact the record
    # calibrated, by calibrated expert name and exact file, version,
    # and digest.
    _verify_record_artifact_bindings(record, bindings, context)
    # Check 8: the record's transform equals the run config's transform.
    configuration = record.get("configuration")
    if not isinstance(configuration, dict):
        configuration = {}
    representations = configuration.get("representations")
    if not isinstance(representations, dict):
        representations = {}
    temperatures = configuration.get("temperatures")
    if not isinstance(temperatures, dict):
        temperatures = {}
    for spec in config.experts:
        if representations.get(spec.name) != spec.representation:
            raise EnsembleError(
                "weight source refused at check transform: expert "
                f"{spec.name!r} representation {representations.get(spec.name)!r} "
                f"!= run config {spec.representation!r} {context}"
            )
        recorded_temperature = temperatures.get(spec.name)
        if recorded_temperature is None or float(recorded_temperature) != spec.temperature:
            raise EnsembleError(
                "weight source refused at check transform: expert "
                f"{spec.name!r} temperature {recorded_temperature!r} "
                f"!= run config {spec.temperature} {context}"
            )
    report = {
        "type": "calibration_record",
        "path": str(record_path),
        "sha256": declared_sha256,
        "manifest_sha256": manifest_sha256,
        "artifact_bindings": {name: dict(binding) for name, binding in sorted(bindings.items())},
        "weights": weights,
        "governed": True,
    }
    return ResolvedWeightSource(
        kind="calibration_record", governed=True, weights=weights, report=report
    )


class TensorExpertSource:
    def __init__(self, spec: ExpertSpec):
        spec.validate()
        if spec.path is None:
            raise EnsembleError(f"expert {spec.name}: tensor path is required")
        self.spec = spec
        self._values = np.load(spec.path, mmap_mode="r")
        if self._values.ndim != 3:
            raise EnsembleError(f"expert {spec.name}: tensor must have frames,height,width shape")

    @property
    def shape(self) -> tuple[int, int, int]:
        frames, height, width = self._values.shape
        return int(frames), int(height), int(width)

    def read(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray(self._values[list(indices)], dtype=np.float64)


class CommandExpertSource:
    def __init__(
        self,
        spec: ExpertSpec,
        frame_paths: Sequence[Path],
        shape: tuple[int, int, int],
        *,
        runner: CommandRunner | None = None,
    ):
        spec.validate()
        if len(frame_paths) != shape[0]:
            raise EnsembleError(f"expert {spec.name}: frame count does not match video")
        self.spec = spec
        self.frame_paths = tuple(frame_paths)
        self._shape = shape
        self._runner = runner or self._run_command

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @staticmethod
    def _run_command(command: list[str], request: dict[str, Any]) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise EnsembleError(
                f"expert command exited {completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise EnsembleError("expert command returned invalid JSON") from error

    def _coordinate_number(self, value: Any, region: dict[str, Any]) -> float:
        """Return one coordinate as a finite float, or raise.

        JSON numbers arrive as int/float. Anything else -- a numeric
        string, a bool, None, a nested list -- is a malformed box:
        float() would coerce some of those silently, so the type is
        checked before any conversion. The offending region is named
        in the error so the bad box is in the run record, never
        silently discarded.
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EnsembleError(
                f"expert {self.spec.name}: region coordinate must be a number, "
                f"got {value!r} in region {region!r}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise EnsembleError(
                f"expert {self.spec.name}: region coordinate must be finite, "
                f"got {value!r} in region {region!r}"
            )
        return number

    def _rasterize(
        self,
        regions: Sequence[dict[str, Any]],
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        normalized = self.spec.normalized_coordinates
        coordinate_limit = 1.0 if normalized else None
        for region in regions:
            if not isinstance(region, dict):
                raise EnsembleError(
                    f"expert {self.spec.name}: region must be an object, got {region!r}"
                )
            if "confidence" not in region:
                raise EnsembleError(
                    f"expert {self.spec.name}: region is missing confidence: {region!r}"
                )
            raw_confidence = region["confidence"]
            if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, int | float):
                raise EnsembleError(
                    f"expert {self.spec.name}: region confidence must be a number, "
                    f"got {raw_confidence!r} in region {region!r}"
                )
            confidence = float(raw_confidence)
            if not 0 <= confidence <= 1:
                raise EnsembleError(f"expert {self.spec.name}: region confidence must be in [0,1]")
            value = int(round(confidence * 255))
            kind = region.get("type")
            coordinates = region.get("coordinates")
            if not isinstance(coordinates, list):
                raise EnsembleError(f"expert {self.spec.name}: region coordinates must be an array")
            if kind == "box":
                if len(coordinates) != 4:
                    raise EnsembleError(f"expert {self.spec.name}: box needs four coordinates")
                numbers = [self._coordinate_number(c, region) for c in coordinates]
                if coordinate_limit is not None and any(
                    not 0 <= number <= coordinate_limit for number in numbers
                ):
                    raise EnsembleError(
                        f"expert {self.spec.name}: normalized coordinates must be in [0,1]"
                    )
                if normalized:
                    x1, y1, x2, y2 = (
                        numbers[0] * width,
                        numbers[1] * height,
                        numbers[2] * width,
                        numbers[3] * height,
                    )
                else:
                    x1, y1, x2, y2 = numbers
                if x2 <= x1 or y2 <= y1:
                    raise EnsembleError(f"expert {self.spec.name}: box has no area")
                if not normalized and (x1 < 0 or y1 < 0 or x2 > width or y2 > height):
                    raise EnsembleError(
                        f"expert {self.spec.name}: absolute box is outside the frame"
                    )
                draw.rectangle((x1, y1, x2, y2), fill=value)
            elif kind == "polygon":
                if len(coordinates) < 3:
                    raise EnsembleError(f"expert {self.spec.name}: polygon needs three points")
                points: list[tuple[float, float]] = []
                for point in coordinates:
                    if not isinstance(point, list) or len(point) != 2:
                        raise EnsembleError(
                            f"expert {self.spec.name}: polygon point must be an "
                            f"[x, y] pair, got {point!r} in region {region!r}"
                        )
                    points.append(
                        (
                            self._coordinate_number(point[0], region),
                            self._coordinate_number(point[1], region),
                        )
                    )
                if coordinate_limit is not None and any(
                    not 0 <= component <= coordinate_limit
                    for point in points
                    for component in point
                ):
                    raise EnsembleError(
                        f"expert {self.spec.name}: normalized coordinates must be in [0,1]"
                    )
                if normalized:
                    points = [(x * width, y * height) for x, y in points]
                if not normalized and any(
                    x < 0 or y < 0 or x > width or y > height for x, y in points
                ):
                    raise EnsembleError(
                        f"expert {self.spec.name}: absolute polygon is outside the frame"
                    )
                draw.polygon(points, fill=value)
            else:
                raise EnsembleError(f"expert {self.spec.name}: region type must be box or polygon")
        if self.spec.boundary_sigma:
            image = image.filter(ImageFilter.GaussianBlur(self.spec.boundary_sigma))
        return np.asarray(image, dtype=np.float64) / 255.0

    def read(self, indices: Sequence[int]) -> np.ndarray:
        request = {
            "prompt": self.spec.prompt,
            "normalized_coordinates": self.spec.normalized_coordinates,
            "frames": [
                {"frame_index": index, "path": str(self.frame_paths[index])} for index in indices
            ],
        }
        response = self._runner(list(self.spec.command), request)
        if not isinstance(response, dict):
            raise EnsembleError(f"expert {self.spec.name}: response must be an object")
        regions_by_frame = response.get("regions_by_frame")
        if not isinstance(regions_by_frame, list):
            raise EnsembleError(f"expert {self.spec.name}: regions_by_frame must be an array")
        if len(regions_by_frame) != len(indices):
            raise EnsembleError(f"expert {self.spec.name}: expected {len(indices)} region records")
        for record, expected_index in zip(regions_by_frame, indices, strict=True):
            if not isinstance(record, dict):
                raise EnsembleError(
                    f"expert {self.spec.name}: region record must be an object, got {record!r}"
                )
            if record.get("frame_index") != expected_index:
                raise EnsembleError(
                    f"expert {self.spec.name}: region records are not in request order"
                )
            if "regions" not in record:
                raise EnsembleError(
                    f"expert {self.spec.name}: region record for frame "
                    f"{expected_index} is missing the regions key: {record!r}"
                )
            if not isinstance(record["regions"], list):
                raise EnsembleError(
                    f"expert {self.spec.name}: regions must be an array, got {record['regions']!r}"
                )
        _, height, width = self._shape
        grids = [
            self._rasterize(record["regions"], width=width, height=height)
            for record in regions_by_frame
        ]
        return np.stack(grids)


def expert_probabilities(values: np.ndarray, specs: Sequence[ExpertSpec]) -> np.ndarray:
    """Convert aligned expert values into calibrated probabilities.

    Probability inputs are converted to logits before temperature scaling, and
    logit inputs are scaled directly. Fusion and calibration share this single
    transform so that a weight fitted on one domain applies to the other.
    """
    if values.ndim != 4:
        raise EnsembleError("expert batch must have experts,frames,height,width shape")
    if values.shape[0] != len(specs):
        raise EnsembleError("expert count does not match tensor count")
    if not np.isfinite(values).all():
        raise EnsembleError("expert values contain NaN or infinity")
    for index, spec in enumerate(specs):
        if spec.representation == "probabilities" and (
            np.min(values[index]) < 0 or np.max(values[index]) > 1
        ):
            raise EnsembleError(f"expert {spec.name}: probabilities must be in [0,1]")
    return np.stack(
        [
            _sigmoid(
                (
                    _logit(values[index])
                    if specs[index].representation == "probabilities"
                    else values[index]
                )
                / specs[index].temperature
            )
            for index in range(len(specs))
        ]
    )


# Anchor gate (S4 verdict, ens-proof): an expert seeded from a source
# mask -- Samurai seeded from the frame-0 SAM 3.1 mask, through its
# bounding box -- can already miss part of that source at frame 0, and
# propagation then carries the incomplete mask forward. In ens-proof
# the frame-0 Samurai mask missed 22,236 px of its 359,464 px seed
# mask (6.19%), at a median 33.6 px from the mask edge: the head and
# bare arm/shoulder were lost at the anchor, not in propagation. This
# gate is the operational fix S4 names: compare the expert's frame-0
# mask against the source mask at the anchor and stop loud on a gap,
# before any propagation or fusion consumes the seeded expert.
# S4 names no fraction; it names the gap. The default limit is 5% of
# the source pixels: the measured ens-proof gap (6.19%) trips it,
# and a boundary-level difference under 5% passes.
ANCHOR_GATE_MAX_MISSED_FRACTION = 0.05


@dataclass(frozen=True)
class AnchorGateReport:
    """The anchor-gate measurement for one seeded expert."""

    expert_name: str
    source_pixels: int
    expert_pixels: int
    missed_pixels: int
    missed_fraction: float
    extra_pixels: int
    max_missed_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "expert_name": self.expert_name,
            "source_pixels": self.source_pixels,
            "expert_pixels": self.expert_pixels,
            "missed_pixels": self.missed_pixels,
            "missed_fraction": self.missed_fraction,
            "extra_pixels": self.extra_pixels,
            "max_missed_fraction": self.max_missed_fraction,
        }


def _anchor_binary_mask(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise EnsembleError(f"anchor gate: {label} mask must be 2D, got shape {array.shape}")
    if array.dtype == bool:
        return array
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise EnsembleError(f"anchor gate: {label} mask contains NaN or infinity")
        # Probability masks binarize at 0.5, the rule the S1 geometry
        # measurement used on the persisted probability tensors.
        return array >= 0.5
    return array > 0


def _anchor_expert_binary_mask(
    values: np.ndarray,
    label: str,
    *,
    representation: str,
    temperature: float,
) -> np.ndarray:
    """Binarize an expert's frame-0 values through the fusion contract.

    The gate judges the mask the fusion would consume, so float
    expert values go through the same transform expert_probabilities
    applies -- sigmoid(values/temperature) for a logits tensor, and
    sigmoid(logit(p)/temperature) for a probabilities tensor -- and
    binarize at 0.5 on the transformed values. Thresholding the raw
    values at 0.5 instead reads a logit of 0.25 (transformed
    probability 0.56) as absent. Bool masks pass through, and
    integer masks keep the > 0 rule.
    """
    if representation not in {"logits", "probabilities"}:
        raise EnsembleError(f"anchor gate: {label} has invalid representation {representation!r}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise EnsembleError(f"anchor gate: {label} temperature must be positive, got {temperature}")
    array = np.asarray(values)
    if array.ndim != 2:
        raise EnsembleError(f"anchor gate: {label} mask must be 2D, got shape {array.shape}")
    if array.dtype == bool:
        return array
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise EnsembleError(f"anchor gate: {label} mask contains NaN or infinity")
        if representation == "probabilities":
            if np.min(array) < 0 or np.max(array) > 1:
                raise EnsembleError(f"anchor gate: {label} probabilities must be in [0, 1]")
            logits = _logit(array)
        else:
            logits = array
        return _sigmoid(logits / temperature) >= 0.5
    return array > 0


_ANCHOR_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_ANCHOR_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})


def load_frame0_mask(path: Path) -> np.ndarray:
    """Load the frame-0 source mask named by an expert anchor_source.

    Supported inputs: a .npy tensor (a 3D array contributes frame 0; a
    2D array is used as-is), a .png/.jpg image (grayscale values over
    127 are source pixels), and a mask video read through
    decode_first_gray_frame (frame 0 only, grayscale over 127). A
    video anchor decodes one frame, never the whole clip. Any
    unreadable file, unsupported suffix, or wrong tensor rank raises
    EnsembleError: a declared anchor that cannot be read must fail
    closed, never pass silently.
    """
    anchor_path = Path(path)
    if not anchor_path.is_file():
        raise EnsembleError(f"anchor gate: anchor source not found: {anchor_path}")
    suffix = anchor_path.suffix.lower()
    try:
        if suffix == ".npy":
            values = np.load(anchor_path)
            if values.ndim == 3:
                return np.asarray(values[0])
            if values.ndim == 2:
                return np.asarray(values)
            raise EnsembleError(
                f"anchor gate: anchor source {anchor_path} must be a 2D mask or a "
                f"3D (frames, height, width) tensor, got shape {values.shape}"
            )
        if suffix in _ANCHOR_IMAGE_SUFFIXES:
            with Image.open(anchor_path) as image:
                return np.asarray(image.convert("L")) > 127
        if suffix in _ANCHOR_VIDEO_SUFFIXES:
            with TemporaryDirectory(prefix=".samurai-anchor-") as temporary:
                frame = decode_first_gray_frame(anchor_path, Path(temporary) / "frames")
            return np.asarray(frame) > 127
    except EnsembleError:
        raise
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise EnsembleError(
            f"anchor gate: cannot read anchor source {anchor_path}: {error}"
        ) from error
    raise EnsembleError(f"anchor gate: unsupported anchor source suffix {suffix!r}: {anchor_path}")


def check_anchor_gate(
    source_mask: np.ndarray | None,
    expert_frame0_mask: np.ndarray,
    *,
    expert_name: str = "expert",
    max_missed_fraction: float = ANCHOR_GATE_MAX_MISSED_FRACTION,
    expert_representation: str = "probabilities",
    expert_temperature: float = 1.0,
) -> AnchorGateReport | None:
    """Check one seeded expert's frame-0 mask against its source mask.

    The gate is off, explicitly, when ``source_mask`` is None: an
    expert that was not seeded from a source mask has no anchor to
    check, and the call returns None instead of tripping. When a
    source mask exists, the missed pixels are the source pixels the
    expert's frame-0 mask does not hold. A missed fraction strictly
    over ``max_missed_fraction`` raises EnsembleError naming the gap
    size in pixels; at or under the limit the measurement is returned
    as an AnchorGateReport. An empty source mask, a shape mismatch,
    and an invalid limit all raise EnsembleError: a seeded expert
    whose anchor cannot be checked must not pass silently.

    The expert's frame-0 values are binarized through the expert's
    contract transform (see _anchor_expert_binary_mask), so the gate
    judges the mask the fusion would consume. The defaults --
    probabilities at temperature 1.0 -- keep the plain probability
    rule for callers that hold no expert spec; run_ensemble passes
    the spec's representation and temperature.
    """
    if source_mask is None:
        return None
    if not math.isfinite(max_missed_fraction) or not 0 <= max_missed_fraction <= 1:
        raise EnsembleError(
            f"anchor gate: max_missed_fraction must be in [0, 1], got {max_missed_fraction}"
        )
    source = _anchor_binary_mask(source_mask, "source")
    expert = _anchor_expert_binary_mask(
        expert_frame0_mask,
        f"expert {expert_name!r} frame-0",
        representation=expert_representation,
        temperature=expert_temperature,
    )
    if source.shape != expert.shape:
        raise EnsembleError(
            f"anchor gate: expert {expert_name!r} frame-0 mask shape {expert.shape} "
            f"does not match source mask shape {source.shape}"
        )
    source_pixels = int(np.count_nonzero(source))
    if source_pixels == 0:
        raise EnsembleError(
            f"anchor gate refused: expert {expert_name!r} was seeded from an empty "
            "source mask; the anchor cannot be checked"
        )
    missed_pixels = int(np.count_nonzero(source & ~expert))
    missed_fraction = missed_pixels / source_pixels
    report = AnchorGateReport(
        expert_name=expert_name,
        source_pixels=source_pixels,
        expert_pixels=int(np.count_nonzero(expert)),
        missed_pixels=missed_pixels,
        missed_fraction=missed_fraction,
        extra_pixels=int(np.count_nonzero(expert & ~source)),
        max_missed_fraction=float(max_missed_fraction),
    )
    if missed_fraction > max_missed_fraction:
        raise EnsembleError(
            f"anchor gate refused: expert {expert_name!r} frame-0 mask misses "
            f"{missed_pixels} px of the source mask ({missed_fraction:.2%} of "
            f"{source_pixels} px; limit {max_missed_fraction:.2%}); stop at the "
            "anchor, before propagation carries the incomplete mask forward"
        )
    return report


def _effective_weights(
    probabilities: np.ndarray, base_weights: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "static":
        return np.broadcast_to(base_weights[:, None], probabilities.shape[:2])
    epsilon = 1e-7
    safe = np.clip(probabilities, epsilon, 1.0 - epsilon)
    entropy = -np.mean(
        safe * np.log2(safe) + (1.0 - safe) * np.log2(1.0 - safe),
        axis=(2, 3),
    )
    confidence = 1.0 - entropy
    dynamic = confidence * base_weights[:, None]
    totals = np.sum(dynamic, axis=0, keepdims=True)
    if not np.all(totals > 0):
        return np.broadcast_to(base_weights[:, None], probabilities.shape[:2])
    return dynamic / totals


def _selected_count(ratio: float, pixel_count: int) -> int:
    if ratio <= 1.0:
        return max(1, int(round(ratio * pixel_count)))
    return max(1, min(pixel_count, int(ratio)))


def _fuse_batch_with_probabilities(
    values: np.ndarray,
    specs: Sequence[ExpertSpec],
    fusion: FusionSpec,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Fuse aligned expert values into masks and report effective weights.

    D2 (gate D2): an expert supports a pixel when its probability
    is over the support threshold, and the active set at a pixel is
    its supporting experts. ``combined`` is the weighted average
    over the active set only, renormalized per pixel over active
    expert mass using the frame-level effective weights (static, or
    entropy mode, which stays per frame): an expert under the
    threshold abstains and adds neither mass nor probability. One
    threshold defines both support and abstention. A pixel with an
    empty active set gets combined 0 and is ineligible. Each frame
    report carries a support_histogram: pixel counts by active set
    size.
    """
    probabilities = expert_probabilities(values, specs)
    base_weights = np.asarray([spec.weight for spec in specs], dtype=np.float64)
    base_weights = base_weights / np.sum(base_weights)
    weights = _effective_weights(probabilities, base_weights, fusion.dynamic_weights)
    supporting = probabilities > fusion.support_threshold
    support = np.sum(supporting, axis=0)
    active_weights = np.where(supporting, weights[..., None, None], 0.0)
    active_mass = np.sum(active_weights, axis=0)
    numerator = np.sum(probabilities * active_weights, axis=0, dtype=np.float64)
    combined = np.zeros_like(numerator)
    np.divide(numerator, active_mass, out=combined, where=active_mass > 0)
    if not np.isfinite(combined).all() or not np.isfinite(weights).all():
        raise EnsembleError("fused probabilities contain NaN or infinity")
    eligible = support >= fusion.min_expert_support
    masks = np.zeros(combined.shape, dtype=bool)
    for frame_index, frame_probability in enumerate(combined):
        if fusion.mode == "threshold":
            masks[frame_index] = (frame_probability >= fusion.threshold) & eligible[frame_index]
            continue
        flat_probability = frame_probability.reshape(-1)
        flat_eligible = eligible[frame_index].reshape(-1)
        candidates = np.flatnonzero(flat_eligible & (flat_probability >= fusion.threshold))
        if candidates.size:
            selected_count = min(
                _selected_count(fusion.top_k_ratio, flat_probability.size),
                candidates.size,
            )
            order = np.argsort(-flat_probability[candidates], kind="stable")[:selected_count]
            chosen = candidates[order]
            masks[frame_index].reshape(-1)[chosen] = True
    reports: list[dict[str, Any]] = []
    for frame_index in range(values.shape[1]):
        report: dict[str, Any] = {
            f"weight_{spec.name}": float(weights[expert_index, frame_index])
            for expert_index, spec in enumerate(specs)
        }
        histogram = np.bincount(support[frame_index].ravel(), minlength=len(specs) + 1)
        report["support_histogram"] = [int(count) for count in histogram]
        reports.append(report)
    return masks, combined, reports, probabilities


def fuse_batch(
    values: np.ndarray,
    specs: Sequence[ExpertSpec],
    fusion: FusionSpec,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Fuse one batch and return masks, combined probabilities, and reports."""
    masks, combined, reports, _expert_probabilities = _fuse_batch_with_probabilities(
        values, specs, fusion
    )
    return masks, combined, reports


def dynamic_batches(
    frame_count: int,
    shape: tuple[int, int, int],
    fusion: FusionSpec,
) -> list[tuple[int, ...]]:
    _, height, width = shape
    pixel_batch_size = max(1, fusion.max_batch_pixels // (height * width))
    size = min(fusion.batch_size, pixel_batch_size)
    return [
        tuple(range(start, min(start + size, frame_count))) for start in range(0, frame_count, size)
    ]


def _open_tensor_sources(
    config: EnsembleConfig,
    tensor_specs: Sequence[ExpertSpec],
    resolved_source: ResolvedWeightSource | None,
) -> tuple[dict[str, TensorExpertSource], tuple[int, int, int] | None]:
    """Open tensor sources, holding D4 locks for the governed panel.

    D1 verifies the panel before the run probes video or opens a tensor.
    A cooperating D4 writer could otherwise replace an artifact in that
    gap. Re-verify the complete panel and create every memory mapping
    while holding each directory's manifest lock; once NumPy returns the
    mapping, replacement by rename cannot change the mapped inode. The
    second verification must produce the same manifest and per-expert
    bindings D1 resolved, otherwise the run fails before output.
    """

    sources: dict[str, TensorExpertSource] = {}
    reference_shape: tuple[int, int, int] | None = None
    governed = resolved_source is not None and resolved_source.governed
    if governed:
        directories = sorted(
            {
                spec.path.expanduser().resolve().parent
                for spec in tensor_specs
                if spec.path is not None
            }
        )
        with ExitStack() as stack:
            for directory in directories:
                stack.enter_context(manifest_lock(directory))
            manifest_sha256, bindings = _verify_consumed_manifest(
                config, "(governed source acquisition after D1 resolution)"
            )
            expected = resolved_source.report if resolved_source is not None else {}
            if manifest_sha256 != expected.get("manifest_sha256") or bindings != expected.get(
                "artifact_bindings"
            ):
                raise EnsembleError(
                    "weight source refused at check manifest_stability: the governed "
                    "D4 panel changed between D1 verification and tensor mapping "
                    f"(expected manifest {expected.get('manifest_sha256')!r}, found "
                    f"{manifest_sha256!r})"
                )
            for spec in tensor_specs:
                source = TensorExpertSource(spec)
                if reference_shape is None:
                    reference_shape = source.shape
                elif source.shape != reference_shape:
                    raise EnsembleError("expert tensors have mismatching frames/height/width")
                sources[spec.name] = source
        return sources, reference_shape

    for spec in tensor_specs:
        source = TensorExpertSource(spec)
        if reference_shape is None:
            reference_shape = source.shape
        elif source.shape != reference_shape:
            raise EnsembleError("expert tensors have mismatching frames/height/width")
        sources[spec.name] = source
    return sources, reference_shape


def run_ensemble(
    config: EnsembleConfig,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Fuse expert tensors/commands into a mask video with bounded frame batches."""
    # D1: resolve the declared weight source before any output is
    # written. A refused source raises here; the governed run never
    # falls back to hand weights or config weight values. A governed
    # run also requires unanimous support (gate D2): min_expert_support
    # equal to the panel size, which is 3 on the three-expert panel.
    if config.expert_persist_mode not in EXPERT_PERSIST_MODES:
        raise EnsembleError(
            f"unknown expert_persist_mode: {config.expert_persist_mode!r} "
            f"(expected one of {sorted(EXPERT_PERSIST_MODES)})"
        )
    resolved_source = resolve_weight_source(config)
    experts = config.experts
    if resolved_source is not None:
        if resolved_source.governed and config.fusion.min_expert_support != len(config.experts):
            raise EnsembleError(
                "governed run requires unanimous support: min_expert_support "
                f"{config.fusion.min_expert_support} != expert count {len(config.experts)}"
            )
        experts = tuple(
            replace(spec, weight=resolved_source.weights[spec.name]) for spec in config.experts
        )
    if config.expert_persist_dir is not None and config.expert_persist_mode != "off":
        consumed_dirs = {
            spec.path.expanduser().resolve().parent
            for spec in experts
            if spec.kind == "tensor" and spec.path
        }
        persist_dir = config.expert_persist_dir.expanduser().resolve()
        if persist_dir in consumed_dirs:
            raise EnsembleError(
                "expert_persist_dir refuses to overwrite a consumed D4 artifact "
                f"directory: {persist_dir}; write the derived run to a new directory"
            )
    # D4 feature flag: "required" makes the governed path always
    # persist, so refuse here -- fail-closed, beside the D1/D2
    # refusal above and before any frame work -- when no persist
    # target is set or the target cannot be written.
    if config.expert_persist_mode == "required":
        if config.expert_persist_dir is None:
            raise EnsembleError(
                "expert_persist_mode required refuses: no expert_persist_dir "
                "is set, and a required run always persists its expert tensors"
            )
        try:
            config.expert_persist_dir.mkdir(parents=True, exist_ok=True)
            probe = config.expert_persist_dir / ".expert-persist-probe"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as error:
            raise EnsembleError(
                "expert_persist_mode required refuses: persist target "
                f"{config.expert_persist_dir} cannot be written: {error}"
            ) from error
    video_info = probe_video(config.input_video)
    tensor_specs = [spec for spec in experts if spec.kind == "tensor"]
    command_specs = [spec for spec in experts if spec.kind == "command"]
    tensor_sources, reference_shape = _open_tensor_sources(config, tensor_specs, resolved_source)
    sources: dict[str, TensorExpertSource | CommandExpertSource] = dict(tensor_sources)
    if reference_shape is None:
        reference_shape = (video_info.frames, video_info.height, video_info.width)
    if reference_shape[0] != video_info.frames:
        raise EnsembleError("expert tensor frame count does not match input video")
    if reference_shape[1:] != (video_info.height, video_info.width):
        raise EnsembleError("expert tensor dimensions do not match input video")
    with TemporaryDirectory(prefix=".samurai-ensemble-") as temporary:
        frame_paths: list[Path] = []
        if command_specs:
            frame_paths = extract_color_frames(config.input_video, Path(temporary) / "frames")
        for spec in command_specs:
            sources[spec.name] = CommandExpertSource(spec, frame_paths, reference_shape)
        batches = dynamic_batches(reference_shape[0], reference_shape, config.fusion)
        frame_reports: list[dict[str, Any] | None] = [None] * reference_shape[0]
        # An expert whose consumed tensor is identically zero in
        # every batch is empty. Track it from the values the fusion
        # consumes, in every persist mode, so an empty expert is a
        # recorded failure in the run report -- never silent, never
        # dropped, never reweighted away.
        expert_nonzero: dict[str, bool] = {spec.name: False for spec in experts}
        # D4: collect the exact per-expert values the fusion consumes
        # (raw expert values, before the probability transform), so
        # every expert -- including command rasterizations -- can be
        # persisted after the run. Buffers exist only when opted in;
        # under the persist feature flag, mode "off" allocates none
        # even when a directory is set, and mode "required" was
        # already refused above if it had no writable target.
        expert_buffers: dict[str, np.ndarray] | None = None
        if config.expert_persist_dir is not None and config.expert_persist_mode != "off":
            # D4 storage dtype: float16 (two bytes per element per
            # expert), not the float64 full-clip buffer (eight bytes per
            # element) this replaces. The buffer is still the clip's size
            # in the storage dtype; it is not streamed to disk.
            expert_buffers = {
                spec.name: np.empty(reference_shape, dtype=PERSISTED_DTYPE) for spec in experts
            }
        # The first batch is read once, here. The anchor gate judges the
        # frame-0 values of that batch -- the values the fusion consumes.
        # A command expert is therefore invoked exactly once for the
        # first batch: the gate never re-invokes it, and it never judges
        # a different frame-0 sample than the one that reaches the
        # fusion.
        first_values: np.ndarray | None = None
        anchor_reports: list[dict[str, Any]] = []
        if batches:
            first_indices = batches[0]
            first_values = np.stack([sources[spec.name].read(first_indices) for spec in experts])
            frame0_position = list(first_indices).index(0)
            # Anchor gate: after every source is built and reference_shape
            # is validated, before GrayVideoWriter writes any frame. A
            # trip raises EnsembleError here, so a refused run writes no
            # output video and no run report.
            for expert_index, spec in enumerate(experts):
                if spec.anchor_source_path is None:
                    continue
                source_mask = load_frame0_mask(spec.anchor_source_path)
                frame0 = first_values[expert_index][frame0_position]
                anchor_report = check_anchor_gate(
                    source_mask,
                    frame0,
                    expert_name=spec.name,
                    expert_representation=spec.representation,
                    expert_temperature=spec.temperature,
                )
                if anchor_report is not None:
                    anchor_reports.append(anchor_report.as_dict())
        elif any(spec.anchor_source_path is not None for spec in experts):
            raise EnsembleError(
                "anchor gate refused: the run has no frames, so the frame-0 "
                "anchor of a seeded expert cannot be checked"
            )
        with GrayVideoWriter(
            output,
            width=reference_shape[2],
            height=reference_shape[1],
            fps=video_info.fps,
        ) as writer:
            for batch_index, indices in enumerate(batches):
                if batch_index == 0 and first_values is not None:
                    values = first_values
                else:
                    values = np.stack([sources[spec.name].read(indices) for spec in experts])
                if expert_buffers is not None:
                    for expert_index, spec in enumerate(experts):
                        expert_buffers[spec.name][list(indices)] = _to_persisted_dtype(
                            values[expert_index], spec.name
                        )
                masks, probabilities, weights, expert_probability_values = (
                    _fuse_batch_with_probabilities(values, experts, config.fusion)
                )
                # Empty means no meaningful transformed support, not merely
                # nonzero raw storage. A logits panel filled with large
                # negative values is non-empty raw storage but dead evidence.
                # The comparison is strict, exactly like fuse_batch: an expert
                # whose transformed probability sits at the support threshold
                # abstains at every pixel and contributes no active mass, so it
                # is empty evidence for this run.
                for expert_index, spec in enumerate(experts):
                    if np.any(
                        expert_probability_values[expert_index] > config.fusion.support_threshold
                    ):
                        expert_nonzero[spec.name] = True
                for local_index, frame_index in enumerate(indices):
                    writer.write(masks[local_index].astype(np.uint8) * 255)
                    frame_reports[frame_index] = {
                        "selected_pixels": int(np.count_nonzero(masks[local_index])),
                        "mean_probability": float(np.mean(probabilities[local_index])),
                        **weights[local_index],
                    }
    if any(report is None for report in frame_reports):
        raise EnsembleError("ensemble did not emit every frame")
    expert_empty = [spec.name for spec in experts if not expert_nonzero[spec.name]]
    expert_artifacts: dict[str, Any] | None = None
    if config.expert_persist_dir is not None and expert_buffers is not None:
        try:
            manifest = persist_expert_tensors(
                config.expert_persist_dir,
                expert_buffers,
                version=config.expert_artifact_version,
                sources={
                    spec.name: {
                        "command": list(spec.command) if spec.command else None,
                        "prompt": spec.prompt,
                        "path": str(spec.path) if spec.path is not None else None,
                    }
                    for spec in experts
                },
            )
        except ExpertPersistError as error:
            if config.expert_persist_mode == "required":
                raise EnsembleError(
                    "expert_persist_mode required refuses: expert tensors "
                    f"could not be persisted to {config.expert_persist_dir}: {error}"
                ) from error
            raise
        expert_artifacts = {
            "manifest": str(config.expert_persist_dir / MANIFEST_NAME),
            "artifacts": manifest["artifacts"],
        }
    result = {
        "input_video": str(config.input_video),
        "output": str(output),
        "frames": reference_shape[0],
        "width": reference_shape[2],
        "height": reference_shape[1],
        "fps": video_info.fps,
        "experts": [spec.name for spec in experts],
        "expert_empty": expert_empty,
        "anchor_gate": anchor_reports,
        "fusion": {
            "mode": config.fusion.mode,
            "dynamic_weights": config.fusion.dynamic_weights,
            "top_k_ratio": config.fusion.top_k_ratio,
            "threshold": config.fusion.threshold,
            "min_expert_support": config.fusion.min_expert_support,
            "support_threshold": config.fusion.support_threshold,
        },
        **({"calibration": config.calibration} if config.calibration else {}),
        **({"weight_source": resolved_source.report} if resolved_source is not None else {}),
        **({"expert_artifacts": expert_artifacts} if expert_artifacts else {}),
        "expert_persistence": (
            {
                "enabled": True,
                "mode": config.expert_persist_mode,
                "dir": str(config.expert_persist_dir),
                "artifact_version": config.expert_artifact_version,
            }
            if expert_artifacts is not None
            else {"enabled": False, "mode": config.expert_persist_mode}
        ),
        "batches": [len(batch) for batch in batches],
        "frames_report": frame_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Required mode is the governed path: an expert that consumed as
    # identically zero on every frame is a failed expert, not a
    # silent success. The evidence is already on disk at this point
    # (the persisted tensors, their manifest expert_empty records,
    # and the run report above, which names the empty experts), so
    # refuse the run here -- fail-closed -- instead of returning a
    # result a caller could treat as a completed governed run.
    # Opt-in and off modes keep the recorded-flag behaviour only.
    if config.expert_persist_mode == "required" and expert_empty:
        raise EnsembleError(
            "expert_persist_mode required refuses: expert(s) "
            f"{', '.join(expert_empty)} are empty (expert_empty) on every frame; "
            "the persisted tensors, manifest, and run report record the failure"
        )
    return result
