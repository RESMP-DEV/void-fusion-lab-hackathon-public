"""Dawid-Skene calibration over persisted expert tensors (fusion gate D3).

The gate D3 ruling: run label-free Dawid-Skene calibration only after
D4, once all aligned tensors exist as persisted artifacts. Store the
calibration inputs, configuration, output weights, diagnostics, and
non-claims. A failed or non-identifiable calibration is a recorded
failure, not a silent fallback.

This module is the D3 gate layer over ``samurai.calibration`` (the
baseline estimator) and ``samurai.expert_persist`` (the D4 layer):

- The only inputs are the persisted artifacts in one expert-tensor
  directory. The D4 manifest must exist and verify (every artifact
  present, SHA-256 matching) before any tensor is read; an absent or
  incomplete manifest is a refusal, recorded as a failure.
- The D4 manifest does not carry each expert's representation, so the
  caller must declare it (``logits`` or ``probabilities``) for every
  persisted expert. An undeclared representation is a refusal: raw
  values are never interpreted under an unrecorded assumption.
- Identifiability is checked, not assumed. A constant expert report
  leaves one row of its confusion matrix unobserved, identical expert
  reports leave the panel's confusion matrices inseparable, a
  non-converged fit or a degenerate latent class produces no valid
  weights, and an expert with no measured skill above chance has no
  valid measured weight. Each case writes a failure record and
  raises; the estimator's uniform-weight fallback is never served.
- A successful run writes one calibration record carrying the input
  manifest SHA-256 and per-artifact digests, the full configuration,
  the output weights, the fit diagnostics, and the non-claims. A
  failed run writes a failure record instead. The hand weights
  0.40/0.35/0.25 are a declared comparison baseline (gate D1) and are
  never substituted by this module.
- A run invalidates any prior success record before it starts, and a
  failed run leaves no readable success record behind: the prior
  record is removed (or overwritten with an invalidated stub), so no
  consumer can read weights that no longer describe the tensors. The
  record and the failure record are mutually exclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import NON_CLAIMS, CalibrationError, CalibrationSettings, calibrate_experts
from .ensemble import EnsembleError, ExpertSpec, expert_probabilities
from .expert_persist import (
    DEFAULT_ARTIFACT_VERSION,
    MANIFEST_NAME,
    ExpertPersistError,
    load_expert_tensor,
    verify_manifest,
)

RECORD_NAME = "calibration_record.json"
FAILURE_NAME = "calibration_failure.json"
RECORD_VERSION = 1

NON_CLAIMS_D3 = (
    "Weights are label-free Dawid-Skene estimates fitted to the persisted "
    "D4 tensors named in the inputs; calibration measures panel agreement, "
    "not semantic ground truth.",
    "On failure this gate records the failure and produces no weights; the "
    "hand weights 0.40/0.35/0.25 are a comparison baseline only and are "
    "never substituted silently.",
)

FAILURE_NON_CLAIMS = (
    "No weights were produced by this run.",
    "The hand weights 0.40/0.35/0.25 were not substituted.",
)

_REPRESENTATIONS = {"logits", "probabilities"}
_DEGENERATE_EPSILON = 1e-6


class CalibrationRunError(RuntimeError):
    """A D3 calibration run was refused or failed; the failure is recorded."""


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file atomically: temp file, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _invalidate_prior_record(output_dir: Path) -> dict[str, Any]:
    """Remove any prior success record so no consumer can read stale weights.

    A calibration record that was fitted on tensors the current run no
    longer accepts must not stay readable: a consumer pins the record
    digest, so a surviving record would keep applying weights that no
    longer describe the panel. The prior record is deleted; when the
    delete fails it is overwritten with an invalidated stub, because a
    consumer must never find a readable, status-recorded record here.
    The returned receipt names what was invalidated.
    """
    path = Path(output_dir) / RECORD_NAME
    if not path.is_file():
        return {"invalidated": False, "reason": "no prior record"}
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    try:
        path.unlink()
    except OSError as error:
        stub = {
            "record_version": RECORD_VERSION,
            "gate": "D3",
            "status": "invalidated",
            "reason": "a later calibration run failed; these weights are stale",
            "weights": None,
            "recorded_at": datetime.now(UTC).isoformat(),
            "non_claims": list(FAILURE_NON_CLAIMS),
        }
        try:
            _atomic_write_json(path, stub)
        except OSError:
            return {
                "invalidated": False,
                "sha256": digest,
                "reason": (f"the prior record could not be removed or invalidated: {error}"),
            }
        return {"invalidated": True, "sha256": digest, "method": "invalidated stub"}
    return {"invalidated": True, "sha256": digest, "method": "removed"}


def _record_failure(
    output_dir: Path,
    *,
    stage: str,
    reason: str,
    manifest_sha256: str | None,
    invalidated: dict[str, Any] | None = None,
) -> Path:
    """Write the D3 failure record and return its path."""
    failure = {
        "record_version": RECORD_VERSION,
        "gate": "D3",
        "status": "failure",
        "stage": stage,
        "reason": reason,
        "inputs": {"manifest_sha256": manifest_sha256},
        "weights": None,
        "invalidated_prior_record": invalidated or {"invalidated": False},
        "recorded_at": datetime.now(UTC).isoformat(),
        "non_claims": list(FAILURE_NON_CLAIMS),
    }
    path = Path(output_dir) / FAILURE_NAME
    _atomic_write_json(path, failure)
    return path


def _fail(
    output_dir: Path,
    *,
    stage: str,
    reason: str,
    manifest_sha256: str | None,
    invalidated: dict[str, Any] | None = None,
) -> CalibrationRunError:
    """Record one failure and return the error to raise for it."""
    path = _record_failure(
        output_dir,
        stage=stage,
        reason=reason,
        manifest_sha256=manifest_sha256,
        invalidated=invalidated,
    )
    return CalibrationRunError(f"{reason} (failure record: {path})")


def _settings_payload(settings: CalibrationSettings) -> dict[str, Any]:
    """Return the calibration configuration as stored in the record."""
    return {
        "tolerance": settings.tolerance,
        "max_iterations": settings.max_iterations,
        "max_samples": settings.max_samples,
        "max_batch_pixels": settings.max_batch_pixels,
        "seed": settings.seed,
        "smoothing": {
            "positive_prior": vars(settings.positive_prior),
            "sensitivity": vars(settings.sensitivity),
            "false_positive_rate": vars(settings.false_positive_rate),
        },
    }


def run_calibration(
    expert_dir: Path,
    output_dir: Path,
    *,
    representations: dict[str, str],
    temperatures: dict[str, float] | None = None,
    settings: CalibrationSettings | None = None,
    artifact_version: str = DEFAULT_ARTIFACT_VERSION,
) -> dict[str, Any]:
    """Run the D3 calibration over one directory of D4 artifacts.

    Returns the calibration record on success. Every refusal or
    failure writes ``calibration_failure.json`` in ``output_dir`` and
    raises ``CalibrationRunError``; no weights are produced on any
    failure path.
    """
    bounds = settings or CalibrationSettings()
    directory = Path(expert_dir)
    out = Path(output_dir)
    temps = temperatures or {}

    manifest_path = directory / MANIFEST_NAME
    manifest_sha256: str | None = None
    # Invalidate any prior success record before this run does any work:
    # until this run records a fresh success, the directory holds no
    # readable weights, so a consumer can never apply weights that this
    # run has not confirmed.
    invalidated = _invalidate_prior_record(out)

    def fail(stage: str, reason: str) -> CalibrationRunError:
        """Record this run's failure and return the error to raise."""
        return _fail(
            out,
            stage=stage,
            reason=reason,
            manifest_sha256=manifest_sha256,
            invalidated=invalidated,
        )

    try:
        if manifest_path.is_file():
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest = verify_manifest(directory)
    except (ExpertPersistError, OSError, ValueError) as error:
        raise fail(stage="manifest", reason=str(error)) from error

    records = [
        item for item in manifest["artifacts"] if str(item.get("version")) == str(artifact_version)
    ]
    seen: set[str] = set()
    for item in records:
        expert = str(item.get("expert"))
        if expert in seen:
            raise fail(
                stage="manifest",
                reason=f"duplicate D4 artifacts for expert {expert!r} at version "
                f"{artifact_version!r}; the calibration input set is ambiguous",
            )
        seen.add(expert)
    if len(records) < 3:
        raise fail(
            stage="manifest",
            reason=f"D3 calibration requires at least three persisted experts, "
            f"found {len(records)} at version {artifact_version!r}",
        )
    shapes = {tuple(int(axis) for axis in item["shape"]) for item in records}
    if len(shapes) != 1:
        raise fail(
            stage="manifest",
            reason=f"persisted expert tensors are not aligned: shapes {sorted(shapes)}",
        )
    shape = shapes.pop()
    if len(shape) != 3:
        raise fail(
            stage="manifest",
            reason=f"persisted expert tensors must have frames,height,width shape, got {shape}",
        )
    if any(axis == 0 for axis in shape):
        raise fail(
            stage="inputs",
            reason=f"persisted expert tensors have a zero-length axis, got shape {shape}; "
            "an empty panel has no calibration observations",
        )

    for item in records:
        expert = str(item["expert"])
        representation = representations.get(expert)
        if representation is None:
            raise fail(
                stage="inputs",
                reason=f"expert {expert!r}: representation is not declared; the D4 "
                "manifest does not carry it and raw values are never interpreted "
                "under an unrecorded assumption",
            )
        if representation not in _REPRESENTATIONS:
            raise fail(
                stage="inputs",
                reason=f"expert {expert!r}: invalid representation {representation!r}",
            )

    # Every expert spec is validated before a single tensor is read: a
    # nonpositive or non-finite temperature would fit weights under a
    # probability transform that no governed run can reproduce, because
    # the run config rejects the same value, so the calibration record
    # would be unusable by construction.
    specs: list[ExpertSpec] = []
    for item in records:
        expert = str(item["expert"])
        try:
            temperature = float(temps.get(expert, 1.0))
        except (TypeError, ValueError) as error:
            raise fail(
                stage="inputs",
                reason=f"expert {expert!r}: temperature is not a number: {error}",
            ) from error
        if not math.isfinite(temperature) or temperature <= 0:
            raise fail(
                stage="inputs",
                reason=f"expert {expert!r}: temperature must be a finite positive "
                f"number, got {temperature!r}; a nonpositive or non-finite "
                "temperature fits weights under a probability transform that no "
                "governed run can reproduce",
            )
        spec = ExpertSpec(
            name=expert,
            weight=1.0,
            temperature=temperature,
            kind="tensor",
            path=directory / str(item["file"]),
            representation=representations[expert],
        )
        try:
            spec.validate()
        except EnsembleError as error:
            raise fail(stage="inputs", reason=str(error)) from error
        specs.append(spec)

    # One float64 stack is filled expert by expert. During identifiability
    # checks a second full float64 probability panel and a boolean vote panel
    # are also resident; both are released before the EM fit.
    values = np.empty((len(records), *shape), dtype=np.float64)
    for index, item in enumerate(records):
        expert = str(item["expert"])
        try:
            array = load_expert_tensor(directory, expert, version=str(artifact_version))
        except (ExpertPersistError, OSError, ValueError) as error:
            raise fail(stage="manifest", reason=str(error)) from error
        artifact_path = directory / str(item["file"])
        try:
            loaded_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        except OSError as error:
            raise fail(stage="manifest", reason=str(error)) from error
        if loaded_sha256 != str(item["sha256"]):
            raise fail(
                stage="manifest",
                reason=f"expert {expert!r}: artifact changed after D4 verification: "
                f"loaded {loaded_sha256} != recorded {str(item['sha256'])}",
            )
        values[index] = array
    try:
        final_manifest_sha256 = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else None
        )
    except OSError as error:
        raise fail(stage="manifest", reason=str(error)) from error
    if final_manifest_sha256 != manifest_sha256:
        raise fail(
            stage="manifest",
            reason="D4 manifest changed during D3 tensor loading: "
            f"initial {manifest_sha256!r}, final {final_manifest_sha256!r}",
        )

    try:
        probabilities = expert_probabilities(values, specs)
    except EnsembleError as error:
        raise fail(stage="inputs", reason=str(error)) from error
    votes = probabilities >= 0.5
    for index, spec in enumerate(specs):
        if bool(votes[index].min()) == bool(votes[index].max()):
            raise fail(
                stage="identifiability",
                reason=f"expert {spec.name!r}: report is constant at the 0.5 vote "
                "threshold, so one row of its confusion matrix is unobserved and "
                "the calibration is not identifiable",
            )
    if all(
        np.array_equal(probabilities[0], probabilities[index]) for index in range(1, len(specs))
    ):
        raise fail(
            stage="identifiability",
            reason="all expert reports are identical, so the panel confusion "
            "matrices are not separately identifiable",
        )
    del probabilities, votes

    try:
        result = calibrate_experts(values, specs, bounds)
    except CalibrationError as error:
        raise fail(stage="calibration", reason=str(error)) from error

    if not result.converged:
        raise fail(
            stage="calibration",
            reason=f"Dawid-Skene fit did not converge within {result.iterations} "
            "iterations; no weights are recorded from an unconverged fit",
        )
    for metric in result.expert_metrics:
        if float(metric["youden_j"]) <= 0.0:
            raise fail(
                stage="calibration",
                reason=f"expert {metric['name']!r}: measured Youden J is "
                f"{float(metric['youden_j']):.6f} (no skill above chance), so it "
                "has no valid measured weight and the calibration is recorded "
                "as failed",
            )
    selected = result.latent_selected_fraction
    if selected <= _DEGENERATE_EPSILON or selected >= 1.0 - _DEGENERATE_EPSILON:
        raise fail(
            stage="calibration",
            reason=f"latent class is degenerate (selected fraction {selected:.6f}); "
            "the two latent classes are not identified",
        )
    weights = result.confidence_weights
    if (
        not all(np.isfinite(weight) and weight >= 0.0 for weight in weights.values())
        or abs(sum(weights.values()) - 1.0) > 1e-6
    ):
        raise fail(
            stage="calibration",
            reason=f"calibrated weights are not a valid distribution: {weights}",
        )

    record: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "gate": "D3",
        "status": "recorded",
        "method": "dawid_skene_bernoulli_em",
        "labels": "none",
        "recorded_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "expert_dir": str(directory),
            "manifest_file": MANIFEST_NAME,
            "manifest_sha256": manifest_sha256,
            "artifact_version": str(artifact_version),
            "shape": {"frames": shape[0], "height": shape[1], "width": shape[2]},
            "artifacts": [
                {
                    "expert": str(item["expert"]),
                    "version": str(item["version"]),
                    "file": str(item["file"]),
                    "shape": [int(axis) for axis in item["shape"]],
                    "dtype": str(item["dtype"]),
                    "sha256": str(item["sha256"]),
                }
                for item in records
            ],
        },
        "configuration": {
            "representations": {
                str(item["expert"]): representations[str(item["expert"])] for item in records
            },
            "temperatures": {
                str(item["expert"]): float(temps.get(str(item["expert"]), 1.0)) for item in records
            },
            "settings": _settings_payload(bounds),
        },
        "weights": weights,
        "diagnostics": {
            "iterations": result.iterations,
            "converged": result.converged,
            "relabelled": result.relabelled,
            "samples": result.samples,
            "latent_prior": result.latent_prior,
            "latent_positive_fraction": result.latent_positive_fraction,
            "latent_selected_fraction": result.latent_selected_fraction,
            "anchor": result.anchor,
            "anchor_agreement": result.anchor_agreement,
            "expert_metrics": list(result.expert_metrics),
            "pairwise": list(result.pairwise_metrics),
        },
        "non_claims": [*NON_CLAIMS, *NON_CLAIMS_D3],
    }
    _atomic_write_json(out / RECORD_NAME, record)
    # The record and the failure record are mutually exclusive: a
    # failure record from an earlier run would misdescribe this one.
    (out / FAILURE_NAME).unlink(missing_ok=True)
    return record


def _parse_assignments(items: list[str], *, flag: str) -> dict[str, str]:
    """Parse repeated ``EXPERT=VALUE`` flag values into a mapping."""
    parsed: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or not value:
            raise CalibrationRunError(f"{flag} expects EXPERT=VALUE, got {item!r}")
        parsed[name] = value
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Console entry point: 0 on a recorded calibration, 2 on a recorded failure."""
    parser = argparse.ArgumentParser(
        prog="samurai.calibration_run",
        description="Run gate D3 Dawid-Skene calibration over persisted D4 expert tensors.",
    )
    parser.add_argument(
        "expert_dir", type=Path, help="directory holding the D4 manifest and artifacts"
    )
    parser.add_argument(
        "output_dir", type=Path, help="directory for the calibration or failure record"
    )
    parser.add_argument(
        "--representation",
        action="append",
        default=[],
        metavar="EXPERT=REPRESENTATION",
        help="representation of one expert's raw values (logits or probabilities)",
    )
    parser.add_argument(
        "--temperature",
        action="append",
        default=[],
        metavar="EXPERT=TEMPERATURE",
        help="temperature of one expert (default 1.0); repeat per expert",
    )
    parser.add_argument("--version", default=DEFAULT_ARTIFACT_VERSION, help="D4 artifact version")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args(argv)

    try:
        representations = _parse_assignments(args.representation, flag="--representation")
        raw_temperatures = _parse_assignments(args.temperature, flag="--temperature")
        temperatures = {name: float(value) for name, value in raw_temperatures.items()}
    except (CalibrationRunError, ValueError) as error:
        parser.error(str(error))
    settings = CalibrationSettings()
    if args.max_iterations is not None or args.tolerance is not None:
        settings = replace(
            settings,
            max_iterations=(
                args.max_iterations if args.max_iterations is not None else settings.max_iterations
            ),
            tolerance=args.tolerance if args.tolerance is not None else settings.tolerance,
        )
    try:
        record = run_calibration(
            args.expert_dir,
            args.output_dir,
            representations=representations,
            temperatures=temperatures,
            settings=settings,
            artifact_version=args.version,
        )
    except CalibrationRunError as error:
        print(f"D3 calibration failed: {error}", file=sys.stderr)
        return 2
    print(f"D3 calibration recorded: {args.output_dir / RECORD_NAME} weights={record['weights']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
