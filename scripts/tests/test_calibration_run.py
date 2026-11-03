"""D3 calibration-run tests (fusion gate D3).

The gate consumes only persisted D4 artifacts. A clean panel records
weights with the input manifest digest; a missing or incomplete
manifest is a recorded refusal; identical or constant expert reports
are recorded non-identifiable failures. No failure path produces
weights or substitutes the hand-weight baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import samurai.calibration_run as calibration_run_module
from samurai.calibration import CalibrationSettings
from samurai.calibration_run import (
    FAILURE_NAME,
    RECORD_NAME,
    CalibrationRunError,
    main,
    run_calibration,
)
from samurai.expert_persist import (
    MANIFEST_NAME,
    artifact_filename,
    persist_expert_tensors,
)

REPRESENTATIONS = {"sam31": "probabilities", "samurai": "probabilities", "muse": "probabilities"}

# (sensitivity, false positive rate) per expert in the synthetic panel.
SKILLS = {"sam31": (0.95, 0.05), "samurai": (0.88, 0.10), "muse": (0.80, 0.15)}


def _expert_probabilities(
    rng: np.random.Generator, latent: np.ndarray, sensitivity: float, fpr: float
) -> np.ndarray:
    draw = rng.random(latent.shape)
    vote = np.where(latent, draw < sensitivity, draw < fpr)
    return np.where(vote, 0.9, 0.1).astype(np.float64)


def _panel_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261028)
    latent = rng.random((4, 30, 40)) < 0.35
    return {
        name: _expert_probabilities(rng, latent, sensitivity, fpr)
        for name, (sensitivity, fpr) in SKILLS.items()
    }


def _persist_panel(tmp_path: Path) -> Path:
    expert_dir = tmp_path / "experts"
    persist_expert_tensors(expert_dir, _panel_tensors(), version="1")
    return expert_dir


def _read_failure(output_dir: Path) -> dict:
    return json.loads((output_dir / FAILURE_NAME).read_text(encoding="utf-8"))


def test_clean_calibration_records_weights_inputs_and_diagnostics(tmp_path):
    expert_dir = _persist_panel(tmp_path)
    output_dir = tmp_path / "calibration"
    record = run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)

    assert record["gate"] == "D3"
    assert record["status"] == "recorded"
    # Inputs: the manifest digest and per-artifact digests are stored.
    manifest_bytes = (expert_dir / MANIFEST_NAME).read_bytes()
    assert record["inputs"]["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    artifacts = {item["expert"]: item for item in record["inputs"]["artifacts"]}
    assert set(artifacts) == set(SKILLS)
    for item in artifacts.values():
        artifact_bytes = (expert_dir / item["file"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    # Configuration: representations and settings are stored.
    assert record["configuration"]["representations"] == REPRESENTATIONS
    assert record["configuration"]["settings"]["max_iterations"] >= 1
    # Weights: a distribution ordered by the panel's true skill.
    weights = record["weights"]
    assert set(weights) == set(SKILLS)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["sam31"] > weights["samurai"] > weights["muse"] > 0.0
    # Diagnostics and non-claims are stored.
    diagnostics = record["diagnostics"]
    assert diagnostics["converged"] is True
    metrics = {item["name"]: item for item in diagnostics["expert_metrics"]}
    assert all(metrics[name]["youden_j"] > 0.0 for name in SKILLS)
    assert any("panel agreement" in claim for claim in record["non_claims"])
    # The record on disk matches the returned record; no failure record.
    on_disk = json.loads((output_dir / RECORD_NAME).read_text(encoding="utf-8"))
    assert on_disk["weights"] == weights
    assert not (output_dir / FAILURE_NAME).exists()


def test_missing_manifest_is_a_recorded_refusal(tmp_path):
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="manifest not found"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["status"] == "failure"
    assert failure["stage"] == "manifest"
    assert failure["weights"] is None
    assert not (output_dir / RECORD_NAME).exists()
    # The console entry point reports the same refusal as a nonzero exit.
    cli_output = tmp_path / "cli-calibration"
    assert main([str(expert_dir), str(cli_output)]) == 2
    assert _read_failure(cli_output)["stage"] == "manifest"


def test_incomplete_manifest_is_a_recorded_refusal(tmp_path):
    expert_dir = tmp_path / "experts"
    tensors = _panel_tensors()
    persist_expert_tensors(
        expert_dir, {name: tensors[name] for name in ("sam31", "muse")}, version="1"
    )
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="at least three persisted experts"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "manifest"
    assert failure["inputs"]["manifest_sha256"] is not None
    assert failure["weights"] is None


def test_corrupt_artifact_is_a_recorded_refusal(tmp_path):
    expert_dir = _persist_panel(tmp_path)
    artifact = expert_dir / artifact_filename("muse", "1")
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 0xFF
    artifact.write_bytes(bytes(data))
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="sha256 mismatch"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    assert _read_failure(output_dir)["stage"] == "manifest"


def test_undeclared_representation_is_a_recorded_refusal(tmp_path):
    expert_dir = _persist_panel(tmp_path)
    output_dir = tmp_path / "calibration"
    partial = {"sam31": "probabilities", "samurai": "probabilities"}
    with pytest.raises(CalibrationRunError, match="representation is not declared"):
        run_calibration(expert_dir, output_dir, representations=partial)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "inputs"
    assert failure["weights"] is None


def test_identical_reports_are_a_recorded_non_identifiable_failure(tmp_path):
    expert_dir = tmp_path / "experts"
    shared = _panel_tensors()["sam31"]
    persist_expert_tensors(expert_dir, {name: shared.copy() for name in SKILLS}, version="1")
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="not separately identifiable"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "identifiability"
    assert failure["weights"] is None
    assert not (output_dir / RECORD_NAME).exists()


def test_nonpositive_temperature_is_a_recorded_input_failure(tmp_path):
    """A nonpositive temperature is rejected before a tensor is read.

    Expert temperatures transform logits into probabilities. Zero and
    negative values are not invertible probability transforms and cannot
    produce a calibration record that any governed run can consume.
    """
    expert_dir = _persist_panel(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(CalibrationRunError, match="temperature must be a finite positive number"):
        run_calibration(
            expert_dir,
            out,
            representations=REPRESENTATIONS,
            temperatures={name: -1.0 for name in REPRESENTATIONS},
        )
    failure = json.loads((out / FAILURE_NAME).read_text())
    assert failure["stage"] == "inputs"
    assert not (out / RECORD_NAME).exists()


def test_nonfinite_temperature_is_a_recorded_input_failure(tmp_path):
    """NaN and infinite temperatures fail closed before reading tensors."""
    expert_dir = _persist_panel(tmp_path)
    for bad in (float("nan"), float("inf")):
        out = tmp_path / f"out-{bad}"
        with pytest.raises(
            CalibrationRunError, match="temperature must be a finite positive number"
        ):
            run_calibration(
                expert_dir,
                out,
                representations=REPRESENTATIONS,
                temperatures={name: bad for name in REPRESENTATIONS},
            )
        failure = json.loads((out / FAILURE_NAME).read_text())
        assert failure["stage"] == "inputs"
        assert not (out / RECORD_NAME).exists()


def test_constant_report_is_a_recorded_non_identifiable_failure(tmp_path):
    expert_dir = tmp_path / "experts"
    tensors = _panel_tensors()
    tensors["muse"] = np.full(tensors["muse"].shape, 0.1, dtype=np.float64)
    persist_expert_tensors(expert_dir, tensors, version="1")
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="constant"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "identifiability"
    assert failure["weights"] is None


def test_mismatched_shapes_across_experts_are_a_recorded_refusal(tmp_path):
    """Design 5.5: two shapes in one panel fail closed at the gate,
    and the failure record names both shapes in the conflict."""
    expert_dir = tmp_path / "experts"
    tensors = _panel_tensors()
    tensors["muse"] = tensors["muse"][:, :, :-1]
    persist_expert_tensors(expert_dir, tensors, version="1")
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="not aligned"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "manifest"
    assert failure["weights"] is None
    assert "(4, 30, 39)" in failure["reason"]
    assert "(4, 30, 40)" in failure["reason"]
    assert not (output_dir / RECORD_NAME).exists()


def test_atomic_write_failure_leaves_no_temp_or_partial_output(tmp_path, monkeypatch):
    """Design 5.20: when the record rename fails mid-write, the
    output directory holds no temp file and no partial record."""
    expert_dir = _persist_panel(tmp_path)
    output_dir = tmp_path / "calibration"
    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).endswith(RECORD_NAME):
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", failing_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    assert not (output_dir / RECORD_NAME).exists()
    assert not (output_dir / FAILURE_NAME).exists()
    assert not list(output_dir.glob(".tmp-*"))
    assert not list(output_dir.iterdir())


def _edit_manifest_record(expert_dir: Path, expert: str, **changes) -> None:
    """Rewrite one D4 manifest record in place; the artifact file is untouched."""
    manifest_path = expert_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        if item["expert"] == expert:
            item.update(changes)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_shape_mismatch_with_manifest_is_a_recorded_refusal(tmp_path):
    """Design 5.3: a manifest shape edit that differs from the .npy
    header fails closed at verification, before any sampling or fit."""
    expert_dir = _persist_panel(tmp_path)
    _edit_manifest_record(expert_dir, "muse", shape=[4, 30, 41])
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="shape/dtype mismatch"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "manifest"
    assert failure["weights"] is None
    assert "muse" in failure["reason"]
    assert not (output_dir / RECORD_NAME).exists()


def test_dtype_mismatch_with_manifest_is_a_recorded_refusal(tmp_path):
    """Design 5.3: a manifest dtype edit that differs from the .npy
    header fails closed at verification, before any sampling or fit."""
    expert_dir = _persist_panel(tmp_path)
    _edit_manifest_record(expert_dir, "muse", dtype="float32")
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="shape/dtype mismatch"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "manifest"
    assert failure["weights"] is None
    assert "muse" in failure["reason"]
    assert not (output_dir / RECORD_NAME).exists()


def test_deterministic_same_inputs_same_weights(tmp_path):
    """Design 5.9: two gate runs over one persisted panel with the
    same settings give identical weights, iterations, per-expert
    metrics, and manifest input hashes, with exact float equality."""
    expert_dir = _persist_panel(tmp_path)
    first = run_calibration(expert_dir, tmp_path / "calibration-1", representations=REPRESENTATIONS)
    second = run_calibration(
        expert_dir, tmp_path / "calibration-2", representations=REPRESENTATIONS
    )
    assert first["weights"] == second["weights"]
    assert first["diagnostics"]["iterations"] == second["diagnostics"]["iterations"]
    assert first["diagnostics"]["expert_metrics"] == second["diagnostics"]["expert_metrics"]
    assert first["diagnostics"]["samples"] == second["diagnostics"]["samples"]
    assert first["inputs"]["manifest_sha256"] == second["inputs"]["manifest_sha256"]
    assert first["inputs"]["artifacts"] == second["inputs"]["artifacts"]
    on_disk_first = json.loads(
        (tmp_path / "calibration-1" / RECORD_NAME).read_text(encoding="utf-8")
    )
    on_disk_second = json.loads(
        (tmp_path / "calibration-2" / RECORD_NAME).read_text(encoding="utf-8")
    )
    assert on_disk_first["weights"] == on_disk_second["weights"]


def test_bounded_sample_is_recorded_and_repeatable_at_the_gate(tmp_path):
    """Design 5.12 (gate half): with more pixels than max_samples, the
    record reports samples == max_samples, and two runs over the same
    persisted panel produce identical weights and expert metrics."""
    expert_dir = _persist_panel(tmp_path)
    settings = CalibrationSettings(max_samples=500)
    first = run_calibration(
        expert_dir, tmp_path / "calibration-a", representations=REPRESENTATIONS, settings=settings
    )
    second = run_calibration(
        expert_dir, tmp_path / "calibration-b", representations=REPRESENTATIONS, settings=settings
    )
    assert 4 * 30 * 40 > 500
    assert first["configuration"]["settings"]["max_samples"] == 500
    assert first["diagnostics"]["samples"] == 500
    assert second["diagnostics"]["samples"] == 500
    assert first["weights"] == second["weights"]
    assert first["diagnostics"]["expert_metrics"] == second["diagnostics"]["expert_metrics"]


def test_exact_half_report_is_a_recorded_identifiability_refusal(tmp_path):
    """Design 5.13 (gate half): an expert that reports exactly 0.5 at
    every pixel abstains everywhere; the gate records an identifiability
    failure and produces no weights."""
    expert_dir = tmp_path / "experts"
    tensors = _panel_tensors()
    tensors["muse"] = np.full(tensors["muse"].shape, 0.5, dtype=np.float64)
    persist_expert_tensors(expert_dir, tensors, version="1")
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match=r"constant at the 0\.5 vote threshold"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "identifiability"
    assert failure["weights"] is None
    assert not (output_dir / RECORD_NAME).exists()


def test_below_threshold_perfect_report_earns_high_weight_at_the_gate(tmp_path):
    """Design 5.14: an expert that reports 0.0 on true-negative pixels
    and 1.0 on true-positive pixels counts its below-threshold reports
    as negative evidence: it earns the largest weight and a near-zero
    false-positive rate. Dropping below-threshold reports would leave
    its false-positive rate unmeasured."""
    rng = np.random.default_rng(20261028)
    latent = rng.random((4, 30, 40)) < 0.35
    tensors = {
        name: _expert_probabilities(rng, latent, sensitivity, fpr)
        for name, (sensitivity, fpr) in SKILLS.items()
    }
    tensors["muse"] = np.where(latent, 1.0, 0.0).astype(np.float64)
    expert_dir = tmp_path / "experts"
    persist_expert_tensors(expert_dir, tensors, version="1")
    output_dir = tmp_path / "calibration"
    record = run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    assert record["status"] == "recorded"
    weights = record["weights"]
    assert weights["muse"] > weights["sam31"]
    assert weights["muse"] > weights["samurai"]
    metrics = {item["name"]: item for item in record["diagnostics"]["expert_metrics"]}
    assert metrics["muse"]["sensitivity"] > 0.99
    assert metrics["muse"]["false_positive_rate"] < 0.01


def test_record_hashes_and_inputs_verify_round_trip(tmp_path):
    """Design 5.18, branch-form half: after a recorded run, independent
    SHA-256 computations match every digest the record carries, the
    record's manifest digest equals the D4 manifest file bytes, each
    stored artifact record equals the D4 record on every field the
    branch record carries, and the record file parses and carries the
    full branch record field set.

    The design form (section 4.2 weights file, section 4.3 output
    manifest, artifact records copied verbatim including bytes) does
    not exist in the branch form; that half needs a production change
    and stays open (see the harness skip reason).
    """
    expert_dir = _persist_panel(tmp_path)
    output_dir = tmp_path / "calibration"
    record = run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)

    # The record file parses and equals the returned record.
    on_disk = json.loads((output_dir / RECORD_NAME).read_text(encoding="utf-8"))
    assert on_disk == record

    # Branch record field set (the section 4.2 analogue in branch form).
    assert set(record) == {
        "record_version",
        "gate",
        "status",
        "method",
        "labels",
        "recorded_at",
        "inputs",
        "configuration",
        "weights",
        "diagnostics",
        "non_claims",
    }
    assert set(record["inputs"]) == {
        "expert_dir",
        "manifest_file",
        "manifest_sha256",
        "artifact_version",
        "shape",
        "artifacts",
    }
    assert set(record["configuration"]) == {"representations", "temperatures", "settings"}
    assert set(record["diagnostics"]) == {
        "iterations",
        "converged",
        "relabelled",
        "samples",
        "latent_prior",
        "latent_positive_fraction",
        "latent_selected_fraction",
        "anchor",
        "anchor_agreement",
        "expert_metrics",
        "pairwise",
    }

    # The manifest digest equals the hash of the D4 manifest file bytes.
    manifest_bytes = (expert_dir / MANIFEST_NAME).read_bytes()
    assert record["inputs"]["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    d4_records = {item["expert"]: item for item in json.loads(manifest_bytes)["artifacts"]}

    # Each stored artifact record: independent hash of the artifact file
    # bytes, and verbatim equality on every carried field.
    carried = ("expert", "version", "file", "shape", "dtype", "sha256")
    artifacts = {item["expert"]: item for item in record["inputs"]["artifacts"]}
    assert set(artifacts) == set(d4_records)
    for expert, item in artifacts.items():
        assert set(item) == set(carried)
        artifact_bytes = (expert_dir / item["file"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
        for field in carried:
            assert item[field] == d4_records[expert][field]
    assert sum(record["weights"].values()) == pytest.approx(1.0)


def test_verify_manifest_failure_record_preserves_manifest_sha256(tmp_path):
    """R2-F6: when verify_manifest fails on an existing manifest, the
    failure record keeps the SHA-256 computed from that manifest file;
    the old code hard-coded manifest_sha256=None on this one path."""
    expert_dir = _persist_panel(tmp_path)
    artifact = expert_dir / artifact_filename("muse", "1")
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 0xFF
    artifact.write_bytes(bytes(data))
    expected_sha256 = hashlib.sha256((expert_dir / MANIFEST_NAME).read_bytes()).hexdigest()
    output_dir = tmp_path / "calibration"
    with pytest.raises(CalibrationRunError, match="sha256 mismatch"):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    failure = _read_failure(output_dir)
    assert failure["stage"] == "manifest"
    assert failure["inputs"]["manifest_sha256"] == expected_sha256
    assert failure["weights"] is None


def test_failed_calibration_invalidates_the_prior_success_record(tmp_path):
    """Finding 1: a failed calibration removes the prior success record,
    so no consumer can read weights the failing run no longer stands
    behind. A consumer that pins the prior digest is refused at
    record_exists (see test_weight_source.py)."""
    expert_dir = _persist_panel(tmp_path)
    output_dir = tmp_path / "calibration"
    run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    record_path = output_dir / RECORD_NAME
    assert record_path.is_file()
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()

    # The panel changes and the next calibration fails on it.
    tensors = _panel_tensors()
    tensors["muse"] = np.full(tensors["muse"].shape, 0.1, dtype=np.float64)
    persist_expert_tensors(expert_dir, tensors, version="1")
    with pytest.raises(CalibrationRunError):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)

    assert not record_path.exists()
    failure = _read_failure(output_dir)
    assert failure["status"] == "failure"
    assert failure["weights"] is None
    assert failure["invalidated_prior_record"]["invalidated"] is True
    assert failure["invalidated_prior_record"]["sha256"] == digest


def test_successful_calibration_leaves_no_stale_failure_record(tmp_path):
    """Finding 1 (other direction): the record and the failure record
    are mutually exclusive, so a successful run does not leave an older
    failure record describing it."""
    expert_dir = _persist_panel(tmp_path)
    output_dir = tmp_path / "calibration"
    tensors = _panel_tensors()
    tensors["muse"] = np.full(tensors["muse"].shape, 0.1, dtype=np.float64)
    persist_expert_tensors(expert_dir, tensors, version="1")
    with pytest.raises(CalibrationRunError):
        run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    assert (output_dir / FAILURE_NAME).is_file()

    persist_expert_tensors(expert_dir, _panel_tensors(), version="1")
    run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    assert (output_dir / RECORD_NAME).is_file()
    assert not (output_dir / FAILURE_NAME).exists()


def test_calibration_record_names_the_d4_manifest_file(tmp_path):
    """Finding 7: the D3 record names the manifest file the D4 layer
    owns, so a reader can find the bytes the digest covers."""
    expert_dir = _persist_panel(tmp_path)
    output_dir = tmp_path / "calibration"
    record = run_calibration(expert_dir, output_dir, representations=REPRESENTATIONS)
    assert record["inputs"]["manifest_file"] == MANIFEST_NAME
    assert (expert_dir / MANIFEST_NAME).is_file()


def test_unreadable_manifest_invalidates_prior_success_and_records_failure(tmp_path):
    expert_dir = _persist_panel(tmp_path)
    out = tmp_path / "out"
    run_calibration(expert_dir, out, representations=REPRESENTATIONS)
    assert (out / RECORD_NAME).exists()
    manifest = expert_dir / MANIFEST_NAME
    manifest.chmod(0)
    try:
        with pytest.raises(CalibrationRunError, match="Permission denied"):
            run_calibration(expert_dir, out, representations=REPRESENTATIONS)
    finally:
        manifest.chmod(0o644)
    failure = json.loads((out / FAILURE_NAME).read_text())
    assert failure["stage"] == "manifest"
    assert not (out / RECORD_NAME).exists()


def test_tensor_replacement_after_d4_verification_fails_closed(tmp_path, monkeypatch):
    expert_dir = _persist_panel(tmp_path)
    tensors = _panel_tensors()
    replaced = {**tensors, "sam31": tensors["samurai"].copy()}
    real_load = calibration_run_module.load_expert_tensor
    calls = 0

    def racing_load(directory, expert, *, version):
        nonlocal calls
        array = real_load(directory, expert, version=version)
        calls += 1
        if expert == "sam31":
            persist_expert_tensors(directory, replaced, version=version)
        return array

    monkeypatch.setattr(calibration_run_module, "load_expert_tensor", racing_load)
    with pytest.raises(CalibrationRunError, match="artifact changed after D4 verification"):
        run_calibration(expert_dir, tmp_path / "out", representations=REPRESENTATIONS)
    assert calls >= 2
    assert not (tmp_path / "out" / RECORD_NAME).exists()


def test_zero_length_aligned_panel_records_an_input_failure(tmp_path):
    expert_dir = tmp_path / "experts"
    persist_expert_tensors(
        expert_dir,
        {name: np.zeros((0, 4, 6), dtype=np.float64) for name in REPRESENTATIONS},
        version="1",
    )
    out = tmp_path / "out"
    with pytest.raises(CalibrationRunError, match="zero-length axis"):
        run_calibration(expert_dir, out, representations=REPRESENTATIONS)
    failure = json.loads((out / FAILURE_NAME).read_text())
    assert failure["stage"] == "inputs"
    assert not (out / RECORD_NAME).exists()
