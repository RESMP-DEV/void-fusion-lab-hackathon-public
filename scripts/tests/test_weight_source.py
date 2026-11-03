"""D1 weight-source loader tests (fusion gate D1).

The governed run takes its weights from a calibration record whose
SHA-256 is pinned in the run config. Every load check fails closed:
a refused source raises EnsembleError, writes no mask and no report,
and never falls back to the hand weights or the config weights. The
hand weights 0.40/0.35/0.25 run only as a declared comparison
baseline and are never labeled governed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from samurai import ensemble as ensemble_module
from samurai.ensemble import (
    EnsembleError,
    TensorExpertSource,
    load_config,
    resolve_weight_source,
    run_ensemble,
)
from samurai.expert_persist import MANIFEST_NAME, persist_expert_tensors
from samurai.media import write_gray_video

EXPERTS = ("a", "b", "c")
RECORD_WEIGHTS = {"a": 0.5, "b": 0.3, "c": 0.2}
HAND_WEIGHTS = {"a": 0.40, "b": 0.35, "c": 0.25}


def _panel_tensors() -> dict[str, np.ndarray]:
    """The deterministic three-expert panel: the same bytes every call."""
    rng = np.random.default_rng(20261028)
    return {name: rng.normal(0.0, 2.0, size=(2, 4, 6)) for name in EXPERTS}


def _panel(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Persist three tiny tensor experts; return (dir, name -> artifact path)."""
    expert_dir = tmp_path / "experts"
    persist_expert_tensors(
        expert_dir,
        _panel_tensors(),
        version="1",
        sources={name: {"path": f"fixture://{name}"} for name in EXPERTS},
    )
    return expert_dir, {name: expert_dir / f"{name}.v1.npy" for name in EXPERTS}


def _record(expert_dir: Path, *, artifact_version: str = "1") -> dict[str, Any]:
    """A governed D3 record over the persisted panel.

    The shape mirrors what ``samurai.calibration_run`` writes: the
    manifest digest, the artifact version, and one entry per
    calibrated artifact naming the calibrated expert, the artifact
    file, the artifact version, and the artifact digest.
    """
    manifest_path = expert_dir / MANIFEST_NAME
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = [
        {
            "expert": str(item["expert"]),
            "version": str(item["version"]),
            "file": str(item["file"]),
            "shape": [int(axis) for axis in item["shape"]],
            "dtype": str(item["dtype"]),
            "sha256": str(item["sha256"]),
        }
        for item in manifest["artifacts"]
        if str(item["version"]) == artifact_version
    ]
    return {
        "record_version": 1,
        "gate": "D3",
        "status": "recorded",
        "inputs": {
            "manifest_sha256": manifest_sha256,
            "artifact_version": artifact_version,
            "artifacts": artifacts,
        },
        "configuration": {
            "representations": {name: "logits" for name in EXPERTS},
            "temperatures": {name: 1.0 for name in EXPERTS},
        },
        "weights": dict(RECORD_WEIGHTS),
    }


def _config(
    tmp_path: Path,
    paths: dict[str, Path],
    weight_source: dict[str, Any],
    *,
    experts: tuple[str, ...] = EXPERTS,
    representations: dict[str, str] | None = None,
    temperatures: dict[str, float] | None = None,
    min_expert_support: int = 3,
) -> Path:
    video = tmp_path / "input.mp4"
    if not video.exists():
        write_gray_video(
            [np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12
        )
    representations = representations or {}
    temperatures = temperatures or {}
    payload = {
        "input_video": str(video),
        "experts": [
            {
                "name": name,
                "path": str(paths[name]),
                "representation": representations.get(name, "logits"),
                "weight": 1.0,
                "temperature": temperatures.get(name, 1.0),
            }
            for name in experts
        ],
        "fusion": {
            "mode": "threshold",
            "threshold": 0.5,
            "min_expert_support": min_expert_support,
            "support_threshold": 0.5,
            "dynamic_weights": "static",
        },
        "weight_source": weight_source,
    }
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _governed_config(
    tmp_path: Path,
    *,
    mutate_record: Callable[[dict[str, Any]], None] | None = None,
    digest_override: str | None = None,
    record_name: str = "calibration_record.json",
    **config_kwargs: Any,
) -> Path:
    expert_dir, paths = _panel(tmp_path)
    record = _record(expert_dir)
    if mutate_record is not None:
        mutate_record(record)
    record_path = tmp_path / record_name
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    source = {
        "type": "calibration_record",
        "path": str(record_path),
        "sha256": digest_override or digest,
    }
    return _config(tmp_path, paths, source, **config_kwargs)


def _assert_refused(tmp_path: Path, config_path: Path, match: str) -> None:
    with pytest.raises(EnsembleError, match=match):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_calibration_weights_enter_run(tmp_path):
    expert_dir, paths = _panel(tmp_path)
    record_path = tmp_path / "calibration_record.json"
    record = _record(expert_dir)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    source = {"type": "calibration_record", "path": str(record_path), "sha256": digest}
    config_path = _config(tmp_path, paths, source)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    applied = result["weight_source"]
    assert applied["type"] == "calibration_record"
    assert applied["governed"] is True
    assert applied["weights"] == RECORD_WEIGHTS
    assert applied["sha256"] == digest
    manifest_sha256 = hashlib.sha256((expert_dir / MANIFEST_NAME).read_bytes()).hexdigest()
    assert applied["manifest_sha256"] == manifest_sha256
    assert result["frames_report"][0]["weight_a"] == pytest.approx(0.5)
    assert result["frames_report"][0]["weight_b"] == pytest.approx(0.3)
    assert result["frames_report"][0]["weight_c"] == pytest.approx(0.2)
    assert (tmp_path / "mask.mp4").is_file()
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["weight_source"]["weights"] == RECORD_WEIGHTS


def test_weight_source_digest_mismatch_refused(tmp_path):
    config_path = _governed_config(tmp_path, digest_override="0" * 64)
    _assert_refused(tmp_path, config_path, "record_digest")


def test_missing_calibration_record_refused(tmp_path):
    _expert_dir, paths = _panel(tmp_path)
    source = {
        "type": "calibration_record",
        "path": str(tmp_path / "no_such_record.json"),
        "sha256": "0" * 64,
    }
    config_path = _config(tmp_path, paths, source)
    _assert_refused(tmp_path, config_path, "record_exists")


def test_failure_record_is_not_a_weight_source(tmp_path):
    def fail(record: dict[str, Any]) -> None:
        record["status"] = "failure"
        record["weights"] = None

    config_path = _governed_config(
        tmp_path, mutate_record=fail, record_name="calibration_failure.json"
    )
    _assert_refused(tmp_path, config_path, "record_shape")


def test_record_expert_set_mismatch_refused(tmp_path):
    _expert_dir, paths = _panel(tmp_path)
    paths["d"] = tmp_path / "experts" / "d.v1.npy"
    record = _record(_expert_dir)
    record_path = tmp_path / "calibration_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    source = {"type": "calibration_record", "path": str(record_path), "sha256": digest}
    config_path = _config(tmp_path, paths, source, experts=("a", "b", "d"))
    with pytest.raises(EnsembleError, match="expert_set") as excinfo:
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert "'d'" in str(excinfo.value) and "'c'" in str(excinfo.value)
    assert not (tmp_path / "mask.mp4").exists()


def _command_config(tmp_path: Path, weight_source: dict[str, Any]) -> Path:
    """Config whose panel is three command experts and no tensor experts."""
    video = tmp_path / "input.mp4"
    if not video.exists():
        write_gray_video(
            [np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12
        )
    payload = {
        "input_video": str(video),
        "experts": [
            {
                "name": name,
                "kind": "command",
                "command": ["fixture-rasterizer", name],
                "weight": 1.0,
                "temperature": 1.0,
            }
            for name in EXPERTS
        ],
        "fusion": {
            "mode": "threshold",
            "threshold": 0.5,
            "min_expert_support": 3,
            "support_threshold": 0.5,
            "dynamic_weights": "static",
        },
        "weight_source": weight_source,
    }
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_command_only_panel_refused_at_manifest_check(tmp_path):
    # The record passes checks 1-5: it is the governed fixture record
    # (pinned digest, D3 shape, exact expert set, valid distribution).
    # Check 6 binds the record to the persisted tensor set the run
    # consumes; a command-only panel consumes none, so the binding
    # cannot be proven and the run is refused before any output.
    expert_dir, _paths = _panel(tmp_path)
    record = _record(expert_dir)
    record_path = tmp_path / "calibration_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    source = {"type": "calibration_record", "path": str(record_path), "sha256": digest}
    config_path = _command_config(tmp_path, source)
    with pytest.raises(EnsembleError, match="check manifest") as excinfo:
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert "consumes no persisted tensor set" in str(excinfo.value)
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_record_manifest_mismatch_refused(tmp_path):
    def corrupt(record: dict[str, Any]) -> None:
        record["inputs"]["manifest_sha256"] = "0" * 64

    config_path = _governed_config(tmp_path, mutate_record=corrupt)
    _assert_refused(tmp_path, config_path, "check manifest")


def test_transform_mismatch_refused(tmp_path):
    representation_dir = tmp_path / "representation"
    representation_dir.mkdir()
    config_path = _governed_config(
        representation_dir, representations={"a": "probabilities", "b": "logits", "c": "logits"}
    )
    _assert_refused(representation_dir, config_path, "check transform")

    temperature_dir = tmp_path / "temperature"
    temperature_dir.mkdir()
    config_path = _governed_config(temperature_dir, temperatures={"a": 2.0, "b": 1.0, "c": 1.0})
    _assert_refused(temperature_dir, config_path, "check transform")


def test_invalid_weight_distribution_refused(tmp_path):
    def shrink(record: dict[str, Any]) -> None:
        record["weights"] = {"a": 0.5, "b": 0.3, "c": 0.1}

    config_path = _governed_config(tmp_path, mutate_record=shrink)
    _assert_refused(tmp_path, config_path, "weight_distribution")


def test_hand_baseline_declared_not_governed(tmp_path):
    _expert_dir, paths = _panel(tmp_path)
    source = {"type": "hand_baseline", "weights": dict(HAND_WEIGHTS)}
    config_path = _config(tmp_path, paths, source, min_expert_support=1)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    applied = result["weight_source"]
    assert applied["type"] == "hand_baseline"
    assert applied["governed"] is False
    assert applied["weights"] == HAND_WEIGHTS
    assert "baseline" in applied["non_claim"]
    assert result["frames_report"][0]["weight_a"] == pytest.approx(0.40)
    assert (tmp_path / "mask.mp4").is_file()


def _record_source(tmp_path: Path, expert_dir: Path) -> dict[str, Any]:
    """Write the governed fixture record; return its weight-source declaration."""
    record = _record(expert_dir)
    record_path = tmp_path / "calibration_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    return {
        "type": "calibration_record",
        "path": str(record_path),
        "sha256": digest,
    }


def test_governed_run_refuses_a_consumed_tensor_that_does_not_match_the_manifest(tmp_path):
    """Finding 2: a governed run verifies the D4 manifest and every
    consumed tensor before fusion. A tensor whose bytes no longer match
    the manifest record is refused, even when the calibration record
    itself is intact and pinned."""
    expert_dir, paths = _panel(tmp_path)
    source = _record_source(tmp_path, expert_dir)
    config_path = _config(tmp_path, paths, source)

    artifact = paths["a"]
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 0xFF
    artifact.write_bytes(bytes(data))

    with pytest.raises(EnsembleError, match="check manifest") as excinfo:
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert "does not verify" in str(excinfo.value)
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_governed_run_reverifies_after_d1_before_mapping_tensors(tmp_path, monkeypatch):
    """Round-5 concurrency finding: replacement after D1 must fail.

    The injected replacement happens after the real D1 resolver succeeds but
    before run_ensemble opens its memory mappings. The pre-repair path opened
    the replacement silently; the repaired path re-verifies under the D4 lock
    and refuses the manifest mismatch before writing output.
    """
    expert_dir, paths = _panel(tmp_path)
    source = _record_source(tmp_path, expert_dir)
    config_path = _config(tmp_path, paths, source)
    real_resolve_weight_source = resolve_weight_source

    def replace_after_d1(config):
        resolved = real_resolve_weight_source(config)
        staged = expert_dir / ".replacement.npy"
        np.save(staged, np.ones((2, 4, 6), dtype=np.float64))
        os.replace(staged, paths["a"])
        return resolved

    monkeypatch.setattr(ensemble_module, "resolve_weight_source", replace_after_d1)
    with pytest.raises(EnsembleError, match="does not verify") as excinfo:
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert "governed source acquisition after D1 resolution" in str(excinfo.value)
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_governed_tensor_sources_open_inside_the_d4_manifest_lock(tmp_path, monkeypatch):
    """Round-5 concurrency finding: verification and mmap creation share one
    protected region. A source constructor may not run before the D4 lock is
    entered or after it is released."""
    config_path = _governed_config(tmp_path)
    events: list[str] = []

    @contextmanager
    def recording_lock(directory):
        events.append("lock-enter")
        yield
        events.append("lock-exit")

    real_tensor_init = TensorExpertSource.__init__

    def recording_tensor_init(source, spec):
        events.append("tensor-enter")
        real_tensor_init(source, spec)
        events.append("tensor-exit")

    monkeypatch.setattr(ensemble_module, "manifest_lock", recording_lock)
    monkeypatch.setattr(TensorExpertSource, "__init__", recording_tensor_init)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["weight_source"]["governed"] is True
    assert events[0] == "lock-enter"
    assert events[-1] == "lock-exit"
    assert events.count("tensor-enter") == len(EXPERTS)
    assert all(
        event in {"lock-enter", "tensor-enter", "tensor-exit", "lock-exit"} for event in events
    )
    assert events.index("tensor-enter") > events.index("lock-enter")
    assert events.index("tensor-exit") < events.index("lock-exit")


def test_governed_run_refuses_a_tensor_the_manifest_does_not_list(tmp_path):
    """Finding 2: a consumed tensor the D4 manifest does not list is
    refused; the run cannot prove which bytes it fused."""
    expert_dir, paths = _panel(tmp_path)
    source = _record_source(tmp_path, expert_dir)
    unlisted = expert_dir / "unlisted.v1.npy"
    np.save(unlisted, np.zeros((2, 4, 6)))
    paths = dict(paths)
    paths["a"] = unlisted
    config_path = _config(tmp_path, paths, source)
    with pytest.raises(EnsembleError, match="check manifest") as excinfo:
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert "is not listed in the" in str(excinfo.value)
    assert not (tmp_path / "mask.mp4").exists()


def test_identical_repersist_keeps_the_calibration_binding(tmp_path):
    """Finding 3: an identical D4 re-persist does not mutate the pinned
    manifest bytes, so the calibration record stays bound to the tensors
    the run consumes and the governed run still applies its weights."""
    expert_dir, paths = _panel(tmp_path)
    source = _record_source(tmp_path, expert_dir)
    config_path = _config(tmp_path, paths, source)

    persist_expert_tensors(
        expert_dir,
        _panel_tensors(),
        version="1",
        sources={name: {"path": f"fixture://{name}"} for name in EXPERTS},
    )

    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["weight_source"]["governed"] is True
    assert result["weight_source"]["weights"] == RECORD_WEIGHTS
    assert result["weight_source"]["manifest_sha256"] == (
        hashlib.sha256((expert_dir / MANIFEST_NAME).read_bytes()).hexdigest()
    )


def test_changed_tensors_invalidate_the_calibration_binding(tmp_path):
    """Finding 3: changed tensors legitimately invalidate the
    calibration. The manifest digest moves, so the governed run is
    refused instead of applying weights fitted to other tensors."""
    expert_dir, paths = _panel(tmp_path)
    source = _record_source(tmp_path, expert_dir)
    config_path = _config(tmp_path, paths, source)

    changed = _panel_tensors()
    changed["a"] = changed["a"] + 1.0
    persist_expert_tensors(
        expert_dir,
        changed,
        version="1",
        sources={name: {"path": f"fixture://{name}"} for name in EXPERTS},
    )

    with pytest.raises(EnsembleError, match="check manifest"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def _panel_with_command_expert(
    tmp_path: Path, paths: dict[str, Path], weight_source: dict[str, Any]
) -> Path:
    """Config whose panel is two tensor experts and one command expert."""
    video = tmp_path / "input.mp4"
    if not video.exists():
        write_gray_video(
            [np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12
        )
    payload = {
        "input_video": str(video),
        "experts": [
            {"name": "a", "path": str(paths["a"]), "weight": 1.0, "temperature": 1.0},
            {"name": "b", "path": str(paths["b"]), "weight": 1.0, "temperature": 1.0},
            {
                "name": "c",
                "kind": "command",
                "command": ["fixture-rasterizer", "c"],
                "weight": 1.0,
                "temperature": 1.0,
            },
        ],
        "fusion": {
            "mode": "threshold",
            "threshold": 0.5,
            "min_expert_support": 3,
            "support_threshold": 0.5,
            "dynamic_weights": "static",
        },
        "weight_source": weight_source,
    }
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_governed_run_refuses_a_command_expert_in_the_panel(tmp_path):
    """Round-2 finding 1: a governed run fails closed when any
    configured expert is a command expert. A command expert consumes
    values the run itself produces, so no pre-run D4 artifact binds
    them: the calibration cannot be shown to describe the panel the
    run fuses. D1 is not weakened to skip command experts."""
    expert_dir, paths = _panel(tmp_path)
    source = _record_source(tmp_path, expert_dir)
    config_path = _panel_with_command_expert(tmp_path, paths, source)
    with pytest.raises(EnsembleError, match="check command_expert_binding") as excinfo:
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    message = str(excinfo.value)
    assert "c" in message
    assert "no pre-run D4 artifact binding" in message
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_governed_run_refuses_an_artifact_the_record_calibrated_at_another_version(tmp_path):
    """Round-2 finding 2: the record binds each run expert to the
    calibrated expert name and the exact D4 artifact file and version.
    Re-persisting one expert at a second version keeps the manifest,
    so the manifest digest still matches; only the per-expert
    artifact binding shows that the calibration fitted the version 1
    artifact while the run would consume version 2."""
    expert_dir, paths = _panel(tmp_path)
    changed = _panel_tensors()
    changed["a"] = changed["a"] + 1.0
    persist_expert_tensors(
        expert_dir,
        {"a": changed["a"]},
        version="2",
        sources={"a": {"path": "fixture://a"}},
    )
    # The calibration ran at version 1 over the current manifest, so
    # it lists the version 1 artifacts only.
    record = _record(expert_dir, artifact_version="1")
    record_path = tmp_path / "calibration_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    source = {"type": "calibration_record", "path": str(record_path), "sha256": digest}
    consumed = dict(paths)
    consumed["a"] = expert_dir / "a.v2.npy"
    config_path = _config(tmp_path, consumed, source)
    with pytest.raises(EnsembleError, match="check artifact_binding") as excinfo:
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    message = str(excinfo.value)
    assert "a.v2.npy" in message
    assert "a.v1.npy" in message
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_governed_run_accepts_the_artifact_the_record_calibrated(tmp_path):
    """Round-2 finding 2, positive control: the same record binds the
    version 1 artifacts, so a run that consumes those exact files
    stays governed."""
    expert_dir, paths = _panel(tmp_path)
    changed = _panel_tensors()
    changed["a"] = changed["a"] + 1.0
    persist_expert_tensors(
        expert_dir,
        {"a": changed["a"]},
        version="2",
        sources={"a": {"path": "fixture://a"}},
    )
    record = _record(expert_dir, artifact_version="1")
    record_path = tmp_path / "calibration_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    source = {"type": "calibration_record", "path": str(record_path), "sha256": digest}
    config_path = _config(tmp_path, paths, source)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["weight_source"]["governed"] is True
    assert result["weight_source"]["artifact_bindings"]["a"]["file"] == "a.v1.npy"
    assert result["weight_source"]["artifact_bindings"]["a"]["version"] == "1"


def test_governed_run_refuses_to_persist_into_its_consumed_d4_directory(tmp_path):
    """A derived governed run must not overwrite the artifact directory whose
    bytes its calibration record pinned. Persisting float16 derivatives there
    would replace the calibrated artifacts and self-invalidate the binding."""
    config_path = _governed_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    consumed_dir = Path(payload["experts"][0]["path"]).parent
    payload["expert_persist_dir"] = str(consumed_dir)
    payload["expert_persist_mode"] = "opt-in"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EnsembleError, match="overwrite a consumed D4 artifact directory"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_governed_run_refuses_a_record_without_artifact_bindings(tmp_path):
    """Round-2 finding 2: a record that pins only the manifest digest
    cannot prove which bytes the calibration fitted, so it is
    refused instead of trusted."""

    def strip(record: dict[str, Any]) -> None:
        record["inputs"]["artifacts"] = []

    config_path = _governed_config(tmp_path, mutate_record=strip)
    _assert_refused(tmp_path, config_path, "check artifact_binding")


def test_governed_run_refuses_a_record_whose_artifact_names_another_expert(tmp_path):
    """Round-2 finding 2: the record binds the calibrated expert name.
    An artifact entry that names a different expert does not bind the
    run's expert."""

    def rename(record: dict[str, Any]) -> None:
        for item in record["inputs"]["artifacts"]:
            if item["expert"] == "a":
                item["expert"] = "alpha"

    config_path = _governed_config(tmp_path, mutate_record=rename)
    _assert_refused(tmp_path, config_path, "check artifact_binding")
