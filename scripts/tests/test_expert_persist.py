"""D4 expert-tensor persistence tests (fusion gate D4).

Round trip: tensor in -> versioned .npy artifact -> manifest
SHA-256 verifies against an independent hashlib computation ->
tensor loaded back is byte-equal to the input. A corrupted file
must fail the manifest hash check.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from samurai.ensemble import (
    CommandExpertSource,
    EnsembleError,
    ExpertSpec,
    load_config,
    run_ensemble,
)
from samurai.expert_persist import (
    MANIFEST_LOCK_NAME,
    MANIFEST_NAME,
    ExpertPersistError,
    artifact_filename,
    load_expert_tensor,
    persist_expert_tensor,
    persist_expert_tensors,
    read_manifest,
    verify_manifest,
)
from samurai.media import write_gray_video
from samurai.persist import MANIFEST_LOCK_NAME as LEGACY_MANIFEST_LOCK_NAME
from samurai.persist import MANIFEST_NAME as LEGACY_MANIFEST_NAME

PROV_TENSORS = {
    "sam31": np.arange(24, dtype=np.float16).reshape(2, 3, 4) / 7.0,
    "samurai": np.full((2, 3, 4), 0.25, dtype=np.float32),
    "muse": np.linspace(0.0, 1.0, 24, dtype=np.float64).reshape(2, 3, 4),
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_trip_byte_equal_and_manifest_sha_verifies(tmp_path):
    manifest = persist_expert_tensors(tmp_path, PROV_TENSORS, version="1")
    assert manifest["manifest_version"] == 1
    records = {item["expert"]: item for item in manifest["artifacts"]}
    assert set(records) == set(PROV_TENSORS)
    for expert, tensor in PROV_TENSORS.items():
        record = records[expert]
        # Version is in the filename and in the manifest record.
        assert record["version"] == "1"
        assert record["file"] == f"{expert}.v1.npy"
        artifact = tmp_path / record["file"]
        assert artifact.is_file()
        # Manifest SHA-256 verifies against independent hashlib.
        assert record["sha256"] == _file_sha256(artifact)
        assert record["shape"] == list(tensor.shape)
        assert record["dtype"] == str(tensor.dtype)
        # Loaded back after verification: byte-equal to the input.
        loaded = load_expert_tensor(tmp_path, expert, version="1")
        assert loaded.shape == tensor.shape
        assert loaded.dtype == tensor.dtype
        assert loaded.tobytes() == tensor.tobytes()
    # The manifest-level verifier agrees.
    assert verify_manifest(tmp_path)["artifacts"]


def test_corrupted_file_fails_manifest_hash_check(tmp_path):
    persist_expert_tensor(tmp_path, "muse", PROV_TENSORS["muse"], version="1")
    artifact = tmp_path / artifact_filename("muse", "1")
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 0xFF
    artifact.write_bytes(bytes(data))
    with pytest.raises(ExpertPersistError, match="sha256 mismatch"):
        verify_manifest(tmp_path)
    with pytest.raises(ExpertPersistError, match="sha256 mismatch"):
        load_expert_tensor(tmp_path, "muse", version="1")


def test_versions_coexist_and_no_temp_files_remain(tmp_path):
    first = np.zeros((1, 2), dtype=np.float32)
    second = np.ones((1, 2), dtype=np.float32)
    persist_expert_tensor(tmp_path, "sam31", first, version="1")
    persist_expert_tensor(tmp_path, "sam31", second, version="2")
    assert (tmp_path / "sam31.v1.npy").is_file()
    assert (tmp_path / "sam31.v2.npy").is_file()
    assert load_expert_tensor(tmp_path, "sam31", version="1").tobytes() == first.tobytes()
    assert load_expert_tensor(tmp_path, "sam31", version="2").tobytes() == second.tobytes()
    assert not list(tmp_path.glob(".tmp-*"))
    records = read_manifest(tmp_path)["artifacts"]
    assert {(item["expert"], item["version"]) for item in records} == {
        ("sam31", "1"),
        ("sam31", "2"),
    }


def test_muse_rasterization_round_trip(tmp_path):
    """A Muse command expert's rasterized tensor persists and verifies."""
    image = tmp_path / "frame.png"
    from PIL import Image

    Image.new("RGB", (10, 8), "black").save(image)

    def runner(command, request):
        return {
            "regions_by_frame": [
                {
                    "frame_index": 0,
                    "regions": [
                        {
                            "type": "box",
                            "coordinates": [0.1, 0.25, 0.6, 0.75],
                            "confidence": 0.8,
                        }
                    ],
                }
            ]
        }

    source = CommandExpertSource(
        ExpertSpec(
            name="muse",
            weight=1,
            temperature=1,
            kind="command",
            command=("fake-expert",),
            representation="probabilities",
        ),
        [image],
        (1, 8, 10),
        runner=runner,
    )
    rasterized = source.read([0])
    persist_expert_tensor(tmp_path, "muse", rasterized, version="1")
    loaded = load_expert_tensor(tmp_path, "muse", version="1")
    assert loaded.tobytes() == rasterized.tobytes()
    assert verify_manifest(tmp_path)


def test_run_ensemble_persists_every_expert_including_command(tmp_path):
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    tensor_path = tmp_path / "sam31_input.npy"
    tensor_values = np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)])
    np.save(tensor_path, tensor_values)
    fake_expert = tmp_path / "fake_expert.py"
    fake_expert.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "frames = [{'frame_index': f['frame_index'], 'regions': "
        "[{'type': 'box', 'coordinates': [0.0, 0.0, 0.5, 0.5], 'confidence': 0.8}]} "
        "for f in request['frames']]\n"
        "print(json.dumps({'regions_by_frame': frames}))\n"
    )
    persist_dir = tmp_path / "experts"
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "sam31", "path": str(tensor_path), "weight": 1.0},
                    {
                        "name": "muse",
                        "kind": "command",
                        "command": [sys.executable, str(fake_expert)],
                        "weight": 1.0,
                    },
                ],
                "fusion": {"mode": "threshold", "threshold": 0.5, "batch_size": 1},
                "expert_persist_dir": str(persist_dir),
                "expert_artifact_version": "3",
            }
        )
    )
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["frames"] == 2
    manifest = verify_manifest(persist_dir)
    records = {item["expert"]: item for item in manifest["artifacts"]}
    assert set(records) == {"sam31", "muse"}
    for record in records.values():
        assert record["version"] == "3"
        assert record["file"].endswith(".v3.npy")
        assert record["sha256"] == _file_sha256(persist_dir / record["file"])
    # The tensor expert persisted in the float16 storage dtype: the
    # values are the source tensor rounded to float16.
    loaded_tensor = load_expert_tensor(persist_dir, "sam31", version="3")
    assert loaded_tensor.dtype == np.float16
    assert loaded_tensor.tobytes() == tensor_values.astype(np.float16).tobytes()
    # The Muse rasterization persisted: shape (frames, height, width),
    # values are the rasterized probabilities the fusion consumed.
    loaded_muse = load_expert_tensor(persist_dir, "muse", version="3")
    assert loaded_muse.shape == (2, 4, 6)
    assert loaded_muse.dtype == np.float16
    assert float(loaded_muse.max()) == pytest.approx(0.8, abs=1e-3)
    assert result["expert_artifacts"]["manifest"] == str(persist_dir / MANIFEST_NAME)


def test_manifest_records_frame_count_and_tensor_digest(tmp_path):
    """D4 gap 2/3: the record carries an explicit frame count and a
    SHA-256 of the tensor data itself, separate from the output
    digest of the .npy file bytes."""
    tensor = PROV_TENSORS["muse"]
    record = persist_expert_tensor(tmp_path, "muse", tensor, version="1")
    assert record["frames"] == tensor.shape[0]
    assert record["tensor_sha256"] == hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
    # The output digest still covers the exact artifact file bytes
    # and differs from the tensor-data digest (the .npy header is
    # part of the file, not of the tensor).
    assert record["sha256"] == _file_sha256(tmp_path / record["file"])
    assert record["tensor_sha256"] != record["sha256"]


def test_tampered_tensor_digest_fails_verification(tmp_path):
    """D4 gap 3: a manifest whose tensor digest does not match the
    stored tensor data fails closed."""
    persist_expert_tensor(tmp_path, "muse", PROV_TENSORS["muse"], version="1")
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["tensor_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExpertPersistError, match="tensor sha256 mismatch"):
        verify_manifest(tmp_path)
    with pytest.raises(ExpertPersistError, match="tensor sha256 mismatch"):
        load_expert_tensor(tmp_path, "muse", version="1")


def test_manifest_records_source_command_prompt_and_path(tmp_path):
    """D4 gap 1: the record names the source that produced the
    tensor -- command and prompt for a command expert, path for a
    tensor expert, nulls where a source field does not apply."""
    sources = {
        "muse": {
            "command": ["fake-expert", "--model", "m"],
            "prompt": "Locate the object.",
            "path": None,
        },
        "sam31": {"command": None, "prompt": None, "path": "/models/sam31.npy"},
    }
    manifest = persist_expert_tensors(tmp_path, PROV_TENSORS, version="1", sources=sources)
    records = {item["expert"]: item for item in manifest["artifacts"]}
    assert records["muse"]["source_command"] == ["fake-expert", "--model", "m"]
    assert records["muse"]["source_prompt"] == "Locate the object."
    assert records["muse"]["source_path"] is None
    assert records["sam31"]["source_command"] is None
    assert records["sam31"]["source_prompt"] is None
    assert records["sam31"]["source_path"] == "/models/sam31.npy"
    # An expert with no declared source still carries the fields.
    assert records["samurai"]["source_command"] is None
    assert records["samurai"]["source_prompt"] is None
    assert records["samurai"]["source_path"] is None


def test_failed_set_leaves_no_partial_artifacts(tmp_path):
    """D4 gap 5: a set persist that fails part-way leaves the
    directory exactly as it was -- no subset artifacts, no manifest
    for a subset."""
    bad = np.empty(1, dtype=object)
    bad[0] = {"not": "a tensor"}
    with pytest.raises(ExpertPersistError):
        persist_expert_tensors(tmp_path, {"sam31": PROV_TENSORS["sam31"], "bad": bad})
    assert not list(tmp_path.glob("*.npy"))
    assert not (tmp_path / MANIFEST_NAME).exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_manifest_lock_file_and_concurrent_writers(tmp_path):
    """D4 gap 5: persists serialize on the manifest lock, so two
    writers to one directory cannot lose each other's records."""
    import threading

    persist_expert_tensor(tmp_path, "sam31", PROV_TENSORS["sam31"], version="1")
    assert (tmp_path / MANIFEST_LOCK_NAME).is_file()

    errors: list[BaseException] = []

    def write(expert: str) -> None:
        try:
            persist_expert_tensor(tmp_path, expert, PROV_TENSORS[expert], version="1")
        except BaseException as error:  # noqa: BLE001 - collected and asserted below
            errors.append(error)

    threads = [threading.Thread(target=write, args=(expert,)) for expert in ("samurai", "muse")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    records = {item["expert"] for item in read_manifest(tmp_path)["artifacts"]}
    assert records == {"sam31", "samurai", "muse"}
    assert verify_manifest(tmp_path)


def _write_fake_ensemble(tmp_path, *, persist: bool):
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    tensor_path = tmp_path / "sam31_input.npy"
    np.save(tensor_path, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    fake_expert = tmp_path / "fake_expert.py"
    fake_expert.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "frames = [{'frame_index': f['frame_index'], 'regions': "
        "[{'type': 'box', 'coordinates': [0.0, 0.0, 0.5, 0.5], 'confidence': 0.8}]} "
        "for f in request['frames']]\n"
        "print(json.dumps({'regions_by_frame': frames}))\n"
    )
    config: dict = {
        "input_video": str(video),
        "experts": [
            {"name": "sam31", "path": str(tensor_path), "weight": 1.0},
            {
                "name": "muse",
                "kind": "command",
                "command": [sys.executable, str(fake_expert)],
                "prompt": "Locate the object.",
                "weight": 1.0,
            },
        ],
        "fusion": {"mode": "threshold", "threshold": 0.5, "batch_size": 1},
    }
    if persist:
        config["expert_persist_dir"] = str(tmp_path / "experts")
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(json.dumps(config))
    return config_path, tensor_path, fake_expert


def test_run_ensemble_records_sources_and_persistence_choice(tmp_path):
    """D4 gaps 1/4: an ensemble run's manifest records each expert's
    source, and the run report states the persistence choice."""
    config_path, tensor_path, fake_expert = _write_fake_ensemble(tmp_path, persist=True)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["expert_persistence"] == {
        "enabled": True,
        "mode": "opt-in",
        "dir": str(tmp_path / "experts"),
        "artifact_version": "1",
    }
    records = {item["expert"]: item for item in read_manifest(tmp_path / "experts")["artifacts"]}
    assert records["muse"]["source_command"] == [sys.executable, str(fake_expert)]
    assert records["muse"]["source_prompt"] == "Locate the object."
    assert records["sam31"]["source_path"] == str(tensor_path)
    assert records["sam31"]["frames"] == 2


def test_run_ensemble_without_persistence_reports_it_disabled(tmp_path):
    """D4 gap 4: a run that persists nothing says so in its report;
    the choice is explicit, never silent."""
    config_path, _, _ = _write_fake_ensemble(tmp_path, persist=False)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["expert_persistence"] == {"enabled": False, "mode": "opt-in"}
    assert "expert_artifacts" not in result


def test_pre_field_manifest_record_verifies_and_loads_byte_equal(tmp_path):
    """D4 gap 3: a manifest record written before the frames,
    tensor_sha256, and source fields existed still verifies, and
    its tensor loads byte-equal. The newer checks gate on a field
    being present; absence is acceptance, not rejection."""
    tensor = PROV_TENSORS["muse"]
    persist_expert_tensor(tmp_path, "muse", tensor, version="1")
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["artifacts"][0]
    for field in (
        "frames",
        "tensor_sha256",
        "source_command",
        "source_prompt",
        "source_path",
    ):
        record.pop(field, None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(tmp_path)["artifacts"]
    loaded = load_expert_tensor(tmp_path, "muse", version="1")
    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype
    assert loaded.tobytes() == tensor.tobytes()


def _write_flag_ensemble(tmp_path, *, mode, persist: bool, persist_dir=None):
    """The fake two-expert ensemble, with a persist feature-flag mode.

    mode=None leaves the key out of the config entirely (the flag
    absent case). persist_dir overrides the directory the base
    builder sets, for the unwritable-target case.
    """
    config_path, _, _ = _write_fake_ensemble(tmp_path, persist=persist)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if mode is not None:
        raw["expert_persist_mode"] = mode
    if persist_dir is not None:
        raw["expert_persist_dir"] = str(persist_dir)
    config_path.write_text(json.dumps(raw))
    return config_path


def test_persist_mode_absent_defaults_to_opt_in(tmp_path):
    """Flag absent = opt-in: the field defaults, persistence runs
    when a directory is set, and the report names the mode."""
    config_path = _write_flag_ensemble(tmp_path, mode=None, persist=True)
    config = load_config(config_path)
    assert config.expert_persist_mode == "opt-in"
    result = run_ensemble(config, tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["expert_persistence"]["mode"] == "opt-in"
    assert result["expert_persistence"]["enabled"] is True
    assert (tmp_path / "experts" / MANIFEST_NAME).is_file()


def test_persist_mode_off_writes_no_records_even_with_dir_set(tmp_path):
    """Mode off: the run completes and writes its mask, but no
    buffers are persisted -- no manifest, no artifacts -- and the
    report states the mode with persistence disabled."""
    config_path = _write_flag_ensemble(tmp_path, mode="off", persist=True)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["expert_persistence"] == {"enabled": False, "mode": "off"}
    assert "expert_artifacts" not in result
    persist_dir = tmp_path / "experts"
    assert not (persist_dir / MANIFEST_NAME).exists()
    assert not list(persist_dir.glob("*.npy"))
    assert (tmp_path / "mask.mp4").is_file()


def test_persist_mode_required_persists_and_reports_mode(tmp_path):
    """Mode required with a writable target: every expert persists
    and the report names the mode."""
    config_path = _write_flag_ensemble(tmp_path, mode="required", persist=True)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["expert_persistence"] == {
        "enabled": True,
        "mode": "required",
        "dir": str(tmp_path / "experts"),
        "artifact_version": "1",
    }
    records = {item["expert"] for item in read_manifest(tmp_path / "experts")["artifacts"]}
    assert records == {"sam31", "muse"}


def test_persist_mode_required_without_dir_refuses_before_output(tmp_path):
    """Mode required with no target: the run refuses fail-closed
    with EnsembleError and writes neither mask nor report."""
    config_path = _write_flag_ensemble(tmp_path, mode="required", persist=False)
    with pytest.raises(EnsembleError, match="required"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_persist_mode_required_unwritable_target_refuses(tmp_path):
    """Mode required with a target that cannot be created (a path
    under a regular file): the run refuses with EnsembleError and
    writes neither mask nor report."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    config_path = _write_flag_ensemble(
        tmp_path, mode="required", persist=True, persist_dir=blocker / "experts"
    )
    with pytest.raises(EnsembleError, match="required"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert not (tmp_path / "mask.mp4").exists()
    assert not (tmp_path / "report.json").exists()


def test_persist_mode_invalid_value_refused_at_parse(tmp_path):
    """An unknown mode is refused when the config is parsed, before
    any run starts."""
    config_path = _write_flag_ensemble(tmp_path, mode="sometimes", persist=True)
    with pytest.raises(EnsembleError, match="expert_persist_mode"):
        load_config(config_path)


def test_required_mode_api_command_expert_persists_with_digest(tmp_path):
    """An API-shaped command expert (the ens-proof Muse shape: a
    provider command, --provider baseten --model
    meta-models/muse-spark-open) persists through run_ensemble in
    required mode, with a manifest entry whose SHA-256 and tensor
    digest match the artifact on disk."""
    config_path = _write_flag_ensemble(tmp_path, mode="required", persist=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    fake_expert = raw["experts"][1]["command"][1]
    raw["experts"][1]["command"] = [
        sys.executable,
        fake_expert,
        "--provider",
        "baseten",
        "--model",
        "meta-models/muse-spark-open",
    ]
    config_path.write_text(json.dumps(raw))
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    persist_dir = tmp_path / "experts"
    records = {item["expert"]: item for item in read_manifest(persist_dir)["artifacts"]}
    assert set(records) == {"sam31", "muse"}
    record = records["muse"]
    assert record["sha256"] == _file_sha256(persist_dir / record["file"])
    loaded = load_expert_tensor(persist_dir, "muse", version="1")
    assert record["tensor_sha256"] == hashlib.sha256(loaded.tobytes(order="C")).hexdigest()
    assert record["source_command"][-1] == "meta-models/muse-spark-open"
    assert record["expert_empty"] is False
    assert result["expert_empty"] == []


def test_all_zero_command_expert_persists_and_is_flagged_expert_empty(tmp_path):
    """A command expert that returns no regions for any frame has an
    all-zero tensor. The tensor still persists (it is the evidence),
    the manifest record and the run report flag it expert_empty, and
    the expert is not dropped: its weight still appears in every
    frame report. The flag is recorded even when nothing persists."""
    config_path, _, _ = _write_fake_ensemble(tmp_path, persist=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    empty_expert = tmp_path / "empty_expert.py"
    empty_expert.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "frames = [{'frame_index': f['frame_index'], 'regions': []} "
        "for f in request['frames']]\n"
        "print(json.dumps({'regions_by_frame': frames}))\n"
    )
    raw["experts"][1]["command"] = [sys.executable, str(empty_expert)]
    config_path.write_text(json.dumps(raw))
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    persist_dir = tmp_path / "experts"
    records = {item["expert"]: item for item in read_manifest(persist_dir)["artifacts"]}
    assert records["muse"]["expert_empty"] is True
    assert records["sam31"]["expert_empty"] is False
    loaded = load_expert_tensor(persist_dir, "muse", version="1")
    assert loaded.shape == (2, 4, 6)
    assert not np.any(loaded)
    assert result["expert_empty"] == ["muse"]
    assert all("weight_muse" in report for report in result["frames_report"])
    # No persistence configured: the empty expert is still flagged.
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    config_path2, _, _ = _write_fake_ensemble(bare_dir, persist=False)
    raw2 = json.loads(config_path2.read_text(encoding="utf-8"))
    raw2["experts"][1]["command"] = [sys.executable, str(empty_expert)]
    config_path2.write_text(json.dumps(raw2))
    result2 = run_ensemble(
        load_config(config_path2), bare_dir / "mask.mp4", bare_dir / "report.json"
    )
    assert result2["expert_empty"] == ["muse"]
    assert result2["expert_persistence"] == {"enabled": False, "mode": "opt-in"}


def test_required_mode_empty_expert_refuses_after_recording(tmp_path):
    """Required mode + an all-zero expert: the run is refused.

    The B7 smoke (2026-10-28) exposed this defect: a Muse=0 tensor
    substitution in required mode exited 0 with expert_empty recorded
    but no refusal. Required mode is the governed path, so an empty
    expert must fail the run. The evidence is recorded before the
    refusal: the persisted manifest flags the expert expert_empty and
    the run report names it, so the failure is never silent.
    """
    config_path = _write_flag_ensemble(tmp_path, mode="required", persist=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    empty_expert = tmp_path / "empty_expert.py"
    empty_expert.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "frames = [{'frame_index': f['frame_index'], 'regions': []} "
        "for f in request['frames']]\n"
        "print(json.dumps({'regions_by_frame': frames}))\n"
    )
    raw["experts"][1]["command"] = [sys.executable, str(empty_expert)]
    config_path.write_text(json.dumps(raw))
    with pytest.raises(EnsembleError, match="expert_empty"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["expert_empty"] == ["muse"]
    records = {item["expert"]: item for item in read_manifest(tmp_path / "experts")["artifacts"]}
    assert records["muse"]["expert_empty"] is True
    assert records["sam31"]["expert_empty"] is False


def test_required_mode_persist_failure_fails_run_without_report(tmp_path, monkeypatch):
    """Required mode: a persist-stage failure (after the start-of-run
    writability probe has passed) fails the run with EnsembleError
    under the flag's fail-closed semantics, and no run report is
    written."""
    import samurai.ensemble as ensemble_module

    def fail_persist(*args, **kwargs):
        raise ExpertPersistError("simulated persist failure")

    monkeypatch.setattr(ensemble_module, "persist_expert_tensors", fail_persist)
    config_path = _write_flag_ensemble(tmp_path, mode="required", persist=True)
    with pytest.raises(EnsembleError, match="required"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def _dir_snapshot(directory: Path) -> dict[str, str]:
    """Map every file in ``directory`` to the SHA-256 of its bytes."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


@pytest.mark.parametrize(
    "corrupt",
    [
        "{ not valid json",
        json.dumps({"manifest_version": 1}),
        json.dumps({"manifest_version": 1, "artifacts": "not-a-list"}),
        json.dumps(["not", "a", "dict"]),
    ],
)
def test_corrupt_manifest_fails_closed_and_leaves_directory_unchanged(tmp_path, corrupt):
    """Defect A: a corrupt existing manifest is not 'no records'.

    Malformed JSON and a manifest without an artifacts list must
    abort the persist before any artifact is replaced, with the
    directory bytes exactly unchanged.
    """
    persist_expert_tensor(tmp_path, "sam31", PROV_TENSORS["sam31"], version="1")
    (tmp_path / MANIFEST_NAME).write_text(corrupt, encoding="utf-8")
    before = _dir_snapshot(tmp_path)
    with pytest.raises(ExpertPersistError):
        persist_expert_tensor(tmp_path, "muse", PROV_TENSORS["muse"], version="1")
    after = _dir_snapshot(tmp_path)
    assert after == before


def test_rename_phase_failure_rolls_back_to_prior_directory(tmp_path, monkeypatch):
    """Defect B: a failure on the 2nd rename restores the old set.

    The fault fires exactly once, on the second staged-artifact
    rename. The directory must end exactly as before the call: old
    artifact bytes and old manifest intact, no staged temps, no
    backup files, no other orphans.
    """
    old_tensors = {
        "sam31": np.zeros((2, 3, 4), dtype=np.float32),
        "muse": np.ones((2, 3, 4), dtype=np.float32),
    }
    persist_expert_tensors(tmp_path, old_tensors, version="1")
    before = _dir_snapshot(tmp_path)
    new_tensors = {
        "sam31": np.full((2, 3, 4), 7.0, dtype=np.float32),
        "muse": np.full((2, 3, 4), 9.0, dtype=np.float32),
    }
    real_replace = os.replace
    state = {"count": 0, "fired": False}

    def failing_replace(src, dst, *args, **kwargs):
        src_name = Path(src).name
        dst_name = Path(dst).name
        is_staged_rename = src_name.startswith(".tmp-") and dst_name.endswith(".v1.npy")
        if is_staged_rename and not state["fired"]:
            state["count"] += 1
            if state["count"] == 2:
                state["fired"] = True
                raise OSError("injected rename failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected rename failure"):
        persist_expert_tensors(tmp_path, new_tensors, version="1")
    monkeypatch.undo()
    assert state["fired"]
    after = _dir_snapshot(tmp_path)
    assert after == before
    assert not list(tmp_path.glob(".tmp-*"))
    assert not list(tmp_path.glob(".backup-*"))
    assert not list(tmp_path.glob(".backup-manifest-*"))
    # Old manifest still verifies and old tensors load byte-equal.
    assert verify_manifest(tmp_path)
    for expert, tensor in old_tensors.items():
        loaded = load_expert_tensor(tmp_path, expert, version="1")
        assert loaded.tobytes() == tensor.tobytes()


def _muse_record(directory: Path) -> dict:
    """The manifest record of one expert in a D4 directory."""
    records = {item["expert"]: item for item in read_manifest(directory)["artifacts"]}
    return records["muse"]


def test_identical_repersist_keeps_the_manifest_bytes(tmp_path):
    """Finding 3: an identical re-persist must not mutate the manifest
    bytes. A calibration record pins the manifest digest, so rewriting
    an unchanged manifest would invalidate a calibration that still
    matches the tensors it was fitted on."""
    persist_expert_tensors(tmp_path, PROV_TENSORS, version="1")
    manifest_path = tmp_path / MANIFEST_NAME
    before = manifest_path.read_bytes()
    before_digest = hashlib.sha256(before).hexdigest()

    persist_expert_tensors(tmp_path, PROV_TENSORS, version="1")

    after = manifest_path.read_bytes()
    assert after == before
    assert hashlib.sha256(after).hexdigest() == before_digest
    # The untouched manifest still verifies against the artifacts.
    assert verify_manifest(tmp_path)["artifacts"]
    for expert, tensor in PROV_TENSORS.items():
        loaded = load_expert_tensor(tmp_path, expert, version="1")
        assert loaded.tobytes() == tensor.tobytes()


def test_changed_tensors_change_the_manifest_bytes(tmp_path):
    """Finding 3: changed tensors legitimately invalidate the pinned
    manifest digest, because the artifact digests change with them."""
    persist_expert_tensors(tmp_path, PROV_TENSORS, version="1")
    manifest_path = tmp_path / MANIFEST_NAME
    before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    before_muse = _muse_record(tmp_path)["sha256"]

    changed = dict(PROV_TENSORS)
    changed["muse"] = PROV_TENSORS["muse"] + 0.5
    persist_expert_tensors(tmp_path, changed, version="1")

    after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert after != before
    assert _muse_record(tmp_path)["sha256"] != before_muse
    assert verify_manifest(tmp_path)["artifacts"]


def test_rollback_keeps_a_backup_it_failed_to_restore(tmp_path, monkeypatch):
    """Finding 6: a rollback never deletes a backup it failed to
    restore. That backup holds the only copy of the previous bytes, so
    it stays on disk and the failure names it."""
    old_tensors = {"sam31": np.zeros((2, 3, 4), dtype=np.float32)}
    persist_expert_tensors(tmp_path, old_tensors, version="1")
    old_bytes = (tmp_path / artifact_filename("sam31", "1")).read_bytes()

    new_tensors = {"sam31": np.full((2, 3, 4), 7.0, dtype=np.float32)}
    real_replace = os.replace

    def failing_replace(src, dst, *args, **kwargs):
        src_name = Path(src).name
        dst_name = Path(dst).name
        if src_name.startswith(".backup-"):
            # The restore of the backup fails.
            raise OSError("injected restore failure")
        if src_name.startswith(".tmp-") and dst_name.endswith(".v1.npy"):
            raise OSError("injected rename failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected rename failure") as excinfo:
        persist_expert_tensors(tmp_path, new_tensors, version="1")
    monkeypatch.undo()

    backups = sorted(tmp_path.glob(".backup-*.npy"))
    assert backups, "the failed restore deleted the backup that held the previous bytes"
    assert backups[0].read_bytes() == old_bytes
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("unrestored backup" in note for note in notes)


def test_d4_manifest_and_lock_names_are_distinct_from_the_legacy_store():
    """Finding 7: the D4 expert layer and the legacy per-chunk store
    must not share a manifest or a lock file name. One filename with two
    schemas lets a reader take one layer's manifest for the other's."""
    assert MANIFEST_NAME == "experts_manifest.json"
    assert MANIFEST_LOCK_NAME == ".experts_manifest.lock"
    assert MANIFEST_NAME != LEGACY_MANIFEST_NAME
    assert MANIFEST_LOCK_NAME != LEGACY_MANIFEST_LOCK_NAME
    assert LEGACY_MANIFEST_NAME == "manifest.json"
    assert LEGACY_MANIFEST_LOCK_NAME == ".manifest.lock"
