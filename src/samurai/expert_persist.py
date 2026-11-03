"""Versioned expert-tensor persistence (fusion gate D4).

The gate D4 ruling: persist every expert tensor, including Muse
rasterizations, as versioned ``.npy`` artifacts. Use atomic writes and
a manifest entry containing expert identity, dtype, shape, frame count,
source command/prompt, output digest, and tensor SHA-256. In-memory-only
command-expert fields are not acceptable for a rerunnable
fusion/calibration run.

This module is the D4 layer. It is additive: it does not change any
scoring or fusion behaviour, and it does not replace
``samurai.persist`` (the per-chunk provenance store). A caller that
opts in gets, for one output directory:

- one artifact per expert: ``<expert>.v<version>.npy`` (the artifact
  version is in the filename and in the manifest record);
- one manifest: ``experts_manifest.json`` in the same directory (a
  name of its own: the legacy per-chunk store in ``samurai.persist``
  owns ``manifest.json``), with one record per artifact carrying
  ``expert``, ``version``, ``file``, ``shape``, ``frames`` (the frame
  count, ``shape[0]``), ``dtype``, ``bytes``, ``sha256`` (the output
  digest: SHA-256 of the exact ``.npy`` file bytes, computed by
  reading the file back after the atomic rename), ``tensor_sha256``
  (SHA-256 of the tensor data
  itself, over the array's bytes in C order), ``expert_empty``
  (true when every value in the tensor is zero: an empty expert is
  a recorded failure, never a silent drop), and the source that
  produced the tensor: ``source_command``, ``source_prompt`` and
  ``source_path`` (null where a field does not apply);
- atomic writes only: each file is written to a temp file in the
  same directory, flushed, fsynced, then renamed with
  ``os.replace``. A failed write leaves no partial artifact and no
  manifest entry;
- set-level atomicity and locking: one persist call stages every
  artifact to a temp file first, then renames the whole set into
  place and rewrites the manifest once, all under an exclusive
  advisory lock (``.experts_manifest.lock`` in the output directory).
  A failure during staging, the rename phase, the manifest write, or
  manifest verification rolls the directory back to exactly as it was: no
  subset artifacts and no manifest for a subset. The lock serializes
  writers across threads and processes on the same host, so two
  writers to one directory cannot lose each other's records. The
  lock file is never unlinked: unlinking it would let a second
  writer lock a fresh inode while a first writer still holds the
  old one.

Loading verifies the manifest hashes before the tensor is served:
a corrupted or replaced file fails the hash check and raises.
Records written before the frame-count / tensor-digest / source
fields existed still verify: those checks apply only when the
record carries the field.

The manifest bytes are stable across an identical re-persist: when the
new artifact records equal the records already on disk, the manifest
is not rewritten, so the SHA-256 a calibration record pins over the
manifest still matches the tensors it was fitted on. Changed tensors
change the records, so the manifest is rewritten and the pinned
digest no longer matches.

A rollback never deletes a backup it failed to restore: a backup whose
restore fails still holds the only copy of the previous bytes, so it
stays on disk and the failure is reported to the caller.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Filename boundary (2026-10-29): the D4 layer owns its own manifest
# and lock file names. The legacy per-chunk store in ``samurai.persist``
# writes ``manifest.json`` and ``.manifest.lock`` with a different
# schema; one filename with two schemas let a reader take one layer's
# manifest for the other's. The two layers never share a directory.
MANIFEST_NAME = "experts_manifest.json"
MANIFEST_LOCK_NAME = ".experts_manifest.lock"
MANIFEST_VERSION = 1
DEFAULT_ARTIFACT_VERSION = "1"

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExpertPersistError(RuntimeError):
    """A persisted expert artifact is missing, corrupt, or unverifiable."""


def _validate_token(value: Any, *, field: str) -> str:
    token = str(value)
    if not _TOKEN.match(token) or ".." in token:
        raise ExpertPersistError(f"invalid {field}: {value!r}")
    return token


def artifact_filename(expert: str, version: str = DEFAULT_ARTIFACT_VERSION) -> str:
    """Return the versioned artifact filename for one expert."""
    name = _validate_token(expert, field="expert name")
    ver = _validate_token(version, field="artifact version")
    return f"{name}.v{ver}.npy"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(array: np.ndarray) -> str:
    """SHA-256 of the tensor data itself (the array's bytes, C order)."""
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def manifest_lock(directory: Path) -> Generator[None, None, None]:
    """Hold the exclusive advisory lock for one expert directory.

    The lock is a file lock (``fcntl.flock``) on
    ``<directory>/.experts_manifest.lock``. It serializes cooperating
    readers and writers across threads and processes on the same host.
    A governed reader uses the same lock while verifying artifact bytes
    and opening their memory mappings, so a D4 writer cannot replace an
    artifact between those operations. The lock file is never unlinked:
    unlinking it would let a second writer lock a fresh inode while a
    first writer still holds the old one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / MANIFEST_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _stage_npy(directory: Path, array: np.ndarray) -> Path:
    """Write ``array`` to a temp ``.npy`` in ``directory``; return its path.

    The caller renames the staged file into place with
    ``os.replace``. A failed write unlinks the temp file and leaves
    no partial artifact behind.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return Path(tmp)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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
    _fsync_dir(path.parent)


def read_manifest(output_dir: Path) -> dict[str, Any]:
    """Read the D4 manifest in ``output_dir``; raise if absent or invalid."""
    path = Path(output_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ExpertPersistError(f"expert manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertPersistError(f"cannot read expert manifest {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("artifacts"), list):
        raise ExpertPersistError(f"expert manifest has no artifacts list: {path}")
    return data


def _existing_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    if not (Path(output_dir) / MANIFEST_NAME).is_file():
        return []
    # Any ExpertPersistError from here is corruption: fail closed.
    return list(read_manifest(output_dir)["artifacts"])


def _source_fields(source: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize one expert's source declaration into record fields."""
    source = source or {}
    command = source.get("command")
    prompt = source.get("prompt")
    path = source.get("path")
    return {
        "source_command": [str(part) for part in command] if command else None,
        "source_prompt": str(prompt) if prompt is not None else None,
        "source_path": str(path) if path is not None else None,
    }


def _persist_set(
    output_dir: Path,
    items: list[tuple[str, Any, dict[str, Any] | None]],
    *,
    version: str,
) -> dict[str, Any]:
    """Persist a set of expert tensors as one locked, staged write.

    ``items`` is a list of ``(expert, tensor, source)`` triples.
    Every artifact is staged to a temp file before any rename, the
    set is renamed into place, and the manifest is rewritten once
    with the merged records. The existing manifest is read before
    any staging or rename, so a corrupt manifest fails closed with
    the directory untouched. The rename phase is all-or-rollback:
    each existing target is backed up before the rename loop, and
    the renames, fsync, manifest write, and manifest verification
    all run inside the protected region; on any failure the backups
    are restored, newly placed targets are removed, staged temps
    are unlinked, and the old manifest is restored. A backup whose
    restore fails is kept on disk, never deleted. Returns the
    manifest; when the new records equal the records already on disk
    the manifest bytes are left untouched.
    """
    directory = Path(output_dir)
    prepared: list[tuple[str, np.ndarray, dict[str, Any] | None]] = []
    for expert, tensor, source in items:
        array = np.asarray(tensor)
        if array.dtype.hasobject:
            raise ExpertPersistError(f"expert {expert!r}: object dtype cannot be persisted")
        prepared.append((expert, array, source))
    with manifest_lock(directory):
        # Read prior records inside the lock, before staging or renaming,
        # and reuse this list in the merge.
        existing_artifacts = _existing_artifacts(directory)
        manifest_path = directory / MANIFEST_NAME
        manifest_existed = manifest_path.is_file()
        staged: list[tuple[Path, Path]] = []
        records: list[dict[str, Any]] = []
        backups: list[tuple[Path, Path | None]] = []
        manifest_backup: Path | None = None
        manifest: dict[str, Any] | None = None
        try:
            # Back up the old manifest by copy (its path stays in place
            # until the atomic manifest write replaces it).
            if manifest_existed:
                fd, btmp = tempfile.mkstemp(
                    dir=directory, prefix=".backup-manifest-", suffix=".json"
                )
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(manifest_path.read_bytes())
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    try:
                        os.unlink(btmp)
                    except OSError:
                        pass
                    raise
                manifest_backup = Path(btmp)
            for expert, array, source in prepared:
                filename = artifact_filename(expert, version)
                tmp = _stage_npy(directory, array)
                staged.append((tmp, directory / filename))
                records.append(
                    {
                        "expert": str(expert),
                        "version": str(version),
                        "file": filename,
                        "shape": [int(axis) for axis in array.shape],
                        "frames": int(array.shape[0]) if array.ndim >= 1 else 0,
                        "dtype": str(array.dtype),
                        "bytes": int(tmp.stat().st_size),
                        "sha256": _sha256_file(tmp),
                        "tensor_sha256": _tensor_sha256(array),
                        "expert_empty": bool(array.size == 0 or not np.any(array)),
                        **_source_fields(source),
                    }
                )
            # Back up each existing target before any rename.
            for _tmp, target in staged:
                if target.is_file():
                    fd, btmp = tempfile.mkstemp(dir=directory, prefix=".backup-", suffix=".npy")
                    os.close(fd)
                    backup = Path(btmp)
                    try:
                        os.replace(target, backup)
                    except BaseException:
                        try:
                            os.unlink(backup)
                        except OSError:
                            pass
                        raise
                    backups.append((target, backup))
                else:
                    backups.append((target, None))
            for tmp, target in staged:
                os.replace(tmp, target)
            _fsync_dir(directory)
            replaced = {(record["expert"], record["version"]) for record in records}
            artifacts = [
                item
                for item in existing_artifacts
                if (str(item.get("expert")), str(item.get("version"))) not in replaced
            ]
            artifacts.extend(records)
            artifacts.sort(key=lambda item: (str(item.get("expert")), str(item.get("version"))))
            if manifest_existed and artifacts == existing_artifacts:
                # Identical re-persist: the artifact records are exactly
                # the records already on disk, so the manifest keeps its
                # bytes (and its created_at). A calibration record pins
                # the manifest digest; rewriting an unchanged manifest
                # would invalidate a calibration that still matches the
                # tensors it was fitted on.
                manifest = read_manifest(directory)
            else:
                manifest = {
                    "manifest_version": MANIFEST_VERSION,
                    "created_at": datetime.now(UTC).isoformat(),
                    "artifacts": artifacts,
                }
                _atomic_write_json(manifest_path, manifest)
            # Read-back: the manifest on disk must verify against the tensors.
            verify_manifest(directory)
        except BaseException as error:
            # Roll back artifacts in reverse order. A backup whose
            # restore fails is NEVER deleted: it still holds the only
            # copy of the previous bytes, so it stays on disk and the
            # caller is told which files were not restored.
            unrestored: list[Path] = []
            for target, backup in reversed(backups):
                if backup is None:
                    try:
                        os.unlink(target)
                    except OSError:
                        pass
                    continue
                try:
                    os.replace(backup, target)
                except OSError:
                    unrestored.append(backup)
            # Roll back the manifest.
            try:
                if manifest_backup is not None:
                    os.replace(manifest_backup, manifest_path)
                    manifest_backup = None
                elif not manifest_existed:
                    try:
                        os.unlink(manifest_path)
                    except OSError:
                        pass
            except OSError:
                if manifest_backup is not None:
                    unrestored.append(manifest_backup)
            for tmp, _target in staged:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            # Clean any backup placeholders that were not restored. A
            # backup in `unrestored` is kept: deleting it would destroy
            # the bytes the rollback could not put back.
            for _target, backup in backups:
                if backup is not None and backup not in unrestored:
                    try:
                        os.unlink(backup)
                    except OSError:
                        pass
            if manifest_backup is not None and manifest_backup not in unrestored:
                try:
                    os.unlink(manifest_backup)
                except OSError:
                    pass
            try:
                _fsync_dir(directory)
            except OSError:
                pass
            if unrestored:
                error.add_note(
                    "rollback kept unrestored backup files; the previous bytes are "
                    f"still on disk: {', '.join(str(item) for item in unrestored)}"
                )
            raise
        # Success: unlink backups, then fsync the directory.
        for _target, backup in backups:
            if backup is not None:
                try:
                    os.unlink(backup)
                except OSError:
                    pass
        if manifest_backup is not None:
            try:
                os.unlink(manifest_backup)
            except OSError:
                pass
        _fsync_dir(directory)
        assert manifest is not None
        return manifest


def persist_expert_tensor(
    output_dir: Path,
    expert: str,
    tensor: Any,
    *,
    version: str = DEFAULT_ARTIFACT_VERSION,
    source_command: Any = None,
    source_prompt: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Persist one expert tensor as a versioned ``.npy`` artifact.

    The write is atomic and the manifest is updated atomically
    after the artifact is in place, under the directory's manifest
    lock. Re-persisting the same expert at the same version
    replaces its artifact and its manifest record; a different
    version creates a second artifact. The record names the source
    that produced the tensor (``source_command`` / ``source_prompt``
    for a command expert, ``source_path`` for a tensor expert);
    fields that do not apply are recorded as null.

    Returns the manifest record for the artifact.
    """
    source = {"command": source_command, "prompt": source_prompt, "path": source_path}
    manifest = _persist_set(output_dir, [(expert, tensor, source)], version=version)
    return next(
        item
        for item in manifest["artifacts"]
        if item["expert"] == str(expert) and item["version"] == str(version)
    )


def persist_expert_tensors(
    output_dir: Path,
    tensors: dict[str, Any],
    *,
    version: str = DEFAULT_ARTIFACT_VERSION,
    sources: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist every tensor in ``tensors`` and return the manifest.

    Every expert in the mapping is persisted; there is no silent
    skip. The set is staged and renamed as one locked write (see
    ``_persist_set``): a failure leaves the directory exactly as it
    was, never a subset with a manifest for the subset. ``sources``
    optionally maps an expert name to its source declaration
    (``command`` / ``prompt`` / ``path``) for the manifest record.
    """
    if not tensors:
        raise ExpertPersistError("no expert tensors to persist")
    sources = sources or {}
    items = [(expert, tensor, sources.get(expert)) for expert, tensor in tensors.items()]
    return _persist_set(output_dir, items, version=version)


def verify_manifest(output_dir: Path) -> dict[str, Any]:
    """Verify every artifact in the manifest against its hashes.

    Returns the manifest on success. Raises ``ExpertPersistError``
    naming the expert when a file is missing, its output digest
    differs from the manifest, its shape/dtype/frame count differ
    from the record, or its tensor-data digest differs from the
    record. Frame-count and tensor-digest checks apply only to
    records that carry those fields.
    """
    directory = Path(output_dir)
    manifest = read_manifest(directory)
    for record in manifest["artifacts"]:
        path = directory / str(record.get("file", ""))
        expert = record.get("expert")
        if not path.is_file():
            raise ExpertPersistError(f"expert {expert!r}: artifact missing: {path}")
        actual = _sha256_file(path)
        if actual != record.get("sha256"):
            raise ExpertPersistError(
                f"expert {expert!r}: sha256 mismatch: file {actual} != manifest "
                f"{record.get('sha256')}"
            )
        header = np.load(path, allow_pickle=False, mmap_mode="r")
        if [int(axis) for axis in header.shape] != record.get("shape") or str(
            header.dtype
        ) != record.get("dtype"):
            raise ExpertPersistError(
                f"expert {expert!r}: shape/dtype mismatch with manifest record"
            )
        if record.get("frames") is not None:
            frames = int(header.shape[0]) if header.ndim >= 1 else 0
            if frames != int(record["frames"]):
                raise ExpertPersistError(
                    f"expert {expert!r}: frame count mismatch with manifest record"
                )
        if record.get("tensor_sha256") is not None:
            array = np.load(path, allow_pickle=False)
            actual_tensor = _tensor_sha256(array)
            if actual_tensor != record["tensor_sha256"]:
                raise ExpertPersistError(
                    f"expert {expert!r}: tensor sha256 mismatch: data {actual_tensor} "
                    f"!= manifest {record['tensor_sha256']}"
                )
    return manifest


def load_expert_tensor(
    output_dir: Path,
    expert: str,
    *,
    version: str = DEFAULT_ARTIFACT_VERSION,
) -> np.ndarray:
    """Load one persisted expert tensor after manifest verification.

    The file's SHA-256 must match the manifest record for
    ``(expert, version)`` before the tensor is loaded; a corrupted
    file raises ``ExpertPersistError`` and is never served. When
    the record carries a tensor-data digest, the loaded tensor
    must match it as well.
    """
    directory = Path(output_dir)
    manifest = read_manifest(directory)
    record = next(
        (
            item
            for item in manifest["artifacts"]
            if item.get("expert") == str(expert) and item.get("version") == str(version)
        ),
        None,
    )
    if record is None:
        raise ExpertPersistError(
            f"expert {expert!r} version {version!r}: no manifest record in {directory}"
        )
    path = directory / str(record["file"])
    if not path.is_file():
        raise ExpertPersistError(f"expert {expert!r}: artifact missing: {path}")
    actual = _sha256_file(path)
    if actual != record.get("sha256"):
        raise ExpertPersistError(
            f"expert {expert!r}: sha256 mismatch: file {actual} != manifest {record['sha256']}"
        )
    array = np.load(path, allow_pickle=False)
    if [int(axis) for axis in array.shape] != record.get("shape") or str(array.dtype) != record.get(
        "dtype"
    ):
        raise ExpertPersistError(f"expert {expert!r}: shape/dtype mismatch with manifest record")
    if record.get("tensor_sha256") is not None:
        actual_tensor = _tensor_sha256(array)
        if actual_tensor != record["tensor_sha256"]:
            raise ExpertPersistError(
                f"expert {expert!r}: tensor sha256 mismatch: data {actual_tensor} "
                f"!= manifest {record['tensor_sha256']}"
            )
    return array
