"""Tensor persistence for quadmask expert chunks.

Implements the expert tensor persistence specification at
docs/tensor-persistence-spec.md (spec v1.0.0).

Layout (defaults, used where the specification is absent):
- One directory per chunk: ``chunk_dir``.
- One file per expert: ``<chunk_dir>/<expert_id>.npy``.
- One manifest per chunk: ``<chunk_dir>/manifest.json``.
- All writes are atomic: write to a temp file in the same directory,
  fsync, then ``os.replace`` (rename) over the target.
- Every saved tensor is verified by reading the file back and
  comparing its SHA-256 with the hash recorded in the manifest.

Manifest format (version 1)::

    {
      "version": 1,
      "experts": {
        "<expert_id>": {
          "expert_id": str,
          "file": "<expert_id>.npy",
          "model": str,
          "prompt": str,
          "seed": ...,
          "host": str,
          "code_hash": str,
          "timestamp": str,          # ISO 8601 UTC
          "determinism_class": "local" | "api",
          "shape": [int, ...],
          "dtype": str,
          "sha256": str              # SHA-256 of the .npy file bytes
        }
      },
      "updated_at": str              # ISO 8601 UTC, optional
    }

Pure stdlib + numpy only. No imports from production code.

Persistence boundary (2026-10-29): this module owns ``manifest.json``
and ``.manifest.lock`` in its own per-chunk directories, with the
``{"version": 1, "experts": {...}}`` schema. The D4 versioned expert
layer (``samurai.expert_persist``) writes ``experts_manifest.json``
and ``.experts_manifest.lock`` with a different schema
(``{"manifest_version": 1, "artifacts": [...]}``). One directory
belongs to exactly one layer: the two never share a directory, so
neither manifest can be read as the other's. See
docs/tensor-persistence-spec.md section 8.

Specification divergence note (2026-10-27): the specification (spec
1.0.0) was absent when this module was built, so the defaults above
were used. The specification landed during the build and diverges from
those defaults: it specifies ``experts/<expert_id>/<expert_id>__<model_slug>__prob.npy``
paths, a manifest ``experts`` list with fields such as
``model_name`` / ``prompt_text`` / ``code_hash`` as an object, and
``determinism_class`` values ``A`` / ``B`` / ``C``. This module keeps
the module's own interface (``<expert_id>.npy``, manifest ``experts`` dict,
``determinism_class`` ``'local'`` / ``'api'``). Aligning the two is a
maintainer decision; do not silently change this layout.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np

MANIFEST_NAME = "manifest.json"
MANIFEST_LOCK_NAME = ".manifest.lock"
MANIFEST_VERSION = 1
_ALLOWED_DETERMINISM = ("local", "api")
_REQUIRED_PROVENANCE = (
    "model",
    "prompt",
    "seed",
    "host",
    "code_hash",
    "determinism_class",
)

PathLike = str | os.PathLike


def _utc_now() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _validate_expert_id(expert_id: Any) -> str:
    """Validate and return ``expert_id`` as a safe file stem.

    Raises:
        ValueError: If the id is empty or contains path separators,
            ``..``, or NUL, or is an absolute path.
    """
    eid = str(expert_id)
    if not eid or eid in (".", ".."):
        raise ValueError(f"invalid expert_id: {expert_id!r}")
    if "/" in eid or "\\" in eid or "\x00" in eid or ".." in eid:
        raise ValueError(f"expert_id must not contain path separators or '..': {expert_id!r}")
    if os.path.isabs(eid):
        raise ValueError(f"expert_id must not be absolute: {expert_id!r}")
    return eid


def _chunk_path(chunk_dir: PathLike) -> str:
    """Return ``chunk_dir`` as a string path."""
    return os.fspath(chunk_dir)


def _expert_path(chunk_dir: PathLike, expert_id: str) -> str:
    """Return the .npy path for ``expert_id`` inside ``chunk_dir``."""
    return os.path.join(_chunk_path(chunk_dir), f"{expert_id}.npy")


def _manifest_path(chunk_dir: PathLike) -> str:
    """Return the manifest path inside ``chunk_dir``."""
    return os.path.join(_chunk_path(chunk_dir), MANIFEST_NAME)


@contextmanager
def _chunk_lock(chunk_dir: PathLike) -> Generator[None, None, None]:
    """Hold the exclusive advisory lock for one chunk directory.

    The lock is a file lock (``fcntl.flock``) on
    ``<chunk_dir>/.manifest.lock``. It serializes writers across
    threads and processes on the same host. The lock file is never
    unlinked: unlinking it would let a second writer lock a fresh
    inode while a first writer still holds the old one.
    """
    chunk = _chunk_path(chunk_dir)
    os.makedirs(chunk, exist_ok=True)
    fd = os.open(os.path.join(chunk, MANIFEST_LOCK_NAME), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _sha256_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (tmp file + fsync + rename)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    """Save ``array`` as .npy to ``path`` atomically."""
    tmp = _stage_npy(path, array)
    os.replace(tmp, path)


def _stage_npy(path: str, array: np.ndarray) -> str:
    """Write ``array`` beside ``path`` without replacing the target yet."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return tmp


def _backup_file(path: str) -> str:
    """Copy ``path`` to a same-directory backup and return the backup path."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, backup = tempfile.mkstemp(dir=directory, prefix=".backup-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as output, open(path, "rb") as source:
            shutil.copyfileobj(source, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.unlink(backup)
        except OSError:
            pass
        raise
    return backup


def read_manifest(chunk_dir: PathLike) -> dict[str, Any]:
    """Read the manifest for a chunk.

    Args:
        chunk_dir: Directory that holds the chunk's tensors.

    Returns:
        The manifest dict. If no manifest exists, returns a new
        empty manifest ``{"version": 1, "experts": {}}``.
    """
    path = _manifest_path(chunk_dir)
    if not os.path.exists(path):
        return {"version": MANIFEST_VERSION, "experts": {}}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    data.setdefault("version", MANIFEST_VERSION)
    data.setdefault("experts", {})
    if not isinstance(data["experts"], dict):
        raise ValueError(f"manifest 'experts' is not an object: {path}")
    return data


def write_manifest(chunk_dir: PathLike, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write ``manifest`` to a chunk directory atomically.

    Args:
        chunk_dir: Directory that holds the chunk's tensors.
        manifest: Manifest dict; ``version`` and ``experts`` are
            filled with defaults when absent.

    Returns:
        The manifest dict that was written.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    out: dict[str, Any] = dict(manifest)
    out.setdefault("version", MANIFEST_VERSION)
    out.setdefault("experts", {})
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(_manifest_path(chunk_dir), text.encode("utf-8"))
    return out


def save_expert_tensor(
    chunk_dir: PathLike,
    expert_id: str | int,
    array: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Save one expert tensor and update the chunk manifest atomically.

    The .npy file is written atomically (tmp + rename). Its SHA-256
    is computed from the written file bytes, the manifest is updated
    atomically, and the file is read back: the read-back hash must
    match and the reloaded shape/dtype must match the input array.
    The whole save runs under the per-chunk lock (see
    ``_chunk_lock``), so concurrent writers cannot interleave the
    manifest read-modify-write and silently drop an entry (F7).

    Args:
        chunk_dir: Per-chunk directory. Created if absent.
        expert_id: Expert identifier; used as the file stem
            ``<expert_id>.npy``. Must not contain path separators.
        array: Array-like data. Converted with ``np.asarray``.
        provenance: Dict that must carry ``model``, ``prompt``,
            ``seed``, ``host``, ``code_hash`` and
            ``determinism_class`` (``'local'`` or ``'api'``).
            ``timestamp`` is filled with current UTC time if absent.
            ``shape``, ``dtype`` and ``sha256`` are always set from
            the actual array / written file, overriding any supplied
            values, so the record cannot lie about the stored bytes.

    Returns:
        The manifest record for this expert (a dict with all
        provenance fields plus ``expert_id``, ``file``, ``shape``,
        ``dtype`` and ``sha256``).

    Raises:
        ValueError: On an invalid expert_id, missing/invalid
            provenance, or a read-back verification failure.
    """
    eid = _validate_expert_id(expert_id)
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a dict")
    missing = [k for k in _REQUIRED_PROVENANCE if k not in provenance]
    if missing:
        raise ValueError(f"provenance missing required fields: {missing}")
    if provenance["determinism_class"] not in _ALLOWED_DETERMINISM:
        raise ValueError(
            "determinism_class must be one of "
            f"{_ALLOWED_DETERMINISM}, got {provenance['determinism_class']!r}"
        )
    arr = np.asarray(array)
    chunk = _chunk_path(chunk_dir)
    with _chunk_lock(chunk):
        os.makedirs(chunk, exist_ok=True)
        target = _expert_path(chunk, eid)
        manifest_path = _manifest_path(chunk)
        manifest = read_manifest(chunk)
        staged = _stage_npy(target, arr)
        sha256 = _sha256_file(staged)

        # Read-back verification: hash + reload shape/dtype/value check.
        if _sha256_file(staged) != sha256:
            raise ValueError(f"readback hash changed for expert {eid!r} in {chunk}")
        try:
            loaded = np.load(staged, allow_pickle=False)
            if list(loaded.shape) != list(arr.shape) or str(loaded.dtype) != str(arr.dtype):
                raise ValueError(f"readback shape/dtype mismatch for expert {eid!r} in {chunk}")
            if not np.array_equal(loaded, arr):
                raise ValueError(f"readback data mismatch for expert {eid!r} in {chunk}")
        except BaseException:
            try:
                os.unlink(staged)
            except OSError:
                pass
            raise

        record: dict[str, Any] = {
            "expert_id": eid,
            "file": f"{eid}.npy",
            "model": provenance["model"],
            "prompt": provenance["prompt"],
            "seed": provenance["seed"],
            "host": provenance["host"],
            "code_hash": provenance["code_hash"],
            "timestamp": provenance.get("timestamp") or _utc_now(),
            "determinism_class": provenance["determinism_class"],
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "sha256": sha256,
        }
        # Preserve any extra provenance fields the caller supplied,
        # except the authoritative fields set above.
        for key, value in provenance.items():
            if key not in record:
                record[key] = value

        manifest["experts"][eid] = record
        manifest["updated_at"] = _utc_now()
        target_backup = _backup_file(target) if os.path.exists(target) else None
        manifest_backup = _backup_file(manifest_path) if os.path.exists(manifest_path) else None
        try:
            os.replace(staged, target)
            write_manifest(chunk, manifest)
            on_disk = read_manifest(chunk)
            stored = on_disk.get("experts", {}).get(eid)
            if stored is None or stored.get("sha256") != sha256:
                raise ValueError(f"manifest verification failed for expert {eid!r} in {chunk}")
            if _sha256_file(target) != stored["sha256"]:
                raise ValueError(f"readback hash mismatch for expert {eid!r} in {chunk}")
        except BaseException:
            if target_backup is not None:
                os.replace(target_backup, target)
            elif os.path.exists(target):
                os.unlink(target)
            if manifest_backup is not None:
                os.replace(manifest_backup, manifest_path)
            elif os.path.exists(manifest_path):
                os.unlink(manifest_path)
            raise
        for backup in (target_backup, manifest_backup):
            if backup is not None:
                try:
                    os.unlink(backup)
                except OSError:
                    pass
        return record


def load_expert_tensor(
    chunk_dir: PathLike,
    expert_id: str | int,
    verify: bool = True,
) -> np.ndarray:
    """Load one expert tensor from a chunk directory.

    Args:
        chunk_dir: Per-chunk directory.
        expert_id: Expert identifier (file stem ``<expert_id>.npy``).
        verify: If True (default), compare the file's SHA-256 with
            the manifest record and raise on mismatch. Verification
            requires a manifest entry for the expert; if none exists,
            a ValueError is raised because verification is impossible.

    Returns:
        The loaded numpy array.

    Raises:
        FileNotFoundError: If the .npy file does not exist.
        ValueError: On hash mismatch, missing manifest entry when
            ``verify`` is True, or shape/dtype mismatch with the
            manifest record.
    """
    eid = _validate_expert_id(expert_id)
    path = _expert_path(chunk_dir, eid)
    if not os.path.exists(path):
        raise FileNotFoundError(f"expert tensor not found: {path}")
    record: Any = None
    if verify:
        manifest = read_manifest(chunk_dir)
        record = manifest.get("experts", {}).get(eid)
        if record is None:
            raise ValueError(f"cannot verify expert {eid!r}: no manifest entry in {chunk_dir}")
        actual = _sha256_file(path)
        expected = record.get("sha256")
        if actual != expected:
            raise ValueError(
                f"sha256 mismatch for expert {eid!r}: file {actual} != manifest {expected}"
            )
    arr = np.load(path, allow_pickle=False)
    if verify and record is not None:
        if "shape" in record and [int(x) for x in record["shape"]] != list(arr.shape):
            raise ValueError(f"shape mismatch for expert {eid!r} in {chunk_dir}")
        if "dtype" in record and str(record["dtype"]) != str(arr.dtype):
            raise ValueError(f"dtype mismatch for expert {eid!r} in {chunk_dir}")
    return arr


def chunk_completeness(
    chunk_dir: PathLike,
    required_experts: Sequence[str | int],
) -> dict[str, Any]:
    """Check which required experts are present in a chunk.

    An expert counts as present when its ``<expert_id>.npy`` file
    exists in ``chunk_dir``.

    Args:
        chunk_dir: Per-chunk directory.
        required_experts: Expert identifiers that must be present.

    Returns:
        ``{"complete": bool, "present": [...], "missing": [...]}``
        with present/missing in the order of ``required_experts``
        (duplicates removed, first occurrence wins).
    """
    required: list[str] = []
    seen = set()
    for item in required_experts:
        eid = _validate_expert_id(item)
        if eid not in seen:
            seen.add(eid)
            required.append(eid)
    present: list[str] = []
    missing: list[str] = []
    for eid in required:
        if os.path.exists(_expert_path(chunk_dir, eid)):
            present.append(eid)
        else:
            missing.append(eid)
    return {"complete": len(missing) == 0, "present": present, "missing": missing}


def _cli_verify(chunk_dir: str, required: list[str]) -> dict[str, Any]:
    """Implement the ``verify`` CLI command. Returns a JSON-ready dict.

    Fail closed (owner ruling): verification fails when the chunk path
    is missing, the manifest is missing, the artifact set is empty, a
    .npy on disk is not listed in the
    manifest, or a manifest entry has no usable sha256. A pass
    requires every required expert present and hash-checked against
    a manifest entry, and no unlisted .npy files on disk.
    """
    if not os.path.isdir(chunk_dir):
        return {"complete": False, "present": [], "missing": [], "error": "chunk_dir_not_found"}
    if not os.path.exists(_manifest_path(chunk_dir)):
        if not required:
            return {
                "complete": False,
                "present": [],
                "missing": [],
                "error": "manifest_missing",
            }
        manifest: dict[str, Any] = {"version": MANIFEST_VERSION, "experts": {}}
    else:
        try:
            manifest = read_manifest(chunk_dir)
        except (OSError, ValueError):
            return {
                "complete": False,
                "present": [],
                "missing": [],
                "error": "manifest_unreadable",
            }
    if not required:
        required = sorted(manifest.get("experts", {}).keys())
    if not required:
        return {"complete": False, "present": [], "missing": [], "error": "empty_required_set"}
    try:
        result = chunk_completeness(chunk_dir, required)
    except ValueError:
        # An invalid expert id -- a manifest key the required set was
        # taken from, or an id named on the command line -- is a
        # verification failure with a JSON diagnostic, not a crash.
        return {
            "complete": False,
            "present": [],
            "missing": [],
            "error": "invalid_expert_id",
        }
    experts = manifest.get("experts", {})
    # Hash-check every present expert. A present expert with no
    # manifest entry, or no usable sha256 in its entry, is unverified
    # and fails the gate -- the same rule load_expert_tensor applies.
    mismatches: list[str] = []
    unverified: list[str] = []
    for eid in result["present"]:
        record = experts.get(eid)
        if not isinstance(record, dict) or not record.get("sha256"):
            unverified.append(eid)
        elif _sha256_file(_expert_path(chunk_dir, eid)) != record["sha256"]:
            mismatches.append(eid)
    # A manifest entry without a usable hash can never be verified,
    # even when its expert was not in the required set.
    for eid in sorted(experts):
        entry = experts[eid]
        if not isinstance(entry, dict) or not entry.get("sha256"):
            if eid not in unverified:
                unverified.append(eid)
    # A .npy on disk that the manifest does not list is unverified by
    # definition. Enumerate the directory; do not trust the required
    # set to name every file present.
    on_disk = {os.path.splitext(name)[0] for name in os.listdir(chunk_dir) if name.endswith(".npy")}
    unlisted = sorted(on_disk - set(experts))
    if not on_disk:
        result["error"] = "empty_artifact_set"
    if mismatches or unverified or unlisted or not on_disk:
        result["complete"] = False
    if mismatches:
        result["hash_mismatches"] = mismatches
    if unverified:
        result["unverified"] = sorted(unverified)
    if unlisted:
        result["unlisted"] = unlisted
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Usage:
        python3 persist.py verify <chunk_dir> [expert_id ...]

    Prints the completeness dict as JSON. When no expert ids are
    given, the required set is taken from the chunk manifest.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[0] == "verify":
        result = _cli_verify(args[1], args[2:])
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["complete"] else 1
    print("usage: python3 persist.py verify <chunk_dir> [expert_id ...]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
