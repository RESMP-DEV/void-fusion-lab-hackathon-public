# ruff: noqa: B017, UP031
# Formatting note: ported A-lane tests keep their original
# blind-exception asserts and % formatting verbatim.
"""Persist tests, ported from the baseline A-lane suites (synthetic
coverage plus failure modes) built against the persist module.

Port adaptations (only these):
  - the module under test is imported as samurai.persist (its canonical
    home, src/samurai/persist.py) instead of a sys.path insert of the
    module directory;
  - the failure suite's PROVENANCE dict is renamed PROVENANCE_F so the
    two suites keep their own provenance fixtures in one file;
  - the original docstrings (which named machine paths) are replaced by
    this header.
Test logic, assertions and expected values are the originals:
synth 6 tests, failure modes 11 tests.
"""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from samurai import persist

persist_mod = persist
PERSIST_PATH = str(Path(persist.__file__).resolve())
NUMPY_ERR = None

# Provenance supplied by the tests. Test 4 requires every field here to be
# present (same key, same value) in the stored record for that expert.
PROVENANCE = {
    # Required by persist.py _REQUIRED_PROVENANCE (read from the live
    # module after it landed, 2026-10-27): model, prompt, seed, host,
    # code_hash, determinism_class ('local' or 'api').
    "model": "synthetic-test-model",
    "prompt": "synthetic prompt",
    "seed": 42,
    "host": "synthetic-host",
    "code_hash": "synthetic-code-hash",
    "determinism_class": "local",
    # Extra fields: persist.py preserves extras in the record.
    "expert_id": "sam31",
    "source": "synthetic",
    "model_sha256": "0" * 64,
    "chunk_id": "synth-chunk-0001",
    "frame_range": [0, 4],
    "created_by": "A-W4-unit-test",
    "version": "1",
}


def prov_for(expert_id):
    p = dict(PROVENANCE)
    p["expert_id"] = expert_id
    return p


def unwrap_array(loaded):
    if isinstance(loaded, np.ndarray):
        return loaded
    if isinstance(loaded, tuple) and loaded and isinstance(loaded[0], np.ndarray):
        return loaded[0]
    if isinstance(loaded, dict):
        for k in ("array", "data", "tensor", "values"):
            if k in loaded and isinstance(loaded[k], np.ndarray):
                return loaded[k]
    raise AssertionError(f"load returned unexpected type: {type(loaded)!r}")


def normalise_completeness(result):
    """Return (complete: bool, missing: list[str])."""
    if isinstance(result, tuple) and len(result) == 2:
        complete, missing = result
        return bool(complete), list(missing or [])
    if isinstance(result, dict):
        complete = result.get("complete", result.get("is_complete"))
        missing = result.get("missing", result.get("missing_experts", []))
        return bool(complete), list(missing or [])
    complete = getattr(result, "complete", getattr(result, "is_complete", None))
    missing = getattr(result, "missing", getattr(result, "missing_experts", []))
    if complete is not None:
        return bool(complete), list(missing or [])
    raise AssertionError(f"chunk_completeness returned unexpected value: {result!r}")


def manifest_entries(chunk_dir):
    """Return {expert_id: record_dict} using manifest helpers if present,
    else by reading a manifest JSON file under chunk_dir."""
    for name in ("load_manifest", "read_manifest", "get_manifest", "manifest"):
        fn = getattr(persist, name, None)
        if callable(fn):
            try:
                data = fn(chunk_dir)
            except TypeError:
                try:
                    data = fn()
                except Exception:
                    data = None
            if data:
                return _entries_from(data)
    candidates = []
    for root, _dirs, files in os.walk(chunk_dir):
        for f in files:
            if "manifest" in f.lower() and f.lower().endswith(".json"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise AssertionError(f"no manifest JSON found under {chunk_dir}")
    with open(sorted(candidates)[0], encoding="utf-8") as fh:
        return _entries_from(json.load(fh))


def _entries_from(data):
    if isinstance(data, dict):
        experts = data.get("experts", data.get("tensors", data))
        if isinstance(experts, dict):
            out = {}
            for k, v in experts.items():
                if isinstance(v, dict):
                    rec = dict(v)
                    rec.setdefault("expert_id", k)
                    out[k] = rec
            if out:
                return out
        if isinstance(experts, list):
            return _entries_from(experts)
    if isinstance(data, list):
        out = {}
        for rec in data:
            if isinstance(rec, dict):
                eid = rec.get("expert_id", rec.get("expert", rec.get("id", rec.get("name"))))
                if eid:
                    out[str(eid)] = rec
        return out
    raise AssertionError(f"cannot parse manifest structure: {type(data)!r}")


def sha_of_record(rec):
    for k in ("sha256", "sha", "hash", "checksum", "sha_256"):
        if k in rec and isinstance(rec[k], str) and len(rec[k]) >= 32:
            return rec[k].lower()
    # provenance may be nested; sha may live at top level only per spec.
    raise AssertionError(f"no sha256 field in manifest record keys={sorted(rec)}")


def record_provenance(rec):
    """Provenance may be nested under 'provenance' or flattened in record."""
    merged = dict(rec)
    nested = rec.get("provenance")
    if isinstance(nested, dict):
        merged.update(nested)
    return merged


def candidate_hashes(chunk_dir, expert_id, array):
    hashes = {
        "raw_tobytes": hashlib.sha256(array.tobytes()).hexdigest(),
        "contig_tobytes": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }
    for root, _dirs, files in os.walk(chunk_dir):
        for f in files:
            if expert_id in f and "manifest" not in f.lower():
                with open(os.path.join(root, f), "rb") as fh:
                    hashes[f"file:{f}"] = hashlib.sha256(fh.read()).hexdigest()
    return hashes


class TestPersistSynth(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.chunk = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    # 1. Round-trip: byte-identical (shape, dtype, values), f32 and f16.
    def test_01_round_trip(self):
        rng = np.random.default_rng(42)
        for dtype in (np.float32, np.float16):
            arr = rng.standard_normal((4, 8, 8)).astype(dtype)
            persist.save_expert_tensor(self.chunk, "sam31", arr, prov_for("sam31"))
            loaded = unwrap_array(persist.load_expert_tensor(self.chunk, "sam31", verify=True))
            self.assertEqual(loaded.shape, arr.shape)
            self.assertEqual(loaded.dtype, arr.dtype)
            self.assertTrue(np.array_equal(loaded, arr))
            self.assertEqual(loaded.tobytes(), arr.tobytes())

    # 2. Manifest: 2 experts, sha256 matches independent hashlib.
    def test_02_manifest_hashes(self):
        rng = np.random.default_rng(7)
        arrays = {
            "sam31": rng.standard_normal((2, 4)).astype(np.float32),
            "samurai": rng.standard_normal((2, 4)).astype(np.float32),
        }
        for eid, arr in arrays.items():
            persist.save_expert_tensor(self.chunk, eid, arr, prov_for(eid))
        entries = manifest_entries(self.chunk)
        self.assertEqual(set(entries), {"sam31", "samurai"})
        for eid, arr in arrays.items():
            recorded = sha_of_record(entries[eid])
            cands = candidate_hashes(self.chunk, eid, arr)
            self.assertIn(
                recorded, cands.values(), f"{eid}: manifest sha {recorded} not in {cands}"
            )

    # 3. Completeness: 2 of 3 saved -> incomplete, missing is muse.
    #    Task text writes missing=['Muse']; compare case-insensitively so
    #    the check follows the required ids as passed in (['muse']).
    def test_03_completeness(self):
        rng = np.random.default_rng(3)
        for eid in ("sam31", "samurai"):
            arr = rng.standard_normal((2, 2)).astype(np.float32)
            persist.save_expert_tensor(self.chunk, eid, arr, prov_for(eid))
        complete, missing = normalise_completeness(
            persist.chunk_completeness(self.chunk, ["sam31", "samurai", "muse"])
        )
        self.assertFalse(complete)
        self.assertEqual([str(m).lower() for m in missing], ["muse"])

    # 4. Provenance: every required field present in the record.
    def test_04_provenance_fields(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        prov = prov_for("sam31")
        persist.save_expert_tensor(self.chunk, "sam31", arr, prov)
        entries = manifest_entries(self.chunk)
        self.assertIn("sam31", entries)
        stored = record_provenance(entries["sam31"])
        for key, value in prov.items():
            self.assertIn(key, stored, f"provenance field missing: {key}")
            self.assertEqual(stored[key], value, f"provenance field differs: {key}")

    # 5. dtype/shape edge: non-contiguous, 2D and 3D arrays.
    def test_05_noncontiguous_and_shapes(self):
        base = np.arange(64, dtype=np.float32).reshape(8, 8)
        noncontig = base[::2, ::2]
        self.assertFalse(noncontig.flags["C_CONTIGUOUS"])
        persist.save_expert_tensor(self.chunk, "sam31", noncontig, prov_for("sam31"))
        loaded = unwrap_array(persist.load_expert_tensor(self.chunk, "sam31", verify=True))
        self.assertEqual(loaded.shape, noncontig.shape)
        self.assertTrue(np.array_equal(loaded, noncontig))

        arr2d = np.ones((5, 7), dtype=np.float16)
        arr3d = np.ones((3, 5, 7), dtype=np.float16)
        persist.save_expert_tensor(self.chunk, "samurai", arr2d, prov_for("samurai"))
        persist.save_expert_tensor(self.chunk, "muse", arr3d, prov_for("muse"))
        self.assertTrue(
            np.array_equal(
                unwrap_array(persist.load_expert_tensor(self.chunk, "samurai", verify=True)), arr2d
            )
        )
        loaded3 = unwrap_array(persist.load_expert_tensor(self.chunk, "muse", verify=True))
        self.assertEqual(loaded3.shape, (3, 5, 7))
        self.assertTrue(np.array_equal(loaded3, arr3d))

    # 6. Overwrite: same expert twice -> hash updates, no duplicate entry.
    def test_06_overwrite(self):
        arr_v1 = np.zeros((2, 2), dtype=np.float32)
        arr_v2 = np.ones((2, 2), dtype=np.float32)
        persist.save_expert_tensor(self.chunk, "sam31", arr_v1, prov_for("sam31"))
        sha_v1 = sha_of_record(manifest_entries(self.chunk)["sam31"])
        persist.save_expert_tensor(self.chunk, "sam31", arr_v2, prov_for("sam31"))
        entries = manifest_entries(self.chunk)
        self.assertEqual(list(entries).count("sam31"), 1)
        self.assertEqual(len(entries), 1)
        sha_v2 = sha_of_record(entries["sam31"])
        self.assertNotEqual(sha_v1, sha_v2)
        self.assertIn(sha_v2, candidate_hashes(self.chunk, "sam31", arr_v2).values())
        loaded = unwrap_array(persist.load_expert_tensor(self.chunk, "sam31", verify=True))
        self.assertTrue(np.array_equal(loaded, arr_v2))


PROVENANCE_F = {
    "model": "synthetic-model",
    "prompt": "synthetic prompt",
    "seed": 1234,
    "host": "synthetic-host",
    "code_hash": "deadbeef",
    "determinism_class": "local",
}


class PersistFailureTests(unittest.TestCase):
    def setUp(self):
        if persist_mod is None or not os.path.exists(PERSIST_PATH):
            self.skipTest("persist.py absent: IMPLEMENTATION UNKNOWN (A-W9)")
        if np is None:
            self.skipTest("numpy unavailable: %r" % (NUMPY_ERR,))
        self.tmp = tempfile.mkdtemp(prefix="persist_fail_")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.chunk_dir = os.path.join(self.tmp, "chunk0000")
        self.arr = np.arange(64, dtype=np.float32).reshape(8, 8) / 7.0
        self.expert = "expert_0"

    # -- helpers (exact real signatures) ------------------------------------
    def _save(self, expert=None, arr=None, provenance=None):
        return persist_mod.save_expert_tensor(
            self.chunk_dir,
            self.expert if expert is None else expert,
            self.arr if arr is None else arr,
            dict(PROVENANCE_F) if provenance is None else provenance,
        )

    def _load(self, expert=None, verify=True):
        return persist_mod.load_expert_tensor(
            self.chunk_dir, self.expert if expert is None else expert, verify=verify
        )

    def _completeness(self, required=None):
        return persist_mod.chunk_completeness(
            self.chunk_dir, [self.expert] if required is None else required
        )

    def _npy(self, expert=None):
        return os.path.join(self.chunk_dir, (self.expert if expert is None else expert) + ".npy")

    def _manifest_path(self):
        return os.path.join(self.chunk_dir, "manifest.json")

    def _edit_manifest(self, fn):
        with open(self._manifest_path()) as fh:
            data = json.load(fh)
        fn(data)
        with open(self._manifest_path(), "w") as fh:
            json.dump(data, fh)
        return data

    # -- baseline ------------------------------------------------------------
    def test_baseline_save_load_roundtrip(self):
        rec = self._save()
        self.assertEqual(rec["sha256"] and len(rec["sha256"]), 64)
        out = self._load(verify=True)
        self.assertTrue(np.array_equal(out, self.arr))
        comp = self._completeness()
        self.assertEqual(comp, {"complete": True, "present": [self.expert], "missing": []})

    # -- F3: corrupt one byte -> load(verify=True) must raise, never serve ---
    def test_F3_corrupt_one_byte_load_must_raise(self):
        self._save()
        target = self._npy()
        with open(target, "r+b") as fh:
            fh.seek(-1, os.SEEK_END)
            b = fh.read(1)
            fh.seek(-1, os.SEEK_END)
            fh.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(Exception, msg="corrupted tensor was silently loaded"):
            self._load(verify=True)

    # -- F4a: delete tensor file -> completeness missing, load raises cleanly
    def test_F4a_deleted_tensor_completeness_missing_and_load_raises(self):
        self._save()
        os.remove(self._npy())
        comp = self._completeness()
        self.assertFalse(comp["complete"], "deleted expert reported complete: %r" % (comp,))
        self.assertIn(self.expert, comp["missing"])
        with self.assertRaises(FileNotFoundError):
            self._load(verify=True)

    # -- F4b: file present, manifest silent -> verified load must refuse -----
    def test_F4b_orphan_file_no_manifest_entry_load_must_refuse(self):
        self._save()  # creates chunk + manifest for expert_0 only
        orphan = "orphan_1"
        np.save(self._npy(orphan), self.arr)  # raw file, no manifest entry
        with self.assertRaises(Exception, msg="orphan file was served under verify=True"):
            self._load(expert=orphan, verify=True)

    # -- F3/F10: hand-edit manifest hash -> verify must fail -----------------
    def test_F3_manifest_hash_edited_verify_must_fail(self):
        self._save()

        def tamper(data):
            rec = data["experts"][self.expert]
            rec["sha256"] = ("0" if rec["sha256"][0] != "0" else "1") + rec["sha256"][1:]

        self._edit_manifest(tamper)
        with self.assertRaises(Exception, msg="load(verify=True) accepted tampered manifest hash"):
            self._load(verify=True)

    # -- F2: truncated .npy (partial write) -> refused, never silently loaded
    def test_F2_truncated_npy_must_be_refused(self):
        self._save()
        target = self._npy()
        size = os.path.getsize(target)
        self.assertGreater(size, 16)
        with open(target, "r+b") as fh:
            fh.truncate(max(10, size // 4))
        with self.assertRaises(Exception, msg="truncated tensor was silently loaded"):
            self._load(verify=True)

    # -- F7: manifest dtype/shape edited -> load must raise ------------------
    def test_F7_manifest_dtype_edited_must_fail(self):
        self._save()

        def tamper(data):
            data["experts"][self.expert]["dtype"] = "int64"

        self._edit_manifest(tamper)
        with self.assertRaises(
            Exception, msg="load accepted dtype that disagrees with manifest/file"
        ):
            self._load(verify=True)

    def test_F7_manifest_shape_edited_must_fail(self):
        self._save()

        def tamper(data):
            data["experts"][self.expert]["shape"] = [9, 8]

        self._edit_manifest(tamper)
        with self.assertRaises(
            Exception, msg="load accepted shape that disagrees with manifest/file"
        ):
            self._load(verify=True)

    # -- F8: provenance field missing at save -> save must refuse ------------
    def test_F8_provenance_missing_field_save_must_refuse(self):
        bad = dict(PROVENANCE_F)
        del bad["code_hash"]
        with self.assertRaises(Exception, msg="save accepted provenance without code_hash"):
            self._save(provenance=bad)
        self.assertFalse(
            os.path.exists(self._npy()), "tensor file was written despite bad provenance"
        )

    # -- F10: corrupt manifest JSON -> verified load must raise --------------
    def test_F10_corrupt_manifest_load_must_raise(self):
        self._save()
        with open(self._manifest_path(), "w") as fh:
            fh.write("{not json")
        with self.assertRaises(Exception, msg="load served tensor with corrupt manifest"):
            self._load(verify=True)

    # -- F12: unsafe expert id -> save and load must refuse -------------------
    def test_F12_unsafe_expert_id_must_refuse(self):
        with self.assertRaises(Exception):
            self._save(expert="../evil")
        with self.assertRaises(Exception):
            self._load(expert="../evil", verify=True)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.npy")))


if __name__ == "__main__":
    unittest.main(verbosity=2)


def run_verify_cli(chunk_dir, *expert_ids):
    """Run the persist CLI verify command; return (exit_code, parsed_json)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = persist.main(["verify", str(chunk_dir), *expert_ids])
    return code, json.loads(buf.getvalue())


class TestPersistVerifyFailClosed(unittest.TestCase):
    """Item #5 (owner ruling): CLI verify must fail closed.

    Each fail case below passed verification on the pre-fix code
    (verified 2026-10-28, L3-persist-verify-verification.md).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.chunk = os.path.join(self._tmp.name, "chunk")

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, eid, chunk=None, value=1.0):
        arr = np.full((2, 2), value, dtype=np.float32)
        return persist.save_expert_tensor(chunk or self.chunk, eid, arr, dict(PROVENANCE_F))

    def _raw(self, eid, chunk=None):
        target = chunk or self.chunk
        os.makedirs(target, exist_ok=True)
        np.save(os.path.join(target, eid + ".npy"), np.zeros((2, 2), dtype=np.float32))

    # Case 1: missing manifest.
    def test_verify_missing_manifest_fails_closed(self):
        self._raw("expert1")
        self._raw("expert2")
        code, out = run_verify_cli(self.chunk)
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["error"], "manifest_missing")

    def test_verify_missing_manifest_explicit_id_fails_closed(self):
        self._raw("expert1")
        code, out = run_verify_cli(self.chunk, "expert1")
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["unverified"], ["expert1"])

    # Case 2: missing path.
    def test_verify_missing_path_fails_closed(self):
        code, out = run_verify_cli(os.path.join(self._tmp.name, "does-not-exist"))
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["error"], "chunk_dir_not_found")

    # Case 3: empty artifact set.
    def test_verify_empty_artifact_set_fails_closed(self):
        os.makedirs(self.chunk)
        Path(self.chunk, "manifest.json").write_text(
            json.dumps({"version": 1, "experts": {}}), encoding="utf-8"
        )
        code, out = run_verify_cli(self.chunk)
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["error"], "empty_required_set")
        code, out = run_verify_cli(self.chunk, "expert1")
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["error"], "empty_artifact_set")
        self.assertEqual(out["missing"], ["expert1"])

    # Case 4: unlisted .npy on disk that the manifest does not name.
    def test_verify_unlisted_npy_fails_closed(self):
        self._save("listed")
        self._raw("extra")
        code, out = run_verify_cli(self.chunk)
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["unlisted"], ["extra"])
        code, out = run_verify_cli(self.chunk, "listed", "extra")
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["unverified"], ["extra"])

    # Case 5: manifest entry without a usable hash.
    def test_verify_manifest_entry_without_usable_hash_fails_closed(self):
        for mutate in ("delete", "empty"):
            with self.subTest(mutate=mutate):
                chunk = os.path.join(self._tmp.name, "chunk-" + mutate)
                self._save("expert1", chunk=chunk)
                mpath = Path(chunk, "manifest.json")
                data = json.loads(mpath.read_text(encoding="utf-8"))
                if mutate == "delete":
                    del data["experts"]["expert1"]["sha256"]
                else:
                    data["experts"]["expert1"]["sha256"] = ""
                mpath.write_text(json.dumps(data), encoding="utf-8")
                code, out = run_verify_cli(chunk)
                self.assertEqual(code, 1)
                self.assertFalse(out["complete"])
                self.assertEqual(out["unverified"], ["expert1"])

    # Controls: valid cases still pass, prior failure modes still fail.
    def test_verify_wellformed_chunk_passes_with_unchanged_shape(self):
        self._save("listed")
        code, out = run_verify_cli(self.chunk)
        self.assertEqual(code, 0)
        self.assertEqual(out, {"complete": True, "missing": [], "present": ["listed"]})

    def test_verify_tampered_tensor_still_fails(self):
        self._save("listed")
        self._raw("listed")
        code, out = run_verify_cli(self.chunk)
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["hash_mismatches"], ["listed"])

    def test_verify_missing_required_expert_still_fails(self):
        self._save("listed")
        code, out = run_verify_cli(self.chunk, "listed", "ghost")
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["missing"], ["ghost"])

    # Case 6: an invalid expert id -- a manifest key the required
    # set is taken from, or an id named on the command line --
    # returns the JSON diagnostic, never a traceback (R3-09).
    def test_verify_invalid_expert_id_returns_json_not_traceback(self):
        os.makedirs(self.chunk)
        Path(self.chunk, "manifest.json").write_text(
            json.dumps({"version": 1, "experts": {"bad/id": {"sha256": "abc"}}}),
            encoding="utf-8",
        )
        code, out = run_verify_cli(self.chunk)
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["error"], "invalid_expert_id")
        code, out = run_verify_cli(self.chunk, "bad/id")
        self.assertEqual(code, 1)
        self.assertFalse(out["complete"])
        self.assertEqual(out["error"], "invalid_expert_id")


class TestPersistConcurrentSaves(unittest.TestCase):
    """Item #6 / F7: concurrent saves into one chunk must not lose entries."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.chunk = os.path.join(self._tmp.name, "chunk")

    def tearDown(self):
        self._tmp.cleanup()

    def _run_writers(self, n, slow_read=False):
        barrier = threading.Barrier(n)

        def worker(i):
            barrier.wait(timeout=30)
            arr = np.full((4, 4), float(i), dtype=np.float32)
            return persist.save_expert_tensor(
                self.chunk, "expert_%02d" % i, arr, dict(PROVENANCE_F)
            )

        original = persist.read_manifest

        def slowed(*args, **kwargs):
            data = original(*args, **kwargs)
            time.sleep(0.05)
            return data

        patch_ctx = (
            mock.patch.object(persist, "read_manifest", wraps=slowed)
            if slow_read
            else contextlib.nullcontext()
        )
        with patch_ctx, ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(worker, i) for i in range(n)]
            for fut in futures:
                fut.result(timeout=120)

    def _assert_all_present(self, n):
        entries = persist.read_manifest(self.chunk)["experts"]
        self.assertEqual(set(entries), {"expert_%02d" % i for i in range(n)})
        for i in range(n):
            arr = persist.load_expert_tensor(self.chunk, "expert_%02d" % i, verify=True)
            self.assertTrue(np.array_equal(arr, np.full((4, 4), float(i), dtype=np.float32)))
        code, out = run_verify_cli(self.chunk)
        self.assertEqual(code, 0, out)
        self.assertTrue(out["complete"])

    def test_concurrent_writers_distinct_experts_all_survive(self):
        self._run_writers(12)
        self._assert_all_present(12)

    def test_concurrent_writers_slowed_manifest_read_all_survive(self):
        # Regression test for F7. Slowing read_manifest widens the
        # read-modify-write window deterministically: on the unlocked
        # code every writer reads the same starting manifest and the
        # last write wins, dropping the other entries. The per-chunk
        # lock serializes the writers, so all entries survive.
        self._run_writers(8, slow_read=True)
        self._assert_all_present(8)


class TestSaveRollback(unittest.TestCase):
    def test_failed_manifest_write_restores_previous_tensor_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix="persist_rollback_") as td:
            chunk = os.path.join(td, "chunk")
            persist.save_expert_tensor(chunk, "expert", np.full((2, 2), 1.0), dict(PROVENANCE_F))
            target = os.path.join(chunk, "expert.npy")
            manifest_path = os.path.join(chunk, persist.MANIFEST_NAME)
            old_tensor = Path(target).read_bytes()
            old_manifest = Path(manifest_path).read_bytes()

            with mock.patch.object(
                persist, "write_manifest", side_effect=OSError(" simulated manifest failure")
            ):
                with contextlib.suppress(OSError):
                    persist.save_expert_tensor(
                        chunk, "expert", np.full((2, 2), 9.0), dict(PROVENANCE_F)
                    )

            self.assertEqual(Path(target).read_bytes(), old_tensor)
            self.assertEqual(Path(manifest_path).read_bytes(), old_manifest)
            self.assertTrue(
                np.array_equal(
                    persist.load_expert_tensor(chunk, "expert", verify=True),
                    np.full((2, 2), 1.0),
                )
            )
