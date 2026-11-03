# Expert Tensor Persistence Specification

> **Public-export note (2026-11-03):** every machine path in this document
> is a placeholder (`<research-root>/...`); the unredacted original is not
> distributed with this repository. The file-name and schema
> boundary between the layers that exist in this repository is recorded
> in section 8. **Conflict C-1 is UNRESOLVED
> and is flagged here, not decided:** section 2 of this spec mandates the
> layout `experts/<expert_id>/<expert_id>__<model_slug>__prob.npy` with a
> manifest `experts` LIST, while the staged module `src/samurai/persist.py`
> implements a flat `<chunk_dir>/<expert_id>.npy` layout with a manifest
> `experts` DICT (its docstring calls that layout the default used where
> this spec is absent). Neither text silently overrules the other; one
> Phase-2 decision must align them.

- Spec version: `1.0.0` (see §7, field `spec_version`)
- Status: SPEC ONLY. No code, no production change, no run authorised by this document.
- Date: 2026-10-27
- Lane / owner: A-W2
- Terminology: aligned to the companion cache design record (K1). Terms `determinism_class`, `PINNED`, `code_closure_hash`, `entry manifest`, `quarantine`, `MISS` keep the K1 meanings. Where this spec and K1 differ, K1 governs cache serving and this spec governs the act of persisting an expert tensor.
- ASD-STE100 style: plain, precise, one action per step.

## 0. Defect this spec closes

1. Proven defect (verified 2026-10-27, on the primary host): for the recorded chunk only two expert tensors exist on disk:
   - `sam31_prob.npy` — MD5 `872fe31c0de3c43b20b13b2c5f204a1f` at `<research-root>/<chunk-slug>/sam31_prob.npy`
   - `samurai_prob.npy` — MD5 `c9a88de786f78c5d0a333131d2e80caf` at `<research-root>/<chunk-slug>/samurai_prob.npy`
   - Same pair, same naming, also at `<research-root>/<second-chunk-slug>/` and `<research-root>/<fusion-proof>/results/`.
2. The third fusion expert, `muse-spark-baseten` (model `meta-models/muse-spark-open`, via Baseten), exists nowhere as a tensor. Its contribution lived only in memory during fusion. K1 §1 C3 states the same fact.
3. Consequence: weighted fusion cannot be re-run or re-calibrated from stored data.
   - Fusion config on record (`<research-root>/<chunk-slug>/ens.json`): Samurai weight `0.4`, SAM 3.1 weight `0.35`, Muse weight `0.25`; threshold `0.5`; `dynamic_weights: "entropy"`; Muse `temperature: 1.4` (tensor experts `1.0`); `top_k_ratio: 0.02`, `min_expert_support: 1`, `support_threshold: 0.5`.
   - Dawid-Skene and any other multi-expert calibration need at least 3 aligned expert tensors. Below 3 aligned tensors the calibration MUST fail closed. It MUST NOT silently fall back to 2 experts.
4. Rule that follows: an expert that contributes to a fusion MUST persist its tensor before, or atomically with, the fusion that consumes it. In-memory-only contribution is prohibited from spec `1.0.0` onward.

## 1. Scope

1. This spec covers per-frame probability tensors produced by fusion experts for one chunk, plus — for API experts — the raw response artifact that the tensor was rasterised from.
2. This spec does NOT cover: fused tensors, fused masks, quadmask, plates, pose, or model weights. Those are K1 stages C4–C12 and keep their own rules.
3. A tensor under this spec is a K1 cache entry candidate (stages C1, C2, C3). Persisting under this spec does not by itself admit a tensor to the K1 cache; K1 §3–§4 govern admission, eviction, and serving.

## 2. On-disk format

### 2.1 Directory layout per chunk

Canonical root on the primary store, mirroring K1 §3.1:

```
<store>/chunks/<chunk_id>/
    manifest.json
    experts/
        <expert_id>/
            <expert_file>.npy
            response.json            # API experts only, §2.4; absent for local experts
    fused/                           # NOT governed by this spec; reserved, may be absent
```

- `<store>` default: `<research-root>/quadmask/persist/store`. A different store root MUST be recorded in `manifest.json:store_root`.
- `<chunk_id>` pattern, exact: `<source_slug>-f<first_frame>-f<last_frame>-<yyyymmddThhmmssZ>`
  - Example (the recorded chunk, retro-fitted): `<source-slug>-f0-f95-20261027T061545Z`
  - `<source_slug>`: lower-case `[a-z0-9-]`, identifies the source clip only, never a person.
  - Frame indices are inclusive, zero-based, and are frame INDEX values, never timestamps (K1 §2.1, P2 rule).
  - The timestamp is the chunk run start time in UTC.
- One directory = one chunk = one `manifest.json`. Do not place tensors from two chunks in one directory.
- Do not reuse a `<chunk_id>`. A re-run creates a new `<chunk_id>` with its own timestamp.
- Legacy locations (`<research-root>/<chunk-slug>/`, `<research-root>/<second-chunk-slug>/`, `<research-root>/<fusion-proof>/results/`) are read-only precedent. Do not write new tensors there. Backfill into the canonical layout is a separate task and MUST copy, never move, legacy files.

### 2.2 Per-expert file naming

Pattern, exact:

```
experts/<expert_id>/<expert_id>__<model_slug>__prob.npy
```

- `<expert_id>`: closed set for spec `1.0.0`; extend only by a spec minor-version bump (§7):
  - `samurai` — Samurai tracker (SAM 2.1 family)
  - `sam31` — SAM 3.1 anchor / segmentation
  - `muse` — Muse Spark API expert
  - `glm`, `minimax`, `meta-api` — reserved API expert ids, unused until named in a chunk manifest
- `<model_slug>`: model id with `/` replaced by `-`, lower-case, e.g.:
  - `samurai__sam2.1-large__prob.npy`
  - `sam31__sam3.1-multiplex__prob.npy`
  - `muse__meta-models-muse-spark-open__prob.npy`
- Legacy alias note: legacy files `sam31_prob.npy` and `samurai_prob.npy` map to expert ids `sam31` and `samurai`. A backfill manifest MUST record the legacy filename in `experts[].legacy_path`. New writes MUST NOT use the legacy filename.
- Exactly one `.npy` per expert per chunk. A second tensor from the same expert for the same chunk is a new chunk directory, not a second file.

### 2.3 Array format — decision

DECISION: `.npy`, NumPy format v1.0, dtype `float16` (`<f2`, little-endian), `fortran_order: false` (C order), shape `(T, H, W)`, uncompressed. No `.npz`. No header padding variation: write with `numpy.save` defaults.

Justification:

1. Precedent: every stored expert tensor on record is exactly this format. Verified header of `<research-root>/<chunk-slug>/sam31_prob.npy`: `{'descr': '<f2', 'fortran_order': False, 'shape': (96, 720, 1280)}`; file size 176,947,328 bytes = 128-byte header + `96 × 720 × 1280 × 2` bytes.
2. Comparability: hashes, shape checks, and fusion replay are comparable across chunks only if the format does not change per chunk. A `float32` chunk and a `float16` chunk are not aligned inputs for calibration without an explicit, recorded conversion.
3. Fusion math does NOT need `float32` on disk:
   - Step 1: the fusion loader upcasts each tensor to `float32` in memory before any arithmetic.
   - Step 2: weighted sum (weights `0.4 / 0.35 / 0.25`), entropy-derived dynamic weights, and temperature scaling (`1.4` for Muse, `1.0` for tensor experts) run in `float32` or higher.
   - Step 3: thresholding at `0.5` runs on the `float32` fused result, never on the stored `float16` values directly.
   - Therefore on-disk precision bounds only the input quantisation, which is already the precision the recorded fusions used.
4. Cost of `float32` on disk: exactly 2× bytes — 353,894,400 bytes of data per 96-frame expert at 720×1280, about 23 GB for a 3-expert full-clip set becomes about 46 GB (derived from K1 §3.3). This cost buys no new information for a local model whose native output was already reduced to `float16`, and it breaks precedent comparability.
5. Exception rule: if a future expert produces native `float32` probabilities AND a calibration record shows `float16` quantisation changes a fused verdict, that expert MAY persist `float32` only under a spec minor-version bump (§7), with `dtype: "float32"` in the manifest and the reason recorded in `experts[].notes`. Mixed dtypes in one chunk are then allowed, and the fusion loader MUST upcast all to `float32`. Silent per-run dtype choice is prohibited.
6. Value domain: probabilities in `[0.0, 1.0]`. Values outside that range, `NaN`, or `Inf` make the write FAIL (§5.3). Do not clip silently; clipping is a producing-code decision and, if applied, MUST be recorded in `experts[].notes`.
7. Axis order is fixed: `(frame, height, width)` = `(T, H, W)`. Any other order is a different format and needs a major spec bump.

### 2.4 API expert response artifact

1. For every API expert (K1 Class C, §4), persist the raw response bytes that produced the tensor, in the same expert directory, filename exactly `response.json` (or `response.bin` plus `response_content_type` in the manifest if the provider does not return JSON).
2. The response artifact is PINNED (K1 §3.3): never auto-evicted. It is the only audit copy of that expert's work.
3. Persist the tensor rasterised from that response as well, in the §2.3 format. Replay rule (K1 §2.5 Class C): a later replay re-rasterises from `response.json` through the recorded rasteriser code hash; it does not re-call the provider.
4. Never persist an API key, authorisation header, or signed URL. The manifest records provider and model id only (§3, `licence_privacy_notes` and the redaction rule in §3.3).

### 2.5 Compression — decision

DECISION: no compression of the canonical `.npy`.

1. Reason 1 — access: uncompressed `.npy` supports memory-mapping and per-frame slicing; compressed formats force a full decompress before fusion or eval can read one frame.
2. Reason 2 — hash stability: the manifest hash (§3) is over the exact stored bytes. Compression introduces encoder-version variance into a byte identity that K1 keys and provenance rely on.
3. Reason 3 — size is already bounded: `float16` is the compact form; the 96-frame expert is 176,947,200 bytes of data.
4. Transport compression (for example `rsync -z`, or a `.zst` copy made only for transfer) is allowed. The stored canonical bytes remain the uncompressed `.npy`, and verification (§5) runs on the stored bytes after transfer and decompression.
5. Archival compression after a chunk is sealed is out of scope for `1.0.0`. Propose it by spec amendment, not by local practice.

## 3. Manifest — exact JSON schema per chunk

### 3.1 File

- Filename: exactly `manifest.json`, at the chunk directory root (§2.1).
- Encoding: UTF-8, JSON, keys in the order shown, 1-space indent is the recorded precedent style in `ens.json`; any valid JSON formatting is acceptable because the manifest is hashed as stored (§5.2) — formatting changes change the hash and therefore need a re-seal (§5.4).
- One manifest per chunk. No sidecar manifests.

### 3.2 Schema

```json
{
  "spec_version": "1.0.0",
  "chunk_id": "<source-slug>-f0-f95-20261027T061545Z",
  "status": "COMPLETE",
  "created_at": "2026-10-27T06:15:45Z",
  "sealed_at": "2026-10-27T06:20:00Z",
  "store_root": "<research-root>/quadmask/persist/store",
  "source": {
    "clip_path": "<research-root>/<chunk-slug>/clip.mp4",
    "clip_md5": "<md5 hex, 32 lower-case>",
    "clip_sha256": "<sha-256 hex, 64 lower-case>",
    "first_frame": 0,
    "last_frame": 95,
    "frame_count": 96,
    "width": 1280,
    "height": 720,
    "fps": 23.976023976023978
  },
  "fusion_intent": {
    "experts_expected": ["samurai", "sam31", "muse"],
    "weights": {"samurai": 0.4, "sam31": 0.35, "muse": 0.25},
    "threshold": 0.5,
    "dynamic_weights": "entropy",
    "temperature": {"samurai": 1.0, "sam31": 1.0, "muse": 1.4},
    "min_aligned_experts_for_calibration": 3
  },
  "experts": [
    {
      "expert_id": "sam31",
      "model_name": "SAM 3.1 multiplex",
      "model_version": "sam3.1_multiplex.pt",
      "model_id": "facebookresearch/sam3@7f473cf",
      "model_checkpoint_sha256": "<sha-256 hex of checkpoint file, or null with reason in notes>",
      "path": "experts/sam31/sam31__sam3.1-multiplex__prob.npy",
      "legacy_path": null,
      "bytes": 176947328,
      "shape": [96, 720, 1280],
      "dtype": "float16",
      "md5": "<md5 hex, 32 lower-case>",
      "sha256": "<sha-256 hex, 64 lower-case>",
      "prompt_text": "<exact prompt string, or null for a non-prompt expert>",
      "seed": null,
      "conditioning": {
        "kind": "seed_box | init_box | prompt_only | none",
        "box_xyxy": [0, 0, 0, 0],
        "box_frame": 0,
        "normalized_coordinates": false
      },
      "host": "<host>",
      "determinism_class": "B",
      "code_hash": {
        "kind": "git_sha | file_md5",
        "value": "<git SHA of producing repo, or MD5 of producing script>",
        "script_path": "<path of producing script as executed>"
      },
      "code_closure_hash": "<sha-256 hex per K1 §2.2, or null with reason in notes>",
      "environment": {
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "tf32": true,
        "offload_mode": null
      },
      "response_artifact": null,
      "timestamp": "2026-10-27T06:12:00Z",
      "licence_privacy_notes": "Local open-weights model. No third-party data leaves the host. Tensor describes a person in the source clip; treat as personal-data-bearing, do not publish.",
      "notes": ""
    }
  ],
  "missing_experts": [],
  "completeness": {
    "rule": "COMPLETE iff every id in fusion_intent.experts_expected has one verified experts[] entry",
    "verified_at": "2026-10-27T06:20:00Z",
    "verifier": "<agent or script id that ran §5 verification>"
  }
}
```

### 3.3 Field rules

1. Required per expert, no exceptions: `expert_id`, `model_name`, `model_version`, `path`, `bytes`, `shape`, `dtype`, `md5`, `sha256`, `prompt_text` (value or explicit `null`), `seed` (value or explicit `null`), `conditioning`, `host`, `code_hash`, `timestamp`, `determinism_class`, `licence_privacy_notes`.
   - A field that does not apply is `null` plus a one-line reason in `notes`. An absent field makes the manifest INVALID.
2. `model_name` + `model_version` are human identity. `model_id` and `model_checkpoint_sha256` are the strong identity where they exist. Reason: K1 §2.1 — a name or version string never substitutes for a checkpoint hash. If the checkpoint hash cannot be obtained, write `null` and state why in `notes`; do not invent a hash.
3. `prompt_text` is the EXACT string sent to the model, byte for byte. Example of the recorded shape for Muse: `Locate the seated person in the sleeveless shirt on the right, beside the small robed figure.` Paraphrase in the manifest is a defect.
4. `seed`: the integer seed as executed, or `null` if the expert has no seed. If a config seed differs from the executed seed, record the executed seed and list the config field in `notes` as a dead key (K1 §2.3).
5. `conditioning`: record the box or conditioning that actually ran. For Samurai, the init box. For SAM 3.1, the seed box / prompt conditioning. For an API expert, the box / normalised-coordinate flag as sent (on record for Muse: `normalized_coordinates: true`, `boundary_sigma: 1.5` — extra provider parameters go in `conditioning.extra` as an object).
6. `code_hash.kind`: use `git_sha` only if the producing tree was clean at that commit; otherwise use `file_md5` of the exact script file that executed (K1 §2.2: repo HEAD does not identify what ran on a dirty tree).
7. `timestamp`: per-expert completion time, UTC, format `YYYY-MM-DDThh:mm:ssZ`. `created_at` / `sealed_at` at chunk level use the same format.
8. `determinism_class`: exactly one of (K1 §2.5):
   - `A` — byte-deterministic, proven by a double-run record. Claim only with that record cited in `notes`.
   - `B` — deterministic modulo declared environment. Default for local GPU experts (SAM 3.1, Samurai). `environment` is then required and key-relevant.
   - `C` — nondeterministic. Mandatory for every API expert (Muse, GLM, MiniMax, Meta API). `response_artifact` is then required: `{"path": "experts/muse/response.json", "md5": "...", "sha256": "...", "bytes": 0, "pinned": true}`.
   - Local-model vs API-response is therefore a recorded field, not an inference: local model = `A` or `B` with `response_artifact: null`; API response = `C` with a PINNED response artifact.
9. `licence_privacy_notes`: state, per expert, (a) model licence class in plain words, (b) whether data left the host, (c) that a probability tensor over a named or visible person is personal-data-bearing. Never write a credential, a signed URL, or a private address in this field or anywhere in the manifest.
10. Hashes: `md5` for precedent compatibility (all existing records use MD5), `sha256` required in addition for tensors (K1 §3.2: SHA-256 required for tensors / weights class). Both are over the exact stored file bytes.
11. `shape` order matches the array: `[T, H, W]`. `shape` MUST equal the `.npy` header shape and MUST have `T = source.frame_count`, `H = source.height`, `W = source.width`. A mismatch makes the entry INVALID (§5.3).

## 4. Completeness rule

1. A chunk manifest has exactly two allowed `status` values: `COMPLETE` or `INCOMPLETE`.
2. `COMPLETE` iff ALL of these hold:
   - Every `expert_id` in `fusion_intent.experts_expected` has exactly one `experts[]` entry.
   - Every such entry passed §5 verification-on-write (readback hash matches, header matches manifest).
   - For every Class C expert, the PINNED response artifact exists and its hashes verify.
3. Otherwise the status is `INCOMPLETE`, and `missing_experts` MUST list every expected `expert_id` that has no verified entry. An `INCOMPLETE` manifest with an empty `missing_experts` list is INVALID.
4. Worked example — the recorded chunk under this spec: `experts_expected = ["samurai", "sam31", "muse"]`, verified entries = `samurai`, `sam31`, therefore `status: "INCOMPLETE"`, `missing_experts: ["muse"]`. This is the correct retro-fitted stamp for every existing stored chunk. Do not stamp them `COMPLETE`.
5. Consumer rule (fail closed):
   - A fusion re-run or a Dawid-Skene / calibration run MUST refuse an `INCOMPLETE` chunk if the missing expert contributed to the original fusion.
   - A calibration run MUST refuse any chunk with fewer than 3 aligned verified expert tensors (`fusion_intent.min_aligned_experts_for_calibration: 3`), regardless of `status`.
   - Refusal is an explicit error naming the chunk and the missing experts. Silent 2-expert fallback is prohibited.
   - A consumer MAY use an `INCOMPLETE` chunk for single-expert inspection or display. It MUST label the output with the chunk status and the missing list.
6. Status changes only by re-sealing: add the missing verified entry, re-run §5, write a new `manifest.json` by the §5 atomic protocol, and update `sealed_at` and `completeness.verifier`. Do not edit `status` in place without re-verification.

## 5. Atomic write protocol and verification-on-write

### 5.1 Write steps (one action per step)

1. Step 1: the producer writes the tensor to a temp file in the SAME expert directory: `<expert_file>.npy.tmp-<pid>-<epoch>`. A temp file in another filesystem is prohibited (rename would stop being atomic).
2. Step 2: the producer flushes and `fsync`s the temp file.
3. Step 3: the producer closes the temp file.
4. Step 4: the verifier reads the temp file back and computes MD5 and SHA-256 over its bytes.
5. Step 5: the verifier opens the `.npy` header and checks `descr`, `fortran_order`, and `shape` against the intended manifest entry, and checks the value domain (§2.3 rule 6).
6. Step 6: if any check in Step 4 or Step 5 fails, the writer deletes the temp file, records the failure in the run log, and the expert is treated as missing (§4). It MUST NOT rename a failed temp file into place.
7. Step 7: on pass, the writer renames the temp file to the final filename with `rename(2)` / `os.replace` (atomic, same directory).
8. Step 8: the writer `fsync`s the expert directory, then the chunk directory.
9. Step 9: the writer repeats Steps 1–8 for the API `response.json` artifact, for Class C experts, before the manifest is sealed.
10. Step 10: the writer writes `manifest.json` last, by the same protocol: write `manifest.json.tmp-<pid>-<epoch>`, `fsync`, verify by re-reading and parsing it as JSON and re-checking every listed hash against the files now on disk, then rename into place and `fsync` the chunk directory. Sealing the manifest is the commit point: a chunk without a sealed `manifest.json` does not exist for consumers.

### 5.2 Readback verification

1. Verification-on-write is a READBACK hash: hash the bytes read from disk after write, not the in-memory buffer.
2. A consumer that reads a tensor for fusion or calibration MUST re-hash the file before use and compare to the manifest (K1 §0.3, §4.2). On mismatch: treat as a MISS, move the chunk expert entry to quarantine per K1 §4.2, and do not serve or fuse the bytes. Never repair an entry in place.

### 5.3 Failure conditions (any one = write FAILS, expert counts as missing)

- Readback MD5 or SHA-256 differs from the hash of the bytes the producer intended to write.
- Header `shape` / `dtype` differs from the manifest entry, or from `(frame_count, height, width)`.
- File `bytes` differs from `128 + T × H × W × element_size` for a `numpy.save` v1.0 file (allow only the exact header length read from the file itself; record that length implicitly via `bytes`).
- Any value is `NaN`, `Inf`, or outside `[0.0, 1.0]`.
- The temp file, the rename, or an `fsync` returns an error.

### 5.4 Re-seal

Replacing a tensor or a manifest field for an existing chunk follows §5.1 in full and produces a new `sealed_at`. The old manifest bytes are preserved as `manifest.prev-<sealed_at>.json` in the chunk directory before the rename. Do not overwrite history silently.

## 6. Retention and PINNED class

1. PINNED — never auto-evicted (K1 §3.3): (a) every Class C `response.json` artifact, (b) every tensor cited as evidence by a run record or a PR, (c) every tensor belonging to a gold-set / eval reference chunk.
2. Non-pinned tensors follow K1 §3.3 budgets and value-density eviction. Eviction deletes the tensor bytes, sets that expert entry's `notes` to an eviction tombstone with date and order reference, and re-seals the manifest as `INCOMPLETE` with the evicted expert in `missing_experts` — because, for fusion purposes, an evicted tensor is a missing tensor.
3. No deletion under this spec is authorised by this spec. Deletion needs its own explicit order (one-way door).

## 7. Versioning

1. Every manifest carries `spec_version` (semantic version, string). This document is `1.0.0`.
2. PATCH (`1.0.x`): wording or example fixes only. No schema, format, or rule change. Old manifests remain valid without action.
3. MINOR (`1.x.0`): additive changes only — a new optional field, a new `expert_id`, the §2.3 rule-5 `float32` exception. Readers at an older minor version MUST ignore unknown fields and MUST still verify all fields they know. A writer MUST NOT require a new minor field for a chunk to be `COMPLETE` until a major bump makes it required.
4. MAJOR (`x.0.0`): any breaking change — array axis order, default dtype, naming pattern, a new required field, a changed completeness rule.
   - Migration rule: old chunks are NEVER rewritten in place to a new major version. A migration copies the chunk to a new `<chunk_id>` directory under the new spec, verifies every hash in the copy (§5), writes a new manifest at the new `spec_version`, and records `migrated_from: "<old chunk_id>@<old spec_version>"` in the new manifest. The old chunk stays readable under its own spec version.
   - A consumer states which major versions it accepts. Accepting a chunk of an unknown major version = fail closed.
5. K1 key interaction: a major spec bump that changes stored bytes or their identity also bumps the affected K1 `key_schema_version` (K1 §2.3 rule 4); entries keyed under the old schema are quarantined per K1, not trusted and not deleted.

## 8. Implementation boundary in this repository (2026-10-29)

This document is still SPEC ONLY: nothing in sections 2 to 7 is
implemented here, and the spec's chunk manifest (section 3.1,
`manifest.json`, with the section 3.2 schema) does not exist in this
repository. Three manifest layers are distinct on disk, and one
directory belongs to exactly one of them.

| Layer | Module in this repository | Manifest file | Lock file | Schema |
|---|---|---|---|---|
| Spec chunk manifest (not implemented) | none | `manifest.json` | none | section 3.2 |
| Legacy per-chunk expert store | `src/samurai/persist.py` | `manifest.json` | `.manifest.lock` | `{"version": 1, "experts": {...}}` |
| D4 versioned expert tensors | `src/samurai/expert_persist.py` | `experts_manifest.json` | `.experts_manifest.lock` | `{"manifest_version": 1, "artifacts": [...]}` |

1. The D4 layer's filenames were renamed from `manifest.json` and
   `.manifest.lock` on 2026-10-29. The legacy store and the D4 layer
   wrote different schemas under one filename, so a reader could take
   one layer's manifest for the other's. They never shared a
   directory, and they still do not.
2. The D4 layer is not this spec. It stores no `spec_version`, no
   `chunk_id`, no `experts[]` LIST, no `determinism_class`, no
   `response_artifact`, and no `code_closure_hash`. It implements the
   versioned-artifact subset the D4 fusion gate needs, and claims
   nothing else.
3. Storage dtype: the D4 layer stores the caller's dtype as given;
   the governed ensemble persistence path (D4) stores `float16`, the
   dtype section 2.3 fixes for stored expert tensors. The legacy
   store also stores the caller's dtype. Neither layer implements the
   section 2.3 file-naming pattern
   (`experts/<expert_id>/<expert_id>__<model_slug>__prob.npy`).
4. The D4 manifest is byte-stable across an identical re-persist, so
   the SHA-256 a calibration record pins over the manifest keeps
   matching the tensors it was fitted on. Changed tensors change the
   manifest digest.
5. The D4 lock is shared by writers and governed readers. A governed run holds
   it while re-verifying the manifest and creating tensor memory mappings, so a
   cooperating writer cannot replace an artifact in the verification-to-load
   gap. Once mapped, atomic replacement publishes a different inode and cannot
   alter bytes already mapped by the run.
6. Conflict C-1 (the public-export note at the top of this document)
   is unchanged: aligning the legacy store with sections 2 and 3
   remains an open maintainer decision. This section records the
   file-name and schema boundary that decision must respect.

## 9. Non-claims

1. This spec does not claim any existing chunk is `COMPLETE`. On the verified evidence, every existing stored chunk is `INCOMPLETE` with `missing_experts: ["muse"]`.
2. This spec does not recover the lost Muse tensor. No re-call of the Muse expert is authorised by this spec, and a re-call would be a new Class C artifact, not the original.
3. No determinism class above B is claimed for any local expert without its double-run record (K1 §7).
4. This spec writes no code and changes no production file. Implementation, backfill, and any expert re-run are separate tasks with their own approval.
