# Architecture and status

Void Fusion is a model-conditioned video replacement runtime. It combines
Meta SAM 3.1 perception, bounded reasoning/fusion, Netflix VOID inpainting, a
fitted 3D insertion lane, and measured Best-of-N selection.

## Current pipeline

1. A natural-language request is converted into a strict target description and
   candidate preserve/affected semantics. Muse Spark participates in a bounded
   generation/critique loop; deterministic validators reject invalid prompt
   shapes before downstream use.
2. Meta SAM 3.1 receives an explicit concrete object concept and produces the
   dense frame-aligned anchor mask. It is the perception anchor, not an
   unconstrained semantic oracle.
3. Reasoning experts are normalized to coordinate/probability fields. The fusion
   layer converts probabilities to logits, applies shared temperature scaling,
   calibrates expert weights, records disagreement, and escalates ambiguous
   selections instead of silently choosing a different object.
4. The fused primary and affected regions become the exact VOID quadmask:
   `0` remove, `63` overlap, `127` affected, and `255` keep. VOID Pass 1 removes
   the original object and reconstructs a clean plate.
5. A fitted 3D proxy is posed from measured scene cues and composited onto the
   clean plate. The inserted object is marked keep. A dilated cleanup ring is
   marked affected/remove. VOID Pass 2 refines only that ring.
6. A strict comparator aligns exact rational timestamps, geometry, and frame
   counts, then measures object preservation, cleanup-ring change, background
   change, and temporal stability. Best-of-N ranks complete candidates using the
   pre-insertion score plus these measured Pass2 signals.

## Real result

Four fitted candidates were exported as complete 85-frame, 672x384 chains and
executed through VOID Pass2 with the same seed. All four completed 50/50 steps.
The full-chain selector ranked:

1. `smooth-00`, objective `0.4767313284`
2. `smooth-06`, objective `0.4765965357`
3. `smooth-02`, objective `0.4760929340`
4. `smooth-04`, objective `0.4753241035`

`smooth-00` was selected as the strongest complete chain under the declared
objective.

The full-chain objective is:

```text
0.35 * pre_pass2_objective
+ 0.25 * (1 - object_preservation_changed_ratio)
+ 0.20 * (1 - background_changed_ratio)
+ 0.20 * (1 - temporal_pop_score)
```

The sanitized submission evidence carries the exact pre-Pass2 objectives and
metrics needed to reproduce this ranking.

## Important boundaries

- Comparator values quantify change from the no-Pass2 insertion; they are not
  an artistic-quality score.
- Every current real comparator reports temporal pop `1.0` and a frame-zero
  lead-in violation. These warnings are retained.
- Sparse lower-body ownership passes only frames `0`, `1`, `7`, and `13` of 96.
  The system does not claim continuous pose ownership or full-body physics.
- The fitted asset is a model-free pneumatic proxy, not learned image-to-3D
  generation.
- Private source media, model weights, generated outputs, machine paths, and
  review history are not redistributed. Sanitized digests bind the executed
  artifacts in `evidence/submission-evidence.json`.

## Code boundaries

- Canonical Python lives in `src/samurai/` and does not import `sam2`.
- SAM 3.1 and VOID remain separate execution lanes with separate dependencies.
- Gray masks are encoded as lossless H.264 `yuv444p` at `qp=0`.
- Best-of-N must consume complete, digest-bound candidate chains; missing or
  stale Pass2 artifacts fail closed.
- The base checks are `just check` and `just quality`.
