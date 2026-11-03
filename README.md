# Void Fusion

Void Fusion is a model-conditioned video replacement system. It gives a film or
previsualization team a repeatable way to change the identity of an object in an
existing shot: track the original object, remove it, insert a fitted 3D
replacement, refine only the seam, and measure what changed. The goal is to
avoid a reshoot when the action is right but an object, product, stand-in, or
character must change.

**Watch the 2 minute 36 second demo:**
[void-fusion-demo-2026-11-03-public.mp4](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v1.0.0/void-fusion-demo-2026-11-03-public.mp4)

The narrated demo shows one real scene as a four-panel comparison: the original
shot, the selected object mask, the fitted 3D insertion before refinement, and
the selected result after the second VOID pass. It explains the model roles,
summarizes the four-candidate measurement, names the selected candidate, and
states the measured limitations in plain language.

## What it does

1. **Select and track.** Meta SAM 3.1 provides the dense object anchor. Model
   reviewers propose target, preserve, and affected semantics; those judgments
   are converted into aligned probability fields and fused numerically.
2. **Remove.** The decision becomes the exact VOID quadmask. VOID Pass 1 erases
   the original object and reconstructs a clean plate.
3. **Insert.** A model-free fitted 3D proxy is posed over that plate. The
   inserted object is marked keep; only the surrounding blend ring is marked
   for cleanup.
4. **Refine.** VOID Pass 2 refines the cleanup ring; the comparator then
   reports how much the inserted object actually changed.
5. **Measure.** The no-Pass2 and Pass2 results are compared across exact
   timestamps, how much the inserted object and untouched background changed,
   and temporal stability. Best-of-N runs the same complete chain over
   multiple fitted candidates and selects the strongest complete result.

This is not a manual video editor and not an unconstrained text-to-video
generator. The contribution is the auditable control system around the models.

## Real full-chain result

The retained real-scene run exported four fitted candidates, created
deterministic 85-frame execution derivatives, and executed VOID Pass2 once per
candidate with seed 42 and 50 steps. All four completed.

A changed ratio measures the fraction of a region that changed; lower is less
change, not better artistic quality. The temporal warning is `1.0`, the maximum
reported pop. The full-chain objective combines the pre-insertion score with
these change and stability measurements; higher is better only within this
declared formula.

| Candidate | Object changed | Background changed | Full-chain objective |
|---|---:|---:|---:|
| smooth-00 | 0.594590 | 0.151184 | **0.476731** |
| smooth-06 | 0.595163 | 0.152393 | 0.476597 |
| smooth-02 | 0.596139 | 0.153477 | 0.476093 |
| smooth-04 | 0.595397 | 0.153359 | 0.475324 |

The full-chain selector chose **smooth-00**. Every comparator currently reports
a maximum temporal warning and a frame-zero lead-in violation, a structural
warning at the first frame; those failures remain visible rather than being
presented as artistic quality. The execution derivatives cover 85 frames while
pose ownership is measured over the 96-frame source cadence. Sparse pose
ownership passes on only 4 of those 96 frames, so the demo claims a fitted
proxy, not continuous full-body physics.

Sanitized execution parameters, model/checkpoint identity, execution log and
output digests, comparator-report digests, the demo asset digest, ranking, and
non-claims are retained in
[`evidence/submission-evidence.json`](evidence/submission-evidence.json).
Private media, full command lines, provider response bodies, and private review
history are not redistributed.

## Why Meta AI is central

- **SAM 3.1** supplies the dense, frame-aligned perception anchor for the
  selected object.
- **Muse Spark** participates in the bounded target/affected prompt loop.
- Reasoning-model outputs are never accepted as prose. Coordinates and concepts
  are validated, rasterized, temperature-scaled, and fused under the same
  numerical contract as the dense expert.
- Low-confidence selection escalates instead of silently changing the target.

## Quick start

Prerequisites are `uv`, `just`, `shellcheck`, and `ffmpeg`/`ffprobe`.
The SAM 3.1 and VOID lanes additionally require their documented CUDA,
checkpoint, and Python dependencies.

```bash
uv sync
just check
just quality
```

The optional SAM 3.1 dependencies use `just setup-sam31`. The separately pinned
VOID lane uses `just setup-void`; it is intentionally not merged into the SAM
environment.

## Evidence and provenance

- [`docs/architecture.md`](docs/architecture.md) is the canonical technical
  status document.
- [`evidence/submission-evidence.json`](evidence/submission-evidence.json)
  binds the real four-candidate execution.
- [`provenance/daily/`](provenance/daily) is the anonymized cumulative
  eligibility timeline.
- [`docs/contribution-manifest.md`](docs/contribution-manifest.md) separates the
  pre-event baseline from event-built work.

Private source media, model weights, private pipeline outputs, credentials,
machine paths, and private review history are intentionally absent. They are
bound by SHA-256 in sanitized receipts rather than redistributed here.

## Team

Kearm (representative), Eden Chan, and Jason Lu built Void Fusion for the Meta
Global AI Developer Hackathon.
