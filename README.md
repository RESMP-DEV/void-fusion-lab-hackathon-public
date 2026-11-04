# VOID Auto-Edit: automatic object-removal review for real footage

VOID Auto-Edit turns a single target-point choice into an auditable removal
pipeline. Meta SAM 3.1 supplies frame-aligned object perception, the runtime
locks one referent and converts it into VOID's exact remove / overlap /
affected / keep classes, VOID removes the target, and measured gates inspect
bystander spill, subject retention, cadence, and ghosting.

This replaces the most tedious part of a manual edit: watching every frame,
finding ambiguous referents, correcting masks, rerunning inpainting, and
comparing outputs by eye.

## Watch the 2:58 demo

- [VOID Auto-Edit removal-only demo](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v4.0.0/void-auto-edit-removal-only.mp4)
- [v4.0.0 release](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/tag/v4.0.0)
- [Judge presentation guide](docs/presentation.md)
- [Submission evidence](evidence/auto-edit-submission-2026-11-04.json)

## What the demo proves

| Claim | Measured result |
| --- | ---: |
| Text-only `person` selection is unsafe | 68,355-70,233 bystander pixels spilled on all 96 frames |
| One target point fixes referent ambiguity | 0 bystander spill on judged frames |
| Intended subject remains selected | 99.995% subject retention |
| Corrected mask can drive VOID | Ghost scores 41.77 and 14.91, below threshold 50 |
| Meta Muse rescues ensemble dropout | 485,191 / 503,058 lost pixels recovered, or 96.4483% |
| Quadmask execution is temporal | 90-frame run with frame-0 ghost score 7.84 |

The Muse recovery is deliberately bounded to verified frames 16-23. The
corrected localization mean IoU is 0.467366, narrowly below a proposed but
not-in-force 0.470294 comparison bar. These limitations stay in the demo rather
than being hidden.

## Why this is not ordinary rotoscoping

A professional artist can certainly reproduce a visual result manually. VOID
Auto-Edit's contribution is not artistic impossibility. It is the automatic,
measured review path:

1. select one referent;
2. propagate SAM object identity;
3. encode lossless four-class quadmasks;
4. execute VOID;
5. measure target retention, bystander spill, ghosting, and cadence;
6. fail closed instead of silently accepting the wrong object.

## Reproduce the software gates

```bash
uv sync
just check
just quality
```

The SAM 3.1 and VOID execution lanes require their documented CUDA
environments and checkpoints. Repository tests do not download checkpoints,
commit source footage, or call external APIs.

## Limitations

The target choice is a human decision. Candidate C's referent correction is
post hoc, identity beyond the measured subject mask is not evaluated, and the
Muse rescue covers frames 16-23 only. Pixel metrics do not prove artistic
quality or production readiness.
