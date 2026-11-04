# VOID Auto-Edit judge guide

## One sentence

**VOID Auto-Edit selects one object, removes it with VOID, and automatically
audits every frame for the wrong target, missing subject, cadence break, or
ghosted removal.**

For a layperson: **instead of a person watching every frame and fixing masks by
hand, the computer checks the edit and rejects the bad ones.**

## Demo map

The submitted demo is 178.000 seconds, 1920×1080, 24 fps, and narrated.

| Time | Section | Claim |
| --- | --- | --- |
| 0:00-0:08 | Hook | Remove the object while preserving the scene. |
| 0:08-0:38 | Referent failure and correction | Text-only SAM unions two people; a single target point locks one referent. |
| 0:38-0:41 | Target-lock card | Identity must pass before GPU work. |
| 0:41-1:02 | Review interface | A creator triages model disagreement and locks one referent. |
| 1:02-1:38 | Muse rescue | A missing head-and-arm region returns when the live Muse vote is restored. |
| 1:38-2:23 | 90-frame VOID endurance | The quadmask contract sustains temporal removal. |
| 2:23-2:53 | Measurements and limits | Zero judged-frame spill, 99.995% retention, bounded Muse scope. |

## Rubric alignment

### Impact, 30%

Manual object removal is slow because a human must inspect footage, detect an
ambiguous referent, correct masks, rerun inpainting, and eyeball the result.
VOID Auto-Edit makes those checks automatic after one target choice. The same
audit can run across a full shot instead of relying on a person watching every
frame.

### Meta AI tooling, 25%

SAM 3.1 provides frame-aligned object perception. The critical improvement is
not asking text alone to decide identity: a single target point selects the
referent, and SAM object identity propagates through the shot. Meta Muse then
participates in ensemble perception. The demo measures what happens when Muse
is removed and restored.

### Technical depth, execution, 20%

The runtime preserves the four VOID classes, uses lossless mask media, aligns
cadence, executes VOID, and computes:

- bystander spill;
- subject retention;
- mean IoU;
- VOID ghost scores;
- frame count and temporal coverage;
- Muse dropout and recovery.

### Demo, 25%

The video opens with the failure mode, shows the corrected target, exposes the
review workflow, demonstrates Muse ablation, and closes with a sustained VOID
run. Narration states both the measured wins and boundaries.

## Honest limits

- Candidate C's correction is post hoc.
- Mean IoU is 0.467366 against a proposed, not-in-force 0.470294 bar.
- The Muse rescue covers frames 16-23 only.
- Metrics do not prove artistic quality or production readiness.
- A skilled artist could reproduce a similar visual result manually; the claim
  is automated audit and execution, not visual impossibility.
