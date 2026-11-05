# VOID Auto-Edit architecture

VOID Auto-Edit turns one selected object into a checked video removal. The
creator identifies the object once. SAM 3.1 follows that object through the
processed shot, Muse contributes a second perception view while it moves, and
VOID rebuilds the background behind the selected object.

## Runtime path

1. Read the source shot and verify its geometry, cadence, and frame count.
2. Accept the creator's one-object selection.
3. Run SAM 3.1 perception and preserve frame-aligned mask tensors.
4. Add the Muse perception view and combine the aligned views under bounded
   weights.
5. Convert the stable selection into VOID's four mask classes.
6. Run VOID to rebuild the background.
7. Fail the run if the selected object is never located. Frames where SAM
   omits the object retain empty target masks.
8. Write the result, receipt, and digest.

## Owned boundaries

| Area | Owner |
| --- | --- |
| CLI and model adapters | `src/samurai/cli.py`, `src/samurai/sam31.py` |
| Perception fusion and calibration | `src/samurai/ensemble.py`, `src/samurai/calibration*.py` |
| Tensor persistence | `src/samurai/persist.py`, `src/samurai/expert_persist.py` |
| Media and mask packaging | `src/samurai/media.py`, `src/samurai/quadmask.py`, `scripts/package_quadmask.py` |
| Result review | `src/samurai/pass2_metrics.py`, `scripts/pass2_compare.py`, `src/samurai/remeasure.py` |
| Candidate selection | `src/samurai/best_of_n.py`, `scripts/best_of_n.py` |
| Submission record | `evidence/auto-edit-submission-2026-11-04.json` |

## Execution boundaries

- SAM 3.1 and VOID run in separate pinned environments.
- A mask is media with exact values and geometry, not prose.
- Checks quantify target selection, retention, stability, and removal change.
- Source media, credentials, machine paths, and checkpoints are not committed.
- Comparator values support review; they do not establish artistic quality or
  production readiness.
