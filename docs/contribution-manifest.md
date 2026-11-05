# Product implementation map

This map explains the public implementation surfaces behind VOID Auto-Edit. It
does not attribute individual files to development periods.

## Product-critical surfaces

| Capability | Public implementation |
| --- | --- |
| Selected-object removal workflow | `src/samurai/sam31.py`, `src/samurai/ensemble.py`, `src/samurai/quadmask.py` |
| Stable mask packaging and media handling | `scripts/package_quadmask.py`, `src/samurai/media.py`, `src/samurai/quadmask.py` |
| Automatic review and comparison | `src/samurai/pass2_metrics.py`, `scripts/pass2_compare.py`, `src/samurai/remeasure.py` |
| Candidate selection | `src/samurai/best_of_n.py`, `scripts/best_of_n.py` |
| Perception persistence and calibration | `src/samurai/persist.py`, `src/samurai/expert_persist.py`, `src/samurai/calibration_run.py` |
| Contact and body-boundary checks | `src/samurai/contact_grounding.py`, `src/samurai/owned_frame_contact.py`, `src/samurai/lower_body.py` |
| Staged insertion experiments | `src/samurai/insertion.py`, `scripts/post_mask_iteration.py`, `scripts/preserve_pass2_insertion.py` |
| Scene segmentation | `scripts/scene_boundaries.py` |
| Prompt critique loop | `scripts/prompt_critique.py`, `scripts/critique_harness.py`, `scripts/critique/` |
| Public demo and submission record | `README.md`, `docs/presentation.md`, `evidence/auto-edit-submission-2026-11-04.json` |

## Supporting surfaces

The CLI, ensemble, SAM 3.1 adapter, media reader, quadmask packaging,
calibration tests, model-boundary tests, and examples support the product path.
The broad execution-and-test suite gives perception, mask, comparison,
persistence, and candidate selection deterministic local contracts.

## Current submission

The submitted product is VOID Auto-Edit: click one object, remove it with
VOID, and run automatic checks across the processed shot. The release asset and
digest are recorded in `evidence/auto-edit-submission-2026-11-04.json`.
