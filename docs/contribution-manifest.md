# Hackathon contribution manifest

This manifest separates the pre-hackathon baseline from event-built work. It is
an eligibility aid, not a demo-quality claim.

## Baseline

The source tree begins from the team's recorded pre-kickoff research baseline.
That baseline contained initial mask/fusion/calibration/SAM/VOID/CLI scaffolding
and is not claimed as event-built work. The private baseline identity and
pull-request discussion are intentionally withheld; the anonymized cumulative
timeline below records only post-kickoff contributions.

## Event-built work

| Area | State | Public evidence boundary |
| --- | --- | --- |
| Official Meta rules, personal-key workflow, and bounded API lane | implemented and exercised | `evidence/submission-evidence.json` |
| Bounded Muse-to-critic prompt loop | implemented, exercised, and measured | `scripts/prompt_critique.py`; usage summary in submission evidence |
| Fitted insertion iteration and per-candidate export | implemented and exercised | `scripts/post_mask_iteration.py`; output digests in submission evidence |
| Strict Pass2 comparator | implemented and measured on real outputs | `scripts/pass2_compare.py`; metrics in submission evidence |
| Diegetic clean-plate insertion and real VOID Pass2 | implemented and executed | `scripts/post_mask_iteration.py`; model/checkpoint identity and digests in submission evidence |
| Lower-body ownership measurement | implemented and measured; 4/96 pass | `src/samurai/lower_body.py`; result summary in submission evidence |
| Contact and shadow grounding | implemented and exercised | `src/samurai/contact_grounding.py`; sparse-owned-frame summary in submission evidence |
| Pass2-aware Best-of-N full-chain ranking | implemented and exercised | `src/samurai/best_of_n.py`; four-candidate ranking in submission evidence |
| Four real complete VOID chains and selected full-chain winner | executed and measured | `evidence/submission-evidence.json` |
| Under-three-minute submission demo | produced | Release asset `void-fusion-demo-2026-11-03-public.mp4` at tag `v1.0.0`; digest and metadata in submission evidence |

The anonymized cumulative event timeline is
[`../provenance/daily/2026-11-03.json`](../provenance/daily/2026-11-03.json).

## Non-claims

A timeline or code path is not a quality verdict. Comparator values quantify
change, and the current temporal/ownership limitations are recorded in
`evidence/submission-evidence.json`.
