"""Claims hygiene guards for the retained public claim carriers.

Two rules are enforced here:

1. an invalidated gated PASS must never be written into a public artifact
   as fact; the guard names the recorded wording literally instead of only
   its neighbours, and also fires on near variants;
2. the expert-tensor persistence spec must hold no private host token and
   must keep its example host as a placeholder.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs/tensor-persistence-spec.md"

# The retained public files that state findings, verdicts, or completion
# claims a reader could take as fact.
CLAIM_CARRIERS = (
    SPEC,
    ROOT / "scripts/critique/critique.py",
    ROOT / "scripts/critique/score.py",
    ROOT / "scripts/critique_harness.py",
)

# The invalidated gated PASS must not be written into an item set as fact.
# "PASS, IoU 0.9923" is the exact wording the recorded regression used, so
# the guard names it literally instead of only its neighbours.
FORBIDDEN_FACT_PASS_WORDINGS: tuple[str, ...] = (
    "verdict PASS",
    "PASS (IoU",
    "PASS, IoU 0.9923",
)

# Near variants of the same regression: a PASS stated next to the invalidated
# IoU figure, with flexible punctuation and spacing.
FORBIDDEN_FACT_PASS_PATTERN = re.compile(r"PASS\W{0,3}IoU\W{0,3}0?\.?9923", re.IGNORECASE)


def fact_pass_wording(text: str) -> list[str]:
    """Return every fact-pass wording found in text."""
    found = [wording for wording in FORBIDDEN_FACT_PASS_WORDINGS if wording in text]
    if FORBIDDEN_FACT_PASS_PATTERN.search(text):
        found.append("PASS, IoU 0.9923")
    return list(dict.fromkeys(found))


def test_public_claim_carriers_do_not_state_invalidated_pass_as_fact() -> None:
    for carrier in CLAIM_CARRIERS:
        text = carrier.read_text(encoding="utf-8")
        # The invalidated gated PASS must not be written into a public
        # artifact as fact.
        assert fact_pass_wording(text) == [], carrier.name


def test_f35_guard_catches_the_documented_regression_wording() -> None:
    # The guard must fire on the exact regression wording and its near
    # variants; otherwise it does not cover the recorded defect.
    for wording in (
        "PASS, IoU 0.9923",
        "known: PASS, IoU 0.9923",
        "PASS,IoU 0.9923",
        "PASS IoU 0.9923",
        "PASS, IoU 0.9923 (gated run)",
    ):
        assert fact_pass_wording(wording), wording


def test_f35_guard_allows_the_corrected_claim_wording() -> None:
    # The corrected citation keeps the invalidated figure on record while
    # refusing to state it as a PASS; the guard must not block it.
    corrected = (
        "its own verdict.json records iou_mean 0.9923, but that result is NOT "
        "CERTIFIABLE from storage: the Muse expert tensor was never persisted"
    )
    assert fact_pass_wording(corrected) == []


def test_a1_spec_holds_no_private_host_tokens() -> None:
    text = SPEC.read_text(encoding="utf-8")
    # Build the labels here so this guard does not itself introduce the
    # private-provider and rig-name tokens into a published tree.
    assert ("Lino" + "de") not in text
    assert ("tira" + "misu") not in text


def test_a1_spec_example_host_is_a_placeholder() -> None:
    text = SPEC.read_text(encoding="utf-8")
    # The example manifest keeps the required `host` field, with a redacted
    # placeholder value in the same style as the spec's other placeholders.
    assert '"host": "<host>"' in text
