"""Anchor-gate tests (S4 verdict, ens-proof Samurai anchor loss).

The ens-proof frame-0 geometry is measured, not invented (S1 record
69af94c8, S4 record cc745abc): the frame-0 SAM 3.1 seed mask holds
359,464 px, Samurai's frame-0 mask holds 338,969 px, and Samurai
already misses 22,236 px of the seed mask at frame 0 (6.19%), with
1,741 px held outside the seed mask. The synthetic fixture below
reproduces those exact counts on the measured 720x1280 frame; the
persisted-artifact test re-reads the real tensors where they exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from samurai.ensemble import (
    ANCHOR_GATE_MAX_MISSED_FRACTION,
    EnsembleError,
    check_anchor_gate,
)

SOURCE_PIXELS = 359_464
EXPERT_PIXELS = 338_969
MISSED_PIXELS = 22_236
EXTRA_PIXELS = 1_741
FRAME_SHAPE = (720, 1280)

ENS_PROOF = Path(os.environ.get("ENS_PROOF_DIR", ""))


def ens_proof_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Build masks with the exact measured ens-proof frame-0 counts."""
    source = np.zeros(FRAME_SHAPE, dtype=bool)
    source.flat[:SOURCE_PIXELS] = True
    expert = source.copy()
    # Drop the last MISSED_PIXELS source pixels, then hold
    # EXTRA_PIXELS pixels just past the source extent, so the expert
    # total is the measured 338,969 px.
    expert.flat[SOURCE_PIXELS - MISSED_PIXELS : SOURCE_PIXELS] = False
    expert.flat[SOURCE_PIXELS : SOURCE_PIXELS + EXTRA_PIXELS] = True
    assert int(np.count_nonzero(source)) == SOURCE_PIXELS
    assert int(np.count_nonzero(expert)) == EXPERT_PIXELS
    assert int(np.count_nonzero(source & ~expert)) == MISSED_PIXELS
    assert int(np.count_nonzero(expert & ~source)) == EXTRA_PIXELS
    return source, expert


def test_ens_proof_frame0_gap_trips_the_gate():
    source, expert = ens_proof_fixture()
    with pytest.raises(EnsembleError, match="misses 22236 px"):
        check_anchor_gate(source, expert, expert_name="samurai-sam2.1L")


def test_ens_proof_persisted_artifacts_trip_the_gate():
    old_mask = ENS_PROOF / "old_masks" / "00000.png"
    samurai_tensor = ENS_PROOF / "results" / "samurai_prob.npy"
    if not ENS_PROOF or not old_mask.is_file() or not samurai_tensor.is_file():
        pytest.skip("ens-proof persisted artifacts are not on this host")
    source = np.asarray(Image.open(old_mask)) > 127
    probabilities = np.load(samurai_tensor, mmap_mode="r")
    expert = np.asarray(probabilities[0], dtype=np.float32) >= 0.5
    assert int(np.count_nonzero(source)) == SOURCE_PIXELS
    assert int(np.count_nonzero(expert)) == EXPERT_PIXELS
    assert int(np.count_nonzero(source & ~expert)) == MISSED_PIXELS
    with pytest.raises(EnsembleError, match="misses 22236 px"):
        check_anchor_gate(source, expert, expert_name="samurai-sam2.1L")


def test_clean_anchor_under_the_limit_passes():
    source = np.zeros((100, 200), dtype=bool)
    source.flat[:10_000] = True
    expert = source.copy()
    expert.flat[9_600:10_000] = False  # 400 missed px = 4.00% of the source
    report = check_anchor_gate(source, expert, expert_name="samurai-sam2.1L")
    assert report is not None
    assert report.source_pixels == 10_000
    assert report.expert_pixels == 9_600
    assert report.missed_pixels == 400
    assert report.missed_fraction == pytest.approx(0.04)
    assert report.max_missed_fraction == ANCHOR_GATE_MAX_MISSED_FRACTION


def test_anchor_at_the_limit_passes_and_over_it_trips():
    source = np.zeros((100, 200), dtype=bool)
    source.flat[:10_000] = True
    expert = source.copy()
    expert.flat[9_500:10_000] = False  # 500 missed px = exactly 5.00%
    report = check_anchor_gate(source, expert)
    assert report is not None
    assert report.missed_fraction == pytest.approx(0.05)
    expert.flat[9_499] = False  # 501 missed px = 5.01%
    with pytest.raises(EnsembleError, match="misses 501 px"):
        check_anchor_gate(source, expert)


def test_no_source_mask_means_the_gate_is_off():
    expert = np.zeros((4, 4), dtype=bool)
    assert check_anchor_gate(None, expert, expert_name="samurai-sam2.1L") is None
    assert check_anchor_gate(None, np.ones((4, 4), dtype=bool)) is None


def test_empty_source_mask_fails_closed():
    with pytest.raises(EnsembleError, match="empty source mask"):
        check_anchor_gate(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool))


def test_shape_mismatch_fails_closed():
    with pytest.raises(EnsembleError, match="does not match source mask shape"):
        check_anchor_gate(np.ones((4, 4), dtype=bool), np.ones((4, 5), dtype=bool))


def test_invalid_limit_fails_closed():
    with pytest.raises(EnsembleError, match="max_missed_fraction"):
        check_anchor_gate(
            np.ones((4, 4), dtype=bool),
            np.ones((4, 4), dtype=bool),
            max_missed_fraction=1.5,
        )


def test_logits_expert_mask_uses_the_contract_transform():
    # A logit of +0.25 sits under the raw 0.5 cut, but the contract
    # transform gives sigmoid(0.25) = 0.56: the pixel is held. The
    # raw-threshold rule binarized this frame as empty and the gate
    # refused the run (round-3 finding R3-06).
    source = np.zeros((4, 6), dtype=bool)
    source.flat[:20] = True
    expert = np.full((4, 6), -4.0)
    expert.flat[:20] = 0.25
    report = check_anchor_gate(source, expert, expert_representation="logits")
    assert report is not None
    assert report.expert_pixels == 20
    assert report.missed_pixels == 0


def test_probabilities_expert_default_rule_is_unchanged():
    source = np.zeros((4, 6), dtype=bool)
    source.flat[:20] = True
    expert = np.full((4, 6), 0.1)
    expert.flat[:20] = 0.9
    report = check_anchor_gate(source, expert)
    assert report is not None
    assert report.expert_pixels == 20
    assert report.missed_pixels == 0
