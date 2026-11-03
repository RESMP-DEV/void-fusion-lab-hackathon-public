from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from samurai.calibration import (
    CalibrationError,
    CalibrationSettings,
    _sample_indices,
    calibrate_config,
    calibrate_experts,
)
from samurai.cli import main
from samurai.ensemble import ExpertSpec, expert_probabilities, load_config


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(name: str, *, temperature: float = 1.0, representation: str = "logits") -> ExpertSpec:
    return ExpertSpec(
        name=name,
        weight=1.0,
        temperature=temperature,
        kind="tensor",
        representation=representation,
    )


def panel_values() -> np.ndarray:
    """Three experts over a 6x8 frame: decisive, moderate, and near-uncertain."""
    latent = np.zeros((1, 6, 8), dtype=bool)
    latent[0, :4, :4] = True
    decisive = np.where(latent, 4.0, -4.0)
    moderate = np.where(latent, 2.0, -1.0)
    uncertain = np.where(latent, 0.4, -0.2)
    return np.stack([decisive, moderate, uncertain])


def test_bernoulli_em_recovers_latent_mask_and_ranks_expert_confidence():
    specs = [spec("decisive"), spec("moderate"), spec("uncertain")]
    result = calibrate_experts(
        panel_values(),
        specs,
        CalibrationSettings(tolerance=1e-7, max_iterations=100),
    )
    assert result.converged
    assert 1 <= result.iterations <= 100
    assert np.count_nonzero(result.latent_masks[0, :4, :4]) == 16
    assert np.count_nonzero(result.latent_masks[0, 4:, :]) == 0
    assert 0.2 < result.latent_positive_fraction < 0.5
    assert 0.2 < result.latent_selected_fraction < 0.5

    metrics = {item["name"]: item for item in result.expert_metrics}
    assert metrics["decisive"]["sensitivity"] > metrics["uncertain"]["sensitivity"]
    assert metrics["decisive"]["false_positive_rate"] < metrics["uncertain"]["false_positive_rate"]
    assert (
        result.confidence_weights["decisive"]
        > result.confidence_weights["moderate"]
        > result.confidence_weights["uncertain"]
    )
    assert sum(result.confidence_weights.values()) == pytest.approx(1.0)


def test_iteration_bound_is_respected_when_convergence_is_not_reached():
    specs = [spec("decisive"), spec("moderate"), spec("uncertain")]
    bounded = calibrate_experts(
        panel_values(),
        specs,
        CalibrationSettings(tolerance=1e-12, max_iterations=1),
    )
    assert bounded.iterations == 1
    assert not bounded.converged


def test_pairwise_summaries_report_dice_brier_and_disagreement():
    result = calibrate_experts(panel_values(), [spec("a"), spec("b"), spec("c")])
    assert len(result.pairwise_metrics) == 3
    for summary in result.pairwise_metrics:
        assert {"experts", "dice", "brier", "disagreement"} <= set(summary)
        assert 0.0 <= summary["dice"] <= 1.0
        assert 0.0 <= summary["disagreement"] <= 1.0
        assert summary["brier"] >= 0.0
    with pytest.raises(CalibrationError, match="at least three experts"):
        calibrate_experts(
            np.stack([np.full((1, 4, 4), 3.0), np.full((1, 4, 4), 3.0)]),
            [spec("a"), spec("b")],
        )


def test_probability_inputs_use_logit_then_temperature_scaling():
    latent = np.zeros((1, 2, 2), dtype=bool)
    latent[0, 0, 0] = True
    votes = np.where(latent, 0.92, 0.04)
    values = np.stack([votes, votes, votes])
    specs = [
        spec("sharp", temperature=0.5, representation="probabilities"),
        spec("soft", temperature=4.0, representation="probabilities"),
        spec("neutral", temperature=2.0, representation="probabilities"),
    ]
    transformed = expert_probabilities(values, specs)
    assert transformed[0, 0, 0, 0] > transformed[1, 0, 0, 0]
    assert transformed[0, 0, 0, 1] < transformed[1, 0, 0, 1]

    result = calibrate_experts(values, specs)
    metrics = {item["name"]: item for item in result.expert_metrics}
    assert metrics["sharp"]["confidence_weight"] > metrics["soft"]["confidence_weight"]
    assert result.confidence_weights["sharp"] > result.confidence_weights["soft"]


def test_identical_and_uninformative_panels_fall_back_to_equal_weights():
    identical = calibrate_experts(
        np.stack([np.full((1, 4, 4), 3.0)] * 3),
        [spec("a"), spec("b"), spec("c")],
    )
    for weight in identical.confidence_weights.values():
        assert weight == pytest.approx(1.0 / 3)

    uninformative = calibrate_experts(
        np.zeros((3, 1, 4, 4)),
        [spec("a"), spec("b"), spec("c")],
    )
    for weight in uninformative.confidence_weights.values():
        assert weight == pytest.approx(1.0 / 3)
    assert sum(uninformative.confidence_weights.values()) == pytest.approx(1.0)


def test_reported_weights_match_the_reported_agreement():
    result = calibrate_experts(panel_values(), [spec("a"), spec("b"), spec("c")])
    metrics = {item["name"]: item for item in result.expert_metrics}
    excess = np.array([metrics[name]["soft_agreement"] - 0.5 for name in ("a", "b", "c")])
    expected = excess / excess.sum()
    for index, name in enumerate(("a", "b", "c")):
        assert metrics[name]["confidence_weight"] == pytest.approx(expected[index])
        assert result.confidence_weights[name] == pytest.approx(expected[index])


def test_positive_class_follows_panel_consensus_not_first_expert():
    latent = np.zeros((1, 4, 4), dtype=bool)
    latent[0, :2, :2] = True
    inverted = np.where(latent, -6.0, 6.0)
    decisive = np.where(latent, 6.0, -6.0)
    values = np.stack([inverted, decisive, decisive])
    result = calibrate_experts(values, [spec("inverted"), spec("correct-a"), spec("correct-b")])
    assert np.array_equal(result.latent_masks[0], latent[0])
    assert result.anchor_agreement == pytest.approx(0.0)
    assert result.confidence_weights["inverted"] == pytest.approx(0.0)
    assert result.confidence_weights["correct-a"] > 0
    assert result.confidence_weights["correct-b"] > 0


def test_report_keeps_detailed_expert_metrics_and_names():
    result = calibrate_experts(panel_values(), [spec("a"), spec("b"), spec("c")])
    report = result.report()
    assert report["experts"] == ["a", "b", "c"]
    assert [item["name"] for item in report["expert_metrics"]] == ["a", "b", "c"]
    assert {"sensitivity", "false_positive_rate", "soft_agreement"} <= set(
        report["expert_metrics"][0]
    )


def test_calibration_bounds_and_inputs_fail_closed():
    with pytest.raises(CalibrationError):
        calibrate_experts(
            np.zeros((2, 1, 2, 2)), [spec("a"), spec("b")], CalibrationSettings(tolerance=-1)
        )
    with pytest.raises(CalibrationError):
        calibrate_experts(
            np.zeros((2, 1, 2, 2)), [spec("a"), spec("b")], CalibrationSettings(max_iterations=0)
        )

    probability_spec = spec("a", representation="probabilities")
    values = np.zeros((2, 1, 2, 2))
    values[0, 0, 0, 0] = 1.5
    with pytest.raises(CalibrationError):
        calibrate_experts(values, [probability_spec, spec("b")])

    values = np.zeros((2, 1, 2, 2))
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(CalibrationError):
        calibrate_experts(values, [spec("a"), spec("b")])

    with pytest.raises(CalibrationError):
        calibrate_experts(np.zeros((1, 1, 2, 2)), [spec("a")])


def test_calibrate_cli_writes_preserving_config_and_json_report(tmp_path, capsys):
    values = panel_values()
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    third = tmp_path / "third.npy"
    np.save(first, values[0])
    np.save(second, values[1])
    np.save(third, values[2])
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"")
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(input_video),
                "experts": [
                    {"name": "first", "path": str(first), "prompt": "keep me"},
                    {"name": "second", "path": str(second), "temperature": 1.5},
                    {"name": "third", "path": str(third)},
                ],
                "fusion": {
                    "mode": "top_k",
                    "top_k_ratio": 0.25,
                    "min_expert_support": 1,
                    "batch_size": 2,
                },
            }
        )
    )
    generated_config = tmp_path / "generated.json"
    report_path = tmp_path / "calibration.json"
    latent_path = tmp_path / "latent.npy"
    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config),
                "--config-output",
                str(generated_config),
                "--report",
                str(report_path),
                "--latent-output",
                str(latent_path),
                "--max-iterations",
                "50",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["method"] == "dawid_skene_bernoulli_em"
    assert report["labels"] == "none"
    assert report["iterations"] <= 50
    assert set(report["confidence_weights"]) == {"first", "second", "third"}
    assert sum(report["confidence_weights"].values()) == pytest.approx(1.0)
    assert len(report["pairwise"]) == 3
    assert len(report["frames_report"]) == 1
    assert {
        "mean_latent_probability",
        "latent_selected_fraction",
        "mean_expert_support",
        "mean_pairwise_disagreement",
        "mean_pairwise_brier",
    } <= set(report["frames_report"][0])
    for name, path in (("first", first), ("second", second), ("third", third)):
        assert report["artifacts"]["expert_sha256"][name] == _sha256(path)
    assert report["non_claims"]

    generated = load_config(generated_config)
    original = load_config(config)
    assert generated.experts[0].prompt == "keep me"
    assert generated.experts[0].path == original.experts[0].path
    assert generated.experts[1].temperature == original.experts[1].temperature
    assert generated.fusion == original.fusion
    assert generated.experts[0].weight > generated.experts[2].weight
    assert sum(expert.weight for expert in generated.experts) == pytest.approx(1.0)
    assert generated.calibration["source_config_sha256"] == _sha256(config)
    assert generated.calibration["expert_sha256"]["first"] == _sha256(first)

    latent = np.load(latent_path)
    assert latent.shape == (1, 6, 8)
    assert float(np.count_nonzero(latent >= 0.5)) == 16


def test_calibrate_config_rejects_out_of_range_probability_tensors(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.full((1, 2, 2), 1.5))
    np.save(second, np.zeros((1, 2, 2)))
    third = tmp_path / "third.npy"
    np.save(third, np.zeros((1, 2, 2)))
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(tmp_path / "input.mp4"),
                "experts": [
                    {"name": "first", "path": str(first), "representation": "probabilities"},
                    {"name": "second", "path": str(second)},
                    {"name": "third", "path": str(third)},
                ],
                "fusion": {"mode": "top_k"},
            }
        )
    )
    with pytest.raises(CalibrationError):
        calibrate_config(config, tmp_path / "generated.json", tmp_path / "report.json")


def test_calibrate_config_rejects_mismatched_tensor_shapes(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.zeros((1, 2, 2)))
    np.save(second, np.zeros((2, 2, 2)))
    third = tmp_path / "third.npy"
    np.save(third, np.zeros((1, 2, 2)))
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(tmp_path / "input.mp4"),
                "experts": [
                    {"name": "first", "path": str(first)},
                    {"name": "second", "path": str(second)},
                    {"name": "third", "path": str(third)},
                ],
                "fusion": {"mode": "top_k"},
            }
        )
    )
    with pytest.raises(CalibrationError):
        calibrate_config(config, tmp_path / "generated.json", tmp_path / "report.json")


def test_calibrate_config_rejects_command_experts(tmp_path):
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "input_video": str(tmp_path / "input.mp4"),
                "experts": [
                    {"name": "first", "path": str(tmp_path / "first.npy")},
                    {"name": "second", "path": str(tmp_path / "second.npy")},
                    {"name": "adapter", "kind": "command", "command": ["fake"]},
                ],
                "fusion": {"mode": "top_k"},
            }
        )
    )
    with pytest.raises(CalibrationError):
        calibrate_config(config, tmp_path / "generated.json", tmp_path / "report.json")


def _noisy_probabilities(rng, latent, sensitivity, fpr):
    draw = rng.random(latent.shape)
    vote = np.where(latent, draw < sensitivity, draw < fpr)
    return np.where(vote, 0.9, 0.1)


def test_sample_indices_are_bounded_sorted_unique_and_repeatable():
    """Design 5.12 (draw half): the bounded sample draw returns exactly
    max_samples indices, sorted, unique, and identical for the same seed."""
    indices = _sample_indices(1000, 100, 0)
    assert indices.size == 100
    assert np.all(np.diff(indices) > 0)
    assert np.unique(indices).size == indices.size
    assert np.array_equal(indices, _sample_indices(1000, 100, 0))
    assert not np.array_equal(indices, _sample_indices(1000, 100, 1))
    # At or above the total, the draw is the full range in order.
    assert np.array_equal(_sample_indices(50, 100, 0), np.arange(50))
    assert np.array_equal(_sample_indices(50, 50, 0), np.arange(50))


def test_exact_half_report_contributes_zero_evidence():
    """Design 5.13 (fit half): an expert at exactly 0.5 everywhere has
    zero confidence on every report, so its fitted sensitivity and
    false-positive rate stay at the prior-smoothed neutral value 0.5,
    its soft agreement is chance, and its weight is 0. The other
    experts' diagnostics are still reported."""
    rng = np.random.default_rng(7)
    latent = rng.random((2, 20, 25)) < 0.35
    values = np.stack(
        [
            _noisy_probabilities(rng, latent, 0.95, 0.05),
            _noisy_probabilities(rng, latent, 0.85, 0.12),
            np.full(latent.shape, 0.5),
        ]
    )
    specs = [
        spec("good", representation="probabilities"),
        spec("ok", representation="probabilities"),
        spec("half", representation="probabilities"),
    ]
    result = calibrate_experts(values, specs)
    metrics = {item["name"]: item for item in result.expert_metrics}
    assert metrics["half"]["sensitivity"] == pytest.approx(0.5)
    assert metrics["half"]["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["half"]["mean_probability"] == pytest.approx(0.5)
    assert metrics["half"]["soft_agreement"] == pytest.approx(0.5)
    assert metrics["half"]["confidence_weight"] == pytest.approx(0.0)
    assert result.confidence_weights["half"] == pytest.approx(0.0)
    assert metrics["good"]["sensitivity"] > 0.5
    assert metrics["ok"]["sensitivity"] > 0.5
    assert sum(result.confidence_weights.values()) == pytest.approx(1.0)


def test_below_threshold_report_is_negative_evidence():
    """Design 5.14 (fit half): an expert that reports 0.0 on
    true-negative pixels and 1.0 on true-positive pixels receives a
    high weight and a low false-positive rate. Its reports below the
    support threshold were counted as negative evidence."""
    rng = np.random.default_rng(7)
    latent = rng.random((2, 20, 25)) < 0.35
    values = np.stack(
        [
            _noisy_probabilities(rng, latent, 0.95, 0.05),
            _noisy_probabilities(rng, latent, 0.85, 0.12),
            np.where(latent, 1.0, 0.0),
        ]
    )
    specs = [
        spec("good", representation="probabilities"),
        spec("ok", representation="probabilities"),
        spec("perfect", representation="probabilities"),
    ]
    result = calibrate_experts(values, specs)
    metrics = {item["name"]: item for item in result.expert_metrics}
    assert metrics["perfect"]["sensitivity"] > 0.98
    assert metrics["perfect"]["false_positive_rate"] < 0.02
    assert result.confidence_weights["perfect"] > result.confidence_weights["good"]
    assert result.confidence_weights["perfect"] > result.confidence_weights["ok"]


def test_support_mean_matches_hand_counted_histogram_panel():
    """Design 5.15, reported half: on a 16-pixel hand-built panel the
    reported per-frame mean_expert_support equals the value hand-counted
    from the support histogram exactly.

    Hand count (support = probability > 0.5, design section 3.3; the
    panel holds no 0.5 values, so the vote rule >= 0.5 used for
    mean_expert_support gives the same counts): support histogram for
    support 0..3 is [4, 3, 4, 5]; unanimous fraction 5/16; below
    min_expert_support=3 fraction 11/16; mean support
    (0*4 + 1*3 + 2*4 + 3*5) / 16 = 26/16 = 1.625.

    The branch record carries no support_histogram, unanimous_fraction,
    or below_min_support_fraction field; those three fields need a
    production change and stay open (see the harness skip reason).
    """
    # Support patterns per pixel (a, b, c); 1 = reports 0.9, 0 = reports 0.1.
    patterns = [
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    ]
    values = np.empty((3, 1, 4, 4), dtype=np.float64)
    for expert_index in range(3):
        for pixel_index, pattern in enumerate(patterns):
            row, column = divmod(pixel_index, 4)
            values[expert_index, 0, row, column] = 0.9 if pattern[expert_index] else 0.1
    specs = [
        spec("a", representation="probabilities"),
        spec("b", representation="probabilities"),
        spec("c", representation="probabilities"),
    ]
    result = calibrate_experts(values, specs)
    report = result.report()
    assert len(report["frames_report"]) == 1
    assert report["frames_report"][0]["mean_expert_support"] == 1.625


def test_all_abstain_pixels_equal_fitted_prior():
    """Design 5.16, posterior half: pixels where every expert reports
    exactly 0.5 carry zero confidence from every expert, so the fitted
    posterior at those pixels equals the fitted prior within 1e-9.

    Hand count: 4 of the 16 pixels are all-abstain, a known fraction of
    0.25. No branch output reports all_abstain_fraction; that reported
    count needs a production change and stays open (see the harness
    skip reason).
    """
    latent = np.zeros((1, 4, 4), dtype=bool)
    latent[0, :2, :] = True
    base = np.where(latent, 0.9, 0.1)
    abstain = np.zeros((1, 4, 4), dtype=bool)
    abstain[0, 0, 0] = True
    abstain[0, 1, 3] = True
    abstain[0, 2, 1] = True
    abstain[0, 3, 2] = True
    experts = []
    for shift in (0.0, 0.05, -0.05):
        panel = np.clip(base + shift * np.where(latent, 1.0, -1.0), 0.05, 0.95)
        experts.append(np.where(abstain, 0.5, panel))
    values = np.stack(experts)
    # The known all-abstain set: every expert reports exactly 0.5.
    assert np.count_nonzero(np.all(values == 0.5, axis=0)) == 4
    specs = [
        spec("a", representation="probabilities"),
        spec("b", representation="probabilities"),
        spec("c", representation="probabilities"),
    ]
    result = calibrate_experts(values, specs)
    assert result.latent_probabilities is not None
    posterior = result.latent_probabilities
    assert np.all(np.abs(posterior[abstain] - result.latent_prior) <= 1e-9)
    # Non-abstain pixels are moved off the prior by the panel evidence.
    assert np.any(np.abs(posterior[~abstain] - result.latent_prior) > 1e-3)
