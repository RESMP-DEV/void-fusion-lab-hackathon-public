"""Label-free Dawid-Skene calibration for aligned expert mask tensors.

samurai fuses expert masks with configured weights, but no expert in the
current panel carries trusted weights or ground-truth labels. This module fits
a bounded latent-label model to aligned expert reports so that confidence
scales come from measured agreement instead of assumption.

Model
-----
For every sampled pixel ``i`` the estimator assumes a latent binary class
``z_i`` (mask positive versus negative) with prior ``pi``. Expert ``k`` reports
a soft probability ``p_ik`` in ``[0, 1]``, which is read as a hard vote
``v_ik = 1[p_ik >= 0.5]`` and a report confidence ``c_ik = |2p_ik - 1|``.
Calibration uses a powered Bernoulli likelihood:

    L_ik(z_i)^c_ik = P(v_ik | z_i)^c_ik

The exponent is an effective-sample-size weight, not a claim that the model
literally flips votes. A report at exactly 0.5 has zero exponent and contributes
no evidence. The vote is class-conditionally Bernoulli:

    P(v_ik = 1 | z_i = 1) = sensitivity_k
    P(v_ik = 1 | z_i = 0) = false_positive_rate_k

The expectation step uses the confidence-scaled Dawid-Skene likelihood ratio,
and the maximisation step updates confidence-weighted class rates and the prior
with beta smoothing. With unit confidence on every report the estimator reduces
to classic Dawid-Skene.

The latent classes are exchangeable without labels. The positive class is
oriented toward the panel consensus (the mean hard vote), and the anchor
expert's agreement with the latent mask is reported separately. That
orientation is a documented convention, not a claim of ground truth.

Non-claims: this estimator measures inter-expert agreement, not real-model
segmentation quality. It never changes VOID quadmask values, never calls an
external API, and never promotes an expert above its measured skill.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .ensemble import EnsembleError, ExpertSpec, expert_probabilities

NON_CLAIMS = (
    "Latent classes are label-free; the positive class follows the panel "
    "consensus convention, not ground truth.",
    "Agreement metrics describe expert panels, not real-model segmentation quality.",
    "No SAM 3.1 checkpoint, VOID model, or external API was exercised here.",
)


class CalibrationError(RuntimeError):
    """Calibration input, bounds, or generated ensemble output is invalid."""


@dataclass(frozen=True)
class BetaSmoothing:
    """Symmetric Beta pseudocounts for one bounded Bernoulli parameter."""

    alpha: float = 1.0
    beta: float = 1.0

    def validate(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise CalibrationError("beta smoothing counts must be positive")


@dataclass(frozen=True)
class CalibrationSettings:
    """Bounds, smoothing, and sampling limits for the latent-label model."""

    positive_prior: BetaSmoothing = BetaSmoothing()
    sensitivity: BetaSmoothing = BetaSmoothing()
    false_positive_rate: BetaSmoothing = BetaSmoothing()
    tolerance: float = 1e-6
    max_iterations: int = 100
    max_samples: int = 200_000
    max_batch_pixels: int = 8_000_000
    seed: int = 0

    def validate(self) -> None:
        for smoothing in (
            self.positive_prior,
            self.sensitivity,
            self.false_positive_rate,
        ):
            smoothing.validate()
        if self.tolerance < 0:
            raise CalibrationError("convergence tolerance must be nonnegative")
        if self.max_iterations < 1:
            raise CalibrationError("max iterations must be at least one")
        if self.max_samples < 1:
            raise CalibrationError("max samples must be at least one")
        if self.max_batch_pixels < 1:
            raise CalibrationError("max batch pixels must be at least one")


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted latent-label model and its aggregate summaries."""

    expert_metrics: tuple[dict[str, Any], ...]
    confidence_weights: dict[str, float]
    latent_positive_fraction: float
    latent_selected_fraction: float
    latent_prior: float
    pairwise_metrics: tuple[dict[str, Any], ...]
    frame_reports: tuple[dict[str, Any], ...]
    latent_probabilities: np.ndarray | None
    iterations: int
    converged: bool
    relabelled: bool
    anchor: str
    anchor_agreement: float
    samples: int
    shape: tuple[int, int, int]
    settings: CalibrationSettings

    @property
    def latent_masks(self) -> np.ndarray:
        """Thresholded latent mask.

        The in-memory calibration path always retains the latent field. The
        file path streams the field to disk instead, so it exposes an empty
        array rather than pretending the latent field is available.
        """
        if self.latent_probabilities is None:
            return np.zeros((0, 0, 0), dtype=bool)
        return self.latent_probabilities >= 0.5

    def report(self) -> dict[str, Any]:
        return {
            "method": "dawid_skene_bernoulli_em",
            "labels": "none",
            "shape": {
                "frames": self.shape[0],
                "height": self.shape[1],
                "width": self.shape[2],
            },
            "samples": self.samples,
            "iterations": self.iterations,
            "converged": self.converged,
            "latent_positive_fraction": self.latent_positive_fraction,
            "latent_selected_fraction": self.latent_selected_fraction,
            "latent_prior": self.latent_prior,
            "confidence_weights": dict(self.confidence_weights),
            "expert_metrics": list(self.expert_metrics),
            "experts": [item["name"] for item in self.expert_metrics],
            "pairwise": list(self.pairwise_metrics),
            "frames_report": list(self.frame_reports),
            "smoothing": {
                "positive_prior": vars(self.settings.positive_prior),
                "sensitivity": vars(self.settings.sensitivity),
                "false_positive_rate": vars(self.settings.false_positive_rate),
            },
            "bounds": {
                "tolerance": self.settings.tolerance,
                "max_iterations": self.settings.max_iterations,
                "max_samples": self.settings.max_samples,
                "max_batch_pixels": self.settings.max_batch_pixels,
                "seed": self.settings.seed,
            },
            "orientation": {
                "anchor": self.anchor,
                "anchor_agreement": self.anchor_agreement,
                "relabelled": self.relabelled,
                "convention": "positive class follows the panel consensus",
            },
            "non_claims": list(NON_CLAIMS),
        }


@dataclass(frozen=True)
class _Fit:
    """Class-conditional vote rates fitted by the expectation-maximisation loop."""

    sensitivity: np.ndarray
    false_positive_rate: np.ndarray
    prior: float
    iterations: int
    converged: bool
    relabelled: bool


class _Accumulator:
    """Streaming aggregation of expert metrics over bounded frame batches."""

    def __init__(self, expert_count: int) -> None:
        self.expert_count = expert_count
        self.observation_total = 0
        self.vote_sum = np.zeros(expert_count)
        self.probability_sum = np.zeros(expert_count)
        self.squared_error_sum = np.zeros(expert_count)
        self.agreement_sum = np.zeros(expert_count)
        self.soft_agreement_sum = np.zeros(expert_count)
        self.posterior_sum = 0.0
        self.selected_sum = 0.0
        self.pair_vote_sum = np.zeros((expert_count, expert_count))
        self.pair_intersection = np.zeros((expert_count, expert_count))
        self.pair_disagreement = np.zeros((expert_count, expert_count))
        self.pair_brier = np.zeros((expert_count, expert_count))
        self.frame_reports: list[dict[str, Any]] = []

    def update(self, probabilities: np.ndarray, posterior: np.ndarray) -> None:
        """Fold one frame batch of expert probabilities into the aggregates."""
        flat = probabilities.reshape(self.expert_count, -1)
        flat_posterior = posterior.reshape(-1)
        votes = (flat >= 0.5).astype(np.float64)
        latent = (flat_posterior >= 0.5).astype(np.float64)
        for frame_index in range(posterior.shape[0]):
            frame_probabilities = probabilities[:, frame_index].reshape(self.expert_count, -1)
            frame_votes = (frame_probabilities >= 0.5).astype(np.float64)
            frame_posterior = posterior[frame_index].reshape(-1)
            pair_disagreements = [
                float(np.mean(frame_votes[first] != frame_votes[second]))
                for first in range(self.expert_count)
                for second in range(first + 1, self.expert_count)
            ]
            pair_briers = [
                float(np.mean((frame_probabilities[first] - frame_probabilities[second]) ** 2))
                for first in range(self.expert_count)
                for second in range(first + 1, self.expert_count)
            ]
            self.frame_reports.append(
                {
                    "mean_latent_probability": float(np.mean(frame_posterior)),
                    "latent_selected_fraction": float(np.mean(frame_posterior >= 0.5)),
                    "mean_expert_support": float(np.mean(np.sum(frame_votes, axis=0))),
                    "mean_pairwise_disagreement": (
                        float(np.mean(pair_disagreements)) if pair_disagreements else 0.0
                    ),
                    "mean_pairwise_brier": float(np.mean(pair_briers)) if pair_briers else 0.0,
                }
            )
        for index in range(self.expert_count):
            self.vote_sum[index] += float(votes[index].sum())
            self.probability_sum[index] += float(flat[index].sum())
            self.squared_error_sum[index] += float(np.sum((flat[index] - flat_posterior) ** 2))
            self.agreement_sum[index] += float(np.count_nonzero(votes[index] == latent))
            self.soft_agreement_sum[index] += float(
                np.sum(flat[index] * flat_posterior + (1.0 - flat[index]) * (1.0 - flat_posterior))
            )
        self.posterior_sum += float(flat_posterior.sum())
        self.selected_sum += float(latent.sum())
        self.observation_total += flat_posterior.size
        for first in range(self.expert_count):
            for second in range(first + 1, self.expert_count):
                self.pair_vote_sum[first, second] += float(votes[first].sum())
                self.pair_vote_sum[second, first] += float(votes[second].sum())
                self.pair_intersection[first, second] += float(
                    np.count_nonzero(votes[first] * votes[second])
                )
                self.pair_disagreement[first, second] += float(
                    np.count_nonzero(votes[first] != votes[second])
                )
                self.pair_brier[first, second] += float(np.sum((flat[first] - flat[second]) ** 2))

    def soft_agreement(self) -> np.ndarray:
        """Return each expert's soft agreement with the latent mask."""
        total = max(self.observation_total, 1)
        return self.soft_agreement_sum / total

    def expert_metrics(
        self,
        fit: _Fit,
        weights: np.ndarray,
        specs: Sequence[ExpertSpec],
    ) -> list[dict[str, Any]]:
        """Aggregate per-expert metrics against the estimated latent mask."""
        total = max(self.observation_total, 1)
        metrics = []
        for index, spec in enumerate(specs):
            metrics.append(
                {
                    "name": spec.name,
                    "representation": spec.representation,
                    "temperature": spec.temperature,
                    "sensitivity": float(fit.sensitivity[index]),
                    "false_positive_rate": float(fit.false_positive_rate[index]),
                    "youden_j": float(fit.sensitivity[index] - fit.false_positive_rate[index]),
                    "balanced_accuracy": float(
                        (fit.sensitivity[index] + (1.0 - fit.false_positive_rate[index])) / 2.0
                    ),
                    "latent_agreement": float(self.agreement_sum[index] / total),
                    "soft_agreement": float(self.soft_agreement_sum[index] / total),
                    "brier_latent": float(self.squared_error_sum[index] / total),
                    "positive_vote_fraction": float(self.vote_sum[index] / total),
                    "mean_probability": float(self.probability_sum[index] / total),
                    "confidence_weight": float(weights[index]),
                }
            )
        return metrics

    def pairwise_metrics(self, specs: Sequence[ExpertSpec]) -> list[dict[str, Any]]:
        """Aggregate pairwise Dice, Brier, and disagreement summaries."""
        total = max(self.observation_total, 1)
        summaries = []
        for first in range(len(specs)):
            for second in range(first + 1, len(specs)):
                denominator = self.pair_vote_sum[first, second] + self.pair_vote_sum[second, first]
                summaries.append(
                    {
                        "experts": [specs[first].name, specs[second].name],
                        "dice": (
                            (2.0 * self.pair_intersection[first, second] / denominator)
                            if denominator
                            else 1.0
                        ),
                        "brier": float(self.pair_brier[first, second] / total),
                        "disagreement": float(self.pair_disagreement[first, second] / total),
                        "positive_vote_fraction": [
                            float(self.pair_vote_sum[first, second] / total),
                            float(self.pair_vote_sum[second, first] / total),
                        ],
                    }
                )
        return summaries


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply the logistic function without overflowing on large magnitudes."""
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _transform_experts(values: np.ndarray, specs: Sequence[ExpertSpec]) -> np.ndarray:
    """Apply the shared expert transform with calibration-specific errors."""
    try:
        return expert_probabilities(values, specs)
    except EnsembleError as error:
        raise CalibrationError(str(error)) from error


def _sha256_path(path: Path) -> str:
    """Hash a source artifact in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_indices(total: int, count: int, seed: int) -> np.ndarray:
    """Choose a deterministic bounded subsample of flat observation indices."""
    if count >= total:
        return np.arange(total)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(total, size=count, replace=False))


def _sign_and_confidence(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split soft reports into a hard sign and a per-report confidence weight."""
    sign = (probabilities >= 0.5).astype(np.float64)
    confidence = np.abs(2.0 * probabilities - 1.0)
    return sign, confidence


def _posterior(
    sign: np.ndarray,
    confidence: np.ndarray,
    sensitivity: np.ndarray,
    false_positive_rate: np.ndarray,
    prior: float,
) -> np.ndarray:
    """Compute the posterior probability of the latent positive class.

    The log-odds use the classic Dawid-Skene likelihood ratio for each expert
    vote, scaled by that report's confidence so that a report near 0.5 moves
    the latent posterior far less than a decisive report.
    """
    log_odds = np.full(sign.shape[1:], np.log(prior) - np.log1p(-prior))
    for index in range(sign.shape[0]):
        sens = sensitivity[index]
        fpr = false_positive_rate[index]
        positive_ratio = np.where(
            sign[index] >= 0.5,
            sens / np.clip(fpr, 1e-9, None),
            (1.0 - sens) / np.clip(1.0 - fpr, 1e-9, None),
        )
        log_odds = log_odds + confidence[index] * np.log(positive_ratio)
    return _sigmoid(log_odds)


def _fit_bernoulli_em(
    probabilities: np.ndarray,
    settings: CalibrationSettings,
) -> _Fit:
    """Fit per-expert sensitivity and false-positive rate on a bounded sample.

    The sufficient statistics are confidence-weighted, so a report near 0.5
    contributes little evidence in either the expectation or the maximisation
    step. With unit confidence the estimator reduces to classic Dawid-Skene.
    """
    sign, confidence = _sign_and_confidence(probabilities)
    expert_count, observation_count = sign.shape
    prior = float(np.clip(np.mean(sign), 1e-3, 1.0 - 1e-3))
    sensitivity = np.full(expert_count, 0.9)
    false_positive_rate = np.full(expert_count, 0.1)
    converged = False
    iterations = 0
    for _step in range(1, settings.max_iterations + 1):
        iterations = _step
        posterior = _posterior(sign, confidence, sensitivity, false_positive_rate, prior)
        positive_votes = (sign >= 0.5).astype(np.float64)
        next_sensitivity = np.empty(expert_count)
        next_false_positive_rate = np.empty(expert_count)
        for index in range(expert_count):
            positive_weight = confidence[index] * posterior
            negative_weight = confidence[index] * (1.0 - posterior)
            next_sensitivity[index] = (
                float(np.sum(positive_weight * positive_votes[index])) + settings.sensitivity.alpha
            ) / (
                float(positive_weight.sum())
                + settings.sensitivity.alpha
                + settings.sensitivity.beta
            )
            next_false_positive_rate[index] = (
                float(np.sum(negative_weight * positive_votes[index]))
                + settings.false_positive_rate.alpha
            ) / (
                float(negative_weight.sum())
                + settings.false_positive_rate.alpha
                + settings.false_positive_rate.beta
            )
        next_prior = (float(posterior.sum()) + settings.positive_prior.alpha) / (
            observation_count + settings.positive_prior.alpha + settings.positive_prior.beta
        )
        change = max(
            float(np.max(np.abs(next_sensitivity - sensitivity))),
            float(np.max(np.abs(next_false_positive_rate - false_positive_rate))),
            abs(float(next_prior) - prior),
        )
        sensitivity = next_sensitivity
        false_positive_rate = next_false_positive_rate
        prior = float(next_prior)
        if change <= settings.tolerance:
            converged = True
            break

    fit = _Fit(
        sensitivity=sensitivity,
        false_positive_rate=false_positive_rate,
        prior=prior,
        iterations=iterations,
        converged=converged,
        relabelled=False,
    )
    posterior = _posterior(sign, confidence, fit.sensitivity, fit.false_positive_rate, fit.prior)
    panel_vote = np.mean(sign, axis=0)
    agreement = float(np.mean(panel_vote * posterior + (1.0 - panel_vote) * (1.0 - posterior)))
    if agreement < 0.5:
        fit = _Fit(
            sensitivity=fit.false_positive_rate,
            false_positive_rate=fit.sensitivity,
            prior=1.0 - fit.prior,
            iterations=fit.iterations,
            converged=fit.converged,
            relabelled=True,
        )
    return fit


def _confidence_weights(agreement: np.ndarray) -> np.ndarray:
    """Normalize measured skill above chance into fusion weights.

    The input is each expert's soft agreement with the estimated latent
    mask. Agreement above 0.5 is what separates a useful expert from a
    coin-flipping one, so the excess above chance becomes the weight.
    """
    scores = np.clip(agreement - 0.5, 0.0, None)
    total = float(scores.sum())
    if total <= 0:
        return np.full(agreement.shape, 1.0 / agreement.size)
    return scores / total


def _validate_experts(values: np.ndarray, specs: Sequence[ExpertSpec]) -> None:
    """Reject malformed expert batches before any estimation happens."""
    if values.ndim != 4:
        raise CalibrationError("expert batch must have experts,frames,height,width shape")
    if values.shape[0] != len(specs):
        raise CalibrationError("expert count does not match tensor count")
    if len(specs) < 3:
        raise CalibrationError("at least three experts are required for label-free triangulation")
    if not np.isfinite(values).all():
        raise CalibrationError("expert values contain NaN or infinity")
    for index, spec in enumerate(specs):
        if spec.kind != "tensor":
            raise CalibrationError(f"expert {spec.name}: calibration requires tensor paths")
        if spec.representation == "probabilities" and (
            np.min(values[index]) < 0 or np.max(values[index]) > 1
        ):
            raise CalibrationError(f"expert {spec.name}: probabilities must be in [0,1]")


def _latent_posterior(
    probabilities: np.ndarray,
    fit: _Fit,
) -> np.ndarray:
    """Apply the fitted latent-label model to a batch of expert probabilities."""
    sign, confidence = _sign_and_confidence(probabilities)
    return _posterior(sign, confidence, fit.sensitivity, fit.false_positive_rate, fit.prior)


def _calibrate_arrays(
    arrays: Sequence[np.ndarray],
    specs: Sequence[ExpertSpec],
    settings: CalibrationSettings,
    latent_output: Path | None = None,
    keep_latent: bool = False,
) -> CalibrationResult:
    """Fit the latent-label model over memory-mapped or in-memory expert tensors."""
    expert_count = len(specs)
    shapes = {tuple(int(size) for size in array.shape) for array in arrays}
    if len(shapes) != 1:
        raise CalibrationError("expert tensors have mismatching frames/height/width")
    frames, height, width = shapes.pop()
    if frames < 1 or height < 1 or width < 1:
        raise CalibrationError("expert tensors must not be empty")
    total = frames * height * width

    sample_indices = _sample_indices(total, settings.max_samples, settings.seed)
    sample_raw = np.stack(
        [np.asarray(array.reshape(-1)[sample_indices], dtype=np.float64) for array in arrays]
    )
    sample_probabilities = _transform_experts(
        sample_raw.reshape(expert_count, 1, 1, -1), specs
    ).reshape(expert_count, -1)
    fit = _fit_bernoulli_em(sample_probabilities, settings)

    latent_store: np.ndarray | None = None
    if latent_output is not None:
        latent_output.parent.mkdir(parents=True, exist_ok=True)
        latent_store = np.lib.format.open_memmap(
            latent_output, mode="w+", dtype=np.float32, shape=(frames, height, width)
        )
    latent_frames: list[np.ndarray] = [] if keep_latent else []

    accumulator = _Accumulator(expert_count)
    frame_step = max(1, settings.max_batch_pixels // max(1, expert_count * height * width))
    for start in range(0, frames, frame_step):
        stop = min(start + frame_step, frames)
        raw = np.stack([np.asarray(array[start:stop], dtype=np.float64) for array in arrays])
        probabilities = _transform_experts(raw, specs)
        posterior = _latent_posterior(probabilities, fit)
        accumulator.update(probabilities, posterior)
        if latent_store is not None:
            latent_store[start:stop] = posterior.reshape(stop - start, height, width).astype(
                np.float32
            )
        if keep_latent:
            latent_frames.append(posterior.reshape(stop - start, height, width).copy())
    if latent_store is not None:
        latent_store.flush()

    latent_probabilities = np.concatenate(latent_frames, axis=0) if keep_latent else None
    weights = _confidence_weights(accumulator.soft_agreement())
    metrics = accumulator.expert_metrics(fit, weights, specs)
    anchor_agreement = float(metrics[0]["latent_agreement"]) if metrics else 0.0
    confidence_weights = {spec.name: float(weights[index]) for index, spec in enumerate(specs)}
    return CalibrationResult(
        expert_metrics=tuple(metrics),
        confidence_weights=confidence_weights,
        latent_positive_fraction=float(
            accumulator.posterior_sum / max(accumulator.observation_total, 1)
        ),
        latent_selected_fraction=float(
            accumulator.selected_sum / max(accumulator.observation_total, 1)
        ),
        latent_prior=fit.prior,
        pairwise_metrics=tuple(accumulator.pairwise_metrics(specs)),
        frame_reports=tuple(accumulator.frame_reports),
        latent_probabilities=latent_probabilities,
        iterations=fit.iterations,
        converged=fit.converged,
        relabelled=fit.relabelled,
        anchor=specs[0].name,
        anchor_agreement=anchor_agreement,
        samples=int(sample_indices.size),
        shape=(frames, height, width),
        settings=settings,
    )


def _load_tolerant_config(config_path: Path) -> dict[str, Any]:
    """Read an ensemble config without requiring the input video to exist."""
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot load ensemble config {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise CalibrationError("ensemble config must be a JSON object")
    experts = raw.get("experts")
    if not isinstance(experts, list) or len(experts) < 3:
        raise CalibrationError("at least three experts are required for label-free triangulation")
    fusion = raw.get("fusion")
    if not isinstance(fusion, dict):
        raise CalibrationError("ensemble config fusion must be an object")
    raw.setdefault("input_video", "")
    for item in experts:
        if not isinstance(item, dict):
            raise CalibrationError("ensemble config experts must be objects")
        if not item.get("name"):
            raise CalibrationError("ensemble config expert requires a name")
        if item.get("kind", "tensor") != "tensor":
            raise CalibrationError(
                f"expert {item.get('name')}: calibration requires tensor experts"
            )
        if not item.get("path"):
            raise CalibrationError(f"expert {item.get('name')}: tensor path is required")
        item.setdefault("weight", 1.0)
        item.setdefault("temperature", 1.0)
        item.setdefault("representation", "logits")
    return raw


def _expert_specs(raw: dict[str, Any]) -> list[ExpertSpec]:
    """Build validated expert specs from a raw ensemble config."""
    specs = []
    for item in raw["experts"]:
        spec = ExpertSpec(
            name=item["name"],
            weight=float(item.get("weight", 1.0)),
            temperature=float(item.get("temperature", 1.0)),
            kind="tensor",
            path=Path(item["path"]).expanduser(),
            representation=item.get("representation", "logits"),
            prompt=item.get("prompt"),
        )
        spec.validate()
        specs.append(spec)
    return specs


def _load_expert_tensors(specs: Sequence[ExpertSpec]) -> list[np.ndarray]:
    """Memory-map expert tensors and enforce identical frames,height,width."""
    arrays: list[np.ndarray] = []
    shape: tuple[int, int, int] | None = None
    for spec in specs:
        if spec.path is None:
            raise CalibrationError(f"expert {spec.name}: tensor path is required")
        try:
            array = np.load(spec.path, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise CalibrationError(f"expert {spec.name}: cannot read tensor: {error}") from error
        if array.ndim != 3:
            raise CalibrationError(
                f"expert {spec.name}: tensor must have frames,height,width shape"
            )
        current = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
        if shape is None:
            shape = current
        elif current != shape:
            raise CalibrationError("expert tensors have mismatching frames/height/width")
        arrays.append(array)
    return arrays


def _generated_config(
    raw: dict[str, Any],
    weights: dict[str, float],
    config_path: Path,
    report_path: Path,
    expert_sha256: dict[str, str],
    source_config_sha256: str,
) -> dict[str, Any]:
    """Return the input config with calibrated weights and preserved paths and prompts."""
    generated = json.loads(json.dumps(raw))
    for item in generated["experts"]:
        name = item.get("name")
        if name not in weights:
            raise CalibrationError(f"calibrated expert {name} is missing from config")
        item["weight"] = weights[name]
    generated["calibration"] = {
        "method": "dawid_skene_bernoulli_em",
        "labels": "none",
        "source_config": str(config_path),
        "report": str(report_path),
        "confidence_weights": weights,
        "expert_sha256": expert_sha256,
        "source_config_sha256": source_config_sha256,
    }
    return generated


def calibrate_config(
    config_path: Path,
    generated_config_path: Path,
    report_path: Path,
    settings: CalibrationSettings | None = None,
    latent_output: Path | None = None,
) -> dict[str, Any]:
    """Calibrate configured tensor experts and emit a weighted ensemble config."""
    bounds = settings or CalibrationSettings()
    bounds.validate()
    raw = _load_tolerant_config(config_path)
    specs = _expert_specs(raw)
    arrays = _load_expert_tensors(specs)
    expert_sha256 = {spec.name: _sha256_path(spec.path) for spec in specs if spec.path is not None}
    source_config_sha256 = _sha256_path(config_path)
    result = _calibrate_arrays(arrays, specs, bounds, latent_output=latent_output)
    report = result.report()
    report.update(
        {
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input_config": str(config_path),
            "generated_config": str(generated_config_path),
            "latent_output": str(latent_output) if latent_output else None,
            "input_video": raw.get("input_video", ""),
            "artifacts": {
                "source_config_sha256": source_config_sha256,
                "expert_sha256": expert_sha256,
            },
            "experts": [spec.name for spec in specs],
        }
    )
    generated = _generated_config(
        raw,
        result.confidence_weights,
        config_path,
        report_path,
        expert_sha256,
        source_config_sha256,
    )
    generated_config_path.parent.mkdir(parents=True, exist_ok=True)
    generated_config_path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def calibrate_experts(
    values: np.ndarray,
    specs: Sequence[ExpertSpec],
    settings: CalibrationSettings | None = None,
) -> CalibrationResult:
    """Estimate a latent binary mask and calibrated confidence weights in memory."""
    bounds = settings or CalibrationSettings()
    bounds.validate()
    _validate_experts(values, specs)
    arrays = [np.asarray(values[index]) for index in range(values.shape[0])]
    return _calibrate_arrays(arrays, specs, bounds, keep_latent=True)
