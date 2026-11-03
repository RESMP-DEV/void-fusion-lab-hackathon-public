from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PUBLISHED_FORMULA_EXPRESSION = (
    "0.35 * pre_pass2_objective + 0.25 * (1 - object_preservation_changed_ratio) "
    "+ 0.20 * (1 - background_changed_ratio) + 0.20 * (1 - temporal_pop_score)"
)
PUBLISHED_FORMULA_TERMS = (
    (0.35, "pre_pass2_objective", False),
    (0.25, "object_preservation_changed_ratio", True),
    (0.20, "background_changed_ratio", True),
    (0.20, "temporal_pop_score", True),
)
RETAINED_PRE_PASS2_OBJECTIVES = {
    "smooth-00": 0.5874731897309333,
    "smooth-02": 0.5880659070744884,
    "smooth-04": 0.5852719043703201,
    "smooth-06": 0.5881883040922771,
}
EXPECTED_RANKING = ("smooth-00", "smooth-06", "smooth-02", "smooth-04")
DEMO_ASSET_SHA256 = "b3e0ca9530662da0a59826e185e3cd801b9f2dcb85524d93ef2eeb8080cfccf1"
DEMO_ASSET_SIZE_BYTES = 47089693
DEMO_RELEASE_URL = (
    "https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/"
    "v1.0.0/void-fusion-demo-2026-11-03-public.mp4"
)


def load_evidence() -> dict:
    return json.loads((ROOT / "evidence/submission-evidence.json").read_text(encoding="utf-8"))


def recompute_objective(formula: dict, candidate: dict) -> float:
    """Recompute one candidate objective from the published formula and its metrics."""
    total = 0.0
    for term in formula["terms"]:
        metric = term["metric"]
        if metric == "pre_pass2_objective":
            value = candidate["pre_pass2_objective"]
        else:
            value = candidate["metrics"][metric]
        if term["complemented"]:
            value = 1.0 - value
        total += term["weight"] * value
    return total


def test_public_submission_evidence_parses_and_has_required_sections() -> None:
    payload = load_evidence()

    assert payload["schema"] == "void-fusion-lab.submission-evidence.v1"
    assert payload["meta_model_usage"]["dense_perception"] == "Meta SAM 3.1"
    assert payload["real_full_chain"]["execution"]["succeeded"] == 4
    assert payload["real_full_chain"]["full_chain_selection"]["selected_candidate_id"] == (
        "smooth-00"
    )
    assert payload["real_full_chain"]["owned_frame_contact"]["owned_frames"] == [0, 1, 7, 13]
    assert payload["demo"]["duration_seconds"] < 180.0
    assert payload["non_claims"]


def test_the_published_objective_formula_is_the_documented_one() -> None:
    formula = load_evidence()["real_full_chain"]["full_chain_objective_formula"]

    assert formula["expression"] == PUBLISHED_FORMULA_EXPRESSION
    published = [
        (term["weight"], term["metric"], term["complemented"]) for term in formula["terms"]
    ]
    assert published == list(PUBLISHED_FORMULA_TERMS)


def test_the_retained_pre_pass2_objectives_are_the_published_ones() -> None:
    candidates = load_evidence()["real_full_chain"]["candidates"]

    assert set(candidates) == set(RETAINED_PRE_PASS2_OBJECTIVES)
    for candidate_id, expected in RETAINED_PRE_PASS2_OBJECTIVES.items():
        assert candidates[candidate_id]["pre_pass2_objective"] == expected


def test_every_candidate_objective_recomputes_from_the_published_formula() -> None:
    payload = load_evidence()
    formula = payload["real_full_chain"]["full_chain_objective_formula"]
    candidates = payload["real_full_chain"]["candidates"]

    for candidate_id, candidate in candidates.items():
        recomputed = recompute_objective(formula, candidate)
        assert recomputed == pytest.approx(candidate["full_chain_objective"], abs=1e-9), (
            candidate_id
        )


def test_the_candidate_objective_ranking_is_smooth_00_smooth_06_smooth_02_smooth_04() -> None:
    payload = load_evidence()
    formula = payload["real_full_chain"]["full_chain_objective_formula"]
    candidates = payload["real_full_chain"]["candidates"]

    recomputed = {
        candidate_id: recompute_objective(formula, candidate)
        for candidate_id, candidate in candidates.items()
    }
    ordered = tuple(sorted(recomputed, key=lambda candidate_id: -recomputed[candidate_id]))

    assert ordered == EXPECTED_RANKING

    selection = payload["real_full_chain"]["full_chain_selection"]
    assert selection["require_full_chain"] is True
    assert selection["selected_basis"] == "full_chain"
    assert selection["selected_candidate_id"] == "smooth-00"
    recorded = [(entry["candidate_id"], entry["objective"]) for entry in selection["ranking"]]
    assert tuple(candidate_id for candidate_id, _ in recorded) == EXPECTED_RANKING
    for candidate_id, objective in recorded:
        assert objective == pytest.approx(recomputed[candidate_id], abs=1e-9)


def test_the_demo_asset_is_bound_by_digest_size_and_release_url() -> None:
    demo = load_evidence()["demo"]

    assert demo["asset_sha256"] == DEMO_ASSET_SHA256
    assert demo["asset_size_bytes"] == DEMO_ASSET_SIZE_BYTES
    assert demo["asset_duration_seconds"] == pytest.approx(155.798549, abs=1e-6)
    assert demo["asset_resolution"] == "1280x720"
    assert demo["asset_fps"] == "24000/1001"
    assert demo["asset_file_name"] == "void-fusion-demo-2026-11-03-public.mp4"
    assert demo["release_tag"] == "v1.0.0"
    assert demo["release_url"] == DEMO_RELEASE_URL
    assert demo["release_url"].endswith(f"/{demo['release_tag']}/{demo['asset_file_name']}")

    # The asset binding and the summary fields describe the same rendered file.
    assert demo["duration_seconds"] == pytest.approx(demo["asset_duration_seconds"], abs=1e-6)
    assert demo["resolution"] == demo["asset_resolution"]
    assert demo["fps"] == demo["asset_fps"]
    assert demo["duration_seconds"] < 180.0

    # The published link in the README is the same release asset the evidence binds.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert DEMO_RELEASE_URL in readme


def test_the_non_claims_separate_the_demo_from_private_pipeline_outputs() -> None:
    claims = load_evidence()["non_claims"]

    assert any(
        "generated presentation recording" in claim
        and "not a redistribution" in claim
        and "private pipeline outputs" in claim
        for claim in claims
    )
    assert any("private pipeline outputs are not redistributed" in claim for claim in claims)


def test_the_public_evidence_and_readme_carry_the_layperson_premise() -> None:
    payload = load_evidence()
    premise = payload["demo"]["one_sentence_premise"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert premise.startswith("Void Fusion changes who or what appears")
    assert payload["demo"]["presentation_guide"] == "docs/presentation.md"
    assert (ROOT / payload["demo"]["presentation_guide"]).is_file()
    assert premise in normalized_readme


def test_every_tracked_evidence_json_parses() -> None:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "evidence"],
        check=True,
        capture_output=True,
        text=True,
    )
    relative_paths = [
        Path(line) for line in completed.stdout.splitlines() if line.endswith(".json")
    ]
    assert relative_paths
    for relative_path in relative_paths:
        payload = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        assert isinstance(payload, dict), relative_path
