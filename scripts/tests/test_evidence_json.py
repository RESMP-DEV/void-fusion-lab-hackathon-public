from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/advertising-campaign-evidence.json"


def load_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_advertising_campaign_evidence_binds_the_public_result() -> None:
    payload = load_evidence()

    assert payload["schema"] == "void-fusion-lab.advertising-campaign.v2"
    assert payload["selection"]["recovered_runs"] == 17
    assert payload["selection"]["recovered_frames"] == 465
    assert payload["selection"]["pass1_sequences_packaged"] == 17
    assert payload["selection"]["retained_shots"] == 16
    assert payload["selection"]["retained_frames"] == 457
    assert payload["selection"]["retained_brand_visible_frames"] == 457


def test_advertising_campaign_evidence_binds_release_artifacts() -> None:
    artifacts = load_evidence()["public_artifacts"]
    long_form = artifacts["long_form"]

    assert long_form["sha256"] == (
        "821eadc309574bf286c92172c31f493b3f0acb062f8633def778f0f5aca3c0cf"
    )
    assert long_form["bytes"] == 62363142
    assert long_form["frames"] == 2573
    assert long_form["duration_seconds"] == 107.208333
    assert long_form["url"].endswith("/v2.0.0/void-fusion-advertising-2026-11-03.mp4")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    presentation = (ROOT / "docs/presentation.md").read_text(encoding="utf-8")
    assert long_form["url"] in readme
    assert artifacts["comparison_master"]["url"] in readme
    assert "107.208 seconds" in presentation


def test_advertising_campaign_evidence_preserves_limitations() -> None:
    claims = load_evidence()["non_claims"]

    assert any("not learned image-to-3D reconstruction" in claim for claim in claims)
    assert any("not official-predictor ground truth" in claim for claim in claims)
    assert any("do not prove artistic quality" in claim for claim in claims)
    assert any("does not claim production readiness" in claim for claim in claims)


def test_every_tracked_evidence_json_parses_without_private_paths() -> None:
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
        raw = (ROOT / relative_path).read_text(encoding="utf-8")
        payload = json.loads(raw)
        assert isinstance(payload, dict), relative_path
        assert "/Users/" not in raw, relative_path
        assert "/home/" not in raw, relative_path
