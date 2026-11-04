from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/true3d-tire-campaign-2026-11-03.json"


def load_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_true3d_campaign_evidence_binds_the_public_result() -> None:
    payload = load_evidence()
    results = payload["results"]

    assert payload["schema"] == "void-fusion-lab.true3d-tire-campaign.v2"
    assert len(results) == 17
    assert len({result["id"] for result in results}) == 17
    assert sum(result["frames"] for result in results) == 465
    assert payload["summary"]["pre_pass2_shots"] == len(results)
    assert payload["summary"]["pre_pass2_frames"] == sum(result["frames"] for result in results)
    assert payload["summary"]["pass2_shots"] == sum(
        result["brand_gate_passed"] for result in results
    )
    assert payload["summary"]["pass2_frames_preserved"] == sum(
        result["frames"] for result in results if result["brand_gate_passed"]
    )
    assert payload["summary"]["all_brand_gates_passed"] is True
    assert payload["summary"]["all_object_regions_unchanged"] is True
    assert payload["summary"]["all_cleanup_rings_changed"] is True
    assert all(result["glb_valid"] for result in results)
    assert all(result["inserted_mask_binary"] for result in results)
    assert all(result["cleanup_mask_binary"] for result in results)


def test_true3d_campaign_evidence_binds_release_artifacts() -> None:
    artifacts = load_evidence()["final_artifacts"]
    long_form = artifacts["long_form_demo"]

    assert long_form["sha256"] == (
        "be8c783362b39e89105a934dfcc97fae4a2c66beac97aa3a975e8590093f0e26"
    )
    assert long_form["frames"] == 2613
    assert long_form["duration_seconds"] == 108.875

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    presentation = (ROOT / "docs/presentation.md").read_text(encoding="utf-8")
    demo_url = (
        "https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/"
        "download/v3.0.0/void-fusion-true3d-final.mp4"
    )
    comparison_url = (
        "https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/"
        "download/v3.0.0/void-fusion-true3d-five-stage-comparison.mp4"
    )
    assert demo_url in readme
    assert comparison_url in readme
    assert (
        "https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/"
        "download/v3.0.0/void-fusion-true3d-five-stage-proof.jpg" in readme
    )
    assert "108.875 seconds" in presentation
    assert artifacts["comparison_master"]["sha256"] == (
        "57531ddea479fc5dfe6ecfc4e02f3e150420f1ff4ced8775fa053ce54f6c2bfe"
    )
    assert artifacts["proof_sheet"]["sha256"] == (
        "3a11ec460ea9c393cbfca7ddd5981c7e83159515b72a38f50ba8feec6336142f"
    )


def test_true3d_campaign_evidence_preserves_limitations() -> None:
    claims = load_evidence()["non_claims"]

    assert any("not a scan" in claim for claim in claims)
    assert any("not recovered 3D ground truth" in claim for claim in claims)
    assert any("not OCR" in claim for claim in claims)
    assert any("not artistic quality" in claim for claim in claims)


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
        assert "/private/" not in raw, relative_path
        assert "/Volumes/" not in raw, relative_path
        assert "10.243." not in raw, relative_path
        assert "b550-zt" not in raw, relative_path
        assert "strix-zt" not in raw, relative_path
        assert "source_revision" not in raw, relative_path
