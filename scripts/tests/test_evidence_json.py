from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/auto-edit-submission-2026-11-04.json"


def load_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_auto_edit_evidence_binds_the_submission_demo() -> None:
    payload = load_evidence()
    demo = payload["final_demo"]

    assert payload["project_name"] == "VOID Auto-Edit"
    assert demo["duration_seconds"] == 178.0
    assert demo["frames"] == 4267
    assert demo["width"] == 1920
    assert demo["height"] == 1080
    assert demo["sha256"] == ("77ef47319734f639b86ffff186d1780a6d381b3659b2445d02082611fa8d7fc0")
    assert payload["release_artifacts"]["demo_url"].endswith(
        "/v4.0.0/void-auto-edit-removal-only.mp4"
    )


def test_auto_edit_evidence_preserves_claims_and_limits() -> None:
    payload = load_evidence()
    candidate_c = payload["headline_results"]["candidate_C"]
    candidate_f = payload["headline_results"]["candidate_F"]
    candidate_d = payload["headline_results"]["candidate_D"]

    assert candidate_c["target_point_referent_lock_judged_bystander_spill"] == 0
    assert candidate_c["subject_retention"] == 0.99995
    assert candidate_c["void_ghost_frame_0"] < candidate_c["ghost_threshold"]
    assert candidate_f["recovery_fraction"] == 0.964483
    assert candidate_d["frames"] == 90
    assert any("post hoc" in claim for claim in payload["non_claims"])
    assert any("frames 16-23" in claim for claim in payload["non_claims"])


def test_readme_and_presentation_bind_the_live_demo() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    presentation = (ROOT / "docs/presentation.md").read_text(encoding="utf-8")
    demo_url = (
        "https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/"
        "download/v4.0.0/void-auto-edit-removal-only.mp4"
    )

    assert demo_url in readme
    assert "178.000 seconds" in presentation
    assert "68,355-70,233" in readme
    assert "96.4483%" in readme


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
        for forbidden in (
            "/Users/",
            "/home/",
            "/private/",
            "/Volumes/",
            "10.243.",
            "b550-zt",
            "strix-zt",
            "source_revision",
        ):
            assert forbidden not in raw, (relative_path, forbidden)
