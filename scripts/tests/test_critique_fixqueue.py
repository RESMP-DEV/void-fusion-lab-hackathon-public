"""R2-F1/F2 regression tests for the critique verdict-class contract.

Contract: the critique item builder writes each planted item's
"expected" as the dict {"verdict_class": ..., "planted_type": ...}.
Both readers consume that one contract: critique_harness.score() (which
called expected.get(...) and crashed with AttributeError on the old bare
string) and scripts/critique/score.py (which compared the bare string).
score.py must also survive an unknown planted_type (R2-F2): the old
TYPE_MATCH[...] index raised KeyError and crashed the scorer.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {rel}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harness = _load("critique_harness_fixqueue", "scripts/critique_harness.py")

FINDING = {"type": "hole", "severity": "high", "location": "arm", "detail": "hole"}


def _fd(verdict: str, findings: list[dict[str, str]]) -> dict[str, Any]:
    return {"verdict": verdict, "findings": findings}


def _writer_expected(verdict: str, planted_type: str) -> dict[str, str]:
    """The exact shape the critique item builder writes for a planted item."""
    return {"verdict_class": verdict, "planted_type": planted_type}


# ---------------- R2-F1: writer contract ----------------


def test_writer_contract_feeds_harness_score() -> None:
    flagged = {"findings": _fd("FAIL", [FINDING])}
    clean = {"findings": _fd("PASS", [])}
    defect = _writer_expected("defect", "hole")
    ok = _writer_expected("ok", "none")
    # No AttributeError; the dict the writer emits scores in the harness.
    assert harness.score(flagged, defect)["agreement"] == "DETECTED_TYPED"
    assert harness.score(clean, defect)["agreement"] == "MISSED"
    assert harness.score(clean, ok)["agreement"] == "MATCH"
    assert harness.score(flagged, ok)["agreement"] == "FALSE_ALARM"


def _run_score_py(tmp_path: Path, items: list[dict[str, Any]], records: dict[str, Any]):
    work = tmp_path / "score-run"
    work.mkdir()
    shutil.copy(ROOT / "scripts/critique/score.py", work / "score.py")
    (work / "items.json").write_text(json.dumps(items))
    recs = [{"item_id": iid, "findings": fd} for iid, fd in records.items()]
    (work / "results.json").write_text(json.dumps({"calls_used": len(recs), "records": recs}))
    proc = subprocess.run(
        [sys.executable, str(work / "score.py")],
        capture_output=True,
        text=True,
        cwd=work,
        check=False,
    )
    return work, proc


def _run_score_files(tmp_path: Path, items_text: str | None, results_text: str):
    work = tmp_path / "score-input-run"
    work.mkdir()
    shutil.copy(ROOT / "scripts/critique/score.py", work / "score.py")
    if items_text is not None:
        (work / "items.json").write_text(items_text)
    (work / "results.json").write_text(results_text)
    return work, subprocess.run(
        [sys.executable, str(work / "score.py")],
        capture_output=True,
        text=True,
        cwd=work,
        check=False,
    )


def test_score_py_names_a_missing_items_file(tmp_path: Path) -> None:
    work, proc = _run_score_files(tmp_path, None, '{"calls_used":0,"records":[]}')
    assert proc.returncode != 0
    assert "cannot read items file" in proc.stderr
    assert not (work / "score.json").exists()


def test_score_py_names_malformed_results_json(tmp_path: Path) -> None:
    work, proc = _run_score_files(tmp_path, "[]", "{not json")
    assert proc.returncode != 0
    assert "results file is not valid JSON" in proc.stderr
    assert not (work / "score.json").exists()


@pytest.mark.parametrize("missing_key", ["records", "calls_used"])
def test_score_py_names_a_missing_results_key(tmp_path: Path, missing_key: str) -> None:
    payload = {"calls_used": 0, "records": []}
    payload.pop(missing_key)
    work, proc = _run_score_files(tmp_path, "[]", json.dumps(payload))
    assert proc.returncode != 0
    expected = (
        "results file records must" if missing_key == "records" else "results file calls_used must"
    )
    assert expected in proc.stderr
    assert not (work / "score.json").exists()


def test_writer_contract_feeds_score_py(tmp_path: Path) -> None:
    items = [
        {
            "item_id": "D1",
            "set": "v9planted",
            "expected": _writer_expected("defect", "hole"),
            "planted_type": "hole",
        },
        {
            "item_id": "G1",
            "set": "v9planted",
            "expected": _writer_expected("ok", "none"),
            "planted_type": "none",
        },
    ]
    work, proc = _run_score_py(
        tmp_path, items, {"D1": _fd("FAIL", [FINDING]), "G1": _fd("PASS", [])}
    )
    assert proc.returncode == 0, proc.stderr
    planted = json.loads((work / "score.json").read_text())["planted"]
    assert planted["n_defect"] == 1
    assert planted["detected"] == 1
    assert planted["typed"] == 1
    assert planted["good_n"] == 1
    assert planted["good_fp"] == 0


def test_score_py_ok_items_are_not_defects(tmp_path: Path) -> None:
    # Refuted half of the original finding, pinned: an "ok" verdict_class
    # lands in the good branch exactly as the bare "ok" string did on the
    # real manifest; only "defect" counts as a defect.
    items = [
        {
            "item_id": "G1",
            "set": "v9planted",
            "expected": _writer_expected("ok", "hole"),
            "planted_type": "hole",
        }
    ]
    work, proc = _run_score_py(tmp_path, items, {"G1": _fd("FAIL", [FINDING])})
    assert proc.returncode == 0, proc.stderr
    planted = json.loads((work / "score.json").read_text())["planted"]
    assert planted["n_defect"] == 0
    assert planted["good_n"] == 1
    assert planted["good_fp"] == 1


def test_score_py_records_a_missing_verdict_class_as_an_input_error(tmp_path: Path) -> None:
    items = [
        {
            "item_id": "I1",
            "set": "v9planted",
            "expected": {"planted_type": "hole"},
            "planted_type": "hole",
        }
    ]
    work, proc = _run_score_py(tmp_path, items, {"I1": _fd("FAIL", [FINDING])})
    assert proc.returncode == 0, proc.stderr
    planted = json.loads((work / "score.json").read_text())["planted"]
    assert planted["input_errors"] == 1
    assert planted["good_n"] == 0
    assert planted["good_fp"] == 0
    assert "input errors: 1" in proc.stdout


# ---------------- R2-F2: unknown planted_type must not crash ----------------


def test_score_py_unknown_planted_type_warns_and_scores_untyped(tmp_path: Path) -> None:
    items = [
        {
            "item_id": "D1",
            "set": "v9planted",
            "expected": _writer_expected("defect", "brand_new_type"),
            "planted_type": "brand_new_type",
        }
    ]
    work, proc = _run_score_py(tmp_path, items, {"D1": _fd("FAIL", [FINDING])})
    assert proc.returncode == 0, proc.stderr
    assert "unknown planted_type" in proc.stderr
    assert "brand_new_type" in proc.stderr
    planted = json.loads((work / "score.json").read_text())["planted"]
    assert planted["detected"] == 1  # detection still counts
    assert planted["typed"] == 0  # typed hit is False, not a KeyError


# ---------------- F31c: the two scorers agree on the defect class -----------


def test_harness_and_score_py_agree_on_which_class_is_a_defect(tmp_path):
    classes = ["defect", "ok", "warn", "unknown"]
    items = [
        {
            "item_id": f"I{index}",
            "set": "v9planted",
            "expected": _writer_expected(klass, "hole"),
            "planted_type": "hole",
        }
        for index, klass in enumerate(classes)
    ]
    records = {f"I{index}": _fd("FAIL", [FINDING]) for index in range(len(classes))}
    work, proc = _run_score_py(tmp_path, items, records)
    assert proc.returncode == 0, proc.stderr
    planted = json.loads((work / "score.json").read_text())["planted"]
    # score.py: only the defect class is a defect item.
    assert planted["n_defect"] == 1
    assert planted["good_n"] == len(classes) - 1
    # The harness agrees item by item.
    for index, klass in enumerate(classes):
        expected = _writer_expected(klass, "hole")
        agreement = harness.score({"findings": records[f"I{index}"]}, expected)["agreement"]
        is_defect = agreement in ("DETECTED_TYPED", "DETECTED_UNTYPED", "MISSED")
        assert is_defect == (klass == "defect"), (klass, agreement)


def test_harness_and_score_py_agree_on_a_clean_item(tmp_path):
    classes = ["defect", "ok"]
    items = [
        {
            "item_id": f"I{index}",
            "set": "v9planted",
            "expected": _writer_expected(klass, "hole"),
            "planted_type": "hole",
        }
        for index, klass in enumerate(classes)
    ]
    records = {f"I{index}": _fd("PASS", []) for index in range(len(classes))}
    work, proc = _run_score_py(tmp_path, items, records)
    assert proc.returncode == 0, proc.stderr
    planted = json.loads((work / "score.json").read_text())["planted"]
    assert planted["detected"] == 0
    assert planted["good_fp"] == 0
    for index, klass in enumerate(classes):
        expected = _writer_expected(klass, "hole")
        agreement = harness.score({"findings": records[f"I{index}"]}, expected)["agreement"]
        if klass == "defect":
            assert agreement == "MISSED"
        else:
            assert agreement == "MATCH"


def test_score_py_refuses_an_item_with_no_expected_answer(tmp_path):
    # The harness scores a missing expected answer as unscored, never as a
    # clean item, so the same item must not inflate the false-positive
    # count here either.
    items = [
        {
            "item_id": "D1",
            "set": "v9planted",
            "planted_type": "hole",
        }
    ]
    work, proc = _run_score_py(tmp_path, items, {"D1": _fd("FAIL", [FINDING])})
    assert proc.returncode == 0, proc.stderr
    planted = json.loads((work / "score.json").read_text())["planted"]
    assert planted["input_errors"] == 1
    assert planted["good_n"] == 0
    assert planted["good_fp"] == 0
    assert planted["n_defect"] == 0
    assert harness.score({"findings": _fd("FAIL", [FINDING])}, None)["agreement"] is None


def test_score_py_writes_its_score_file_atomically(tmp_path):
    source = (ROOT / "scripts" / "critique" / "score.py").read_text()
    assert "write_json_atomic(" in source
    assert 'open(os.path.join(HERE, "score.json"), "w")' not in source
    items = [
        {
            "item_id": "D1",
            "set": "v9planted",
            "expected": _writer_expected("defect", "hole"),
            "planted_type": "hole",
        }
    ]
    work, proc = _run_score_py(tmp_path, items, {"D1": _fd("FAIL", [FINDING])})
    assert proc.returncode == 0, proc.stderr
    assert json.loads((work / "score.json").read_text())["planted"]["detected"] == 1
    assert [path.name for path in work.iterdir() if path.name.startswith(".critique-")] == []


# ---------------- F40: the persistence spec names its implementation ---------


def _spec_text() -> str:
    raw = (ROOT / "docs/tensor-persistence-spec.md").read_text(encoding="utf-8")
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("> "):
            stripped = stripped[2:]
        elif stripped == ">":
            stripped = ""
        lines.append(stripped)
    # The note is hard-wrapped, so the assertions below read the text with
    # its line wrapping collapsed.
    return " ".join("\n".join(lines).split())


def test_tensor_persistence_spec_separates_the_two_implemented_manifest_layers():
    spec = _spec_text()
    assert (ROOT / "src/samurai/persist.py").exists()
    assert (ROOT / "src/samurai/expert_persist.py").exists()
    assert "Legacy per-chunk expert store" in spec
    assert "`manifest.json` | `.manifest.lock`" in spec
    assert "D4 versioned expert tensors" in spec
    assert "`experts_manifest.json` | `.experts_manifest.lock`" in spec


def test_tensor_persistence_spec_bounds_the_d4_implementation_claim():
    spec = _spec_text()
    assert "The D4 layer is not this spec." in spec
    assert "versioned-artifact subset" in spec
    assert "Conflict C-1" in spec
