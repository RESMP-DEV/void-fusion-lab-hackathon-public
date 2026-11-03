from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import urllib.error
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


critique = _load("critique_under_test", "scripts/critique/critique.py")
harness = _load("critique_harness_under_test", "scripts/critique_harness.py")

FINDING = {"type": "hole", "severity": "high", "location": "arm", "detail": "hole"}


def _fd(verdict: str, findings: list[dict[str, str]]) -> dict[str, Any]:
    return {"verdict": verdict, "findings": findings}


def _item(tmp_path: Path, image_bytes: bytes, **overrides: Any) -> dict[str, Any]:
    image = tmp_path / "frame.png"
    image.write_bytes(image_bytes)
    item: dict[str, Any] = {
        "item_id": "X1",
        "image": str(image),
        "legend": "quad",
        "scene": "scene",
        "stats": "0: 10",
        "set": "set1",
    }
    item.update(overrides)
    return item


# ---------------- F28: resume must not reuse a verdict for changed input ----------------


def test_fingerprint_changes_with_image_bytes(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    fp1 = critique.item_fingerprint(item)
    (tmp_path / "frame.png").write_bytes(b"image-v2-changed-mask")
    fp2 = critique.item_fingerprint(item)
    assert fp1 is not None and fp2 is not None and fp1 != fp2


def test_fingerprint_changes_with_prompt_inputs(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    base = critique.item_fingerprint(item)
    assert critique.item_fingerprint(item, prompt_suffix="retry") != base
    assert critique.item_fingerprint(item, max_tokens=12000) != base
    assert critique.item_fingerprint({**item, "scene": "other"}) != base
    assert critique.item_fingerprint({**item, "stats": "0: 99"}) != base
    assert critique.item_fingerprint({**item, "legend": "binary"}) != base
    assert critique.item_fingerprint(item) == base


def test_fingerprint_missing_image_is_none(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    item["image"] = str(tmp_path / "absent.png")
    assert critique.item_fingerprint(item) is None


def test_resume_reuses_only_on_fingerprint_match(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    rec = {
        "item_id": "X1",
        "findings": _fd("PASS", []),
        "input_fingerprint": critique.item_fingerprint(item),
    }
    done, carried = critique.resume_state([rec], [item])
    assert done == {"X1": rec}
    assert carried == [rec]


def test_resume_rejects_changed_image(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    rec = {
        "item_id": "X1",
        "findings": _fd("PASS", []),
        "input_fingerprint": critique.item_fingerprint(item),
    }
    (tmp_path / "frame.png").write_bytes(b"image-v2-changed-mask")
    done, carried = critique.resume_state([rec], [item])
    assert done == {}
    assert carried == []  # stale record is dropped, not carried as current


def test_resume_rejects_legacy_record_without_fingerprint(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    rec = {"item_id": "X1", "findings": _fd("PASS", [])}
    done, carried = critique.resume_state([rec], [item])
    assert done == {}
    assert carried == []


def test_resume_carries_out_of_run_record_without_reusing_it(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    other = {"item_id": "ZZ", "findings": _fd("FAIL", [FINDING])}
    unparsed = {"item_id": "X1", "findings": None}
    done, carried = critique.resume_state([other, unparsed], [item])
    assert done == {}
    assert carried == [other]


# ---------------- F31: one canonical flagged definition in both scorers ----------------


def test_is_flagged_truth_table() -> None:
    assert harness.is_flagged(_fd("PASS", [])) is False
    assert harness.is_flagged(_fd("PASS", [FINDING])) is True
    assert harness.is_flagged(_fd("WARN", [])) is True
    assert harness.is_flagged(_fd("FAIL", [])) is True
    assert harness.is_flagged(_fd("WARN", [FINDING])) is True
    assert harness.is_flagged(_fd("FAIL", [FINDING])) is True


def test_harness_score_uses_canonical_flagged() -> None:
    rec = {"findings": _fd("PASS", [FINDING])}
    assert harness.score(rec, {"verdict_class": "ok"})["agreement"] == "FALSE_ALARM"
    defect = {"verdict_class": "defect", "planted_type": "hole"}
    assert harness.score(rec, defect)["agreement"] == "DETECTED_TYPED"
    clean = {"findings": _fd("PASS", [])}
    assert harness.score(clean, {"verdict_class": "ok"})["agreement"] == "MATCH"
    assert harness.score(clean, defect)["agreement"] == "MISSED"


def _run_score_py(tmp_path: Path, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    work = tmp_path / "score-run"
    work.mkdir()
    shutil.copy(ROOT / "scripts/critique/score.py", work / "score.py")
    items = [
        {"item_id": "D1", "set": "v9planted", "expected": "defect", "planted_type": "hole"},
        {"item_id": "G1", "set": "v9planted", "expected": "good", "planted_type": "hole"},
    ]
    (work / "items.json").write_text(json.dumps(items))
    recs = [{"item_id": iid, "findings": fd} for iid, fd in records.items()]
    (work / "results.json").write_text(json.dumps({"calls_used": 2, "records": recs}))
    proc = subprocess.run(
        [sys.executable, str(work / "score.py")],
        capture_output=True,
        text=True,
        cwd=work,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads((work / "score.json").read_text())["planted"]


def test_score_py_pass_with_findings_is_flagged(tmp_path: Path) -> None:
    # The F31 divergent case: PASS verdict carrying a finding.
    planted = _run_score_py(tmp_path, {"D1": _fd("PASS", [FINDING]), "G1": _fd("PASS", [FINDING])})
    assert planted["detected"] == 1  # defect item counts as detected, as in the harness
    assert planted["good_fp"] == 1  # good item counts as a false positive, as in the harness


def test_score_py_pass_without_findings_is_not_flagged(tmp_path: Path) -> None:
    planted = _run_score_py(tmp_path, {"D1": _fd("PASS", []), "G1": _fd("PASS", [])})
    assert planted["detected"] == 0
    assert planted["good_fp"] == 0


# ---------------- T02: fingerprint covers prompt template + model ----------------


def test_fingerprint_changes_with_prompt_template(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    base = critique.item_fingerprint(item)
    original = critique.PROMPT_QUAD
    try:
        critique.PROMPT_QUAD = original + "\nEdited critic instructions."
        assert critique.item_fingerprint(item) != base
    finally:
        critique.PROMPT_QUAD = original
    assert critique.item_fingerprint(item) == base


def test_fingerprint_changes_with_model(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1")
    base = critique.item_fingerprint(item)
    original = critique.MODEL
    try:
        critique.MODEL = "Other-Model"
        assert critique.item_fingerprint(item) != base
    finally:
        critique.MODEL = original


def test_fingerprint_binary_item_uses_binary_template(tmp_path: Path) -> None:
    item = _item(tmp_path, b"image-v1", legend="binary")
    base = critique.item_fingerprint(item)
    original = critique.PROMPT_BINARY
    try:
        critique.PROMPT_BINARY = original + "\nEdited."
        assert critique.item_fingerprint(item) != base
    finally:
        critique.PROMPT_BINARY = original


# ---------------- T09: empty ENV_FILE fails with a named SystemExit ----------------


def test_load_key_empty_env_file_names_minimax_env_file() -> None:
    with pytest.raises(SystemExit, match="MINIMAX_ENV_FILE"):
        critique.load_key("")


# ---------------- F30: env_file placeholders expand variables and user paths --


def test_expand_env_path_expands_variables_and_user_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SPLIT_CRITIQUE_KEY_FILE", str(tmp_path / "key.env"))
    assert harness.expand_env_path("$SPLIT_CRITIQUE_KEY_FILE") == str(tmp_path / "key.env")
    assert harness.expand_env_path("${SPLIT_CRITIQUE_KEY_FILE}") == str(tmp_path / "key.env")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert harness.expand_env_path("~") == str(tmp_path)
    # The user-home path is assembled here so this guard does not itself put
    # a home-relative path into the published tree.
    user_path = Path("~") / "secrets" / "minimax.env"
    assert harness.expand_env_path(str(user_path)) == str(tmp_path / "secrets/minimax.env")


def test_harness_load_key_reads_the_expanded_env_file(monkeypatch, tmp_path):
    env = tmp_path / "key.env"
    env.write_text("MINIMAX_KEY=sekret\n")
    monkeypatch.setenv("SPLIT_CRITIQUE_KEY_FILE", str(env))
    assert harness.load_key("$SPLIT_CRITIQUE_KEY_FILE") == "sekret"
    assert harness.load_key(str(env)) == "sekret"


def test_harness_load_key_fails_closed_on_an_unset_placeholder(monkeypatch):
    monkeypatch.delenv("SPLIT_CRITIQUE_ABSENT", raising=False)
    with pytest.raises(SystemExit, match="placeholder"):
        harness.load_key("$SPLIT_CRITIQUE_ABSENT")


def test_harness_load_key_allows_a_literal_dollar_in_the_resolved_path(tmp_path):
    env_file = tmp_path / "my$key.env"
    env_file.write_text("MINIMAX_KEY=sekret\n")
    assert harness.load_key(str(env_file), [str(tmp_path)]) == "sekret"


def test_critique_load_key_reads_the_expanded_env_file(monkeypatch, tmp_path):
    env = tmp_path / "key.env"
    env.write_text("MINIMAX_KEY=sekret\n")
    monkeypatch.setenv("SPLIT_CRITIQUE_KEY_FILE", str(env))
    assert critique.load_key("$SPLIT_CRITIQUE_KEY_FILE", [str(tmp_path)]) == "sekret"
    with pytest.raises(SystemExit, match="placeholder"):
        critique.load_key("$SPLIT_CRITIQUE_ABSENT")


# ---------------- F31b: only the defect class is a defect (harness side) ----


def test_harness_treats_only_the_defect_class_as_a_defect():
    flagged = {"findings": _fd("FAIL", [FINDING])}
    clean = {"findings": _fd("PASS", [])}
    for verdict_class in ("ok", "warn", "unknown", None):
        expected = {"verdict_class": verdict_class}
        assert harness.score(clean, expected)["agreement"] == "MATCH"
        assert harness.score(flagged, expected)["agreement"] == "FALSE_ALARM"
    defect = {"verdict_class": "defect", "planted_type": "hole"}
    assert harness.score(flagged, defect)["agreement"] == "DETECTED_TYPED"
    assert harness.score(clean, defect)["agreement"] == "MISSED"


def test_harness_reads_the_legacy_bare_verdict_string():
    flagged = {"findings": _fd("FAIL", [FINDING])}
    clean = {"findings": _fd("PASS", [])}
    assert harness.score(flagged, "defect")["agreement"] == "DETECTED_UNTYPED"
    assert harness.score(clean, "defect")["agreement"] == "MISSED"
    assert harness.score(flagged, "ok")["agreement"] == "FALSE_ALARM"
    assert harness.score(clean, "ok")["agreement"] == "MATCH"


# ---------------- F33: one fingerprint covers the retry parameters ----------


def test_fingerprint_covers_the_retry_parameters(tmp_path):
    item = _item(tmp_path, b"image-v1")
    base = critique.item_fingerprint(item)
    assert base is not None
    # The judged input changes with the documented retry parameters ...
    assert (
        critique.item_fingerprint(item, critique.RETRY_PROMPT_SUFFIX, critique.RETRY_MAX_TOKENS)
        != base
    )
    # ... and each retry parameter on its own changes it too, so a verdict
    # judged under one prompt can never match a run under another.
    assert critique.item_fingerprint(item, critique.RETRY_PROMPT_SUFFIX) != base
    assert critique.item_fingerprint(item, max_tokens=critique.RETRY_MAX_TOKENS) != base


def test_fingerprint_tracks_the_judged_input(tmp_path):
    item = _item(tmp_path, b"image-v1")
    base = critique.item_fingerprint(item)
    assert critique.item_fingerprint({**item, "scene": "other"}) != base
    assert critique.item_fingerprint({**item, "stats": "0: 99"}) != base
    assert critique.item_fingerprint({**item, "legend": "binary"}) != base
    (tmp_path / "frame.png").write_bytes(b"image-v2-changed-mask")
    assert critique.item_fingerprint(item) != base


def test_fingerprint_tracks_the_prompt_template(tmp_path):
    item = _item(tmp_path, b"image-v1")
    base = critique.item_fingerprint(item)
    original = critique.PROMPT_QUAD
    try:
        critique.PROMPT_QUAD = original + "\nEdited critic instructions."
        assert critique.item_fingerprint(item) != base
    finally:
        critique.PROMPT_QUAD = original
    assert critique.item_fingerprint(item) == base


def test_resume_reuses_a_verdict_judged_under_the_same_prompt(tmp_path):
    item = _item(tmp_path, b"image-v1")
    rec = {
        "item_id": "X1",
        "findings": _fd("PASS", []),
        "input_fingerprint": critique.item_fingerprint(item),
    }
    done, carried = critique.resume_state([rec], [item])
    assert done == {"X1": rec}
    assert carried == [rec]


def _legacy_verdict_fingerprint(item) -> str:
    """The pre-repair verdict fingerprint: the judged input without the
    retry parameters. A record that carries it must still not be reused
    under a different prompt."""
    digest = hashlib.sha256()
    digest.update(Path(item["image"]).read_bytes())
    digest.update(b"\0")
    digest.update(
        json.dumps(
            {
                "item_id": item.get("item_id"),
                "legend": item.get("legend"),
                "model": critique.MODEL,
                "prompt_template": critique.PROMPT_QUAD
                if item.get("legend") == "quad"
                else critique.PROMPT_BINARY,
                "scene": item.get("scene"),
                "stats": item.get("stats"),
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def test_resume_does_not_reuse_a_verdict_judged_under_another_prompt(tmp_path):
    # The documented retry changes the prompt and the token budget, so the
    # prior verdict answers a different question and must be re-made.
    item = _item(tmp_path, b"image-v1")
    rec = {
        "item_id": "X1",
        "findings": _fd("PASS", []),
        "input_fingerprint": critique.item_fingerprint(item),
        "verdict_fingerprint": _legacy_verdict_fingerprint(item),
    }
    done, carried = critique.resume_state(
        [rec], [item], critique.RETRY_PROMPT_SUFFIX, critique.RETRY_MAX_TOKENS
    )
    assert done == {}
    assert carried == []


def test_resume_does_not_reuse_a_verdict_judged_with_another_token_budget(tmp_path):
    item = _item(tmp_path, b"image-v1")
    rec = {
        "item_id": "X1",
        "findings": _fd("PASS", []),
        "input_fingerprint": critique.item_fingerprint(item),
        "verdict_fingerprint": _legacy_verdict_fingerprint(item),
    }
    done, carried = critique.resume_state([rec], [item], max_tokens=critique.RETRY_MAX_TOKENS)
    assert done == {}
    assert carried == []


def test_retry_parameters_do_not_reuse_a_changed_image(tmp_path):
    item = _item(tmp_path, b"image-v1")
    rec = {
        "item_id": "X1",
        "findings": _fd("PASS", []),
        "input_fingerprint": critique.item_fingerprint(item),
    }
    (tmp_path / "frame.png").write_bytes(b"image-v2-changed-mask")
    done, carried = critique.resume_state(
        [rec], [item], critique.RETRY_PROMPT_SUFFIX, critique.RETRY_MAX_TOKENS
    )
    assert done == {}
    assert carried == []


def _valid_answer(item_id: str) -> str:
    return json.dumps(
        {
            "item_id": item_id,
            "target_present": True,
            "target_fully_covered": True,
            "wrong_subject": False,
            "kept_subject_touched": False,
            "fit": "good",
            "findings": [],
            "affected_assessment": "plausible",
            "verdict": "PASS",
            "one_line": "clean",
        }
    )


def _run_critique_main(tmp_path, monkeypatch, argv, payloads):
    """Run critique.main() against a fake transport (no network, no model)."""
    state = {item_id: 0 for item_id in payloads}

    def fake_call(key, messages, max_tokens=8000, max_attempts=2):
        text = messages[0]["content"][0]["text"]
        item_id = next(name for name in payloads if f"Item: {name}." in text)
        content = payloads[item_id][state[item_id]]
        state[item_id] += 1
        return 200, {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 7}}

    monkeypatch.setattr(critique, "call", fake_call)
    monkeypatch.setattr(critique, "load_key", lambda path, allowed_dirs: "test-key")
    monkeypatch.setattr(sys, "argv", ["critique.py"] + argv)
    critique.main()
    return state


def _write_items(tmp_path, items):
    for item in items:
        image = tmp_path / f"{item['item_id']}.png"
        image.write_bytes(b"image-v1")
        item["image"] = str(image)
    path = tmp_path / "items.json"
    path.write_text(json.dumps(items))
    return path


def test_documented_retry_rejudges_under_the_new_prompt_and_keeps_its_call_budget(
    tmp_path, monkeypatch
):
    items = [
        {"item_id": "A", "legend": "quad", "scene": "scene", "stats": "0: 1", "set": "s1"},
        {"item_id": "B", "legend": "quad", "scene": "scene", "stats": "0: 1", "set": "s1"},
    ]
    items_path = _write_items(tmp_path, items)
    out = tmp_path / "results.json"
    payloads = {"A": [_valid_answer("A")], "B": ["no json in this answer"]}

    _run_critique_main(
        tmp_path,
        monkeypatch,
        ["--items", str(items_path), "--out", str(out), "--max-calls", "10"],
        payloads,
    )
    first = json.loads(out.read_text())
    assert first["calls_used"] == 2
    assert first["run_calls_used"] == 2
    assert first["records"][0]["findings"] is not None
    assert first["records"][1]["findings"] is None

    # The documented retry: new token budget and the brevity suffix. The
    # transport state is per run, so B now answers with a parseable object.
    payloads["B"] = [_valid_answer("B")]
    state = _run_critique_main(
        tmp_path,
        monkeypatch,
        [
            "--items",
            str(items_path),
            "--out",
            str(out),
            "--max-calls",
            "1",
            "--max-tokens",
            str(critique.RETRY_MAX_TOKENS),
            "--prompt-suffix",
            critique.RETRY_PROMPT_SUFFIX,
        ],
        payloads,
    )
    second = json.loads(out.read_text())
    # The verdict judged under the first prompt is not reused: the retry
    # judges A again under its own prompt, so A costs one call.
    assert state == {"A": 1, "B": 0}
    # The per-invocation call budget keeps the retry bounded: this
    # invocation paid one call and stopped before the second item.
    assert second["run_calls_used"] == 1
    # The prior cumulative count is kept, and the call cap of this run was
    # not pre-empted by it (the previous run already used two calls).
    assert second["calls_used"] == 3
    by_id = {rec["item_id"]: rec for rec in second["records"]}
    assert by_id["A"]["findings"] is not None
    # The stored fingerprint is the retry run's judged input, not the
    # fingerprint of the first prompt.
    assert by_id["A"]["input_fingerprint"] == critique.item_fingerprint(
        items[0], critique.RETRY_PROMPT_SUFFIX, critique.RETRY_MAX_TOKENS
    )
    assert by_id["A"]["input_fingerprint"] != first["records"][0]["input_fingerprint"]


# ---------------- F37: the fingerprint is the bytes that were sent ----------


def test_recorded_fingerprint_is_computed_from_the_bytes_sent_to_the_model(tmp_path, monkeypatch):
    item = {"item_id": "A", "legend": "quad", "scene": "scene", "stats": "0: 1", "set": "s1"}
    items_path = _write_items(tmp_path, [item])
    out = tmp_path / "results.json"
    reads = []
    sent = {}

    def fake_read(path):
        reads.append(path)
        return b"first-read" if len(reads) == 1 else b"second-read"

    def fake_call(key, messages, max_tokens=8000, max_attempts=2):
        url = messages[0]["content"][1]["image_url"]["url"]
        sent["bytes"] = base64.b64decode(url.split(",", 1)[1])
        return 200, {"choices": [{"message": {"content": _valid_answer("A")}}], "usage": {}}

    monkeypatch.setattr(critique, "read_image_bytes", fake_read)
    monkeypatch.setattr(critique, "call", fake_call)
    monkeypatch.setattr(critique, "load_key", lambda path, allowed_dirs: "test-key")
    monkeypatch.setattr(sys, "argv", ["critique.py", "--items", str(items_path), "--out", str(out)])
    critique.main()

    record = json.loads(out.read_text())["records"][0]
    # The image bytes are read once, and the bytes sent to the model are
    # the bytes the fingerprint was computed from.
    assert len(reads) == 1
    assert sent["bytes"] == b"first-read"
    assert record["input_fingerprint"] == critique.item_fingerprint(item, "", 8000, b"first-read")
    # The fingerprint is not a later re-read of the file, and not the
    # bytes a second read would have produced.
    assert record["input_fingerprint"] != critique.item_fingerprint(item, "", 8000, b"second-read")
    assert record["input_fingerprint"] != critique.item_fingerprint(item, "", 8000, b"image-v1")


def test_load_key_accepts_only_named_minimax_credentials(tmp_path):
    env_file = tmp_path / "minimax.env"
    env_file.write_text(
        "AWS_KEY=not-this\nDB_TOKEN=not-this\nMINIMAX_KEY=minimax-value\n",
        encoding="utf-8",
    )
    assert critique.load_key(str(env_file), [str(tmp_path)]) == "minimax-value"


def test_load_key_requires_an_env_directory_allowlist(tmp_path, monkeypatch):
    env = tmp_path / "minimax.env"
    env.write_text("MINIMAX_KEY=sekret\n")

    def fail_open(*_args, **_kwargs):
        raise AssertionError("the env file must not be opened without an allowlist")

    monkeypatch.setattr(critique.Path, "open", fail_open)
    with pytest.raises(SystemExit, match="no allowed env-file directory"):
        critique.load_key(str(env))


def test_load_key_refuses_a_path_outside_the_allowlist_before_reading(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.env"
    outside.write_text("MINIMAX_KEY=sekret\n")

    def fail_open(*_args, **_kwargs):
        raise AssertionError("an out-of-allowlist env file must not be opened")

    monkeypatch.setattr(critique.Path, "open", fail_open)
    with pytest.raises(SystemExit, match="outside the allowed directories"):
        critique.load_key(str(outside), [str(allowed)])
    assert outside.exists()


def test_load_key_allows_a_literal_dollar_in_the_resolved_path(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    env_file = allowed / "my$key.env"
    env_file.write_text("MINIMAX_KEY=sekret\n")
    assert critique.load_key(str(env_file), [str(allowed)]) == "sekret"


def test_call_retries_one_transient_transport_failure(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"choices": []}).encode()

    state = {"calls": 0}

    def fake_urlopen(_request, timeout):
        state["calls"] += 1
        if state["calls"] == 1:
            raise urllib.error.URLError("transient")
        return Response()

    monkeypatch.setattr(critique.urllib.request, "urlopen", fake_urlopen)
    status, response = critique.call("key", [])
    assert status == 200
    assert state["calls"] == 2
    assert response["attempts"] == 2


def test_call_returns_an_http_200_non_json_body_as_an_error_record(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"<html>not json</html>"

    monkeypatch.setattr(critique.urllib.request, "urlopen", lambda request, timeout: Response())
    status, response = critique.call("key", [])
    assert status == 200
    assert response["attempts"] == 1
    assert "invalid_json" in response["transport_error"]
    assert "not json" in response["error_body"]


def test_call_returns_valid_non_object_json_as_an_error_record(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"[1, 2, 3]"

    monkeypatch.setattr(critique.urllib.request, "urlopen", lambda request, timeout: Response())
    status, response = critique.call("key", [])
    assert status == 200
    assert response["transport_error"] == "response is not a JSON object"
    assert response["attempts"] == 1


def test_main_clamps_the_transport_retry_to_the_remaining_call_budget(tmp_path, monkeypatch):
    item = {"item_id": "A", "legend": "quad", "scene": "scene", "stats": "0: 1", "set": "s1"}
    items_path = _write_items(tmp_path, [item])
    out = tmp_path / "results.json"
    attempts = []

    def failing_call(key, messages, max_tokens=8000, max_attempts=2):
        attempts.append(max_attempts)
        return 0, {
            "transport_error": "URLError",
            "error_body": "transient",
            "attempts": max_attempts,
        }

    monkeypatch.setattr(critique, "call", failing_call)
    monkeypatch.setattr(critique, "load_key", lambda path, allowed_dirs: "test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["critique.py", "--items", str(items_path), "--out", str(out), "--max-calls", "1"],
    )
    critique.main()
    assert attempts == [1]
    result = json.loads(out.read_text())
    assert result["run_calls_used"] == 1
    assert result["calls_used"] == 1
    assert result["records"][0]["transport_attempts"] == 1
    assert result["records"][0]["parse_error"]


def test_parse_findings_selects_the_schema_complete_object():
    response = 'reasoning {"nested": true} ' + _valid_answer("A")
    parsed = critique.parse_findings(response)
    assert parsed["item_id"] == "A"
    assert parsed["target_present"] is True


def test_parse_findings_requires_target_present():
    payload = json.loads(_valid_answer("A"))
    del payload["target_present"]
    with pytest.raises(ValueError, match="missing key target_present"):
        critique.parse_findings(json.dumps(payload))


def test_results_record_response_digest_instead_of_raw_model_text(tmp_path, monkeypatch):
    item = {"item_id": "A", "legend": "quad", "scene": "scene", "stats": "0: 1", "set": "s1"}
    items_path = _write_items(tmp_path, [item])
    out = tmp_path / "results.json"
    _run_critique_main(
        tmp_path,
        monkeypatch,
        ["--items", str(items_path), "--out", str(out), "--max-calls", "1"],
        {"A": [_valid_answer("A")]},
    )
    record = json.loads(out.read_text())["records"][0]
    assert "raw_tail" not in record
    assert record["response_digest"] == hashlib.sha256(_valid_answer("A").encode()).hexdigest()
    assert record["response_chars"] == len(_valid_answer("A"))


# ---------------- F34: the item set is validated before any paid call -------


def test_validate_items_rejects_a_malformed_item_set(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"image-v1")
    good = {
        "item_id": "A",
        "image": str(image),
        "legend": "quad",
        "scene": "scene",
        "stats": "0: 1",
        "set": "s1",
    }
    harness.validate_items([good])
    critique.validate_items([good])
    for broken in (
        {k: v for k, v in good.items() if k != "scene"},
        {**good, "legend": "triple"},
        {**good, "image": str(tmp_path / "absent.png")},
    ):
        with pytest.raises(SystemExit):
            harness.validate_items([broken])
        with pytest.raises(SystemExit):
            critique.validate_items([broken])
    with pytest.raises(SystemExit):
        harness.validate_items([good, dict(good)])
    with pytest.raises(SystemExit):
        critique.validate_items([good, dict(good)])


def test_validate_items_rejects_an_unknown_only_id_or_set(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"image-v1")
    good = {
        "item_id": "A",
        "image": str(image),
        "legend": "quad",
        "scene": "scene",
        "stats": "0: 1",
        "set": "s1",
    }
    with pytest.raises(SystemExit, match="not in the items file"):
        harness.validate_items([good], "MISSING")
    with pytest.raises(SystemExit, match="not in the items file"):
        critique.validate_items([good], "MISSING")


def test_harness_main_validates_the_item_set_before_any_call(tmp_path, monkeypatch):
    env = tmp_path / "key.env"
    env.write_text("MINIMAX_KEY=sekret\n")
    image = tmp_path / "frame.png"
    image.write_bytes(b"image-v1")
    config = tmp_path / "providers.json"
    config.write_text(
        json.dumps(
            {
                "providers": {
                    "minimax": {
                        "provider": "minimax",
                        "model_id": "MiniMax-M3.1-Flash",
                        "base_url": "https://example.invalid/v1",
                        "endpoint_style": "openai-compat",
                        "chat_path": "/chat/completions",
                        "env_file": "$SPLIT_CRITIQUE_KEY_FILE",
                        "auth": "bearer",
                        "max_tokens": 8000,
                    }
                }
            }
        )
    )
    monkeypatch.setenv("SPLIT_CRITIQUE_KEY_FILE", str(env))
    items = [
        {
            "item_id": "A",
            "image": str(image),
            "legend": "quad",
            "scene": "scene",
            "stats": "0: 1",
            "set": "s1",
        }
    ]
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps(items))
    called: list[int] = []
    monkeypatch.setattr(harness, "call_openai", lambda *a, **k: called.append(1))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "critique_harness.py",
            "--config",
            str(config),
            "--provider",
            "minimax",
            "--items",
            str(items_path),
            "--out",
            str(tmp_path / "out.json"),
            "--only",
            "MISSING",
        ],
    )
    with pytest.raises(SystemExit):
        harness.main()
    assert called == []


def test_harness_main_validates_the_provider_schema_before_any_call(tmp_path, monkeypatch):
    env = tmp_path / "key.env"
    env.write_text("MINIMAX_KEY=sekret\n")
    config = tmp_path / "providers.json"
    config.write_text(
        json.dumps(
            {
                "providers": {
                    "minimax": {
                        "provider": "minimax",
                        "model_id": "MiniMax-M3.1-Flash",
                        "base_url": "https://example.invalid/v1",
                        "endpoint_style": "openai-compat",
                        "chat_path": "/chat/completions",
                        "env_file": str(env),
                        "auth": "x-api-key",
                        "max_tokens": 8000,
                    }
                }
            }
        )
    )
    image = tmp_path / "frame.png"
    image.write_bytes(b"image-v1")
    items_path = tmp_path / "items.json"
    items_path.write_text(
        json.dumps(
            [
                {
                    "item_id": "A",
                    "image": str(image),
                    "legend": "quad",
                    "scene": "scene",
                    "stats": "0: 1",
                    "set": "s1",
                }
            ]
        )
    )
    called: list[int] = []
    monkeypatch.setattr(harness, "call_openai", lambda *a, **k: called.append(1))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "critique_harness.py",
            "--config",
            str(config),
            "--provider",
            "minimax",
            "--items",
            str(items_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(SystemExit, match="auth"):
        harness.main()
    assert called == []


def test_example_provider_config_matches_the_enforced_schema():
    config = json.loads((ROOT / "configs" / "critique-providers.example.json").read_text())
    for name, provider in config["providers"].items():
        harness.validate_provider(name, provider)
    assert "expands" in config["_note"]
    for key in harness.REQUIRED_PROVIDER_KEYS:
        assert key in config["schema"]


# ---------------- F35: the results file is written atomically ----------------


@pytest.mark.parametrize("module", [harness, critique])
def test_write_json_atomic_keeps_the_previous_file_on_a_crash(tmp_path, monkeypatch, module):
    target = tmp_path / "results.json"
    module.write_json_atomic(target, {"calls_used": 1})

    def boom(payload, *args, **kwargs):
        raise RuntimeError("crash mid-write")

    monkeypatch.setattr(module.json, "dump", boom)
    with pytest.raises(RuntimeError):
        module.write_json_atomic(target, {"calls_used": 2})
    assert json.loads(target.read_text())["calls_used"] == 1
    assert [path.name for path in tmp_path.iterdir()] == ["results.json"]


@pytest.mark.parametrize("module", [harness, critique])
def test_write_json_atomic_replaces_the_file(tmp_path, module):
    target = tmp_path / "results.json"
    module.write_json_atomic(target, {"calls_used": 1})
    module.write_json_atomic(target, {"calls_used": 2})
    assert json.loads(target.read_text())["calls_used"] == 2
    assert [path.name for path in tmp_path.iterdir()] == ["results.json"]


# ---------------- F38: pathlib joins and with-blocked file handles ----------


def _script_tree(relative_path: str) -> ast.AST:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)


def _url_constant_names(tree: ast.AST) -> set[str]:
    """Names bound at module level to a URL string literal."""
    names = set()
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        if "://" not in node.value.value:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _path_string_additions(tree: ast.AST) -> list[tuple[int, str]]:
    """Every '+' that builds a filesystem path by string concatenation.

    A URL base joined with a URL path is not a filesystem path, so the
    module-level URL constants are excluded.
    """
    url_names = _url_constant_names(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            sides = ((node.left, node.right), (node.right, node.left))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            sides = ((node.value, None),)
        else:
            continue
        for operand, other in sides:
            if not isinstance(operand, ast.Constant) or not isinstance(operand.value, str):
                continue
            literal = operand.value
            # A URL path, a data-URL prefix, and a scheme are not
            # filesystem paths.
            if "/" not in literal or ":" in literal:
                continue
            if isinstance(other, ast.Name) and other.id in url_names:
                continue
            found.append((node.lineno, literal))
    return found


def _os_path_calls(tree: ast.AST) -> list[int]:
    """Every os.path.<join-family> call, which pathlib replaces."""
    joined = {"join", "abspath", "dirname", "basename", "isfile", "exists"}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in joined
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
        ):
            found.append(node.lineno)
    return found


def _open_calls_outside_with(tree: ast.AST) -> list[int]:
    """Every open() call that is not the context expression of a with."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                for sub in ast.walk(item.context_expr):
                    inside.add(id(sub))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_open = (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "open"
        )
        if is_open and id(node) not in inside:
            found.append(node.lineno)
    return found


def test_critique_joins_paths_with_pathlib() -> None:
    tree = _script_tree("scripts/critique/critique.py")
    assert _os_path_calls(tree) == []
    assert _path_string_additions(tree) == []
    assert "from pathlib import Path" in (ROOT / "scripts/critique/critique.py").read_text(
        encoding="utf-8"
    )


def test_critique_closes_every_file_handle_in_a_with_block() -> None:
    tree = _script_tree("scripts/critique/critique.py")
    assert _open_calls_outside_with(tree) == []
