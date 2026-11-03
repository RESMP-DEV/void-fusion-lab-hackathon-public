"""Round-2 repair tests for the critique harness (round-2 review findings).

Every test here maps to one numbered outcome of the round-2 work order:

  1  the example config note distinguishes environment placeholders from
     real secret-file paths;
  2  expand_env_path results are constrained to an operator-supplied
     allowlist, and a path outside it is refused before any read;
  3  temperature and the token budgets are validated as positive numbers
     before an API call;
  4  provider auth is required and must match the endpoint style;
  5  item images are validated as readable images, not just existing paths;
  6  atomic writes use a non-predictable temporary name;
  7  prompt assembly does not use percent formatting on user data;
  8  transient URL errors are returned as error records like HTTP errors;
  9  JSON extraction handles multiple objects and stray braces;
 10  the findings parser validates every key the schema block requires,
     target_present included;
 11  a missing expected verdict_class is an explicit recorded input
     error, not an implicit clean verdict;
 12  the ledger file is closed deterministically;
 13  raw model response tails are not persisted in the results file.

No test in this file calls a network service or a model: every transport
is a local fake.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from email.message import Message
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


harness = _load("critique_harness_round2", "scripts/critique_harness.py")

EXAMPLE_CONFIG = ROOT / "configs" / "critique-providers.example.json"

FINDINGS = {
    "item_id": "A",
    "target_present": True,
    "target_fully_covered": True,
    "wrong_subject": False,
    "kept_subject_touched": False,
    "fit": "good",
    "findings": [],
    "affected_assessment": "plausible",
    "verdict": "PASS",
    "one_line": "clean mask",
}


def _item(image: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "item_id": "A",
        "image": image,
        "legend": "quad",
        "scene": "scene",
        "stats": "0: 10",
        "set": "set1",
    }
    item.update(overrides)
    return item


def _valid_findings(**overrides: Any) -> dict[str, Any]:
    payload = json.loads(json.dumps(FINDINGS))
    payload.update(overrides)
    return payload


def _png(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (4, 4), (10, 20, 30)).save(path)
    return path


def _schema_top_level_keys(schema_text: str) -> list[str]:
    """The top-level keys the schema block asks a findings object for."""
    start = schema_text.index("{")
    depth = 0
    keys: list[str] = []
    index = start
    while index < len(schema_text):
        char = schema_text[index]
        if char == '"':
            end = index + 1
            while schema_text[end] != '"':
                end += 2 if schema_text[end] == "\\" else 1
            after = end + 1
            while after < len(schema_text) and schema_text[after] in " \n\t":
                after += 1
            if depth == 1 and after < len(schema_text) and schema_text[after] == ":":
                keys.append(schema_text[index + 1 : end])
            index = end + 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    return keys


# ---------------- 1: the example config note ---------------------------


def test_example_config_note_separates_placeholders_from_real_paths() -> None:
    config = json.loads(EXAMPLE_CONFIG.read_text())
    note = config["_note"]
    assert "PLACEHOLDER" in note
    assert "$MINIMAX_ENV_FILE" in note
    assert "must never be committed" in note
    assert "REAL SECRET-FILE PATH" in note
    # The note names both sides of the distinction, so a reader can tell
    # which value belongs in this tracked file.
    assert "ENVIRONMENT PLACEHOLDERS" in note
    assert "Never commit a key value" in note


def test_example_config_env_files_are_placeholders_not_real_paths() -> None:
    config = json.loads(EXAMPLE_CONFIG.read_text())
    for name, provider in config["providers"].items():
        env_file = provider["env_file"]
        # Every env_file in the tracked example is an environment
        # placeholder, never a path to a file on a machine.
        assert env_file.startswith("$"), name
        assert env_file[1:].strip("{}").isidentifier(), name
    # The note names no real file either.
    assert ".env" not in config["_note"]


def test_example_config_schema_states_the_required_auth_and_positive_budgets() -> None:
    config = json.loads(EXAMPLE_CONFIG.read_text())
    assert "auth" in harness.REQUIRED_PROVIDER_KEYS
    for name, provider in config["providers"].items():
        harness.validate_provider(name, provider)
        style = provider["endpoint_style"]
        assert provider["auth"] == harness.ENDPOINT_STYLES[style]


# ---------------- 2: the env-file allowlist ----------------------------


def test_env_path_outside_the_allowlist_is_refused_before_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.env"
    outside.write_text("MINIMAX_KEY=k\n")
    opened: list[str] = []
    real_open = open

    def watched_open(file, *args, **kwargs):  # noqa: ANN001, ANN202
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", watched_open)
    with pytest.raises(SystemExit, match="outside the allowed directories"):
        harness.load_key(str(outside), [str(allowed)])
    assert opened == []
    assert outside.exists()


def test_env_path_inside_the_allowlist_is_read(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    env_file = allowed / "key.env"
    env_file.write_text("MINIMAX_KEY=k\n")
    assert harness.load_key(str(env_file), [str(allowed)]) == "k"


def test_env_path_allowlist_resolves_a_placeholder(tmp_path: Path, monkeypatch) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    env_file = allowed / "key.env"
    env_file.write_text("MINIMAX_KEY=k\n")
    monkeypatch.setenv("ROUND2_ENV_FILE", str(env_file))
    assert harness.load_key("$ROUND2_ENV_FILE", [str(allowed)]) == "k"
    # The same placeholder outside the allowlist is refused.
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("ROUND2_ENV_FILE", str(other / "key.env"))
    with pytest.raises(SystemExit, match="outside the allowed directories"):
        harness.load_key("$ROUND2_ENV_FILE", [str(allowed)])


def test_env_path_allowlist_refuses_a_parent_traversal(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    escape = tmp_path / "outside.env"
    escape.write_text("MINIMAX_KEY=k\n")
    with pytest.raises(SystemExit, match="outside the allowed directories"):
        harness.load_key(str(allowed / ".." / "outside.env"), [str(allowed)])


def test_main_refuses_to_run_without_an_allowlist(tmp_path: Path, monkeypatch) -> None:
    config, items_path, image = _harness_inputs(tmp_path)
    monkeypatch.delenv(harness.ALLOWED_ENV_DIRS_ENV, raising=False)
    called: list[int] = []
    monkeypatch.setattr(harness, "call_openai", lambda *a, **k: called.append(1))
    monkeypatch.setattr(
        sys, "argv", _argv(config, items_path, tmp_path / "out.json", allowed=False)
    )
    with pytest.raises(SystemExit, match="allowed env-file directory"):
        harness.main()
    assert called == []


def test_main_accepts_the_allowlist_from_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    monkeypatch.setenv(harness.ALLOWED_ENV_DIRS_ENV, str(tmp_path))
    monkeypatch.setattr(harness, "call_openai", _fake_call(json.dumps(FINDINGS)))
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json"))
    harness.main()
    assert (tmp_path / "out.json").exists()


# ---------------- 3 and 4: the provider schema -------------------------


def _provider(**overrides: Any) -> dict[str, Any]:
    provider: dict[str, Any] = {
        "provider": "minimax",
        "model_id": "MiniMax-M3.1-Flash",
        "base_url": "https://example.invalid/v1",
        "endpoint_style": "openai-compat",
        "chat_path": "/chat/completions",
        "env_file": "$ROUND2_ENV_FILE",
        "auth": "bearer",
        "temperature": 0.1,
        "max_tokens": 8000,
        "retry_max_tokens": 12000,
    }
    provider.update(overrides)
    return provider


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", 0),
        ("temperature", -0.5),
        ("temperature", "hot"),
        ("temperature", True),
        ("max_tokens", 0),
        ("max_tokens", -5),
        ("max_tokens", 8000.5),
        ("retry_max_tokens", 0),
        ("retry_max_tokens", -1),
    ],
)
def test_provider_budgets_must_be_positive_numbers(key: str, value: Any) -> None:
    with pytest.raises(SystemExit, match=key):
        harness.validate_provider("minimax", _provider(**{key: value}))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", 0.1),
        ("temperature", 1),
        ("max_tokens", 8000),
        ("retry_max_tokens", 12000),
    ],
)
def test_provider_accepts_positive_budgets(key: str, value: Any) -> None:
    harness.validate_provider("minimax", _provider(**{key: value}))


def test_provider_auth_is_required() -> None:
    provider = _provider()
    del provider["auth"]
    with pytest.raises(SystemExit, match="auth"):
        harness.validate_provider("minimax", provider)


def test_provider_auth_must_match_the_endpoint_style() -> None:
    with pytest.raises(SystemExit, match="auth"):
        harness.validate_provider("minimax", _provider(auth="x-api-key"))


def test_provider_auth_matches_the_anthropic_style() -> None:
    harness.validate_provider(
        "minimax-anthropic",
        _provider(
            endpoint_style="anthropic-compat", auth="x-api-key", anthropic_version="2023-06-01"
        ),
    )


# ---------------- 5: the item images -----------------------------------


def test_validate_item_images_rejects_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    broken = tmp_path / "frame.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(SystemExit, match="not a readable image"):
        harness.validate_item_images([_item(str(broken))])


def test_validate_item_images_rejects_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "frame.png"
    empty.write_bytes(b"")
    with pytest.raises(SystemExit, match="not a readable image"):
        harness.validate_item_images([_item(str(empty))])


def test_validate_item_images_accepts_a_real_image(tmp_path: Path) -> None:
    image = _png(tmp_path / "frame.png")
    harness.validate_item_images([_item(str(image))])


def test_validate_items_rejects_an_empty_image_file(tmp_path: Path) -> None:
    empty = tmp_path / "frame.png"
    empty.write_bytes(b"")
    with pytest.raises(SystemExit, match="empty"):
        harness.validate_items([_item(str(empty))])


def test_main_refuses_a_non_image_item_before_any_call(tmp_path: Path, monkeypatch) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    items = json.loads(items_path.read_text())
    items[0]["image"] = str(broken)
    items_path.write_text(json.dumps(items))
    called: list[int] = []
    monkeypatch.setattr(harness, "call_openai", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json"))
    with pytest.raises(SystemExit, match="not a readable image"):
        harness.main()
    assert called == []


# ---------------- 6: the atomic write ----------------------------------


def test_write_json_atomic_uses_a_non_predictable_temporary_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "results.json"
    created: list[tuple[str, int]] = []
    real_open = os.open

    def watched_open(path, flags, mode=0o777):  # noqa: ANN001, ANN202
        created.append((str(path), mode))
        return real_open(path, flags, mode)

    monkeypatch.setattr(harness.os, "open", watched_open)
    harness.write_json_atomic(target, {"calls_used": 1})
    names = [Path(path).name for path, _mode in created]
    assert len(names) == 1
    name = names[0]
    assert name.startswith(".critique-")
    assert name.endswith(".tmp")
    # The random part is not a fixed literal: two writes must not reuse
    # the same temporary name.
    monkeypatch.setattr(harness.os, "open", watched_open)
    created.clear()
    harness.write_json_atomic(target, {"calls_used": 2})
    assert created[0][0] != Path(created[0][0]).name or True
    second = Path(created[0][0]).name
    assert second != name
    assert json.loads(target.read_text())["calls_used"] == 2


def test_write_json_atomic_does_not_use_a_predictable_decoy(tmp_path: Path) -> None:
    target = tmp_path / "results.json"
    decoy = tmp_path / ".critique-results.json.tmp"
    decoy.write_text("decoy")
    harness.write_json_atomic(target, {"calls_used": 1})
    assert decoy.read_text() == "decoy"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".critique-results.json.tmp",
        "results.json",
    ]


def test_write_json_atomic_creates_the_temporary_file_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "results.json"
    modes: list[int] = []
    real_open = os.open

    def watched_open(path, flags, mode=0o777):  # noqa: ANN001, ANN202
        modes.append(mode)
        return real_open(path, flags, mode)

    monkeypatch.setattr(harness.os, "open", watched_open)
    harness.write_json_atomic(target, {"calls_used": 1})
    assert modes == [0o600]


# ---------------- 7: the prompt assembly -------------------------------


def test_prompt_keeps_percent_signs_and_braces_in_the_item_data() -> None:
    item = _item(
        "/tmp/frame.png",
        scene="100% covered {odd} %s %d",
        stats="0: 50%",
    )
    prompt = harness.build_prompt(item)
    assert "100% covered {odd} %s %d" in prompt
    assert "0: 50%" in prompt
    assert "Item: A." in prompt


def test_prompt_assembly_does_not_percent_format_user_data() -> None:
    source = (ROOT / "scripts" / "critique_harness.py").read_text()
    tree = ast.parse(source, filename="scripts/critique_harness.py")
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_prompt"
    )
    percent_formatting = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
    ]
    assert percent_formatting == []


def test_prompt_escapes_the_item_id_as_json() -> None:
    item = _item("/tmp/frame.png", item_id='a"b')
    prompt = harness.build_prompt(item)
    assert '"a\\"b"' in prompt


# ---------------- 8: the transport failures ----------------------------


class _FakeResponse:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_transient_url_error_is_returned_as_an_error_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(req, timeout=None):  # noqa: ANN001, ANN202
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    status, body = harness.request_json(
        urllib.request.Request("https://example.invalid/v1/chat/completions")
    )
    assert status == 0
    assert body["error_kind"] == "transport"
    assert "ConnectionRefusedError" in body["error_body"]


def test_timeout_is_returned_as_an_error_record(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req, timeout=None):  # noqa: ANN001, ANN202
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    status, body = harness.request_json(
        urllib.request.Request("https://example.invalid/v1/chat/completions")
    )
    assert status == 0
    assert body["error_kind"] == "transport"
    assert "TimeoutError" in body["error_body"]


def test_http_error_is_still_an_error_record(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req, timeout=None):  # noqa: ANN001, ANN202
        raise urllib.error.HTTPError("https://example.invalid/v1", 429, "too many", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    status, body = harness.request_json(
        urllib.request.Request("https://example.invalid/v1/chat/completions")
    )
    assert status == 429
    assert body["error_kind"] == "http"


def test_non_json_body_is_returned_as_an_error_record(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req, timeout=None):  # noqa: ANN001, ANN202
        return _FakeResponse(200, b"<html>not json</html>")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    status, body = harness.request_json(
        urllib.request.Request("https://example.invalid/v1/chat/completions")
    )
    assert status == 0
    assert body["error_kind"] == "decode"


def test_call_openai_returns_an_error_record_for_a_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(req, timeout=None):  # noqa: ANN001, ANN202
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    status, body = harness.call_openai(_provider(), "k", "prompt", "aGk=", 8000)
    assert status == 0
    assert body["error_kind"] == "transport"


# ---------------- 9 and 10: the JSON extraction and the parser ----------


def test_extractor_picks_the_findings_object_after_another_object() -> None:
    text = '{"note": "thinking about the frame"} then ' + json.dumps(FINDINGS)
    parsed = harness.extract_findings_object(text)
    assert parsed["item_id"] == "A"
    assert parsed["verdict"] == "PASS"


def test_extractor_handles_stray_braces() -> None:
    text = "analysis { unbalanced { brace\n" + json.dumps(FINDINGS)
    parsed = harness.extract_findings_object(text)
    assert parsed["item_id"] == "A"


def test_extractor_raises_when_no_object_carries_the_schema() -> None:
    with pytest.raises(ValueError, match="no findings object"):
        harness.extract_findings_object('{"note": "only prose"}')


def test_extractor_raises_when_there_is_no_json_at_all() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        harness.extract_findings_object("no json here")


def test_parser_accepts_a_response_with_prose_around_the_object() -> None:
    text = "Sure. " + json.dumps(FINDINGS) + "\nThat is my verdict."
    assert harness.parse_findings(text)["verdict"] == "PASS"


@pytest.mark.parametrize("key", list(harness.REQUIRED_FINDINGS_KEYS))
def test_parser_requires_every_schema_key(key: str) -> None:
    payload = _valid_findings()
    del payload[key]
    with pytest.raises(ValueError, match="missing key"):
        harness.parse_findings(json.dumps(payload))


def test_parser_requires_target_present() -> None:
    payload = _valid_findings()
    del payload["target_present"]
    with pytest.raises(ValueError, match="missing key target_present"):
        harness.parse_findings(json.dumps(payload))


def test_parser_rejects_a_non_boolean_flag() -> None:
    payload = _valid_findings(target_present="yes")
    with pytest.raises(ValueError, match="not a boolean"):
        harness.parse_findings(json.dumps(payload))


def test_parser_rejects_a_bad_verdict() -> None:
    payload = _valid_findings(verdict="MAYBE")
    with pytest.raises(ValueError, match="bad verdict"):
        harness.parse_findings(json.dumps(payload))


def test_required_keys_match_the_schema_block() -> None:
    assert _schema_top_level_keys(harness.SCHEMA_BLOCK) == list(harness.REQUIRED_FINDINGS_KEYS)


# ---------------- 11: the expected-answer input error ------------------


def test_score_records_an_input_error_for_a_missing_verdict_class() -> None:
    rec = {"findings": _valid_findings()}
    result = harness.score(rec, {"planted_type": "hole"})
    assert result["agreement"] == harness.INPUT_ERROR
    assert result["scored"] is False
    assert "verdict_class" in result["error"]


def test_score_does_not_treat_a_missing_verdict_class_as_a_clean_verdict() -> None:
    rec = {"findings": _valid_findings()}
    assert harness.score(rec, {"planted_type": "hole"})["agreement"] != "MATCH"
    assert harness.score(rec, {"planted_type": "hole"})["agreement"] != "FALSE_ALARM"


def test_score_keeps_a_present_but_unknown_class_on_the_clean_branch() -> None:
    rec = {"findings": _valid_findings()}
    assert harness.score(rec, {"verdict_class": None})["agreement"] == "MATCH"
    assert harness.score(rec, {"verdict_class": "unknown"})["agreement"] == "MATCH"


def test_score_records_the_input_error_in_the_results_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    items = json.loads(items_path.read_text())
    items[0]["expected"] = {"planted_type": "hole"}
    items_path.write_text(json.dumps(items))
    monkeypatch.setattr(harness, "call_openai", _fake_call(json.dumps(FINDINGS)))
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json"))
    harness.main()
    results = json.loads((tmp_path / "out.json").read_text())
    record = results["records"][0]
    assert record["score"]["agreement"] == harness.INPUT_ERROR
    assert record["score"]["scored"] is False


# ---------------- 12: the ledger ---------------------------------------


def test_the_ledger_is_closed_after_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    ledger_path = tmp_path / "ledger.jsonl"
    handles: list[Any] = []
    real_open = open

    def watched_open(file, *args, **kwargs):  # noqa: ANN001, ANN202
        handle = real_open(file, *args, **kwargs)
        if str(file) == str(ledger_path):
            handles.append(handle)
        return handle

    monkeypatch.setattr("builtins.open", watched_open)
    monkeypatch.setattr(harness, "call_openai", _fake_call(json.dumps(FINDINGS)))
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json", ledger_path))
    harness.main()
    assert handles
    assert all(handle.closed for handle in handles)


def test_the_ledger_is_closed_even_when_the_run_fails(tmp_path: Path, monkeypatch) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    ledger_path = tmp_path / "ledger.jsonl"
    handles: list[Any] = []
    real_open = open

    def watched_open(file, *args, **kwargs):  # noqa: ANN001, ANN202
        handle = real_open(file, *args, **kwargs)
        if str(file) == str(ledger_path):
            handles.append(handle)
        return handle

    monkeypatch.setattr("builtins.open", watched_open)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model call exploded")

    monkeypatch.setattr(harness, "call_openai", boom)
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json", ledger_path))
    with pytest.raises(RuntimeError):
        harness.main()
    assert handles
    assert all(handle.closed for handle in handles)


def test_the_ledger_holds_one_json_line_per_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(harness, "call_openai", _fake_call(json.dumps(FINDINGS)))
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json", ledger_path))
    harness.main()
    lines = [line for line in ledger_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["item_id"] == "A"
    assert entry["http"] == 200


# ---------------- 13: no raw model text in the results file -------------


def test_the_results_file_holds_no_model_text(tmp_path: Path, monkeypatch) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    marker = "PRIVATE-CONTENT-MARKER"
    # The model answers with reasoning around the JSON object. The raw
    # text must not reach the results file; the parsed findings object is
    # the deliverable and is kept.
    response = "reasoning: " + marker + "\n" + json.dumps(_valid_findings())
    monkeypatch.setattr(harness, "call_openai", _fake_call(response))
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json"))
    harness.main()
    results_text = (tmp_path / "out.json").read_text()
    assert marker not in results_text
    record = json.loads(results_text)["records"][0]
    assert record["findings"]["verdict"] == "PASS"
    assert "raw_tail" not in record["attempts"][0]
    assert record["attempts"][0]["response_digest"].startswith("sha256:")
    assert record["attempts"][0]["response_chars"] > 0


def test_a_parse_failure_records_no_model_text(tmp_path: Path, monkeypatch) -> None:
    config, items_path, _image = _harness_inputs(tmp_path)
    marker = "PRIVATE-CONTENT-MARKER"
    monkeypatch.setattr(harness, "call_openai", _fake_call("no json here: " + marker))
    monkeypatch.setattr(harness, "call_openai", _fake_call("no json here: " + marker))
    monkeypatch.setattr(sys, "argv", _argv(config, items_path, tmp_path / "out.json"))
    harness.main()
    results_text = (tmp_path / "out.json").read_text()
    assert marker not in results_text
    record = json.loads(results_text)["records"][0]
    assert record["attempts"][0]["parse"].startswith("FAIL:")
    assert record["attempts"][0]["parse"] != "OK"


# ---------------- helpers for the end-to-end runs ----------------------


def _fake_call(payload: str):
    def call(cfg, key, prompt, b64, max_tokens):  # noqa: ANN001, ANN202
        return 200, {
            "choices": [{"message": {"content": payload}}],
            "usage": {"total_tokens": 7},
        }

    return call


def _harness_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A provider config, an items file, and a real image, all under tmp_path."""
    allowed = tmp_path / "secrets"
    allowed.mkdir(exist_ok=True)
    env_file = allowed / "key.env"
    env_file.write_text("MINIMAX_KEY=k\n")
    image = _png(tmp_path / "frame.png")
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
                        "env_file": str(env_file),
                        "auth": "bearer",
                        "temperature": 0.1,
                        "max_tokens": 8000,
                        "retry_max_tokens": 12000,
                    }
                }
            }
        )
    )
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
    return config, items_path, image


def _argv(
    config: Path,
    items_path: Path,
    out: Path,
    ledger: Path | None = None,
    allowed: bool = True,
) -> list[str]:
    argv = [
        "critique_harness.py",
        "--config",
        str(config),
        "--provider",
        "minimax",
        "--items",
        str(items_path),
        "--out",
        str(out),
    ]
    if allowed:
        argv.extend(["--allowed-env-dir", str(config.parent)])
    if ledger is not None:
        argv.extend(["--ledger", str(ledger)])
    return argv
