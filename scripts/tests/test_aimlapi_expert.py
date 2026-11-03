from __future__ import annotations

import base64
import hashlib
import io
import json
import sys

import aimlapi_expert
import pytest
from aimlapi_expert import (
    build_diagnostics,
    build_payload,
    extract_message_text,
    extract_message_text_with_source,
    parse_model_content,
    write_diagnostics,
)
from PIL import Image


def test_build_payload_batches_frames_and_hides_image_bytes(tmp_path):
    image = Image.new("RGB", (20, 10), "blue")
    image_path = tmp_path / "frame.png"
    image.save(image_path)
    request = {
        "prompt": "Find the target",
        "frames": [{"frame_index": 0, "path": str(image_path)}],
    }

    payload = build_payload(
        model="xiaomi/mimo-v2.5",
        request=request,
        max_image_size=8,
        max_output_tokens=123,
    )

    assert payload["model"] == "xiaomi/mimo-v2.5"
    assert payload["max_tokens"] == 123
    assert len(payload["messages"][0]["content"]) == 2
    data_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1])


def test_parse_model_content_accepts_fenced_and_prose_wrapped_array():
    records = [{"frame_index": 0, "regions": []}]
    fenced = f"```json\n{json.dumps(records)}\n```"
    assert parse_model_content(fenced) == {"regions_by_frame": records}
    prose = f"Here it is:\n{json.dumps(records)}\nDone."
    assert parse_model_content(prose) == {"regions_by_frame": records}


SYNTHETIC_ARRAY = '[{"frame_index": 0, "regions": []}]'


def test_extract_message_text_accepts_plain_string_content():
    message = {"role": "assistant", "content": SYNTHETIC_ARRAY}
    assert extract_message_text(message) == SYNTHETIC_ARRAY


def test_extract_message_text_joins_list_text_parts():
    message = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": '[{"frame_index": 0, '},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": '"regions": []}]'},
        ],
    }
    assert extract_message_text(message) == SYNTHETIC_ARRAY


def test_extract_message_text_falls_back_when_content_key_absent():
    # Synthetic fixture of the MiniMax-M3.1-Flash served shape that
    # stopped the MK13 run at aimlapi_expert line 130: the message
    # carries reasoning_content and no content key at all.
    message = {"role": "assistant", "reasoning_content": SYNTHETIC_ARRAY}
    text = extract_message_text(message)
    assert text == SYNTHETIC_ARRAY
    assert parse_model_content(text) == {"regions_by_frame": [{"frame_index": 0, "regions": []}]}


def test_extract_message_text_falls_back_when_content_is_null():
    message = {"role": "assistant", "content": None, "reasoning_content": SYNTHETIC_ARRAY}
    assert extract_message_text(message) == SYNTHETIC_ARRAY


def test_extract_message_text_prefers_content_over_reasoning():
    message = {
        "role": "assistant",
        "content": SYNTHETIC_ARRAY,
        "reasoning_content": "synthetic reasoning text",
    }
    assert extract_message_text(message) == SYNTHETIC_ARRAY


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant"},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": []},
        {"role": "assistant", "content": [{"type": "image_url", "image_url": {}}]},
        {"role": "assistant", "content": None, "reasoning_content": ""},
    ],
)
def test_extract_message_text_fails_closed_without_text(message):
    with pytest.raises(KeyError, match="content"):
        extract_message_text(message)


def _synthetic_response(message, *, finish_reason="stop", usage=None):
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_extract_message_text_with_source_names_each_field():
    text, source = extract_message_text_with_source({"content": SYNTHETIC_ARRAY})
    assert (text, source) == (SYNTHETIC_ARRAY, "content")
    text, source = extract_message_text_with_source(
        {"content": [{"type": "text", "text": SYNTHETIC_ARRAY}]}
    )
    assert (text, source) == (SYNTHETIC_ARRAY, "content_list")
    text, source = extract_message_text_with_source({"reasoning_content": SYNTHETIC_ARRAY})
    assert (text, source) == (SYNTHETIC_ARRAY, "reasoning_content")


def test_diagnostics_record_describes_content_string_shape():
    message = {"role": "assistant", "content": SYNTHETIC_ARRAY, "reasoning_content": "abc"}
    text, source = extract_message_text_with_source(message)
    diag = build_diagnostics(
        _synthetic_response(message),
        model="m",
        endpoint="e",
        extracted_text=text,
        extracted_from=source,
    )
    assert diag["message_keys"] == ["content", "reasoning_content", "role"]
    assert diag["content"] == {"present": True, "type": "string", "length": len(SYNTHETIC_ARRAY)}
    assert diag["reasoning_content"]["length"] == 3
    assert diag["extracted_from"] == "content"
    assert diag["extracted_text_length"] == len(SYNTHETIC_ARRAY)
    assert (
        diag["extracted_text_sha256"] == hashlib.sha256(SYNTHETIC_ARRAY.encode("utf-8")).hexdigest()
    )
    assert diag["finish_reason"] == "stop"
    assert diag["usage"]["total_tokens"] == 15


def test_diagnostics_record_describes_absent_content_reasoning_fallback():
    reasoning = "synthetic reasoning that carries the array"
    message = {"role": "assistant", "reasoning_content": reasoning}
    text, source = extract_message_text_with_source(message)
    diag = build_diagnostics(
        _synthetic_response(message, finish_reason="length"),
        model="m",
        endpoint="e",
        extracted_text=text,
        extracted_from=source,
    )
    assert diag["content"] == {"present": False, "type": "absent", "length": None}
    assert diag["reasoning_content"] == {
        "present": True,
        "type": "string",
        "length": len(reasoning),
    }
    assert diag["extracted_from"] == "reasoning_content"
    assert diag["finish_reason"] == "length"


def test_diagnostics_record_describes_list_content():
    message = {
        "content": [
            {"type": "text", "text": "ab"},
            {"type": "image_url", "image_url": {}},
            {"type": "text", "text": "cde"},
        ]
    }
    text, source = extract_message_text_with_source(message)
    diag = build_diagnostics(
        _synthetic_response(message),
        model="m",
        endpoint="e",
        extracted_text=text,
        extracted_from=source,
    )
    assert diag["content"]["type"] == "list"
    assert diag["content"]["length"] == 5
    assert diag["content"]["parts"] == 3
    assert diag["extracted_from"] == "content_list"


def test_diagnostics_record_when_no_text_extractable():
    diag = build_diagnostics(
        _synthetic_response({"role": "assistant", "content": None}),
        model="m",
        endpoint="e",
        extracted_text=None,
        extracted_from=None,
    )
    assert diag["content"]["type"] == "null"
    assert diag["reasoning_content"]["type"] == "absent"
    assert diag["extracted_from"] is None
    assert diag["extracted_text_length"] is None
    assert diag["extracted_text_sha256"] is None


def test_write_diagnostics_writes_flag_path_and_capture_copy(tmp_path):
    diag = {"adapter": "aimlapi_expert", "extracted_from": "content"}
    flag_path = tmp_path / "nested" / "diag.json"
    capture_dir = tmp_path / "capture"
    write_diagnostics(
        diag,
        explicit_path=str(flag_path),
        capture_dir=capture_dir,
        stem="response-x",
    )
    assert json.loads(flag_path.read_text()) == diag
    assert json.loads((capture_dir / "response-x.diag.json").read_text()) == diag


def _run_main(monkeypatch, tmp_path, response, extra_argv=()):
    image = Image.new("RGB", (20, 10), "blue")
    image_path = tmp_path / "frame.png"
    image.save(image_path)
    request = {
        "prompt": "Find the target",
        "frames": [{"frame_index": 0, "path": str(image_path)}],
    }
    monkeypatch.setattr(
        sys,
        "argv",
        ["aimlapi_expert", "--model", "MiniMax-M3.1-Flash", *extra_argv],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(aimlapi_expert, "call_api", lambda payload, **kwargs: response)
    monkeypatch.setenv("AIMLAPI_KEY", "test-key")
    return aimlapi_expert.main()


def test_main_reasoning_fallback_parse_failure_names_the_shape(monkeypatch, tmp_path, capsys):
    reasoning = "plan: [1, oops] end"
    response = _synthetic_response({"role": "assistant", "reasoning_content": reasoning})
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv("EXPERT_RESPONSE_CAPTURE", str(capture_dir))
    assert _run_main(monkeypatch, tmp_path, response) == 2
    err = capsys.readouterr().err
    assert "reasoning-fallback parse failure" in err
    assert "content absent" in err
    diag_files = list(capture_dir.glob("*.diag.json"))
    raw_files = [
        path for path in capture_dir.glob("response-*.json") if not path.name.endswith(".diag.json")
    ]
    assert len(diag_files) == 1
    assert len(raw_files) == 1
    diag = json.loads(diag_files[0].read_text())
    assert diag["extracted_from"] == "reasoning_content"
    assert diag["content"]["type"] == "absent"
    assert diag["raw_response_file"] == raw_files[0].name


def test_main_content_parse_failure_keeps_plain_error(monkeypatch, tmp_path, capsys):
    response = _synthetic_response({"role": "assistant", "content": "answer: [1, oops] end"})
    assert _run_main(monkeypatch, tmp_path, response) == 2
    err = capsys.readouterr().err
    assert err.startswith("aimlapi_expert:")
    assert "reasoning-fallback" not in err


def test_main_no_extractable_text_keeps_keyerror_surface(monkeypatch, tmp_path, capsys):
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv("EXPERT_RESPONSE_CAPTURE", str(capture_dir))
    response = _synthetic_response({"role": "assistant", "content": None})
    assert _run_main(monkeypatch, tmp_path, response) == 2
    err = capsys.readouterr().err
    assert "'content'" in err
    diag_files = list(capture_dir.glob("*.diag.json"))
    assert len(diag_files) == 1
    diag = json.loads(diag_files[0].read_text())
    assert diag["extracted_from"] is None
    assert diag["extracted_text_sha256"] is None


def test_main_success_writes_diagnostics_to_flag_path(monkeypatch, tmp_path, capsys):
    diag_path = tmp_path / "diag.json"
    response = _synthetic_response({"role": "assistant", "content": SYNTHETIC_ARRAY})
    rc = _run_main(
        monkeypatch,
        tmp_path,
        response,
        extra_argv=("--diagnostics-out", str(diag_path)),
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"regions_by_frame": [{"frame_index": 0, "regions": []}]}
    diag = json.loads(diag_path.read_text())
    assert diag["extracted_from"] == "content"
    assert (
        diag["extracted_text_sha256"] == hashlib.sha256(SYNTHETIC_ARRAY.encode("utf-8")).hexdigest()
    )
