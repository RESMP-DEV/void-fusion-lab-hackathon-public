from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture"

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prompt_critique", ROOT / "scripts/prompt_critique.py"
)
if spec is None or spec.loader is None:
    raise AssertionError("unable to load prompt_critique test module")
prompt_critique = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_critique)


def test_candidate_schema_is_meta_strict_subset() -> None:
    schema = prompt_critique.CANDIDATE_SCHEMA
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_parse_json_object_rejects_missing_object() -> None:
    with pytest.raises(prompt_critique.PromptLoopError, match="JSON"):
        prompt_critique.parse_json_object("no object")


def test_parse_json_object_unwraps_fenced_json() -> None:
    value = prompt_critique.parse_json_object('```json\n{"passed": true}\n```')
    assert value == {"passed": True}


def test_parse_json_object_rejects_extra_objects_or_ambiguous_text() -> None:
    with pytest.raises(prompt_critique.PromptLoopError, match="multiple JSON objects"):
        prompt_critique.parse_json_object('prefix {"a":1} suffix {"b":2}')


def test_read_image_enforces_extension_signature_and_size(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_BYTES)
    data, media_type = prompt_critique.read_image(image)
    assert data == PNG_BYTES
    assert media_type == "image/png"

    text_disguise = tmp_path / "disguise.png"
    text_disguise.write_text("not an image", encoding="utf-8")
    with pytest.raises(prompt_critique.PromptLoopError, match="do not match"):
        prompt_critique.read_image(text_disguise)


def test_sam_guard_rejects_commands_questions_spatial_and_multi_concepts() -> None:
    valid = [
        "object",
        "red car",
        "gray oval shadow",
        "black rectangular podium",
        "logo",
        "tattoo",
        "man",
        "woman",
        "traffic cone",
        "white van",
        "red sports car",
    ]
    invalid = [
        "erase the car",
        "near the car",
        "which car",
        "cat dog",
        "shadow reflection",
        "remove the red object",
    ]
    assert all(prompt_critique.is_bare_noun_phrase(value) for value in valid)
    assert not any(prompt_critique.is_bare_noun_phrase(value) for value in invalid)


def test_candidate_schema_rejects_missing_strict_fields() -> None:
    with pytest.raises(prompt_critique.PromptLoopError, match="exact strict-schema"):
        prompt_critique.validate_candidate_schema({"sam_concept": "red car"})


def test_generation_payload_uses_constrained_decoding_and_labeled_frame() -> None:
    payload = prompt_critique.build_generation_payload(
        image_data="data:image/png;base64,eA==",
        user_request="remove the red object",
        feedback=None,
        model="muse-test",
    )
    assert payload["reasoning_effort"] == "minimal"
    assert payload["response_format"]["json_schema"]["strict"] is True
    user = payload["messages"][1]["content"]
    assert user[0]["type"] == "text"
    assert user[1]["type"] == "image_url"
    assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_run_loop_revises_once_and_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_BYTES)
    generation_prompts: list[str] = []

    def fake_meta_chat(
        *, api_key: str, payload: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        generation_prompts.append(payload["messages"][1]["content"][0]["text"])
        candidate = {
            "target_description": "red object" if len(generation_prompts) == 1 else "red rectangle",
            "sam_concept": "red object" if len(generation_prompts) == 1 else "red rectangle",
            "preserve_concepts": ["black podium"],
            "affected_concepts": ["shadow"],
            "evidence": ["red object is visible"],
            "ambiguous": False,
            "question_if_ambiguous": "",
            "confidence": 0.8,
        }
        return candidate, {"raw": {"id": f"meta-{len(generation_prompts)}"}, "usage": {}}

    def fake_run_critic(
        *,
        command: str,
        model: str,
        endpoint: str | None,
        image: Path,
        image_data: str,
        image_bytes: bytes,
        user_request: str,
        candidate: dict[str, Any],
        critic_api_key: str,
        timeout: float,
        allowed_hosts: Sequence[str] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        passed = candidate["sam_concept"] == "red rectangle"
        critique = {
            "passed": passed,
            "failed_criteria": [] if passed else ["sam_concept_is_bare_noun_phrase"],
            "corrected_sam_concept": None if passed else "red rectangle",
            "confidence": 0.9,
            "reason": "candidate wording revised to a concrete noun phrase",
            "visual_evidence": ["red object and black podium are visible"],
        }
        return critique, {"raw": {"id": "critic"}, "usage": {}}

    monkeypatch.setattr(prompt_critique, "meta_chat", fake_meta_chat)
    monkeypatch.setattr(prompt_critique, "run_critic", fake_run_critic)
    result = prompt_critique.run_loop(
        image=image,
        user_request="remove the red object",
        api_key="test-key",
        meta_model="muse-test",
        critic_command="zai-vision",
        critic_model="glm-test",
        critic_endpoint=None,
        critic_api_key="",
        timeout=10,
        max_iterations=2,
    )
    assert len(generation_prompts) == 2
    assert "critic_feedback" not in generation_prompts[0]
    assert "critic_feedback" in generation_prompts[1]
    assert result["status"] == "passed"
    assert result["iterations_used"] == 2
    assert result["final_candidate"]["sam_concept"] == "red rectangle"


def test_critic_output_rejects_unknown_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    critique = {
        "passed": False,
        "failed_criteria": ["made_up"],
        "corrected_sam_concept": None,
        "confidence": 0.9,
        "reason": "unknown rubric",
        "visual_evidence": ["frame"],
    }
    outer = {
        "choices": [{"message": {"content": json.dumps(critique)}}],
        "usage": {},
    }

    class Completed:
        returncode = 0
        stdout = json.dumps(outer)
        stderr = ""

    monkeypatch.setattr(prompt_critique.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(prompt_critique.PromptLoopError, match="unknown rubric"):
        prompt_critique.run_critic(
            command="zai-vision",
            model="glm-test",
            endpoint=None,
            image=Path("/tmp/image.png"),
            image_data="data:image/png;base64,eA==",
            image_bytes=PNG_BYTES,
            user_request="remove the object",
            candidate={"sam_concept": "object"},
            critic_api_key="",
            timeout=5,
        )


def test_critic_output_rejects_contradictory_pass_flag() -> None:
    with pytest.raises(prompt_critique.PromptLoopError, match="contradicts"):
        prompt_critique.validate_critique(
            {
                "passed": True,
                "failed_criteria": ["target_visible"],
                "corrected_sam_concept": None,
                "confidence": 0.9,
                "reason": "contradiction",
                "visual_evidence": ["frame"],
            }
        )


def test_invalid_advisory_correction_is_cleared_without_aborting() -> None:
    critique = {
        "passed": False,
        "failed_criteria": ["sam_concept_is_bare_noun_phrase"],
        "corrected_sam_concept": "the red car",
        "confidence": 0.9,
        "reason": "invalid advisory correction",
        "visual_evidence": ["frame"],
    }
    prompt_critique.validate_critique(critique)
    assert critique["corrected_sam_concept"] is None


def test_endpoint_critic_adds_bearer_only_when_key_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_request(url: str, data: bytes, headers: dict[str, str], method: str) -> dict[str, Any]:
        captured.update({"url": url, "headers": headers, "method": method})
        return captured

    def fake_open(*_args: Any, **_kwargs: Any) -> None:
        raise prompt_critique.PromptLoopError("stop after header capture")

    monkeypatch.setattr(prompt_critique.urllib.request, "Request", fake_request)
    monkeypatch.setattr(prompt_critique.urllib.request, "urlopen", fake_open)
    with pytest.raises(prompt_critique.PromptLoopError, match="stop after"):
        prompt_critique.run_critic(
            command="zai-vision",
            model="glm-5.3-flash",
            endpoint="https://critic.invalid/v1/chat/completions",
            image=Path("/tmp/image.png"),
            image_data="data:image/png;base64,eA==",
            image_bytes=PNG_BYTES,
            user_request="remove the object",
            candidate={"sam_concept": "red object"},
            critic_api_key="critic-secret",
            timeout=5,
            allowed_hosts=["critic.invalid"],
        )
    assert captured["headers"]["Authorization"] == "Bearer critic-secret"


def test_parse_args_uses_public_model_for_generic_endpoint() -> None:
    args = prompt_critique.parse_args(
        [
            "--image",
            "frame.png",
            "--request",
            "remove the object",
            "--output",
            "result.json",
            "--critic-endpoint",
            "https://critic.invalid/v1/chat/completions",
        ]
    )
    assert args.critic_model == "glm-5.3-flash"
    assert args.ledger_output == Path("result.ledger.json")


def test_remote_http_critic_endpoint_is_rejected() -> None:
    with pytest.raises(prompt_critique.PromptLoopError, match="HTTPS"):
        prompt_critique.run_critic(
            command="zai-vision",
            model="glm-5.3-flash",
            endpoint="http://critic.invalid/v1/chat/completions",
            image=Path("/tmp/image.png"),
            image_data="data:image/png;base64,eA==",
            image_bytes=PNG_BYTES,
            user_request="remove the object",
            candidate={"sam_concept": "red object"},
            critic_api_key="",
            timeout=5,
        )


def test_subprocess_timeout_is_ceiled_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_run(command: list[str], **_kwargs: Any) -> None:
        captured.extend(command)
        raise prompt_critique.PromptLoopError("stop after command capture")

    monkeypatch.setattr(prompt_critique.subprocess, "run", capture_run)
    with pytest.raises(prompt_critique.PromptLoopError, match="stop after"):
        prompt_critique.run_critic(
            command="zai-vision",
            model="glm-test",
            endpoint=None,
            image=Path("/tmp/image.png"),
            image_data="data:image/png;base64,eA==",
            image_bytes=PNG_BYTES,
            user_request="remove the object",
            candidate={"sam_concept": "red object"},
            critic_api_key="",
            timeout=0.5,
        )
    assert captured[captured.index("--timeout") + 1] == "1"


def test_critic_request_digest_includes_exact_image() -> None:
    candidate = {"sam_concept": "red rectangle"}
    first = prompt_critique.critique_request_digest(
        image_bytes=b"frame-a", user_request="remove the object", candidate=candidate
    )
    second = prompt_critique.critique_request_digest(
        image_bytes=b"frame-b", user_request="remove the object", candidate=candidate
    )
    assert first != second


def test_run_loop_enforces_sam_boundary_before_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_BYTES)
    concepts = ["remove red object keep podium", "red object"]

    def fake_meta_chat(
        *, api_key: str, payload: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        concept = concepts.pop(0)
        candidate = {
            "target_description": concept,
            "sam_concept": concept,
            "preserve_concepts": ["black podium"],
            "affected_concepts": ["shadow"],
            "evidence": ["red object is visible"],
            "ambiguous": False,
            "question_if_ambiguous": "",
            "confidence": 0.8,
        }
        return candidate, {"raw": {"id": f"meta-{concept}"}, "usage": {}}

    seen_concepts: list[str] = []

    def fake_run_critic(
        *, candidate: dict[str, Any], **_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seen_concepts.append(candidate["sam_concept"])
        return (
            {
                "passed": True,
                "failed_criteria": [],
                "corrected_sam_concept": None,
                "confidence": 0.9,
                "reason": "valid concept",
                "visual_evidence": ["red object is visible"],
            },
            {"raw": {"id": "critic"}, "usage": {}},
        )

    monkeypatch.setattr(prompt_critique, "meta_chat", fake_meta_chat)
    monkeypatch.setattr(prompt_critique, "run_critic", fake_run_critic)
    result = prompt_critique.run_loop(
        image=image,
        user_request="remove the red object",
        api_key="test-key",
        meta_model="muse-test",
        critic_command="zai-vision",
        critic_model="glm-test",
        critic_endpoint=None,
        critic_api_key="",
        timeout=5,
        max_iterations=2,
    )
    assert result["status"] == "passed"
    assert result["iterations_used"] == 2
    assert result["iterations"][0]["critique"]["model"] == "local-validation"
    assert seen_concepts == ["red object"]


def test_failed_critic_preserves_paid_generation_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_BYTES)

    def fake_meta_chat(
        *, api_key: str, payload: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate = {
            "target_description": "red rectangle",
            "sam_concept": "red rectangle",
            "preserve_concepts": ["black podium"],
            "affected_concepts": ["shadow"],
            "evidence": ["red rectangle"],
            "ambiguous": False,
            "question_if_ambiguous": "",
            "confidence": 0.8,
        }
        return candidate, {"raw": {"id": "paid-generation"}, "usage": {}}

    def failing_critic(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        raise prompt_critique.PromptLoopError("critic transport failed")

    monkeypatch.setattr(prompt_critique, "meta_chat", fake_meta_chat)
    monkeypatch.setattr(prompt_critique, "run_critic", failing_critic)
    with pytest.raises(prompt_critique.PromptLoopError) as caught:
        prompt_critique.run_loop(
            image=image,
            user_request="remove the red object",
            api_key="test-key",
            meta_model="muse-test",
            critic_command="zai-vision",
            critic_model="glm-test",
            critic_endpoint=None,
            critic_api_key="",
            timeout=5,
            max_iterations=2,
        )
    assert caught.value.details is not None
    record = caught.value.details["iterations"][0]["generation"]
    assert record["response_id"] == "paid-generation"


def test_subprocess_timeout_preserves_paid_generation_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_BYTES)

    def fake_meta_chat(
        *, api_key: str, payload: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate = {
            "target_description": "red rectangle",
            "sam_concept": "red rectangle",
            "preserve_concepts": ["black podium"],
            "affected_concepts": ["shadow"],
            "evidence": ["red rectangle"],
            "ambiguous": False,
            "question_if_ambiguous": "",
            "confidence": 0.8,
        }
        return candidate, {"raw": {"id": "paid-generation"}, "usage": {}}

    def timeout_run(*_args: Any, **_kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="zai-vision", timeout=5)

    monkeypatch.setattr(prompt_critique, "meta_chat", fake_meta_chat)
    monkeypatch.setattr(prompt_critique.subprocess, "run", timeout_run)
    with pytest.raises(prompt_critique.PromptLoopError, match="timed out") as caught:
        prompt_critique.run_loop(
            image=image,
            user_request="remove the red object",
            api_key="test-key",
            meta_model="muse-test",
            critic_command="zai-vision",
            critic_model="glm-test",
            critic_endpoint=None,
            critic_api_key="",
            timeout=5,
            max_iterations=2,
        )
    assert caught.value.details is not None
    assert caught.value.details["iterations"][0]["generation"]["response_id"] == "paid-generation"


def test_meta_ledger_fragment_models_usage() -> None:
    fragment = prompt_critique.build_meta_ledger_fragment(
        {
            "iterations": [
                {
                    "generation": {
                        "model": "muse-test",
                        "response_id": "meta-1",
                        "request_sha256": "request",
                        "output_sha256": "output",
                        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                        "elapsed_seconds": 1.0,
                    }
                }
            ]
        }
    )
    assert fragment["model_api_calls"][0]["modeled_debit"] == 5.5


def test_ledger_fragment_tolerates_null_usage_and_records_models() -> None:
    fragment = prompt_critique.build_meta_ledger_fragment(
        {
            "iterations": [
                {
                    "generation": {
                        "model": "served-model",
                        "requested_model": "requested-model",
                        "served_model": "served-model",
                        "response_id": "meta-1",
                        "request_sha256": "request",
                        "output_sha256": "output",
                        "usage": None,
                    }
                }
            ]
        }
    )
    call = fragment["model_api_calls"][0]
    assert call["usage"] == {}
    assert call["requested_model"] == "requested-model"
    assert call["served_model"] == "served-model"


def test_meta_transport_failure_retains_attempt_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_open(*_args: Any, **_kwargs: Any) -> None:
        raise TimeoutError("network timeout")

    monkeypatch.setattr(prompt_critique.urllib.request, "urlopen", failing_open)
    with pytest.raises(prompt_critique.PromptLoopError, match="network timeout") as caught:
        prompt_critique.meta_chat(
            api_key="test-key",
            payload={"model": "muse-test"},
            timeout=0.5,
        )
    assert caught.value.details is not None
    assert caught.value.details["api_partial"] == {"raw": None, "usage": None}


def test_preflight_failure_writes_receipt_and_empty_ledger_fragment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "not-an-image.png"
    image.write_text("not image bytes", encoding="utf-8")
    output = tmp_path / "result.json"
    ledger = tmp_path / "ledger.json"
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    status = prompt_critique.main(
        [
            "--image",
            str(image),
            "--request",
            "remove the object",
            "--output",
            str(output),
            "--ledger-output",
            str(ledger),
        ]
    )
    assert status == 1
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["image_sha256"] is None
    assert failure["iterations"] == []
    assert json.loads(ledger.read_text(encoding="utf-8"))["model_api_calls"] == []


def test_sam_guard_rejects_spatial_relational_and_two_object_phrases() -> None:
    invalid = [
        "side of car",
        "handle of door",
        "atop black podium",
        "surfboard person",
        "next to the car",
    ]
    assert not any(prompt_critique.is_bare_noun_phrase(value) for value in invalid)


def test_sam_guard_accepts_known_compounds_and_open_vocabulary() -> None:
    valid = [
        "car door",
        "toy car",
        "bottle cap",
        "window frame",
        "phone screen",
        "table lamp",
        "coffee cup",
        "water bottle",
        "red sports car",
        "traffic cone",
        "logo",
        "surfboard",
    ]
    assert all(prompt_critique.is_bare_noun_phrase(value) for value in valid)


def test_remote_critic_endpoint_requires_explicit_host_allowlist() -> None:
    with pytest.raises(prompt_critique.PromptLoopError, match="not allowlisted"):
        prompt_critique.run_critic(
            command="zai-vision",
            model="glm-5.3-flash",
            endpoint="https://critic.invalid/v1/chat/completions",
            image=Path("/tmp/image.png"),
            image_data="data:image/png;base64,eA==",
            image_bytes=PNG_BYTES,
            user_request="remove the object",
            candidate={"sam_concept": "red object"},
            critic_api_key="",
            timeout=5,
        )


def test_allowed_critic_host_passes_the_endpoint_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open(*_args: Any, **_kwargs: Any) -> None:
        raise prompt_critique.PromptLoopError("request attempted")

    monkeypatch.setattr(prompt_critique.urllib.request, "urlopen", fake_open)
    with pytest.raises(prompt_critique.PromptLoopError, match="request attempted"):
        prompt_critique.run_critic(
            command="zai-vision",
            model="glm-5.3-flash",
            endpoint="https://critic.invalid/v1/chat/completions",
            image=Path("/tmp/image.png"),
            image_data="data:image/png;base64,eA==",
            image_bytes=PNG_BYTES,
            user_request="remove the object",
            candidate={"sam_concept": "red object"},
            critic_api_key="",
            timeout=5,
            allowed_hosts=["critic.invalid"],
        )


def test_critic_command_must_be_allowlisted_or_an_absolute_executable() -> None:
    prompt_critique.validate_critic_command("zai-vision")
    prompt_critique.validate_critic_command(sys.executable)
    with pytest.raises(prompt_critique.PromptLoopError, match="allowlisted"):
        prompt_critique.validate_critic_command("rm -rf /")
    with pytest.raises(prompt_critique.PromptLoopError, match="allowlisted"):
        prompt_critique.validate_critic_command("./relative-critic")


def test_critic_cli_prompt_carries_the_critic_system_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class Completed:
        returncode = 0
        stdout = json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": {}})
        stderr = ""

    def capture_run(command: list[str], **_kwargs: Any) -> Completed:
        captured.extend(command)
        return Completed()

    monkeypatch.setattr(prompt_critique.subprocess, "run", capture_run)
    with pytest.raises(prompt_critique.PromptLoopError):
        prompt_critique.run_critic(
            command="zai-vision",
            model="glm-test",
            endpoint=None,
            image=Path("/tmp/image.png"),
            image_data="data:image/png;base64,eA==",
            image_bytes=PNG_BYTES,
            user_request="remove the object",
            candidate={"sam_concept": "red object"},
            critic_api_key="",
            timeout=5,
        )
    prompt = captured[-1]
    assert prompt_critique.CRITIC_SYSTEM in prompt
    assert "critique the candidate" in prompt.lower()


def test_http_client_exception_is_reported_as_a_prompt_loop_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_open(*_args: Any, **_kwargs: Any) -> None:
        raise http.client.IncompleteRead(b"partial", 64)

    monkeypatch.setattr(prompt_critique.urllib.request, "urlopen", failing_open)
    with pytest.raises(prompt_critique.PromptLoopError, match="request failed") as caught:
        prompt_critique.meta_chat(
            api_key="test-key",
            payload={"model": "muse-test"},
            timeout=0.5,
        )
    assert caught.value.details is not None
    assert caught.value.details["api_partial"] == {"raw": None, "usage": None}


def test_schema_invalid_paid_generation_keeps_output_and_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_BYTES)
    raw_content = '{"sam_concept": "red car"}'

    def failing_meta_chat(
        *, api_key: str, payload: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        failure = prompt_critique.PromptLoopError("candidate schema rejected")
        failure.details = {
            "api_partial": {
                "raw": {
                    "id": "chatcmpl-paid",
                    "model": "muse-spark-1.3",
                    "choices": [{"message": {"content": raw_content}}],
                },
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            }
        }
        raise failure

    monkeypatch.setattr(prompt_critique, "meta_chat", failing_meta_chat)
    with pytest.raises(prompt_critique.PromptLoopError) as caught:
        prompt_critique.run_loop(
            image=image,
            user_request="remove the red object",
            api_key="test-key",
            meta_model="muse-spark-1.3",
            critic_command="zai-vision",
            critic_model="glm-test",
            critic_endpoint=None,
            critic_api_key="",
            timeout=5,
            max_iterations=2,
        )
    details = caught.value.details
    assert details is not None
    assert details["api_partial"]["raw"]["id"] == "chatcmpl-paid"
    record = details["iterations"][0]["generation"]
    assert record["response_id"] == "chatcmpl-paid"
    assert record["raw_output"] == raw_content
    assert record["raw_output_sha256"] == hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    assert record["output"] is None
    assert record["output_sha256"] is None
    assert record["served_model"] == "muse-spark-1.3"
    assert record["requested_model"] == "muse-spark-1.3"


def test_partial_ledger_entry_does_not_claim_the_requested_model_was_served() -> None:
    fragment = prompt_critique.build_meta_ledger_fragment(
        {
            "iterations": [
                {
                    "generation": {
                        "model": "muse-spark-1.3",
                        "requested_model": "muse-spark-1.3",
                        "served_model": None,
                        "response_id": None,
                        "request_sha256": "request",
                        "output_sha256": None,
                        "usage": {},
                    }
                }
            ]
        }
    )
    call = fragment["model_api_calls"][0]
    assert call["requested_model"] == "muse-spark-1.3"
    assert call["served_model"] is None


def test_ledger_fragment_records_pricing_and_vocabulary_versions() -> None:
    fragment = prompt_critique.build_meta_ledger_fragment({"iterations": []})
    assert fragment["sam_vocab_version"] == prompt_critique.SAM_VOCAB_VERSION
    calls = prompt_critique.build_meta_ledger_fragment(
        {
            "iterations": [
                {
                    "generation": {
                        "model": "muse-test",
                        "response_id": "meta-1",
                        "request_sha256": "request",
                        "output_sha256": "output",
                        "usage": {
                            "prompt_tokens": 1_000_000,
                            "completion_tokens": 1_000_000,
                        },
                    }
                }
            ]
        }
    )
    basis = calls["model_api_calls"][0]["pricing_basis"]
    assert basis["version"] == prompt_critique.META_PRICING_VERSION
    assert basis["as_of"] == prompt_critique.META_PRICING_AS_OF


def test_missing_api_key_writes_receipt_and_empty_ledger_fragment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    output = tmp_path / "result.json"
    ledger = tmp_path / "ledger.json"
    status = prompt_critique.main(
        [
            "--image",
            str(tmp_path / "frame.png"),
            "--request",
            "remove the object",
            "--output",
            str(output),
            "--ledger-output",
            str(ledger),
        ]
    )
    assert status == 2
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert "MODEL_API_KEY" in failure["error"]
    assert json.loads(ledger.read_text(encoding="utf-8"))["model_api_calls"] == []


def test_iteration_bound_has_one_source_of_truth() -> None:
    assert prompt_critique.validate_max_iterations(1) == 1
    assert prompt_critique.validate_max_iterations(2) == 2
    with pytest.raises(prompt_critique.PromptLoopError, match="max iterations"):
        prompt_critique.validate_max_iterations(0)
    with pytest.raises(prompt_critique.PromptLoopError, match="max iterations"):
        prompt_critique.validate_max_iterations(3)
