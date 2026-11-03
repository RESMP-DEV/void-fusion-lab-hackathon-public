#!/usr/bin/env python3
"""Adapter from an AIMLAPI multimodal model to samurai expert regions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_ENDPOINT = "https://api.aimlapi.com/v1/chat/completions"


def _image_data_url(path: Path, max_image_size: int) -> str:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_image_size:
        scale = max_image_size / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_payload(
    *,
    model: str,
    request: dict[str, Any],
    max_image_size: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    prompt = request.get("prompt") or "Identify the target object to remove."
    contract = (
        "Return only a JSON array named regions_by_frame. For each input image, "
        'in the same order, emit {"frame_index": <integer>, "regions": [...]}. '
        'Each region is {"type": "box" or "polygon", "coordinates": [...], '
        '"confidence": <0..1>}. Box coordinates are normalized x1,y1,x2,y2. '
        "Polygon coordinates are normalized x,y pairs. No prose."
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{prompt}\n\n{contract}"},
    ]
    for frame in request["frames"]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(Path(frame["path"]), max_image_size)},
            }
        )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_output_tokens,
        "temperature": 0,
    }


def extract_message_text_with_source(message: dict[str, Any]) -> tuple[str, str]:
    """Return the model's answer text and the name of the field that held it.

    Providers differ in shape: content may be a plain string, a list of
    text parts, or absent/null with the text carried in reasoning_content
    (MiniMax-M3.1-Flash omits the content key when its reasoning consumes
    the whole output budget). The source is one of "content",
    "content_list", "reasoning_content". A message with no extractable
    text raises KeyError("content"), the same failure the direct read
    produced.
    """
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content, "content"
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        joined = "".join(parts)
        if joined.strip():
            return joined, "content_list"
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning, "reasoning_content"
    raise KeyError("content")


def extract_message_text(message: dict[str, Any]) -> str:
    """Return the model's answer text from a chat-completion message."""
    return extract_message_text_with_source(message)[0]


def _content_state(message: dict[str, Any]) -> str:
    """Name why content supplied no text: absent, null, or a textless shape."""
    if "content" not in message:
        return "absent"
    content = message["content"]
    if content is None:
        return "null"
    if isinstance(content, str):
        return "empty string"
    if isinstance(content, list):
        return "list without text"
    return type(content).__name__


def _describe_field(message: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in message:
        return {"present": False, "type": "absent", "length": None}
    value = message[name]
    if value is None:
        return {"present": True, "type": "null", "length": None}
    if isinstance(value, str):
        return {"present": True, "type": "string", "length": len(value)}
    if isinstance(value, list):
        text_length = sum(
            len(part["text"])
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        return {
            "present": True,
            "type": "list",
            "length": text_length,
            "parts": len(value),
        }
    return {"present": True, "type": type(value).__name__, "length": None}


def build_diagnostics(
    response: dict[str, Any],
    *,
    model: str,
    endpoint: str,
    extracted_text: str | None,
    extracted_from: str | None,
    raw_response_file: str | None = None,
) -> dict[str, Any]:
    """Build the per-call diagnostics record: the response shape, never prose.

    The record names the message keys present, the character length of
    the content and reasoning_content fields, which field supplied the
    extracted text, the finish reason, the usage counts, and the
    SHA-256 of the extracted text. It never carries the text itself.
    """
    choice: dict[str, Any] = {}
    message: dict[str, Any] = {}
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            if isinstance(choice.get("message"), dict):
                message = choice["message"]
    usage = response.get("usage") if isinstance(response, dict) else None
    return {
        "adapter": "aimlapi_expert",
        "recorded_at": datetime.now(UTC).isoformat(),
        "model": model,
        "endpoint": endpoint,
        "message_keys": sorted(message.keys()),
        "content": _describe_field(message, "content"),
        "reasoning_content": _describe_field(message, "reasoning_content"),
        "extracted_from": extracted_from,
        "extracted_text_length": len(extracted_text) if extracted_text is not None else None,
        "extracted_text_sha256": (
            hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()
            if extracted_text is not None
            else None
        ),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage if isinstance(usage, dict) else None,
        "raw_response_file": raw_response_file,
    }


def _capture_stem() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"response-{stamp}-{os.getpid()}-{time.time_ns() % 1_000_000:06d}"


def capture_raw_response(response: dict[str, Any], directory: Path, stem: str) -> str | None:
    """Write the raw response JSON into the capture directory (best effort)."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{stem}.json"
        target.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return target.name
    except OSError:
        return None


def write_diagnostics(
    diagnostics: dict[str, Any],
    *,
    explicit_path: str | None,
    capture_dir: Path | None,
    stem: str | None,
) -> None:
    """Write the diagnostics record to each configured sink (best effort).

    Sinks: the --diagnostics-out path when given, and
    <EXPERT_RESPONSE_CAPTURE>/<stem>.diag.json when response capture is
    on. Diagnostics are observability only: a failed write never changes
    the call's outcome or its fail-closed behavior.
    """
    targets: list[Path] = []
    if explicit_path:
        targets.append(Path(explicit_path))
    if capture_dir is not None and stem is not None:
        targets.append(capture_dir / f"{stem}.diag.json")
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass


def parse_model_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("model response does not contain a JSON array")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("regions_by_frame must be an array")
    return {"regions_by_frame": parsed}


def call_api(
    payload: dict[str, Any], *, endpoint: str, api_key: str, timeout: float
) -> dict[str, Any]:
    http_request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "samurai-openai-vision-adapter/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"AIMLAPI request failed: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-image-size", type=int, default=1280)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--diagnostics-out", default=None)
    args = parser.parse_args()
    try:
        request = json.loads(sys.stdin.read())
        payload = build_payload(
            model=args.model,
            request=request,
            max_image_size=args.max_image_size,
            max_output_tokens=args.max_output_tokens,
        )
        api_key = os.environ.get("AIMLAPI_KEY")
        if not api_key:
            raise RuntimeError("AIMLAPI_KEY is not set")
        response = call_api(
            payload,
            endpoint=args.endpoint,
            api_key=api_key,
            timeout=args.timeout,
        )
        capture_env = os.environ.get("EXPERT_RESPONSE_CAPTURE")
        capture_dir = Path(capture_env) if capture_env else None
        stem = _capture_stem() if capture_dir is not None else None
        raw_name = (
            capture_raw_response(response, capture_dir, stem)
            if capture_dir is not None and stem is not None
            else None
        )
        message = response["choices"][0]["message"]
        try:
            content, source = extract_message_text_with_source(message)
        except KeyError:
            write_diagnostics(
                build_diagnostics(
                    response,
                    model=args.model,
                    endpoint=args.endpoint,
                    extracted_text=None,
                    extracted_from=None,
                    raw_response_file=raw_name,
                ),
                explicit_path=args.diagnostics_out,
                capture_dir=capture_dir,
                stem=stem,
            )
            raise
        write_diagnostics(
            build_diagnostics(
                response,
                model=args.model,
                endpoint=args.endpoint,
                extracted_text=content,
                extracted_from=source,
                raw_response_file=raw_name,
            ),
            explicit_path=args.diagnostics_out,
            capture_dir=capture_dir,
            stem=stem,
        )
        try:
            result = parse_model_content(content)
        except ValueError as error:
            if source == "reasoning_content":
                raise ValueError(
                    "reasoning-fallback parse failure "
                    f"(content {_content_state(message)}; text from "
                    f"reasoning_content, {len(content)} chars): {error}"
                ) from error
            raise
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f"aimlapi_expert: {error}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
