#!/usr/bin/env python3
"""Run a bounded Muse Spark draft / GLM vision critique prompt loop."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Long prompt prose is intentionally kept as readable multiline text.
# ruff: noqa: E501

META_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_META_MODEL = "muse-spark-1.3"
DEFAULT_CRITIC_MODEL = "glm-5.3-flash"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_ITERATIONS = 2
MAX_SAM_CONCEPT_WORDS = 6
META_INPUT_USD_PER_1M = 1.25
META_OUTPUT_USD_PER_1M = 4.25
META_PRICING_VERSION = "meta-standard-2026-10-26"
META_PRICING_AS_OF = "2026-10-26"
# Vocabulary revision for the deterministic SAM text-boundary guard below.
SAM_VOCAB_VERSION = "sam-3.1-2026-10-27"
# Bare command names the critic CLI may launch. Absolute paths to an existing
# executable are also accepted; everything else is rejected.
ALLOWED_CRITIC_COMMANDS = frozenset({"zai-vision"})
# Environment fallback for the explicit critic-host allowlist.
CRITIC_ALLOWED_HOSTS_ENV = "PROMPT_CRITIC_ALLOWED_HOSTS"
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
FORBIDDEN_SAM_TERMS = frozenset(
    {
        "and",
        "or",
        "not",
        "no",
        "to",
        "without",
        "exclude",
        "excluding",
        "keep",
        "preserve",
        "remove",
        "segment",
        "find",
        "where",
        "above",
        "below",
        "left",
        "right",
        "behind",
        "under",
        "over",
        "with",
        "on",
        "in",
        "at",
        "erase",
        "delete",
        "select",
        "mask",
        "protect",
        "identify",
        "locate",
        "show",
        "choose",
        "isolate",
        "crop",
        "cut",
        "outline",
        "draw",
        "highlight",
        "point",
        "click",
        "near",
        "beside",
        "between",
        "next",
        "top",
        "bottom",
        "front",
        "back",
        "inside",
        "outside",
        "toward",
        "towards",
        "across",
        "along",
        "around",
        "beyond",
        "against",
        "among",
        "beneath",
        "upon",
        "within",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "why",
        "how",
        "the",
        "a",
        "an",
        "of",
        "atop",
        "onto",
        "into",
        "underneath",
        "alongside",
        "adjacent",
        "nearby",
        "opposite",
        "facing",
        "amid",
        "amidst",
        "past",
        "through",
        "throughout",
        "plus",
        "versus",
        "vs",
        "either",
        "neither",
        "both",
        "also",
        "per",
        "via",
    }
)
SAM_COMPOUND_PREFIXES = frozenset(
    {
        # attribute nouns that legitimately modify a head noun
        "bus",
        "cable",
        "container",
        "engine",
        "face",
        "fish",
        "flower",
        "glass",
        "hand",
        "plant",
        "road",
        "rock",
        "screen",
        "train",
        "tree",
        "truck",
        "wall",
        "wire",
        "bottle",
        "book",
        "car",
        "clock",
        "coffee",
        "computer",
        "cup",
        "dining",
        "door",
        "garden",
        "home",
        "house",
        "kitchen",
        "lamp",
        "light",
        "mirror",
        "office",
        "paper",
        "phone",
        "photo",
        "picture",
        "plate",
        "race",
        "restaurant",
        "school",
        "shelf",
        "sign",
        "sports",
        "street",
        "swimming",
        "table",
        "toy",
        "traffic",
        "water",
        "window",
    }
)
SAM_OBJECT_NOUNS = frozenset(
    {
        "animal",
        "artifact",
        "athlete",
        "bag",
        "ball",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "box",
        "bus",
        "cable",
        "car",
        "cat",
        "chair",
        "cone",
        "cup",
        "dog",
        "door",
        "face",
        "fish",
        "flower",
        "glass",
        "hand",
        "horse",
        "laptop",
        "marker",
        "object",
        "person",
        "phone",
        "plant",
        "player",
        "plate",
        "podium",
        "rectangle",
        "reflection",
        "road",
        "rock",
        "screen",
        "shadow",
        "sign",
        "surface",
        "table",
        "toy",
        "track",
        "train",
        "tree",
        "truck",
        "utensil",
        "vehicle",
        "wall",
        "window",
        "wire",
        "backpack",
        "balloon",
        "banana",
        "basket",
        "bean",
        "belt",
        "bench",
        "blanket",
        "blender",
        "book",
        "boot",
        "bowl",
        "branch",
        "brick",
        "brush",
        "bucket",
        "bulb",
        "bush",
        "cabinet",
        "calendar",
        "camera",
        "candle",
        "cap",
        "cart",
        "carton",
        "case",
        "chain",
        "chalk",
        "charger",
        "chart",
        "clock",
        "cloth",
        "coat",
        "coffee",
        "collar",
        "computer",
        "cord",
        "couch",
        "counter",
        "curtain",
        "cushion",
        "deck",
        "desk",
        "dial",
        "dinner",
        "dish",
        "doll",
        "dress",
        "drill",
        "drum",
        "dumbbell",
        "earring",
        "engine",
        "fan",
        "fence",
        "flag",
        "flashlight",
        "flute",
        "fork",
        "fridge",
        "fruit",
        "guitar",
        "hat",
        "helmet",
        "hose",
        "jacket",
        "jar",
        "kettle",
        "keyboard",
        "knife",
        "ladder",
        "lamp",
        "leaf",
        "light",
        "logo",
        "luggage",
        "magnet",
        "mailbox",
        "mask",
        "mat",
        "microwave",
        "mirror",
        "monitor",
        "mouse",
        "mug",
        "napkin",
        "necklace",
        "newspaper",
        "notebook",
        "oven",
        "package",
        "pan",
        "pants",
        "paper",
        "pen",
        "pencil",
        "photo",
        "piano",
        "pillow",
        "pipe",
        "pole",
        "poster",
        "pot",
        "purse",
        "radio",
        "remote",
        "ring",
        "rope",
        "rug",
        "sand",
        "scarf",
        "scissors",
        "scooter",
        "screw",
        "shelf",
        "shirt",
        "shoe",
        "skateboard",
        "snowboard",
        "soap",
        "socket",
        "sofa",
        "spoon",
        "spray",
        "sticker",
        "stool",
        "strap",
        "stroller",
        "surfboard",
        "sweater",
        "switch",
        "tank",
        "tattoo",
        "tea",
        "tent",
        "tie",
        "tile",
        "timer",
        "tin",
        "tire",
        "towel",
        "tractor",
        "umbrella",
        "vase",
        "vest",
        "watch",
        "wheel",
        "wrench",
        "yard",
    }
)

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "target_description",
        "sam_concept",
        "preserve_concepts",
        "affected_concepts",
        "evidence",
        "ambiguous",
        "question_if_ambiguous",
        "confidence",
    ],
    "properties": {
        "target_description": {"type": "string", "minLength": 3, "maxLength": 160},
        "sam_concept": {"type": "string", "minLength": 1, "maxLength": 80},
        "preserve_concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "affected_concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "ambiguous": {"type": "boolean"},
        "question_if_ambiguous": {"type": "string", "maxLength": 240},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

CRITERIA = (
    "target_visible",
    "target_matches_user_request",
    "sam_concept_is_bare_noun_phrase",
    "preserve_object_excluded",
    "affected_semantics_supported",
    "no_contradictions",
)

GENERATOR_SYSTEM = """You are the target and affected-region prompt generator for video object removal.
## Objective
Produce one canonical SAM 3.1 target concept and bounded affected-region concepts from the labeled frame and user request.
## Priority rules
1. Preserve the object or region explicitly protected by the user.
2. Make `sam_concept` a short bare noun phrase, never a command, question, negation, or multi-concept list.
3. Return `ambiguous: true` and a concrete question when target identity is unresolved; do not guess.
4. Report only visual evidence present in the frame.
## Output
Return exactly the required JSON object. Use an empty array for absent affected concepts and an empty string when no question is needed.
## Completion
Stop after all fields are supported. If evidence is unavailable, mark the candidate ambiguous.
"""

CRITIC_SYSTEM = """You are an independent vision critic for object-removal prompts.
Evaluate only the supplied frame, user request, and candidate JSON. Apply each rubric item as pass/fail.
Never rewrite unrelated fields. A failed criterion must cite concrete visual evidence.
Return JSON only with keys passed, failed_criteria, corrected_sam_concept, confidence, reason, and visual_evidence.
Use corrected_sam_concept only when the candidate's target is visible but its wording is not a bare noun phrase; otherwise use null.
"""


class PromptLoopError(RuntimeError):
    """A bounded prompt-loop request or response was invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.details: dict[str, Any] | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```"):
            value = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    candidates: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    candidate = json.loads(value[start : index + 1])
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, dict):
                    candidates.append(candidate)
                start = None
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise PromptLoopError("model response did not contain exactly one JSON object")
    raise PromptLoopError("model response contained multiple JSON objects")


def read_image(path: Path) -> tuple[bytes, str]:
    # Strict extension/magic contract: renaming without transcoding fails closed.
    image_type = IMAGE_TYPES.get(path.suffix.lower())
    if image_type is None:
        raise PromptLoopError(f"unsupported image type {path.suffix!r}; use JPEG or PNG")
    try:
        with path.open("rb") as source:
            data = source.read(MAX_IMAGE_BYTES + 1)
    except OSError as error:
        raise PromptLoopError(f"could not read image: {error}") from error
    if len(data) > MAX_IMAGE_BYTES:
        raise PromptLoopError(f"image is {len(data)} bytes; maximum is {MAX_IMAGE_BYTES}")
    if not data:
        raise PromptLoopError("image is empty")
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    if (image_type == "image/png" and not is_png) or (image_type == "image/jpeg" and not is_jpeg):
        raise PromptLoopError("image contents do not match the PNG or JPEG file extension")
    return data, image_type


def image_data_url(data: bytes, image_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{image_type};base64,{encoded}"


def is_bare_noun_phrase(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    concept = value.strip().lower()
    if not concept or len(concept) > 80:
        return False
    if not re.fullmatch(r"[a-z0-9]+(?:[ \-'][a-z0-9]+)*", concept):
        return False
    words = concept.replace("-", " ").replace("'", " ").split()
    if not 1 <= len(words) <= MAX_SAM_CONCEPT_WORDS:
        return False
    if set(words) & FORBIDDEN_SAM_TERMS:
        return False
    # A modifier that is itself a known object noun means the phrase names two
    # objects, which the SAM text boundary forbids. Known compounds such as
    # "car door" are exempted by the compound-prefix list. The head noun is
    # never checked, so the target vocabulary stays open.
    earlier_words = words[:-1]
    return not any(
        word in SAM_OBJECT_NOUNS and word not in SAM_COMPOUND_PREFIXES for word in earlier_words
    )


def validate_candidate_schema(candidate: dict[str, Any]) -> None:
    properties = CANDIDATE_SCHEMA["properties"]
    if set(candidate) != set(CANDIDATE_SCHEMA["required"]):
        raise PromptLoopError("candidate did not contain the exact strict-schema fields")
    for field in ("target_description", "sam_concept", "question_if_ambiguous"):
        value = candidate[field]
        rule = properties[field]
        if not isinstance(value, str):
            raise PromptLoopError(f"candidate field {field} was not a string")
        if len(value) < rule.get("minLength", 0) or len(value) > rule.get("maxLength", 80):
            raise PromptLoopError(f"candidate field {field} violated its length contract")
    for field in ("preserve_concepts", "affected_concepts", "evidence"):
        value = candidate[field]
        rule = properties[field]
        if not isinstance(value, list) or len(value) > rule["maxItems"]:
            raise PromptLoopError(f"candidate field {field} violated its array contract")
        if any(not isinstance(item, str) for item in value):
            raise PromptLoopError(f"candidate field {field} contained a non-string")
    if type(candidate["ambiguous"]) is not bool:
        raise PromptLoopError("candidate field ambiguous was not boolean")
    confidence = candidate["confidence"]
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise PromptLoopError("candidate field confidence was outside 0..1")


def build_generation_payload(
    *,
    image_data: str,
    user_request: str,
    feedback: dict[str, Any] | None,
    model: str,
) -> dict[str, Any]:
    prompt = f"""<task>
Create the candidate object-removal prompt from Frame 1.
</task>
<user_request>
{user_request.strip()}
</user_request>
"""
    if feedback is not None:
        prompt += f"""<critic_feedback>
{canonical_json(feedback)}
</critic_feedback>
Revise only to address failed criteria. Preserve fields already supported by visual evidence.
"""
    prompt += """<sam_boundary>
The `sam_concept` field must contain one short concrete noun phrase suitable for SAM 3.1. Never place a command, question, negation, boolean operator, or multiple concepts in this field.
</sam_boundary>
"""
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        },
    ]
    return {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ObjectRemovalPrompt",
                "strict": True,
                "schema": CANDIDATE_SCHEMA,
            },
        },
        "reasoning_effort": "minimal",
        "temperature": 0.0,
        "max_tokens": 1200,
    }


def meta_chat(
    *, api_key: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{META_BASE_URL}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", "replace")
        failure = PromptLoopError(f"Meta API returned HTTP {error.code}: {detail}")
        try:
            raw = json.loads(detail)
        except json.JSONDecodeError:
            raw = None
        usage = raw.get("usage") if isinstance(raw, dict) else None
        failure.details = {"api_partial": {"raw": raw, "usage": usage}}
        raise failure from error
    except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError) as error:
        failure = PromptLoopError(f"Meta API request failed: {error}")
        failure.details = {"api_partial": {"raw": None, "usage": None}}
        raise failure from error
    try:
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
    except (KeyError, IndexError, TypeError) as error:
        failure = PromptLoopError("Meta API response lacked choices/content/usage")
        failure.details = {
            "api_partial": {
                "raw": result,
                "usage": result.get("usage") if isinstance(result, dict) else None,
            }
        }
        raise failure from error
    if not isinstance(content, str):
        failure = PromptLoopError("Meta API content was not text")
        failure.details = {"api_partial": {"raw": result, "usage": usage}}
        raise failure
    try:
        parsed = parse_json_object(content)
        validate_candidate_schema(parsed)
    except PromptLoopError as error:
        error.details = {"api_partial": {"raw": result, "usage": usage}}
        raise
    return parsed, {"raw": result, "usage": usage}


def critique_prompt(user_request: str, candidate: dict[str, Any]) -> str:
    criteria = "\n".join(f"- {name}" for name in CRITERIA)
    return f"""<task>
Critique the candidate object-removal prompt against Frame 1.
</task>
<user_request>
{user_request.strip()}
</user_request>
<critic_rubric>
{criteria}
</critic_rubric>
<candidate>
{canonical_json(candidate)}
</candidate>
<output_contract>
Return one JSON object only:
{{"passed":true,"failed_criteria":[],"corrected_sam_concept":null,"confidence":0.0,"reason":"...","visual_evidence":["..."]}}
Every item in failed_criteria must be one of the rubric names. Use null for corrected_sam_concept unless a correction is required.
</output_contract>
"""


def critique_request_digest(
    *, image_bytes: bytes, user_request: str, candidate: dict[str, Any]
) -> str:
    return digest(
        {
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "prompt": critique_prompt(user_request, candidate),
        }
    )


def run_critic(
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
    if endpoint is not None:
        parsed_endpoint = urlparse(endpoint)
        endpoint_host = (parsed_endpoint.hostname or "").lower()
        try:
            endpoint_ip = ipaddress.ip_address(endpoint_host) if endpoint_host else None
        except ValueError:
            endpoint_ip = None
        local_endpoint = endpoint_host == "localhost" or (
            endpoint_ip is not None and endpoint_ip.is_loopback
        )
        if parsed_endpoint.scheme == "http" and not local_endpoint:
            raise PromptLoopError("remote critic endpoint must use HTTPS")
        if not local_endpoint and endpoint_host not in allowed_hosts:
            raise PromptLoopError(
                f"critic endpoint host is not allowlisted: {endpoint_host!r}; "
                "pass --allow-critic-host to authorize it"
            )
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": critique_prompt(user_request, candidate)},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                },
            ],
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if critic_api_key:
            headers["Authorization"] = f"Bearer {critic_api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                outer = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", "replace")
            raise PromptLoopError(
                f"critic endpoint returned HTTP {error.code}: {detail}"
            ) from error
        except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError) as error:
            raise PromptLoopError(f"critic endpoint request failed: {error}") from error
    else:
        validate_critic_command(command)
        with tempfile.TemporaryDirectory(prefix="prompt-critique-") as temporary:
            stable_image = Path(temporary) / f"image{image.suffix.lower()}"
            stable_image.write_bytes(image_bytes)
            try:
                completed = subprocess.run(
                    [
                        command,
                        "--json",
                        "--model",
                        model,
                        "--temperature",
                        "0.0",
                        "--max-tokens",
                        "2000",
                        "--retries",
                        "0",
                        "--timeout",
                        str(max(1, math.ceil(timeout))),
                        "image",
                        str(stable_image),
                        f"{CRITIC_SYSTEM}\n\n{critique_prompt(user_request, candidate)}",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout + 5,
                )
            except subprocess.TimeoutExpired as error:
                raise PromptLoopError(f"critic command timed out: {error}") from error
            except (OSError, UnicodeDecodeError) as error:
                raise PromptLoopError(f"critic command could not run: {error}") from error
        if completed.returncode != 0:
            raise PromptLoopError(
                f"critic command failed ({completed.returncode}): "
                f"{completed.stderr.strip()[-1000:]}"
            )
        try:
            outer = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise PromptLoopError(f"critic response was not JSON: {error}") from error
    try:
        content = outer["choices"][0]["message"]["content"]
        usage = outer.get("usage", {})
    except (KeyError, IndexError, TypeError) as error:
        raise PromptLoopError(f"critic response lacked JSON content: {error}") from error
    if not isinstance(content, str):
        raise PromptLoopError("critic content was not text")
    parsed = parse_json_object(content)
    validate_critique(parsed)
    return parsed, {"raw": outer, "usage": usage}


def validate_critic_command(command: str) -> None:
    """Reject a critic command that is neither an allowlisted name nor a real binary."""
    if command in ALLOWED_CRITIC_COMMANDS:
        return
    if os.path.isabs(command) and os.path.isfile(command) and os.access(command, os.X_OK):
        return
    raise PromptLoopError(
        "critic command must be an allowlisted critic binary or an absolute path "
        f"to an executable file: {command!r}"
    )


def validate_critique(parsed: dict[str, Any]) -> None:
    required = {
        "passed",
        "failed_criteria",
        "corrected_sam_concept",
        "confidence",
        "reason",
        "visual_evidence",
    }
    if set(parsed) != required:
        raise PromptLoopError("critic response did not contain the exact output contract")
    if type(parsed["passed"]) is not bool:
        raise PromptLoopError("critic response lacked boolean passed")
    failed = parsed["failed_criteria"]
    if not isinstance(failed, list) or any(item not in CRITERIA for item in failed):
        raise PromptLoopError("critic failed_criteria contained an unknown rubric name")
    if parsed["passed"] != (not failed):
        raise PromptLoopError("critic passed flag contradicts failed_criteria")
    corrected = parsed["corrected_sam_concept"]
    if corrected is not None and (
        not isinstance(corrected, str) or not is_bare_noun_phrase(corrected)
    ):
        # The correction is advisory; invalid wording must not kill a paid loop.
        parsed["corrected_sam_concept"] = None
    confidence = parsed["confidence"]
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise PromptLoopError("critic confidence was outside 0..1")
    if not isinstance(parsed["reason"], str) or not parsed["reason"].strip():
        raise PromptLoopError("critic response lacked a reason")
    evidence = parsed["visual_evidence"]
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        raise PromptLoopError("critic response lacked visual evidence")


def _partial_generation_record(
    *,
    error: PromptLoopError,
    model: str,
    started: float,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    partial = error.details.get("api_partial") if error.details is not None else None
    if not isinstance(partial, dict):
        return None
    raw = partial.get("raw")
    usage = partial.get("usage")
    raw_content: str | None = None
    if isinstance(raw, dict):
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if isinstance(content, str):
            raw_content = content
    parsed_output: dict[str, Any] | None = None
    if raw_content is not None:
        try:
            candidate = parse_json_object(raw_content)
            validate_candidate_schema(candidate)
        except PromptLoopError:
            candidate = None
        if candidate is not None:
            parsed_output = candidate
    served_model = raw.get("model") if isinstance(raw, dict) else None
    return {
        "kind": "generation",
        "model": served_model or model,
        "requested_model": model,
        "served_model": served_model,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "response_id": raw.get("id") if isinstance(raw, dict) else None,
        "request_sha256": digest(payload),
        "output_sha256": digest(parsed_output) if parsed_output is not None else None,
        "raw_output": raw_content,
        "raw_output_sha256": (
            hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            if raw_content is not None
            else None
        ),
        "usage": usage if isinstance(usage, dict) else {},
        "output": parsed_output,
        "error": str(error),
    }


def build_meta_ledger_fragment(result: dict[str, Any]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for iteration in result.get("iterations", []):
        generation = iteration.get("generation")
        if not isinstance(generation, dict):
            continue
        usage = generation.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        modeled_debit: float | None = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            modeled_debit = round(
                prompt_tokens / 1_000_000 * META_INPUT_USD_PER_1M
                + completion_tokens / 1_000_000 * META_OUTPUT_USD_PER_1M,
                8,
            )
        calls.append(
            {
                "provider": "Meta Model API",
                "model": generation.get("model"),
                "requested_model": generation.get("requested_model", generation.get("model")),
                "served_model": (
                    generation["served_model"]
                    if "served_model" in generation
                    else generation.get("model")
                ),
                "response_id": generation.get("response_id"),
                "request_sha256": generation.get("request_sha256"),
                "output_sha256": generation.get("output_sha256"),
                "usage": usage,
                "elapsed_seconds": generation.get("elapsed_seconds"),
                "modeled_debit": modeled_debit,
                "pricing_basis": {
                    "version": META_PRICING_VERSION,
                    "as_of": META_PRICING_AS_OF,
                    "input_usd_per_1m_tokens": META_INPUT_USD_PER_1M,
                    "output_usd_per_1m_tokens": META_OUTPUT_USD_PER_1M,
                },
            }
        )
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "sam_vocab_version": SAM_VOCAB_VERSION,
        "model_api_calls": calls,
        "non_claim": "Merge into the tracked spend ledger; monetary debit is modeled unless Billing confirms it.",
    }


def _record(
    *,
    kind: str,
    model: str,
    started: float,
    parsed: dict[str, Any],
    metadata: dict[str, Any],
    request_digest: str,
) -> dict[str, Any]:
    usage = metadata.get("usage", {})
    raw = metadata.get("raw", {})
    served_model = raw.get("model") if isinstance(raw, dict) else None
    return {
        "kind": kind,
        "model": served_model or model,
        "requested_model": model,
        "served_model": served_model,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "response_id": raw.get("id") if isinstance(raw, dict) else None,
        "request_sha256": request_digest,
        "output_sha256": digest(parsed),
        "usage": usage,
        "output": parsed,
    }


def validate_max_iterations(max_iterations: int) -> int:
    if not 1 <= max_iterations <= MAX_ITERATIONS:
        raise PromptLoopError(f"max iterations must be in 1..{MAX_ITERATIONS}")
    return max_iterations


def parse_allowed_hosts(values: Sequence[str] | None) -> frozenset[str]:
    """Collect the explicit critic-host allowlist from flags and the environment."""
    hosts: set[str] = set()
    for value in values or ():
        for host in value.split(","):
            cleaned = host.strip().lower()
            if cleaned:
                hosts.add(cleaned)
    for host in os.environ.get(CRITIC_ALLOWED_HOSTS_ENV, "").split(","):
        cleaned = host.strip().lower()
        if cleaned:
            hosts.add(cleaned)
    return frozenset(hosts)


def run_loop(
    *,
    image: Path,
    user_request: str,
    api_key: str,
    meta_model: str,
    critic_command: str,
    critic_model: str,
    critic_endpoint: str | None,
    critic_api_key: str,
    timeout: float,
    max_iterations: int,
    allowed_hosts: Sequence[str] = (),
) -> dict[str, Any]:
    image_bytes = b""
    image_digest: str | None = None
    feedback: dict[str, Any] | None = None
    iterations: list[dict[str, Any]] = []
    final_candidate: dict[str, Any] | None = None
    final_critique: dict[str, Any] | None = None

    try:
        image_bytes, image_type = read_image(image)
        image_data = image_data_url(image_bytes, image_type)
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        validate_max_iterations(max_iterations)
        for number in range(1, max_iterations + 1):
            payload = build_generation_payload(
                image_data=image_data,
                user_request=user_request,
                feedback=feedback,
                model=meta_model,
            )
            started = time.monotonic()
            try:
                candidate, metadata = meta_chat(api_key=api_key, payload=payload, timeout=timeout)
            except PromptLoopError as error:
                partial = _partial_generation_record(
                    error=error, model=meta_model, started=started, payload=payload
                )
                if partial is not None:
                    iterations.append({"number": number, "generation": partial})
                raise
            generation_record = _record(
                kind="generation",
                model=meta_model,
                started=started,
                parsed=candidate,
                metadata=metadata,
                request_digest=digest(payload),
            )
            iteration: dict[str, Any] = {"number": number, "generation": generation_record}
            iterations.append(iteration)
            started = time.monotonic()
            if not is_bare_noun_phrase(candidate.get("sam_concept")):
                critique = {
                    "passed": False,
                    "failed_criteria": ["sam_concept_is_bare_noun_phrase"],
                    "corrected_sam_concept": None,
                    "confidence": 1.0,
                    "reason": "Local deterministic validation rejected the SAM concept.",
                    "visual_evidence": [
                        "The sam_concept failed the local command, negation, boolean, spatial, or multi-concept boundary."
                    ],
                }
                critic_metadata = {
                    "raw": {"id": f"local-validation-{digest(critique)}"},
                    "usage": {},
                }
                critic_record_model = "local-validation"
            else:
                critique, critic_metadata = run_critic(
                    command=critic_command,
                    model=critic_model,
                    endpoint=critic_endpoint,
                    image=image,
                    image_data=image_data,
                    image_bytes=image_bytes,
                    user_request=user_request,
                    candidate=candidate,
                    critic_api_key=critic_api_key,
                    timeout=timeout,
                    allowed_hosts=allowed_hosts,
                )
                critic_record_model = critic_model
            critique_record = _record(
                kind="critique",
                model=critic_record_model,
                started=started,
                parsed=critique,
                metadata=critic_metadata,
                request_digest=critique_request_digest(
                    image_bytes=image_bytes,
                    user_request=user_request,
                    candidate=candidate,
                ),
            )
            iteration["critique"] = critique_record
            final_candidate = candidate
            final_critique = critique
            if critique["passed"]:
                break
            feedback = critique
    except (
        PromptLoopError,
        OSError,
        ValueError,
        OverflowError,
        subprocess.TimeoutExpired,
    ) as error:
        wrapped = error if isinstance(error, PromptLoopError) else PromptLoopError(str(error))
        details: dict[str, Any] = {
            "image_sha256": image_digest,
            "user_request": user_request,
            "iterations": iterations,
        }
        # A paid Meta call may have completed before the failure. Keep its
        # captured provider body so the receipt still proves what was billed.
        original = error.details if isinstance(error, PromptLoopError) else None
        if isinstance(original, dict) and original.get("api_partial") is not None:
            details["api_partial"] = original["api_partial"]
        wrapped.details = details
        raise wrapped from error

    assert final_candidate is not None
    assert final_critique is not None
    return {
        "status": "passed" if final_critique["passed"] else "failed_after_max_iterations",
        "iterations_used": len(iterations),
        "max_iterations": max_iterations,
        "sam_vocab_version": SAM_VOCAB_VERSION,
        "image_sha256": image_digest,
        "user_request": user_request,
        "iterations": iterations,
        "final_candidate": final_candidate,
        "final_critique": final_critique,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    parser.add_argument("--critic-command", default="zai-vision")
    parser.add_argument("--critic-model")
    parser.add_argument("--critic-endpoint")
    parser.add_argument(
        "--allow-critic-host",
        action="append",
        default=None,
        help=(
            "Authorize one remote critic endpoint host. Repeatable. "
            f"Also read from {CRITIC_ALLOWED_HOSTS_ENV}."
        ),
    )
    parser.add_argument("--critic-api-key-env", default="GLM_CRITIC_API_KEY")
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args(argv)
    if not args.request.strip():
        parser.error("--request must contain text")
    try:
        validate_max_iterations(args.max_iterations)
    except PromptLoopError as error:
        parser.error(str(error))
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    if args.critic_endpoint is not None and not args.critic_endpoint.startswith(
        ("http://", "https://")
    ):
        parser.error("--critic-endpoint must use http:// or https://")
    if args.critic_model is None:
        args.critic_model = DEFAULT_CRITIC_MODEL
    if args.ledger_output is None:
        args.ledger_output = args.output.with_suffix(".ledger.json")
    args.allowed_hosts = parse_allowed_hosts(args.allow_critic_host)
    return args


def write_failure_outputs(
    args: argparse.Namespace, error_text: str, details: dict[str, Any] | None = None
) -> None:
    """Write the failure receipt and its Meta ledger fragment."""
    failure: dict[str, Any] = {"status": "failed", "error": error_text}
    if details:
        failure.update(details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.ledger_output is not None:
        args.ledger_output.parent.mkdir(parents=True, exist_ok=True)
        args.ledger_output.write_text(
            json.dumps(build_meta_ledger_fragment(failure), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"missing {args.api_key_env}", file=sys.stderr)
        write_failure_outputs(args, f"missing {args.api_key_env}")
        return 2
    critic_api_key = os.environ.get(args.critic_api_key_env, "").strip()
    try:
        result = run_loop(
            image=args.image,
            user_request=args.request,
            api_key=api_key,
            meta_model=args.meta_model,
            critic_command=args.critic_command,
            critic_model=args.critic_model,
            critic_endpoint=args.critic_endpoint,
            critic_api_key=critic_api_key,
            timeout=args.timeout,
            max_iterations=args.max_iterations,
            allowed_hosts=args.allowed_hosts,
        )
    except (
        PromptLoopError,
        OSError,
        ValueError,
        OverflowError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"prompt critique loop failed: {error}", file=sys.stderr)
        details = error.details if isinstance(error, PromptLoopError) else None
        write_failure_outputs(args, str(error), details)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.ledger_output is not None:
        args.ledger_output.parent.mkdir(parents=True, exist_ok=True)
        args.ledger_output.write_text(
            json.dumps(build_meta_ledger_fragment(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        canonical_json(
            {
                "status": result["status"],
                "iterations_used": result["iterations_used"],
                "final_candidate": result["final_candidate"],
                "final_critique": result["final_critique"],
            }
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
