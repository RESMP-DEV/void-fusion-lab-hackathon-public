#!/usr/bin/env python3
# ruff: noqa: UP031, E501
# Staging note: the remaining % formatting is in internal messages and
# prints only. The prompt assembly concatenates the item data, so a
# percent sign, a brace, or a newline in a scene reaches the model
# verbatim.
"""Model-agnostic quadmask critique harness (MiniMax loop 1).

Design rule (user order 2026-10-27 19:49): never design around one model.
The MODEL IS CONFIGURATION: provider, model_id, endpoint style, token
budget and prompt quirks live in providers.json as DATA. This file holds
only model-free loop logic: item loading, the fixed findings schema,
prompt assembly (schema-first), one parse-fail retry (12,000 tokens +
brevity instruction), the UNSCORED rule (an item with no parsed findings
object is UNSCORED, never PASS), scoring against expected answers, and
model-use ledger appends.

Fixed findings schema (same as the original critique loop):
  {item_id, target_present, target_fully_covered, wrong_subject,
   kept_subject_touched, fit: too_tight|good|too_loose|not_applicable,
   findings: [{type, severity, location, detail}],
   affected_assessment, verdict: PASS|WARN|FAIL, one_line}

Reliability fixes baked in as adapter defaults:
  1. FRAME-0 RULE in every quad prompt: frame 0 has NO affected ring by
     contract; never penalise its absence. (The earlier critique-full frame-0
     WARN lean was a prompt defect, not a mask defect.)
  2. Parse-fail retry: 12,000 tokens + brevity instruction, one retry.
  3. Schema-first prompt order: the JSON contract is stated BEFORE the
     scene and judging instructions, so the model plans the answer
     object first (reasoning-budget guard).

Usage:
  python3 harness.py --config providers.json --provider minimax \
      --items items_proof.json --out results-minimax.json \
      --ledger ../model-use-ledger.jsonl [--max-calls 30] [--only id1,id2] \
      --allowed-env-dir /path/to/secrets

Keys are read at runtime from each provider's env_file on this host and
are never printed or written anywhere. The env_file value is expanded
before it is opened: $VAR / ${VAR} placeholders and ~ user paths are
resolved, so the example config's "$MINIMAX_ENV_FILE" resolves to the
operator's real env file. The expanded path must then resolve inside a
directory the operator allowlisted (--allowed-env-dir, repeatable, or
the CRITIQUE_HARNESS_ALLOWED_ENV_DIRS environment variable); a path
outside that allowlist is refused before the file is read, and no
allowlist means no env file is read at all.

Everything that can fail is checked before the first paid call: the
provider object against the documented config schema (including a
required auth that matches the endpoint style, and positive
temperature and token budgets), the item set (required fields, legend,
unique ids, image files, --only ids), the item images (each one must
decode as an image, not merely exist), and the key file.
The item set is a JSON list of {item_id, image, legend: quad|binary,
scene, stats}; the optional fields are set and expected.
The results file is written atomically after every item, so a crash
never leaves a truncated results file.

The results file records no raw model text: a response tail is never
written to it, only a digest and a length, so an unreviewed model
output cannot leak a credential or personal content into a tracked
artifact. The parsed findings object is the deliverable of the run and
is kept. Transport failures (a refused connection, a DNS failure, a
timeout) are recorded as error records like an HTTP error, so one bad
call never ends the run.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import secrets
import urllib.error
import urllib.request
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

from PIL import Image

CST = timezone(timedelta(hours=8))
FINDING_TYPES = [
    "hole",
    "missing_part",
    "wrongly_included",
    "wrong_subject",
    "kept_subject_damage",
    "mask_too_small",
    "mask_too_large",
    "mask_shifted",
    "boundary_misaligned",
    "empty_mask_subject_present",
    "other",
]
FRAME0_RULE = (
    "FRAME-0 RULE (contract): frame 0 of a quadmask sequence has NO "
    "affected (cyan, class 127) ring by contract. If this item is "
    "frame 0, the absence of a cyan ring is CORRECT; never "
    "penalise it and never report it as a finding."
)
BREVITY = (
    "IMPORTANT: reason briefly (a few short sentences at most), then "
    "output the JSON object. Do not write long analysis."
)

SCHEMA_BLOCK = """Respond with ONE JSON object and no other text. The object must have exactly these keys:
{"item_id": {item_id},
 "target_present": true/false,
 "target_fully_covered": true/false,
 "wrong_subject": true/false,
 "kept_subject_touched": true/false,
 "fit": "too_tight" | "good" | "too_loose",
 "findings": [{"type": "hole|missing_part|wrongly_included|wrong_subject|kept_subject_damage|mask_too_small|mask_too_large|mask_shifted|boundary_misaligned|empty_mask_subject_present|other", "severity": "low|medium|high|critical", "location": "where in the frame", "detail": "what exactly is wrong"}],
 "affected_assessment": "plausible" | "too_wide" | "too_narrow" | "absent_but_needed" | "not_applicable",
 "verdict": "PASS" | "WARN" | "FAIL",
 "one_line": "one sentence summary"}
Use an empty findings list only when the mask is truly clean. WARN = minor defect, FAIL = the mask would visibly damage the removal."""

LEGEND_QUAD = """You are a strict quality critic for video object-removal masks (quadmasks).
Colour legend of the overlay: RED = class 0 REMOVE (the subject to delete), ORANGE = class 63 OVERLAP, CYAN = class 127 AFFECTED (pixels the removal may disturb, e.g. shadows/contact), uncoloured = class 255 KEEP (must stay untouched).
"""

LEGEND_BINARY = """You are a strict quality critic for video segmentation masks.
The overlay shows ONE binary mask in RED over the video frame (red = the region selected for removal). There are no other mask classes in this image."""

TAIL_ITEM_PREFIX = "Item: "
TAIL_ITEM_MIDDLE = ". Mask pixel counts this frame (class: pixels): "
TAIL_JUDGE = """.
Judge ONLY the mask quality in this frame. Be specific: if part of the target is uncovered, name the body part; if something that must be kept is covered, name it."""

# The one verdict-class contract shared with scripts/critique/score.py:
# "defect" is the only defect class. Every other class ("ok", an unknown
# class, a legacy bare string) is the clean branch.
DEFECT_VERDICT_CLASS = "defect"

# Item contract: the harness reads these fields from every item and
# refuses the item set before the first paid call when one is missing.
REQUIRED_ITEM_FIELDS = ("item_id", "image", "legend", "scene", "stats")
VALID_LEGENDS = ("quad", "binary")

# Provider-config schema (configs/critique-providers.example.json states
# the same set). The harness enforces it so a config that drifts from the
# documented schema fails before any paid call.
REQUIRED_PROVIDER_KEYS = (
    "provider",
    "model_id",
    "base_url",
    "endpoint_style",
    "chat_path",
    "env_file",
    "auth",
    "max_tokens",
)
ENDPOINT_STYLES = {"openai-compat": "bearer", "anthropic-compat": "x-api-key"}

# Token and sampling defaults, applied when the config leaves them out.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 8000
DEFAULT_RETRY_MAX_TOKENS = 12000

# The operator-supplied allowlist for env-file paths. The expanded path
# must resolve inside one of these directories; anything else is refused
# before the file is read.
ALLOWED_ENV_DIRS_ENV = "CRITIQUE_HARNESS_ALLOWED_ENV_DIRS"
UNRESOLVED_ENV_PLACEHOLDER = re.compile(r"\$(?:\{[A-Z_][A-Z0-9_]*\}|[A-Z_][A-Z0-9_]*)")

# Every key the schema block asks a findings object to carry. The parser
# checks all of them, so a response that drops one (target_present was
# the one that was never checked) is a parse failure, not a partial
# verdict.
REQUIRED_FINDINGS_KEYS = (
    "item_id",
    "target_present",
    "target_fully_covered",
    "wrong_subject",
    "kept_subject_touched",
    "fit",
    "findings",
    "affected_assessment",
    "verdict",
    "one_line",
)

# The findings keys that must hold a JSON boolean.
BOOLEAN_FINDINGS_KEYS = (
    "target_present",
    "target_fully_covered",
    "wrong_subject",
    "kept_subject_touched",
)

# The findings keys that must hold a JSON string.
STRING_FINDINGS_KEYS = ("fit", "affected_assessment", "one_line")

# The agreement value recorded when the expected answer itself is
# broken. It is never a clean verdict: an input error is scored as
# nothing and reported.
INPUT_ERROR = "INPUT_ERROR"


def expand_env_path(path, allowed_dirs=None):
    """Expand a configured env-file path: $VAR / ${VAR} placeholders and ~.

    The provider config ships placeholders such as "$MINIMAX_ENV_FILE", and
    the operators may write a path that starts with a tilde. Without
    expansion the harness opened a literal file named "$MINIMAX_ENV_FILE"
    and failed on the first paid call, so the placeholder never reached the
    key.

    When the operator supplies an allowlist of directories, the expanded
    path must resolve inside one of them. A path outside the allowlist
    is refused here, before the file is opened, so a config that points
    at an arbitrary file cannot make the harness read it.
    """
    if not path:
        return ""
    expanded = os.path.expanduser(os.path.expandvars(path))
    if not allowed_dirs:
        return expanded
    if UNRESOLVED_ENV_PLACEHOLDER.search(expanded):
        raise SystemExit("env_file placeholder is not set: " + expanded)
    resolved = os.path.realpath(expanded)
    if not within_allowed_dirs(resolved, allowed_dirs):
        raise SystemExit("env file path is outside the allowed directories: " + resolved)
    return resolved


def within_allowed_dirs(resolved_path, allowed_dirs):
    """True when a resolved path sits inside one of the allowed directories."""
    for directory in allowed_dirs:
        base = os.path.realpath(os.path.expanduser(os.path.expandvars(directory)))
        if resolved_path == base or resolved_path.startswith(base + os.sep):
            return True
    return False


def load_key(env_file, allowed_dirs=None):
    """Read the provider key from the env file named by the config.

    The path is expanded first (variables and user paths) and constrained
    to the operator's allowlist when one is supplied, then checked: an
    unset placeholder, a missing file, or a file without a KEY/TOKEN
    variable fails closed before any paid call.
    """
    path = expand_env_path(env_file, allowed_dirs)
    if not path:
        raise SystemExit("no env_file configured for this provider")
    if UNRESOLVED_ENV_PLACEHOLDER.search(path):
        raise SystemExit("env_file placeholder is not set: " + path)
    if not os.path.isfile(path):
        raise SystemExit("env file missing: " + path)
    with open(path) as handle:
        lines = [ln.strip() for ln in handle if ln.strip() and not ln.strip().startswith("#")]
    if lines and all("=" not in ln for ln in lines):
        return lines[0].strip('"').strip("'")  # bare-key env file
    for line in lines:
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if "KEY" in k.upper() or "TOKEN" in k.upper():
            return v.strip().strip('"').strip("'")
    raise SystemExit("no KEY/TOKEN variable in " + path)


def validate_provider(provider_name, cfg):
    """Check the provider object against the documented config schema.

    configs/critique-providers.example.json states this schema; the
    harness enforces the same set, so a config that drifts from the
    documentation fails before any paid call instead of failing on the
    first request.
    """
    if not isinstance(cfg, dict):
        raise SystemExit("provider %s is not an object" % provider_name)
    missing = [key for key in REQUIRED_PROVIDER_KEYS if not cfg.get(key)]
    if missing:
        raise SystemExit(
            "provider %s is missing config key(s): %s" % (provider_name, ", ".join(missing))
        )
    style = cfg["endpoint_style"]
    if style not in ENDPOINT_STYLES:
        raise SystemExit("provider %s has unknown endpoint_style %r" % (provider_name, style))
    auth = cfg.get("auth")
    if auth != ENDPOINT_STYLES[style]:
        raise SystemExit(
            "provider %s declares auth %r but endpoint_style %r requires %r"
            % (provider_name, auth, style, ENDPOINT_STYLES[style])
        )
    if style == "anthropic-compat" and not cfg.get("anthropic_version"):
        raise SystemExit(
            "provider %s uses anthropic-compat and must set anthropic_version" % provider_name
        )
    validate_positive_number(
        provider_name, "temperature", cfg.get("temperature", DEFAULT_TEMPERATURE)
    )
    validate_positive_int(provider_name, "max_tokens", cfg["max_tokens"])
    validate_positive_int(
        provider_name, "retry_max_tokens", cfg.get("retry_max_tokens", DEFAULT_RETRY_MAX_TOKENS)
    )


def validate_positive_number(provider_name, key, value):
    """Refuse a config value that is not a positive, finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            "provider %s has %s %r; a positive number is required" % (provider_name, key, value)
        )
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(
            "provider %s has %s %r; a positive number is required" % (provider_name, key, value)
        )


def validate_positive_int(provider_name, key, value):
    """Refuse a token budget that is not a positive whole number."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(
            "provider %s has %s %r; a positive integer is required" % (provider_name, key, value)
        )
    if value <= 0:
        raise SystemExit(
            "provider %s has %s %r; a positive integer is required" % (provider_name, key, value)
        )


def validate_items(items, only=None):
    """Validate the item set before any paid call.

    Every failure below is raised before the first model call: a missing
    field, an unknown legend, a duplicate item id, a missing image file,
    or an --only id that the items file does not hold would otherwise be
    discovered mid-run, after calls were already paid for.
    """
    if not isinstance(items, list):
        raise SystemExit("items file must hold a JSON list of items")
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SystemExit("item %d is not an object" % index)
        for field in REQUIRED_ITEM_FIELDS:
            if field not in item:
                raise SystemExit("item %d is missing required field %r" % (index, field))
        if item["legend"] not in VALID_LEGENDS:
            raise SystemExit("item %s has unknown legend %r" % (item["item_id"], item["legend"]))
        if item["item_id"] in seen:
            raise SystemExit("duplicate item_id %r" % item["item_id"])
        seen.add(item["item_id"])
        if not os.path.isfile(item["image"]):
            raise SystemExit("item %s image file is missing: %s" % (item["item_id"], item["image"]))
        if os.path.getsize(item["image"]) == 0:
            raise SystemExit("item %s image file is empty: %s" % (item["item_id"], item["image"]))
        try:
            with open(item["image"], "rb") as handle:
                if not handle.read(1):
                    raise SystemExit(
                        "item %s image file is not readable: %s" % (item["item_id"], item["image"])
                    )
        except OSError as error:
            raise SystemExit(
                "item %s image file is not readable: %s (%s)"
                % (item["item_id"], item["image"], type(error).__name__)
            ) from error
    if only:
        requested = set(only.split(","))
        missing = sorted(requested - seen)
        if missing:
            raise SystemExit(
                "requested item id(s) are not in the items file: " + ", ".join(missing)
            )


def validate_item_images(items):
    """Refuse an item image that is not a readable image.

    An image path that exists but does not decode (a truncated download,
    an HTML error page saved under a .png name) would otherwise be sent
    to the model as a broken payload and paid for. Every image is opened
    and verified here, before the first paid call.
    """
    for item in items:
        path = item["image"]
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("image has no pixels")
        except Exception as error:
            raise SystemExit(
                "item %s image is not a readable image: %s (%s)"
                % (item["item_id"], path, type(error).__name__)
            ) from error


def write_json_atomic(path, payload):
    """Write JSON atomically: temp file in the same directory, then rename.

    A crash mid-write must not leave a truncated results file behind: the
    previous complete file survives until the new one is renamed over it.

    The temporary name is not predictable: it carries 96 bits of random
    entropy and the file is created exclusively, so a name guessed in
    advance cannot be pre-created (as a symlink or a decoy) to capture
    the write. The temporary file is owner-only.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = ""
    handle_fd = -1
    for _ in range(5):
        candidate = os.path.join(directory, ".critique-" + secrets.token_hex(12) + ".tmp")
        try:
            handle_fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        tmp_path = candidate
        break
    if handle_fd == -1:
        raise SystemExit("could not create a temporary file in " + directory)
    try:
        with os.fdopen(handle_fd, "w") as handle:
            json.dump(payload, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def build_prompt(item):
    """Assemble the schema-first prompt for one item.

    The item data is concatenated, never percent-formatted: a scene that
    carries a percent sign, a brace, or a newline must reach the model
    exactly as the item set holds it. The item id is inserted as JSON,
    so a quote or a backslash in it cannot break the schema block.
    """
    schema = SCHEMA_BLOCK.replace("{item_id}", json.dumps(str(item["item_id"])))
    if item["legend"] == "quad":
        legend = LEGEND_QUAD + FRAME0_RULE
    else:
        legend = LEGEND_BINARY
    tail = (
        str(item["scene"])
        + "\n"
        + TAIL_ITEM_PREFIX
        + str(item["item_id"])
        + TAIL_ITEM_MIDDLE
        + str(item["stats"])
        + TAIL_JUDGE
    )
    # schema-first order: JSON contract, then legend/rules, then the item
    return schema + "\n\n" + legend + "\n" + tail


def call_openai(cfg, key, prompt, b64, max_tokens):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            ],
        }
    ]
    body = json.dumps(
        {
            "model": cfg["model_id"],
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": cfg.get("temperature", DEFAULT_TEMPERATURE),
        }
    ).encode()
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + cfg["chat_path"],
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    return request_json(req)


def call_anthropic(cfg, key, prompt, b64, max_tokens):
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    body = json.dumps(
        {
            "model": cfg["model_id"],
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": cfg.get("temperature", DEFAULT_TEMPERATURE),
        }
    ).encode()
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + cfg["chat_path"],
        data=body,
        headers={
            "x-api-key": key,
            "Content-Type": "application/json",
            "anthropic-version": cfg.get("anthropic_version", "2023-06-01"),
        },
    )
    return request_json(req)


def request_json(req, timeout=300):
    """Send one request and return (http status, body).

    A transport failure (a refused connection, a DNS failure, a reset,
    a timeout) is returned as an error record, exactly like an HTTP
    error: one unreachable call must not end the run or lose the
    attempts already paid for. The error record names the failure kind
    and the exception type only, never the endpoint, the key, or the
    response text.
    """
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, {"error_body": error.read().decode()[:1500], "error_kind": "http"}
    except urllib.error.URLError as error:
        return 0, {
            "error_body": "transport error: " + error_label(error),
            "error_kind": "transport",
        }
    except (TimeoutError, OSError) as error:
        return 0, {
            "error_body": "transport error: " + error_label(error),
            "error_kind": "transport",
        }
    except ValueError:
        return 0, {"error_body": "response body is not JSON", "error_kind": "decode"}


def error_label(error):
    """The exception type behind a transport failure, without its text."""
    reason = getattr(error, "reason", None)
    target = reason if isinstance(reason, BaseException) else error
    return type(target).__name__


def text_of(style, resp):
    try:
        if style == "anthropic-compat":
            return "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
        msg = resp["choices"][0]["message"]
        return msg.get("content") or ""
    except Exception:
        return ""


def usage_of(style, resp):
    if style == "anthropic-compat":
        u = resp.get("usage") or {}
        if not u:
            return None
        return {
            "prompt_tokens": u.get("input_tokens"),
            "completion_tokens": u.get("output_tokens"),
            "total_tokens": (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0),
        }
    return resp.get("usage")


def response_digest(text):
    """A stable digest of one model response.

    The digest identifies the response that was parsed without carrying
    any of its text, so the results file can prove which answer was
    judged while holding no model output.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]


def has_required_findings_keys(candidate):
    """True when a parsed object carries every key the schema block names."""
    return isinstance(candidate, dict) and all(key in candidate for key in REQUIRED_FINDINGS_KEYS)


def extract_findings_object(txt):
    """Find the findings object in a model response, deterministically.

    The schema-first prompt asks for one JSON object, but a response can
    carry prose, a reasoning object, a second object, or stray braces.
    The text is scanned for every JSON object it can decode, in document
    order, and the one that carries the whole findings schema is
    returned. When no object is complete, the object that carries the
    most schema keys is returned, so the parser can name the key that
    is missing instead of failing with a vague error. Nothing is
    guessed, and no response text is copied into an error message.
    """
    decoder = json.JSONDecoder()
    index = 0
    decoded = 0
    best = None
    best_score = 0
    while True:
        start = txt.find("{", index)
        if start == -1:
            break
        try:
            candidate, end = decoder.raw_decode(txt, start)
        except ValueError:
            index = start + 1
            continue
        decoded += 1
        if isinstance(candidate, dict):
            score = sum(1 for key in REQUIRED_FINDINGS_KEYS if key in candidate)
            if score > best_score:
                best = candidate
                best_score = score
                if has_required_findings_keys(candidate):
                    return candidate
        index = max(end, start + 1)
    if best is not None:
        return best
    if decoded:
        raise ValueError("no findings object in the response")
    raise ValueError("no JSON object in the response")


def parse_findings(txt):
    """Parse and validate one findings object.

    Every key the schema block requires is checked, target_present
    included: a response that omits one is a parse failure, so a partial
    verdict is never scored. The value types are checked too, so a
    string where the schema asks for a boolean is a parse failure as
    well.
    """
    obj = extract_findings_object(txt)
    for key in REQUIRED_FINDINGS_KEYS:
        if key not in obj:
            raise ValueError("missing key " + key)
    for key in BOOLEAN_FINDINGS_KEYS:
        if not isinstance(obj[key], bool):
            raise ValueError("key is not a boolean: " + key)
    for key in STRING_FINDINGS_KEYS:
        if not isinstance(obj[key], str):
            raise ValueError("key is not a string: " + key)
    if not isinstance(obj["item_id"], str) or not obj["item_id"]:
        raise ValueError("key is not a non-empty string: item_id")
    if not isinstance(obj["findings"], list):
        raise ValueError("key is not a list: findings")
    if obj["verdict"] not in ("PASS", "WARN", "FAIL"):
        raise ValueError("bad verdict")
    for fd in obj["findings"]:
        if not isinstance(fd, dict):
            raise ValueError("finding entry is not an object")
        if fd.get("type") not in FINDING_TYPES:
            fd["type"] = "other"
        if fd.get("severity") not in ("low", "medium", "high", "critical"):
            fd["severity"] = "medium"
    return obj


def is_flagged(fd):
    """Canonical "flagged" definition, shared with scripts/critique/score.py.

    An item is flagged when its verdict is WARN or FAIL, or its findings
    list is non-empty. The prompt contract states an empty findings list
    is used only when the mask is truly clean, so a PASS verdict that
    still carries findings is inconsistent and must not score as clean
    (F31: the two staged scorers defined flagged differently).
    """
    return fd["verdict"] in ("WARN", "FAIL") or len(fd.get("findings") or []) > 0


def expected_verdict_class(expected):
    """The one verdict-class contract, shared with scripts/critique/score.py.

    A dict carries {"verdict_class": ..., "planted_type": ...}; a legacy
    bare string is read as the class itself, so old runs still score.
    """
    if isinstance(expected, dict):
        return expected.get("verdict_class")
    return expected


def planted_type_of(expected):
    """The planted defect type carried by the expected answer, if any."""
    if isinstance(expected, dict):
        return expected.get("planted_type")
    return None


def score(rec, expected):
    """Model-free scoring vs the item's expected answer (if any).

    Only the defect class is a defect. The harness used to treat every
    class other than "ok" as a defect while scripts/critique/score.py
    treated only "defect" as one, so an unknown or legacy class scored
    differently in the two scorers; both now read the same contract.

    An expected answer that carries no verdict_class at all is an input
    error, not a clean verdict: it is recorded as one, so a broken item
    can never be counted as a mask the model judged correctly.
    """
    if expected is None:
        return {"scored": rec.get("findings") is not None, "agreement": None}
    if isinstance(expected, dict) and "verdict_class" not in expected:
        return {
            "scored": False,
            "agreement": INPUT_ERROR,
            "error": "expected answer has no verdict_class",
        }
    if rec.get("findings") is None:
        return {"scored": False, "agreement": "UNSCORED"}
    fd = rec["findings"]
    flagged = is_flagged(fd)
    if expected_verdict_class(expected) != DEFECT_VERDICT_CLASS:
        return {"scored": True, "agreement": "MATCH" if not flagged else "FALSE_ALARM"}
    # expected a defect
    if not flagged:
        return {"scored": True, "agreement": "MISSED"}
    types = [f.get("type") for f in (fd.get("findings") or [])]
    want = planted_type_of(expected)
    return {
        "scored": True,
        "agreement": "DETECTED_TYPED" if want in types else "DETECTED_UNTYPED",
        "types": types,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default="")
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--allowed-env-dir",
        action="append",
        default=[],
        help=(
            "directory the provider env file must resolve inside; repeatable. "
            "Also read from " + ALLOWED_ENV_DIRS_ENV + "."
        ),
    )
    a = ap.parse_args()

    with open(a.config) as handle:
        providers = json.load(handle).get("providers", {})
    if a.provider not in providers:
        raise SystemExit("provider %s is not in the config file" % a.provider)
    cfg = providers[a.provider]
    if not cfg.get("model_id"):
        raise SystemExit(
            "provider %s has no model_id set (UNTESTED stub); refusing to call" % a.provider
        )
    # Config schema, item set, and key file are all checked before the
    # first paid call: a bad config, a malformed item set, or an --only id
    # the items file does not hold must not cost a call.
    validate_provider(a.provider, cfg)
    with open(a.items) as handle:
        items = json.load(handle)
    validate_items(items, a.only)
    if a.only:
        keep = set(a.only.split(","))
        items = [it for it in items if it["item_id"] in keep]
    # The item images must decode, not merely exist: a broken file would
    # be paid for as a broken payload.
    validate_item_images(items)
    allowed_dirs = list(a.allowed_env_dir)
    allowed_dirs.extend(
        part for part in os.environ.get(ALLOWED_ENV_DIRS_ENV, "").split(os.pathsep) if part
    )
    if not allowed_dirs:
        raise SystemExit(
            "no allowed env-file directory supplied: pass --allowed-env-dir or set "
            + ALLOWED_ENV_DIRS_ENV
        )
    key = load_key(cfg["env_file"], allowed_dirs)
    style = cfg["endpoint_style"]
    call = call_anthropic if style == "anthropic-compat" else call_openai

    calls = 0
    records = []
    # The ledger is opened once and closed deterministically, also when
    # the run fails part way through.
    ledger_context = open(a.ledger, "a") if a.ledger else nullcontext(None)
    with ledger_context as ledger:
        for it in items:
            with open(it["image"], "rb") as handle:
                b64 = base64.b64encode(handle.read()).decode()
            prompt = build_prompt(it)
            attempts = []
            findings = None
            for attempt, (mtoks, suffix) in enumerate(
                [
                    (cfg.get("max_tokens", DEFAULT_MAX_TOKENS), ""),
                    (cfg.get("retry_max_tokens", DEFAULT_RETRY_MAX_TOKENS), "\n" + BREVITY),
                ],
                start=1,
            ):
                if calls >= a.max_calls:
                    print("CALL CAP REACHED")
                    break
                p = prompt + suffix if suffix else prompt
                st, resp = call(cfg, key, p, b64, mtoks)
                calls += 1
                txt = text_of(style, resp)
                usg = usage_of(style, resp)
                # No model text is written to the results file: a digest
                # and a length are enough to trace a run, and neither can
                # carry a credential or personal content.
                att = {
                    "attempt": attempt,
                    "http": st,
                    "max_tokens": mtoks,
                    "usage": usg,
                    "response_digest": response_digest(txt),
                    "response_chars": len(txt),
                }
                try:
                    findings = parse_findings(txt)
                    att["parse"] = "OK"
                except Exception as e:
                    att["parse"] = "FAIL: " + str(e)[:120]
                attempts.append(att)
                if ledger:
                    ledger.write(
                        json.dumps(
                            {
                                "ts_cst": datetime.now(CST).isoformat(),
                                "lane": "minimax-loop1",
                                "provider": cfg["provider"],
                                "model_id": cfg["model_id"],
                                "endpoint_style": style,
                                "item_id": it["item_id"],
                                "attempt": attempt,
                                "max_tokens": mtoks,
                                "http": st,
                                "prompt_tokens": (usg or {}).get("prompt_tokens"),
                                "completion_tokens": (usg or {}).get("completion_tokens"),
                                "total_tokens": (usg or {}).get("total_tokens"),
                                "parse": att["parse"],
                                "cost": "provider plan quota; no per-call currency exposed",
                                "artifact": os.path.basename(a.out),
                            }
                        )
                        + "\n"
                    )
                    ledger.flush()
                print(
                    "%s attempt %d: HTTP %s %s (calls=%d)"
                    % (it["item_id"], attempt, st, att["parse"], calls),
                    flush=True,
                )
                if findings is not None:
                    break
            rec = {
                "item_id": it["item_id"],
                "set": it.get("set", ""),
                "provider": cfg["provider"],
                "model_id": cfg["model_id"],
                "endpoint_style": style,
                "attempts": attempts,
                "findings": findings,
                "status": "SCORED" if findings is not None else "UNSCORED",
            }
            rec["score"] = score(rec, it.get("expected"))
            records.append(rec)
            write_json_atomic(
                a.out,
                {
                    "provider": a.provider,
                    "model_id": cfg["model_id"],
                    "endpoint_style": style,
                    "calls_used": calls,
                    "records": records,
                },
            )
    print("TOTAL CALLS:", calls)
    for r in records:
        fd = r.get("findings") or {}
        print(
            "%s: %s verdict=%s agreement=%s"
            % (
                r["item_id"],
                r["status"],
                fd.get("verdict", "-"),  # ty: ignore[unresolved-attribute]
                (r.get("score") or {}).get("agreement"),  # ty: ignore[unresolved-attribute]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
