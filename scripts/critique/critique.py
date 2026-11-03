#!/usr/bin/env python3
# ruff: noqa: UP031, E501
# Formatting note: the percent-style formatting and the long prompt and data
# strings are kept verbatim; converting them is a follow-up cleanup.
"""Quadmask self-critique loop (vision critic, structured findings).

Critic: MiniMax-M3.1-Flash via the OpenAI-compatible route
(https://api.minimax.io/v1/chat/completions), key read at runtime from
the file named by the MINIMAX_ENV_FILE env var (never printed). GLM (Z.AI) is NOT used:
an earlier probe proved the Z.AI key unlocks no vision model (glm-4v-flash ->
400/1211) and generation is balance-blocked (429/1113); see
the GLM critique record and the critique-run record
(records held outside this repository).

Input : items.json in this directory, or any JSON
        list of {item_id, image, legend: quad|binary, scene, stats, set}.
        The loop refuses the item set before the first paid call when a
        field is missing, the legend is unknown, an item id repeats, an
        image file is absent, or --only names a set the file does not
        hold.
Output: results.json (this directory) — one record per item with the parsed
        findings object, per-call token usage, and a running call count.
        Written incrementally after every call so a crash loses nothing.
        Items already scored are skipped only when the stored
        input_fingerprint matches the current item (image bytes +
        prompt inputs, including the prompt suffix and the token
        budget); a changed image or prompt is re-scored, and a record
        with no fingerprint (written before this check) is never reused.
        The fingerprint of a record is computed from the same single read
        of the image bytes that is sent to the model, so the stored
        fingerprint always describes the bytes the verdict answers.

Fixed findings schema (no free-text-only verdicts):
  {item_id, target_present, target_fully_covered, wrong_subject,
   kept_subject_touched, fit: too_tight|good|too_loose|not_applicable,
   findings: [{type, severity: low|medium|high|critical, location, detail}],
   affected_assessment: plausible|too_wide|too_narrow|absent_but_needed|not_applicable,
   verdict: PASS|WARN|FAIL, one_line}

Failure mode on record: on some items the model's reasoning consumes the
whole token budget and the final JSON is never emitted (content empty,
completion_tokens == max_tokens). Retry those items with --max-tokens 12000
and --prompt-suffix "IMPORTANT: reason briefly, then output the JSON object."
(RETRY_MAX_TOKENS and RETRY_PROMPT_SUFFIX hold exactly those two values).

A retry run does NOT reuse the prior verdicts. The documented retry changes
the prompt (a brevity suffix) and the token budget, so every prior verdict
answers a different question and is re-made: a verdict judged under one
prompt is never reused under another. --max-calls caps the calls made by ONE
invocation, so a retry run gets its own budget instead of inheriting an
exhausted cap; results.json records the cumulative total as calls_used and
this run's calls as run_calls_used.

Everything that can fail is checked before the first paid call: the item
set (required fields, legend, unique ids, image files, --only sets) and
the key file. The key file must resolve inside an operator-supplied
directory selected by repeatable --allowed-env-dir arguments or
CRITIQUE_HARNESS_ALLOWED_ENV_DIRS. The results file is written atomically
after every item, so a crash never leaves a truncated results file behind.

Usage:
  python3 critique.py [--items items.json] [--max-calls 40] [--only set1,set2]
                      [--max-tokens 8000] [--prompt-suffix ""]
Cost: MiniMax Coding Plan quota draw (no per-call currency exposed);
~1,800-3,200 tokens per call at max_tokens 8000. A 38-item pass ~= 38 calls.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "https://api.minimax.io/v1"
MODEL = "MiniMax-M3.1-Flash"
ENV_FILE = os.environ.get("MINIMAX_ENV_FILE", "")
ALLOWED_ENV_DIRS_ENV = "CRITIQUE_HARNESS_ALLOWED_ENV_DIRS"
UNRESOLVED_ENV_PLACEHOLDER = re.compile(r"\$(?:\{[A-Z_][A-Z0-9_]*\}|[A-Z_][A-Z0-9_]*)")

# The documented recovery run for items whose JSON was never emitted.
RETRY_MAX_TOKENS = 12000
RETRY_PROMPT_SUFFIX = "IMPORTANT: reason briefly, then output the JSON object."

# Item contract: the loop reads these fields from every item and refuses
# the item set before the first paid call when one is missing.
REQUIRED_ITEM_FIELDS = ("item_id", "image", "legend", "scene", "stats", "set")
VALID_LEGENDS = ("quad", "binary")

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
BOOLEAN_FINDINGS_KEYS = (
    "target_present",
    "target_fully_covered",
    "wrong_subject",
    "kept_subject_touched",
)
STRING_FINDINGS_KEYS = ("fit", "affected_assessment", "one_line")
ALLOWED_KEY_VARIABLES = {"MINIMAX_KEY", "MINIMAX_TOKEN", "KEY", "TOKEN"}

PROMPT_QUAD = """You are a strict quality critic for video object-removal masks (quadmasks).
Colour legend of the overlay: RED = class 0 REMOVE (the subject to delete),
ORANGE = class 63 OVERLAP, CYAN = class 127 AFFECTED (pixels the removal may disturb, e.g. shadows/contact),
uncoloured = class 255 KEEP (must stay untouched).
{scene}
Item: {item_id}. Mask pixel counts this frame (class: pixels): {stats}.
Judge ONLY the mask quality in this frame. Be specific: if part of the target is uncovered,
name the body part; if something that must be kept is covered, name it.
Answer with one JSON object and no other text:
{{"item_id": "{item_id}",
 "target_present": true/false,
 "target_fully_covered": true/false,
 "wrong_subject": true/false,
 "kept_subject_touched": true/false,
 "fit": "too_tight" | "good" | "too_loose",
 "findings": [{{"type": "hole|missing_part|wrongly_included|wrong_subject|kept_subject_damage|mask_too_small|mask_too_large|mask_shifted|boundary_misaligned|empty_mask_subject_present|other", "severity": "low|medium|high|critical", "location": "where in the frame", "detail": "what exactly is wrong"}}],
 "affected_assessment": "plausible" | "too_wide" | "too_narrow" | "absent_but_needed" | "not_applicable",
 "verdict": "PASS" | "WARN" | "FAIL",
 "one_line": "one sentence summary"}}
Use an empty findings list only when the mask is truly clean. WARN = minor defect, FAIL = the mask would visibly damage the removal."""

PROMPT_BINARY = """You are a strict quality critic for video segmentation masks.
The overlay shows ONE binary mask in RED over the video frame (red = the region selected for removal).
There are no other mask classes in this image.
{scene}
Item: {item_id}. Mask pixel counts: {stats}.
Judge ONLY the mask quality in this frame. Be specific: if part of the target is uncovered,
name the body part; if something that must not be masked is covered, name it.
Answer with one JSON object and no other text:
{{"item_id": "{item_id}",
 "target_present": true/false,
 "target_fully_covered": true/false,
 "wrong_subject": true/false,
 "kept_subject_touched": true/false,
 "fit": "too_tight" | "good" | "too_loose",
 "findings": [{{"type": "hole|missing_part|wrongly_included|wrong_subject|kept_subject_damage|mask_too_small|mask_too_large|mask_shifted|boundary_misaligned|empty_mask_subject_present|other", "severity": "low|medium|high|critical", "location": "where in the frame", "detail": "what exactly is wrong"}}],
 "affected_assessment": "not_applicable",
 "verdict": "PASS" | "WARN" | "FAIL",
 "one_line": "one sentence summary"}}
Use an empty findings list only when the mask is truly clean. WARN = minor defect, FAIL = the mask would visibly damage the removal."""


def _prompt_template(item):
    """The prompt template this item is judged with."""
    return PROMPT_QUAD if item.get("legend") == "quad" else PROMPT_BINARY


def read_image_bytes(path):
    """Read the image bytes once and return them, or None when unreadable."""
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def item_fingerprint(item, prompt_suffix="", max_tokens=8000, image_bytes=None):
    """Hash everything a verdict depends on: image bytes + prompt inputs.

    The prompt suffix and the token budget are part of the judged input:
    a verdict produced under one prompt answers a different question than
    the same item under another prompt, so the two must never share a
    fingerprint.

    image_bytes lets the caller pass the exact bytes it sends to the
    model, so a recorded fingerprint can never describe a different read
    than the one the verdict answers. When it is None the image is read
    once here.

    Returns None when the image cannot be read; None never matches a
    stored fingerprint, so an unreadable image is never resumed.
    """
    if image_bytes is None:
        image_bytes = read_image_bytes(item.get("image", ""))
    if image_bytes is None:
        return None
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b"\0")
    h.update(
        json.dumps(
            {
                "item_id": item.get("item_id"),
                "legend": item.get("legend"),
                "model": MODEL,
                "prompt_template": _prompt_template(item),
                "scene": item.get("scene"),
                "stats": item.get("stats"),
                "prompt_suffix": prompt_suffix,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()


def resume_state(prev_records, items, prompt_suffix="", max_tokens=8000):
    """Split previous records into (reusable, carried) for this run.

    A record is reusable only when it holds a parsed findings object AND
    the stored input_fingerprint equals the current item's fingerprint for
    this run's prompt parameters. The fingerprint covers the image bytes,
    the item fields, the model, the prompt template, the prompt suffix,
    and the token budget, so a verdict judged under a different prompt is
    never reused here.
    (F28: keying resume on item_id alone silently reused a verdict for a
    changed image. F33: the documented retry changed every fingerprint
    and discarded every prior verdict. The round-2 repair keeps one
    fingerprint that covers the retry parameters too: a retry re-judges
    under its own prompt, and the per-invocation call budget keeps that
    retry bounded.)
    A stale record for a current item is dropped, not carried. A record
    for an item outside this run is carried unchanged: no verdict is
    reused for any current item from it.
    """
    current = {it["item_id"]: it for it in items}
    done = {}
    carried = []
    for rec in prev_records:
        if not rec.get("findings"):
            continue
        item = current.get(rec.get("item_id"))
        if item is None:
            carried.append(rec)
            continue
        fp = item_fingerprint(item, prompt_suffix, max_tokens)
        if fp is not None and rec.get("input_fingerprint") == fp:
            done[rec["item_id"]] = rec
            carried.append(rec)
            continue
    return done, carried


def expand_env_path(path):
    """Expand an env-file path: $VAR / ${VAR} placeholders and ~ .

    The operator writes the path into MINIMAX_ENV_FILE, sometimes with a
    leading tilde. Without expansion the loader opened a literal file
    named after the unexpanded path and failed at the first paid call.
    """
    if not path:
        return ""
    return os.path.expanduser(os.path.expandvars(path))


def within_allowed_dirs(path, allowed_dirs):
    """Whether a resolved path is the directory itself or lies beneath it."""
    resolved = Path(path).resolve()
    for directory in allowed_dirs:
        allowed = Path(os.path.expanduser(os.path.expandvars(directory))).resolve()
        if resolved == allowed or resolved.is_relative_to(allowed):
            return True
    return False


def load_key(env_file, allowed_dirs=None):
    """Read the key only from a path inside the operator's allowlist."""
    if not env_file:
        raise SystemExit("MINIMAX_ENV_FILE env var not set: no env file path given")
    path = expand_env_path(env_file)
    if UNRESOLVED_ENV_PLACEHOLDER.search(path):
        raise SystemExit("env file placeholder is not set: " + path)
    if not allowed_dirs:
        raise SystemExit(
            "no allowed env-file directory supplied: pass --allowed-env-dir or set "
            + ALLOWED_ENV_DIRS_ENV
        )
    if not within_allowed_dirs(path, allowed_dirs):
        raise SystemExit("env file path is outside the allowed directories: " + path)
    key_file = Path(path)
    if not key_file.is_file():
        raise SystemExit("env file missing: " + path)
    with key_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k in ALLOWED_KEY_VARIABLES:
                return v.strip().strip('"').strip("'")
    raise SystemExit("no MINIMAX_KEY/MINIMAX_TOKEN/KEY/TOKEN variable in env file")


def validate_items(items, only=None):
    """Validate the item set before any paid call.

    Every failure below is raised before the first model call: a missing
    field, an unknown legend, a duplicate item id, a missing image file,
    or an --only set that the items file does not hold would otherwise be
    discovered mid-run, after calls were already paid for.
    """
    if not isinstance(items, list):
        raise SystemExit("items file must hold a JSON list of items")
    seen = set()
    sets = set()
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
        sets.add(item["set"])
        if not Path(item["image"]).is_file():
            raise SystemExit("item %s image file is missing: %s" % (item["item_id"], item["image"]))
    if only:
        missing = sorted(set(only.split(",")) - sets)
        if missing:
            raise SystemExit("requested set(s) are not in the items file: " + ", ".join(missing))


def write_json_atomic(path, payload):
    """Write JSON atomically: temp file in the same directory, then rename.

    A crash mid-write must not leave a truncated results file behind: the
    previous complete file survives until the new one is renamed over it.
    """
    target = Path(path)
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".critique-", suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w") as handle:
            json.dump(payload, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def call(key, messages, max_tokens=8000, *, max_attempts=2):
    body = json.dumps(
        {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.1}
    ).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                status = r.status
                raw = r.read().decode("utf-8", errors="replace")
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as error:
                return status, {
                    "error_body": raw[:1500],
                    "transport_error": f"invalid_json: {error}",
                    "attempts": attempt,
                }
            if not isinstance(response, dict):
                return status, {
                    "error_body": raw[:1500],
                    "transport_error": "response is not a JSON object",
                    "attempts": attempt,
                }
            response["attempts"] = attempt
            return status, response
        except urllib.error.HTTPError as error:
            response = {"error_body": error.read().decode(errors="replace")[:1500]}
            retryable = error.code == 429 or error.code >= 500
            if retryable and attempt < max_attempts:
                continue
            response["attempts"] = attempt
            return error.code, response
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            response = {
                "error_body": f"{type(error).__name__}: {error}"[:1500],
                "transport_error": type(error).__name__,
            }
            if attempt < max_attempts:
                continue
            response["attempts"] = attempt
            return 0, response
    raise AssertionError("unreachable call retry state")


def text_of(resp):
    try:
        msg = resp["choices"][0]["message"]
        return msg.get("content") or ""
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"model response has no choices[0].message.content: {error}") from error


def extract_findings_object(txt):
    """Return the response object carrying the complete findings schema."""
    decoder = json.JSONDecoder()
    index = 0
    best = None
    best_score = 0
    decoded = 0
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
                best, best_score = candidate, score
                if score == len(REQUIRED_FINDINGS_KEYS):
                    return candidate
        index = max(end, start + 1)
    if best is not None:
        return best
    if decoded:
        raise ValueError("no findings object in the response")
    raise ValueError("no JSON object in the response")


def parse_findings(txt):
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


def results_payload(calls, run_calls, records):
    """The results-file payload written after every item."""
    return {
        "model": MODEL,
        "base": BASE,
        "calls_used": calls,
        "run_calls_used": run_calls,
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(HERE / "items.json"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--prompt-suffix", default="")
    ap.add_argument(
        "--allowed-env-dir",
        action="append",
        default=[],
        help=(
            "directory the MiniMax env file must resolve inside; repeatable. "
            "Also read from " + ALLOWED_ENV_DIRS_ENV + "."
        ),
    )
    a = ap.parse_args()
    with Path(a.items).open(encoding="utf-8") as handle:
        items = json.load(handle)
    # The item set is validated before the first paid call.
    validate_items(items, a.only)
    if a.only:
        keep = set(a.only.split(","))
        items = [it for it in items if it["set"] in keep]
    allowed_dirs = list(a.allowed_env_dir)
    allowed_dirs.extend(
        part for part in os.environ.get(ALLOWED_ENV_DIRS_ENV, "").split(os.pathsep) if part
    )
    key = load_key(ENV_FILE, allowed_dirs)
    done = {}
    records = []
    prev_calls = 0
    if Path(a.out).exists():
        with Path(a.out).open(encoding="utf-8") as handle:
            prev = json.load(handle)
        prev_calls = prev.get("calls_used", 0)
        done, records = resume_state(prev.get("records", []), items, a.prompt_suffix, a.max_tokens)
    calls = prev_calls
    run_calls = 0
    for it in items:
        if it["item_id"] in done:
            print("skip (done)", it["item_id"])
            continue
        # The cap is per invocation: a documented retry run gets its own
        # budget instead of inheriting an already exhausted cap.
        if run_calls >= a.max_calls:
            print("CALL CAP REACHED")
            break
        tmpl = PROMPT_QUAD if it["legend"] == "quad" else PROMPT_BINARY
        prompt = tmpl.format(item_id=it["item_id"], scene=it["scene"], stats=it["stats"])
        if a.prompt_suffix:
            prompt += "\n" + a.prompt_suffix
        # One read of the image bytes: the same bytes are sent to the model
        # and hashed into the record's fingerprint, so the fingerprint can
        # never describe a different read than the verdict answers.
        image_bytes = read_image_bytes(it["image"])
        if image_bytes is None:
            # The image cannot be read: record the failure and pay no call.
            rec = {
                "item_id": it["item_id"],
                "input_fingerprint": None,
                "set": it["set"],
                "http": None,
                "usage": None,
                "counted": False,
                "response_digest": hashlib.sha256(b"").hexdigest(),
                "response_chars": 0,
                "findings": None,
                "parse_error": "image is unreadable: %s" % it["image"],
            }
            records = [r for r in records if r["item_id"] != it["item_id"]]
            records.append(rec)
            write_json_atomic(a.out, results_payload(calls, run_calls, records))
            print("image unreadable", it["item_id"], flush=True)
            continue
        fingerprint = item_fingerprint(it, a.prompt_suffix, a.max_tokens, image_bytes)
        b64 = base64.b64encode(image_bytes).decode()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                ],
            }
        ]
        remaining = a.max_calls - run_calls
        attempts_allowed = min(2, remaining)
        st, resp = call(key, msgs, max_tokens=a.max_tokens, max_attempts=attempts_allowed)
        attempts = max(1, int(resp.get("attempts", 1)))
        calls += attempts
        run_calls += attempts
        structure_error = None
        try:
            txt = text_of(resp)
        except ValueError as error:
            txt = ""
            structure_error = str(error)
        rec = {
            "item_id": it["item_id"],
            "input_fingerprint": fingerprint,
            "set": it["set"],
            "http": st,
            "usage": resp.get("usage"),
            "counted": True,
            "response_digest": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
            "response_chars": len(txt),
            "transport_attempts": attempts,
        }
        try:
            if structure_error is not None:
                raise ValueError(structure_error)
            rec["findings"] = parse_findings(txt)
        except Exception as e:
            rec["findings"] = None
            rec["parse_error"] = str(e)
        # replace any earlier failed record for this item
        records = [r for r in records if r["item_id"] != it["item_id"]]
        records.append(rec)
        write_json_atomic(a.out, results_payload(calls, run_calls, records))
        fd = rec.get("findings") or {}
        print(
            "%s: HTTP %s verdict=%s findings=%d (calls=%d)"
            % (
                it["item_id"],
                st,
                fd.get("verdict", "PARSE_FAIL"),  # ty: ignore[unresolved-attribute]
                len(fd.get("findings", [])),  # ty: ignore[unresolved-attribute]
                calls,
            ),
            flush=True,
        )
    print("TOTAL CALLS:", calls)


if __name__ == "__main__":
    main()
