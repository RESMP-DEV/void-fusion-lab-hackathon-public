#!/usr/bin/env python3
# ruff: noqa: UP031
# Formatting note: the percent-style formatting and the long prompt and data
# strings are kept verbatim; converting them is a follow-up cleanup.
"""Score the critique run: planted-defect precision/recall + real-frame summary.
Reads items.json + results.json in this directory; writes score.json and
prints the compact scorecard-shaped summary for CRITIQUE-RUN.md.
"""

import json
import os
import sys
import tempfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(path, label):
    """Read a required JSON file, with presence and format failures named."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise SystemExit("cannot read %s: %s (%s)" % (label, path, error)) from error
    except json.JSONDecodeError as error:
        raise SystemExit("%s is not valid JSON: %s (%s)" % (label, path, error)) from error


def load_score_inputs(items_path, results_path):
    """Load and shape-check the scorer's two required input files."""
    items_payload = load_json(items_path, "items file")
    if not isinstance(items_payload, list) or not all(
        isinstance(item, dict) and isinstance(item.get("item_id"), str) for item in items_payload
    ):
        raise SystemExit("items file must hold a JSON list of item objects with item_id strings")
    items = {item["item_id"]: item for item in items_payload}
    if len(items) != len(items_payload):
        raise SystemExit("items file repeats an item_id")

    results = load_json(results_path, "results file")
    if not isinstance(results, dict):
        raise SystemExit("results file must hold a JSON object")
    records = results.get("records")
    calls_used = results.get("calls_used")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) and isinstance(record.get("item_id"), str) for record in records
    ):
        raise SystemExit("results file records must be a JSON list of objects with item_id strings")
    if not isinstance(calls_used, int) or isinstance(calls_used, bool) or calls_used < 0:
        raise SystemExit("results file calls_used must be a non-negative integer")
    return items, results


items, res = load_score_inputs(os.path.join(HERE, "items.json"), os.path.join(HERE, "results.json"))
recs = {r["item_id"]: r for r in res["records"]}

# The one verdict-class contract shared with scripts/critique_harness.py:
# "defect" is the only defect class. Every other class ("ok", an unknown
# class, a legacy bare string) is the clean branch.
DEFECT_VERDICT_CLASS = "defect"
INPUT_ERROR = "INPUT_ERROR"

TYPE_MATCH = {
    "too_small": ({"mask_too_small"}, "too_tight"),
    "too_large": ({"mask_too_large"}, "too_loose"),
    "shifted": ({"mask_shifted", "boundary_misaligned"}, None),
    "hole": ({"hole"}, None),
    "missing_part": ({"missing_part"}, None),
    "wrong_subject": ({"wrong_subject"}, None),
}


def is_flagged(fd):
    """Canonical "flagged" definition, shared with critique_harness.py.

    An item is flagged when its verdict is WARN or FAIL, or its findings
    list is non-empty. The prompt contract states an empty findings list
    is used only when the mask is truly clean, so a PASS verdict that
    still carries findings is inconsistent and must not score as clean
    (F31: the two staged scorers defined flagged differently).
    """
    return fd["verdict"] in ("WARN", "FAIL") or len(fd.get("findings") or []) > 0


def expected_verdict_class(it):
    """The one items contract for "expected" (the critique item builder
    writes it, critique_harness.py reads the same dict): a dict holding
    {"verdict_class": ..., "planted_type": ...}. Legacy items.json files
    with a bare verdict string are read as that string, so old runs
    still score; "defect" is the only defect class either way."""
    expected = it.get("expected")
    if isinstance(expected, dict):
        return expected.get("verdict_class")
    return expected


def is_defect_class(it):
    """True only for the defect class (see DEFECT_VERDICT_CLASS).

    The harness and this scorer must agree on which items are defects:
    only "defect" is one.
    """
    return expected_verdict_class(it) == DEFECT_VERDICT_CLASS


def is_input_error(it):
    """True for the expected-answer shapes the harness refuses to score.

    The harness (scripts/critique_harness.py score()) records an expected
    answer that carries no verdict_class as an INPUT_ERROR and records a
    missing expected answer as unscored. Neither is a clean item, so this
    scorer refuses both instead of counting them as good items.
    """
    expected = it.get("expected")
    if expected is None:
        return True
    return isinstance(expected, dict) and "verdict_class" not in expected


def planted_type_of(it):
    planted = it.get("planted_type")
    if planted is None and isinstance(it.get("expected"), dict):
        planted = it["expected"].get("planted_type")
    return planted


def typed_hit(it, fd):
    planted = planted_type_of(it)
    entry = TYPE_MATCH.get(planted)
    if entry is None:
        print(
            "warning: unknown planted_type %r; treating item as untyped" % (planted,),
            file=sys.stderr,
        )
        return False
    types, fit = entry
    if planted == "wrong_subject" and fd.get("wrong_subject"):
        return True
    if any(f["type"] in types for f in fd.get("findings", [])):
        return True
    if fit and fd.get("fit") == fit:
        return True
    return False


planted = {
    "n": 0,
    "errors": 0,
    "input_errors": 0,
    "detected": 0,
    "typed": 0,
    "good_n": 0,
    "good_fp": 0,
    "by_type": defaultdict(lambda: [0, 0, 0]),
}
for iid, it in items.items():
    if it["set"] != "v9planted":
        continue
    r = recs.get(iid)
    fd = (r or {}).get("findings")
    if fd is None:
        planted["errors"] += 1
        continue
    if is_input_error(it):
        planted["input_errors"] += 1
        continue
    flagged = is_flagged(fd)
    if is_defect_class(it):
        planted["n"] += 1
        det = flagged
        typ = det and typed_hit(it, fd)
        planted["detected"] += det
        planted["typed"] += typ
        b = planted["by_type"][planted_type_of(it)]
        b[0] += 1
        b[1] += det
        b[2] += typ
    else:
        planted["good_n"] += 1
        planted["good_fp"] += flagged

flagged_total = sum(
    1
    for iid, it in items.items()
    if it["set"] == "v9planted"
    and not is_input_error(it)
    and (recs.get(iid) or {}).get("findings")
    and is_flagged(recs[iid]["findings"])
)
tp = planted["detected"]
precision = (tp / flagged_total) if flagged_total else None
recall = (tp / planted["n"]) if planted["n"] else None

print("== PLANTED SET ==")
print(
    "defect items scored: %d, parse errors: %d, input errors: %d"
    % (planted["n"], planted["errors"], planted["input_errors"])
)
print(
    "detection recall: %d/%d = %s"
    % (tp, planted["n"], round(recall, 3) if recall is not None else "-")
)
print("typed recall:     %d/%d" % (planted["typed"], planted["n"]))
print(
    "precision (flagged & truly defect): %s over %d flagged"
    % (round(precision, 3) if precision is not None else "-", flagged_total)
)
print("good items: %d, false positives: %d" % (planted["good_n"], planted["good_fp"]))
for t, (n, d, ty) in sorted(planted["by_type"].items()):
    print("  %-14s n=%d detected=%d typed=%d" % (t, n, d, ty))

print("\n== REAL ARTIFACTS ==")
sev_count = Counter()
type_count = Counter()
worst = []
for iid, it in items.items():
    if it["set"] == "v9planted":
        continue
    r = recs.get(iid)
    fd = (r or {}).get("findings")
    if fd is None:
        print("%-14s ERROR %s" % (iid, (r or {}).get("parse_error", "no record")))
        continue
    for f in fd.get("findings", []):
        sev_count[f["severity"]] += 1
        type_count[f["type"]] += 1
    print(
        "%-14s %-4s findings=%d  %s"
        % (iid, fd["verdict"], len(fd.get("findings", [])), fd.get("one_line", "")[:110])
    )
    if is_flagged(fd):
        worst.append((fd["verdict"], iid))
print("\nfindings by type:", dict(type_count))
print("findings by severity:", dict(sev_count))
print("flagged frames:", worst)
print("\nTOTAL CRITIC CALLS:", res["calls_used"])
tok = sum((r.get("usage") or {}).get("total_tokens", 0) for r in res["records"])
print("TOTAL TOKENS:", tok)


def write_json_atomic(path, payload):
    """Write JSON atomically: temp file in the same directory, then rename.

    A crash mid-write must not leave a truncated score file behind: the
    previous complete file survives until the new one is renamed over it.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle_fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".critique-", suffix=".tmp")
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


write_json_atomic(
    os.path.join(HERE, "score.json"),
    {
        "planted": {
            "n_defect": planted["n"],
            "detected": tp,
            "typed": planted["typed"],
            "precision": precision,
            "recall": recall,
            "good_n": planted["good_n"],
            "good_fp": planted["good_fp"],
            "errors": planted["errors"],
            "input_errors": planted["input_errors"],
            "by_type": {k: v for k, v in planted["by_type"].items()},
        },
        "real_type_counts": dict(type_count),
        "real_sev_counts": dict(sev_count),
        "calls_used": res["calls_used"],
        "total_tokens": tok,
    },
)
