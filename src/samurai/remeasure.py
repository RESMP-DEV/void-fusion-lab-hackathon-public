"""Re-measure receipt for the fuse change (fusion gate D2).

The gate D2 ruling requires the mask to be re-measured after the
unanimous-support and abstention change. The receipt compares a
pre-change run and a post-change run over identical inputs and an
identical config, and both runs consume the tensor set named by one
calibration record's inputs.manifest_sha256, which the receipt names.

The receipt records numbers only: both mask digests, both report
digests, per-frame and total selected pixel counts for both runs,
mask agreement (IoU and symmetric-difference pixel count), and the
support histogram shift. It makes no PASS or FAIL claim and no
quality claim. It is not a D5 judgement.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import numpy as np

from .ensemble import EnsembleError

NON_CLAIMS = (
    "Numbers only: this receipt makes no PASS or FAIL claim and no quality claim.",
    "This receipt is not a D5 judgement.",
)


def mask_digest(masks: np.ndarray) -> str:
    """SHA-256 of a mask array's bytes (bool, C order)."""
    array = np.ascontiguousarray(masks, dtype=bool)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def report_digest(report: dict[str, Any]) -> str:
    """SHA-256 of a report's canonical JSON bytes."""
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _histogram_total(
    histogram: list[int] | list[list[int] | tuple[int, ...]] | None, sizes: int
) -> list[int] | None:
    if histogram is None:
        return None
    total = [0] * sizes
    # Support both an already-aggregated one-dimensional histogram and the
    # nested per-frame form emitted by a report collector. A flat histogram is
    # copied; per-frame rows are summed into the clip total.
    rows: list[list[int]]
    if any(isinstance(value, list | tuple) for value in histogram):
        rows = [
            [int(value) for value in row if isinstance(value, int)]
            for row in histogram
            if isinstance(row, list | tuple)
        ]
    else:
        flat = cast(list[int], histogram)
        rows = [[int(value) for value in flat]]
    for row in rows:
        for index, count in enumerate(row):
            if index < sizes:
                total[index] += int(count)
    return total


def build_remeasure_receipt(
    *,
    manifest_sha256: str,
    pre_mask_digest: str,
    post_mask_digest: str,
    pre_report_digest: str,
    post_report_digest: str,
    pre_masks: np.ndarray,
    post_masks: np.ndarray,
    pre_support_histogram: list[int] | list[list[int] | tuple[int, ...]] | None = None,
    post_support_histogram: list[int] | list[list[int] | tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Build the D2 re-measure receipt from two stored runs.

    The digests are the caller's stored-artifact digests (mask video
    and report file bytes); the mask arrays give the selected counts
    and the agreement numbers. Support histograms are per-run pixel
    counts by active set size, summed over frames.
    """
    pre = np.asarray(pre_masks, dtype=bool)
    post = np.asarray(post_masks, dtype=bool)
    if pre.shape != post.shape:
        raise EnsembleError(
            f"re-measure runs are not comparable: mask shapes {pre.shape} != {post.shape}"
        )
    intersection = int(np.count_nonzero(pre & post))
    union = int(np.count_nonzero(pre | post))

    def histogram_width(histogram: list[int] | list[list[int] | tuple[int, ...]] | None) -> int:
        if histogram is None:
            return 0
        if any(isinstance(value, list | tuple) for value in histogram):
            return max(
                (len(value) for value in histogram if isinstance(value, list | tuple)),
                default=0,
            )
        return len(histogram)

    histogram_sizes = max(
        histogram_width(pre_support_histogram), histogram_width(post_support_histogram)
    )
    pre_histogram = _histogram_total(pre_support_histogram, histogram_sizes)
    post_histogram = _histogram_total(post_support_histogram, histogram_sizes)
    shift = (
        [
            post_count - pre_count
            for pre_count, post_count in zip(pre_histogram, post_histogram, strict=True)
        ]
        if pre_histogram is not None and post_histogram is not None
        else None
    )
    return {
        "gate": "D2",
        "kind": "remeasure_receipt",
        "inputs": {"manifest_sha256": manifest_sha256},
        "pre": {
            "mask_digest": pre_mask_digest,
            "report_digest": pre_report_digest,
            "selected_per_frame": [
                int(np.count_nonzero(frame)) for frame in pre.reshape(pre.shape[0], -1)
            ],
            "selected_total": int(np.count_nonzero(pre)),
            "support_histogram": pre_histogram,
        },
        "post": {
            "mask_digest": post_mask_digest,
            "report_digest": post_report_digest,
            "selected_per_frame": [
                int(np.count_nonzero(frame)) for frame in post.reshape(post.shape[0], -1)
            ],
            "selected_total": int(np.count_nonzero(post)),
            "support_histogram": post_histogram,
        },
        "agreement": {
            "iou": (intersection / union) if union else 1.0,
            "symmetric_difference_pixels": int(np.count_nonzero(pre != post)),
        },
        "support_histogram_shift": shift,
        "non_claims": list(NON_CLAIMS),
    }
