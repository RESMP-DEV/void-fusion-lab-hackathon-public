from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import samurai.ensemble as ensemble_module
from samurai.ensemble import (
    CommandExpertSource,
    EnsembleError,
    ExpertSpec,
    FusionSpec,
    TensorExpertSource,
    dynamic_batches,
    expert_probabilities,
    fuse_batch,
    load_config,
    run_ensemble,
)
from samurai.expert_persist import load_expert_tensor, verify_manifest
from samurai.media import write_gray_video
from samurai.remeasure import build_remeasure_receipt, mask_digest, report_digest


def spec(name: str, *, weight: float = 1.0, temperature: float = 1.0) -> ExpertSpec:
    return ExpertSpec(
        name=name,
        weight=weight,
        temperature=temperature,
        kind="tensor",
        representation="logits",
    )


def test_temperature_is_applied_before_sigmoid_and_weights_normalize():
    values = np.array(
        [
            [[[2.0, -2.0], [0.0, 0.0]]],
            [[[-2.0, 2.0], [0.0, 0.0]]],
        ]
    )
    specs = [spec("a", weight=3.0, temperature=2.0), spec("b", weight=1.0, temperature=1.0)]
    masks, probabilities, reports = fuse_batch(
        values,
        specs,
        FusionSpec(mode="top_k", top_k_ratio=0.25, dynamic_weights="static"),
    )
    assert masks.shape == probabilities.shape == (1, 2, 2)
    assert np.count_nonzero(masks) == 1
    assert reports[0]["weight_a"] == 0.75
    assert reports[0]["weight_b"] == 0.25


def test_entropy_reweights_uncertain_expert_and_support_filters_outliers():
    confident = np.full((1, 2, 2), 8.0)
    uncertain = np.full((1, 2, 2), 0.0)
    hallucination = np.array([[[8.0, 8.0], [-8.0, -8.0]]])
    values = np.stack([confident, uncertain, hallucination])
    specs = [spec("anchor"), spec("uncertain"), spec("semantic")]
    masks, _probabilities, reports = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            dynamic_weights="entropy",
            min_expert_support=2,
        ),
    )
    assert reports[0]["weight_anchor"] > reports[0]["weight_uncertain"]
    assert np.array_equal(masks[0], np.array([[True, True], [False, False]]))


def test_all_uncertain_panel_and_top_k_score_floor_fail_closed():
    uncertain = np.zeros((3, 1, 2, 2))
    specs = [spec("a"), spec("b"), spec("c")]
    masks, probabilities, reports = fuse_batch(
        uncertain,
        specs,
        FusionSpec(
            mode="top_k",
            top_k_ratio=1.0,
            threshold=0.5,
            min_expert_support=1,
            support_threshold=0.5,
            dynamic_weights="entropy",
        ),
    )
    assert np.isfinite(probabilities).all()
    assert not masks.any()
    assert all(
        np.isfinite([value for value in report.values() if isinstance(value, float)]).all()
        for report in reports
    )

    # D2: the average renormalizes over the active set, so a lone
    # supporting expert is no longer diluted by the full panel mass.
    # The fail-closed floor is now eligibility and score: the same
    # pixels are refused when min_expert_support rises above their
    # support of 1, and a pixel whose renormalized probability stays
    # under the threshold is refused on score.
    weak_support = np.array([[[[0.51], [0.2]]], [[[-6.0], [-6.0]]], [[[-6.0], [-6.0]]]])
    masks, _, _ = fuse_batch(
        weak_support,
        [spec("a"), spec("b"), spec("c")],
        FusionSpec(
            mode="top_k",
            top_k_ratio=2.0,
            threshold=0.5,
            min_expert_support=1,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert masks.any()
    masks, _, _ = fuse_batch(
        weak_support,
        [spec("a"), spec("b"), spec("c")],
        FusionSpec(
            mode="top_k",
            top_k_ratio=2.0,
            threshold=0.5,
            min_expert_support=2,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert not masks.any()
    below_floor = np.array([[[[-0.41], [-0.41]]], [[[-6.0], [-6.0]]], [[[-6.0], [-6.0]]]])
    masks, _, _ = fuse_batch(
        below_floor,
        [spec("a"), spec("b"), spec("c")],
        FusionSpec(
            mode="top_k",
            top_k_ratio=2.0,
            threshold=0.5,
            min_expert_support=1,
            support_threshold=0.3,
            dynamic_weights="static",
        ),
    )
    assert not masks.any()


def test_dynamic_batches_are_bounded_by_pixels():
    fusion = FusionSpec(batch_size=8, max_batch_pixels=8)
    batches = dynamic_batches(5, (5, 2, 2), fusion)
    assert batches == [(0, 1), (2, 3), (4,)]


def test_nonfinite_and_out_of_range_probabilities_fail():
    values = np.zeros((2, 1, 2, 2))
    values[0, 0, 0, 0] = np.nan
    try:
        fuse_batch(values, [spec("a"), spec("b")], FusionSpec())
    except EnsembleError:
        pass
    else:
        raise AssertionError("nonfinite expert value was accepted")

    probability_spec = ExpertSpec(name="a", weight=1, temperature=1, representation="probabilities")
    values[0, 0, 0, 0] = 1.5
    try:
        fuse_batch(values, [probability_spec, spec("b")], FusionSpec())
    except EnsembleError:
        pass
    else:
        raise AssertionError("out-of-range probability was accepted")


def test_tensor_expert_sources_must_align(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.zeros((2, 3, 4)))
    assert TensorExpertSource(
        ExpertSpec(name="first", weight=1, temperature=1, path=first)
    ).shape == (2, 3, 4)
    np.save(second, np.zeros((3, 4)))
    try:
        TensorExpertSource(
            ExpertSpec(name="second", weight=1, temperature=1, path=second, kind="tensor")
        )
    except EnsembleError:
        pass
    else:
        raise AssertionError("invalid source was accepted")


def test_run_ensemble_writes_video_and_report(tmp_path):
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    np.save(second, np.array([np.full((4, 6), -4.0), np.full((4, 6), 4.0)]))
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "first", "path": str(first), "weight": 2.0},
                    {
                        "name": "second",
                        "path": str(second),
                        "representation": "logits",
                        "weight": 1.0,
                    },
                ],
                "fusion": {
                    "mode": "top_k",
                    "top_k_ratio": 0.25,
                    "dynamic_weights": "entropy",
                    "batch_size": 1,
                },
                "calibration": {"method": "test", "confidence_weights": {"first": 0.5}},
            }
        )
    )
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["frames"] == 2
    assert result["batches"] == [1, 1]
    assert (tmp_path / "mask.mp4").is_file()
    saved_report = json.loads((tmp_path / "report.json").read_text())
    assert saved_report["experts"] == ["first", "second"]
    assert saved_report["calibration"]["method"] == "test"


def test_command_expert_rasterizes_normalized_regions(tmp_path):
    image = tmp_path / "frame.png"
    from PIL import Image

    Image.new("RGB", (10, 8), "black").save(image)

    def runner(command, request):
        assert command == ["fake-expert"]
        assert request["frames"][0]["frame_index"] == 0
        return {
            "regions_by_frame": [
                {
                    "frame_index": 0,
                    "regions": [
                        {
                            "type": "box",
                            "coordinates": [0.1, 0.25, 0.6, 0.75],
                            "confidence": 0.8,
                        }
                    ],
                }
            ]
        }

    source = CommandExpertSource(
        ExpertSpec(
            name="semantic",
            weight=1,
            temperature=1,
            kind="command",
            command=("fake-expert",),
            representation="probabilities",
        ),
        [image],
        (1, 8, 10),
        runner=runner,
    )
    probabilities = source.read([0])
    assert probabilities.shape == (1, 8, 10)
    assert probabilities.max() == np.float64(0.8)
    assert probabilities[0, 0, 0] == 0


def _muse_source(tmp_path, response):
    """A Muse command expert whose runner returns one fixed response."""
    image = tmp_path / "frame.png"
    if not image.exists():
        from PIL import Image

        Image.new("RGB", (10, 8), "black").save(image)

    def runner(command, request):
        return response

    return CommandExpertSource(
        ExpertSpec(
            name="muse",
            weight=1,
            temperature=1,
            kind="command",
            command=("fake-expert",),
            representation="probabilities",
        ),
        [image],
        (1, 8, 10),
        runner=runner,
    )


def _record(regions):
    return {"regions_by_frame": [{"frame_index": 0, "regions": regions}]}


def _box(coordinates, confidence=0.8):
    return {"type": "box", "coordinates": coordinates, "confidence": confidence}


MALFORMED_REGIONS = [
    pytest.param([_box(["0.0", "0.0", "0.5", "0.5"])], id="string-coordinates"),
    pytest.param([_box(["west", 0.0, 0.5, 0.5])], id="alpha-coordinate"),
    pytest.param([_box([float("nan"), 0.0, 0.5, 0.5])], id="nan-coordinate"),
    pytest.param([_box([float("inf"), 0.0, 0.5, 0.5])], id="infinite-coordinate"),
    pytest.param([_box([None, 0.0, 0.5, 0.5])], id="none-coordinate"),
    pytest.param([_box([True, 0.0, 0.5, 0.5])], id="bool-coordinate"),
    pytest.param([_box([0.5, 0.0, 0.25, 0.5])], id="negative-width"),
    pytest.param([_box([0.5, 0.0, 0.5, 0.5])], id="zero-area-box"),
    pytest.param([_box([0.0, 0.6, 0.5, 0.2])], id="swapped-y"),
    pytest.param([_box([0.0, 0.0, 1.5, 0.5])], id="coordinate-above-one"),
    pytest.param([_box([-0.1, 0.0, 0.5, 0.5])], id="negative-coordinate"),
    pytest.param([_box([0.0, 0.0, 0.5])], id="three-coordinates"),
    pytest.param([_box([0.0, 0.0, 0.5, 0.5, 0.9, 0.9])], id="six-coordinates"),
    pytest.param([{"type": "box", "confidence": 0.8}], id="missing-coordinates"),
    pytest.param([{"coordinates": [0.0, 0.0, 0.5, 0.5], "confidence": 0.8}], id="missing-type"),
    pytest.param([{"type": "box", "coordinates": [0.0, 0.0, 0.5, 0.5]}], id="missing-confidence"),
    pytest.param([_box([0.0, 0.0, 0.5, 0.5], confidence="0.8")], id="string-confidence"),
    pytest.param([_box([0.0, 0.0, 0.5, 0.5], confidence=float("nan"))], id="nan-confidence"),
    pytest.param([_box([0.0, 0.0, 0.5, 0.5], confidence=1.5)], id="confidence-above-one"),
    pytest.param(["not-a-region"], id="region-not-an-object"),
    pytest.param(
        [
            {
                "type": "polygon",
                "coordinates": [0.0, 0.0, 0.5, 0.0, 0.5, 0.5],
                "confidence": 0.8,
            }
        ],
        id="polygon-flat-coordinates",
    ),
]


@pytest.mark.parametrize("regions", MALFORMED_REGIONS)
def test_command_expert_malformed_region_fails_loud(tmp_path, regions):
    """A malformed region raises EnsembleError; it is never dropped
    silently on the way to an all-zero raster (M4 hardening)."""
    source = _muse_source(tmp_path, _record(regions))
    with pytest.raises(EnsembleError):
        source.read([0])


@pytest.mark.parametrize(
    "record",
    [
        pytest.param({"frame_index": 0}, id="no-regions-key"),
        pytest.param(
            {"frame_index": 0, "boxes": [_box([0.0, 0.0, 0.5, 0.5])]},
            id="regions-under-boxes-key",
        ),
    ],
)
def test_command_expert_missing_regions_key_fails_loud(tmp_path, record):
    """A record without a regions key raises; it does not default to
    an empty list and an all-zero map. Regions carried under another
    key (M1) hit the same raise: the key is missing, not empty."""
    source = _muse_source(tmp_path, {"regions_by_frame": [record]})
    with pytest.raises(EnsembleError, match="missing the regions key"):
        source.read([0])


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            {"regions_by_frame": [{"frame_index": 0, "regions": None}]}, id="regions-none"
        ),
        pytest.param(
            {"regions_by_frame": [{"frame_index": 0, "regions": "abc"}]}, id="regions-string"
        ),
        pytest.param({"regions_by_frame": ["not-a-record"]}, id="record-not-an-object"),
        pytest.param(["not-a-response"], id="response-not-an-object"),
    ],
)
def test_command_expert_response_shape_fails_loud(tmp_path, response):
    source = _muse_source(tmp_path, response)
    with pytest.raises(EnsembleError):
        source.read([0])


def test_command_expert_empty_regions_list_is_zero(tmp_path):
    """An explicitly empty regions list stays legal input. Its raster
    is all-zero, and run_ensemble records that as expert_empty via the
    0cb7e6c path (see test_expert_persist) -- recorded, not silent."""
    source = _muse_source(tmp_path, _record([]))
    probabilities = source.read([0])
    assert probabilities.shape == (1, 8, 10)
    assert not probabilities.any()


def test_command_expert_valid_polygon_rasterizes(tmp_path):
    source = _muse_source(
        tmp_path,
        _record(
            [
                {
                    "type": "polygon",
                    "coordinates": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
                    "confidence": 0.8,
                }
            ]
        ),
    )
    probabilities = source.read([0])
    assert probabilities.max() == np.float64(0.8)


def test_muse_text_shapes_fail_loud_or_parse(tmp_path):
    """End to end over the adapter text path: parse_model_content text
    feeds CommandExpertSource.read, the Muse consumer path."""
    from aimlapi_expert import parse_model_content

    valid = json.dumps([{"frame_index": 0, "regions": [_box([0.0, 0.0, 0.5, 0.5])]}])
    prose = f"Here is the result:\n{valid}\nDone."
    source = _muse_source(tmp_path, parse_model_content(prose))
    assert source.read([0]).max() == np.float64(0.8)

    malformed = json.dumps([{"frame_index": 0, "regions": [_box(["0.0", "0.0", "0.5", "0.5"])]}])
    source = _muse_source(tmp_path, parse_model_content(malformed))
    with pytest.raises(EnsembleError):
        source.read([0])

    missing_regions = '[{"frame_index": 0}]'
    source = _muse_source(tmp_path, parse_model_content(missing_regions))
    with pytest.raises(EnsembleError, match="missing the regions key"):
        source.read([0])

    nan_text = (
        '[{"frame_index": 0, "regions": [{"type": "box",'
        ' "coordinates": [NaN, 0.0, 0.5, 0.5], "confidence": 0.8}]}]'
    )
    source = _muse_source(tmp_path, parse_model_content(nan_text))
    with pytest.raises(EnsembleError):
        source.read([0])

    empty = '[{"frame_index": 0, "regions": []}]'
    source = _muse_source(tmp_path, parse_model_content(empty))
    assert not source.read([0]).any()

    with pytest.raises(ValueError):
        parse_model_content(valid[: len(valid) // 2])


def prob_spec(name: str, *, weight: float = 1.0) -> ExpertSpec:
    return ExpertSpec(
        name=name,
        weight=weight,
        temperature=1.0,
        kind="tensor",
        representation="probabilities",
    )


def test_unanimous_support_required():
    values = np.array([[[[0.9]]], [[[0.9]]], [[[0.1]]]])
    specs = [prob_spec("a"), prob_spec("b"), prob_spec("c")]
    masks, _, _ = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            min_expert_support=3,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert not masks.any()
    masks, _, _ = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            min_expert_support=2,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert masks[0, 0, 0]


def test_abstention_renormalizes_over_active_mass():
    values = np.array([[[[0.9]]], [[[0.9]]], [[[0.1]]]])
    specs = [prob_spec("a"), prob_spec("b"), prob_spec("c")]
    _masks, combined, _reports = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            min_expert_support=2,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert combined[0, 0, 0] == pytest.approx(0.9)
    assert combined[0, 0, 0] != pytest.approx(0.6333, abs=0.01)


def test_single_expert_cannot_select():
    values = np.array([[[[1.0]]], [[[0.0]]], [[[0.0]]]])
    specs = [prob_spec("a"), prob_spec("b"), prob_spec("c")]
    masks, combined, _reports = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            min_expert_support=3,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert combined[0, 0, 0] == pytest.approx(1.0)
    assert not masks.any()


def test_no_active_expert_pixel_is_background():
    values = np.array([[[[0.4]]], [[[0.3]]], [[[0.2]]]])
    specs = [prob_spec("a"), prob_spec("b"), prob_spec("c")]
    masks, combined, _reports = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            min_expert_support=1,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert combined[0, 0, 0] == 0.0
    assert not masks.any()


def test_support_histogram_reported():
    frame0 = [
        [(0.9, 0.9, 0.9), (0.9, 0.9, 0.1)],
        [(0.9, 0.1, 0.1), (0.1, 0.1, 0.1)],
    ]
    frame1 = [
        [(0.9, 0.9, 0.9), (0.9, 0.9, 0.9)],
        [(0.9, 0.9, 0.9), (0.9, 0.9, 0.9)],
    ]
    values = np.transpose(np.array([frame0, frame1], dtype=np.float64), (3, 0, 1, 2))
    specs = [prob_spec("a"), prob_spec("b"), prob_spec("c")]
    _masks, _combined, reports = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="threshold",
            threshold=0.5,
            min_expert_support=3,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert reports[0]["support_histogram"] == [1, 1, 1, 1]
    assert reports[1]["support_histogram"] == [0, 0, 0, 4]


def test_top_k_respects_unanimous_eligibility():
    values = np.array([[[[0.95, 0.9]]], [[[0.95, 0.9]]], [[[0.1, 0.9]]]])
    specs = [prob_spec("a"), prob_spec("b"), prob_spec("c")]
    masks, _combined, _reports = fuse_batch(
        values,
        specs,
        FusionSpec(
            mode="top_k",
            top_k_ratio=0.5,
            threshold=0.5,
            min_expert_support=3,
            support_threshold=0.5,
            dynamic_weights="static",
        ),
    )
    assert not masks[0, 0, 0]
    assert masks[0, 0, 1]


def test_remeasure_receipt_shape():
    specs = [prob_spec("a"), prob_spec("b"), prob_spec("c")]
    frame = [
        [(0.9, 0.9, 0.9), (0.9, 0.9, 0.1)],
        [(0.9, 0.1, 0.1), (0.1, 0.1, 0.1)],
    ]
    values = np.transpose(np.array([frame], dtype=np.float64), (3, 0, 1, 2))
    fusion = FusionSpec(
        mode="threshold",
        threshold=0.5,
        min_expert_support=3,
        support_threshold=0.5,
        dynamic_weights="static",
    )
    post_masks, _combined, post_reports = fuse_batch(values, specs, fusion)
    # Pre-D2 reference: full-panel-mass average, a support of 1 eligible.
    probabilities = expert_probabilities(values, specs)
    base = np.full(len(specs), 1.0 / len(specs))
    pre_combined = np.sum(probabilities * base[:, None, None, None], axis=0)
    pre_support = np.sum(probabilities > fusion.support_threshold, axis=0)
    pre_masks = (pre_combined >= fusion.threshold) & (pre_support >= 1)
    pre_histogram = [
        int(count) for count in np.bincount(pre_support[0].ravel(), minlength=len(specs) + 1)
    ]
    post_histogram = post_reports[0]["support_histogram"]
    manifest = "ab" * 32
    receipt = build_remeasure_receipt(
        manifest_sha256=manifest,
        pre_mask_digest=mask_digest(pre_masks),
        post_mask_digest=mask_digest(post_masks),
        pre_report_digest=report_digest({"frames_report": [{"support_histogram": pre_histogram}]}),
        post_report_digest=report_digest({"frames_report": post_reports}),
        pre_masks=pre_masks,
        post_masks=post_masks,
        pre_support_histogram=pre_histogram,
        post_support_histogram=post_histogram,
    )
    # Both runs consumed the one input set the receipt names.
    assert receipt["inputs"]["manifest_sha256"] == manifest
    assert receipt["pre"]["mask_digest"] == mask_digest(pre_masks)
    assert receipt["post"]["mask_digest"] == mask_digest(post_masks)
    assert receipt["pre"]["mask_digest"] != receipt["post"]["mask_digest"]
    assert receipt["pre"]["report_digest"]
    assert receipt["post"]["report_digest"]
    assert receipt["pre"]["selected_total"] == int(np.count_nonzero(pre_masks))
    assert receipt["post"]["selected_total"] == int(np.count_nonzero(post_masks))
    assert receipt["post"]["selected_per_frame"] == [1]
    difference = int(np.count_nonzero(pre_masks != post_masks))
    assert receipt["agreement"]["symmetric_difference_pixels"] == difference
    intersection = int(np.count_nonzero(pre_masks & post_masks))
    union = int(np.count_nonzero(pre_masks | post_masks))
    assert receipt["agreement"]["iou"] == pytest.approx(intersection / union)
    assert receipt["support_histogram_shift"] == [
        post_count - pre_count
        for pre_count, post_count in zip(pre_histogram, post_histogram, strict=True)
    ]


def test_remeasure_sums_nested_per_frame_support_histograms():
    masks = np.zeros((2, 1, 1), dtype=bool)
    receipt = build_remeasure_receipt(
        manifest_sha256="ab" * 32,
        pre_mask_digest=mask_digest(masks),
        post_mask_digest=mask_digest(masks),
        pre_report_digest=report_digest({}),
        post_report_digest=report_digest({}),
        pre_masks=masks,
        post_masks=masks,
        pre_support_histogram=[[1, 2, 0], [0, 3, 4]],
        post_support_histogram=[[0, 1, 1], [2, 0, 0]],
    )
    assert receipt["pre"]["support_histogram"] == [1, 5, 4]
    assert receipt["post"]["support_histogram"] == [2, 1, 1]
    assert receipt["support_histogram_shift"] == [1, -4, -3]


COUNTING_EXPERT = r"""import json, pathlib, sys

request = json.load(sys.stdin)
log = pathlib.Path({log!r})
calls = log.read_text().splitlines() if log.exists() else []
index = len(calls)
calls.append(str(index))
log.write_text("\n".join(calls) + "\n")
boxes = {boxes!r}
box = boxes[min(index, len(boxes) - 1)]
frames = [
    {{"frame_index": frame["frame_index"], "regions": [] if box is None else [box]}}
    for frame in request["frames"]
]
print(json.dumps({{"regions_by_frame": frames}}))
"""


def _counting_expert(tmp_path, boxes):
    """Write a command expert that returns a different box per call.

    The first call returns ``boxes[0]``; later calls return the next
    box, and ``None`` means no regions at all. Every call is logged, so
    a test can count the invocations the run made.
    """
    log = tmp_path / "expert-calls.log"
    script = tmp_path / "counting_expert.py"
    script.write_text(COUNTING_EXPERT.format(log=str(log), boxes=boxes))
    return script, log


def _anchor_run(tmp_path, boxes, *, persist: bool = False):
    """Run a two-expert panel whose command expert carries an anchor.

    The anchor source is the left half of a 4x6 frame; the command
    expert's first box covers the left half too, so the gate passes.
    """
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    tensor_path = tmp_path / "other.npy"
    np.save(tensor_path, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    source = np.zeros((4, 6), dtype=bool)
    source[:, :3] = True
    np.save(tmp_path / "anchor.npy", source)
    script, log = _counting_expert(tmp_path, boxes)
    payload = {
        "input_video": str(video),
        "experts": [
            {"name": "other", "path": str(tensor_path), "weight": 1.0},
            {
                "name": "cmd",
                "kind": "command",
                "command": [sys.executable, str(script)],
                "weight": 1.0,
                "anchor_source": "anchor.npy",
            },
        ],
        "fusion": {"mode": "threshold", "threshold": 0.5, "batch_size": 1},
    }
    if persist:
        payload["expert_persist_dir"] = str(tmp_path / "experts")
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(json.dumps(payload))
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    return result, log


LEFT_BOX = {"type": "box", "coordinates": [0.0, 0.0, 0.5, 1.0], "confidence": 0.8}


def test_anchor_gate_does_not_reinvoke_the_command_expert(tmp_path):
    """Finding 4: the anchor gate must not re-invoke a command expert.

    The command is called once per batch, and the gate judges the
    values of that one call: two frames, two batches, two calls.
    """
    result, log = _anchor_run(tmp_path, [LEFT_BOX, LEFT_BOX])
    calls = log.read_text().splitlines()
    assert len(calls) == 2, "the anchor gate invoked the command expert a second time"
    assert result["anchor_gate"][0]["expert_name"] == "cmd"


def test_anchor_gate_judges_the_frame_zero_values_the_fusion_consumes(tmp_path):
    """Finding 4: the anchor gate judges the frame-0 values the fusion
    consumes, not a different sample.

    The command expert returns the left half on its first call and
    nothing on its second. The gate's report must describe the same
    frame-0 pixels the fusion consumed and persisted.
    """
    result, log = _anchor_run(tmp_path, [LEFT_BOX, None], persist=True)
    assert len(log.read_text().splitlines()) == 2
    persisted = load_expert_tensor(tmp_path / "experts", "cmd", version="1")
    consumed_pixels = int(np.count_nonzero(persisted[0]))
    assert consumed_pixels > 0
    assert result["anchor_gate"][0]["expert_pixels"] == consumed_pixels


def test_persisted_expert_artifacts_use_float16_storage(tmp_path):
    """Finding 5: the D4 artifacts are stored in float16, not float64.

    The stored values are the consumed values rounded to float16, and
    the manifest record names that dtype.
    """
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    np.save(second, np.array([np.full((4, 6), -4.0), np.full((4, 6), 4.0)]))
    persist_dir = tmp_path / "experts"
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "first", "path": str(first), "weight": 1.0},
                    {"name": "second", "path": str(second), "weight": 1.0},
                ],
                "fusion": {"mode": "threshold", "threshold": 0.5, "batch_size": 1},
                "expert_persist_dir": str(persist_dir),
            }
        )
    )
    run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    manifest = verify_manifest(persist_dir)
    assert manifest["artifacts"]
    for record in manifest["artifacts"]:
        assert record["dtype"] == "float16"
        stored = np.load(persist_dir / record["file"], mmap_mode="r")
        assert stored.dtype == np.float16
    stored_first = load_expert_tensor(persist_dir, "first", version="1")
    source = np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)])
    assert stored_first.tobytes() == source.astype(np.float16).tobytes()


def test_required_mode_rejects_an_at_threshold_dead_expert(tmp_path):
    """An expert at the support threshold abstains and is dead evidence.

    The empty-expert gate uses the same strict support comparison as
    fuse_batch: logits of exactly zero transform to probability 0.5,
    which does not exceed support_threshold 0.5 and therefore adds no
    active mass. The prior ``>=`` gate let this dead expert pass.
    """
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.zeros((2, 4, 6)))
    np.save(second, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "dead", "path": str(first), "representation": "logits"},
                    {"name": "live", "path": str(second), "representation": "logits"},
                ],
                "fusion": {
                    "mode": "threshold",
                    "threshold": 0.5,
                    "support_threshold": 0.5,
                    "batch_size": 1,
                },
                "expert_persist_mode": "required",
                "expert_persist_dir": str(tmp_path / "derived"),
            }
        )
    )
    with pytest.raises(EnsembleError, match="expert\\(s\\) dead"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")


def test_required_mode_rejects_a_dead_logits_expert(tmp_path):
    """Raw nonzero logits are not meaningful support after transformation."""
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.full((2, 4, 6), -100.0))
    np.save(second, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "dead", "path": str(first), "representation": "logits"},
                    {"name": "live", "path": str(second), "representation": "logits"},
                ],
                "fusion": {
                    "mode": "threshold",
                    "threshold": 0.5,
                    "support_threshold": 0.5,
                    "batch_size": 1,
                },
                "expert_persist_mode": "required",
                "expert_persist_dir": str(tmp_path / "derived"),
            }
        )
    )
    with pytest.raises(EnsembleError, match="expert\\(s\\) dead"):
        run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")


def test_run_ensemble_transforms_expert_probabilities_once_per_batch(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    write_gray_video([np.full((4, 6), value, dtype=np.uint8) for value in (0, 255)], video, fps=12)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.array([np.full((4, 6), 4.0), np.full((4, 6), -4.0)]))
    np.save(second, np.array([np.full((4, 6), -4.0), np.full((4, 6), 4.0)]))
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "input_video": str(video),
                "experts": [
                    {"name": "first", "path": str(first), "weight": 2.0},
                    {"name": "second", "path": str(second), "weight": 1.0},
                ],
                "fusion": {"mode": "threshold", "threshold": 0.5, "batch_size": 1},
            }
        )
    )
    real_transform = ensemble_module.expert_probabilities
    calls = 0

    def counting_transform(values, specs):
        nonlocal calls
        calls += 1
        return real_transform(values, specs)

    monkeypatch.setattr(ensemble_module, "expert_probabilities", counting_transform)
    result = run_ensemble(load_config(config_path), tmp_path / "mask.mp4", tmp_path / "report.json")
    assert result["batches"] == [1, 1]
    assert calls == 2


def test_remeasure_sums_nested_tuple_support_histogram_rows():
    masks = np.zeros((1, 1, 1), dtype=bool)
    receipt = build_remeasure_receipt(
        manifest_sha256="ab" * 32,
        pre_mask_digest=mask_digest(masks),
        post_mask_digest=mask_digest(masks),
        pre_report_digest=report_digest({}),
        post_report_digest=report_digest({}),
        pre_masks=masks,
        post_masks=masks,
        pre_support_histogram=[(1, 2, 0), (0, 3, 4)],
        post_support_histogram=[(0, 1, 1), (2, 0, 0)],
    )
    assert receipt["pre"]["support_histogram"] == [1, 5, 4]
    assert receipt["post"]["support_histogram"] == [2, 1, 1]
    assert receipt["support_histogram_shift"] == [1, -4, -3]
