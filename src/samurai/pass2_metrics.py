"""No-Pass2 versus Pass 2 comparison metrics for the diegetic insertion lane.

The comparator answers one question with numbers: given a baseline inserted
video (the composite before VOID Pass 2) and a Pass 2 video, how much did Pass 2
change the inserted object, the cleanup ring around it, and the untouched
background, and how stable were those changes over time?

Conventions are taken from contracts that already exist in this repository:

* Both masks are ROI-black, as defined by ``src/samurai/quadmask.py``: black
  (0) is the region of interest, white (255) is not.
* The cleanup ring is the cleanup ROI minus the inserted ROI, exactly the
  region ``quadmask._combine_insertion`` hands to VOID Pass 2.
* A frame difference is the mean absolute per-channel difference of the two
  RGB frames.  A pixel counts as changed above ``changed_threshold``.
* ``cleanup_value`` names the VOID quadmask value of the cleanup ring, 0 for
  remove and 127 for affected.  ``quadmask._combine_insertion`` clears an
  affected-only ring on frame zero, so with 127 the first frame's ring is
  outside the cleanup ROI and is reported as background instead.
* With 127 that first frame is also a structural lead-in: the whole frame-zero
  quadmask is keep, so VOID leaves frame zero unchanged by contract.  Frame
  zero is therefore excluded from the temporal measurement window, and any
  change measured inside it is reported as ``lead_in_changed_pixels`` rather
  than being scored as a temporal artifact.
* Temporal stability uses the second-difference pop contract already used by
  ``samurai.insertion`` and ``scripts/best_of_n.py``: ordinary motion is not
  penalized, a pop is.  The reduction is a maximum over the interior second
  differences, not an average, so one severe pop is not diluted by a long
  steady stretch.  A difference that is present in some measured frames and
  absent in others is a temporal dropout and scores the maximum pop.
* The timeline is compared with the exact per-frame presentation timestamps,
  read as integers in the stream time base and compared as rational numbers.
  A nominal frame rate cannot separate a constant frame rate clip from a
  variable one, so both the exact timestamps and the constant-versus-variable
  character have to agree.

The comparator fails closed on geometry, frame-count, or frame-rate mismatch
between the two videos or between a video and its masks, on a per-frame
timestamp mismatch, on a constant-versus-variable frame-rate mismatch, and it
refuses to measure a region that is empty in every frame.  It issues no quality
verdict and claims nothing about what produced either video.

Frames are consumed one at a time.  No function in this module retains a whole
clip's difference maps, so a long clip costs one frame of memory rather than
one frame per second of video.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .media import VideoInfo, iter_color_frames, iter_gray_frames, probe_video
from .quadmask import AFFECTED, REMOVE

ROI_THRESHOLD = 128
DEFAULT_CHANGED_THRESHOLD = 10.0
DEFAULT_FPS_TOLERANCE = 0.005
CLEANUP_VALUES = (REMOVE, AFFECTED)
MAX_LEVEL = 255.0
# A one- or two-pixel second difference is threshold and boundary-counting
# noise, so a four-pixel second difference is the smallest full-strength mass
# pop.  With this floor a one-pixel boundary step scores 0.25 and a one-pixel
# alternation scores 0.5 instead of saturating the metric.
MIN_MASS_SCALE = 4.0
# Sub-pixel centroid wobble is the same kind of noise, so a two-pixel centroid
# second difference is the smallest full-strength centroid pop.
MIN_CENTROID_SCALE = 2.0
# A second difference needs one interior frame, so the temporal window needs
# at least three frames to be measurable at all.
MIN_MEASURED_FRAMES = 3
# With an affected cleanup the frame-zero quadmask is entirely keep, so frame
# zero is a structural lead-in and not part of the temporal window.
STRUCTURAL_LEAD_IN_FRAMES = 1
NON_CLAIMS = (
    "These numbers compare two supplied videos frame by frame; they do not "
    "prove that Pass 2 ran, that VOID was used, or that either video is diegetic.",
    "Object preservation is baseline-versus-Pass2 agreement inside the inserted ROI; "
    "it is not agreement with a ground-truth object render.",
    "No PASS/FAIL quality verdict is issued; this report states numbers only.",
    "A null psnr_db means the two videos are identical inside that region, where PSNR "
    "is unbounded rather than finite.",
    "With cleanup_value 127 the first frame's cleanup ring is outside the cleanup ROI, "
    "because VOID clears an affected-only ring on frame zero; that ring is reported as "
    "background.",
    "With cleanup_value 127 the first frame is also excluded from the temporal window, "
    "because its quadmask is entirely keep; a change measured there is reported as "
    "lead_in_changed_pixels instead of being scored as a temporal artifact.",
    "no_changed_pixels is threshold-relative: a difference below changed_threshold is "
    "not counted as a change. exactly_identical requires equal decoded values and is "
    "independent of the threshold.",
    "The temporal metric uses changed-pixel mass and centroid second differences. It "
    "cannot see a spatial rearrangement whose per-frame mass and centroid are identical, "
    "so per-frame coordinates remain the diagnostic for that case.",
    "Timestamp alignment proves the two videos and the two masks share one presentation "
    "timeline; it does not prove that the frames show the same scene.",
    "No model call, credential, or network access is made by this comparator.",
)


class Pass2MetricsError(RuntimeError):
    """A Pass 2 comparison input or measurement is invalid."""


def fraction_text(value: Fraction) -> str:
    """Return the exact rational text of one timestamp."""
    return f"{value.numerator}/{value.denominator}"


def _finite(name: str, value: float) -> float:
    """Reject a non-finite threshold before it can poison a reported number."""
    if not math.isfinite(value):
        raise Pass2MetricsError(f"{name} must be finite, got {value}")
    return value


def _timestamp_digest(pts: tuple[Fraction, ...]) -> str:
    """Return the SHA-256 of the canonical timestamp text."""
    payload = ",".join(fraction_text(value) for value in pts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InputAlignment:
    """Probed geometry and timing of one comparison input."""

    name: str
    width: int
    height: int
    frames: int
    fps: float


@dataclass(frozen=True)
class FrameAlignment:
    """Whether the two videos and the two masks describe the same timeline."""

    baseline: InputAlignment
    pass2: InputAlignment
    inserted_mask: InputAlignment
    cleanup_mask: InputAlignment
    dimensions_aligned: bool
    frames_aligned: bool
    fps_aligned: bool
    aligned: bool
    fps_delta: float
    fps_tolerance: float


@dataclass(frozen=True)
class FrameTimestamps:
    """Exact per-frame presentation timestamps of one comparison input.

    ``pts`` holds one rational timestamp per decoded frame, and ``digest`` is
    the SHA-256 of their canonical text so two timelines can be compared
    without carrying the full list.
    """

    name: str
    count: int
    pts: tuple[Fraction, ...]
    variable_frame_rate: bool
    cfr_delta: Fraction | None
    digest: str


@dataclass(frozen=True)
class TimestampAlignment:
    """Whether the four inputs share one exact presentation timeline."""

    baseline: FrameTimestamps
    pass2: FrameTimestamps
    inserted_mask: FrameTimestamps
    cleanup_mask: FrameTimestamps
    pts_aligned: bool
    frame_rate_character_aligned: bool
    aligned: bool


@dataclass(frozen=True)
class RegionMetrics:
    """Baseline-versus-Pass2 agreement inside one masked region."""

    name: str
    pixels: int
    frames_measured: int
    frames_total: int
    mae: float
    mse: float
    psnr_db: float | None
    changed_pixels: int
    changed_ratio: float
    max_abs_difference: float


@dataclass(frozen=True)
class TemporalStability:
    """Stability over time of the baseline-versus-Pass2 difference map.

    ``frames`` counts every compared frame and ``measured_frames`` counts the
    frames inside the temporal window.  The window starts after
    ``structural_lead_in`` frames, which the quadmask contract makes
    unchanged by construction; a difference measured inside the lead-in is
    reported as ``lead_in_changed_pixels`` instead of being scored as a pop.

    ``mean_mass``, ``mass_step_max``, ``centroid_step_mean``, and
    ``centroid_step_max`` describe the measured window, while ``frames``,
    ``frames_with_difference``, ``per_frame_mass``, and ``per_frame_centroid``
    describe every compared frame.  The series itself is frame-wide, not
    restricted to the cleanup ring, because the temporal metric watches every
    pixel the two videos disagree about.

    ``no_changed_pixels`` is threshold-relative: no pixel anywhere exceeded
    ``changed_threshold``.  ``exactly_identical`` is threshold-independent: the
    decoded values of the two videos are equal in every frame.
    """

    frames: int
    frames_with_difference: int
    no_changed_pixels: bool
    exactly_identical: bool
    measured_frames: int
    structural_lead_in: int
    lead_in_changed_pixels: int
    lead_in_violation: bool
    mean_mass: float
    mass_step_max: int
    mass_pop: float
    centroid_pop: float
    temporal_pop_score: float
    centroid_step_mean: float
    centroid_step_max: float
    per_frame_mass: list[int]
    per_frame_centroid: list[list[float] | None]


@dataclass(frozen=True)
class FrameComparison:
    """Region and temporal numbers for one baseline/Pass 2 frame sequence."""

    object_preservation: RegionMetrics
    cleanup_ring: RegionMetrics
    background: RegionMetrics
    temporal: TemporalStability
    changed_threshold: float
    cleanup_value: int


@dataclass(frozen=True)
class Pass2Comparison:
    """Every number the comparator measures for one baseline/Pass 2 pair."""

    frame_alignment: FrameAlignment
    timestamps: TimestampAlignment
    object_preservation: RegionMetrics
    cleanup_ring: RegionMetrics
    background: RegionMetrics
    temporal: TemporalStability
    changed_threshold: float
    cleanup_value: int


def roi_from_gray(frame: np.ndarray) -> np.ndarray:
    """Return the ROI-black selection of one grayscale mask frame.

    The repository's mask convention puts the region of interest at black (0)
    and everything else at white (255), so the selection is a strict ``< 128``
    test, identical to ``samurai.quadmask._combine_insertion``.
    """
    if frame.ndim != 2:
        raise Pass2MetricsError(f"mask frame must be two-dimensional, got shape {frame.shape}")
    return frame < ROI_THRESHOLD


def _frame_maps(baseline: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-pixel mean absolute difference and mean squared difference.

    The maps are computed in single precision to bound the transient cost of
    one frame; every sum that becomes a reported number is accumulated in
    double precision, so the reported values keep full double accuracy.
    """
    if baseline.shape != other.shape:
        raise Pass2MetricsError(f"frame geometry mismatch: {baseline.shape} != {other.shape}")
    if baseline.ndim != 3 or baseline.shape[2] != 3:
        raise Pass2MetricsError(f"video frames must be RGB, got shape {baseline.shape}")
    delta = baseline.astype(np.float32) - other.astype(np.float32)
    return np.abs(delta).mean(axis=2), (delta * delta).mean(axis=2)


class _RegionAccumulator:
    """Accumulate one region's error sums without retaining any frame."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.pixels = 0
        self.frames_measured = 0
        self.frames_total = 0
        self.absolute_sum = 0.0
        self.squared_sum = 0.0
        self.changed_pixels = 0
        self.max_abs_difference = 0.0

    def add(
        self,
        absolute_map: np.ndarray,
        squared_map: np.ndarray,
        region: np.ndarray,
        *,
        changed_threshold: float,
    ) -> None:
        """Fold one frame's region into the running sums."""
        self.frames_total += 1
        if absolute_map.shape != region.shape:
            raise Pass2MetricsError(
                f"difference map {absolute_map.shape} does not match region {region.shape}"
            )
        if not np.any(region):
            return
        self.frames_measured += 1
        values = absolute_map[region]
        self.pixels += int(values.size)
        self.absolute_sum += float(values.sum(dtype=np.float64))
        self.squared_sum += float(squared_map[region].sum(dtype=np.float64))
        self.changed_pixels += int(np.count_nonzero(values > changed_threshold))
        self.max_abs_difference = max(self.max_abs_difference, float(values.max()))

    def result(self) -> RegionMetrics:
        """Return the measured region, or fail closed when it was never measured."""
        if self.pixels == 0:
            raise Pass2MetricsError(f"region {self.name!r} is empty in every frame")
        mae = self.absolute_sum / self.pixels
        mse = self.squared_sum / self.pixels
        psnr_db = None if mse <= 0.0 else 10.0 * math.log10((MAX_LEVEL * MAX_LEVEL) / mse)
        return RegionMetrics(
            name=self.name,
            pixels=self.pixels,
            frames_measured=self.frames_measured,
            frames_total=self.frames_total,
            mae=mae,
            mse=mse,
            psnr_db=psnr_db,
            changed_pixels=self.changed_pixels,
            changed_ratio=self.changed_pixels / self.pixels,
            max_abs_difference=self.max_abs_difference,
        )


class _TemporalAccumulator:
    """Accumulate the per-frame difference mass and centroid of a clip."""

    def __init__(self) -> None:
        self.masses: list[int] = []
        self.centroids: list[list[float] | None] = []
        self.exactly_identical = True
        self.lead_in_changed_pixels = 0

    def add(
        self,
        absolute_map: np.ndarray,
        *,
        changed_threshold: float,
        structural_lead_in: int,
    ) -> None:
        """Fold one frame's difference map into the temporal series."""
        changed = absolute_map > changed_threshold
        rows, columns = np.nonzero(changed)
        count = int(rows.size)
        self.masses.append(count)
        if count == 0:
            self.centroids.append(None)
        else:
            self.centroids.append([float(rows.mean()), float(columns.mean())])
        if float(absolute_map.max()) > 0.0:
            self.exactly_identical = False
        if len(self.masses) <= structural_lead_in:
            self.lead_in_changed_pixels += count

    def result(self, *, structural_lead_in: int) -> TemporalStability:
        """Reduce the accumulated series into the reported temporal numbers."""
        masses = self.masses
        centroids = self.centroids
        frames = len(masses)
        if structural_lead_in < 0:
            raise Pass2MetricsError(
                f"structural_lead_in must be nonnegative, got {structural_lead_in}"
            )
        window_mass = masses[structural_lead_in:]
        window_centroid = centroids[structural_lead_in:]
        measured = len(window_mass)
        if measured < MIN_MEASURED_FRAMES:
            raise Pass2MetricsError(
                "temporal stability requires at least "
                f"{MIN_MEASURED_FRAMES} measured frames after the structural lead-in "
                f"of {structural_lead_in} frame(s), got {measured}"
            )

        mass_steps = [
            abs(window_mass[index] - window_mass[index - 1]) for index in range(1, measured)
        ]
        centroid_steps: list[float] = []
        for index in range(1, measured):
            current = window_centroid[index]
            previous = window_centroid[index - 1]
            if current is None or previous is None:
                continue
            centroid_steps.append(math.dist(current, previous))
        mass_scale = (
            max(float(np.median(mass_steps)), MIN_MASS_SCALE) if mass_steps else MIN_MASS_SCALE
        )
        centroid_scale = (
            max(float(np.median(centroid_steps)), MIN_CENTROID_SCALE)
            if centroid_steps
            else MIN_CENTROID_SCALE
        )

        empty = [mass == 0 for mass in window_mass]
        dropout = any(empty) and not all(empty)
        if dropout:
            mass_pop = 1.0
            centroid_pop = 1.0
        else:
            mass_pops: list[float] = []
            for index in range(1, measured - 1):
                second = window_mass[index + 1] - 2 * window_mass[index] + window_mass[index - 1]
                mass_pops.append(min(1.0, abs(float(second)) / mass_scale))
            centroid_pops: list[float] = []
            for index in range(1, measured - 1):
                before = window_centroid[index - 1]
                current = window_centroid[index]
                after = window_centroid[index + 1]
                if before is None or current is None or after is None:
                    continue
                second = (
                    after[0] - 2.0 * current[0] + before[0],
                    after[1] - 2.0 * current[1] + before[1],
                )
                centroid_pops.append(min(1.0, math.hypot(second[0], second[1]) / centroid_scale))
            mass_pop = max(mass_pops) if mass_pops else 0.0
            centroid_pop = max(centroid_pops) if centroid_pops else 0.0

        frames_with_difference = sum(1 for mass in masses if mass > 0)
        return TemporalStability(
            frames=frames,
            frames_with_difference=frames_with_difference,
            no_changed_pixels=frames_with_difference == 0,
            exactly_identical=self.exactly_identical,
            measured_frames=measured,
            structural_lead_in=structural_lead_in,
            lead_in_changed_pixels=self.lead_in_changed_pixels,
            lead_in_violation=self.lead_in_changed_pixels > 0,
            mean_mass=float(np.mean(window_mass)),
            mass_step_max=max(mass_steps) if mass_steps else 0,
            mass_pop=mass_pop,
            centroid_pop=centroid_pop,
            temporal_pop_score=0.5 * centroid_pop + 0.5 * mass_pop,
            centroid_step_mean=float(np.mean(centroid_steps)) if centroid_steps else 0.0,
            centroid_step_max=max(centroid_steps) if centroid_steps else 0.0,
            per_frame_mass=masses,
            per_frame_centroid=centroids,
        )


def region_metrics(
    name: str,
    absolute_maps: list[np.ndarray],
    squared_maps: list[np.ndarray],
    regions: list[np.ndarray],
    *,
    changed_threshold: float,
) -> RegionMetrics:
    """Measure baseline-versus-Pass2 agreement inside one region.

    MAE is the mean absolute per-channel error over the region, PSNR is derived
    from the mean squared per-channel error over the same pixels, and the
    changed ratio counts region pixels whose mean absolute difference exceeds
    ``changed_threshold``.  Frames whose region is empty are skipped and
    counted, because a region may legitimately be empty in some frames; a
    region that is empty in every frame fails closed.
    """
    if not (len(absolute_maps) == len(squared_maps) == len(regions)):
        raise Pass2MetricsError("region measurement requires equal, nonempty map lists")
    if not absolute_maps:
        raise Pass2MetricsError("region measurement requires at least one frame")
    _finite("changed_threshold", changed_threshold)
    accumulator = _RegionAccumulator(name)
    for absolute_map, squared_map, region in zip(absolute_maps, squared_maps, regions, strict=True):
        accumulator.add(
            absolute_map,
            squared_map,
            region,
            changed_threshold=changed_threshold,
        )
    return accumulator.result()


def temporal_stability(
    absolute_maps: list[np.ndarray],
    *,
    changed_threshold: float,
    structural_lead_in: int = 0,
) -> TemporalStability:
    """Measure how stable the difference map is over the clip.

    Mass is the number of changed pixels in a frame and the centroid is the
    mean position of those pixels.  Both are reduced to the second-difference
    pop used by ``samurai.insertion``: a steady difference scores zero, a
    flicker scores high.

    The reduction is a maximum over the interior second differences rather
    than an average, so one severe pop is not diluted by a long steady
    stretch.  A difference that is present in some measured frames and absent
    in others is a temporal dropout: the change appeared or vanished inside
    the clip, which the second difference cannot see, so a dropout scores the
    maximum pop.  A difference present in every measured frame is scored by
    the maximum normalized interior second difference.

    ``structural_lead_in`` names the number of leading frames the quadmask
    contract leaves unchanged, so they are excluded from the measured window
    instead of being read as a dropout.  The frames are still counted in
    ``frames`` and reported in ``per_frame_mass``, and a change inside them is
    reported as ``lead_in_changed_pixels``.
    """
    if not absolute_maps:
        raise Pass2MetricsError("temporal stability requires at least one frame")
    _finite("changed_threshold", changed_threshold)
    accumulator = _TemporalAccumulator()
    for absolute_map in absolute_maps:
        accumulator.add(
            absolute_map,
            changed_threshold=changed_threshold,
            structural_lead_in=structural_lead_in,
        )
    return accumulator.result(structural_lead_in=structural_lead_in)


def _lockstep(
    streams: tuple[Iterator[np.ndarray], ...],
) -> Iterator[tuple[int, tuple[np.ndarray, ...]]]:
    """Yield the frames of equal-length streams together, and fail closed otherwise.

    The streams are consumed one frame at a time, so a clip is never held in
    memory as a whole.  A stream that ends before the others is a frame-count
    mismatch, which is an invalid comparison rather than a measurement.
    """
    counts = [0] * len(streams)
    index = 0
    while True:
        frames: list[np.ndarray] = []
        exhausted = 0
        for position, stream in enumerate(streams):
            try:
                frames.append(next(stream))
            except StopIteration:
                exhausted += 1
            else:
                counts[position] += 1
        if exhausted == len(streams):
            return
        if exhausted:
            raise Pass2MetricsError(
                "comparison requires equal, nonempty baseline/pass2/mask frame counts, got "
                f"{counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}"
            )
        yield index, (frames[0], frames[1], frames[2], frames[3])
        index += 1


def compare_sequences(
    *,
    baseline: Iterable[np.ndarray],
    pass2: Iterable[np.ndarray],
    inserted_masks: Iterable[np.ndarray],
    cleanup_masks: Iterable[np.ndarray],
    changed_threshold: float = DEFAULT_CHANGED_THRESHOLD,
    cleanup_value: int = AFFECTED,
) -> FrameComparison:
    """Measure one baseline/Pass 2 frame sequence against its ROI-black masks.

    The three measured regions are the inserted ROI (object preservation), the
    cleanup ROI minus the inserted ROI (the cleanup ring), and everything
    outside both ROIs (untouched background).  A frame whose region is empty
    is skipped and counted; a region that is empty in every frame fails
    closed, because an unmeasurable region cannot be reported as a number.

    ``cleanup_value`` names the VOID quadmask value of the cleanup ring.  With
    127 (affected) VOID clears an affected-only ring on frame zero, so the
    first frame's ring is outside the cleanup ROI and belongs to the
    background region instead, and the whole frame-zero quadmask is keep, so
    frame zero is a structural lead-in that is excluded from the temporal
    window.  With 0 (remove) the ring is a cleanup region in every frame and
    no frame is a lead-in.  The three regions partition every measured pixel
    in every frame under both settings.

    The four sequences are consumed one frame at a time.  No difference map
    survives the frame that produced it, so the memory cost of a comparison is
    one frame rather than one frame per second of clip.
    """
    _finite("changed_threshold", changed_threshold)
    if changed_threshold < 0:
        raise Pass2MetricsError(f"changed_threshold must be nonnegative, got {changed_threshold}")
    if cleanup_value not in CLEANUP_VALUES:
        raise Pass2MetricsError(
            f"cleanup_value must be {REMOVE} (remove) or {AFFECTED} (affected), got {cleanup_value}"
        )
    structural_lead_in = STRUCTURAL_LEAD_IN_FRAMES if cleanup_value == AFFECTED else 0
    object_accumulator = _RegionAccumulator("object_preservation")
    ring_accumulator = _RegionAccumulator("cleanup_ring")
    background_accumulator = _RegionAccumulator("background")
    temporal_accumulator = _TemporalAccumulator()
    frames = 0
    for index, (source, other, inserted_mask, cleanup_mask) in _lockstep(
        (
            iter(baseline),
            iter(pass2),
            iter(inserted_masks),
            iter(cleanup_masks),
        )
    ):
        if source.shape[:2] != inserted_mask.shape or source.shape[:2] != cleanup_mask.shape:
            raise Pass2MetricsError(
                f"frame {index}: mask geometry {inserted_mask.shape}/{cleanup_mask.shape} "
                f"does not match video {source.shape[:2]}"
            )
        absolute_map, squared_map = _frame_maps(source, other)
        inserted_roi = roi_from_gray(inserted_mask)
        cleanup_ring = roi_from_gray(cleanup_mask) & ~inserted_roi
        if structural_lead_in and index < structural_lead_in:
            cleanup_ring = np.zeros_like(cleanup_ring)
        background = ~(inserted_roi | cleanup_ring)
        object_accumulator.add(
            absolute_map, squared_map, inserted_roi, changed_threshold=changed_threshold
        )
        ring_accumulator.add(
            absolute_map, squared_map, cleanup_ring, changed_threshold=changed_threshold
        )
        background_accumulator.add(
            absolute_map, squared_map, background, changed_threshold=changed_threshold
        )
        temporal_accumulator.add(
            absolute_map,
            changed_threshold=changed_threshold,
            structural_lead_in=structural_lead_in,
        )
        frames += 1
    if frames == 0:
        raise Pass2MetricsError(
            "comparison requires equal, nonempty baseline/pass2/mask frame counts, got 0/0/0/0"
        )
    return FrameComparison(
        object_preservation=object_accumulator.result(),
        cleanup_ring=ring_accumulator.result(),
        background=background_accumulator.result(),
        temporal=temporal_accumulator.result(structural_lead_in=structural_lead_in),
        changed_threshold=changed_threshold,
        cleanup_value=cleanup_value,
    )


def _alignment_record(name: str, info: VideoInfo) -> InputAlignment:
    return InputAlignment(
        name=name,
        width=info.width,
        height=info.height,
        frames=info.frames,
        fps=info.fps,
    )


def probe_frame_timestamps(path: Path, *, name: str) -> FrameTimestamps:
    """Return the exact per-frame presentation timestamps of one video.

    The shared ``samurai.media`` probe reports a nominal frame rate, and a
    nominal rate cannot separate a constant frame rate clip from a variable
    one.  This probe reads the frame timestamps as integers in the stream time
    base, so every timestamp is an exact rational number and two timelines can
    be compared without a float tolerance.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=time_base:frame=pts,best_effort_timestamp",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise Pass2MetricsError(f"ffprobe failed for {path}: {error}") from error
    try:
        payload = json.loads(completed.stdout)
        time_base = Fraction(payload["streams"][0]["time_base"])
        frames = payload["frames"]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise Pass2MetricsError(f"could not parse ffprobe timestamps for {path}") from error
    pts: list[Fraction] = []
    for frame in frames:
        raw = frame.get("pts", frame.get("best_effort_timestamp"))
        if raw is None:
            raise Pass2MetricsError(f"frame without a timestamp in {path}")
        pts.append(Fraction(int(raw)) * time_base)
    if not pts:
        raise Pass2MetricsError(f"no frame timestamps decoded from {path}")
    deltas = tuple(pts[index + 1] - pts[index] for index in range(len(pts) - 1))
    variable = len(set(deltas)) > 1
    return FrameTimestamps(
        name=name,
        count=len(pts),
        pts=tuple(pts),
        variable_frame_rate=variable,
        cfr_delta=None if variable or not deltas else deltas[0],
        digest=_timestamp_digest(tuple(pts)),
    )


def check_timestamp_alignment(
    *,
    baseline: FrameTimestamps,
    pass2: FrameTimestamps,
    inserted: FrameTimestamps,
    cleanup: FrameTimestamps,
) -> TimestampAlignment:
    """Fail closed unless the four inputs share one exact presentation timeline.

    Every frame of the two videos and the two masks must carry the same exact
    timestamp, and the four inputs must agree on whether the clip has a
    constant or a variable frame rate.  A frame-by-frame comparison across two
    different timelines is not a measurement, so a mismatch is an error rather
    than a reported metric.
    """
    records = (baseline, pass2, inserted, cleanup)
    if len({record.count for record in records}) != 1:
        raise Pass2MetricsError(
            "timestamp count mismatch: baseline "
            f"{baseline.count}, pass2 {pass2.count}, inserted mask {inserted.count}, "
            f"cleanup mask {cleanup.count}"
        )
    if len({record.variable_frame_rate for record in records}) != 1:
        raise Pass2MetricsError(
            "frame rate character mismatch: "
            + ", ".join(
                f"{record.name} is "
                f"{'variable' if record.variable_frame_rate else 'constant'} frame rate"
                for record in records
            )
        )
    for index in range(baseline.count):
        if len({record.pts[index] for record in records}) != 1:
            raise Pass2MetricsError(
                f"timestamp mismatch at frame {index}: "
                + ", ".join(
                    f"{record.name} {fraction_text(record.pts[index])}" for record in records
                )
            )
    return TimestampAlignment(
        baseline=baseline,
        pass2=pass2,
        inserted_mask=inserted,
        cleanup_mask=cleanup,
        pts_aligned=True,
        frame_rate_character_aligned=True,
        aligned=True,
    )


def check_alignment(
    *,
    baseline: VideoInfo,
    pass2: VideoInfo,
    inserted: VideoInfo,
    cleanup: VideoInfo,
    fps_tolerance: float = DEFAULT_FPS_TOLERANCE,
    require_nominal_fps: bool = True,
) -> FrameAlignment:
    """Return the alignment record, or fail closed when the inputs disagree.

    Geometry, frame count, and frame rate must agree across the two videos and
    their two masks.  A frame-by-frame comparison across different geometries,
    different frame counts, or different frame rates is not a measurement, so
    the mismatch is an error rather than a reported metric.

    The nominal frame rate of a variable frame rate clip is not a timing
    contract, so ``require_nominal_fps`` lets the caller drop that gate when
    the exact per-frame timestamps are the authority.
    """
    _finite("fps_tolerance", fps_tolerance)
    if fps_tolerance < 0:
        raise Pass2MetricsError(f"fps_tolerance must be nonnegative, got {fps_tolerance}")
    dimensions_aligned = (
        baseline.width == pass2.width == inserted.width == cleanup.width
        and baseline.height == pass2.height == inserted.height == cleanup.height
    )
    frames_aligned = baseline.frames == pass2.frames == inserted.frames == cleanup.frames
    fps_values = [baseline.fps, pass2.fps, inserted.fps, cleanup.fps]
    fps_delta = max(fps_values) - min(fps_values)
    fps_aligned = not require_nominal_fps or fps_delta <= fps_tolerance * max(max(fps_values), 1e-9)
    if not dimensions_aligned:
        raise Pass2MetricsError(
            "geometry mismatch: baseline "
            f"{baseline.width}x{baseline.height}, pass2 {pass2.width}x{pass2.height}, "
            f"inserted mask {inserted.width}x{inserted.height}, "
            f"cleanup mask {cleanup.width}x{cleanup.height}"
        )
    if not frames_aligned:
        raise Pass2MetricsError(
            f"frame count mismatch: baseline {baseline.frames}, pass2 {pass2.frames}, "
            f"inserted mask {inserted.frames}, cleanup mask {cleanup.frames}"
        )
    if not fps_aligned:
        raise Pass2MetricsError(
            f"frame rate mismatch: baseline {baseline.fps}, pass2 {pass2.fps}, "
            f"inserted mask {inserted.fps}, cleanup mask {cleanup.fps}"
        )
    return FrameAlignment(
        baseline=_alignment_record("baseline", baseline),
        pass2=_alignment_record("pass2", pass2),
        inserted_mask=_alignment_record("inserted_mask", inserted),
        cleanup_mask=_alignment_record("cleanup_mask", cleanup),
        dimensions_aligned=dimensions_aligned,
        frames_aligned=frames_aligned,
        fps_aligned=fps_aligned,
        aligned=dimensions_aligned and frames_aligned and fps_aligned,
        fps_delta=fps_delta,
        fps_tolerance=fps_tolerance,
    )


def _counted(
    frames: Iterable[np.ndarray],
    *,
    expected: int,
    source: Path,
) -> Iterator[np.ndarray]:
    """Yield a decoded stream lazily and verify it matched the probe frame count."""
    decoded = 0
    for frame in frames:
        decoded += 1
        yield frame
    if decoded != expected:
        raise Pass2MetricsError(f"decoded {decoded} frames from {source}, expected {expected}")


def compare_videos(
    *,
    baseline: Path,
    pass2: Path,
    inserted_mask: Path,
    cleanup_mask: Path,
    changed_threshold: float = DEFAULT_CHANGED_THRESHOLD,
    fps_tolerance: float = DEFAULT_FPS_TOLERANCE,
    cleanup_value: int = AFFECTED,
    workdir: Path | None = None,
) -> Pass2Comparison:
    """Probe, decode, and compare a baseline insert against a Pass 2 video.

    The inputs are the baseline inserted video, the Pass 2 video, the ROI-black
    inserted mask, and the ROI-black cleanup mask.  Every input is probed with
    the shared ``samurai.media`` contract before any frame is decoded, and the
    comparison fails closed when the probed geometry, frame count, or frame
    rate disagree, when the exact per-frame timestamps disagree, or when one
    input has a constant frame rate and another a variable one.

    Frames are decoded lazily and consumed in lockstep, so no input is ever
    materialized as a whole clip in memory.  Only the per-region error sums
    and the per-frame difference mass and centroid survive a frame.
    """
    _finite("changed_threshold", changed_threshold)
    _finite("fps_tolerance", fps_tolerance)
    if changed_threshold < 0:
        raise Pass2MetricsError(f"changed_threshold must be nonnegative, got {changed_threshold}")
    if fps_tolerance < 0:
        raise Pass2MetricsError(f"fps_tolerance must be nonnegative, got {fps_tolerance}")
    if cleanup_value not in CLEANUP_VALUES:
        raise Pass2MetricsError(
            f"cleanup_value must be {REMOVE} (remove) or {AFFECTED} (affected), got {cleanup_value}"
        )
    baseline_info = probe_video(baseline)
    pass2_info = probe_video(pass2)
    inserted_info = probe_video(inserted_mask)
    cleanup_info = probe_video(cleanup_mask)
    baseline_pts = probe_frame_timestamps(baseline, name="baseline")
    pass2_pts = probe_frame_timestamps(pass2, name="pass2")
    inserted_pts = probe_frame_timestamps(inserted_mask, name="inserted_mask")
    cleanup_pts = probe_frame_timestamps(cleanup_mask, name="cleanup_mask")
    alignment = check_alignment(
        baseline=baseline_info,
        pass2=pass2_info,
        inserted=inserted_info,
        cleanup=cleanup_info,
        fps_tolerance=fps_tolerance,
        require_nominal_fps=not baseline_pts.variable_frame_rate,
    )
    timestamps = check_timestamp_alignment(
        baseline=baseline_pts,
        pass2=pass2_pts,
        inserted=inserted_pts,
        cleanup=cleanup_pts,
    )
    for label, info, record in (
        ("baseline", baseline_info, timestamps.baseline),
        ("pass2", pass2_info, timestamps.pass2),
        ("inserted mask", inserted_info, timestamps.inserted_mask),
        ("cleanup mask", cleanup_info, timestamps.cleanup_mask),
    ):
        if info.frames != record.count:
            raise Pass2MetricsError(
                f"frame count mismatch for {label}: probe reported {info.frames} frames, "
                f"timestamps reported {record.count}"
            )
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pass2-compare-", dir=workdir) as temporary:
        root = Path(temporary)
        baseline_frames = _counted(
            iter_color_frames(baseline, root / "baseline"),
            expected=baseline_info.frames,
            source=baseline,
        )
        pass2_frames = _counted(
            iter_color_frames(pass2, root / "pass2"),
            expected=pass2_info.frames,
            source=pass2,
        )
        inserted_frames = iter_gray_frames(
            inserted_mask, root / "inserted", expected_frames=inserted_info.frames
        )
        cleanup_frames = iter_gray_frames(
            cleanup_mask, root / "cleanup", expected_frames=cleanup_info.frames
        )
        comparison = compare_sequences(
            baseline=baseline_frames,
            pass2=pass2_frames,
            inserted_masks=inserted_frames,
            cleanup_masks=cleanup_frames,
            changed_threshold=changed_threshold,
            cleanup_value=cleanup_value,
        )
    return Pass2Comparison(
        frame_alignment=alignment,
        timestamps=timestamps,
        object_preservation=comparison.object_preservation,
        cleanup_ring=comparison.cleanup_ring,
        background=comparison.background,
        temporal=comparison.temporal,
        changed_threshold=comparison.changed_threshold,
        cleanup_value=comparison.cleanup_value,
    )
