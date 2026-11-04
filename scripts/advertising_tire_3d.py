#!/usr/bin/env python3
"""Render a true Blender-native three-dimensional advertising tire.

The lane compiles a bounded English target description into a wheel
specification, decodes the ROI-black tire mask, reuses the fitted and smoothed
tire poses of the advertising tire lane, maps every two-dimensional mask
ellipse onto a three-dimensional wheel rotation, scale, and position, writes
the pose JSON, invokes Blender, composites the true-3D RGBA renders over the
clean plate, and emits the ROI-black inserted-object and cleanup-ring masks
together with a JSON receipt.

The tire is a real Blender mesh, not a painted 2.5D sprite: the carcass, tread
blocks, rim, hub, spokes, and bent sidewall label are three-dimensional
geometry rendered with Cycles.  No third-party asset, checkpoint, credential,
network call, or paid API runs in this lane.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from advertising_tire_asset import (
    AdvertisingTireAssetError,
    TireFit,
    build_inserted_cleanup_masks,
    fit_ellipse,
    load_primary_mask,
    smooth_fit_series,
    write_color_video,
)
from PIL import Image

from samurai.media import (
    GrayVideoWriter,
    MediaError,
    assert_video_alignment,
    iter_color_frames,
    probe_video,
)

SCHEMA = "void-fusion-lab.advertising-tire-3d.v1"
POSE_SCHEMA = "void-fusion-lab.tire-wheel-poses.v1"
EXIT_INPUT_ERROR = 2
DEFAULT_RENDER_WIDTH = 672
DEFAULT_RENDER_HEIGHT = 384
DEFAULT_SAMPLES = 32
DEFAULT_CLEANUP_RADIUS = 16
DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_DEVICE = "CPU"
METERS_PER_PIXEL = 0.01
MAX_DESCRIPTION_LENGTH = 200
MAX_TILT_DEGREES = 78.0
MIN_RADIUS_PIXELS = 1.0

COLOR_WORDS: dict[str, tuple[float, float, float]] = {
    "black": (0.035, 0.035, 0.045),
    "white": (0.86, 0.86, 0.86),
    "silver": (0.62, 0.63, 0.66),
    "gray": (0.26, 0.27, 0.29),
    "grey": (0.26, 0.27, 0.29),
    "red": (0.52, 0.05, 0.05),
    "blue": (0.06, 0.11, 0.42),
    "green": (0.06, 0.28, 0.11),
    "yellow": (0.86, 0.62, 0.04),
    "orange": (0.85, 0.33, 0.03),
}
TARGET_WORDS = frozenset({"tire", "tyre", "wheel"})
BAND_WORDS = frozenset({"band", "stripe", "lettering", "letters", "text", "label", "decal"})
LABEL_MARKERS = frozenset(
    {"labeled", "labelled", "branded", "branding", "word", "words", "reading", "says"}
)
ARTICLES = frozenset({"a", "an", "the"})
FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "with",
        "and",
        "on",
        "in",
        "that",
        "has",
        "have",
        "featuring",
        "showing",
        "for",
        "to",
        "my",
        "this",
        "one",
        "please",
        "render",
        "replace",
        "insert",
        "over",
        "into",
        "from",
        "new",
        "spare",
        "front",
        "rear",
        "left",
        "right",
        "side",
        "visible",
        "ad",
        "advertisement",
        "advertising",
        "tyre",
        "tire",
        "wheel",
    }
)
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9]{1,16}$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

NON_CLAIMS = (
    "The replacement tire is a procedural Blender mesh fitted to a "
    "two-dimensional mask ellipse; it is not a scanned, measured, or "
    "photographic reconstruction of the filmed wheel.",
    "The wheel pose comes from the mask ellipse, the modelled tyre profile, "
    "and the rolling relation between the fitted centre motion and the "
    "fitted radius; it is not a recovered 3D ground-truth pose.",
    "The wheel tilt is solved so that the modelled tyre's projected "
    "silhouette matches the fitted ellipse ratio; the tilt is bounded at 78 "
    "degrees, so a strongly foreshortened ellipse saturates and the rendered "
    "tyre stays wider than the fitted ellipse.",
    "The label is procedural Blender text compiled from the bounded target; "
    "it is not an official brand asset or typographic match.",
    "Compositing the render over the clean plate does not prove that the "
    "original tire was removed; without a VOID Pass 1 plate the original "
    "tire remains outside the rendered footprint.",
    "The cleanup ring is a deterministic dilation of the rendered alpha, not "
    "a model-reasoned shadow, contact, or interaction region.",
    "No SAM 3.1, VOID, perception model, paid API, credential, or network call runs in this lane.",
)


class AdvertisingTire3DError(RuntimeError):
    """An input, geometry, Blender, or media operation for the lane failed."""


@dataclass(frozen=True)
class TireTargetSpec:
    """The compiled wheel specification for one English target description."""

    description: str
    target: str
    tread_color: tuple[float, float, float]
    band_color: tuple[float, float, float]
    label: str | None
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class WheelProfile:
    """The modelled tyre profile that maps a mask ellipse onto a wheel tilt.

    The profile is read from the in-Blender render script so that the pose
    solver and the rendered geometry can never drift apart: the render
    manifest repeats it and the lane fails closed on a mismatch.
    """

    major_radius: float
    minor_radius: float
    tread_half_depth: float
    outer_radius: float


@dataclass(frozen=True)
class WheelPose:
    """One frame's fitted wheel pose in image pixels and Blender world units."""

    frame: int
    source_center_x: float
    source_center_y: float
    radius_x: float
    radius_y: float
    angle_degrees: float
    position: tuple[float, float, float]
    rotation_euler_degrees: tuple[float, float, float]
    rotation_matrix: tuple[tuple[float, float, float], ...]
    axis: tuple[float, float, float]
    tilt_degrees: float
    spin_degrees: float
    scale: float
    held: bool


def _extract_label(tokens: list[str], description: str) -> tuple[str | None, int]:
    """Return the bounded label of a description and the token index that holds it."""
    for index in range(2, len(tokens)):
        if (
            tokens[index] == "label"
            and tokens[index - 2] == "and"
            and LABEL_PATTERN.fullmatch(tokens[index - 1])
            and tokens[index - 1] not in COLOR_WORDS
            and tokens[index - 1] not in TARGET_WORDS
        ):
            return tokens[index - 1].upper(), index - 1
    for index, token in enumerate(tokens):
        if token not in LABEL_MARKERS:
            continue
        cursor = index + 1
        if (
            cursor + 1 < len(tokens)
            and tokens[cursor] in ARTICLES
            and LABEL_PATTERN.fullmatch(tokens[cursor + 1])
            and tokens[cursor + 1] not in COLOR_WORDS
            and tokens[cursor + 1] not in TARGET_WORDS
        ):
            cursor += 1
        if cursor >= len(tokens):
            raise AdvertisingTire3DError(
                f"the label marker {token!r} in {description!r} is followed by no label"
            )
        candidate = tokens[cursor]
        if not LABEL_PATTERN.fullmatch(candidate):
            raise AdvertisingTire3DError(
                f"the label {candidate!r} in {description!r} is not a short alphanumeric word"
            )
        return candidate.upper(), cursor
    return None, -1


def _extract_colors(
    tokens: list[str],
    label_index: int,
    description: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the tread and band colors named by one description."""
    tread: tuple[float, float, float] | None = None
    band: tuple[float, float, float] | None = None
    for index, token in enumerate(tokens):
        if token not in COLOR_WORDS or index == label_index:
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if following in BAND_WORDS:
            if band is not None and band != COLOR_WORDS[token]:
                raise AdvertisingTire3DError(
                    f"the target description {description!r} names two band colors"
                )
            band = COLOR_WORDS[token]
            continue
        if tread is not None and tread != COLOR_WORDS[token]:
            raise AdvertisingTire3DError(
                f"the target description {description!r} names two tread colors"
            )
        tread = COLOR_WORDS[token]
    return tread or COLOR_WORDS["black"], band or COLOR_WORDS["yellow"]


def compile_target(description: str) -> TireTargetSpec:
    """Compile one bounded English target description into a wheel specification.

    The compiler is a closed vocabulary: the description must name a tire, a
    tyre, or a wheel, and every token must be a known target, color, band,
    label, or filler word.  Anything else fails closed instead of guessing.
    """
    text = " ".join(str(description).split())
    if not text:
        raise AdvertisingTire3DError("the target description is empty")
    if len(text) > MAX_DESCRIPTION_LENGTH:
        raise AdvertisingTire3DError(
            f"the target description is {len(text)} characters, "
            f"the bound is {MAX_DESCRIPTION_LENGTH}"
        )
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        raise AdvertisingTire3DError(f"the target description {description!r} holds no words")
    target = next((token for token in tokens if token in TARGET_WORDS), None)
    if target is None:
        raise AdvertisingTire3DError(
            f"unsupported target: {description!r} names no tire, tyre, or wheel"
        )
    label, label_index = _extract_label(tokens, description)
    for index, token in enumerate(tokens):
        if index == label_index:
            continue
        if token in TARGET_WORDS or token in COLOR_WORDS or token in BAND_WORDS:
            continue
        if token in LABEL_MARKERS or token in FILLER_WORDS:
            continue
        raise AdvertisingTire3DError(
            f"unsupported token {token!r} in the target description {description!r}"
        )
    tread, band = _extract_colors(tokens, label_index, description)
    return TireTargetSpec(
        description=text,
        target="tire",
        tread_color=tread,
        band_color=band,
        label=label,
        tokens=tuple(tokens),
    )


def rotation_x(degrees: float) -> np.ndarray:
    """Return the rotation matrix about the X axis."""
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]], dtype=np.float64)


def rotation_z(degrees: float) -> np.ndarray:
    """Return the rotation matrix about the Z axis."""
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def wheel_matrix(angle_degrees: float, tilt_degrees: float, spin_degrees: float) -> np.ndarray:
    """Return the wheel orientation that matches one fitted mask ellipse.

    The fitted ellipse is the projection of a circle, so the major axis is the
    unforeshortened wheel radius and the minor axis carries the tilt.  The
    wheel axis is aligned from the major-axis direction, and the spin is the
    rolling angle of the wheel about its own axis.
    """
    align = rotation_z(-angle_degrees)
    tilt = rotation_x(-tilt_degrees)
    spin = rotation_z(spin_degrees)
    return align @ tilt @ spin


def matrix_from_euler(
    alpha_degrees: float, beta_degrees: float, gamma_degrees: float
) -> np.ndarray:
    """Return the XYZ Euler rotation matrix of three angles in degrees."""
    alpha = math.radians(alpha_degrees)
    beta = math.radians(beta_degrees)
    gamma = math.radians(gamma_degrees)
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=np.float64)
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=np.float64)
    rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return the XYZ Euler angles of one rotation matrix in degrees.

    The extraction follows Blender's XYZ Euler order, where the composed
    rotation is ``Rz(gamma) @ Ry(beta) @ Rx(alpha)``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    beta = math.asin(float(np.clip(-m[2, 0], -1.0, 1.0)))
    cos_beta = math.cos(beta)
    if abs(cos_beta) < 1e-9:
        alpha = 0.0
        gamma = math.atan2(-m[0, 1], m[1, 1])
    else:
        alpha = math.atan2(m[2, 1], m[2, 2])
        gamma = math.atan2(m[1, 0], m[0, 0])
    return (math.degrees(alpha), math.degrees(beta), math.degrees(gamma))


def content_box(
    source_width: int,
    source_height: int,
    render_width: int,
    render_height: int,
) -> tuple[int, int, int, int]:
    """Return the render rectangle that holds the source geometry.

    The render canvas keeps its fixed size, so the source frame is placed in
    the largest centred rectangle of the canvas with the source aspect ratio.
    The composite crops that rectangle, which keeps the wheel's shape when
    the render is resampled back onto the clean plate.
    """
    if source_width <= 0 or source_height <= 0:
        raise AdvertisingTire3DError("the source geometry must be positive")
    factor = min(render_width / source_width, render_height / source_height)
    box_width = min(render_width, max(1, int(round(source_width * factor))))
    box_height = min(render_height, max(1, int(round(source_height * factor))))
    left = (render_width - box_width) // 2
    top = (render_height - box_height) // 2
    return (left, top, left + box_width, top + box_height)


def load_wheel_profile(script: Path) -> WheelProfile:
    """Read the modelled profile constants without executing the render script."""
    names = {
        "TIRE_MAJOR_RADIUS": "major_radius",
        "TIRE_MINOR_RADIUS": "minor_radius",
        "TREAD_DEPTH": "tread_depth",
    }
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except (OSError, SyntaxError) as error:
        raise AdvertisingTire3DError(
            f"could not parse the wheel profile from {script}: {error}"
        ) from error
    values: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            value = (
                getattr(node.value, "value", None) if isinstance(node.value, ast.Constant) else None
            )
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AdvertisingTire3DError(
                    f"the render script {script} defines {target.id} as a non-literal"
                )
            values[target.id] = float(value)
    missing = sorted(set(names) - set(values))
    if missing:
        raise AdvertisingTire3DError(f"the render script {script} omits profile {missing}")
    try:
        profile = WheelProfile(
            major_radius=values["TIRE_MAJOR_RADIUS"],
            minor_radius=values["TIRE_MINOR_RADIUS"],
            tread_half_depth=values["TREAD_DEPTH"] / 2.0,
            outer_radius=values["TIRE_MAJOR_RADIUS"] + values["TIRE_MINOR_RADIUS"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AdvertisingTire3DError(f"could not build a wheel profile from {script}") from error
    if profile.major_radius <= 0 or profile.minor_radius <= 0 or profile.tread_half_depth <= 0:
        raise AdvertisingTire3DError(f"the wheel profile from {script} holds a nonpositive value")
    return profile


def projected_minor_ratio(profile: WheelProfile, tilt_degrees: float) -> float:
    """Return the projected minor semi-axis of the modelled tyre per unit radius.

    A tyre is a thick torus, not a thin disc, so its projected silhouette is
    fatter than the projection of its outer circle.  The torus contributes
    ``major * cos(tilt) + minor`` and the tread blocks contribute
    ``outer * cos(tilt) + tread_half_depth * sin(tilt)``; the larger of the
    two bounds the projected minor semi-axis.
    """
    tilt = math.radians(tilt_degrees)
    torus = profile.major_radius * math.cos(tilt) + profile.minor_radius
    tread = profile.outer_radius * math.cos(tilt) + profile.tread_half_depth * math.sin(tilt)
    return max(torus, tread) / profile.outer_radius


def solve_tilt_degrees(
    profile: WheelProfile,
    ratio: float,
    *,
    max_tilt: float = MAX_TILT_DEGREES,
) -> float:
    """Return the wheel tilt whose projected silhouette matches the mask ratio.

    The search returns the smallest tilt whose projected minor semi-axis is at
    or below the fitted ratio, so a face-on ellipse keeps a zero tilt and a
    strongly foreshortened ellipse saturates at the tilt bound.
    """
    if ratio >= 1.0:
        return 0.0
    step = 0.05
    low = 0.0
    high: float | None = None
    tilt = 0.0
    while tilt <= max_tilt:
        if projected_minor_ratio(profile, tilt) <= ratio:
            high = tilt
            break
        low = tilt
        tilt += step
    if high is None:
        return max_tilt
    for _ in range(48):
        middle = 0.5 * (low + high)
        if projected_minor_ratio(profile, middle) <= ratio:
            high = middle
        else:
            low = middle
    return high


def fit_wheel_poses(
    fits: list[TireFit],
    *,
    profile: WheelProfile,
    source_width: int,
    source_height: int,
    render_width: int,
    render_height: int,
    meters_per_pixel: float,
) -> list[WheelPose]:
    """Map every fitted mask ellipse onto a three-dimensional wheel pose.

    The ellipse is the projection of a circle, so the major semi-axis is the
    wheel radius and the ratio of the two semi-axes gives the tilt of the
    wheel plane away from the camera.  The spin follows the rolling relation
    between the fitted centre motion along the major axis and the fitted
    radius, so the tread advances with the wheel instead of sliding.
    """
    if not fits:
        raise AdvertisingTire3DError("the lane holds no fitted wheel pose")
    left, top, right, bottom = content_box(source_width, source_height, render_width, render_height)
    factor_x = (right - left) / source_width
    factor_y = (bottom - top) / source_height
    centers: list[tuple[float, float]] = []
    radii: list[tuple[float, float]] = []
    for fit in fits:
        center_x = left + fit.center_x * factor_x
        center_y = top + fit.center_y * factor_y
        centers.append((center_x, center_y))
        radii.append((fit.radius_x * factor_x, fit.radius_y * factor_y))
    spins: list[float] = []
    cumulative = 0.0
    for index, fit in enumerate(fits):
        if index > 0:
            travel_x = fits[index].center_x - fits[index - 1].center_x
            travel_y = fits[index].center_y - fits[index - 1].center_y
            theta = math.radians(fit.angle_degrees)
            travel = travel_x * math.cos(theta) + travel_y * math.sin(theta)
            radius = max(fit.radius_x, MIN_RADIUS_PIXELS)
            cumulative += -travel / radius
        spins.append(math.degrees(cumulative))
    poses: list[WheelPose] = []
    for index, fit in enumerate(fits):
        center_x, center_y = centers[index]
        radius_x, radius_y = radii[index]
        ratio = min(1.0, max(0.0, radius_y / max(radius_x, MIN_RADIUS_PIXELS)))
        tilt = solve_tilt_degrees(profile, ratio)
        matrix = wheel_matrix(fit.angle_degrees, tilt, spins[index])
        euler = euler_from_matrix(matrix)
        position = (
            (center_x - render_width / 2.0) * meters_per_pixel,
            (render_height / 2.0 - center_y) * meters_per_pixel,
            0.0,
        )
        poses.append(
            WheelPose(
                frame=index,
                source_center_x=fit.center_x,
                source_center_y=fit.center_y,
                radius_x=fit.radius_x,
                radius_y=fit.radius_y,
                angle_degrees=fit.angle_degrees,
                position=position,
                rotation_euler_degrees=euler,
                rotation_matrix=(
                    (float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])),
                    (float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])),
                    (float(matrix[2][0]), float(matrix[2][1]), float(matrix[2][2])),
                ),
                axis=(float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])),
                tilt_degrees=tilt,
                spin_degrees=spins[index],
                scale=radius_x * meters_per_pixel,
                held=fit.held,
            )
        )
    return poses


def profile_record(profile: WheelProfile) -> dict[str, float]:
    """Return the modelled tyre profile as a JSON record."""
    return {
        "major_radius": profile.major_radius,
        "minor_radius": profile.minor_radius,
        "tread_half_depth": profile.tread_half_depth,
        "outer_radius": profile.outer_radius,
    }


def pose_payload(
    poses: list[WheelPose],
    spec: TireTargetSpec,
    *,
    profile: WheelProfile,
    source_width: int,
    source_height: int,
    render_width: int,
    render_height: int,
    meters_per_pixel: float,
) -> dict[str, Any]:
    """Return the pose JSON document handed to the in-Blender render script."""
    return {
        "schema": POSE_SCHEMA,
        "width": render_width,
        "height": render_height,
        "source_width": source_width,
        "source_height": source_height,
        "meters_per_pixel": meters_per_pixel,
        "frame_count": len(poses),
        "profile": profile_record(profile),
        "target": {
            "description": spec.description,
            "tread_color": list(spec.tread_color),
            "band_color": list(spec.band_color),
            "label": spec.label,
        },
        "poses": [asdict(pose) for pose in poses],
    }


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_blender(
    blender: Path,
    script: Path,
    poses: Path,
    output: Path,
    *,
    samples: int,
    device: str,
    blend: Path | None,
    glb: Path | None,
) -> None:
    """Invoke Blender in the background to render every wheel pose."""
    if output.exists():
        raise AdvertisingTire3DError(f"the Blender output directory already exists: {output}")
    command = [
        str(blender),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "--poses",
        str(poses),
        "--out",
        str(output),
        "--samples",
        str(samples),
        "--device",
        device,
        "--blender-executable",
        str(blender),
    ]
    if blend is not None:
        command += ["--blend", str(blend)]
    if glb is not None:
        command += ["--glb", str(glb)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise AdvertisingTire3DError(f"the Blender render exited {completed.returncode}{suffix}")


def verify_render_manifest(
    manifest: Any,
    *,
    blender_sha256: str,
    script_sha256: str,
    pose_sha256: str,
    samples: int,
    width: int,
    height: int,
    frames: int,
    profile: WheelProfile,
    device: str,
) -> dict[str, Any]:
    """Fail closed unless the Blender render manifest matches this lane."""
    required = {
        "blender_executable_sha256",
        "blender_version",
        "counts",
        "device",
        "engine",
        "frames",
        "height",
        "pose_sha256",
        "samples",
        "script_sha256",
        "width",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise AdvertisingTire3DError("the Blender render manifest is missing required fields")
    expected = {
        "blender_executable_sha256": blender_sha256,
        "device": device,
        "engine": "CYCLES",
        "frames": frames,
        "height": height,
        "pose_sha256": pose_sha256,
        "samples": samples,
        "script_sha256": script_sha256,
        "width": width,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise AdvertisingTire3DError(
                f"the Blender render manifest {key} is {manifest.get(key)!r}, expected {value!r}"
            )
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or int(counts.get("mesh_objects", 0)) < 1:
        raise AdvertisingTire3DError("the Blender render manifest records no mesh objects")
    recorded = manifest.get("profile")
    if recorded != profile_record(profile):
        raise AdvertisingTire3DError(
            f"the Blender render manifest profile {recorded!r} does not match the "
            f"modelled profile {profile_record(profile)!r}"
        )
    return manifest


def load_render(
    path: Path,
    *,
    width: int,
    height: int,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Load one rendered RGBA frame, crop it, and resample it to the plate."""
    with Image.open(path) as image:
        render = np.asarray(image.convert("RGBA")).copy()
    left, top, right, bottom = box
    crop = render[top:bottom, left:right]
    if crop.shape[:2] != (height, width):
        resampled = Image.fromarray(crop, mode="RGBA").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        crop = np.asarray(resampled).copy()
    return crop


def composite_render(plate: np.ndarray, render: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Composite one rendered wheel over one clean plate frame."""
    alpha = render[:, :, 3].astype(np.float64) / 255.0
    rgb = render[:, :, :3].astype(np.float64)
    base = plate.astype(np.float64)
    composite = base * (1.0 - alpha[..., None]) + rgb * alpha[..., None]
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8), alpha


def _source_revision(root: Path) -> dict[str, Any]:
    """Return the checked-out revision and dirty flag for the receipt."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": bool(status)}


def _reject_path_aliases(paths: dict[str, Path | None]) -> None:
    """Fail closed when two CLI roles resolve to one filesystem path."""
    resolved: dict[Path, str] = {}
    for label, path in paths.items():
        if path is None:
            continue
        key = path.expanduser().resolve(strict=False)
        if key in resolved:
            raise AdvertisingTire3DError(
                f"--{label.replace('_', '-')} and --{resolved[key].replace('_', '-')} "
                "must name different files"
            )
        resolved[key] = label


def run_lane(args: argparse.Namespace) -> dict[str, Any]:
    """Run the true-3D advertising tire lane and return its JSON receipt."""
    spec = compile_target(args.target)
    _reject_path_aliases(
        {
            "input": args.input,
            "primary_mask": args.primary_mask,
            "plate": args.plate,
            "blender": args.blender,
            "blender_script": args.blender_script,
            "output": args.output,
            "inserted_mask": args.inserted_mask,
            "cleanup_mask": args.cleanup_mask,
            "poses": args.poses,
            "render_manifest": args.render_manifest,
            "report": args.report,
            "blend": args.blend,
            "glb": args.glb,
        }
    )
    if args.cleanup_radius < 1:
        raise AdvertisingTire3DError("--cleanup-radius must be a positive integer")
    if args.smoothing_window < 1:
        raise AdvertisingTire3DError("--smoothing-window must be a positive integer")
    if args.samples < 1:
        raise AdvertisingTire3DError("--samples must be a positive integer")
    if args.render_width < 8 or args.render_height < 8:
        raise AdvertisingTire3DError("--render-width and --render-height must be at least 8 pixels")
    for label, path in (
        ("input video", args.input),
        ("primary mask", args.primary_mask),
        ("clean plate", args.plate),
        ("Blender executable", args.blender),
        ("Blender script", args.blender_script),
    ):
        if not Path(path).is_file():
            raise AdvertisingTire3DError(f"the {label} does not exist: {path}")
    info = probe_video(args.input)
    if args.cleanup_radius * 2 >= min(info.width, info.height):
        raise AdvertisingTire3DError(
            "--cleanup-radius must be less than half the shortest source dimension"
        )
    mask_info = probe_video(args.primary_mask)
    plate_info = probe_video(args.plate)
    for name, other in (("primary mask", mask_info), ("clean plate", plate_info)):
        if (other.width, other.height, other.frames) != (info.width, info.height, info.frames):
            raise AdvertisingTire3DError(
                f"the {name} is {other.width}x{other.height}x{other.frames}, "
                f"expected {info.width}x{info.height}x{info.frames}"
            )
    assert_video_alignment(
        {
            "input video": args.input,
            "primary mask": args.primary_mask,
            "clean plate": args.plate,
        },
        width=info.width,
        height=info.height,
        frames=info.frames,
    )
    profile = load_wheel_profile(args.blender_script)
    masks = load_primary_mask(
        args.primary_mask,
        frames=info.frames,
        width=info.width,
        height=info.height,
        invert=args.mask_invert,
    )
    raw_fits = [fit_ellipse(mask) for mask in masks]
    fits = smooth_fit_series(raw_fits, window=args.smoothing_window)
    poses = fit_wheel_poses(
        fits,
        profile=profile,
        source_width=info.width,
        source_height=info.height,
        render_width=args.render_width,
        render_height=args.render_height,
        meters_per_pixel=METERS_PER_PIXEL,
    )
    payload = pose_payload(
        poses,
        spec,
        profile=profile,
        source_width=info.width,
        source_height=info.height,
        render_width=args.render_width,
        render_height=args.render_height,
        meters_per_pixel=METERS_PER_PIXEL,
    )
    args.poses.parent.mkdir(parents=True, exist_ok=True)
    args.poses.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    blender_sha = sha256(args.blender)
    script_sha = sha256(args.blender_script)
    pose_sha = sha256(args.poses)
    footprint_counts: list[int] = []
    with (
        TemporaryDirectory(prefix=".samurai-tire3d-render-") as render_root,
        GrayVideoWriter(
            args.inserted_mask, width=info.width, height=info.height, fps=info.fps
        ) as inserted_writer,
        GrayVideoWriter(
            args.cleanup_mask, width=info.width, height=info.height, fps=info.fps
        ) as cleanup_writer,
        TemporaryDirectory(prefix=".samurai-tire3d-frames-") as frame_root,
    ):
        render_dir = Path(render_root) / "render"
        run_blender(
            args.blender,
            args.blender_script,
            args.poses,
            render_dir,
            samples=args.samples,
            device=args.device,
            blend=args.blend,
            glb=args.glb,
        )
        manifest_path = render_dir / "render-manifest.json"
        if not manifest_path.is_file():
            raise AdvertisingTire3DError("the Blender run wrote no render manifest")
        manifest = verify_render_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            blender_sha256=blender_sha,
            script_sha256=script_sha,
            pose_sha256=pose_sha,
            samples=args.samples,
            width=args.render_width,
            height=args.render_height,
            frames=info.frames,
            profile=profile,
            device=args.device,
        )
        renders = sorted(render_dir.glob("frame-*.png"))
        if len(renders) != info.frames:
            raise AdvertisingTire3DError(
                f"Blender rendered {len(renders)} frames, expected {info.frames}"
            )
        composite_dir = Path(frame_root) / "composite"
        composite_dir.mkdir(parents=True, exist_ok=True)
        plate_frames = iter_color_frames(args.plate, Path(frame_root) / "plate")
        for index, (render_path, plate) in enumerate(zip(renders, plate_frames, strict=True)):
            render = load_render(
                render_path,
                width=info.width,
                height=info.height,
                box=content_box(
                    info.width,
                    info.height,
                    args.render_width,
                    args.render_height,
                ),
            )
            composite, alpha = composite_render(plate, render)
            Image.fromarray(composite, mode="RGB").save(composite_dir / f"frame-{index:06d}.png")
            inserted, cleanup = build_inserted_cleanup_masks(
                alpha, cleanup_radius=args.cleanup_radius
            )
            crop = (
                slice(args.cleanup_radius, -args.cleanup_radius),
                slice(args.cleanup_radius, -args.cleanup_radius),
            )
            inserted_writer.write(inserted[crop])
            cleanup_writer.write(cleanup[crop])
            footprint_counts.append(int(np.count_nonzero(alpha >= 0.5)))
        write_color_video(sorted(composite_dir.glob("frame-*.png")), args.output, fps=info.fps)
    manifest_target = args.render_manifest
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "frames": info.frames,
        "render_width": args.render_width,
        "render_height": args.render_height,
        "mean_footprint_pixels": (float(np.mean(footprint_counts)) if footprint_counts else 0.0),
        "min_footprint_pixels": min(footprint_counts) if footprint_counts else 0,
        "max_footprint_pixels": max(footprint_counts) if footprint_counts else 0,
        "frames_with_inserted_pixels": sum(1 for value in footprint_counts if value > 0),
        "frames_holding_a_fit": sum(1 for pose in poses if pose.held),
        "mean_tilt_degrees": float(np.mean([pose.tilt_degrees for pose in poses])),
        "mean_spin_degrees": float(np.mean([pose.spin_degrees for pose in poses])),
    }
    report = {
        "schema": SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(Path(__file__).resolve().parents[1]),
        "scope": (
            "A Blender-native 3D wheel mesh fitted to the ROI-black tire mask "
            "and composited over the clean plate; no perception model, paid API, "
            "credential, or network call ran."
        ),
        "inputs": {
            "video": {
                "path": str(args.input),
                "sha256": sha256(args.input),
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frames": info.frames,
            },
            "primary_mask": {
                "path": str(args.primary_mask),
                "sha256": sha256(args.primary_mask),
                "width": mask_info.width,
                "height": mask_info.height,
                "frames": mask_info.frames,
                "roi_convention": "bright" if args.mask_invert else "dark",
            },
            "plate": {
                "path": str(args.plate),
                "sha256": sha256(args.plate),
                "width": plate_info.width,
                "height": plate_info.height,
                "frames": plate_info.frames,
            },
        },
        "target": {
            "description": spec.description,
            "compiled": {
                "target": spec.target,
                "tread_color": list(spec.tread_color),
                "band_color": list(spec.band_color),
                "label": spec.label,
                "tokens": list(spec.tokens),
            },
        },
        "settings": {
            "blender": str(args.blender),
            "blender_sha256": blender_sha,
            "blender_script": str(args.blender_script),
            "blender_script_sha256": script_sha,
            "samples": args.samples,
            "device": args.device,
            "cleanup_radius": args.cleanup_radius,
            "smoothing_window": args.smoothing_window,
            "mask_invert": bool(args.mask_invert),
            "meters_per_pixel": METERS_PER_PIXEL,
            "composite_base": "plate",
            "wheel_profile": profile_record(profile),
        },
        "poses": [asdict(pose) for pose in poses],
        "render": {
            "manifest_path": str(manifest_target),
            "manifest_sha256": sha256(manifest_target),
            "blender_version": manifest["blender_version"],
            "engine": manifest["engine"],
            "counts": manifest["counts"],
            "wheel": manifest.get("wheel", {}),
            "blend": manifest.get("blend", {}),
            "glb": manifest.get("glb", {}),
        },
        "summary": summary,
        "outputs": {
            "composite": {
                "path": str(args.output),
                "sha256": sha256(args.output),
                "frames": info.frames,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
            },
            "inserted_mask": {
                "path": str(args.inserted_mask),
                "sha256": sha256(args.inserted_mask),
                "frames": info.frames,
                "mask_family": "roi-black",
            },
            "cleanup_mask": {
                "path": str(args.cleanup_mask),
                "sha256": sha256(args.cleanup_mask),
                "frames": info.frames,
                "mask_family": "roi-black",
            },
            "poses": {
                "path": str(args.poses),
                "sha256": pose_sha,
                "schema": POSE_SCHEMA,
            },
            "render_manifest": {
                "path": str(manifest_target),
                "sha256": sha256(manifest_target),
            },
        },
        "non_claims": list(NON_CLAIMS),
    }
    if args.blend is not None and Path(args.blend).is_file():
        report["outputs"]["blend"] = {
            "path": str(args.blend),
            "sha256": sha256(args.blend),
        }
    if args.glb is not None and Path(args.glb).is_file():
        report["outputs"]["glb"] = {
            "path": str(args.glb),
            "sha256": sha256(args.glb),
        }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line contract of the true-3D advertising tire lane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input video")
    parser.add_argument("--primary-mask", type=Path, required=True, help="ROI-black tire mask")
    parser.add_argument("--plate", type=Path, required=True, help="clean plate video")
    parser.add_argument(
        "--target",
        required=True,
        help="bounded English target description, for example 'a black tire with a yellow band'",
    )
    parser.add_argument("--output", type=Path, required=True, help="composited output video")
    parser.add_argument("--inserted-mask", type=Path, required=True, help="ROI-black inserted mask")
    parser.add_argument("--cleanup-mask", type=Path, required=True, help="ROI-black cleanup ring")
    parser.add_argument("--poses", type=Path, required=True, help="pose JSON written for Blender")
    parser.add_argument("--blender", type=Path, required=True, help="Blender executable")
    parser.add_argument(
        "--blender-script",
        type=Path,
        default=Path(__file__).resolve().parent / "blender_tire_render.py",
        help="in-Blender render script",
    )
    parser.add_argument("--report", type=Path, default=None, help="optional JSON receipt")
    parser.add_argument(
        "--render-manifest",
        type=Path,
        default=None,
        help="where the Blender render manifest is kept",
    )
    parser.add_argument("--blend", type=Path, default=None, help="optional .blend export")
    parser.add_argument("--glb", type=Path, default=None, help="optional .glb export")
    parser.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--cleanup-radius", type=int, default=DEFAULT_CLEANUP_RADIUS)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--mask-invert", action="store_true", help="accept a bright-ROI mask")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the true-3D advertising tire lane from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.render_manifest is None:
        args.render_manifest = args.poses.with_suffix(".render-manifest.json")
    try:
        report = run_lane(args)
    except (
        AdvertisingTire3DError,
        AdvertisingTireAssetError,
        MediaError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    summary = report["summary"]
    print(
        "advertising tire 3d: "
        f"{summary['frames']} frames at "
        f"{summary['render_width']}x{summary['render_height']}, "
        f"mean inserted footprint {summary['mean_footprint_pixels']:.1f} pixels"
    )
    for name, record in report["outputs"].items():
        print(f"{name}: {record['path']} sha256={record['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
