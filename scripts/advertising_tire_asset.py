#!/usr/bin/env python3
"""Render and composite a visible advertising replacement tire asset.

The lane fits the primary mask of an existing tire per frame, renders a
synthetic three-dimensional-looking advertising tire at that fitted pose, and
composites it over the source frames or over a clean plate.  It also writes
the ROI-black inserted-object mask and the ROI-black cleanup ring that the
VOID Pass 2 insertion lane consumes, plus a contact sheet that shows the
source, the rendered asset, the composite, and the inserted mask side by side.

The rendered asset is drawn by this script from numeric primitives.  No
third-party tire image, brand asset, checkpoint, model, network call, or
credential is read, and no rendered asset is committed to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from samurai.media import (
    GrayVideoWriter,
    MediaError,
    decode_gray_frames,
    iter_color_frames,
    probe_video,
)

SCHEMA = "void-fusion-lab.advertising-tire-asset.v1"
EXIT_INPUT_ERROR = 2
DEFAULT_BRAND = "MICHELIN"
DEFAULT_CLEANUP_RADIUS = 16
DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_CONTACT_ROWS = 4
CONTACT_PANEL = (320, 180)
MIN_FIT_PIXELS = 16
SUPERSAMPLE = 2
VISIBILITY_MIN_CHANGED_FRACTION = 0.5
VISIBILITY_MIN_BRAND_PIXELS = 24
VISIBILITY_CHANGE_THRESHOLD = 12.0

# Normalized radii of the drawn tire, from the hub outwards.
HUB_CAP_OUTER = 0.08
HUB_OUTER = 0.17
RIM_OUTER = 0.44
SIDEWALL_OUTER = 0.56
BAND_OUTER = 0.78
TREAD_OUTER = 1.00
BAND_TEXT_RADIUS = 0.67
SPOKE_COUNT = 5
TREAD_BLOCKS = 28

PALETTE = {
    "tread": (24, 24, 28),
    "tread_groove": (13, 13, 16),
    "sidewall": (46, 46, 53),
    "band": (247, 200, 0),
    "rim_gap": (25, 25, 29),
    "spoke": (178, 180, 188),
    "hub": (50, 52, 60),
    "hub_cap": (216, 218, 224),
}

NON_CLAIMS = (
    "The tire asset is drawn numerically by this script; it is not a photograph, "
    "a scan, a 3D render, or a third-party brand asset.",
    "The rendered wordmark is a generic advertising-style band and does not "
    "establish trademark, brand, or advertising rights for any manufacturer.",
    "The tire geometry comes from a per-frame ellipse fit of the primary mask; "
    "it is not a measured 3D reconstruction, a material model, or a physical "
    "simulation.",
    "The cleanup ring is a deterministic dilation of the rendered tire alpha, "
    "not a model-reasoned shadow, contact, or interaction region.",
    "Compositing the asset over the source frame does not remove the original "
    "tire; without a clean plate the original tire remains outside the "
    "rendered footprint.",
    "Visibility is a pixel-change measurement against the composite base, not a "
    "human quality verdict or an acceptance decision.",
    "No renderer, SAM 3.1, VOID, model, paid API, credential, or network call runs in this lane.",
)


class AdvertisingTireAssetError(RuntimeError):
    """An input, geometry, or media operation for the advertising lane failed."""


@dataclass(frozen=True)
class TireTile:
    """One rendered advertising tire tile and its placement in the frame."""

    rgb: np.ndarray
    alpha: np.ndarray
    origin: tuple[int, int]
    lettering: bool


@dataclass(frozen=True)
class TireFit:
    """One frame's fitted tire pose in image pixels."""

    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    angle_degrees: float
    area: int
    held: bool = False

    @property
    def radius_pixels(self) -> float:
        """Return the smaller fitted radius, used for pixel-scaled features."""
        return min(self.radius_x, self.radius_y)


def run_tool(command: list[str], *, context: str) -> None:
    """Run an external media tool and fail closed with its own exit code."""
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise AdvertisingTireAssetError(f"{context} exit {completed.returncode}{suffix}")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffmpeg_framerate(fps: float) -> str:
    """Return the canonical rational frame rate used by reproducible encodes."""
    fraction = Fraction(fps).limit_denominator(1001)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def require_file(path: Path, label: str) -> Path:
    """Fail closed unless a required input path is an existing file."""
    if not path.is_file():
        raise AdvertisingTireAssetError(f"{label} does not exist: {path}")
    return path


def load_primary_mask(
    path: Path,
    *,
    frames: int,
    width: int,
    height: int,
    invert: bool,
) -> list[np.ndarray]:
    """Decode the primary mask video into per-frame boolean ROI arrays.

    The repository mask convention is ROI-black: the region of interest is 0
    and the rest is 255.  ``--mask-invert`` accepts the bright-ROI variant.
    """
    with TemporaryDirectory(prefix=".samurai-advertising-mask-") as temporary:
        try:
            decoded = decode_gray_frames(path, Path(temporary), expected_frames=frames)
        except MediaError as error:
            raise AdvertisingTireAssetError(
                f"could not decode the primary mask: {error}"
            ) from error
    if len(decoded) != frames:
        raise AdvertisingTireAssetError(
            f"primary mask holds {len(decoded)} frames, expected {frames}"
        )
    masks: list[np.ndarray] = []
    for frame in decoded:
        if frame.shape != (height, width):
            raise AdvertisingTireAssetError(
                f"primary mask frame has shape {frame.shape}, expected {(height, width)}"
            )
        masks.append(frame >= 128 if invert else frame < 128)
    return masks


def fit_ellipse(mask: np.ndarray) -> TireFit | None:
    """Fit an ellipse to a binary region with the second-moment method.

    The fit returns the region centroid, the two semi-axes recovered from the
    covariance eigenvalues, and the major-axis orientation.  A uniform disk of
    radius ``R`` has covariance eigenvalue ``R**2 / 4``, so each semi-axis is
    ``2 * sqrt(eigenvalue)``.  Regions below ``MIN_FIT_PIXELS`` pixels have no
    usable fit and return ``None``.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < MIN_FIT_PIXELS:
        return None
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    dx = xs.astype(np.float64) - center_x
    dy = ys.astype(np.float64) - center_y
    var_x = float((dx * dx).mean())
    var_y = float((dy * dy).mean())
    covariance = float((dx * dy).mean())
    difference = var_x - var_y
    discriminant = math.hypot(difference, 2.0 * covariance)
    major = 0.5 * (var_x + var_y + discriminant)
    minor = max(0.0, 0.5 * (var_x + var_y - discriminant))
    angle = 0.5 * math.atan2(2.0 * covariance, difference)
    radius_x = 2.0 * math.sqrt(max(major, 0.0))
    radius_y = 2.0 * math.sqrt(minor)
    if radius_x < radius_y:
        radius_x, radius_y = radius_y, radius_x
        angle += math.pi / 2.0
    if not all(math.isfinite(value) for value in (center_x, center_y, radius_x, radius_y)):
        raise AdvertisingTireAssetError("the primary mask produced a non-finite ellipse fit")
    return TireFit(
        center_x=center_x,
        center_y=center_y,
        radius_x=max(radius_x, 2.0),
        radius_y=max(radius_y, 2.0),
        angle_degrees=math.degrees(angle),
        area=int(ys.size),
    )


def smooth_fit_series(fits: list[TireFit | None], *, window: int) -> list[TireFit]:
    """Smooth the fitted trajectory with a bounded centered moving average.

    Frames without a usable fit inherit the nearest fitted pose and are marked
    as held.  The average is taken over the window of fitted poses around each
    frame, the orientation is averaged as a direction vector so that a pose
    crossing the angle wrap does not flip, and the first and last frames
    anchor the series.
    """
    if window < 1:
        raise AdvertisingTireAssetError("--smoothing-window must be a positive integer")
    filled: list[TireFit] = []
    last: TireFit | None = None
    following: list[TireFit | None] = list(fits)
    for index, fit in enumerate(following):
        if fit is not None:
            last = fit
            filled.append(fit)
            continue
        if last is not None:
            filled.append(
                TireFit(
                    center_x=last.center_x,
                    center_y=last.center_y,
                    radius_x=last.radius_x,
                    radius_y=last.radius_y,
                    angle_degrees=last.angle_degrees,
                    area=last.area,
                    held=True,
                )
            )
            continue
        upcoming = next((item for item in following[index + 1 :] if item is not None), None)
        if upcoming is None:
            raise AdvertisingTireAssetError(
                "the primary mask has no fitted tire in any frame; nothing to replace"
            )
        filled.append(
            TireFit(
                center_x=upcoming.center_x,
                center_y=upcoming.center_y,
                radius_x=upcoming.radius_x,
                radius_y=upcoming.radius_y,
                angle_degrees=upcoming.angle_degrees,
                area=upcoming.area,
                held=True,
            )
        )
    half = window // 2
    smoothed: list[TireFit] = []
    for index, fit in enumerate(filled):
        low = max(0, index - half)
        high = min(len(filled), index + half + 1)
        neighbourhood = filled[low:high]
        angles = np.array([math.radians(item.angle_degrees) for item in neighbourhood])
        direction = math.degrees(
            math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
        )
        smoothed.append(
            TireFit(
                center_x=float(np.mean([item.center_x for item in neighbourhood])),
                center_y=float(np.mean([item.center_y for item in neighbourhood])),
                radius_x=float(np.mean([item.radius_x for item in neighbourhood])),
                radius_y=float(np.mean([item.radius_y for item in neighbourhood])),
                angle_degrees=direction,
                area=fit.area,
                held=fit.held,
            )
        )
    return smoothed


def _font_for(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return Pillow's bundled default font at a requested size.

    The bundled font keeps the lane dependency-free and machine-independent.
    When the running Pillow cannot scale its default font, the unscaled font is
    returned and the caller records that the wordmark was not drawn.
    """
    try:
        return ImageFont.load_default(size=size)
    except (AttributeError, OSError, TypeError, ValueError):
        return ImageFont.load_default()


def _fit_basis(fit: TireFit) -> tuple[float, float]:
    """Return the cosine and sine of the fitted major-axis orientation."""
    angle = math.radians(fit.angle_degrees)
    return math.cos(angle), math.sin(angle)


def _tile_geometry(fit: TireFit) -> tuple[int, int, int, int]:
    """Return the tile size and its top-left corner for a fitted pose."""
    cosine, sine = _fit_basis(fit)
    half_x = math.hypot(fit.radius_x * cosine, fit.radius_y * sine)
    half_y = math.hypot(fit.radius_x * sine, fit.radius_y * cosine)
    width = max(8, int(math.ceil(2.0 * half_x)) + 4)
    height = max(8, int(math.ceil(2.0 * half_y)) + 4)
    origin_x = int(round(fit.center_x - width / 2.0))
    origin_y = int(round(fit.center_y - height / 2.0))
    return width, height, origin_x, origin_y


def _local_grid(
    fit: TireFit, width: int, height: int, origin_x: int, origin_y: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the supersampled local coordinates of one tile.

    The local frame follows the fitted ellipse: ``a`` is the normalized major
    axis and ``b`` the normalized minor axis, so a circle in the local frame
    becomes the fitted ellipse in image space.
    """
    scale = SUPERSAMPLE
    cosine, sine = _fit_basis(fit)
    absolute_x = origin_x + (np.arange(width * scale, dtype=np.float64) + 0.5) / scale
    absolute_y = origin_y + (np.arange(height * scale, dtype=np.float64) + 0.5) / scale
    dx = absolute_x[None, :] - fit.center_x
    dy = absolute_y[:, None] - fit.center_y
    u = dx * cosine + dy * sine
    v = -dx * sine + dy * cosine
    a = u / fit.radius_x
    b = v / fit.radius_y
    return a, b, np.hypot(a, b), np.arctan2(b, a)


def _shading(
    fit: TireFit, a: np.ndarray, b: np.ndarray, radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the diffuse shade and the specular term for one tile.

    The light comes from the upper left of the image.  Each radial band uses
    its own surface tilt so that the sidewall reads as a rounded shoulder, the
    tread band as a steeply tilted surface, and the rim as a shallow dome.
    """
    light = np.array([-0.55, -0.62, 0.60])
    light = light / float(np.linalg.norm(light))
    highlight = light + np.array([0.0, 0.0, 1.0])
    highlight = highlight / float(np.linalg.norm(highlight))
    tilt = np.select(
        [radius < RIM_OUTER, radius < SIDEWALL_OUTER, radius < BAND_OUTER],
        [0.45, 1.50, 1.35],
        default=2.00,
    )
    depth = np.select([radius < BAND_OUTER], [1.0], default=0.5)
    cosine, sine = _fit_basis(fit)
    normal_x = a * tilt
    normal_y = b * tilt
    length = np.sqrt(normal_x**2 + normal_y**2 + depth**2)
    normal_x = normal_x / length
    normal_y = normal_y / length
    normal_z = depth / length
    image_x = normal_x * cosine - normal_y * sine
    image_y = normal_x * sine + normal_y * cosine
    diffuse = np.clip(image_x * light[0] + image_y * light[1] + normal_z * light[2], 0.0, 1.0)
    shade = 0.34 + 0.78 * diffuse
    specular_dot = np.clip(
        image_x * highlight[0] + image_y * highlight[1] + normal_z * highlight[2], 0.0, 1.0
    )
    metallic = (radius < RIM_OUTER).astype(np.float64)
    specular = 0.75 * specular_dot**24 * metallic + 0.12 * specular_dot**10 * (1.0 - metallic)
    return shade, specular


def _spoke_mask(angle: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Return the rim spoke mask for the local polar grid."""
    spacing = 2.0 * math.pi / SPOKE_COUNT
    phase = np.mod(angle - 0.35, spacing)
    sector = np.abs(phase - spacing / 2.0)
    span = (radius - HUB_OUTER) / max(RIM_OUTER - HUB_OUTER, 1e-6)
    half_width = 0.16 + 0.22 * np.clip(span, 0.0, 1.0)
    return (radius >= HUB_OUTER) & (radius < RIM_OUTER) & (sector < half_width)


def _tread_groove_mask(angle: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Return the tread groove mask for the local polar grid."""
    block = np.floor((angle / (2.0 * math.pi) + 0.5) * TREAD_BLOCKS)
    lateral = np.mod(block, 2.0) < 1.0
    circumferential = np.abs(radius - 0.90) < 0.015
    return (radius >= BAND_OUTER) & (lateral & (radius > 0.86) | circumferential)


def _band_arc(
    fit: TireFit, *, radius: float = BAND_TEXT_RADIUS, samples: int = 720
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled band-center points in image space and their arc length."""
    cosine, sine = _fit_basis(fit)
    phi = np.linspace(0.0, 2.0 * math.pi, samples + 1)
    u = radius * fit.radius_x * np.cos(phi)
    v = radius * fit.radius_y * np.sin(phi)
    x = fit.center_x + u * cosine - v * sine
    y = fit.center_y + u * sine + v * cosine
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return phi, cumulative


def _render_glyph(
    character: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[np.ndarray, float]:
    """Return one glyph bitmap and its advance width in pixels."""
    advance = float(font.getlength(character))
    bbox = font.getbbox(character)
    width = max(1, int(bbox[2] - bbox[0]) + 4)
    height = max(1, int(bbox[3] - bbox[1]) + 4)
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).text((2 - bbox[0], 2 - bbox[1]), character, fill=255, font=font)
    return np.asarray(image), advance


def _brand_text_mask(
    fit: TireFit,
    *,
    brand_text: str,
    width: int,
    height: int,
    origin_x: int,
    origin_y: int,
) -> tuple[np.ndarray, bool]:
    """Return the wordmark mask drawn along the upper arc of the brand band."""
    mask = np.zeros((height * SUPERSAMPLE, width * SUPERSAMPLE), dtype=bool)
    text = brand_text.strip()
    if not text:
        return mask, False
    phi, cumulative = _band_arc(fit)
    samples = len(phi) - 1
    arc_start = float(cumulative[samples // 2])
    arc_length = float(cumulative[samples] - arc_start)
    size = max(9, int(round(0.17 * fit.radius_pixels)))
    font = _font_for(size * SUPERSAMPLE)
    advances = [float(font.getlength(character)) / SUPERSAMPLE for character in text]
    total = sum(advances) * 1.04
    for _ in range(8):
        if total <= 0.82 * arc_length or size <= 8:
            break
        size = max(8, int(size * 0.85))
        font = _font_for(size * SUPERSAMPLE)
        advances = [float(font.getlength(character)) / SUPERSAMPLE for character in text]
        total = sum(advances) * 1.04
    if total > arc_length:
        return mask, False
    cosine, sine = _fit_basis(fit)
    running = 0.0
    for character, advance in zip(text, advances, strict=True):
        middle = running + advance / 2.0
        running += advance * 1.04
        arc_position = arc_start + arc_length / 2.0 + (middle - total / 2.0)
        arc_position = min(max(arc_position, float(cumulative[0])), float(cumulative[-1]))
        angle = float(np.interp(arc_position, cumulative, phi))
        u = BAND_TEXT_RADIUS * fit.radius_x * math.cos(angle)
        v = BAND_TEXT_RADIUS * fit.radius_y * math.sin(angle)
        x = fit.center_x + u * cosine - v * sine
        y = fit.center_y + u * sine + v * cosine
        normal = np.array([math.cos(angle) / fit.radius_x, math.sin(angle) / fit.radius_y])
        normal = normal / float(np.linalg.norm(normal))
        normal_x = normal[0] * cosine - normal[1] * sine
        normal_y = normal[0] * sine + normal[1] * cosine
        rotation = math.degrees(math.atan2(-normal_x, -normal_y))
        glyph, _ = _render_glyph(character, font)
        rotated = Image.fromarray(glyph).rotate(
            rotation, resample=Image.Resampling.BICUBIC, expand=True
        )
        patch = np.asarray(rotated) > 96
        offset_x = int(round((x - origin_x) * SUPERSAMPLE - patch.shape[1] / 2.0))
        offset_y = int(round((y - origin_y) * SUPERSAMPLE - patch.shape[0] / 2.0))
        low_x = max(0, offset_x)
        low_y = max(0, offset_y)
        high_x = min(mask.shape[1], offset_x + patch.shape[1])
        high_y = min(mask.shape[0], offset_y + patch.shape[0])
        if high_x <= low_x or high_y <= low_y:
            continue
        window = patch[low_y - offset_y : high_y - offset_y, low_x - offset_x : high_x - offset_x]
        mask[low_y:high_y, low_x:high_x] |= window
    return mask, True


def render_tire_tile(fit: TireFit, *, brand_text: str) -> TireTile:
    """Render the advertising tire asset for one fitted pose.

    The returned tile holds the asset RGB and its soft alpha in the frame's
    pixel space, together with the tile origin that places it in the frame and
    a flag that states whether the brand wordmark was drawn.
    """
    width, height, origin_x, origin_y = _tile_geometry(fit)
    a, b, radius, angle = _local_grid(fit, width, height, origin_x, origin_y)
    shade, specular = _shading(fit, a, b, radius)
    in_hub = radius < HUB_OUTER
    in_cap = radius < HUB_CAP_OUTER
    in_rim = (radius >= HUB_OUTER) & (radius < RIM_OUTER)
    in_sidewall = (radius >= RIM_OUTER) & (radius < SIDEWALL_OUTER)
    in_band = (radius >= SIDEWALL_OUTER) & (radius < BAND_OUTER)
    in_tread = radius >= BAND_OUTER
    spoke = _spoke_mask(angle, radius)
    groove = _tread_groove_mask(angle, radius)
    colors = np.full(
        (height * SUPERSAMPLE, width * SUPERSAMPLE, 3),
        np.array(PALETTE["tread"], dtype=np.float64),
    )
    for region, key in (
        (in_rim, "rim_gap"),
        (in_sidewall, "sidewall"),
        (in_band, "band"),
        (in_tread, "tread"),
        (in_hub, "hub"),
    ):
        colors[region] = np.array(PALETTE[key], dtype=np.float64)
    colors[spoke] = np.array(PALETTE["spoke"], dtype=np.float64)
    colors[in_cap] = np.array(PALETTE["hub_cap"], dtype=np.float64)
    colors[in_tread & groove] = np.array(PALETTE["tread_groove"], dtype=np.float64)
    shaded = colors * shade[..., None] + 255.0 * specular[..., None] * 0.85
    text_mask, lettering = _brand_text_mask(
        fit,
        brand_text=brand_text,
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if text_mask.any():
        painted = text_mask & in_band
        if painted.any():
            shaded[painted] = np.array([252.0, 252.0, 255.0]) * shade[painted][..., None]
            lettering = True
    image = Image.fromarray(np.clip(np.rint(shaded), 0, 255).astype(np.uint8), mode="RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    rgb = np.asarray(image).copy()
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    dx = origin_x + grid_x + 0.5 - fit.center_x
    dy = origin_y + grid_y + 0.5 - fit.center_y
    cosine, sine = _fit_basis(fit)
    u = dx * cosine + dy * sine
    v = -dx * sine + dy * cosine
    equation = (u / fit.radius_x) ** 2 + (v / fit.radius_y) ** 2 - TREAD_OUTER**2
    gradient_x = 2.0 * u / (fit.radius_x**2)
    gradient_y = 2.0 * v / (fit.radius_y**2)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2 + 1e-9)
    alpha = np.clip(0.5 - equation / gradient, 0.0, 1.0).astype(np.float32)
    return TireTile(rgb=rgb, alpha=alpha, origin=(origin_x, origin_y), lettering=lettering)


def composite_tile(
    frame: np.ndarray,
    tile_rgb: np.ndarray,
    tile_alpha: np.ndarray,
    origin: tuple[int, int],
) -> np.ndarray:
    """Alpha-composite one rendered tile onto a frame."""
    height, width = frame.shape[:2]
    tile_height, tile_width = tile_alpha.shape
    origin_x, origin_y = origin
    end_x = min(width, origin_x + tile_width)
    end_y = min(height, origin_y + tile_height)
    destination_x = max(0, origin_x)
    destination_y = max(0, origin_y)
    source_x = max(0, -origin_x)
    source_y = max(0, -origin_y)
    if end_x <= destination_x or end_y <= destination_y:
        return frame.copy()
    extent_x = end_x - destination_x
    extent_y = end_y - destination_y
    alpha = tile_alpha[source_y : source_y + extent_y, source_x : source_x + extent_x]
    tile = tile_rgb[source_y : source_y + extent_y, source_x : source_x + extent_x]
    region = frame[destination_y:end_y, destination_x:end_x].astype(np.float64)
    blended = region * (1.0 - alpha[..., None]) + tile.astype(np.float64) * alpha[..., None]
    output = frame.copy()
    output[destination_y:end_y, destination_x:end_x] = np.clip(np.rint(blended), 0, 255).astype(
        np.uint8
    )
    return output


def _dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Return the square dilation of a boolean mask without wrapping."""
    result = mask.copy()
    for _ in range(max(radius, 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(
            [
                padded[index : index + result.shape[0], offset : offset + result.shape[1]]
                for index in range(3)
                for offset in range(3)
            ]
        )
    return result


def build_inserted_cleanup_masks(
    alpha: np.ndarray,
    *,
    cleanup_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ROI-black inserted-object mask and cleanup ring for one tile.

    The inserted mask is black (0) on the rendered tire and white (255)
    elsewhere.  The cleanup mask is black (0) on a deterministic dilation ring
    around the rendered tire and white (255) elsewhere, so VOID Pass 2 can be
    pointed at the ring without touching the inserted object.  Both returned
    masks are padded by ``cleanup_radius`` around the rendered tile so the ring
    is not truncated by the tile boundary.
    """
    if cleanup_radius < 1:
        raise AdvertisingTireAssetError("--cleanup-radius must be a positive integer")
    padded_alpha = np.pad(alpha, cleanup_radius, mode="constant", constant_values=0.0)
    object_mask = padded_alpha >= 0.5
    ring = _dilate_square(object_mask, cleanup_radius) & ~object_mask
    inserted = np.where(object_mask, 0, 255).astype(np.uint8)
    cleanup = np.where(ring, 0, 255).astype(np.uint8)
    return inserted, cleanup


def measure_visibility(
    base: np.ndarray,
    composite: np.ndarray,
    tile_alpha: np.ndarray,
    origin: tuple[int, int],
) -> dict[str, float | int]:
    """Measure how much of the fitted footprint visibly changed.

    The change metric is the share of footprint pixels whose mean absolute
    channel difference against the composite base exceeds a visible
    threshold.  The brand metric counts the yellow advertising band pixels
    that survive inside the footprint.
    """
    height, width = base.shape[:2]
    tile_height, tile_width = tile_alpha.shape
    origin_x, origin_y = origin
    end_x = min(width, origin_x + tile_width)
    end_y = min(height, origin_y + tile_height)
    destination_x = max(0, origin_x)
    destination_y = max(0, origin_y)
    source_x = max(0, -origin_x)
    source_y = max(0, -origin_y)
    empty = {"footprint_pixels": 0, "changed_fraction": 0.0, "brand_pixels": 0}
    if end_x <= destination_x or end_y <= destination_y:
        return empty
    extent_x = end_x - destination_x
    extent_y = end_y - destination_y
    alpha = tile_alpha[source_y : source_y + extent_y, source_x : source_x + extent_x]
    base_tile = base[destination_y:end_y, destination_x:end_x]
    composite_tile = composite[destination_y:end_y, destination_x:end_x]
    footprint = alpha >= 0.5
    pixels = int(np.count_nonzero(footprint))
    if pixels == 0:
        return empty
    difference = np.abs(composite_tile.astype(np.int16) - base_tile.astype(np.int16)).mean(axis=2)
    changed = int(np.count_nonzero((difference > VISIBILITY_CHANGE_THRESHOLD) & footprint))
    red = composite_tile[:, :, 0].astype(np.int16)
    green = composite_tile[:, :, 1].astype(np.int16)
    blue = composite_tile[:, :, 2].astype(np.int16)
    yellow = (red >= 120) & (green >= 90) & ((red - blue) >= 45)
    brand = int(np.count_nonzero(yellow & footprint))
    return {
        "footprint_pixels": pixels,
        "changed_fraction": changed / pixels,
        "brand_pixels": brand,
    }


def write_color_video(frame_paths: list[Path], output: Path, *, fps: float) -> None:
    """Encode the composited frames into one H.264 video."""
    if not frame_paths:
        raise AdvertisingTireAssetError("cannot encode a video without frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            ffmpeg_framerate(fps),
            "-i",
            str(frame_paths[0].parent / "frame-%06d.png"),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
            "-y",
        ],
        context="composite video encode",
    )


def write_contact_sheet(rows: list[list[tuple[str, np.ndarray]]], output: Path) -> None:
    """Write one labeled contact sheet from RGB panels."""
    if not rows:
        raise AdvertisingTireAssetError("cannot write a contact sheet without rows")
    label_height = 22
    strips: list[np.ndarray] = []
    for row in rows:
        panels: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for text, panel in row:
            panels.append(
                np.asarray(Image.fromarray(panel).resize(CONTACT_PANEL, Image.Resampling.LANCZOS))
            )
            label = Image.new("RGB", (CONTACT_PANEL[0], label_height), (0, 0, 0))
            ImageDraw.Draw(label).text((8, 5), text, fill=(255, 255, 255))
            labels.append(np.asarray(label))
        strips.append(
            np.concatenate((np.concatenate(labels, axis=1), np.concatenate(panels, axis=1)), axis=0)
        )
    sheet = np.concatenate(strips, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(output)


def _paste_tile(
    tile: np.ndarray,
    shape: tuple[int, int],
    origin: tuple[int, int],
    *,
    fill: int,
) -> np.ndarray:
    """Place a tile into a full-frame canvas with the given background value."""
    height, width = shape
    canvas = np.full((height, width, *tile.shape[2:]), fill, dtype=np.uint8)
    tile_height, tile_width = tile.shape[:2]
    origin_x, origin_y = origin
    end_x = min(width, origin_x + tile_width)
    end_y = min(height, origin_y + tile_height)
    destination_x = max(0, origin_x)
    destination_y = max(0, origin_y)
    source_x = max(0, -origin_x)
    source_y = max(0, -origin_y)
    if end_x <= destination_x or end_y <= destination_y:
        return canvas
    extent_x = end_x - destination_x
    extent_y = end_y - destination_y
    canvas[destination_y:end_y, destination_x:end_x] = tile[
        source_y : source_y + extent_y, source_x : source_x + extent_x
    ]
    return canvas


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


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line contract of the advertising tire lane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input video")
    parser.add_argument("--primary-mask", type=Path, required=True, help="ROI-black tire mask")
    parser.add_argument("--plate", type=Path, default=None, help="optional clean plate video")
    parser.add_argument("--output", type=Path, required=True, help="composited output video")
    parser.add_argument("--inserted-mask", type=Path, required=True, help="ROI-black inserted mask")
    parser.add_argument("--cleanup-mask", type=Path, required=True, help="ROI-black cleanup ring")
    parser.add_argument("--contact-sheet", type=Path, required=True, help="contact sheet image")
    parser.add_argument("--report", type=Path, default=None, help="optional JSON receipt")
    parser.add_argument("--mask-invert", action="store_true", help="accept a bright-ROI mask")
    parser.add_argument("--cleanup-radius", type=int, default=DEFAULT_CLEANUP_RADIUS)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--contact-rows", type=int, default=DEFAULT_CONTACT_ROWS)
    parser.add_argument("--brand-text", default=DEFAULT_BRAND)
    return parser


def run_lane(args: argparse.Namespace) -> dict[str, Any]:
    """Run the advertising tire lane and return its JSON receipt."""
    if args.cleanup_radius < 1:
        raise AdvertisingTireAssetError("--cleanup-radius must be a positive integer")
    if args.smoothing_window < 1:
        raise AdvertisingTireAssetError("--smoothing-window must be a positive integer")
    if args.contact_rows < 1:
        raise AdvertisingTireAssetError("--contact-rows must be a positive integer")
    require_file(args.input, "input video")
    require_file(args.primary_mask, "primary mask")
    if args.plate is not None:
        require_file(args.plate, "clean plate")
    info = probe_video(args.input)
    expected = (info.width, info.height, info.frames)
    mask_info = probe_video(args.primary_mask)
    if (mask_info.width, mask_info.height, mask_info.frames) != expected:
        raise AdvertisingTireAssetError(
            f"primary mask is {mask_info.width}x{mask_info.height}x{mask_info.frames}, "
            f"expected {info.width}x{info.height}x{info.frames}"
        )
    plate_info = None
    if args.plate is not None:
        plate_info = probe_video(args.plate)
        if (plate_info.width, plate_info.height, plate_info.frames) != expected:
            raise AdvertisingTireAssetError(
                f"clean plate is {plate_info.width}x{plate_info.height}x{plate_info.frames}, "
                f"expected {info.width}x{info.height}x{info.frames}"
            )
    masks = load_primary_mask(
        args.primary_mask,
        frames=info.frames,
        width=info.width,
        height=info.height,
        invert=args.mask_invert,
    )
    raw_fits = [fit_ellipse(mask) for mask in masks]
    fits = smooth_fit_series(raw_fits, window=args.smoothing_window)
    sample_indices = sorted(
        {
            int(round(value))
            for value in np.linspace(
                0, info.frames - 1, min(args.contact_rows, info.frames)
            ).tolist()
        }
    )
    fit_records: list[dict[str, Any]] = []
    visibility_records: list[dict[str, Any]] = []
    lettering_frames = 0
    panels: dict[int, list[tuple[str, np.ndarray]]] = {}
    for output_path in (args.inserted_mask, args.cleanup_mask):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        GrayVideoWriter(
            args.inserted_mask, width=info.width, height=info.height, fps=info.fps
        ) as inserted_writer,
        GrayVideoWriter(
            args.cleanup_mask, width=info.width, height=info.height, fps=info.fps
        ) as cleanup_writer,
        TemporaryDirectory(prefix=".samurai-advertising-frames-") as temporary,
    ):
        root = Path(temporary)
        composite_dir = root / "composite"
        composite_dir.mkdir(parents=True, exist_ok=True)
        source_frames = iter_color_frames(args.input, root / "source")
        plate_frames = iter_color_frames(args.plate, root / "plate") if args.plate else None
        for index, fit in enumerate(fits):
            source = next(source_frames)
            base = next(plate_frames) if plate_frames is not None else source
            tile = render_tire_tile(fit, brand_text=args.brand_text)
            composite = composite_tile(base, tile.rgb, tile.alpha, tile.origin)
            inserted_tile, cleanup_tile = build_inserted_cleanup_masks(
                tile.alpha, cleanup_radius=args.cleanup_radius
            )
            mask_origin = (
                tile.origin[0] - args.cleanup_radius,
                tile.origin[1] - args.cleanup_radius,
            )
            inserted_frame = _paste_tile(
                inserted_tile, (info.height, info.width), mask_origin, fill=255
            )
            cleanup_frame = _paste_tile(
                cleanup_tile, (info.height, info.width), mask_origin, fill=255
            )
            metrics = measure_visibility(base, composite, tile.alpha, tile.origin)
            composite_path = composite_dir / f"frame-{index:06d}.png"
            Image.fromarray(composite, mode="RGB").save(composite_path)
            inserted_writer.write(inserted_frame)
            cleanup_writer.write(cleanup_frame)
            lettering_frames += 1 if tile.lettering else 0
            fit_records.append(
                {
                    "frame": index,
                    **asdict(fit),
                    "fitted": raw_fits[index] is not None,
                }
            )
            visibility_records.append({"frame": index, **metrics})
            if index in sample_indices:
                asset_panel = _paste_tile(tile.rgb, (info.height, info.width), tile.origin, fill=90)
                panels[index] = [
                    ("PLATE" if plate_frames is not None else "SOURCE", base),
                    ("TIRE ASSET", asset_panel),
                    ("COMPOSITE", composite),
                    ("INSERTED MASK", np.repeat(inserted_frame[..., None], 3, axis=2)),
                ]
        write_color_video(sorted(composite_dir.glob("frame-*.png")), args.output, fps=info.fps)
    write_contact_sheet([panels[index] for index in sample_indices], args.contact_sheet)
    changed = [float(record["changed_fraction"]) for record in visibility_records]
    brand = [int(record["brand_pixels"]) for record in visibility_records]
    summary = {
        "frames": info.frames,
        "mean_changed_fraction": float(np.mean(changed)) if changed else 0.0,
        "min_changed_fraction": float(np.min(changed)) if changed else 0.0,
        "mean_brand_pixels": float(np.mean(brand)) if brand else 0.0,
        "min_brand_pixels": int(np.min(brand)) if brand else 0,
        "frames_with_brand_band": sum(1 for value in brand if value > 0),
        "frames_holding_a_fit": sum(1 for record in fit_records if record["held"]),
        "frames_with_lettering": lettering_frames,
    }
    summary["visibility_gate"] = {
        "min_changed_fraction": VISIBILITY_MIN_CHANGED_FRACTION,
        "min_brand_pixels": VISIBILITY_MIN_BRAND_PIXELS,
        "passed": bool(
            summary["min_changed_fraction"] >= VISIBILITY_MIN_CHANGED_FRACTION
            and summary["min_brand_pixels"] >= VISIBILITY_MIN_BRAND_PIXELS
            and summary["frames_with_brand_band"] == info.frames
        ),
    }
    report = {
        "schema": SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(Path(__file__).resolve().parents[1]),
        "scope": (
            "Deterministic advertising tire asset fitted to the primary mask and "
            "composited over the source frames or the clean plate; no renderer, "
            "model, credential, or network call ran."
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
            "plate": (
                {
                    "path": str(args.plate),
                    "sha256": sha256(args.plate),
                    "width": plate_info.width,
                    "height": plate_info.height,
                    "frames": plate_info.frames,
                }
                if plate_info is not None
                else None
            ),
        },
        "settings": {
            "brand_text": args.brand_text,
            "cleanup_radius": args.cleanup_radius,
            "smoothing_window": args.smoothing_window,
            "contact_rows": args.contact_rows,
            "mask_invert": bool(args.mask_invert),
            "composite_base": "plate" if plate_frames is not None else "source",
        },
        "fit": fit_records,
        "visibility": visibility_records,
        "summary": summary,
        "outputs": {
            "output_video": {
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
            "contact_sheet": {
                "path": str(args.contact_sheet),
                "sha256": sha256(args.contact_sheet),
            },
        },
        "non_claims": list(NON_CLAIMS),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the advertising tire lane from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_lane(args)
    except (AdvertisingTireAssetError, MediaError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    summary = report["summary"]
    print(
        "advertising tire asset: "
        f"{summary['frames']} frames, "
        f"mean changed footprint {summary['mean_changed_fraction']:.4f}, "
        f"brand band on {summary['frames_with_brand_band']}/{summary['frames']} frames"
    )
    for name, record in report["outputs"].items():
        print(f"{name}: {record['path']} sha256={record['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
