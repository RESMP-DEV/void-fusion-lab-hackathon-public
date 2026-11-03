#!/usr/bin/env python3
"""package.py -- any-range VOID sequence packager (quadmask lane Q10).

Builds a Netflix VOID sequence package (input_video.mp4 + quadmask_0.mp4 +
prompt.json + MANIFEST.md) for ANY frame range of a source clip, from
existing mask sources. Frames are selected and paired by INDEX (decode
order), never by timestamp.

Usage:
  package.py --clip CLIP.mp4 --start S --end E --mask-source SRC \
             --mask-mode MODE --prompt-file PROMPT --out DIR [options]

  --start / --end   0-based source frame indices, end EXCLUSIVE.
                    Frame count of the package = end - start.
  --mask-source     One of:
                    * a directory of per-frame mask images (PNG/JPG),
                      sorted by natural name order (digit runs compare
                      numerically, so frame_2.png precedes frame_10.png);
                      either one file per clip frame (sliced [start:end])
                      or one per ranged frame;
                    * a mask VIDEO (.mp4/.mov/.mkv/.webm): a binary
                      primary mask (black = subject/remove), a thresholded
                      probability mask, or a prebuilt quadmask video.
                      Its kind is stated with --mask-mode, never
                      auto-detected from its decoded values.
                    * with --affected-source, a binary primary mask
                      (video or directory) combined canonically.
  --mask-mode       Required. What --mask-source holds: "primary" (a
                    binary or probability primary mask, black =
                    subject/remove) or "quadmask" (a prebuilt quadmask,
                    packaged as-is without a frame-zero rewrite).
                    There is no default and no auto-detection: a decoded
                    value set inside {0, 63, 127, 255} fits both kinds,
                    so the mode is a fact the caller must state (F12).
  --affected-source Optional second mask (same forms): black = affected
                    region (ROI = black), matching src/samurai/quadmask.py:
                    keep 255, affected 127, primary 0, overlap 63.
  --prompt-file     A JSON file holding {"bg": "..."} or a plain-text
                    file whose text becomes the bg prompt. The bg text
                    describes the scene AFTER removal; it must not
                    describe the removal.
  --width/--height  Output size, default 672x384 (VOID sample size
                    384x672 HxW). --keep-native-size keeps clip size.
  --kernel          Ellipse dilation kernel for the affected PROXY ring,
                    default 15 (applied at SOURCE resolution, the method
                    that reproduces the reference package histogram).

Mask semantics when only a primary mask is given (the canonical lane):
  primary  = mask < 128
  affected = (dilate(PRIMARY REGION, ellipse KxK) > 127)
             AND NOT primary                                    [PROXY]
  quad     = 255 keep; affected -> 127; primary -> 0. No 63 is
             produced in this mode (the ring excludes primary by
             construction). The 127 class is a geometric dilation-ring
             PROXY, not a VLM-reasoned affected region.
Frame-zero rule for generated primary/affected packages: affected-only
pixels (127) on package frame 0 are cleared to keep (255), matching the
repository combiner. Overlap pixels (63) are preserved. A declared
quadmask is never rewritten on frame 0.

Encodings (stated, per contract):
  input_video.mp4 : libx264, crf 18, yuv420p (same class as reference).
  quadmask_0.mp4  : libx264, crf 0 (lossless), yuv444p. Lossy encodings
                    are forbidden: yuv420p shifts class 127 to 126.

Verification (always run unless --no-verify): the ENCODED outputs are
decoded back and checked: frame counts equal each other and the range
length; quadmask decoded values are a subset of {0, 63, 127, 255};
generated package frame 0 values are a subset of {0, 63, 255}; a declared
quadmask may carry all four values on every frame; dimensions match.
The process exits non-zero if any check fails.

Refusals (fail closed, before any output byte is written):
  * An ffprobe stream that lacks a usable frame count, frame rate, size,
    or pixel format is refused by file and field name, never a KeyError
    and never a silently zero frame count.
  * --kernel must be a positive odd integer. OpenCV's ellipse structuring
    element accepts nothing else and raises a bare assertion otherwise.
  * A package output that aliases an input (the source clip, the source
    mask, the affected source, or the prompt file) is refused, so the run
    can never overwrite the material it was given. Absent paths and
    symlinked paths are compared the same way as ordinary paths.
  * A failed ffmpeg or ffprobe run is refused with the tool, the work it
    was doing and the exit code, never a bare subprocess traceback.
  * The frame-select filter is built only from validated integer bounds,
    so no caller-supplied text can reach ffmpeg's filter parser.
  * A mask directory whose names leave the frame order ambiguous (two
    files naming the same frame number, including different image
    extensions) is refused, because the package
    pairs masks with clip frames by index.
  * A frame image that cannot be written is refused by file name, so a
    failed write never reaches the encoder as a missing frame.

Requires: python3 with numpy, opencv (cv2), Pillow; ffmpeg/ffprobe.
Proven interpreter: the execution environment's python3 (sys.executable)
with numpy, opencv (cv2) and Pillow installed. The script uses no
Python 3.10-only keyword argument, so it stays runnable on that
interpreter rather than on one release newer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2  # ty: ignore[unresolved-import]
import numpy as np

ALLOWED_QUAD = {0, 63, 127, 255}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Every file the packager writes into --out. Each one is compared with
# every input before the run starts, because all of them are written with
# overwrite semantics.
PACKAGE_OUTPUTS = ("input_video.mp4", "quadmask_0.mp4", "prompt.json", "MANIFEST.md")


def _tool_failure_detail(error: subprocess.CalledProcessError) -> str:
    """Summarize a failed tool run: the exit code and the last stderr line."""
    detail = (error.stderr or "").strip().splitlines()
    suffix = f"; {detail[-1]}" if detail else ""
    return f"exit {error.returncode}{suffix}"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_tool(cmd: list[str], context: str) -> subprocess.CompletedProcess[str]:
    """Run an external tool and fail closed with a clear message.

    A failed ffmpeg must name the tool, the work it was doing and the exit
    code, never a bare CalledProcessError traceback. The tool is still
    `run`, so a test or a caller can substitute it.
    """
    try:
        return run(cmd)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{context}: {_tool_failure_detail(error)}") from error
    except FileNotFoundError as error:
        raise SystemExit(f"{context}: {cmd[0]} is not on PATH") from error


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_field(stream: dict[str, Any], key: str, path: Path) -> str:
    """Read one ffprobe stream field, failing closed when it is unusable.

    ffprobe omits a field or reports it as "N/A" when the container cannot
    supply it. Either case is a refusal naming the file and the field,
    never a KeyError and never a silently zero frame count.
    """
    if key not in stream:
        raise SystemExit(
            f"ffprobe: {path} has no '{key}' field in its first video stream; "
            "the packager refuses to guess it"
        )
    raw = stream[key]
    if raw is None or (isinstance(raw, str) and raw.strip() in ("", "N/A")):
        raise SystemExit(
            f"ffprobe: {path} reports '{key}' as {raw!r}; the packager refuses to guess it"
        )
    return str(raw)


def _probe_int(stream: dict[str, Any], key: str, path: Path) -> int:
    raw = _probe_field(stream, key, path)
    try:
        return int(raw)
    except ValueError as error:
        raise SystemExit(
            f"ffprobe: {path} reports '{key}' as {raw!r}, which is not an integer"
        ) from error


def _probe_rate(stream: dict[str, Any], key: str, path: Path) -> tuple[str, float]:
    raw = _probe_field(stream, key, path)
    numerator_text, separator, denominator_text = raw.partition("/")
    if not separator:
        raise SystemExit(f"ffprobe: {path} reports '{key}' as {raw!r}, which is not a frame rate")
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text)
    except ValueError as error:
        raise SystemExit(
            f"ffprobe: {path} reports '{key}' as {raw!r}, which is not a frame rate"
        ) from error
    if denominator == 0 or numerator <= 0:
        raise SystemExit(
            f"ffprobe: {path} reports '{key}' as {raw!r}, which is not a usable "
            "frame rate; the packager refuses to invent one"
        )
    return raw, numerator / denominator


def probe(path: Path) -> dict:
    """Probe the first video stream of `path`.

    Every field the packager relies on is read through the fail-closed
    accessors above, so a missing frame count or frame rate is a clear
    refusal naming the file and the field instead of a KeyError.
    """
    try:
        out = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,codec_name,pix_fmt",
                "-of",
                "json",
                str(path),
            ]
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ffprobe failed for {path}: {_tool_failure_detail(error)}") from error
    except FileNotFoundError as error:
        raise SystemExit(f"ffprobe failed for {path}: ffprobe is not on PATH") from error
    try:
        streams = json.loads(out.stdout)["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(f"ffprobe returned no parsable video stream list for {path}") from error
    if not isinstance(streams, list) or not streams:
        raise SystemExit(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise SystemExit(f"ffprobe returned an unusable video stream record for {path}")
    rate_str, fps = _probe_rate(stream, "r_frame_rate", path)
    return {
        "width": _probe_int(stream, "width", path),
        "height": _probe_int(stream, "height", path),
        "frames": _probe_int(stream, "nb_read_frames", path),
        "fps": fps,
        "fps_str": rate_str,
        "codec": _probe_field(stream, "codec_name", path),
        "pix_fmt": _probe_field(stream, "pix_fmt", path),
    }


def frame_select_filter(start: int, end: int) -> str:
    """Build the frame-select filter from a validated integer range.

    The filter expression is assembled only from integers that passed the
    range check, so no caller-supplied text can reach ffmpeg's filter
    parser. The bounds are validated here because this helper is private
    to the module: the public CLI already refuses an empty or negative
    range in main(), so a rejected range is an impossible state that no
    valid command line can reach.
    """
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"extract_range: {name} must be an integer frame index, got {value!r}")
    if start < 0 or end <= start:
        raise SystemExit(
            f"extract_range: invalid frame range [{start},{end}); the range must be "
            "non-empty and start at frame 0 or later"
        )
    return f"select='between(n\\,{start}\\,{end - 1})'"


def extract_range(video: Path, start: int, end: int, dest: Path, gray: bool) -> list[Path]:
    """Decode coded-orientation frames [start, end) by INDEX.

    ffprobe reports coded width and height. ffmpeg autorotates
    display-matrix streams by default, which would transpose or squash the
    extracted PNGs relative to those dimensions. Extraction therefore opts
    out of autorotation and keeps every frame in the source's coded
    orientation, the same orientation used by the probed size and masks.
    """
    vf = frame_select_filter(start, end)
    dest.mkdir(parents=True, exist_ok=True)
    if gray:
        vf += ",format=gray"
    run_tool(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-noautorotate",
            "-i",
            str(video),
            "-vf",
            vf,
            "-fps_mode",
            "passthrough",
            str(dest / "%06d.png"),
        ],
        f"ffmpeg failed to extract frames [{start},{end}) from {video}",
    )
    files = sorted(dest.glob("*.png"))
    if len(files) != end - start:
        raise SystemExit(
            f"extract_range: got {len(files)} frames from {video} "
            f"for [{start},{end}), expected {end - start}"
        )
    return files


def _read_mask_files(files: list[Path], start: int) -> list[np.ndarray]:
    """Read mask image files; a corrupt/unreadable file is a clean
    refusal naming the file and its frame index, never a None that
    crashes later at .shape / np.unique with no path (R2-F3/F4)."""
    frames: list[np.ndarray] = []
    for i, p in enumerate(files):
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise SystemExit(f"mask source: failed to load {p} (frame index {start + i})")
        frames.append(arr)
    return frames


def _mask_name_key(name: str) -> tuple[tuple[int, int, str], ...]:
    """Natural order key for a mask file name.

    Digit runs compare numerically, so frame_2.png sorts before frame_10.png.
    A plain lexicographic sort pairs the wrong mask with the wrong clip
    frame, and this package pairs masks with clip frames by index, so the
    order is a contract and not a convenience.
    """
    key: list[tuple[int, int, str]] = []
    for part in re.split(r"(\d+)", name):
        if part.isdigit():
            key.append((1, int(part), ""))
        else:
            key.append((0, 0, part))
    return tuple(key)


def _mask_sort_key(path: Path) -> tuple[tuple[tuple[int, int, str], ...], str]:
    """Total order over mask file names: natural order, then the raw name."""
    return (_mask_name_key(path.name), path.name)


def _mask_ambiguity_key(path: Path) -> tuple[tuple[int, int, str], ...]:
    """Return the natural frame identity of `path`, excluding its extension."""
    return _mask_name_key(path.stem)


def _reject_ambiguous_mask_names(files: list[Path]) -> None:
    """Refuse mask names whose frame order is ambiguous.

    Two files whose names carry the same frame number (frame_1.png,
    frame_01.png, or frame_1.jpg) have no defined order, and the package pairs masks with
    clip frames by index, so the run refuses instead of guessing which
    file is frame 1.
    """
    seen: dict[tuple[tuple[int, int, str], ...], Path] = {}
    for path in files:
        key = _mask_ambiguity_key(path)
        previous = seen.get(key)
        if previous is not None:
            raise SystemExit(
                f"mask dir {previous.parent}: {previous.name} and {path.name} name the "
                "same frame number; their order is ambiguous, so the packager "
                "refuses to guess which one is the earlier frame"
            )
        seen[key] = path


def load_mask_frames(
    source: Path, clip_frames: int, start: int, end: int, tmp: Path, tag: str
) -> list[np.ndarray]:
    """Load mask frames for [start, end) from a directory or a video."""
    if source.is_dir():
        files = sorted(
            (p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=_mask_sort_key,
        )
        _reject_ambiguous_mask_names(files)
        if len(files) == clip_frames:
            files = files[start:end]
        elif len(files) == end - start:
            pass
        else:
            raise SystemExit(
                f"mask dir {source}: {len(files)} images; expected "
                f"{clip_frames} (per clip frame) or {end - start} (per ranged frame)"
            )
        if len(files) != end - start:
            # Impossible state: the branches above select exactly the
            # ranged frames, and main() refuses a range outside the clip.
            raise SystemExit(
                f"mask dir {source}: {len(files)} frames selected for [{start},{end}); "
                f"expected {end - start}"
            )
        return _read_mask_files(files, start)
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        mask_info = probe(source)
        if end > int(mask_info["frames"]):
            raise SystemExit(
                f"mask video {source}: range [{start},{end}) exceeds its "
                f"{mask_info['frames']} frames"
            )
        files = extract_range(source, start, end, tmp / f"mask-{tag}", gray=True)
        return _read_mask_files(files, start)
    raise SystemExit(f"mask source {source}: not a directory or a video file")


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (mask.shape[1], mask.shape[0]) == size:
        return mask
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def zip_counts(values: list[int], counts: list[int], context: str) -> list[tuple[int, int]]:
    """Pair a value array with its count array, failing closed on a mismatch.

    The strict keyword argument of zip needs Python 3.10 and this script is
    documented to run on the execution environment's python3, so the length
    contract is checked here instead: a mismatch is a refusal naming the
    context, never a silently truncated histogram.
    """
    if len(values) != len(counts):
        raise SystemExit(f"{context}: {len(values)} values paired with {len(counts)} counts")
    return [(values[index], counts[index]) for index in range(len(values))]


def validate_kernel(kernel: int) -> int:
    """Validate the dilation kernel size before OpenCV sees it.

    cv2.getStructuringElement accepts only a positive, odd kernel size for
    the ellipse shape and raises a bare assertion otherwise, so the packager
    refuses the value first with a message naming the flag and the value.
    """
    if kernel < 1:
        raise SystemExit(f"--kernel must be a positive odd integer, got {kernel}")
    if kernel % 2 == 0:
        raise SystemExit(f"--kernel must be odd (OpenCV ellipse structuring element), got {kernel}")
    return kernel


def _real_path(path: Path) -> Path:
    """Resolve a path to its real location, following every symlink.

    os.path.realpath is used instead of Path.resolve because Path.resolve
    raises RuntimeError on a symlink loop. The old handler turned that
    into a silent "not the same file" verdict, so a path that could not
    be resolved compared as different from itself.
    """
    try:
        return Path(os.path.realpath(path))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot resolve path {path}: {error}") from error


def _stat_or_none(path: Path) -> os.stat_result | None:
    """Stat a path, or None when it is absent (an output not written yet)."""
    try:
        return path.stat()
    except OSError:
        return None


def _same_file(left: Path, right: Path) -> bool:
    """True when two paths name the same file on disk.

    The literal paths are compared first, so two identical paths are the
    same file even when the path cannot be resolved (a symlink loop) or
    does not exist yet (a package output that has not been written).
    Each side is then resolved on its own, so a failure on one side
    cannot hide a match on the other, and the inode comparison catches a
    hardlink.
    """
    if os.path.abspath(left) == os.path.abspath(right):
        return True
    left_path = _real_path(left)
    right_path = _real_path(right)
    if left_path == right_path:
        return True
    left_stat = _stat_or_none(left_path)
    right_stat = _stat_or_none(right_path)
    if left_stat is None or right_stat is None:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def guard_output_aliasing(out: Path, inputs: dict[str, Path]) -> None:
    """Refuse to write a package output over any input file.

    The packager writes input_video.mp4, quadmask_0.mp4, prompt.json and
    MANIFEST.md into --out with overwrite semantics (ffmpeg -y, write_text).
    If --out holds an input under one of those names, the run would destroy
    its own source clip, source mask, or prompt file. Every output name is
    compared with every input, by resolved path and by inode identity,
    before anything is written.
    """
    for name in PACKAGE_OUTPUTS:
        target = out / name
        for label, source in inputs.items():
            if _same_file(target, source):
                raise SystemExit(
                    f"refusing to write package output {target}: it is the same "
                    f"file as the {label} input {source}; choose an --out "
                    "directory that does not hold an input under a package "
                    "output name"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description="Any-range VOID sequence packager")
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, required=True, help="exclusive end index in the source clip")
    ap.add_argument("--mask-source", required=True, type=Path)
    ap.add_argument(
        "--mask-mode",
        required=True,
        choices=("primary", "quadmask"),
        help="what --mask-source holds: a binary/probability primary mask "
        "(black = subject/remove) or a prebuilt quadmask. Required; the mode is "
        "never guessed from the decoded mask values (F12).",
    )
    ap.add_argument("--affected-source", type=Path, default=None)
    ap.add_argument("--prompt-file", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--keep-native-size", action="store_true")
    ap.add_argument("--kernel", type=int, default=15)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    # Both refusals happen before the first output byte is written: an
    # invalid kernel never reaches OpenCV, and no package output may alias
    # an input file.
    validate_kernel(args.kernel)
    inputs: dict[str, Path] = {
        "source clip": args.clip,
        "source mask": args.mask_source,
        "prompt file": args.prompt_file,
    }
    if args.affected_source is not None:
        inputs["affected source"] = args.affected_source
    guard_output_aliasing(args.out, inputs)

    clip_info = probe(args.clip)
    if not (0 <= args.start < args.end <= clip_info["frames"]):
        raise SystemExit(
            f"range [{args.start},{args.end}) outside clip frames 0..{clip_info['frames']}"
        )
    n = args.end - args.start
    size = (
        (clip_info["width"], clip_info["height"])
        if args.keep_native_size
        else (args.width, args.height)
    )
    fps_str = clip_info["fps_str"]

    # Prompt: JSON {"bg": ...} or plain text.
    if not args.prompt_file.is_file():
        raise SystemExit(f"prompt file not found: {args.prompt_file}")
    raw_prompt = args.prompt_file.read_text(encoding="utf-8")
    bg = None
    try:
        parsed = json.loads(raw_prompt)
        if isinstance(parsed, dict) and "bg" in parsed:
            bg = str(parsed["bg"]).strip()
    except json.JSONDecodeError:
        pass
    if bg is None:
        bg = raw_prompt.strip()
    if not bg:
        raise SystemExit("prompt file yields an empty bg prompt")

    args.out.mkdir(parents=True, exist_ok=True)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel, args.kernel))

    # Stage on the output filesystem so publication remains a local rename.
    # A system temporary directory can be on another device, where rename(2)
    # fails with EXDEV only after encoding and verification are complete.
    pack_temp = tempfile.TemporaryDirectory(prefix=".q10-pack-", dir=args.out)
    tmp = Path(pack_temp.name)
    color_files = extract_range(args.clip, args.start, args.end, tmp / "color", gray=False)
    primary_frames = load_mask_frames(
        args.mask_source, clip_info["frames"], args.start, args.end, tmp, "primary"
    )
    affected_frames = None
    if args.affected_source is not None:
        affected_frames = load_mask_frames(
            args.affected_source, clip_info["frames"], args.start, args.end, tmp, "affected"
        )

    # Mode is stated by the caller via --mask-mode, never inferred
    # from the decoded values alone (F12): every value set inside
    # ALLOWED_QUAD fits both a prebuilt quadmask and a binary primary
    # mask, so a value-only rule misclassifies in both directions.
    # The declared mode is validated against the decoded values and
    # an incompatible declaration is refused with a clear error.
    uniques: set[int] = set()
    for f in primary_frames:
        uniques.update(int(v) for v in np.unique(f))
    if affected_frames is not None:
        if args.mask_mode == "quadmask":
            raise SystemExit(
                "--mask-mode quadmask cannot be combined with "
                "--affected-source: a prebuilt quadmask is packaged "
                "as-is and has no combine rule; declare --mask-mode "
                "primary for a primary mask combined with an "
                "affected source"
            )
        mode = "primary+affected"
    elif args.mask_mode == "quadmask":
        if not uniques <= ALLOWED_QUAD:
            raise SystemExit(
                "mask source declared --mask-mode quadmask but holds "
                f"values outside {{0, 63, 127, 255}}: "
                f"{sorted(uniques - ALLOWED_QUAD)}; a prebuilt "
                "quadmask holds only those values -- if this source "
                "is a primary/probability mask, declare "
                "--mask-mode primary"
            )
        mode = "quadmask-passthrough"
    else:
        mode = "primary" if uniques <= {0, 255} else "primary-thresholded"

    vf_dir = tmp / "vf"
    qm_dir = tmp / "qm"
    vf_dir.mkdir()
    qm_dir.mkdir()
    hist: dict[int, int] = {}
    for i in range(n):
        frame = cv2.imread(str(color_files[i]))
        if frame is None:
            raise SystemExit(
                f"failed to load color frame {color_files[i]} (frame index {args.start + i})"
            )
        frame = (
            cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if (frame.shape[1], frame.shape[0]) != size
            else frame
        )
        color_path = vf_dir / f"{i:06d}.png"
        if not cv2.imwrite(str(color_path), frame):
            raise SystemExit(
                f"failed to write color frame {color_path} (frame index {args.start + i})"
            )

        m = primary_frames[i]
        if mode == "quadmask-passthrough":
            q = resize_mask(m, size)
            bad = set(int(v) for v in np.unique(q)) - ALLOWED_QUAD
            if bad:
                raise SystemExit(f"quadmask source frame {i} holds values {sorted(bad)}")
        else:
            prim_full = m < 128
            if affected_frames is not None:
                a = affected_frames[i]
                if a.shape != m.shape:
                    a = cv2.resize(a, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                aff_full = a < 128
            else:
                primary_region = prim_full.astype(np.uint8) * 255
                aff_full = (cv2.dilate(primary_region, ker) > 127) & ~prim_full
            prim = resize_mask((prim_full.astype(np.uint8) * 255), size) > 127
            aff = resize_mask((aff_full.astype(np.uint8) * 255), size) > 127
            if affected_frames is None:
                # Dilation-ring proxy excludes primary by construction.
                aff &= ~prim
            q = np.full((size[1], size[0]), 255, np.uint8)
            q[aff] = 127
            q[prim] = 0
            if affected_frames is not None:
                q[prim & aff] = 63
        if i == 0 and mode != "quadmask-passthrough":
            # Generated packages follow the canonical frame-zero rule.
            # A caller-declared quadmask is a true pass-through.
            q[q == 127] = 255
        vals, counts = np.unique(q, return_counts=True)
        for v, c in zip_counts(vals.tolist(), counts.tolist(), "pre-encode histogram"):
            hist[int(v)] = hist.get(int(v), 0) + int(c)
        quadmask_path = qm_dir / f"{i:06d}.png"
        if not cv2.imwrite(str(quadmask_path), q):
            raise SystemExit(
                f"failed to write quadmask frame {quadmask_path} (frame index {args.start + i})"
            )

    package_stage = tmp / "package"
    package_stage.mkdir()
    input_out = package_stage / "input_video.mp4"
    quad_out = package_stage / "quadmask_0.mp4"
    run_tool(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            fps_str,
            "-i",
            str(vf_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(input_out),
        ],
        f"ffmpeg failed to encode {input_out}",
    )
    run_tool(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            fps_str,
            "-i",
            str(qm_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(quad_out),
        ],
        f"ffmpeg failed to encode {quad_out}",
    )
    (package_stage / "prompt.json").write_text(
        json.dumps({"bg": bg}, indent=2) + "\n", encoding="utf-8"
    )

    # ---- decode-readback verification of the ENCODED outputs ----
    checks: list[tuple[str, bool, str]] = []
    decoded_hist: dict[int, int] = {}
    frame0_values: list[int] = []
    if not args.no_verify:
        in_info = probe(input_out)
        qm_info = probe(quad_out)
        checks.append(
            (
                "input frame count == range length",
                in_info["frames"] == n,
                f"{in_info['frames']} vs {n}",
            )
        )
        checks.append(
            (
                "quadmask frame count == range length",
                qm_info["frames"] == n,
                f"{qm_info['frames']} vs {n}",
            )
        )
        checks.append(
            (
                "frame counts equal across input/quadmask",
                in_info["frames"] == qm_info["frames"],
                f"{in_info['frames']} vs {qm_info['frames']}",
            )
        )
        checks.append(
            (
                "dimensions match",
                (in_info["width"], in_info["height"]) == size
                and (qm_info["width"], qm_info["height"]) == size,
                f"input {in_info['width']}x{in_info['height']}, "
                f"quad {qm_info['width']}x{qm_info['height']}, "
                f"target {size[0]}x{size[1]}",
            )
        )
        checks.append(
            (
                "quadmask pix_fmt is yuv444p",
                str(qm_info["pix_fmt"]).startswith("yuv444p"),
                qm_info["pix_fmt"],
            )
        )
        with tempfile.TemporaryDirectory(prefix="q10-verify-") as td:
            files = extract_range(quad_out, 0, n, Path(td) / "qm", gray=True)
            all_values: set[int] = set()
            for i, p in enumerate(files):
                a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if a is None:
                    checks.append(
                        (
                            f"decoded quadmask frame {i} readable",
                            False,
                            f"failed to load decoded frame {p} (frame index {i})",
                        )
                    )
                    continue
                vals, counts = np.unique(a, return_counts=True)
                all_values.update(int(v) for v in vals)
                for v, c in zip_counts(vals.tolist(), counts.tolist(), "decoded histogram"):
                    decoded_hist[int(v)] = decoded_hist.get(int(v), 0) + int(c)
                if i == 0:
                    frame0_values = sorted(int(v) for v in vals)
            checks.append(
                (
                    "decoded quadmask values subset of {0,63,127,255}",
                    all_values <= ALLOWED_QUAD,
                    f"values {sorted(all_values)}",
                )
            )
            if mode != "quadmask-passthrough":
                checks.append(
                    (
                        "generated frame 0 values subset of {0,63,255}",
                        set(frame0_values) <= {0, 63, 255},
                        f"frame 0 values {frame0_values}",
                    )
                )
        failed = [c for c in checks if not c[1]]
    else:
        failed = []

    mask_md5 = md5_of(args.mask_source) if args.mask_source.is_file() else "n/a (directory)"
    lines = [
        "# VOID sequence package — MANIFEST",
        "",
        f"- Source clip: `{args.clip}`",
        f"  - MD5 {md5_of(args.clip)}; {clip_info['width']}x{clip_info['height']}; "
        f"{clip_info['frames']} frames; {fps_str} fps; codec {clip_info['codec']}",
        f"- Range: source indices [{args.start}, {args.end}) "
        f"(start inclusive, end exclusive) = {n} frames",
        f"- Mask source: `{args.mask_source}` (MD5 {mask_md5}); mode: {mode}",
        f"- Mask mode declared: {args.mask_mode}",
    ]
    if args.mask_mode == "primary":
        lines.append(
            "- Primary/affected mask polarity: black = region of interest, "
            "matching src/samurai/quadmask.py (white = keep)"
        )
    else:
        lines.append("- Declared quadmask handling: pass-through with no frame-zero rewrite")
    if args.affected_source is not None:
        aff_md5 = (
            md5_of(args.affected_source) if args.affected_source.is_file() else "n/a (directory)"
        )
        lines.append(f"- Affected source: `{args.affected_source}` (MD5 {aff_md5})")
    if mode in ("primary", "primary-thresholded"):
        lines.append(
            f"- Affected class: PROXY dilation ring, ellipse "
            f"{args.kernel}x{args.kernel} at source resolution, "
            "dilated-minus-primary. Not VLM-reasoned."
        )
    lines += [
        f"- Prompt file: `{args.prompt_file}` (MD5 {md5_of(args.prompt_file)})",
        f"- bg prompt: {json.dumps(bg, ensure_ascii=False)}",
        "",
        "## Outputs",
        "",
        "| File | Bytes | MD5 | Frames | Format |",
        "|---|---|---|---|---|",
    ]
    for name in ("input_video.mp4", "quadmask_0.mp4", "prompt.json"):
        p = package_stage / name
        if name.endswith(".mp4"):
            info = probe(p)
            fmt = (
                f"{info['codec']}, {info['pix_fmt']}, "
                f"{info['width']}x{info['height']}, {fps_str} fps"
            )
            frames = str(info["frames"])
        else:
            fmt = 'JSON, one "bg" key'
            frames = "-"
        lines.append(f"| {name} | {p.stat().st_size} | {md5_of(p)} | {frames} | {fmt} |")
    lines += [
        "",
        "## Decoded quadmask histogram (readback of the encoded file)",
        "",
        f"- Pre-encode histogram: {dict(sorted(hist.items()))}",
        f"- Decoded histogram: {dict(sorted(decoded_hist.items()))}",
        f"- Frame 0 decoded values: {frame0_values}",
        "",
        "## Pairing and VOID loader facts (read before use)",
        "",
        "- Frames are paired by INDEX (decode order), never by timestamp. "
        "Package frame i is source frame "
        f"{args.start} + i for both input_video.mp4 and quadmask_0.mp4.",
        "- VOID's loader (predict_v2v.py / utils.py) truncates the input "
        "to the configured max_video_length (49 in the proven run-1 "
        "config) and temporally mirror-pads to at least the 85-frame "
        "window; the cap-197 config consumes up to 197 frames and "
        "mirror-pads the tail (a 96-frame package produced 101 output "
        "frames: 96 real + 5 mirror).",
        "- VOID output is NOT frame-aligned with this package: output "
        "frame count differs and fps is stamped from config (12), with "
        "no resampling. Downstream steps must convert explicitly.",
        "- Frame-zero rule applied to generated packages only: "
        "affected-only 127 is cleared to 255 and overlap 63 is preserved. "
        "A declared quadmask is passed through unchanged.",
        "",
        "## Verification (decode-readback)",
        "",
    ]
    if args.no_verify:
        lines.append("- SKIPPED (--no-verify)")
    else:
        for name, ok, detail in checks:
            lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    lines.append("")
    (package_stage / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")

    if failed:
        print(
            "VERIFY FAIL: " + "; ".join(f"{name}: {detail}" for name, _ok, detail in failed),
            file=sys.stderr,
        )
        pack_temp.cleanup()
        return 1

    # Publish only after encoding, verification, and manifest construction.
    # A verification failure leaves no new package files for a consumer.
    args.out.mkdir(parents=True, exist_ok=True)
    for name in PACKAGE_OUTPUTS:
        os.replace(package_stage / name, args.out / name)
    pack_temp.cleanup()
    print(
        f"package: {args.out} frames={n} mode={mode} "
        f"hist={dict(sorted((decoded_hist or hist).items()))}"
    )
    print("VERIFY PASS" if not args.no_verify else "VERIFY SKIPPED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
