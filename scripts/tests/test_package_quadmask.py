"""Port of Q10 proof 1: the packager reproduces the reference package.

The original proof (Q10 lane, 2026-10-27) ran scripts/package_quadmask.py
on the ens-proof clip (frames [0, 96)), the fused-frame mask source and
the shipped bg prompt. What this test asserts, stated exactly: the
packager's quadmask_0.mp4 has decoded pixels identical to the reference
package quadmask (96 frames; decoded-frame MD5
8faacb958de81eb0d0185e15d9375402 over the cv2-decoded frames, in order).

Container bytes are not asserted. They are ffmpeg-build-dependent: the
2026-10-27 proof build produced a container byte-identical to the
reference (container MD5 d4d976c027f14d001cb40335da6f2967, muxer
Lavf60.16.100), while the R7 reproduction under ffmpeg 7.0.2-static
(Lavf61.1.100) differed in 34 bytes, all muxer metadata, with decoded
pixels bit-identical. The produced container MD5 is computed and
reported on failure as information only.

The historical fused-frame mask source used white for the selected subject.
The proof harness converts it to the repository's canonical primary-mask
polarity (black = subject/remove) before declaring --mask-mode primary, so
the test still checks decoded-output continuity with the reference package.

That reproduction needs the fixture media (not in this repo) and opencv,
so the test is gated: set QUADMASK_PROOF_CLIP, QUADMASK_PROOF_MASK_SOURCE
and QUADMASK_PROOF_PROMPT to run it; otherwise it skips. Skipping is not
a pass of the reproduction.
"""

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "package_quadmask.py"
# Reference package quadmask container MD5 (muxer Lavf60.16.100 build).
# Information only: container bytes are ffmpeg-build-dependent.
REFERENCE_QUADMASK_CONTAINER_MD5 = "d4d976c027f14d001cb40335da6f2967"
# MD5 of the reference quadmask's decoded frames (cv2, in frame order).
EXPECTED_QUADMASK_DECODED_MD5 = "8faacb958de81eb0d0185e15d9375402"
EXPECTED_FRAME_COUNT = 96


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decoded_md5(path):
    import cv2  # ty: ignore[unresolved-import]

    cap = cv2.VideoCapture(str(path))
    h = hashlib.md5()
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h.update(frame.tobytes())
        frames += 1
    cap.release()
    return frames, h.hexdigest()


@pytest.mark.skipif(
    not all(
        os.environ.get(k)
        for k in ("QUADMASK_PROOF_CLIP", "QUADMASK_PROOF_MASK_SOURCE", "QUADMASK_PROOF_PROMPT")
    ),
    reason="Q10 proof fixtures not configured (env vars unset)",
)
def test_proof1_reproduction(tmp_path):
    import importlib.util

    import numpy as np

    if importlib.util.find_spec("cv2") is None:
        pytest.skip("opencv (cv2) not installed in this environment")
    legacy_mask = Path(os.environ["QUADMASK_PROOF_MASK_SOURCE"])
    canonical_mask = tmp_path / "canonical-mask"
    canonical_mask.mkdir()
    if legacy_mask.is_dir():
        from PIL import Image

        legacy_frames = sorted(
            path for path in legacy_mask.iterdir() if path.suffix.lower() in {".png", ".jpg"}
        )
        for index, path in enumerate(legacy_frames):
            frame = np.asarray(Image.open(path).convert("L"))
            Image.fromarray(255 - frame, mode="L").save(canonical_mask / f"{index:06d}.png")
    else:
        import cv2  # ty: ignore[unresolved-import]

        capture = cv2.VideoCapture(str(legacy_mask))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(canonical_mask / f"{index:06d}.png"), 255 - gray)
            index += 1
        capture.release()
    assert any(canonical_mask.iterdir()), "canonical proof mask is empty"
    out = tmp_path / "pkg"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--clip",
        os.environ["QUADMASK_PROOF_CLIP"],
        "--start",
        "0",
        "--end",
        "96",
        "--mask-source",
        str(canonical_mask),
        "--mask-mode",
        "primary",
        "--prompt-file",
        os.environ["QUADMASK_PROOF_PROMPT"],
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    produced = out / "quadmask_0.mp4"
    frames, decoded_md5 = _decoded_md5(produced)
    assert frames == EXPECTED_FRAME_COUNT, (
        f"decoded frame count {frames} != {EXPECTED_FRAME_COUNT}; "
        f"produced container MD5 (information only): {_md5(produced)}"
    )
    assert decoded_md5 == EXPECTED_QUADMASK_DECODED_MD5, (
        f"decoded pixels differ from the reference package: {decoded_md5} "
        f"!= {EXPECTED_QUADMASK_DECODED_MD5}; produced container MD5 "
        f"(information only): {_md5(produced)}"
    )


# ---------------------------------------------------------------------------
# F12 mask-mode ruling (Option B): --mask-mode is explicit and required.
#
# The packager must never guess the mode from the decoded mask values:
# every value set inside {0, 63, 127, 255} fits both a prebuilt quadmask
# and a binary primary mask, and the pre-change value-only detection
# misclassified in both directions (verification: the package-quadmask
# verification record of 2026-10-28).
# Fixtures below reuse that verification geometry exactly (64x64, 4 frames,
# --keep-native-size), so the pre-change histograms pin the regression
# expectations. The tests run the packager as a subprocess with
# sys.executable, so they need opencv in that interpreter plus ffmpeg.
# ---------------------------------------------------------------------------

_F12_SIZE = 64
_F12_FRAMES = 4
_F12_LOSSLESS = ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"]

_f12_gate = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None or shutil.which("ffmpeg") is None,
    reason="F12 tests need opencv in the packager interpreter and ffmpeg",
)


def _f12_write_video(tmp_path, name, frames, codec_args):
    from PIL import Image

    png_dir = tmp_path / f"{name}-png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(png_dir / f"{i:06d}.png")
    out = tmp_path / f"{name}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            "24",
            "-i",
            str(png_dir / "%06d.png"),
            *codec_args,
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def _f12_frames(build):
    return [build() for _ in range(_F12_FRAMES)]


def _f12_clip(tmp_path):
    import numpy as np

    def build():
        frame = np.full((_F12_SIZE, _F12_SIZE, 3), (90, 130, 170), np.uint8)
        frame[:, _F12_SIZE // 2 :] = (40, 60, 80)
        return frame

    return _f12_write_video(
        tmp_path,
        "clip",
        _f12_frames(build),
        ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"],
    )


def _f12_two_class(tmp_path, name="two-class"):
    """Left half 0, right half 255: a two-class quadmask AND, read as a
    primary mask, a clean canonical primary (black = subject). Same bytes,
    two meanings -- the ambiguity at the heart of F12."""
    import numpy as np

    def build():
        frame = np.full((_F12_SIZE, _F12_SIZE), 255, np.uint8)
        frame[:, : _F12_SIZE // 2] = 0
        return frame

    return _f12_write_video(tmp_path, name, _f12_frames(build), _F12_LOSSLESS)


def _f12_artifact(tmp_path):
    """Binary primary (black = subject, left half) whose decoded edge
    fringe holds stray 63/127 values (verification case B)."""
    import numpy as np

    def build():
        frame = np.zeros((_F12_SIZE, _F12_SIZE), np.uint8)
        frame[:, 32:] = 255
        frame[:, 30] = 127
        frame[:, 31] = 63
        return frame

    return _f12_write_video(tmp_path, "artifact", _f12_frames(build), _F12_LOSSLESS)


def _f12_four_class(tmp_path):
    import numpy as np

    def build():
        frame = np.zeros((_F12_SIZE, _F12_SIZE), np.uint8)
        half = _F12_SIZE // 2
        frame[:half, half:] = 255
        frame[half:, :half] = 127
        frame[half:, half:] = 63
        return frame

    return _f12_write_video(tmp_path, "four-class", _f12_frames(build), _F12_LOSSLESS)


def _f12_probability(tmp_path):
    import numpy as np

    def build():
        frame = np.zeros((_F12_SIZE, _F12_SIZE), np.uint8)
        frame[:, 16:32] = 100
        frame[:, 32:48] = 200
        frame[:, 48:] = 255
        return frame

    return _f12_write_video(tmp_path, "probability", _f12_frames(build), _F12_LOSSLESS)


def _f12_run(tmp_path, clip, mask, out_name, *extra):
    prompt = tmp_path / "prompt.json"
    prompt.write_text('{"bg": "empty room, plain wall"}\n', encoding="utf-8")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--clip",
        str(clip),
        "--start",
        "0",
        "--end",
        str(_F12_FRAMES),
        "--mask-source",
        str(mask),
        "--prompt-file",
        str(prompt),
        "--out",
        str(tmp_path / out_name),
        "--keep-native-size",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _f12_decoded_hist(video, dest):
    import numpy as np
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            "select='between(n\\,0\\,99)',format=gray",
            "-fps_mode",
            "passthrough",
            str(dest / "%06d.png"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    hist = {}
    per_frame = []
    for png in sorted(dest.glob("*.png")):
        arr = np.asarray(Image.open(png))
        vals, counts = np.unique(arr, return_counts=True)
        vals_list = vals.tolist()
        counts_list = counts.tolist()
        assert len(vals_list) == len(counts_list)
        frame_hist = {int(vals_list[i]): int(counts_list[i]) for i in range(len(vals_list))}
        per_frame.append(frame_hist)
        for value, count in frame_hist.items():
            hist[value] = hist.get(value, 0) + count
    return hist, per_frame


@_f12_gate
def test_mask_mode_is_required(tmp_path):
    clip = _f12_clip(tmp_path)
    mask = _f12_two_class(tmp_path)
    res = _f12_run(tmp_path, clip, mask, "pkg")
    assert res.returncode != 0
    assert "--mask-mode" in res.stderr


@_f12_gate
def test_quadmask_mode_rejects_values_outside_quad_set(tmp_path):
    clip = _f12_clip(tmp_path)
    mask = _f12_probability(tmp_path)
    res = _f12_run(tmp_path, clip, mask, "pkg", "--mask-mode", "quadmask")
    assert res.returncode != 0
    assert "quadmask" in res.stderr
    assert not (tmp_path / "pkg" / "quadmask_0.mp4").exists()


@_f12_gate
def test_quadmask_mode_rejects_affected_source(tmp_path):
    clip = _f12_clip(tmp_path)
    mask = _f12_two_class(tmp_path)
    affected = _f12_two_class(tmp_path, "affected")
    res = _f12_run(
        tmp_path,
        clip,
        mask,
        "pkg",
        "--mask-mode",
        "quadmask",
        "--affected-source",
        str(affected),
    )
    assert res.returncode != 0
    assert "--affected-source" in res.stderr


@_f12_gate
def test_two_class_quadmask_declared_passthrough(tmp_path):
    # Verification case A: the pre-change code misread this {0, 255}
    # prebuilt quadmask as a primary mask, fabricated a 127 ring and
    # inverted keep/remove. Declared quadmask passes it through.
    clip = _f12_clip(tmp_path)
    mask = _f12_two_class(tmp_path)
    res = _f12_run(tmp_path, clip, mask, "pkg", "--mask-mode", "quadmask")
    assert res.returncode == 0, res.stderr
    assert "mode=quadmask-passthrough" in res.stdout
    hist, per_frame = _f12_decoded_hist(tmp_path / "pkg" / "quadmask_0.mp4", tmp_path / "decoded")
    assert hist == {0: 8192, 255: 8192}
    assert per_frame == [{0: 2048, 255: 2048}] * _F12_FRAMES


@_f12_gate
def test_artifact_mask_declared_primary_is_thresholded(tmp_path):
    # Verification case B: the pre-change code passed this binary primary
    # through as a quadmask because its decoded set held stray 63/127.
    # Declared primary thresholds it; no 63 survives.
    clip = _f12_clip(tmp_path)
    mask = _f12_artifact(tmp_path)
    res = _f12_run(tmp_path, clip, mask, "pkg", "--mask-mode", "primary")
    assert res.returncode == 0, res.stderr
    assert "mode=primary-thresholded" in res.stdout
    hist, per_frame = _f12_decoded_hist(tmp_path / "pkg" / "quadmask_0.mp4", tmp_path / "decoded")
    assert hist == {0: 8192, 127: 1344, 255: 6848}
    assert per_frame[1] == {0: 2048, 127: 448, 255: 1600}


@_f12_gate
def test_binary_primary_declared_matches_pre_change_output(tmp_path):
    # Regression: a clean binary {0, 255} primary declared primary
    # reproduces the pre-change primary lane exactly (the verification
    # control histogram on this geometry).
    clip = _f12_clip(tmp_path)
    mask = _f12_two_class(tmp_path)
    res = _f12_run(tmp_path, clip, mask, "pkg", "--mask-mode", "primary")
    assert res.returncode == 0, res.stderr
    assert "mode=primary" in res.stdout
    hist, per_frame = _f12_decoded_hist(tmp_path / "pkg" / "quadmask_0.mp4", tmp_path / "decoded")
    assert hist == {0: 8192, 127: 1344, 255: 6848}
    assert per_frame[0] == {0: 2048, 255: 2048}


@_f12_gate
def test_four_class_quadmask_declared_passthrough(tmp_path):
    # Regression: the one case pre-change auto-detection got right (a
    # four-class quadmask) is unchanged when declared explicitly.
    clip = _f12_clip(tmp_path)
    mask = _f12_four_class(tmp_path)
    res = _f12_run(tmp_path, clip, mask, "pkg", "--mask-mode", "quadmask")
    assert res.returncode == 0, res.stderr
    assert "mode=quadmask-passthrough" in res.stdout
    hist, per_frame = _f12_decoded_hist(tmp_path / "pkg" / "quadmask_0.mp4", tmp_path / "decoded")
    assert hist == {0: 4096, 63: 4096, 127: 4096, 255: 4096}
    assert per_frame[0] == {0: 1024, 63: 1024, 127: 1024, 255: 1024}
    assert per_frame[1] == {0: 1024, 63: 1024, 127: 1024, 255: 1024}


@_f12_gate
def test_primary_and_affected_sources_follow_canonical_polarity(tmp_path):
    """The packager and src/samurai/quadmask.py must agree on ROI-black."""
    import numpy as np

    def primary_frame():
        frame = np.full((_F12_SIZE, _F12_SIZE), 255, np.uint8)
        frame[:, : _F12_SIZE // 2] = 0
        return frame

    def affected_frame():
        frame = np.full((_F12_SIZE, _F12_SIZE), 255, np.uint8)
        frame[:, _F12_SIZE // 4 : _F12_SIZE * 3 // 4] = 0
        return frame

    clip = _f12_clip(tmp_path)
    primary = _f12_write_video(
        tmp_path, "canonical-primary", _f12_frames(primary_frame), _F12_LOSSLESS
    )
    affected = _f12_write_video(
        tmp_path, "canonical-affected", _f12_frames(affected_frame), _F12_LOSSLESS
    )
    res = _f12_run(
        tmp_path,
        clip,
        primary,
        "pkg",
        "--mask-mode",
        "primary",
        "--affected-source",
        str(affected),
    )
    assert res.returncode == 0, res.stderr
    assert "mode=primary+affected" in res.stdout
    hist, per_frame = _f12_decoded_hist(tmp_path / "pkg" / "quadmask_0.mp4", tmp_path / "decoded")
    assert hist == {0: 4096, 63: 4096, 127: 3072, 255: 5120}
    # Canonical frame-zero behavior clears affected-only pixels but keeps overlap.
    assert per_frame[0] == {0: 1024, 63: 1024, 255: 2048}
    assert per_frame[1] == {0: 1024, 63: 1024, 127: 1024, 255: 1024}


@_f12_gate
def test_probability_mask_declared_primary_thresholded(tmp_path):
    # Regression: values outside the quad set were the unambiguous
    # pre-change auto case (primary-thresholded); declared primary
    # reproduces it.
    clip = _f12_clip(tmp_path)
    mask = _f12_probability(tmp_path)
    res = _f12_run(tmp_path, clip, mask, "pkg", "--mask-mode", "primary")
    assert res.returncode == 0, res.stderr
    assert "mode=primary-thresholded" in res.stdout
    hist, _ = _f12_decoded_hist(tmp_path / "pkg" / "quadmask_0.mp4", tmp_path / "decoded")
    assert hist == {0: 8192, 127: 1344, 255: 6848}


# ---------------------------------------------------------------------------
# R2-F3/F4/F5: every cv2.imread result is guarded before use.
#
# A corrupt/unreadable image must produce a clean SystemExit (or, in the
# decode-readback verifier, a recorded failed check) naming the file --
# never a None that crashes later at .shape / np.unique with no path.
# These unit tests drive the module with a fake cv2 whose imread returns
# None for files whose bytes start with b"CORRUPT" and a small array
# otherwise, so they run with or without a real opencv install.
# ---------------------------------------------------------------------------


def _fq_fake_cv2(fail_stage=None):
    """A fake cv2 module.

    `fail_stage` makes imwrite fail for every file written into a directory
    with that name, which is how the two frame-writing call sites are told
    apart (the color frames go to "vf", the quadmask frames to "qm").
    """
    import types

    import numpy as np

    fake: Any = types.ModuleType("cv2")
    fake.IMREAD_GRAYSCALE = 0
    fake.MORPH_ELLIPSE = 2
    fake.INTER_NEAREST = 0
    fake.INTER_AREA = 3

    def imread(path, flags=None):
        data = Path(path).read_bytes() if Path(path).exists() else b""
        if data.startswith(b"CORRUPT"):
            return None
        if flags == fake.IMREAD_GRAYSCALE:
            return np.full((4, 4), 255, np.uint8)
        return np.full((4, 4, 3), 128, np.uint8)

    fake.imread = imread

    def resize(arr, size, interpolation=None):
        if arr.ndim == 3:
            return np.zeros((size[1], size[0], arr.shape[2]), arr.dtype)
        return np.zeros((size[1], size[0]), arr.dtype)

    fake.resize = resize
    fake.getStructuringElement = lambda shape, ksize: np.ones(ksize, np.uint8)
    fake.dilate = lambda arr, ker: arr

    def imwrite(path, arr):
        if fail_stage is not None and Path(path).parent.name == fail_stage:
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")
        return True

    fake.imwrite = imwrite
    return fake


def _fq_load_package(fake_cv2):
    spec = importlib.util.spec_from_file_location("package_quadmask_fixqueue", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("cv2")
    sys.modules["cv2"] = fake_cv2
    try:
        spec.loader.exec_module(mod)
    finally:
        if previous is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous
    return mod


def _fq_write(tmp_path, name, data):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_load_mask_frames_directory_corrupt_names_file_and_frame(tmp_path):
    mod = _fq_load_package(_fq_fake_cv2())
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    bad = _fq_write(tmp_path, "masks/000.png", b"CORRUPT png")
    _fq_write(tmp_path, "masks/001.png", b"OK png")
    with pytest.raises(SystemExit) as excinfo:
        mod.load_mask_frames(mask_dir, 10, 5, 7, tmp_path / "tmp", "primary")
    assert str(bad) in str(excinfo.value)
    assert "frame index 5" in str(excinfo.value)


def test_load_mask_frames_video_corrupt_names_file_and_frame(tmp_path, monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())
    source = _fq_write(tmp_path, "mask.mp4", b"video")
    bad = _fq_write(tmp_path, "extracted/000.png", b"CORRUPT png")
    ok = _fq_write(tmp_path, "extracted/001.png", b"OK png")
    monkeypatch.setattr(
        mod,
        "probe",
        lambda _path: {
            "width": 4,
            "height": 4,
            "frames": 10,
            "fps": 24.0,
            "fps_str": "24/1",
            "codec": "h264",
            "pix_fmt": "yuv420p",
        },
    )
    monkeypatch.setattr(mod, "extract_range", lambda *a, **k: [bad, ok])
    with pytest.raises(SystemExit) as excinfo:
        mod.load_mask_frames(source, 10, 3, 5, tmp_path / "tmp", "primary")
    assert str(bad) in str(excinfo.value)
    assert "frame index 3" in str(excinfo.value)


def test_load_mask_frames_video_range_is_refused_upfront(tmp_path, monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())
    source = _fq_write(tmp_path, "mask.mp4", b"video")
    monkeypatch.setattr(
        mod,
        "probe",
        lambda _path: {
            "width": 4,
            "height": 4,
            "frames": 4,
            "fps": 24.0,
            "fps_str": "24/1",
            "codec": "h264",
            "pix_fmt": "yuv420p",
        },
    )
    monkeypatch.setattr(
        mod, "extract_range", lambda *_args, **_kwargs: pytest.fail("extract_range must not run")
    )
    with pytest.raises(SystemExit, match=r"mask video .* exceeds its 4 frames"):
        mod.load_mask_frames(source, 10, 3, 5, tmp_path / "tmp", "primary")


def _fq_main_setup(tmp_path, monkeypatch, color_files, extract_range, fake_cv2=None):
    mod = _fq_load_package(fake_cv2 if fake_cv2 is not None else _fq_fake_cv2())
    clip = _fq_write(tmp_path, "clip.mp4", b"clip")
    prompt = tmp_path / "prompt.json"
    prompt.write_text('{"bg": "empty room"}\n', encoding="utf-8")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _fq_write(tmp_path, "masks/000.png", b"OK png")
    _fq_write(tmp_path, "masks/001.png", b"OK png")
    info = {
        "width": 4,
        "height": 4,
        "frames": 2,
        "fps": 24.0,
        "fps_str": "24/1",
        "codec": "h264",
        "pix_fmt": "yuv420p",
    }
    monkeypatch.setattr(mod, "probe", lambda path: dict(info))
    monkeypatch.setattr(mod, "extract_range", extract_range)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_quadmask.py",
            "--clip",
            str(clip),
            "--start",
            "0",
            "--end",
            "2",
            "--mask-source",
            str(mask_dir),
            "--mask-mode",
            "primary",
            "--prompt-file",
            str(prompt),
            "--out",
            str(tmp_path / "out"),
            "--keep-native-size",
        ],
    )
    return mod


def test_main_color_frame_corrupt_names_file_and_frame(tmp_path, monkeypatch):
    bad = _fq_write(tmp_path, "color/000.png", b"CORRUPT png")
    ok = _fq_write(tmp_path, "color/001.png", b"OK png")
    mod = _fq_main_setup(
        tmp_path, monkeypatch, [bad, ok], lambda video, start, end, dest, gray=False: [bad, ok]
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert str(bad) in str(excinfo.value)
    assert "frame index 0" in str(excinfo.value)


def test_verify_corrupt_decoded_frame_is_recorded_failure_not_crash(tmp_path, monkeypatch, capsys):
    color = [
        _fq_write(tmp_path, "color/000.png", b"OK png"),
        _fq_write(tmp_path, "color/001.png", b"OK png"),
    ]
    bad_qm = _fq_write(tmp_path, "verify/000.png", b"CORRUPT png")
    ok_qm = _fq_write(tmp_path, "verify/001.png", b"OK png")

    def extract_range(video, start, end, dest, gray=False):
        if Path(dest).name == "qm":
            return [bad_qm, ok_qm]
        return color

    def fake_run(cmd):
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    mod = _fq_main_setup(tmp_path, monkeypatch, color, extract_range)
    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.main() == 1
    captured = capsys.readouterr()
    assert "decoded quadmask frame 0 readable" in captured.err
    assert str(bad_qm) in captured.err
    for name in mod.PACKAGE_OUTPUTS:
        assert not (tmp_path / "out" / name).exists()


def test_missing_prompt_file_is_refused_before_packaging(tmp_path, monkeypatch):
    mod = _fq_main_setup(
        tmp_path,
        monkeypatch,
        [_fq_write(tmp_path, "color/000.png", b"OK png")],
        lambda *_args, **_kwargs: pytest.fail("packaging must not run"),
    )
    missing = tmp_path / "missing-prompt.json"
    argv = sys.argv
    argv[argv.index(str(tmp_path / "prompt.json"))] = str(missing)
    with pytest.raises(SystemExit, match="prompt file not found"):
        mod.main()


# ---------------------------------------------------------------------------
# Packager review repairs (2026-10-30): the four findings repaired on the
# packager are all fail-closed refusals, so each one is asserted here.
#   1. an ffprobe stream without a usable frame count or frame rate is a
#      clear refusal naming the file and the field, not a KeyError;
#   2. a non-positive or even --kernel is refused before OpenCV is called;
#   3. the packager carries no Python 3.10-only zip keyword argument;
#   4. a package output that aliases an input is refused and the input
#      survives byte-identical.
# The tests drive the module with the fake cv2 above, so they run with or
# without a real opencv install.
# ---------------------------------------------------------------------------


def _fq_probe_payload(drop=(), **overrides):
    stream = {
        "width": "4",
        "height": "4",
        "r_frame_rate": "24/1",
        "avg_frame_rate": "24/1",
        "nb_read_frames": "2",
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
    }
    stream.update(overrides)
    for key in drop:
        stream.pop(key, None)
    return json.dumps({"streams": [stream]})


def _fq_probe(mod, monkeypatch, payload):
    monkeypatch.setattr(mod, "run", lambda cmd: subprocess.CompletedProcess(cmd, 0, payload, ""))
    return mod.probe(Path("/nonexistent/clip.mp4"))


def _fq_argv(clip, mask_source, prompt, out, *extra):
    return [
        "package_quadmask.py",
        "--clip",
        str(clip),
        "--start",
        "0",
        "--end",
        "2",
        "--mask-source",
        str(mask_source),
        "--mask-mode",
        "primary",
        "--prompt-file",
        str(prompt),
        "--out",
        str(out),
        "--keep-native-size",
        *extra,
    ]


def _fq_inputs(tmp_path):
    clip = _fq_write(tmp_path, "clip.mp4", b"ORIGINAL CLIP")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _fq_write(tmp_path, "masks/000.png", b"OK png")
    prompt = tmp_path / "prompt.json"
    prompt.write_text('{"bg": "empty room"}\n', encoding="utf-8")
    return clip, mask_dir, prompt


def test_probe_reads_a_complete_stream(monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())
    info = _fq_probe(mod, monkeypatch, _fq_probe_payload())
    assert info["frames"] == 2
    assert info["fps"] == 24.0
    assert info["fps_str"] == "24/1"
    assert (info["width"], info["height"]) == (4, 4)
    assert info["codec"] == "h264"
    assert info["pix_fmt"] == "yuv420p"


@pytest.mark.parametrize(
    ("drop", "overrides", "expected"),
    [
        ((), {"nb_read_frames": "N/A"}, "nb_read_frames"),
        (("nb_read_frames",), {}, "nb_read_frames"),
        ((), {"nb_read_frames": "two"}, "nb_read_frames"),
        ((), {"r_frame_rate": "0/0"}, "r_frame_rate"),
        ((), {"r_frame_rate": "N/A"}, "r_frame_rate"),
        (("r_frame_rate",), {}, "r_frame_rate"),
        ((), {"r_frame_rate": "24"}, "r_frame_rate"),
        ((), {"width": "N/A"}, "width"),
        ((), {"pix_fmt": ""}, "pix_fmt"),
    ],
)
def test_probe_unusable_field_is_a_clear_refusal(monkeypatch, drop, overrides, expected):
    mod = _fq_load_package(_fq_fake_cv2())
    payload = _fq_probe_payload(drop=drop, **overrides)
    with pytest.raises(SystemExit) as excinfo:
        _fq_probe(mod, monkeypatch, payload)
    message = str(excinfo.value)
    assert expected in message
    assert "ffprobe" in message
    assert "KeyError" not in message


def test_probe_ffprobe_failure_is_a_clear_refusal(monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())

    def failing(cmd):
        raise subprocess.CalledProcessError(1, cmd, stderr="Invalid data found")

    monkeypatch.setattr(mod, "run", failing)
    with pytest.raises(SystemExit) as excinfo:
        mod.probe(Path("/nonexistent/clip.mp4"))
    message = str(excinfo.value)
    assert "ffprobe failed" in message
    assert "Invalid data found" in message


def test_probe_without_a_video_stream_is_a_clear_refusal(monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())
    with pytest.raises(SystemExit) as excinfo:
        _fq_probe(mod, monkeypatch, json.dumps({"streams": []}))
    assert "no video stream" in str(excinfo.value)


@pytest.mark.parametrize("kernel", [0, -1, 2, 14])
def test_invalid_kernel_is_refused_before_opencv(tmp_path, monkeypatch, kernel):
    mod = _fq_load_package(_fq_fake_cv2())
    calls = []
    monkeypatch.setattr(mod.cv2, "getStructuringElement", lambda *a, **k: calls.append(a))
    clip, mask_dir, prompt = _fq_inputs(tmp_path)
    monkeypatch.setattr(
        mod,
        "probe",
        lambda path: {
            "width": 4,
            "height": 4,
            "frames": 2,
            "fps": 24.0,
            "fps_str": "24/1",
            "codec": "h264",
            "pix_fmt": "yuv420p",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _fq_argv(clip, mask_dir, prompt, tmp_path / "out", "--kernel", str(kernel)),
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    message = str(excinfo.value)
    assert "--kernel" in message
    assert str(kernel) in message
    assert calls == []


def test_valid_kernel_reaches_opencv(tmp_path, monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())
    assert mod.validate_kernel(15) == 15
    assert mod.validate_kernel(1) == 1
    with pytest.raises(SystemExit):
        mod.validate_kernel(0)
    with pytest.raises(SystemExit):
        mod.validate_kernel(4)


def test_packager_uses_no_zip_strict_keyword():
    """The strict keyword argument of zip needs Python 3.10. The packager
    is documented to run on the execution environment's python3, so no zip
    call in it may use that keyword."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and any(keyword.arg == "strict" for keyword in node.keywords)
    ]
    assert offenders == []


def test_zip_counts_fails_closed_on_a_length_mismatch():
    mod = _fq_load_package(_fq_fake_cv2())
    with pytest.raises(SystemExit) as excinfo:
        mod.zip_counts([0, 127], [3], "pre-encode histogram")
    message = str(excinfo.value)
    assert "pre-encode histogram" in message
    assert "2 values" in message
    assert "1 counts" in message


def _fq_alias_scenario(tmp_path, alias_name, role):
    """Place one input file inside the output directory under a package
    output name. Returns the argv, the aliased path, the input label the
    refusal must name, and the output directory."""
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    clip = _fq_write(tmp_path, "clip.mp4", b"ORIGINAL CLIP")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _fq_write(tmp_path, "masks/000.png", b"OK png")
    prompt = tmp_path / "prompt.json"
    prompt.write_text('{"bg": "empty room"}\n', encoding="utf-8")

    alias = out_dir / alias_name
    if role == "clip":
        alias.write_bytes(b"ORIGINAL CLIP")
        argv = _fq_argv(alias, mask_dir, prompt, out_dir)
        label = "source clip"
    elif role == "mask":
        alias.write_bytes(b"ORIGINAL MASK")
        argv = _fq_argv(clip, alias, prompt, out_dir)
        label = "source mask"
    else:
        alias.write_text('{"bg": "empty room"}\n', encoding="utf-8")
        argv = _fq_argv(clip, mask_dir, alias, out_dir)
        label = "prompt file"
    return argv, alias, label, out_dir


@pytest.mark.parametrize(
    ("alias_name", "role"),
    [
        ("input_video.mp4", "clip"),
        ("quadmask_0.mp4", "mask"),
        ("prompt.json", "prompt"),
    ],
)
def test_package_output_may_not_alias_an_input(tmp_path, monkeypatch, alias_name, role):
    mod = _fq_load_package(_fq_fake_cv2())
    argv, alias, label, out_dir = _fq_alias_scenario(tmp_path, alias_name, role)
    before = alias.read_bytes()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    message = str(excinfo.value)
    assert alias_name in message
    assert label in message
    assert alias.read_bytes() == before
    assert sorted(path.name for path in out_dir.iterdir()) == [alias_name]


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_package_output_may_not_alias_an_input_through_a_link(tmp_path, monkeypatch, link_kind):
    mod = _fq_load_package(_fq_fake_cv2())
    clip, mask_dir, prompt = _fq_inputs(tmp_path)
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    link = out_dir / "input_video.mp4"
    if link_kind == "symlink":
        link.symlink_to(clip)
    else:
        os.link(clip, link)
    monkeypatch.setattr(sys, "argv", _fq_argv(clip, mask_dir, prompt, out_dir))
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    message = str(excinfo.value)
    assert "input_video.mp4" in message
    assert "source clip" in message
    assert clip.read_bytes() == b"ORIGINAL CLIP"


def test_a_distinct_output_directory_is_still_allowed(tmp_path, monkeypatch):
    """The refusal is scoped to aliasing: an output directory that holds no
    input under a package output name stays valid."""
    mod = _fq_load_package(_fq_fake_cv2())
    clip, mask_dir, prompt = _fq_inputs(tmp_path)
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    _fq_write(tmp_path, "pkg/input_video.mp4", b"unrelated old output")
    monkeypatch.setattr(sys, "argv", _fq_argv(clip, mask_dir, prompt, out_dir))
    monkeypatch.setattr(
        mod,
        "probe",
        lambda path: {
            "width": 4,
            "height": 4,
            "frames": 2,
            "fps": 24.0,
            "fps_str": "24/1",
            "codec": "h264",
            "pix_fmt": "yuv420p",
        },
    )
    monkeypatch.setattr(mod, "extract_range", lambda *a, **k: [])
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    # The run proceeds past the aliasing guard and fails later on its own
    # contract (the faked extract_range yields no frames).
    assert "aliases" not in str(excinfo.value)
    assert "refusing to write package output" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Packager second-round review repairs (2026-10-30).
#
#   1. a failed ffmpeg/ffprobe run is a clear refusal naming the tool, the
#      work and the exit code, never a bare subprocess traceback;
#   2. the frame-select filter is built only from validated integer bounds;
#   3. a mask directory is read in natural frame order, and names that leave
#      the frame order ambiguous are refused;
#   4. same-file comparison survives absent paths and symlinked paths;
#   5. a failed cv2.imwrite is a clear refusal naming the file.
#
# The two cv2.imwrite call sites share one parametrized regression test, so
# the duplicate finding is asserted once against both write paths.
# ---------------------------------------------------------------------------


def test_failed_ffmpeg_run_is_a_clear_refusal(tmp_path, monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())

    def failing(cmd):
        raise subprocess.CalledProcessError(
            1, cmd, stderr="Invalid data found when processing input"
        )

    monkeypatch.setattr(mod, "run", failing)
    video = _fq_write(tmp_path, "clip.mp4", b"clip")
    with pytest.raises(SystemExit) as excinfo:
        mod.extract_range(video, 0, 2, tmp_path / "frames", gray=False)
    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "exit 1" in message
    assert "Invalid data found" in message
    assert "CalledProcessError" not in message


def test_missing_ffmpeg_binary_is_a_clear_refusal(tmp_path, monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())

    def missing(cmd):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(mod, "run", missing)
    video = _fq_write(tmp_path, "clip.mp4", b"clip")
    with pytest.raises(SystemExit) as excinfo:
        mod.extract_range(video, 0, 2, tmp_path / "frames", gray=False)
    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "not on PATH" in message
    assert "FileNotFoundError" not in message


def test_missing_ffprobe_binary_is_a_clear_refusal(monkeypatch):
    mod = _fq_load_package(_fq_fake_cv2())

    def missing(cmd):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(mod, "run", missing)
    with pytest.raises(SystemExit) as excinfo:
        mod.probe(Path("/nonexistent/clip.mp4"))
    message = str(excinfo.value)
    assert "ffprobe failed" in message
    assert "not on PATH" in message


def test_frame_select_filter_is_built_from_validated_integers():
    mod = _fq_load_package(_fq_fake_cv2())
    assert mod.frame_select_filter(0, 3) == "select='between(n\\,0\\,2)'"
    assert mod.frame_select_filter(3, 5) == "select='between(n\\,3\\,4)'"


@pytest.mark.parametrize(
    "bounds",
    [(5, 5), (5, 3), (-1, 3), ("0", 3), (0, "3"), (True, 3)],
)
def test_invalid_frame_range_is_refused_before_ffmpeg(tmp_path, monkeypatch, bounds):
    mod = _fq_load_package(_fq_fake_cv2())
    calls = []
    monkeypatch.setattr(mod, "run", lambda cmd: calls.append(cmd))
    video = _fq_write(tmp_path, "clip.mp4", b"clip")
    with pytest.raises(SystemExit) as excinfo:
        mod.extract_range(video, bounds[0], bounds[1], tmp_path / "frames", gray=False)
    assert "extract_range" in str(excinfo.value)
    assert calls == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="rotation probe needs ffmpeg")
def test_rotation_tagged_extraction_preserves_coded_dimensions(tmp_path):
    """A 64x32 stream tagged rotate 90 must stay 64x32 when extracted.

    ffmpeg's default autorotation would write 32x64 PNGs while ffprobe
    reports the coded 64x32 stream, silently transposing every packaged
    pixel. The packager disables autorotation so extraction and probe use
    one orientation.
    """
    from PIL import Image

    mod = _fq_load_package(_fq_fake_cv2())
    base = tmp_path / "base.mp4"
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x32:r=24",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(base),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-display_rotation",
            "90",
            "-i",
            str(base),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = mod.probe(rotated)
    assert (info["width"], info["height"], info["frames"]) == (64, 32, 2)
    frames = mod.extract_range(rotated, 0, 2, tmp_path / "frames", gray=False)
    assert len(frames) == 2
    assert [Image.open(frame).size for frame in frames] == [(64, 32), (64, 32)]


def test_mask_directory_order_is_natural_not_lexicographic(tmp_path):
    """The package pairs masks with clip frames by index, so frame_10 must
    follow frame_2, not precede it as a plain string sort would."""
    import numpy as np

    mod = _fq_load_package(_fq_fake_cv2())
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for name in ("frame_1.png", "frame_2.png", "frame_10.png"):
        _fq_write(tmp_path, f"masks/{name}", b"OK png")
    read = []

    def imread(path, flags=None):
        read.append(Path(path).name)
        return np.full((4, 4), 255, np.uint8)

    mod.cv2.imread = imread
    frames = mod.load_mask_frames(mask_dir, 3, 0, 3, tmp_path / "tmp", "primary")
    assert len(frames) == 3
    assert read == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_mask_directory_with_ambiguous_names_is_refused(tmp_path):
    mod = _fq_load_package(_fq_fake_cv2())
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _fq_write(tmp_path, "masks/frame_1.png", b"OK png")
    _fq_write(tmp_path, "masks/frame_01.png", b"OK png")
    with pytest.raises(SystemExit) as excinfo:
        mod.load_mask_frames(mask_dir, 2, 0, 2, tmp_path / "tmp", "primary")
    message = str(excinfo.value)
    assert "frame_1.png" in message
    assert "frame_01.png" in message
    assert "ambiguous" in message


def test_mask_directory_same_frame_with_different_extensions_is_refused(tmp_path):
    mod = _fq_load_package(_fq_fake_cv2())
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _fq_write(tmp_path, "masks/frame_1.png", b"OK png")
    _fq_write(tmp_path, "masks/frame_1.jpg", b"OK jpeg")
    with pytest.raises(SystemExit) as excinfo:
        mod.load_mask_frames(mask_dir, 2, 0, 2, tmp_path / "tmp", "primary")
    message = str(excinfo.value)
    assert "frame_1.png" in message
    assert "frame_1.jpg" in message
    assert "ambiguous" in message


def test_same_file_treats_an_unresolvable_path_as_itself(tmp_path):
    """A path that cannot be resolved (a symlink loop) is still the same
    path as itself. The pre-repair comparison swallowed the resolution
    error and reported two identical paths as different files."""
    mod = _fq_load_package(_fq_fake_cv2())
    loop = tmp_path / "loop"
    other = tmp_path / "loop2"
    loop.symlink_to(other)
    other.symlink_to(loop)
    assert mod._same_file(loop, loop) is True
    assert mod._same_file(loop, other) is False


def test_same_file_handles_an_absent_path(tmp_path):
    mod = _fq_load_package(_fq_fake_cv2())
    absent = tmp_path / "not-written-yet.mp4"
    other = tmp_path / "other.mp4"
    assert mod._same_file(absent, absent) is True
    assert mod._same_file(absent, other) is False


def test_package_output_alias_through_a_symlinked_out_directory_is_refused(tmp_path, monkeypatch):
    """The output directory is reached through a symlink, so the alias
    comparison must follow it the same way as an ordinary path."""
    mod = _fq_load_package(_fq_fake_cv2())
    real_out = tmp_path / "real-out"
    real_out.mkdir()
    linked_out = tmp_path / "linked-out"
    linked_out.symlink_to(real_out)
    # The clip sits in the real output directory, which the packager reaches
    # through the symlink, so the two paths name one file.
    clip = _fq_write(tmp_path, "real-out/input_video.mp4", b"ORIGINAL CLIP")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _fq_write(tmp_path, "masks/000.png", b"OK png")
    prompt = tmp_path / "prompt.json"
    prompt.write_text('{"bg": "empty room"}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _fq_argv(clip, mask_dir, prompt, linked_out))
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    message = str(excinfo.value)
    assert "input_video.mp4" in message
    assert "source clip" in message
    assert clip.read_bytes() == b"ORIGINAL CLIP"


@pytest.mark.parametrize(
    ("stage", "expected"),
    [("vf", "color frame"), ("qm", "quadmask frame")],
)
def test_failed_frame_write_is_a_clear_refusal(tmp_path, monkeypatch, stage, expected):
    """One regression test for both cv2.imwrite call sites: a frame image
    that cannot be written must be refused by name, never encoded on as a
    missing frame."""
    color = [
        _fq_write(tmp_path, "color/000.png", b"OK png"),
        _fq_write(tmp_path, "color/001.png", b"OK png"),
    ]
    fake = _fq_fake_cv2(fail_stage=stage)
    mod = _fq_main_setup(
        tmp_path,
        monkeypatch,
        color,
        lambda video, start, end, dest, gray=False: color,
        fake_cv2=fake,
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    message = str(excinfo.value)
    assert expected in message
    assert "failed to write" in message
    assert "000000.png" in message
    assert "frame index 0" in message
    assert not (tmp_path / "out" / "input_video.mp4").exists()
    assert not (tmp_path / "out" / "quadmask_0.mp4").exists()


def test_package_staging_is_created_on_the_output_filesystem(tmp_path, monkeypatch):
    """Publication must not depend on a cross-device rename.

    The package staging directory is created inside --out. The verifier's
    working directory may still use the platform default because it never
    publishes by rename.
    """
    color = [
        _fq_write(tmp_path, "color/000.png", b"OK png"),
        _fq_write(tmp_path, "color/001.png", b"OK png"),
    ]
    mod = _fq_main_setup(
        tmp_path, monkeypatch, color, lambda video, start, end, dest, gray=False: color
    )
    sys.argv.extend(["--no-verify"])
    original_tempdir = mod.tempfile.TemporaryDirectory
    staging_calls = []

    def recording_tempdir(*args, **kwargs):
        staging_calls.append((args, kwargs.copy()))
        return original_tempdir(*args, **kwargs)

    def fake_run(cmd):
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mod.tempfile, "TemporaryDirectory", recording_tempdir)
    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.main() == 0
    args, kwargs = staging_calls[0]
    assert args == ()
    assert kwargs.get("prefix", "").startswith(".q10-pack-")
    assert Path(kwargs["dir"]) == tmp_path / "out"
    assert not list((tmp_path / "out").glob(".q10-pack-*"))
