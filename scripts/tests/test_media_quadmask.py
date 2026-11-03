from __future__ import annotations

import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from samurai import media as media_module
from samurai.media import (
    MediaError,
    assert_video_alignment,
    decode_gray_frames,
    iter_color_frames,
    iter_gray_frames,
    probe_frame_timestamps,
    probe_video,
    write_gray_video,
)
from samurai.quadmask import AFFECTED, KEEP, OVERLAP, REMOVE, build_sequence


def make_video(path, frames: list[np.ndarray]) -> None:
    write_gray_video(frames, path, fps=12.0)


def test_frame_timestamps_multiply_by_the_stream_time_base(monkeypatch, tmp_path):
    payload = {
        "streams": [{"time_base": "1001/30000"}],
        "frames": [{"pts": 30000}, {"pts": 60000}],
    }

    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")

    def fake_run(command, cwd=None):
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(media_module, "_run", fake_run)

    assert probe_frame_timestamps(video) == [
        Fraction(1001),
        Fraction(2002),
    ]


def test_alignment_checks_every_timestamp_frame_count(monkeypatch, tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    info = SimpleNamespace(width=2, height=2, frames=3, fps=12.0)
    monkeypatch.setattr(media_module, "probe_video", lambda path: info)
    monkeypatch.setattr(
        media_module,
        "probe_frame_timestamps",
        lambda path: (
            [Fraction(0), Fraction(1), Fraction(2)] if path == first else [Fraction(0), Fraction(1)]
        ),
    )

    with pytest.raises(MediaError, match="second has 2 timestamped frames"):
        assert_video_alignment({"first": first, "second": second}, width=2, height=2, frames=3)


def test_gray_video_round_trip(tmp_path):
    frames = [np.full((5, 7), value, dtype=np.uint8) for value in (0, 127, 255)]
    video = tmp_path / "video.mp4"
    make_video(video, frames)
    info = probe_video(video)
    assert (info.width, info.height, info.frames) == (7, 5, 3)


def write_rgb_frames(frames: list[np.ndarray], directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(directory / f"frame-{index:06d}.png")


def test_iter_gray_frames_matches_decode_gray_frames(tmp_path):
    frames = [np.arange(35, dtype=np.uint8).reshape(5, 7) + value for value in (0, 50, 100)]
    video = tmp_path / "gray.mp4"
    make_video(video, frames)

    decoded = decode_gray_frames(video, tmp_path / "eager")
    streamed = list(iter_gray_frames(video, tmp_path / "lazy", expected_frames=len(decoded)))

    np.testing.assert_array_equal(streamed, decoded)


def test_iter_gray_frames_rejects_a_wrong_expected_frame_count(tmp_path):
    video = tmp_path / "two-frames.mp4"
    make_video(video, [np.zeros((4, 4), dtype=np.uint8) for _ in range(2)])

    with pytest.raises(MediaError, match="decoded 2 frames from .* expected 3"):
        list(iter_gray_frames(video, tmp_path / "lazy", expected_frames=3))


def test_iter_gray_frames_fails_closed_when_no_frame_is_decoded(tmp_path, monkeypatch):
    video = tmp_path / "empty.mp4"
    video.write_text("decodable but empty", encoding="utf-8")
    monkeypatch.setattr("samurai.media._run", lambda command, cwd=None: None)

    with pytest.raises(MediaError, match="no frames decoded from"):
        next(iter_gray_frames(video, tmp_path / "lazy"))


def test_iter_color_frames_yields_the_source_rgb_frames(tmp_path):
    frames = [
        np.stack(
            [np.full((5, 7), channel, dtype=np.uint8) for channel in (red, green, blue)],
            axis=2,
        )
        for red, green, blue in ((10, 20, 30), (200, 150, 100))
    ]
    video = tmp_path / "color.mp4"
    write_rgb_frames(frames, tmp_path / "source")
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "12",
            "-i",
            str(tmp_path / "source" / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv444p",
            "-y",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    streamed = list(iter_color_frames(video, tmp_path / "lazy"))

    assert [frame.shape for frame in streamed] == [(5, 7, 3)] * 2
    for decoded, expected in zip(streamed, frames, strict=True):
        np.testing.assert_allclose(decoded, expected, atol=1)


def test_iter_color_frames_fails_closed_when_no_frame_is_decoded(tmp_path, monkeypatch):
    video = tmp_path / "empty.mp4"
    video.write_text("decodable but empty", encoding="utf-8")
    monkeypatch.setattr("samurai.media._run", lambda command, cwd=None: None)

    with pytest.raises(MediaError, match="no color frames extracted from"):
        next(iter_color_frames(video, tmp_path / "lazy"))


def test_gray_video_preserves_ntsc_framerate(tmp_path):
    from samurai.media import write_gray_video

    video = tmp_path / "ntsc.mp4"
    write_gray_video(
        [np.full((4, 4), 255, dtype=np.uint8) for _ in range(2)], video, fps=24000 / 1001
    )
    assert probe_video(video).fps == pytest.approx(24000 / 1001)


def test_build_sequence_combines_quadmask(tmp_path):
    source = tmp_path / "input.mp4"
    primary = tmp_path / "primary.mp4"
    affected = tmp_path / "affected.mp4"
    make_video(source, [np.full((4, 4), 255, dtype=np.uint8) for _ in range(2)])
    make_video(primary, [np.array([[0, 255, 255, 255]] * 4, dtype=np.uint8) for _ in range(2)])
    make_video(
        affected,
        [np.array([[255, 127, 127, 127]] * 4, dtype=np.uint8) for _ in range(2)],
    )
    result = build_sequence(
        input_video=source,
        primary_mask=primary,
        affected_mask=affected,
        background_prompt="An empty table",
        sequence_dir=tmp_path / "sequence",
    )
    assert result.frames == 2
    assert (tmp_path / "sequence/input_video.mp4").is_file()
    assert (tmp_path / "sequence/quadmask_0.mp4").is_file()
    assert json.loads((tmp_path / "sequence/prompt.json").read_text()) == {"bg": "An empty table"}


def test_quadmask_values_follow_void_contract():
    from samurai.quadmask import _combine

    primary = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    affected = np.array([[255, 0, 0, 255]], dtype=np.uint8)
    combined = _combine(primary, affected, first_frame=False)
    assert combined.tolist() == [[REMOVE, OVERLAP, AFFECTED, KEEP]]
    assert _combine(primary, affected, first_frame=True).tolist()[0][2] == KEEP


def test_insertion_combines_quadmask():
    from samurai.quadmask import _combine_insertion

    inserted = np.array([[0, 255, 255, 255]], dtype=np.uint8)
    cleanup = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    combined = _combine_insertion(inserted, cleanup, first_frame=False, cleanup_value=AFFECTED)
    assert combined.tolist() == [[KEEP, AFFECTED, KEEP, KEEP]]
    first = _combine_insertion(inserted, cleanup, first_frame=True, cleanup_value=AFFECTED)
    assert first.tolist()[0][1] == KEEP


def test_build_insertion_sequence_writes_the_staged_contract(tmp_path):
    from samurai.media import decode_gray_frames
    from samurai.quadmask import build_insertion_sequence

    source = tmp_path / "input.mp4"
    inserted = tmp_path / "inserted.mp4"
    cleanup = tmp_path / "cleanup.mp4"
    make_video(source, [np.full((4, 4), 255, dtype=np.uint8) for _ in range(2)])
    make_video(inserted, [np.array([[0, 255, 255, 255]] * 4, dtype=np.uint8) for _ in range(2)])
    make_video(cleanup, [np.array([[0, 0, 255, 255]] * 4, dtype=np.uint8) for _ in range(2)])
    result = build_insertion_sequence(
        input_video=source,
        inserted_mask=inserted,
        cleanup_mask=cleanup,
        background_prompt="An empty table",
        sequence_dir=tmp_path / "sequence",
    )
    assert result.frames == 2
    assert (tmp_path / "sequence/input_video.mp4").is_file()
    assert (tmp_path / "sequence/quadmask_0.mp4").is_file()
    assert json.loads((tmp_path / "sequence/prompt.json").read_text()) == {"bg": "An empty table"}
    frames = decode_gray_frames(tmp_path / "sequence/quadmask_0.mp4", tmp_path / "decoded")
    assert frames[0].tolist() == [[KEEP, KEEP, KEEP, KEEP]] * 4
    assert frames[1].tolist() == [[KEEP, AFFECTED, KEEP, KEEP]] * 4
