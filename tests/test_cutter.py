"""Tests for autocut.output.cutter — pure functions and subprocess error handling."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, MediaInfo, Segment, SegmentSource
from autocut.output.cutter import (
    _apply_crossfade,
    _boundary_timestamps,
    _compute_kept_segments,
    _effective_crossfade_s,
    _segment_filter_chain,
    _xfade_filter_parts,
    cut_video,
)
from autocut.output import cutter as cutter_module


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bad(start: float, end: float) -> BadSegment:
    return BadSegment(Segment(start, end), SegmentSource.VAD, "silence")


FPS = 25.0
FRAME_S = 2.0 / FPS  # 0.08 s


# ── _compute_kept_segments ────────────────────────────────────────────────────

def test_kept_no_bad_segments_returns_full_duration():
    assert _compute_kept_segments([], 10.0) == [Segment(0.0, 10.0)]


def test_kept_bad_at_start():
    assert _compute_kept_segments([_bad(0.0, 3.0)], 10.0) == [Segment(3.0, 10.0)]


def test_kept_bad_at_end():
    assert _compute_kept_segments([_bad(8.0, 10.0)], 10.0) == [Segment(0.0, 8.0)]


def test_kept_bad_in_middle():
    assert _compute_kept_segments([_bad(3.0, 5.0)], 10.0) == [
        Segment(0.0, 3.0),
        Segment(5.0, 10.0),
    ]


def test_kept_multiple_bad_segments_unsorted():
    bad = [_bad(5.0, 7.0), _bad(1.0, 3.0)]
    kept = _compute_kept_segments(bad, 10.0)
    assert kept == [Segment(0.0, 1.0), Segment(3.0, 5.0), Segment(7.0, 10.0)]


def test_kept_filters_segments_shorter_than_10ms():
    # 10.0 - 9.995 = 0.005 s < 0.01 threshold
    kept = _compute_kept_segments([_bad(0.0, 9.995)], 10.0)
    assert kept == []


def test_kept_overlapping_bad_segments():
    bad = [_bad(1.0, 5.0), _bad(3.0, 7.0)]
    kept = _compute_kept_segments(bad, 10.0)
    assert kept == [Segment(0.0, 1.0), Segment(7.0, 10.0)]


# ── _effective_crossfade_s ────────────────────────────────────────────────────

def test_effective_crossfade_zero_ms():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    assert _effective_crossfade_s(kept, 0, FPS) == 0.0


def test_effective_crossfade_negative_ms():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    assert _effective_crossfade_s(kept, -100, FPS) == 0.0


def test_effective_crossfade_single_segment():
    assert _effective_crossfade_s([Segment(0.0, 5.0)], 200, FPS) == 0.0


def test_effective_crossfade_within_limit():
    # min_dur=5s, limit = 5/2 - 0.08 = 2.42s; request 0.2s → returned as-is
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    assert _effective_crossfade_s(kept, 200, FPS) == pytest.approx(0.2)


def test_effective_crossfade_clamped_to_segment_limit():
    # min_dur=0.5s, limit = 0.25 - 0.08 = 0.17s; request 0.5s → clamped
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    expected = 0.5 / 2.0 - FRAME_S
    assert _effective_crossfade_s(kept, 500, FPS) == pytest.approx(expected)


def test_effective_crossfade_zero_when_segment_too_short():
    # min_dur=0.1s, limit = 0.05 - 0.08 = -0.03 → clamped to 0
    kept = [Segment(0.0, 0.1), Segment(0.1, 0.2)]
    assert _effective_crossfade_s(kept, 200, FPS) == 0.0


def test_effective_crossfade_uses_shortest_segment():
    kept = [Segment(0.0, 10.0), Segment(10.0, 10.5), Segment(10.5, 20.0)]
    # min_dur = 0.5s, limit = 0.25 - 0.08 = 0.17s
    expected = 0.5 / 2.0 - FRAME_S
    assert _effective_crossfade_s(kept, 500, FPS) == pytest.approx(expected)


# ── _apply_crossfade ──────────────────────────────────────────────────────────

SR = 1000  # samples/s for all audio tests


def test_apply_crossfade_no_crossfade_is_plain_concat():
    audio = np.ones(1000, dtype=np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    result = _apply_crossfade(audio, SR, kept, 0.0)
    assert len(result) == 1000
    np.testing.assert_array_almost_equal(result, audio)


def test_apply_crossfade_single_segment_slices_correctly():
    audio = np.arange(1000, dtype=np.float32)
    kept = [Segment(0.1, 0.4)]
    result = _apply_crossfade(audio, SR, kept, 0.0)
    assert len(result) == 300
    np.testing.assert_array_equal(result, audio[100:400])


def test_apply_crossfade_empty_kept_returns_empty():
    audio = np.ones(1000, dtype=np.float32)
    result = _apply_crossfade(audio, SR, [], 0.0)
    assert len(result) == 0
    assert result.dtype == np.float32


def test_apply_crossfade_reduces_total_length():
    audio = np.ones(1000, dtype=np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    crossfade_s = 0.1
    result = _apply_crossfade(audio, SR, kept, crossfade_s)
    # 500 + 500 - 100 overlap = 900
    assert len(result) == 900


def test_apply_crossfade_three_segments_length():
    audio = np.ones(1500, dtype=np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0), Segment(1.0, 1.5)]
    crossfade_s = 0.1
    result = _apply_crossfade(audio, SR, kept, crossfade_s)
    # 3×500 - 2×100 = 1300
    assert len(result) == 1300


def test_apply_crossfade_blend_at_boundary():
    # Segment 1: all 0s; segment 2: all 1s; crossfade of 100 samples
    audio = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    crossfade_s = 0.1
    result = _apply_crossfade(audio, SR, kept, crossfade_s)
    # Midpoint of crossfade region: fade-out~0.5 and fade-in~0.5 → ~0.5
    mid = result[449]
    assert 0.3 < mid < 0.7


# ── _segment_filter_chain ─────────────────────────────────────────────────────

def test_segment_filter_chain_format():
    seg = Segment(1.5, 4.0)
    result = _segment_filter_chain(2, seg, 25.0)
    assert result == "[0:v]trim=start=1.500000:end=4.000000,setpts=PTS-STARTPTS,fps=25[v2]"


def test_segment_filter_chain_index_in_label():
    result = _segment_filter_chain(7, Segment(0.0, 1.0), 25.0)
    assert result.endswith("[v7]")


def test_segment_filter_chain_fps_rounded():
    result = _segment_filter_chain(0, Segment(0.0, 1.0), 29.97)
    assert ",fps=30[v0]" in result


def test_segment_filter_chain_fps_24():
    result = _segment_filter_chain(0, Segment(0.0, 1.0), 23.976)
    assert ",fps=24[v0]" in result


# ── _xfade_filter_parts ───────────────────────────────────────────────────────

def test_xfade_two_segments_one_part():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    parts, label = _xfade_filter_parts(kept, 0.1, FPS)
    assert len(parts) == 1
    assert label == "xf1"


def test_xfade_first_part_inputs():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    parts, _ = _xfade_filter_parts(kept, 0.1, FPS)
    assert parts[0].startswith("[v0][v1]xfade")


def test_xfade_three_segments_chained():
    kept = [Segment(0.0, 3.0), Segment(3.0, 6.0), Segment(6.0, 9.0)]
    parts, label = _xfade_filter_parts(kept, 0.1, FPS)
    assert len(parts) == 2
    assert label == "xf2"
    assert "[v0][v1]" in parts[0]
    assert "[xf1][v2]" in parts[1]


def test_xfade_offset_formula():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    crossfade_s = 0.1
    parts, _ = _xfade_filter_parts(kept, crossfade_s, FPS)
    expected = 5.0 - 1 * (crossfade_s + FRAME_S)
    assert f"offset={expected:.6f}" in parts[0]


def test_xfade_offset_clamped_to_zero():
    # Very short segments where offset would go negative
    kept = [Segment(0.0, 0.05), Segment(0.05, 0.10)]
    parts, _ = _xfade_filter_parts(kept, 0.04, FPS)
    assert "offset=0.000000" in parts[0]


def test_xfade_uses_transition_fade():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    parts, _ = _xfade_filter_parts(kept, 0.1, FPS)
    assert "transition=fade" in parts[0]


# ── _boundary_timestamps ──────────────────────────────────────────────────────

def test_boundary_timestamps_two_segments():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    result = _boundary_timestamps(kept, 0.1, FPS)
    assert len(result) == 1
    expected = max(0.0, 5.0 - 1 * (0.1 + FRAME_S))
    assert result[0] == f"{expected:.3f}"


def test_boundary_timestamps_three_segments():
    kept = [Segment(0.0, 3.0), Segment(3.0, 6.0), Segment(6.0, 9.0)]
    result = _boundary_timestamps(kept, 0.1, FPS)
    assert len(result) == 2


def test_boundary_timestamps_no_crossfade():
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    result = _boundary_timestamps(kept, 0.0, FPS)
    # offset = 5.0 - 1*(0 + 0.08) = 4.92
    expected = max(0.0, 5.0 - FRAME_S)
    assert result[0] == f"{expected:.3f}"


def test_boundary_timestamps_clamped_to_zero():
    kept = [Segment(0.0, 0.05), Segment(0.05, 0.1)]
    result = _boundary_timestamps(kept, 0.04, FPS)
    assert result[0] == "0.000"


# ── _cut_video_reencode error handling ────────────────────────────────────────

def test_cut_video_reencode_raises_on_ffmpeg_failure():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = b"some ffmpeg error"

    with patch("autocut.output.cutter.subprocess.run", return_value=mock_result):
        from autocut.output.cutter import _cut_video_reencode
        from pathlib import Path
        with pytest.raises(RuntimeError, match="FFmpeg re-encode failed"):
            _cut_video_reencode(Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"), 0.0, FPS)


def test_cut_video_reencode_error_includes_stderr():
    mock_result = MagicMock()
    mock_result.returncode = 234
    mock_result.stderr = b"rate of 1/0 is invalid"

    with patch("autocut.output.cutter.subprocess.run", return_value=mock_result):
        from autocut.output.cutter import _cut_video_reencode
        from pathlib import Path
        with pytest.raises(RuntimeError, match="rate of 1/0 is invalid"):
            _cut_video_reencode(Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"), 0.0, FPS)


# ── cut_video path routing ────────────────────────────────────────────────────

def _media(duration: float = 10.0, fps: float = FPS) -> MediaInfo:
    return MediaInfo(duration_s=duration, fps=fps, has_audio=True)


def test_cut_video_uses_stream_copy_when_no_processing(tmp_path):
    bad = [_bad(3.0, 5.0)]
    out = tmp_path / "out.mp4"
    with patch.object(cutter_module, "_cut_stream_copy") as mock_copy, \
         patch.object(cutter_module, "_cut_with_audio_processing") as mock_proc:
        cut_video(tmp_path / "in.mp4", bad, _media(), out, AutoCutConfig())
    mock_copy.assert_called_once()
    mock_proc.assert_not_called()


def test_cut_video_uses_audio_processing_when_crossfade_set(tmp_path):
    bad = [_bad(3.0, 5.0)]
    out = tmp_path / "out.mp4"
    cfg = AutoCutConfig(crossfade_ms=120)
    with patch.object(cutter_module, "_cut_stream_copy") as mock_copy, \
         patch.object(cutter_module, "_cut_with_audio_processing") as mock_proc:
        cut_video(tmp_path / "in.mp4", bad, _media(), out, cfg)
    mock_proc.assert_called_once()
    mock_copy.assert_not_called()


def test_cut_video_uses_audio_processing_when_room_eq_active(tmp_path):
    bad = [_bad(3.0, 5.0)]
    out = tmp_path / "out.mp4"
    cfg = AutoCutConfig(room_eq_enabled=True)
    with patch.object(cutter_module, "_cut_stream_copy") as mock_copy, \
         patch.object(cutter_module, "_cut_with_audio_processing") as mock_proc:
        cut_video(tmp_path / "in.mp4", bad, _media(), out, cfg, resonant_freqs=[200.0])
    mock_proc.assert_called_once()
    mock_copy.assert_not_called()


def test_cut_video_raises_when_all_segments_cut(tmp_path):
    bad = [_bad(0.0, 10.0)]
    with pytest.raises(ValueError, match="No segments left"):
        cut_video(tmp_path / "in.mp4", bad, _media(10.0), tmp_path / "out.mp4", AutoCutConfig())
