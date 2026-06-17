"""Tests for autocut.output.cutter — pure functions and subprocess error handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, MediaInfo, Segment, SegmentSource
from autocut.output import cutter as cutter_module
from autocut.output.cutter import (
    _apply_crossfade,
    _boundary_timestamps,
    _compute_kept_segments,
    _effective_crossfade_s,
    _segment_filter_chain,
    _xfade_filter_parts,
    cut_video,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bad(start: float, end: float) -> BadSegment:
    """Return a VAD-sourced BadSegment spanning [start, end)."""
    return BadSegment(Segment(start, end), SegmentSource.VAD, "silence")


FPS = 25.0
# One frame duration in seconds at FPS (2 frames of slack used in offset formulae)
FRAME_S = 2.0 / FPS


# ── _compute_kept_segments ────────────────────────────────────────────────────

def test_kept_no_bad_segments_returns_full_duration():
    """With no bad segments the entire duration is returned as a single kept segment."""
    assert _compute_kept_segments([], 10.0) == [Segment(0.0, 10.0)]


def test_kept_bad_at_start():
    """A bad segment at the start leaves only the tail as kept."""
    assert _compute_kept_segments([_bad(0.0, 3.0)], 10.0) == [Segment(3.0, 10.0)]


def test_kept_bad_at_end():
    """A bad segment at the end leaves only the head as kept."""
    assert _compute_kept_segments([_bad(8.0, 10.0)], 10.0) == [Segment(0.0, 8.0)]


def test_kept_bad_in_middle():
    """A bad segment in the middle splits the duration into two kept segments."""
    assert _compute_kept_segments([_bad(3.0, 5.0)], 10.0) == [
        Segment(0.0, 3.0),
        Segment(5.0, 10.0),
    ]


def test_kept_multiple_bad_segments_unsorted():
    """Out-of-order bad segments are sorted before computing the complement."""
    bad = [_bad(5.0, 7.0), _bad(1.0, 3.0)]
    kept = _compute_kept_segments(bad, 10.0)
    assert kept == [Segment(0.0, 1.0), Segment(3.0, 5.0), Segment(7.0, 10.0)]


def test_kept_filters_segments_shorter_than_10ms():
    """Kept segments shorter than 10 ms are dropped from the result."""
    # 10.0 - 9.995 = 0.005 s < 0.01 threshold
    kept = _compute_kept_segments([_bad(0.0, 9.995)], 10.0)
    assert kept == []


def test_kept_overlapping_bad_segments():
    """Overlapping bad segments are unified before computing the complement."""
    bad = [_bad(1.0, 5.0), _bad(3.0, 7.0)]
    kept = _compute_kept_segments(bad, 10.0)
    assert kept == [Segment(0.0, 1.0), Segment(7.0, 10.0)]


def test_kept_bad_covers_full_duration():
    """A bad segment spanning the entire duration leaves nothing to keep."""
    assert _compute_kept_segments([_bad(0.0, 10.0)], 10.0) == []


def test_kept_bad_end_beyond_duration():
    """A bad segment extending past the video end leaves only the initial head."""
    kept = _compute_kept_segments([_bad(8.0, 15.0)], 10.0)
    assert kept == [Segment(0.0, 8.0)]


def test_kept_zero_duration_video():
    """A zero-duration video has no kept segments."""
    assert _compute_kept_segments([], 0.0) == []


def test_kept_segment_exactly_10ms_excluded():
    """A kept segment of exactly 0.01 s does not pass the > 0.01 filter."""
    kept = _compute_kept_segments([_bad(0.0, 9.99)], 10.0)
    assert kept == []


def test_kept_segment_just_above_10ms_included():
    """A kept segment of 0.011 s passes the > 0.01 filter."""
    # 10.0 - 9.989 = 0.011 > 0.01 → kept
    kept = _compute_kept_segments([_bad(0.0, 9.989)], 10.0)
    assert len(kept) == 1
    assert kept[0].duration == pytest.approx(0.011)


def test_kept_segment_just_below_10ms_dropped():
    """A kept segment of 0.005 s is below the threshold and is dropped."""
    # 10.0 - 9.995 = 0.005 < 0.01 → dropped
    assert _compute_kept_segments([_bad(0.0, 9.995)], 10.0) == []


def test_kept_two_bads_covering_whole_duration():
    """Two adjacent bad segments covering the whole duration leave nothing to keep."""
    bad = [_bad(0.0, 5.0), _bad(5.0, 10.0)]
    assert _compute_kept_segments(bad, 10.0) == []


# ── _effective_crossfade_s ────────────────────────────────────────────────────

def test_effective_crossfade_zero_ms():
    """A crossfade_ms of 0 always returns 0.0 seconds."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    assert _effective_crossfade_s(kept, 0, FPS) == 0.0


def test_effective_crossfade_negative_ms():
    """A negative crossfade_ms is treated as zero."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    assert _effective_crossfade_s(kept, -100, FPS) == 0.0


def test_effective_crossfade_single_segment():
    """With only one segment there are no transitions, so 0.0 is returned."""
    assert _effective_crossfade_s([Segment(0.0, 5.0)], 200, FPS) == 0.0


def test_effective_crossfade_within_limit():
    """A requested crossfade well within the segment-length limit is returned unchanged."""
    # min_dur=5s, limit = 5/2 - 0.08 = 2.42s; request 0.2s → returned as-is
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    assert _effective_crossfade_s(kept, 200, FPS) == pytest.approx(0.2)


def test_effective_crossfade_clamped_to_segment_limit():
    """A crossfade that would exceed half the shortest segment is clamped."""
    # min_dur=0.5s, limit = 0.25 - 0.08 = 0.17s; request 0.5s → clamped
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    expected = 0.5 / 2.0 - FRAME_S
    assert _effective_crossfade_s(kept, 500, FPS) == pytest.approx(expected)


def test_effective_crossfade_zero_when_segment_too_short():
    """When the shortest segment is too short for any crossfade, 0.0 is returned."""
    # min_dur=0.1s, limit = 0.05 - 0.08 = -0.03 → clamped to 0
    kept = [Segment(0.0, 0.1), Segment(0.1, 0.2)]
    assert _effective_crossfade_s(kept, 200, FPS) == 0.0


def test_effective_crossfade_uses_shortest_segment():
    """The clamping limit is determined by the shortest segment, not the average."""
    kept = [Segment(0.0, 10.0), Segment(10.0, 10.5), Segment(10.5, 20.0)]
    # min_dur = 0.5s, limit = 0.25 - 0.08 = 0.17s
    expected = 0.5 / 2.0 - FRAME_S
    assert _effective_crossfade_s(kept, 500, FPS) == pytest.approx(expected)


# ── _apply_crossfade ──────────────────────────────────────────────────────────

# Sample rate used for all _apply_crossfade tests
SR = 1000


def test_apply_crossfade_no_crossfade_is_plain_concat():
    """With crossfade_s=0 the result is the plain concatenation of kept chunks."""
    audio = np.ones(1000, dtype=np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    result = _apply_crossfade(audio, SR, kept, 0.0)
    assert len(result) == 1000
    np.testing.assert_array_almost_equal(result, audio)


def test_apply_crossfade_single_segment_slices_correctly():
    """A single kept segment returns the exact audio slice without modification."""
    audio = np.arange(1000, dtype=np.float32)
    kept = [Segment(0.1, 0.4)]
    result = _apply_crossfade(audio, SR, kept, 0.0)
    assert len(result) == 300
    np.testing.assert_array_equal(result, audio[100:400])


def test_apply_crossfade_empty_kept_returns_empty():
    """An empty kept list returns an empty float32 array."""
    audio = np.ones(1000, dtype=np.float32)
    result = _apply_crossfade(audio, SR, [], 0.0)
    assert len(result) == 0
    assert result.dtype == np.float32


def test_apply_crossfade_reduces_total_length():
    """Two segments with a crossfade produce output shorter by one crossfade window."""
    audio = np.ones(1000, dtype=np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    crossfade_s = 0.1
    result = _apply_crossfade(audio, SR, kept, crossfade_s)
    # 500 + 500 - 100 overlap = 900
    assert len(result) == 900


def test_apply_crossfade_three_segments_length():
    """Three segments with crossfade produce output shorter by two crossfade windows."""
    audio = np.ones(1500, dtype=np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0), Segment(1.0, 1.5)]
    crossfade_s = 0.1
    result = _apply_crossfade(audio, SR, kept, crossfade_s)
    # 3×500 - 2×100 = 1300
    assert len(result) == 1300


def test_apply_crossfade_blend_at_boundary():
    """The crossfade region blends the tail of one segment into the head of the next."""
    # Segment 1: all 0s; segment 2: all 1s; crossfade of 100 samples
    audio = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    crossfade_s = 0.1
    result = _apply_crossfade(audio, SR, kept, crossfade_s)
    # Midpoint of crossfade region: fade-out~0.5 and fade-in~0.5 → ~0.5
    mid = result[449]
    assert 0.3 < mid < 0.7


def test_apply_crossfade_fade_n_zero_treated_as_plain_concat():
    """When crossfade_s rounds to 0 samples, the function falls back to plain concat."""
    # At sr=10, crossfade_s=0.001 → int(0.01) = 0 → must not crash, treated as no crossfade
    audio = np.ones(20, dtype=np.float32)
    kept = [Segment(0.0, 1.0), Segment(1.0, 2.0)]
    result = _apply_crossfade(audio, 10, kept, 0.001)
    assert len(result) == 20
    np.testing.assert_array_almost_equal(result, audio)


def test_apply_crossfade_chunk_shorter_than_fade():
    """When a chunk is shorter than fade_n, fn is clamped and no error is raised."""
    # next_chunk shorter than fade_n → fn = min(fade_n, ..., len(next_chunk))
    sr = 1000
    audio = np.concatenate([np.ones(500), np.ones(50)]).astype(np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 0.55)]
    # crossfade_s=0.2 → fade_n=200, but next_chunk has 50 samples → fn=50
    result = _apply_crossfade(audio, sr, kept, 0.2)
    # Should not raise; result length < 550
    assert len(result) < 550


def test_apply_crossfade_negative_crossfade_treated_as_zero():
    """A negative crossfade_s is treated as no crossfade (plain concat)."""
    audio = np.ones(1000, dtype=np.float32)
    kept = [Segment(0.0, 0.5), Segment(0.5, 1.0)]
    result = _apply_crossfade(audio, 1000, kept, -0.1)
    assert len(result) == 1000


# ── _segment_filter_chain ─────────────────────────────────────────────────────

def test_segment_filter_chain_format():
    """The filter chain string matches the expected trim+setpts+fps format exactly."""
    seg = Segment(1.5, 4.0)
    result = _segment_filter_chain(2, seg, 25.0)
    assert result == "[0:v]trim=start=1.500000:end=4.000000,setpts=PTS-STARTPTS,fps=25[v2]"


def test_segment_filter_chain_index_in_label():
    """The segment index appears in the output label [vN]."""
    result = _segment_filter_chain(7, Segment(0.0, 1.0), 25.0)
    assert result.endswith("[v7]")


def test_segment_filter_chain_fps_rounded():
    """A fractional fps (29.97) is rounded to the nearest integer for the fps filter."""
    result = _segment_filter_chain(0, Segment(0.0, 1.0), 29.97)
    assert ",fps=30[v0]" in result


def test_segment_filter_chain_fps_24():
    """23.976 fps rounds to 24 in the fps filter."""
    result = _segment_filter_chain(0, Segment(0.0, 1.0), 23.976)
    assert ",fps=24[v0]" in result


# ── _xfade_filter_parts ───────────────────────────────────────────────────────

def test_xfade_two_segments_one_part():
    """Two segments produce exactly one xfade part with label xf1."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    parts, label = _xfade_filter_parts(kept, 0.1, FPS)
    assert len(parts) == 1
    assert label == "xf1"


def test_xfade_first_part_inputs():
    """The first xfade part consumes [v0] and [v1] as inputs."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    parts, _ = _xfade_filter_parts(kept, 0.1, FPS)
    assert parts[0].startswith("[v0][v1]xfade")


def test_xfade_three_segments_chained():
    """Three segments produce two chained xfade parts, ending at xf2."""
    kept = [Segment(0.0, 3.0), Segment(3.0, 6.0), Segment(6.0, 9.0)]
    parts, label = _xfade_filter_parts(kept, 0.1, FPS)
    assert len(parts) == 2
    assert label == "xf2"
    assert "[v0][v1]" in parts[0]
    assert "[xf1][v2]" in parts[1]


def test_xfade_offset_formula():
    """The xfade offset equals segment_duration minus k*(crossfade_s + frame_slack)."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    crossfade_s = 0.1
    parts, _ = _xfade_filter_parts(kept, crossfade_s, FPS)
    expected = 5.0 - 1 * (crossfade_s + FRAME_S)
    assert f"offset={expected:.6f}" in parts[0]


def test_xfade_offset_clamped_to_zero():
    """The xfade offset is clamped to 0 when the formula would produce a negative value."""
    # Very short segments where offset would go negative
    kept = [Segment(0.0, 0.05), Segment(0.05, 0.10)]
    parts, _ = _xfade_filter_parts(kept, 0.04, FPS)
    assert "offset=0.000000" in parts[0]


def test_xfade_uses_transition_fade():
    """The xfade filter always uses the 'fade' transition type."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    parts, _ = _xfade_filter_parts(kept, 0.1, FPS)
    assert "transition=fade" in parts[0]


def test_xfade_single_segment_produces_no_parts():
    """A single-segment list produces no xfade parts (no transitions needed)."""
    kept = [Segment(0.0, 5.0)]
    parts, label = _xfade_filter_parts(kept, 0.1, FPS)
    assert parts == []
    assert label == "xf0"


# ── _boundary_timestamps ──────────────────────────────────────────────────────

def test_boundary_timestamps_two_segments():
    """Two segments produce one boundary timestamp at the first xfade offset."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    result = _boundary_timestamps(kept, 0.1, FPS)
    assert len(result) == 1
    expected = max(0.0, 5.0 - 1 * (0.1 + FRAME_S))
    assert result[0] == f"{expected:.3f}"


def test_boundary_timestamps_three_segments():
    """Three segments produce two boundary timestamps."""
    kept = [Segment(0.0, 3.0), Segment(3.0, 6.0), Segment(6.0, 9.0)]
    result = _boundary_timestamps(kept, 0.1, FPS)
    assert len(result) == 2


def test_boundary_timestamps_no_crossfade():
    """With crossfade_s=0 the boundary is placed one frame-slack before the cut."""
    kept = [Segment(0.0, 5.0), Segment(5.0, 10.0)]
    result = _boundary_timestamps(kept, 0.0, FPS)
    # offset = 5.0 - 1*(0 + 0.08) = 4.92
    expected = max(0.0, 5.0 - FRAME_S)
    assert result[0] == f"{expected:.3f}"


def test_boundary_timestamps_clamped_to_zero():
    """A computed boundary that would go negative is clamped to 0.000."""
    kept = [Segment(0.0, 0.05), Segment(0.05, 0.1)]
    result = _boundary_timestamps(kept, 0.04, FPS)
    assert result[0] == "0.000"


def test_boundary_timestamps_single_segment_returns_empty():
    """A single-segment list has no transitions and returns an empty list."""
    kept = [Segment(0.0, 5.0)]
    assert _boundary_timestamps(kept, 0.1, FPS) == []


# ── _cut_video_reencode error handling ────────────────────────────────────────

def _mock_popen(returncode: int, stderr_text: str) -> MagicMock:
    """Build a subprocess.Popen mock that fails with the given stderr."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = iter([])
    proc.stderr.read.return_value = stderr_text
    proc.wait.return_value = None
    return proc


def test_cut_video_reencode_raises_on_ffmpeg_failure():
    """A non-zero FFmpeg exit code raises RuntimeError mentioning 'FFmpeg re-encode failed'."""
    with patch("autocut.output.cutter.subprocess.Popen", return_value=_mock_popen(1, "some ffmpeg error")):
        from pathlib import Path

        from autocut.output.cutter import _cut_video_reencode
        with pytest.raises(RuntimeError, match="FFmpeg re-encode failed"):
            _cut_video_reencode(Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"), 0.0, FPS)


def test_cut_video_reencode_error_includes_stderr():
    """The RuntimeError message includes the raw FFmpeg stderr text."""
    with patch("autocut.output.cutter.subprocess.Popen", return_value=_mock_popen(234, "rate of 1/0 is invalid")):
        from pathlib import Path

        from autocut.output.cutter import _cut_video_reencode
        with pytest.raises(RuntimeError, match="rate of 1/0 is invalid"):
            _cut_video_reencode(Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"), 0.0, FPS)


def test_cut_video_reencode_single_segment_no_crossfade():
    """Single kept segment uses the v0 label path (no concat, no xfade)."""
    from autocut.output.cutter import _cut_video_reencode
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = iter([])
    proc.stderr.read.return_value = ""
    proc.wait.return_value = None

    with patch("autocut.output.cutter.subprocess.Popen", return_value=proc) as mock_popen:
        _cut_video_reencode(Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"), 0.0, FPS)

    cmd = mock_popen.call_args[0][0]
    # Single-segment path: filter_complex contains only the one trim chain for v0
    fc_idx = cmd.index("-filter_complex")
    fc = cmd[fc_idx + 1]
    assert "[v0]" in fc
    assert "concat" not in fc
    assert "xfade" not in fc


def test_cut_video_reencode_multi_segment_no_crossfade_uses_concat():
    """Multiple segments without crossfade use the concat filter path."""
    from autocut.output.cutter import _cut_video_reencode
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = iter([])
    proc.stderr.read.return_value = ""
    proc.wait.return_value = None

    kept = [Segment(0.0, 3.0), Segment(5.0, 8.0)]
    with patch("autocut.output.cutter.subprocess.Popen", return_value=proc) as mock_popen:
        _cut_video_reencode(Path("in.mp4"), kept, Path("out.mp4"), 0.0, FPS)

    cmd = mock_popen.call_args[0][0]
    fc_idx = cmd.index("-filter_complex")
    fc = cmd[fc_idx + 1]
    assert "concat" in fc
    assert "xfade" not in fc


def test_cut_video_reencode_progress_malformed_out_time_us_ignored():
    """Malformed out_time_us value (e.g. 'N/A') must not crash the progress loop."""
    from autocut.output.cutter import _cut_video_reencode

    calls: list = []
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = iter(["out_time_us=N/A\n", "out_time_us=3000000\n"])
    proc.stderr.read.return_value = ""
    proc.wait.return_value = None

    with patch("autocut.output.cutter.subprocess.Popen", return_value=proc):
        _cut_video_reencode(
            Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"),
            0.0, FPS, lambda c, t: calls.append(c),
        )

    # N/A line silently skipped; valid line fires callback
    assert calls == [pytest.approx(3.0)]


def test_cut_video_reencode_progress_zero_out_time():
    """out_time_us=0 should fire the callback with current_s=0.0."""
    from autocut.output.cutter import _cut_video_reencode

    calls: list = []
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = iter(["out_time_us=0\n"])
    proc.stderr.read.return_value = ""
    proc.wait.return_value = None

    with patch("autocut.output.cutter.subprocess.Popen", return_value=proc):
        _cut_video_reencode(
            Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"),
            0.0, FPS, lambda c, t: calls.append(c),
        )

    assert calls == [pytest.approx(0.0)]


# ── cut_video path routing ────────────────────────────────────────────────────

def _media(duration: float = 10.0, fps: float = FPS) -> MediaInfo:
    """Return a MediaInfo with the given duration and fps."""
    return MediaInfo(duration_s=duration, fps=fps, has_audio=True)


def test_cut_video_uses_stream_copy_when_no_processing(tmp_path):
    """Default config with no crossfade or room EQ uses the fast stream-copy path."""
    bad = [_bad(3.0, 5.0)]
    out = tmp_path / "out.mp4"
    with patch.object(cutter_module, "_cut_stream_copy") as mock_copy, \
         patch.object(cutter_module, "_cut_with_audio_processing") as mock_proc:
        cut_video(tmp_path / "in.mp4", bad, _media(), out, AutoCutConfig())
    mock_copy.assert_called_once()
    mock_proc.assert_not_called()


def test_cut_video_uses_audio_processing_when_crossfade_set(tmp_path):
    """A non-zero crossfade_ms forces the audio-processing path."""
    bad = [_bad(3.0, 5.0)]
    out = tmp_path / "out.mp4"
    cfg = AutoCutConfig(crossfade_ms=120)
    with patch.object(cutter_module, "_cut_stream_copy") as mock_copy, \
         patch.object(cutter_module, "_cut_with_audio_processing") as mock_proc:
        cut_video(tmp_path / "in.mp4", bad, _media(), out, cfg)
    mock_proc.assert_called_once()
    mock_copy.assert_not_called()


def test_cut_video_uses_audio_processing_when_room_eq_active(tmp_path):
    """Room EQ enabled with resonant frequencies forces the audio-processing path."""
    bad = [_bad(3.0, 5.0)]
    out = tmp_path / "out.mp4"
    cfg = AutoCutConfig(room_eq_enabled=True)
    with patch.object(cutter_module, "_cut_stream_copy") as mock_copy, \
         patch.object(cutter_module, "_cut_with_audio_processing") as mock_proc:
        cut_video(tmp_path / "in.mp4", bad, _media(), out, cfg, resonant_freqs=[200.0])
    mock_proc.assert_called_once()
    mock_copy.assert_not_called()


def test_cut_video_raises_when_all_segments_cut(tmp_path):
    """When all segments are removed, cut_video raises ValueError."""
    bad = [_bad(0.0, 10.0)]
    with pytest.raises(ValueError, match="No segments left"):
        cut_video(tmp_path / "in.mp4", bad, _media(10.0), tmp_path / "out.mp4", AutoCutConfig())


def test_cut_video_no_bad_segments_uses_stream_copy(tmp_path):
    """No bad segments → full duration kept → stream-copy path."""
    with patch.object(cutter_module, "_cut_stream_copy") as mock_copy, \
         patch.object(cutter_module, "_cut_with_audio_processing") as mock_proc:
        cut_video(tmp_path / "in.mp4", [], _media(10.0), tmp_path / "out.mp4", AutoCutConfig())
    mock_copy.assert_called_once()
    mock_proc.assert_not_called()
