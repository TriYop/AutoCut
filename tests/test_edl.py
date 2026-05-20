"""Tests for autocut.output.edl — timecode conversion, EDL and JSON export."""

import json
from pathlib import Path

import pytest

from autocut.models import BadSegment, MediaInfo, Segment, SegmentSource
from autocut.output.edl import _seconds_to_timecode, write_edl, write_json


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bad(start: float, end: float, label: str = "silence", source: SegmentSource = SegmentSource.VAD) -> BadSegment:
    return BadSegment(Segment(start, end), source, label)


def _media(fps: float = 25.0) -> MediaInfo:
    return MediaInfo(duration_s=60.0, fps=fps, has_audio=True)


# ── _seconds_to_timecode ──────────────────────────────────────────────────────

def test_tc_zero():
    assert _seconds_to_timecode(0.0, 25.0) == "00:00:00:00"


def test_tc_one_second_25fps():
    assert _seconds_to_timecode(1.0, 25.0) == "00:00:01:00"


def test_tc_one_frame_25fps():
    assert _seconds_to_timecode(1.0 / 25.0, 25.0) == "00:00:00:01"


def test_tc_mid_second_frames_25fps():
    # 0.5 s × 25 fps = 12.5 → round to 12 frames
    assert _seconds_to_timecode(0.5, 25.0) == "00:00:00:12"


def test_tc_cross_minute():
    assert _seconds_to_timecode(60.0, 25.0) == "00:01:00:00"


def test_tc_cross_hour():
    assert _seconds_to_timecode(3600.0, 25.0) == "01:00:00:00"


def test_tc_complex_timestamp():
    # 1h 2m 3s 4f at 25fps → total_frames = (3600+120+3)*25 + 4 = 185_329
    assert _seconds_to_timecode(3723.0 + 4.0 / 25.0, 25.0) == "01:02:03:04"


def test_tc_29_97fps_one_second():
    # 1.0 s × 29.97 = 29.97 → round to 30 frames = frame 0 of next second
    assert _seconds_to_timecode(1.0, 29.97) == "00:00:01:00"


def test_tc_23_976fps_one_second():
    # 1.0 × 23.976 = 23.976 → round to 24 frames = frame 0 of next second
    assert _seconds_to_timecode(1.0, 23.976) == "00:00:01:00"


# ── write_edl ─────────────────────────────────────────────────────────────────

def test_write_edl_empty_segments(tmp_path):
    out = tmp_path / "output.edl"
    write_edl([], _media(), "clip.mp4", out)
    content = out.read_text(encoding="utf-8")
    assert "TITLE: AutoCut - clip.mp4" in content
    assert "FCM: NON-DROP FRAME" in content
    # No event entries
    assert "001" not in content


def test_write_edl_single_event(tmp_path):
    out = tmp_path / "output.edl"
    bad = [_bad(2.0, 4.0, "silence")]
    write_edl(bad, _media(), "video.mp4", out)
    content = out.read_text(encoding="utf-8")
    assert "001" in content
    assert "silence" in content
    assert "vad" in content


def test_write_edl_event_numbering(tmp_path):
    out = tmp_path / "output.edl"
    bad = [_bad(1.0, 2.0), _bad(5.0, 6.0), _bad(9.0, 10.0)]
    write_edl(bad, _media(), "clip.mp4", out)
    content = out.read_text(encoding="utf-8")
    assert "001" in content
    assert "002" in content
    assert "003" in content


def test_write_edl_timecodes_present(tmp_path):
    out = tmp_path / "output.edl"
    # 2.0 s at 25fps → 00:00:02:00
    bad = [_bad(2.0, 4.0)]
    write_edl(bad, _media(fps=25.0), "clip.mp4", out)
    content = out.read_text(encoding="utf-8")
    assert "00:00:02:00" in content
    assert "00:00:04:00" in content


def test_write_edl_source_in_header(tmp_path):
    out = tmp_path / "output.edl"
    write_edl([], _media(), "my_webinar.mp4", out)
    assert "my_webinar.mp4" in out.read_text(encoding="utf-8")


def test_write_edl_comment_contains_source_enum(tmp_path):
    out = tmp_path / "output.edl"
    bad = [_bad(1.0, 2.0, "euh", SegmentSource.WHISPER_FILLER)]
    write_edl(bad, _media(), "clip.mp4", out)
    content = out.read_text(encoding="utf-8")
    assert "whisper_filler" in content


# ── write_json ────────────────────────────────────────────────────────────────

def test_write_json_empty_segments(tmp_path):
    out = tmp_path / "output.json"
    write_json([], "clip.mp4", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source"] == "clip.mp4"
    assert data["cuts"] == []


def test_write_json_single_cut(tmp_path):
    out = tmp_path / "output.json"
    bad = [_bad(1.0, 3.0, "silence")]
    write_json(bad, "clip.mp4", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["cuts"]) == 1
    cut = data["cuts"][0]
    assert cut["start"] == pytest.approx(1.0)
    assert cut["end"] == pytest.approx(3.0)
    assert cut["duration"] == pytest.approx(2.0)
    assert cut["label"] == "silence"
    assert cut["source"] == "vad"
    assert cut["confidence"] == pytest.approx(1.0)


def test_write_json_duration_computed(tmp_path):
    out = tmp_path / "output.json"
    bad = [_bad(2.5, 4.0)]
    write_json(bad, "clip.mp4", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cuts"][0]["duration"] == pytest.approx(1.5)


def test_write_json_unicode_label_not_escaped(tmp_path):
    out = tmp_path / "output.json"
    bad = [_bad(1.0, 2.0, "voilà")]
    write_json(bad, "clip.mp4", out)
    raw = out.read_text(encoding="utf-8")
    assert "voilà" in raw
    assert r"\u" not in raw


def test_write_json_confidence_preserved(tmp_path):
    out = tmp_path / "output.json"
    seg = BadSegment(Segment(1.0, 2.0), SegmentSource.WHISPER_FILLER, "euh", confidence=0.73)
    write_json([seg], "clip.mp4", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cuts"][0]["confidence"] == pytest.approx(0.73)


def test_write_json_multiple_cuts_ordered(tmp_path):
    out = tmp_path / "output.json"
    bad = [_bad(1.0, 2.0), _bad(5.0, 6.0)]
    write_json(bad, "clip.mp4", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["cuts"]) == 2
    assert data["cuts"][0]["start"] == pytest.approx(1.0)
    assert data["cuts"][1]["start"] == pytest.approx(5.0)
