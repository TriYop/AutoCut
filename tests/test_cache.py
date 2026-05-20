"""Tests for autocut.pipeline.cache — load/save/invalidation."""

import json
import time
from pathlib import Path

import pytest

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment, SegmentSource
from autocut.pipeline import cache as pipeline_cache


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bad(start: float, end: float, source: SegmentSource = SegmentSource.VAD) -> BadSegment:
    return BadSegment(Segment(start, end), source, "silence")


def _whisper_bad(start: float, end: float) -> BadSegment:
    return BadSegment(Segment(start, end), SegmentSource.WHISPER_FILLER, "euh", confidence=0.9)


def _rep_bad(start: float, end: float) -> BadSegment:
    return BadSegment(Segment(start, end), SegmentSource.WHISPER_REPETITION, "repetition:donc")


CFG = AutoCutConfig()


# ── cache_path ────────────────────────────────────────────────────────────────

def test_cache_path_suffix(tmp_path):
    p = tmp_path / "video.mp4"
    assert pipeline_cache.cache_path(p) == tmp_path / "video.mp4.autocut-cache.json"


def test_cache_path_preserves_original_suffix(tmp_path):
    p = tmp_path / "clip.mov"
    assert pipeline_cache.cache_path(p).name == "clip.mov.autocut-cache.json"


# ── round-trip ────────────────────────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    vad = [_bad(1.0, 2.0), _bad(5.0, 6.0)]
    whisper = [_whisper_bad(3.0, 3.5), _rep_bad(7.0, 7.3)]

    pipeline_cache.save(video, vad, whisper, CFG)
    result = pipeline_cache.load(video, CFG)

    assert result is not None
    loaded_vad, loaded_whisper = result
    assert len(loaded_vad) == 2
    assert len(loaded_whisper) == 2
    assert loaded_vad[0].segment.start == pytest.approx(1.0)
    assert loaded_vad[1].segment.end == pytest.approx(6.0)
    assert loaded_whisper[0].source == SegmentSource.WHISPER_FILLER
    assert loaded_whisper[1].source == SegmentSource.WHISPER_REPETITION


def test_save_preserves_confidence(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    whisper = [_whisper_bad(1.0, 1.5)]

    pipeline_cache.save(video, [], whisper, CFG)
    _, loaded_whisper = pipeline_cache.load(video, CFG)

    assert loaded_whisper[0].confidence == pytest.approx(0.9)


def test_save_empty_segments(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [], [], CFG)
    result = pipeline_cache.load(video, CFG)
    assert result == ([], [])


# ── invalidation ─────────────────────────────────────────────────────────────

def test_load_returns_none_when_no_cache(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    assert pipeline_cache.load(video, CFG) is None


def test_load_returns_none_after_file_content_change(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"version1")
    pipeline_cache.save(video, [_bad(1.0, 2.0)], [], CFG)

    video.write_bytes(b"version2_longer")
    assert pipeline_cache.load(video, CFG) is None


def test_load_returns_none_after_mtime_change(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [_bad(1.0, 2.0)], [], CFG)

    future_mtime = time.time() + 10
    import os
    os.utime(video, (future_mtime, future_mtime))
    assert pipeline_cache.load(video, CFG) is None


def test_load_returns_none_when_config_changes(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [_bad(1.0, 2.0)], [], CFG)

    different_cfg = AutoCutConfig(vad_min_silence_duration_ms=300)
    assert pipeline_cache.load(video, different_cfg) is None


def test_load_returns_none_on_corrupt_json(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.cache_path(video).write_text("not json", encoding="utf-8")
    assert pipeline_cache.load(video, CFG) is None


def test_load_returns_none_on_wrong_format_version(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [], [], CFG)

    cf = pipeline_cache.cache_path(video)
    data = json.loads(cf.read_text())
    data["format_version"] = 99
    cf.write_text(json.dumps(data), encoding="utf-8")

    assert pipeline_cache.load(video, CFG) is None


def test_save_silently_ignores_write_error(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    read_only_dir = tmp_path / "ro"
    read_only_dir.mkdir()
    video2 = read_only_dir / "video.mp4"
    video2.write_bytes(b"fake")
    read_only_dir.chmod(0o555)

    try:
        pipeline_cache.save(video2, [], [], CFG)
    except Exception as e:
        pytest.fail(f"save() raised unexpectedly: {e}")
    finally:
        read_only_dir.chmod(0o755)


# ── config hash covers all detection fields ───────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("vad_speech_pad_ms", 50),
    ("whisper_model", "medium"),
    ("whisper_language", "fr"),
    ("whisper_compute_type", "float32"),
    ("min_filler_duration_s", 0.5),
    ("detect_repetitions", False),
    ("repetition_window_words", 5),
    ("repetition_min_word_length", 3),
])
def test_config_change_invalidates_cache(tmp_path, field, value):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [_bad(1.0, 2.0)], [], CFG)

    altered = AutoCutConfig(**{field: value})
    assert pipeline_cache.load(video, altered) is None


# ── progress callback passthrough ─────────────────────────────────────────────

def test_progress_callback_called_during_reencode():
    """Verify cut_video forwards progress_cb into _cut_video_reencode."""
    from unittest.mock import MagicMock, patch
    from autocut.output.cutter import _cut_video_reencode

    progress_calls: list[tuple[float, float]] = []

    def _cb(current_s: float, total_s: float) -> None:
        progress_calls.append((current_s, total_s))

    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = iter(["out_time_us=2000000\n", "progress=end\n"])
    proc.stderr.read.return_value = ""
    proc.wait.return_value = None

    with patch("autocut.output.cutter.subprocess.Popen", return_value=proc):
        _cut_video_reencode(Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"), 0.0, 25.0, _cb)

    assert len(progress_calls) == 1
    assert progress_calls[0][0] == pytest.approx(2.0)
    assert progress_calls[0][1] == pytest.approx(5.0)


def test_progress_callback_not_required():
    """_cut_video_reencode must not crash when progress_cb is None."""
    from unittest.mock import MagicMock, patch
    from autocut.output.cutter import _cut_video_reencode

    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = iter(["out_time_us=1000000\n"])
    proc.stderr.read.return_value = ""
    proc.wait.return_value = None

    with patch("autocut.output.cutter.subprocess.Popen", return_value=proc):
        _cut_video_reencode(Path("in.mp4"), [Segment(0.0, 5.0)], Path("out.mp4"), 0.0, 25.0, None)


# ── cache JSON robustness ─────────────────────────────────────────────────────

def test_load_missing_vad_segments_key_returns_empty(tmp_path):
    """Cache with no 'vad_segments' key returns ([], whisper)."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [], [], CFG)

    cf = pipeline_cache.cache_path(video)
    data = json.loads(cf.read_text())
    del data["vad_segments"]
    cf.write_text(json.dumps(data), encoding="utf-8")

    result = pipeline_cache.load(video, CFG)
    assert result == ([], [])


def test_load_missing_whisper_segments_key_returns_empty(tmp_path):
    """Cache with no 'whisper_segments' key returns (vad, [])."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [_bad(1.0, 2.0)], [], CFG)

    cf = pipeline_cache.cache_path(video)
    data = json.loads(cf.read_text())
    del data["whisper_segments"]
    cf.write_text(json.dumps(data), encoding="utf-8")

    result = pipeline_cache.load(video, CFG)
    assert result is not None
    vad, whisper = result
    assert len(vad) == 1
    assert whisper == []


def test_load_missing_fingerprint_key_returns_none(tmp_path):
    """Cache without 'fingerprint' key cannot match → returns None."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [], [], CFG)

    cf = pipeline_cache.cache_path(video)
    data = json.loads(cf.read_text())
    del data["fingerprint"]
    cf.write_text(json.dumps(data), encoding="utf-8")

    assert pipeline_cache.load(video, CFG) is None


def test_load_invalid_segment_source_returns_none(tmp_path):
    """Cache containing an unknown SegmentSource value must return None, not crash."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [_bad(1.0, 2.0)], [], CFG)

    cf = pipeline_cache.cache_path(video)
    data = json.loads(cf.read_text())
    data["vad_segments"][0]["source"] = "unknown_source_value"
    cf.write_text(json.dumps(data), encoding="utf-8")

    assert pipeline_cache.load(video, CFG) is None


def test_load_segment_missing_required_key_returns_none(tmp_path):
    """Cache with a segment dict missing 'start' must return None, not crash."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    pipeline_cache.save(video, [_bad(1.0, 2.0)], [], CFG)

    cf = pipeline_cache.cache_path(video)
    data = json.loads(cf.read_text())
    del data["vad_segments"][0]["start"]
    cf.write_text(json.dumps(data), encoding="utf-8")

    assert pipeline_cache.load(video, CFG) is None


# ── cache_path edge cases ─────────────────────────────────────────────────────

def test_cache_path_file_without_extension(tmp_path):
    """Files without an extension still get a sidecar path."""
    p = tmp_path / "Makefile"
    result = pipeline_cache.cache_path(p)
    assert result.name == "Makefile.autocut-cache.json"


def test_cache_path_double_extension(tmp_path):
    """Files like 'video.mkv' preserve both extension parts in sidecar name."""
    p = tmp_path / "video.tar.gz"
    result = pipeline_cache.cache_path(p)
    assert result.name == "video.tar.gz.autocut-cache.json"
