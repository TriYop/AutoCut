from pathlib import Path
from unittest.mock import patch

import pytest

from autocut.config import AutoCutConfig
from autocut.exceptions import AutoCutError
from autocut.gui.worker import PipelineWorker
from autocut.models import BadSegment, MediaInfo, Segment, SegmentSource

_MEDIA = MediaInfo(duration_s=60.0, fps=25.0, has_audio=True)
_SEG = BadSegment(segment=Segment(1.0, 2.0), source=SegmentSource.VAD, label="silence")
_AUDIO = Path("/tmp/audio.wav")
_INPUT = Path("/tmp/video.mp4")
_OUT_DIR = Path("/tmp")
_CFG = AutoCutConfig()


def _worker(output_mode: str = "edl", config: AutoCutConfig = _CFG) -> PipelineWorker:
    return PipelineWorker(_INPUT, output_mode, _OUT_DIR, config)


@pytest.fixture(autouse=True)
def _patch_pipeline():
    """Replace every I/O-heavy call with a no-op stub."""
    with (
        patch("autocut.gui.worker.extract_audio", return_value=(_AUDIO, _MEDIA)),
        patch("autocut.gui.worker.detect_silences", return_value=[_SEG]),
        patch("autocut.gui.worker.detect_fillers_and_repetitions", return_value=([], [])),
        patch("autocut.gui.worker.merge_bad_segments", return_value=[_SEG]),
        patch("autocut.gui.worker.cut_video"),
        patch("autocut.gui.worker.write_edl"),
        patch("autocut.gui.worker.write_json"),
    ):
        yield


# ── Happy path ────────────────────────────────────────────────────────────────

def test_edl_mode_emits_finished(qtbot):
    worker = _worker("edl")
    with qtbot.waitSignal(worker.finished, timeout=2000):
        worker.run()


def test_video_mode_emits_finished(qtbot):
    worker = _worker("video")
    with qtbot.waitSignal(worker.finished, timeout=2000):
        worker.run()


def test_both_mode_emits_finished(qtbot):
    worker = _worker("both")
    with qtbot.waitSignal(worker.finished, timeout=2000):
        worker.run()


def test_log_lines_cover_every_stage(qtbot):
    worker = _worker()
    lines: list[str] = []
    worker.log_line.connect(lines.append)
    with qtbot.waitSignal(worker.finished, timeout=2000):
        worker.run()
    joined = "\n".join(lines)
    assert "Extracting" in joined
    assert "VAD" in joined
    assert "Whisper" in joined
    assert "Found" in joined


def test_finished_carries_pipeline_result(qtbot):
    from autocut.pipeline.runner import PipelineResult
    worker = _worker()
    results: list = []
    worker.finished.connect(results.append)
    with qtbot.waitSignal(worker.finished, timeout=2000):
        worker.run()
    assert isinstance(results[0], PipelineResult)
    assert results[0].input_path == _INPUT


# ── Output routing ────────────────────────────────────────────────────────────

def test_edl_mode_writes_edl_and_json(qtbot):
    with (
        patch("autocut.gui.worker.write_edl") as mock_edl,
        patch("autocut.gui.worker.write_json") as mock_json,
        patch("autocut.gui.worker.cut_video") as mock_video,
    ):
        worker = _worker("edl")
        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.run()
    mock_edl.assert_called_once()
    mock_json.assert_called_once()
    mock_video.assert_not_called()


def test_video_mode_calls_cut_video_only(qtbot):
    with (
        patch("autocut.gui.worker.write_edl") as mock_edl,
        patch("autocut.gui.worker.cut_video") as mock_video,
    ):
        worker = _worker("video")
        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.run()
    mock_video.assert_called_once()
    mock_edl.assert_not_called()


def test_both_mode_writes_all_outputs(qtbot):
    with (
        patch("autocut.gui.worker.write_edl") as mock_edl,
        patch("autocut.gui.worker.write_json") as mock_json,
        patch("autocut.gui.worker.cut_video") as mock_video,
    ):
        worker = _worker("both")
        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.run()
    mock_edl.assert_called_once()
    mock_json.assert_called_once()
    mock_video.assert_called_once()


# ── Room EQ ───────────────────────────────────────────────────────────────────

def test_room_eq_not_called_when_disabled(qtbot):
    with patch("autocut.gui.worker.analyze_room_resonances") as mock_eq:
        worker = _worker(config=AutoCutConfig(room_eq_enabled=False))
        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.run()
    mock_eq.assert_not_called()


def test_room_eq_called_when_enabled(qtbot):
    with patch("autocut.gui.worker.analyze_room_resonances", return_value=[440.0]) as mock_eq:
        worker = _worker(config=AutoCutConfig(room_eq_enabled=True))
        with qtbot.waitSignal(worker.finished, timeout=2000):
            worker.run()
    mock_eq.assert_called_once()


# ── Error handling ────────────────────────────────────────────────────────────

def test_autocut_error_emits_error_signal(qtbot):
    with patch("autocut.gui.worker.extract_audio", side_effect=AutoCutError("bad file")):
        worker = _worker()
        with qtbot.waitSignal(worker.error, timeout=2000) as blocker:
            worker.run()
    assert "bad file" in blocker.args[0]


def test_autocut_error_does_not_emit_finished(qtbot):
    with patch("autocut.gui.worker.extract_audio", side_effect=AutoCutError("bad file")):
        worker = _worker()
        with qtbot.assertNotEmitted(worker.finished):
            with qtbot.waitSignal(worker.error, timeout=2000):
                worker.run()


def test_unexpected_exception_emits_error_with_prefix(qtbot):
    with patch("autocut.gui.worker.extract_audio", side_effect=RuntimeError("disk full")):
        worker = _worker()
        with qtbot.waitSignal(worker.error, timeout=2000) as blocker:
            worker.run()
    assert "Unexpected error" in blocker.args[0]
    assert "disk full" in blocker.args[0]
