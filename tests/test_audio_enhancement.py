"""Tests for de-esser and click/plosive detection."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import soundfile as sf

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment, SegmentSource
from autocut.pipeline.audio_enhancement import (
    detect_sibilants,
    detect_clicks_and_plosives,
    build_deeser_filter,
    build_click_removal_filter,
)


@pytest.fixture
def config():
    """Standard test config with audio enhancement enabled."""
    return AutoCutConfig(
        deeser_enabled=True,
        deeser_threshold_db=8.0,
        click_removal_enabled=True,
        click_threshold_db=12.0,
    )


@pytest.fixture
def sample_wav():
    """Create a temporary WAV file with synthetic sibilant and click signals."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Sibilant content: peak at 6 kHz (simulates harsh 's' sound)
    sibilant = 0.3 * np.sin(2 * np.pi * 6000 * t)

    # Click: brief sharp transient at 3 kHz (simulates microphone pop)
    click_start, click_end = int(0.5 * sr), int(0.501 * sr)
    click = np.zeros_like(t)
    click[click_start:click_end] = 0.5

    # Plosive: transient at 200 Hz (simulates 'p' or 'b' sound)
    plosive_start, plosive_end = int(0.3 * sr), int(0.304 * sr)
    plosive = np.zeros_like(t)
    plosive[plosive_start:plosive_end] = 0.3 * np.sin(2 * np.pi * 200 * t[plosive_start:plosive_end])

    combined = sibilant + click + plosive
    combined = np.clip(combined, -1, 1).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, combined, sr)
        yield Path(f.name)
        Path(f.name).unlink()


def test_detect_sibilants_finds_peaks(sample_wav, config):
    """De-esser should detect spectral peak around 6 kHz."""
    freqs = detect_sibilants(sample_wav, config)
    assert len(freqs) > 0
    assert any(5500 < f < 6500 for f in freqs), f"Expected peak ~6000 Hz, got {freqs}"


def test_detect_clicks_finds_transients(sample_wav, config):
    """Click detector should find transients in click (~3 kHz) and plosive (~200 Hz) ranges."""
    freqs = detect_clicks_and_plosives(sample_wav, config)
    assert len(freqs) > 0, "Should detect at least one transient"


def test_deeser_filter_generation(config):
    """Build de-esser sidechain compressor filter string."""
    freqs = [6000, 7500]
    filter_str = build_deeser_filter(freqs, config)
    assert filter_str is not None
    assert "sidechaincompress" in filter_str
    assert "highpass" in filter_str  # Sidechain uses highpass at 4 kHz


def test_click_removal_filter_generation(config):
    """Build click removal transient suppressor filter string."""
    freqs = [200, 3000]
    filter_str = build_click_removal_filter(freqs, config)
    assert filter_str is not None
    assert "compand" in filter_str  # Fast-attack compand for transient gating


def test_deeser_disabled_returns_none(config, sample_wav):
    """When deeser disabled, detection should return empty list."""
    config.deeser_enabled = False
    freqs = detect_sibilants(sample_wav, config)
    assert len(freqs) == 0


def test_click_removal_disabled_returns_none(config, sample_wav):
    """When click removal disabled, detection should return empty list."""
    config.click_removal_enabled = False
    freqs = detect_clicks_and_plosives(sample_wav, config)
    assert len(freqs) == 0


def test_empty_filter_when_no_peaks(config):
    """build_deeser_filter returns None when no frequencies detected."""
    assert build_deeser_filter([], config) is None
    assert build_click_removal_filter([], config) is None
