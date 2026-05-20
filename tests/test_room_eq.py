"""Tests for autocut.pipeline.room_eq — build_ffmpeg_eq_filter (no audio I/O)."""

import pytest

from autocut.config import AutoCutConfig
from autocut.pipeline.room_eq import build_ffmpeg_eq_filter


CFG = AutoCutConfig()


# ── build_ffmpeg_eq_filter ────────────────────────────────────────────────────

def test_build_eq_filter_empty_returns_none():
    assert build_ffmpeg_eq_filter([], CFG) is None


def test_build_eq_filter_single_freq_is_string():
    result = build_ffmpeg_eq_filter([200.0], CFG)
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_eq_filter_single_freq_contains_frequency():
    result = build_ffmpeg_eq_filter([200.0], CFG)
    assert "f=200.0" in result


def test_build_eq_filter_single_freq_contains_gain():
    # default gain_db = -10.0
    result = build_ffmpeg_eq_filter([200.0], CFG)
    assert f"g={CFG.room_eq_gain_db:.1f}" in result


def test_build_eq_filter_single_freq_bandwidth_formula():
    # bw = freq / q_factor = 400 / 8.0 = 50.0
    cfg = AutoCutConfig(room_eq_q_factor=8.0)
    result = build_ffmpeg_eq_filter([400.0], cfg)
    assert "width=50.0" in result


def test_build_eq_filter_bandwidth_minimum_10():
    # freq=5 Hz → freq/q = 5/8 = 0.625 < 10 → clamped to 10.0
    result = build_ffmpeg_eq_filter([5.0], CFG)
    assert "width=10.0" in result


def test_build_eq_filter_bandwidth_at_boundary():
    # freq=80 Hz, q=8 → bw=10.0 exactly (boundary)
    result = build_ffmpeg_eq_filter([80.0], CFG)
    assert "width=10.0" in result


def test_build_eq_filter_multiple_freqs_comma_joined():
    result = build_ffmpeg_eq_filter([200.0, 500.0], CFG)
    parts = result.split(",")
    # Each filter section starts with "equalizer="
    assert all("equalizer=" in p for p in parts)
    assert len(parts) == 2


def test_build_eq_filter_multiple_freqs_all_present():
    freqs = [200.0, 500.0, 1000.0]
    result = build_ffmpeg_eq_filter(freqs, CFG)
    assert "f=200.0" in result
    assert "f=500.0" in result
    assert "f=1000.0" in result


def test_build_eq_filter_custom_gain():
    cfg = AutoCutConfig(room_eq_gain_db=-6.0)
    result = build_ffmpeg_eq_filter([300.0], cfg)
    assert "g=-6.0" in result


def test_build_eq_filter_custom_q_factor():
    # q=4.0, freq=400 → bw=100.0
    cfg = AutoCutConfig(room_eq_q_factor=4.0)
    result = build_ffmpeg_eq_filter([400.0], cfg)
    assert "width=100.0" in result


def test_build_eq_filter_uses_width_type_h():
    # Width type must be 'h' (Hz bandwidth) for the notch to be correct
    result = build_ffmpeg_eq_filter([200.0], CFG)
    assert "width_type=h" in result
