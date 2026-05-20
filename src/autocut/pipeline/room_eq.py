"""Room resonance analysis: FFT over VAD silences → notch-filter frequencies."""

from pathlib import Path

import numpy as np
import soundfile as sf

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, SegmentSource


def analyze_room_resonances(
    wav_path: Path,
    silence_segments: list[BadSegment],
    config: AutoCutConfig,
) -> list[float]:
    """
    Identify room resonance frequencies from VAD silence segments via FFT.
    Returns a list of frequencies (Hz) to attenuate.
    """
    data, sr = sf.read(str(wav_path), dtype="float32")

    # Only use VAD silences long enough (>150ms) to be reliable room noise samples
    chunks = []
    for seg in silence_segments:
        if seg.source is not SegmentSource.VAD:
            continue
        start = int(seg.segment.start * sr)
        end = int(seg.segment.end * sr)
        if (end - start) >= int(0.15 * sr):
            chunks.append(data[start:end])

    if not chunks:
        return []

    combined = np.concatenate(chunks)

    # Averaged magnitude spectrum (Welch-style: average over non-overlapping windows)
    n_fft = 8192
    hop = n_fft // 2
    frames = [
        combined[i : i + n_fft]
        for i in range(0, len(combined) - n_fft, hop)
    ]
    if not frames:
        return []

    spectra = np.array([np.abs(np.fft.rfft(f * np.hanning(n_fft))) for f in frames])
    avg_spectrum = spectra.mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

    # Restrict to speech-relevant band: 80 Hz – 4 kHz
    mask = (freqs >= 80) & (freqs <= 4000)
    freqs_r = freqs[mask]
    spec_r = avg_spectrum[mask]

    spec_db = 20 * np.log10(spec_r + 1e-10)
    noise_floor = np.percentile(spec_db, 25)
    threshold = noise_floor + config.room_eq_threshold_db

    # Simple local-maximum peak finder
    # ~40 Hz minimum gap between peaks in bin units
    min_gap = max(1, int(40 / (sr / n_fft)))
    peaks: list[tuple[int, float]] = []
    for i in range(1, len(spec_db) - 1):
        if spec_db[i] < threshold:
            continue
        if spec_db[i] <= spec_db[i - 1] or spec_db[i] <= spec_db[i + 1]:
            continue
        if peaks and (i - peaks[-1][0]) < min_gap:
            # Keep the taller of the two
            if spec_db[i] > peaks[-1][1]:
                peaks[-1] = (i, spec_db[i])
        else:
            peaks.append((i, spec_db[i]))

    # Return top-N by prominence, as floats
    peaks.sort(key=lambda p: p[1], reverse=True)
    return [float(freqs_r[i]) for i, _ in peaks[: config.room_eq_max_filters]]


def build_ffmpeg_eq_filter(resonant_freqs: list[float], config: AutoCutConfig) -> str | None:
    """Build a chained FFmpeg `equalizer` filter string for the given frequencies."""
    if not resonant_freqs:
        return None
    parts = []
    for freq in resonant_freqs:
        bw = max(10.0, freq / config.room_eq_q_factor)
        parts.append(
            f"equalizer=f={freq:.1f}:width_type=h:width={bw:.1f}:g={config.room_eq_gain_db:.1f}"
        )
    return ",".join(parts)
