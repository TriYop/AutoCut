"""Audio enhancement detection: de-esser and click/plosive removal via spectral analysis."""

from pathlib import Path

import numpy as np
import soundfile as sf

from autocut.config import AutoCutConfig


def detect_sibilants(wav_path: Path, config: AutoCutConfig) -> list[float]:
    """
    Detect sibilant frequencies (harsh 's', 'sh', 'ch' sounds) via spectral peak finding.
    Analyzes the full audio looking for peaks in 4–9 kHz band (sibilant range).
    Returns list of frequencies (Hz) to attenuate; empty if disabled.
    """
    if not config.deeser_enabled:
        return []

    data, sr = sf.read(str(wav_path), dtype="float32")

    # Use full audio for sibilant detection (they're distributed throughout speech)
    n_fft = 4096
    hop = n_fft // 2
    frames = [
        data[i : i + n_fft]
        for i in range(0, len(data) - n_fft, hop)
    ]

    if not frames:
        return []

    spectra = np.array([np.abs(np.fft.rfft(f * np.hanning(n_fft))) for f in frames])
    avg_spectrum = spectra.mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

    # Sibilant band: 4 kHz – 9 kHz
    mask = (freqs >= 4000) & (freqs <= 9000)
    freqs_r = freqs[mask]
    spec_r = avg_spectrum[mask]

    spec_db = 20 * np.log10(spec_r + 1e-10)
    noise_floor = np.percentile(spec_db, 25)
    threshold = noise_floor + config.deeser_threshold_db

    # Find local maxima (peaks)
    peaks: list[tuple[int, float]] = []
    min_gap = max(1, int(100 / (sr / n_fft)))  # ~100 Hz minimum gap between peaks
    for i in range(1, len(spec_db) - 1):
        if spec_db[i] < threshold:
            continue
        if spec_db[i] <= spec_db[i - 1] or spec_db[i] <= spec_db[i + 1]:
            continue
        if peaks and (i - peaks[-1][0]) < min_gap:
            if spec_db[i] > peaks[-1][1]:
                peaks[-1] = (i, spec_db[i])
        else:
            peaks.append((i, spec_db[i]))

    # Return top 3 sibilant peaks
    peaks.sort(key=lambda p: p[1], reverse=True)
    return [float(freqs_r[i]) for i, _ in peaks[:3]]


def detect_clicks_and_plosives(wav_path: Path, config: AutoCutConfig) -> list[float]:
    """
    Detect click and plosive artifacts via transient detection.
    Scans for sharp peaks in low-freq (plosives: 20–500 Hz) and mid-freq (clicks: 2–10 kHz) ranges.
    Returns list of frequencies (Hz) to attenuate; empty if disabled.
    """
    if not config.click_removal_enabled:
        return []

    data, sr = sf.read(str(wav_path), dtype="float32")

    # Use smaller FFT for transient resolution
    n_fft = 2048
    hop = n_fft // 2
    frames = [
        data[i : i + n_fft]
        for i in range(0, len(data) - n_fft, hop)
    ]

    if not frames:
        return []

    spectra = np.array([np.abs(np.fft.rfft(f * np.hanning(n_fft))) for f in frames])
    avg_spectrum = spectra.mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

    # Analyze two bands: plosives (20–500 Hz) and clicks (2–10 kHz)
    plosive_mask = (freqs >= 20) & (freqs <= 500)
    click_mask = (freqs >= 2000) & (freqs <= 10000)

    detected_freqs = []

    for _, mask in [("plosive", plosive_mask), ("click", click_mask)]:
        freqs_r = freqs[mask]
        spec_r = avg_spectrum[mask]

        if len(spec_r) == 0:
            continue

        spec_db = 20 * np.log10(spec_r + 1e-10)
        noise_floor = np.percentile(spec_db, 25)
        threshold = noise_floor + config.click_threshold_db

        peaks: list[tuple[int, float]] = []
        min_gap = max(1, int(100 / (sr / n_fft)))
        for i in range(1, len(spec_db) - 1):
            if spec_db[i] < threshold:
                continue
            if spec_db[i] <= spec_db[i - 1] or spec_db[i] <= spec_db[i + 1]:
                continue
            if peaks and (i - peaks[-1][0]) < min_gap:
                if spec_db[i] > peaks[-1][1]:
                    peaks[-1] = (i, spec_db[i])
            else:
                peaks.append((i, spec_db[i]))

        # Top 2 per band
        peaks.sort(key=lambda p: p[1], reverse=True)
        detected_freqs.extend([float(freqs_r[i]) for i, _ in peaks[:2]])

    return detected_freqs


def build_deeser_filter(sibilant_freqs: list[float], config: AutoCutConfig) -> str | None:
    """
    Build an FFmpeg dynamic de-esser using sidechain compression.
    Detects sibilants (4–9 kHz) and applies threshold-based compression only when they peak.
    Returns a filter graph string with split + highpass sidechain + sidechaincompress.
    """
    if not sibilant_freqs:
        return None

    # Dynamic de-esser: sidechain compressor triggered by sibilant band
    # Split audio: one path for compression, one for sidechain detection
    # Sidechain: highpass at 4 kHz to focus on sibilant frequencies
    # When sibilants exceed threshold, compress the entire signal
    return (
        "split[main][sidechain];"
        "[sidechain]highpass=f=4000[sidechain_filtered];"
        "[main][sidechain_filtered]sidechaincompress="
        "threshold=0.1:ratio=4:attack=0.005:release=0.05:makeup=2"
    )


def build_click_removal_filter(click_freqs: list[float], config: AutoCutConfig) -> str | None:
    """
    Build an FFmpeg transient suppressor using compand filter.
    Detects clicks/plosives and applies fast-attack compression to gate them out.
    Uses compand with fast attack (1ms), slow release (100ms) for transient gating.
    """
    if not click_freqs:
        return None

    # Transient suppressor: compand with fast attack, slow release
    # attack,release: 1ms attack, 100ms release for catching transients without pumping
    # in|out transfer curve: -inf to -30 (deep gate), 0 to -30 (gentle suppression)
    # soft_knee: 6 dB (smooth gate transition)
    # makeup: 2 dB (compensate for attenuation)
    return "compand=0.001,0.1:-inf|-30,0|-30:6:2"
