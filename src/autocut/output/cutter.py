import subprocess
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, MediaInfo, Segment
from autocut.pipeline.room_eq import build_ffmpeg_eq_filter


def _compute_kept_segments(
    bad_segments: list[BadSegment],
    duration_s: float,
) -> list[Segment]:
    kept: list[Segment] = []
    prev_end = 0.0

    for bad in sorted(bad_segments, key=lambda s: s.segment.start):
        if bad.segment.start > prev_end:
            kept.append(Segment(prev_end, bad.segment.start))
        prev_end = max(prev_end, bad.segment.end)

    if prev_end < duration_s:
        kept.append(Segment(prev_end, duration_s))

    return [s for s in kept if s.duration > 0.01]


def cut_video(
    input_path: Path,
    bad_segments: list[BadSegment],
    media_info: MediaInfo,
    output_path: Path,
    config: AutoCutConfig,
    resonant_freqs: list[float] | None = None,
) -> None:
    kept = _compute_kept_segments(bad_segments, media_info.duration_s)

    if not kept:
        raise ValueError("No segments left to keep after cuts")

    needs_audio_processing = config.crossfade_ms > 0 or (
        config.room_eq_enabled and resonant_freqs
    )

    if needs_audio_processing:
        _cut_with_audio_processing(input_path, kept, media_info, output_path, config, resonant_freqs or [])
    else:
        _cut_stream_copy(input_path, kept, output_path)


# ── Stream-copy path (fast, lossless) ────────────────────────────────────────

def _cut_stream_copy(input_path: Path, kept: list[Segment], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_path = Path(f.name)
        for seg in kept:
            f.write(f"file '{input_path}'\n")
            f.write(f"inpoint {seg.start:.6f}\n")
            f.write(f"outpoint {seg.end:.6f}\n")

    try:
        ffmpeg.input(str(concat_path), format="concat", safe=0).output(
            str(output_path), c="copy"
        ).overwrite_output().run(quiet=True)
    except ffmpeg.Error:
        ffmpeg.input(str(concat_path), format="concat", safe=0).output(
            str(output_path), vcodec="libx264", acodec="aac", crf=18
        ).overwrite_output().run(quiet=True)
    finally:
        concat_path.unlink(missing_ok=True)


# ── Audio-processing path (crossfade + room EQ) ───────────────────────────────

def _load_audio_native(input_path: Path) -> tuple[np.ndarray, int]:
    """Extract mono audio from the video at its native sample rate via FFmpeg pipe."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "csv=p=0", str(input_path),
        ],
        capture_output=True, text=True, check=True,
    )
    native_sr = int(probe.stdout.strip())

    proc = subprocess.Popen(
        [
            "ffmpeg", "-i", str(input_path),
            "-f", "f32le", "-ac", "1", "-ar", str(native_sr), "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    raw = proc.stdout.read()
    proc.wait()

    return np.frombuffer(raw, dtype=np.float32).copy(), native_sr


def _apply_crossfade(
    audio: np.ndarray, sr: int, kept: list[Segment], crossfade_ms: int
) -> np.ndarray:
    fade_n = int(crossfade_ms / 1000 * sr)
    chunks = []

    for seg in kept:
        start = int(seg.start * sr)
        end = min(int(seg.end * sr), len(audio))
        chunk = audio[start:end].copy()

        if fade_n > 0 and len(chunk) > 2 * fade_n:
            chunk[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            chunk[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

        chunks.append(chunk)

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def _cut_with_audio_processing(
    input_path: Path,
    kept: list[Segment],
    media_info: MediaInfo,
    output_path: Path,
    config: AutoCutConfig,
    resonant_freqs: list[float],
) -> None:
    audio, native_sr = _load_audio_native(input_path)

    # Apply crossfade at cut boundaries
    processed = _apply_crossfade(audio, native_sr, kept, config.crossfade_ms)

    # Write processed audio to a temp WAV
    audio_tmp = Path(tempfile.mktemp(suffix=".wav"))
    sf.write(str(audio_tmp), processed, native_sr, subtype="FLOAT")

    # Apply room EQ via FFmpeg equalizer filter if requested
    eq_filter = build_ffmpeg_eq_filter(resonant_freqs, config) if (config.room_eq_enabled and resonant_freqs) else None

    if eq_filter:
        eq_tmp = Path(tempfile.mktemp(suffix=".wav"))
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_tmp),
                "-af", eq_filter,
                str(eq_tmp),
            ],
            check=True, capture_output=True,
        )
        audio_tmp.unlink(missing_ok=True)
        audio_tmp = eq_tmp

    # Cut video stream only (stream copy, no audio)
    video_tmp = Path(tempfile.mktemp(suffix=".mp4"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_path = Path(f.name)
        for seg in kept:
            f.write(f"file '{input_path}'\n")
            f.write(f"inpoint {seg.start:.6f}\n")
            f.write(f"outpoint {seg.end:.6f}\n")

    try:
        ffmpeg.input(str(concat_path), format="concat", safe=0).output(
            str(video_tmp), vcodec="copy", an=None
        ).overwrite_output().run(quiet=True)
    finally:
        concat_path.unlink(missing_ok=True)

    # Mux video + processed audio
    try:
        video_in = ffmpeg.input(str(video_tmp))
        audio_in = ffmpeg.input(str(audio_tmp))
        ffmpeg.output(
            video_in.video, audio_in.audio,
            str(output_path),
            vcodec="copy", acodec="aac", audio_bitrate="192k",
            shortest=None,
        ).overwrite_output().run(quiet=True)
    finally:
        video_tmp.unlink(missing_ok=True)
        audio_tmp.unlink(missing_ok=True)
