"""Video cutting: stream-copy fast path and audio-processing path (crossfade + room EQ)."""

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
    """Return the complement of bad_segments within [0, duration_s]."""
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
    """Write a new video with bad_segments removed, choosing the fastest safe path."""
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

    result = subprocess.run(
        [
            "ffmpeg", "-i", str(input_path),
            "-f", "f32le", "-ac", "1", "-ar", str(native_sr), "-",
        ],
        capture_output=True,
        check=False,  # ffmpeg exits non-zero when writing to stdout pipe; stdout is valid
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy(), native_sr


def _apply_crossfade(
    audio: np.ndarray, sr: int, kept: list[Segment], crossfade_ms: int
) -> np.ndarray:
    fade_n = int(crossfade_ms / 1000 * sr)
    chunks = []
    n = len(kept)

    for idx, seg in enumerate(kept):
        start = int(seg.start * sr)
        end = min(int(seg.end * sr), len(audio))
        chunk = audio[start:end].copy()

        if fade_n > 0 and len(chunk) > 2 * fade_n:
            # Fade-in only after a cut (not at the natural start of the video)
            if idx > 0:
                chunk[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            # Fade-out only before a cut (not at the natural end of the video)
            if idx < n - 1:
                chunk[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

        chunks.append(chunk)

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def _segment_filter_chain(i: int, seg: Segment, n: int, crossfade_s: float) -> str:
    """Return the trim+setpts+optional-fade filter chain string for one segment."""
    chain = f"[0:v]trim=start={seg.start:.6f}:end={seg.end:.6f},setpts=PTS-STARTPTS"
    if crossfade_s > 0 and seg.duration > 2 * crossfade_s:
        if i > 0:
            chain += f",fade=t=in:st=0:d={crossfade_s:.6f}"
        if i < n - 1:
            chain += f",fade=t=out:st={seg.duration - crossfade_s:.6f}:d={crossfade_s:.6f}"
    return f"{chain}[v{i}]"


def _boundary_timestamps(kept: list[Segment]) -> list[str]:
    """Return cumulative output-timeline timestamps at each segment join."""
    t = 0.0
    result: list[str] = []
    for seg in kept[:-1]:
        t += seg.duration
        result.append(f"{t:.3f}")
    return result


def _cut_video_reencode(
    input_path: Path,
    kept: list[Segment],
    output_path: Path,
    crossfade_ms: int,
) -> None:
    """Re-encode kept video segments with optional fade-through-black at cut boundaries.

    Uses trim+setpts per segment so every cut is frame-accurate regardless of
    keyframe placement.  When crossfade_ms > 0 a fade-out is appended to each
    non-last segment and a fade-in is prepended to each non-first segment,
    mirroring the audio crossfade without changing total duration.
    """
    n = len(kept)
    if n == 0:
        return

    crossfade_s = crossfade_ms / 1000.0
    if crossfade_s > 0 and n > 1:
        crossfade_s = min(crossfade_s, min(seg.duration for seg in kept) / 2.0)

    filter_parts = [_segment_filter_chain(i, seg, n, crossfade_s) for i, seg in enumerate(kept)]

    if n > 1:
        concat_inputs = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vout]")
        final_label = "vout"
    else:
        final_label = "v0"

    # direct=auto (libx264 default) uses temporal B-frame prediction, which
    # looks up co-located motion vectors from frames that no longer exist
    # after a cut → "co located POCs unavailable" decoder warnings in VLC.
    # Spatial direct mode avoids co-location lookups; no-open-gop keeps each
    # GOP self-contained so decoders can seek to any boundary cleanly.
    # IDR keyframes at segment joins guarantee a clean decoder reset per cut.
    boundaries = _boundary_timestamps(kept) if n > 1 else []
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter_complex", ";".join(filter_parts),
            "-map", f"[{final_label}]",
            "-c:v", "libx264", "-crf", "18",
            "-x264-params", "direct=spatial:no-open-gop=1",
            "-an",
            *((["-force_key_frames", ",".join(boundaries)]) if boundaries else []),
            str(output_path),
        ],
        check=True, capture_output=True,
    )


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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_tmp = Path(f.name)
    sf.write(str(audio_tmp), processed, native_sr, subtype="FLOAT")

    # Apply room EQ via FFmpeg equalizer filter if requested
    eq_filter = build_ffmpeg_eq_filter(resonant_freqs, config) if (config.room_eq_enabled and resonant_freqs) else None

    if eq_filter:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            eq_tmp = Path(f.name)
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

    # Re-encode video segments so cuts are frame-accurate (stream copy would
    # snap to keyframes, causing drift against the exact-timestamp audio).
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_tmp = Path(f.name)
    _cut_video_reencode(input_path, kept, video_tmp, config.crossfade_ms)

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
