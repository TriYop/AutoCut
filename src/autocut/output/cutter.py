"""Video cutting: stream-copy fast path and audio-processing path (crossfade + room EQ)."""

import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, MediaInfo, Segment
from autocut.pipeline.audio_enhancement import (
    build_click_removal_filter,
    build_deeser_filter,
)
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
    progress_cb: Callable[[float, float], None] | None = None,
    sibilant_freqs: list[float] | None = None,
    click_freqs: list[float] | None = None,
) -> None:
    """Write a new video with bad_segments removed, choosing the fastest safe path.

    progress_cb(current_s, total_s) is called during re-encode when the audio-processing
    path is active; it is never called on the stream-copy path (near-instant).
    """
    if sibilant_freqs is None:
        sibilant_freqs = []
    if click_freqs is None:
        click_freqs = []
    kept = _compute_kept_segments(bad_segments, media_info.duration_s)

    if not kept:
        raise ValueError("No segments left to keep after cuts")

    needs_audio_processing = config.crossfade_ms > 0 or (
        config.room_eq_enabled and resonant_freqs
    ) or sibilant_freqs or click_freqs

    if needs_audio_processing:
        _cut_with_audio_processing(
            input_path, kept, media_info, output_path, config,
            resonant_freqs or [], progress_cb,
            sibilant_freqs, click_freqs,
        )
    else:
        _cut_stream_copy(input_path, kept, output_path)


# ── Stream-copy path (fast, lossless) ────────────────────────────────────────

def _cut_stream_copy(input_path: Path, kept: list[Segment], output_path: Path) -> None:
    """Concatenate kept segments via stream-copy (no re-encode) using FFmpeg concat demuxer.

    Falls back to libx264/aac re-encode if stream-copy fails (e.g. mixed codec sources).
    """
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
        # ffmpeg exits non-zero when writing to a stdout pipe; the audio data in stdout is valid
        check=False,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy(), native_sr


def _effective_crossfade_s(kept: list[Segment], crossfade_ms: int, fps: float) -> float:
    """Return crossfade duration in seconds, clamped so the audio+video timing stays safe.

    trim rounds to frame boundaries causing up to 2/fps (start + end) of duration error per
    segment.  The video offset uses 2/fps slack per transition; audio must match by overlapping
    crossfade_s + 2/fps.  To prevent a middle segment from being consumed by two consecutive
    audio overlaps, we need 2*(crossfade_s + 2/fps) ≤ min_dur, i.e. crossfade_s ≤ min_dur/2 - 2/fps.
    """
    if crossfade_ms <= 0 or len(kept) <= 1:
        return 0.0
    frame_s = 2.0 / fps
    min_dur = min(seg.duration for seg in kept)
    return max(0.0, min(crossfade_ms / 1000.0, min_dur / 2.0 - frame_s))


def _apply_crossfade(
    audio: np.ndarray, sr: int, kept: list[Segment], crossfade_s: float
) -> np.ndarray:
    """Blend consecutive segments with a true overlap crossfade.

    The output is shorter than a plain concatenation by (n-1)*crossfade_s seconds,
    matching the duration reduction produced by FFmpeg's xfade filter on the video.
    """
    chunks = [
        audio[int(seg.start * sr) : min(int(seg.end * sr), len(audio))].copy()
        for seg in kept
    ]
    if crossfade_s <= 0 or len(chunks) <= 1:
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    fade_n = int(crossfade_s * sr)
    # crossfade_s > 0 but rounds to 0 samples at low sample rates — treat as no crossfade
    if fade_n == 0:
        return np.concatenate(chunks)

    fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)

    result = chunks[0]
    for next_chunk in chunks[1:]:
        fn = min(fade_n, len(result), len(next_chunk))
        result = result.copy()
        result[-fn:] *= fade_out[-fn:]
        result[-fn:] += next_chunk[:fn] * fade_in[:fn]
        result = np.concatenate([result, next_chunk[fn:]])
    return result


def _segment_filter_chain(i: int, seg: Segment, fps: float) -> str:
    """Return trim+setpts+fps filter chain for one segment.

    The fps filter normalises each segment to a known constant frame rate.
    Without it, deeply chained xfade filters lose frame-rate tracking and
    FFmpeg rejects the graph with "current rate of 1/0 is invalid".
    """
    fps_int = round(fps)
    return (
        f"[0:v]trim=start={seg.start:.6f}:end={seg.end:.6f},"
        f"setpts=PTS-STARTPTS,fps={fps_int}[v{i}]"
    )


def _xfade_filter_parts(kept: list[Segment], crossfade_s: float, fps: float) -> tuple[list[str], str]:
    """Build the chained xfade dissolve filters and return (parts, final_output_label).

    Each xfade overlaps consecutive segments by crossfade_s seconds.  The offset for
    transition k is sum(dur[0..k-1]) - k*(crossfade_s + 2/fps).  The 2/fps slack per
    transition absorbs frame-boundary rounding by FFmpeg's trim filter: trim can drop up
    to one frame at the start AND one frame at the end of a segment (2/fps total), so
    1/fps slack leaves zero margin when both boundaries round against us.
    """
    n = len(kept)
    parts: list[str] = []
    t = 0.0
    frame_s = 2.0 / fps
    for k in range(1, n):
        t += kept[k - 1].duration
        offset = max(0.0, t - k * (crossfade_s + frame_s))
        input_a = "v0" if k == 1 else f"xf{k - 1}"
        parts.append(
            f"[{input_a}][v{k}]xfade=transition=fade"
            f":duration={crossfade_s:.6f}:offset={offset:.6f}[xf{k}]"
        )
    return parts, f"xf{n - 1}"


def _boundary_timestamps(kept: list[Segment], crossfade_s: float, fps: float) -> list[str]:
    """Return output-timeline timestamps where each xfade transition begins."""
    t = 0.0
    result: list[str] = []
    frame_s = 2.0 / fps
    for k, seg in enumerate(kept[:-1]):
        t += seg.duration
        result.append(f"{max(0.0, t - (k + 1) * (crossfade_s + frame_s)):.3f}")
    return result


def _cut_video_reencode(
    input_path: Path,
    kept: list[Segment],
    output_path: Path,
    crossfade_s: float,
    fps: float,
    progress_cb: Callable[[float, float], None] | None = None,
) -> None:
    """Re-encode kept segments with optional xfade dissolve at cut boundaries.

    Uses xfade when crossfade_s > 0 (total duration shrinks by (n-1)*crossfade_s),
    plain concat otherwise.  B-frames are disabled (-bf 0) to prevent hardware-decoder
    co-located POC warnings: temporal direct prediction + b-pyramid create reference
    chains that VA-API decoders cannot resolve; quality impact at CRF 18 is imperceptible.

    When progress_cb is provided, FFmpeg's -progress pipe is used to stream
    out_time_us updates; stderr is drained on a background thread to prevent deadlock.
    """
    n = len(kept)
    if n == 0:
        return

    filter_parts = [_segment_filter_chain(i, seg, fps) for i, seg in enumerate(kept)]

    if n > 1 and crossfade_s > 0:
        xfade_parts, final_label = _xfade_filter_parts(kept, crossfade_s, fps)
        filter_parts.extend(xfade_parts)
    elif n > 1:
        concat_inputs = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vout]")
        final_label = "vout"
    else:
        final_label = "v0"

    total_s = sum(seg.duration for seg in kept)
    if crossfade_s > 0 and n > 1:
        total_s -= (n - 1) * crossfade_s

    boundaries = _boundary_timestamps(kept, crossfade_s, fps) if n > 1 else []
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-filter_complex", ";".join(filter_parts),
        "-map", f"[{final_label}]",
        "-c:v", "libx264", "-crf", "18",
        "-bf", "0",
        "-x264-params", "no-open-gop=1",
        "-an",
        *((["-force_key_frames", ",".join(boundaries)]) if boundaries else []),
        "-progress", "pipe:1",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        """Read all FFmpeg stderr so the pipe buffer never blocks the main thread."""
        if proc.stderr:
            stderr_lines.extend(proc.stderr.read().splitlines())

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    if proc.stdout:
        for line in proc.stdout:
            if progress_cb and line.startswith("out_time_us="):
                try:
                    current_s = int(line.split("=", 1)[1]) / 1_000_000
                    progress_cb(min(current_s, total_s), total_s)
                except ValueError:
                    pass

    proc.wait()
    stderr_thread.join()

    if proc.returncode != 0:
        stderr = "\n".join(stderr_lines)
        raise RuntimeError(
            f"FFmpeg re-encode failed (exit {proc.returncode}):\n{stderr}"
        )


def _cut_with_audio_processing(
    input_path: Path,
    kept: list[Segment],
    media_info: MediaInfo,
    output_path: Path,
    config: AutoCutConfig,
    resonant_freqs: list[float],
    progress_cb: Callable[[float, float], None] | None = None,
    sibilant_freqs: list[float] | None = None,
    click_freqs: list[float] | None = None,
) -> None:
    """Re-encode kept segments with crossfade dissolve and optional room EQ.

    crossfade_s is the visual fade duration passed to xfade.  The video offset per
    transition includes 2/fps slack for frame-boundary rounding, so each transition
    shortens the video by (crossfade_s + 2/fps).  The audio overlap is set to the same
    value so both streams shrink identically and A/V transitions remain in sync.
    """
    if sibilant_freqs is None:
        sibilant_freqs = []
    if click_freqs is None:
        click_freqs = []
    crossfade_s = _effective_crossfade_s(kept, config.crossfade_ms, media_info.fps)
    audio_overlap_s = crossfade_s + (2.0 / media_info.fps if crossfade_s > 0 else 0.0)

    audio, native_sr = _load_audio_native(input_path)
    processed = _apply_crossfade(audio, native_sr, kept, audio_overlap_s)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_tmp = Path(f.name)
    sf.write(str(audio_tmp), processed, native_sr, subtype="FLOAT")

    # Build audio filter chain with all enhancements
    # Note: de-esser uses split graph (sidechain), so it must come first
    audio_chain = ""

    # De-esser with sidechain (uses split, so must be first in chain)
    deeser_filter = build_deeser_filter(sibilant_freqs, config)

    # Click removal (compand doesn't split, can be in sequence)
    click_filter = build_click_removal_filter(click_freqs, config)

    # Build the filter graph
    if deeser_filter:
        # Sidechain de-esser outputs [compressed], chain click removal to that label
        audio_chain = deeser_filter

        # Chain click removal after de-esser if both present
        if click_filter:
            audio_chain += f";[compressed]{click_filter}"
    elif click_filter:
        # Only click removal
        audio_chain = click_filter

    # Add room EQ (if not already in chain)
    room_eq_filter = build_ffmpeg_eq_filter(resonant_freqs, config)
    if room_eq_filter:
        if audio_chain:
            audio_chain += f",{room_eq_filter}"
        else:
            audio_chain = room_eq_filter

    # Note: crossfade is already applied via numpy at line 338 (_apply_crossfade).
    # Do NOT add acrossfade FFmpeg filter here — it would apply crossfade twice.

    # Ensure formatting is always applied
    if audio_chain:
        audio_filter_chain = f"aformat=sample_rates=44100;{audio_chain}"
    else:
        audio_filter_chain = "aformat=sample_rates=44100"

    if audio_filter_chain and audio_filter_chain != "aformat=sample_rates=44100":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filtered_tmp = Path(f.name)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_tmp),
                "-af", audio_filter_chain,
                str(filtered_tmp),
            ],
            check=True, capture_output=True,
        )
        audio_tmp.unlink(missing_ok=True)
        audio_tmp = filtered_tmp

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_tmp = Path(f.name)
    _cut_video_reencode(input_path, kept, video_tmp, crossfade_s, media_info.fps, progress_cb)

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
