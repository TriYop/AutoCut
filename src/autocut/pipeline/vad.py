from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment, SegmentSource


def detect_silences(wav_path: Path, duration_s: float, config: AutoCutConfig) -> list[BadSegment]:
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    # torchaudio >= 2.9 dropped its legacy audio backend; load WAV via soundfile instead
    data, _ = sf.read(str(wav_path), dtype="float32")
    wav = torch.from_numpy(data)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        min_silence_duration_ms=config.vad_min_silence_duration_ms,
        speech_pad_ms=config.vad_speech_pad_ms,
        return_seconds=True,
    )

    min_silence_s = config.vad_min_silence_duration_ms / 1000.0
    silences: list[BadSegment] = []
    prev_end = 0.0

    max_silence_s = config.vad_max_silence_duration_s

    def _within_bounds(duration: float) -> bool:
        return duration >= min_silence_s and (max_silence_s is None or duration <= max_silence_s)

    for seg in speech_timestamps:
        gap = seg["start"] - prev_end
        if _within_bounds(gap):
            silences.append(BadSegment(
                segment=Segment(prev_end, seg["start"]),
                source=SegmentSource.VAD,
                label="silence",
            ))
        prev_end = seg["end"]

    trailing = duration_s - prev_end
    if _within_bounds(trailing):
        silences.append(BadSegment(
            segment=Segment(prev_end, duration_s),
            source=SegmentSource.VAD,
            label="silence",
        ))

    return silences
