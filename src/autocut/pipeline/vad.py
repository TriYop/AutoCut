from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment, SegmentSource


def detect_silences(wav_path: Path, duration_s: float, config: AutoCutConfig) -> list[BadSegment]:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    model = load_silero_vad()
    wav = read_audio(str(wav_path), sampling_rate=16000)

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

    for seg in speech_timestamps:
        gap = seg["start"] - prev_end
        if gap >= min_silence_s:
            silences.append(BadSegment(
                segment=Segment(prev_end, seg["start"]),
                source=SegmentSource.VAD,
                label="silence",
            ))
        prev_end = seg["end"]

    trailing = duration_s - prev_end
    if trailing >= min_silence_s:
        silences.append(BadSegment(
            segment=Segment(prev_end, duration_s),
            source=SegmentSource.VAD,
            label="silence",
        ))

    return silences
