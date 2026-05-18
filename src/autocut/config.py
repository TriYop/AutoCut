from dataclasses import dataclass, field


@dataclass
class AutoCutConfig:
    # VAD
    vad_min_silence_duration_ms: int = 500
    vad_speech_pad_ms: int = 100

    # Whisper
    whisper_model: str = "small"
    whisper_language: str | None = None
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Filler detection
    filler_words: list[str] = field(default_factory=lambda: [
        "euh", "hm", "hmm", "donc", "ben", "beh", "voilà", "eh",
    ])
    min_filler_duration_s: float = 0.3

    # Repetition detection
    detect_repetitions: bool = True
    repetition_window_words: int = 3

    # Segment merging
    merge_gap_s: float = 0.2

    # Output padding
    padding_before_s: float = 0.05
    padding_after_s: float = 0.05
