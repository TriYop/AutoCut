"""Central configuration dataclass for all AutoCut pipeline stages."""

from dataclasses import dataclass, field


@dataclass
class AutoCutConfig:
    """All tunable parameters for detection, merging, and output post-processing."""
    # VAD
    vad_min_silence_duration_ms: int = 700
    vad_speech_pad_ms: int = 150

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
    repetition_min_word_length: int = 2  # ignore single-letter noise from transcription

    # VAD silence cap: silences longer than this are kept (Q&A breaks, applause, etc.)
    # None = no cap, cut everything (good for YT replays)
    vad_max_silence_duration_s: float | None = 30.0

    # Segment merging
    merge_gap_s: float = 0.2

    # Output padding
    padding_before_s: float = 0.05
    padding_after_s: float = 0.05

    # Room EQ
    room_eq_enabled: bool = False
    room_eq_threshold_db: float = 10.0   # dB above noise floor to flag a resonance
    room_eq_max_filters: int = 5         # max number of notch filters to apply
    room_eq_q_factor: float = 8.0        # higher = narrower notch
    room_eq_gain_db: float = -10.0       # attenuation in dB (negative = cut)

    # Crossfade at cut points
    crossfade_ms: int = 0                # 0 = disabled; ~120ms recommended

    # Pipeline cache
    use_cache: bool = True
