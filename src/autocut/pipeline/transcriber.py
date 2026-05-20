"""Whisper-based detection of filler words, truncated words, and word repetitions."""

import unicodedata
from collections import deque
from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment, SegmentSource

# Very common words where repetition is grammatically normal
_STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "en",
    "que", "qui", "se", "si", "ne", "pas", "je", "tu", "il", "elle", "nous",
    "vous", "ils", "elles", "on", "the", "a", "an", "and", "or", "of", "in",
    "is", "it", "to", "that", "this", "i", "we", "you", "he", "she", "they",
}


def _normalize(word: str) -> str:
    """Strip punctuation, lowercase, and NFC-normalise a word token."""
    word = word.strip().lower()
    word = word.rstrip("-.,;:!?\"'")
    return unicodedata.normalize("NFC", word)


def detect_fillers_and_repetitions(
    wav_path: Path,
    config: AutoCutConfig,
) -> tuple[list[BadSegment], list[BadSegment]]:
    """Transcribe audio with Whisper and return (filler_segments, repetition_segments)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(
        config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
    )

    segments, _ = model.transcribe(
        str(wav_path),
        language=config.whisper_language,
        word_timestamps=True,
    )

    words = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)

    filler_set = {w.lower() for w in config.filler_words}
    fillers: list[BadSegment] = []
    repetitions: list[BadSegment] = []

    # Pass 1 — filler words
    for word in words:
        normalized = _normalize(word.word)
        if normalized in filler_set:
            duration = word.end - word.start
            if duration >= config.min_filler_duration_s:
                fillers.append(BadSegment(
                    segment=Segment(word.start, word.end),
                    source=SegmentSource.WHISPER_FILLER,
                    label=normalized,
                    confidence=word.probability,
                ))

    if not config.detect_repetitions:
        return fillers, []

    # Pass 2 — repetitions (sliding window)
    # (normalized_word, word-index-in-words-list)
    window: deque[tuple[str, int]] = deque()

    for i, word in enumerate(words):
        normalized = _normalize(word.word)
        if not normalized or len(normalized) < config.repetition_min_word_length or normalized in _STOP_WORDS:
            window.append((normalized, i))
            if len(window) > config.repetition_window_words:
                window.popleft()
            continue

        # Check for repetition in window
        for prev_norm, prev_idx in window:
            if prev_norm == normalized:
                prev_word = words[prev_idx]
                repetitions.append(BadSegment(
                    segment=Segment(prev_word.start, prev_word.end),
                    source=SegmentSource.WHISPER_REPETITION,
                    label=f"repetition:{normalized}",
                    confidence=word.probability,
                ))
                break

        # Detect partial word restart (word ending with hyphen)
        if word.word.rstrip().endswith("-"):
            fillers.append(BadSegment(
                segment=Segment(word.start, word.end),
                source=SegmentSource.WHISPER_FILLER,
                label="truncated-word",
                confidence=word.probability,
            ))

        window.append((normalized, i))
        if len(window) > config.repetition_window_words:
            window.popleft()

    return fillers, repetitions
