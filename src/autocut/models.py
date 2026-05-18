from dataclasses import dataclass
from enum import Enum


class SegmentSource(Enum):
    VAD = "vad"
    WHISPER_FILLER = "whisper_filler"
    WHISPER_REPETITION = "whisper_repetition"


@dataclass(frozen=True, order=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class BadSegment:
    segment: Segment
    source: SegmentSource
    label: str
    confidence: float = 1.0


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    fps: float
    has_audio: bool
