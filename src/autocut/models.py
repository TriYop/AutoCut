"""Domain models shared across the AutoCut pipeline."""

from dataclasses import dataclass
from enum import Enum


class SegmentSource(Enum):
    """Identifies which detector produced a BadSegment."""

    VAD = "vad"
    WHISPER_FILLER = "whisper_filler"
    WHISPER_REPETITION = "whisper_repetition"


@dataclass(frozen=True, order=True)
class Segment:
    """A half-open time interval [start, end) in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class BadSegment:
    """A detected region that should be cut, together with its origin and label."""

    segment: Segment
    source: SegmentSource
    label: str
    confidence: float = 1.0


@dataclass(frozen=True)
class MediaInfo:
    """Basic media properties extracted by FFprobe."""

    duration_s: float
    fps: float
    has_audio: bool
