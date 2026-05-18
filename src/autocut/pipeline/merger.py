from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment


def merge_bad_segments(
    segments: list[BadSegment],
    config: AutoCutConfig,
    total_duration_s: float,
) -> list[BadSegment]:
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.segment.start)

    merged: list[BadSegment] = [sorted_segs[0]]
    for current in sorted_segs[1:]:
        last = merged[-1]
        gap = current.segment.start - last.segment.end
        if gap <= config.merge_gap_s:
            new_end = max(last.segment.end, current.segment.end)
            label = f"{last.label}+{current.label}" if last.label != current.label else last.label
            merged[-1] = BadSegment(
                segment=Segment(last.segment.start, new_end),
                source=last.source,
                label=label,
                confidence=min(last.confidence, current.confidence),
            )
        else:
            merged.append(current)

    result: list[BadSegment] = []
    for seg in merged:
        padded_start = max(0.0, seg.segment.start - config.padding_before_s)
        padded_end = min(total_duration_s, seg.segment.end + config.padding_after_s)
        result.append(BadSegment(
            segment=Segment(padded_start, padded_end),
            source=seg.source,
            label=seg.label,
            confidence=seg.confidence,
        ))

    return result
