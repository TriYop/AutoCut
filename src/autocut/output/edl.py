"""EDL and JSON export of detected cut regions."""

import json
from pathlib import Path

from autocut.models import BadSegment, MediaInfo


def _seconds_to_timecode(seconds: float, fps: float) -> str:
    """Convert fractional seconds to a CMX 3600 non-drop-frame timecode string."""
    total_frames = round(seconds * fps)
    frames = total_frames % round(fps)
    total_seconds = total_frames // round(fps)
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def write_edl(
    bad_segments: list[BadSegment],
    media_info: MediaInfo,
    source_filename: str,
    output_path: Path,
) -> None:
    """Write a CMX 3600 EDL marking each bad segment for deletion."""
    fps = media_info.fps
    lines = [
        f"TITLE: AutoCut - {source_filename}",
        "FCM: NON-DROP FRAME",
        "",
    ]

    for i, bad in enumerate(bad_segments, start=1):
        tc_in = _seconds_to_timecode(bad.segment.start, fps)
        tc_out = _seconds_to_timecode(bad.segment.end, fps)
        lines += [
            f"{i:03d}  AX       V     C        {tc_in} {tc_out} {tc_in} {tc_out}",
            f"* FROM CLIP NAME: {source_filename}",
            f"* COMMENT: {bad.label} [{bad.source.value}]",
            "",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    bad_segments: list[BadSegment],
    source_filename: str,
    output_path: Path,
) -> None:
    """Write a JSON cut list with timestamps, labels, sources, and confidence scores."""
    data = {
        "source": source_filename,
        "cuts": [
            {
                "start": seg.segment.start,
                "end": seg.segment.end,
                "duration": seg.segment.duration,
                "label": seg.label,
                "source": seg.source.value,
                "confidence": seg.confidence,
            }
            for seg in bad_segments
        ],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
