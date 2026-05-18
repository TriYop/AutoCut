import json
from pathlib import Path

from autocut.models import BadSegment, MediaInfo


def _seconds_to_timecode(seconds: float, fps: float) -> str:
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
