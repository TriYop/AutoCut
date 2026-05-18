import tempfile
from pathlib import Path

import ffmpeg

from autocut.models import BadSegment, MediaInfo, Segment


def _compute_kept_segments(
    bad_segments: list[BadSegment],
    duration_s: float,
) -> list[Segment]:
    kept: list[Segment] = []
    prev_end = 0.0

    for bad in sorted(bad_segments, key=lambda s: s.segment.start):
        if bad.segment.start > prev_end:
            kept.append(Segment(prev_end, bad.segment.start))
        prev_end = max(prev_end, bad.segment.end)

    if prev_end < duration_s:
        kept.append(Segment(prev_end, duration_s))

    return [s for s in kept if s.duration > 0.01]


def cut_video(
    input_path: Path,
    bad_segments: list[BadSegment],
    media_info: MediaInfo,
    output_path: Path,
) -> None:
    kept = _compute_kept_segments(bad_segments, media_info.duration_s)

    if not kept:
        raise ValueError("No segments left to keep after cuts")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_path = Path(f.name)
        for seg in kept:
            f.write(f"file '{input_path}'\n")
            f.write(f"inpoint {seg.start:.6f}\n")
            f.write(f"outpoint {seg.end:.6f}\n")

    try:
        _run_concat(concat_path, output_path, reencode=False)
    except ffmpeg.Error:
        _run_concat(concat_path, output_path, reencode=True)
    finally:
        concat_path.unlink(missing_ok=True)


def _run_concat(concat_path: Path, output_path: Path, reencode: bool) -> None:
    input_stream = ffmpeg.input(str(concat_path), format="concat", safe=0)

    if reencode:
        out = input_stream.output(
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            crf=18,
        )
    else:
        out = input_stream.output(str(output_path), c="copy")

    out.overwrite_output().run(quiet=True)
