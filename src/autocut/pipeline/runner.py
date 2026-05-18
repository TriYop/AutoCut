from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, MediaInfo
from autocut.pipeline.audio import extract_audio
from autocut.pipeline.merger import merge_bad_segments
from autocut.pipeline.transcriber import detect_fillers_and_repetitions
from autocut.pipeline.vad import detect_silences


@dataclass
class PipelineResult:
    input_path: Path
    media_info: MediaInfo
    bad_segments: list[BadSegment]


def run(input_path: Path, config: AutoCutConfig, console: Console) -> PipelineResult:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        t1 = progress.add_task("Extracting audio…", total=None)
        audio_path, media_info = extract_audio(input_path, config)
        progress.update(t1, description="[green]Audio extracted", completed=1, total=1)

        t2 = progress.add_task("Voice Activity Detection…", total=None)
        silence_segs = detect_silences(audio_path, media_info.duration_s, config)
        progress.update(t2, description=f"[green]VAD done ({len(silence_segs)} silences)", completed=1, total=1)

        t3 = progress.add_task("Transcribing with Whisper…", total=None)
        filler_segs, repetition_segs = detect_fillers_and_repetitions(audio_path, config)
        progress.update(
            t3,
            description=f"[green]Whisper done ({len(filler_segs)} fillers, {len(repetition_segs)} repetitions)",
            completed=1,
            total=1,
        )

        audio_path.unlink(missing_ok=True)

        all_bad = silence_segs + filler_segs + repetition_segs
        merged = merge_bad_segments(all_bad, config, media_info.duration_s)

    return PipelineResult(
        input_path=input_path,
        media_info=media_info,
        bad_segments=merged,
    )
