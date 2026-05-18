"""Top-level pipeline orchestrator: audio → VAD → Whisper → merge → result."""

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, MediaInfo
from autocut.pipeline.audio import extract_audio
from autocut.pipeline.merger import merge_bad_segments
from autocut.pipeline.room_eq import analyze_room_resonances
from autocut.pipeline.transcriber import detect_fillers_and_repetitions
from autocut.pipeline.vad import detect_silences


@dataclass
class PipelineResult:
    """Aggregated output from the full detection pipeline."""

    input_path: Path
    media_info: MediaInfo
    bad_segments: list[BadSegment]
    resonant_freqs: list[float] = field(default_factory=list)


def run(input_path: Path, config: AutoCutConfig, console: Console) -> PipelineResult:
    """Run the full AutoCut pipeline and return all detected bad segments."""
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

        resonant_freqs: list[float] = []
        if config.room_eq_enabled:
            t4 = progress.add_task("Analysing room resonances…", total=None)
            resonant_freqs = analyze_room_resonances(audio_path, silence_segs, config)
            progress.update(
                t4,
                description=f"[green]Room EQ: {len(resonant_freqs)} resonance(s) identified",
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
        resonant_freqs=resonant_freqs,
    )
