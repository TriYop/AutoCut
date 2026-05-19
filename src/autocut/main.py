"""CLI entry point: parse options, run the pipeline, write EDL/JSON/video output."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn

from autocut.config import AutoCutConfig
from autocut.exceptions import AutoCutError
from autocut.output.cutter import cut_video
from autocut.output.edl import write_edl, write_json
from autocut.pipeline.runner import PipelineResult, run

console = Console()


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", "output_mode",
    type=click.Choice(["edl", "video", "both"]),
    default="edl",
    show_default=True,
    help="Output type: EDL markers, cleaned video, or both.",
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=None,
    help="Directory for output files (default: same as input).",
)
@click.option("--padding-before", default=0.05, show_default=True, type=float,
              help="Seconds of speech to keep before each cut.")
@click.option("--padding-after", default=0.05, show_default=True, type=float,
              help="Seconds of speech to keep after each cut.")
@click.option("--model", default="small", show_default=True,
              help="Whisper model: tiny/base/small/medium/large-v3")
@click.option("--language", default=None,
              help="Language code (e.g. fr, en). Auto-detect if omitted.")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", show_default=True,
              help="Device for Whisper inference.")
@click.option(
    "--compute-type",
    type=click.Choice(["int8", "float16", "float32"]),
    default="int8", show_default=True,
    help="Whisper quantisation: int8 (fastest), float16 (GPU), float32 (full precision).",
)
@click.option("--min-silence-ms", default=700, show_default=True, type=int,
              help="Minimum silence duration to flag (ms).")
@click.option("--speech-pad-ms", default=150, show_default=True, type=int,
              help="Silence padding added around detected speech to avoid clipping (ms).")
@click.option("--merge-gap", default=0.2, show_default=True, type=float,
              help="Merge bad segments closer than this many seconds.")
@click.option("--max-silence-s", default=30.0, show_default=True, type=float,
              help="Silences longer than this are kept (Q&A breaks, applause…).")
@click.option("--no-silence-cap", is_flag=True, default=False,
              help="Cut all silences regardless of duration (good for YT replays).")
@click.option("--fillers", default=None,
              help="Comma-separated filler word list override.")
@click.option("--min-filler-duration", default=0.3, show_default=True, type=float,
              help="Minimum filler word duration to flag (s). Shorter utterances are ignored.")
@click.option("--no-repetitions", is_flag=True, default=False,
              help="Disable repetition detection.")
@click.option("--repetition-window", default=3, show_default=True, type=int,
              help="Number of consecutive words checked for repetitions.")
@click.option("--repetition-min-length", default=2, show_default=True, type=int,
              help="Minimum word length considered for repetition detection.")
@click.option("--crossfade-ms", default=0, show_default=True, type=int,
              help="Fade-out/in duration at each cut point (ms). 0 = disabled. ~120ms recommended.")
@click.option("--room-eq", is_flag=True, default=False,
              help="Detect and attenuate room resonance frequencies from silence segments.")
@click.option("--room-eq-gain", default=-10.0, show_default=True, type=float,
              help="Attenuation applied to each resonance (dB, negative = cut).")
@click.option("--room-eq-threshold", default=10.0, show_default=True, type=float,
              help="Minimum peak height above noise floor to flag a resonance (dB).")
@click.option("--room-eq-filters", default=5, show_default=True, type=int,
              help="Maximum number of notch filters applied by room EQ.")
@click.option("--room-eq-q", default=8.0, show_default=True, type=float,
              help="Q factor for room EQ notch filters (higher = narrower notch).")
@click.option("--no-cache", is_flag=True, default=False,
              help="Force re-analysis even if a fresh cache exists.")
@click.option("--verbose", "-v", is_flag=True)
def cli(
    input_file: Path,
    output_mode: str,
    output_dir: Path | None,
    padding_before: float,
    padding_after: float,
    model: str,
    language: str | None,
    device: str,
    compute_type: str,
    min_silence_ms: int,
    speech_pad_ms: int,
    merge_gap: float,
    max_silence_s: float,
    no_silence_cap: bool,
    fillers: str | None,
    min_filler_duration: float,
    no_repetitions: bool,
    repetition_window: int,
    repetition_min_length: int,
    crossfade_ms: int,
    room_eq: bool,
    room_eq_gain: float,
    room_eq_threshold: float,
    room_eq_filters: int,
    room_eq_q: float,
    no_cache: bool,
    verbose: bool,
) -> None:
    """Detect hesitations and stutters in a webinar video and export cut regions."""
    config = AutoCutConfig(
        whisper_model=model,
        whisper_language=language,
        whisper_device=device,
        whisper_compute_type=compute_type,
        vad_min_silence_duration_ms=min_silence_ms,
        vad_speech_pad_ms=speech_pad_ms,
        vad_max_silence_duration_s=None if no_silence_cap else max_silence_s,
        merge_gap_s=merge_gap,
        padding_before_s=padding_before,
        padding_after_s=padding_after,
        filler_words=[w.strip() for w in fillers.split(",")] if fillers else AutoCutConfig().filler_words,
        min_filler_duration_s=min_filler_duration,
        detect_repetitions=not no_repetitions,
        repetition_window_words=repetition_window,
        repetition_min_word_length=repetition_min_length,
        crossfade_ms=crossfade_ms,
        room_eq_enabled=room_eq,
        room_eq_gain_db=room_eq_gain,
        room_eq_threshold_db=room_eq_threshold,
        room_eq_max_filters=room_eq_filters,
        room_eq_q_factor=room_eq_q,
        use_cache=not no_cache,
    )

    out_dir = output_dir or input_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem

    try:
        result = run(input_file, config, console)
    except AutoCutError as e:
        console.print(Panel(f"[red]{e}[/red]", title="Error", border_style="red"))
        sys.exit(e.exit_code)
    except Exception as e:  # noqa: BLE001
        console.print(Panel(f"[red]Unexpected error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)

    n = len(result.bad_segments)
    total_cut = sum(s.segment.duration for s in result.bad_segments)
    console.print(f"\n[bold]Found {n} region(s) to cut[/bold] ({total_cut:.1f}s total)")

    if verbose:
        for seg in result.bad_segments:
            console.print(
                f"  [{seg.segment.start:.2f}s – {seg.segment.end:.2f}s] "
                f"{seg.label} ({seg.source.value})"
            )

    source_name = input_file.name

    if output_mode in ("edl", "both"):
        edl_path = out_dir / f"{stem}_cuts.edl"
        json_path = out_dir / f"{stem}_cuts.json"
        write_edl(result.bad_segments, result.media_info, source_name, edl_path)
        write_json(result.bad_segments, source_name, json_path)
        console.print(f"[green]EDL:[/green]  {edl_path}")
        console.print(f"[green]JSON:[/green] {json_path}")

    if output_mode in ("video", "both"):
        video_path = out_dir / f"{stem}_cleaned{input_file.suffix}"
        try:
            _encode_with_progress(
                input_file, result, video_path, config, console,
            )
            console.print(f"[green]Video:[/green] {video_path}")
        except AutoCutError as e:
            console.print(Panel(f"[red]{e}[/red]", title="Video cutting error", border_style="red"))
            sys.exit(e.exit_code)


def _encode_with_progress(
    input_file: Path,
    result: PipelineResult,
    video_path: Path,
    config: AutoCutConfig,
    console: Console,
) -> None:
    """Call cut_video with a Rich progress bar showing encode time and ETA."""
    encode_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )
    task_id: TaskID | None = None

    def _on_progress(current_s: float, total_s: float) -> None:
        """Create or advance the Rich progress task for each out_time_us update from FFmpeg."""
        nonlocal task_id
        if task_id is None:
            task_id = encode_progress.add_task("Re-encoding…", total=total_s)
        encode_progress.update(task_id, completed=current_s)

    with encode_progress:
        cut_video(
            input_file,
            result.bad_segments,
            result.media_info,
            video_path,
            config,
            result.resonant_freqs,
            progress_cb=_on_progress,
        )
