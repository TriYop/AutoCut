"""CLI entry point: parse options, run the pipeline, write EDL/JSON/video output."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from autocut.config import AutoCutConfig
from autocut.exceptions import AutoCutError
from autocut.output.cutter import cut_video
from autocut.output.edl import write_edl, write_json
from autocut.pipeline.runner import run

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
@click.option("--model", default="small", show_default=True,
              help="Whisper model: tiny/base/small/medium/large-v3")
@click.option("--language", default=None,
              help="Language code (e.g. fr, en). Auto-detect if omitted.")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", show_default=True,
              help="Device for Whisper inference.")
@click.option("--min-silence-ms", default=700, show_default=True, type=int,
              help="Minimum silence duration to flag (ms).")
@click.option("--merge-gap", default=0.2, show_default=True, type=float,
              help="Merge bad segments closer than this many seconds.")
@click.option("--fillers", default=None,
              help="Comma-separated filler word list override.")
@click.option("--no-repetitions", is_flag=True, default=False,
              help="Disable repetition detection.")
@click.option("--max-silence-s", default=30.0, show_default=True, type=float,
              help="Silences longer than this are kept (Q&A breaks, applause…).")
@click.option("--no-silence-cap", is_flag=True, default=False,
              help="Cut all silences regardless of duration (good for YT replays).")
@click.option("--crossfade-ms", default=0, show_default=True, type=int,
              help="Fade-out/in duration at each cut point (ms). 0 = disabled. ~120ms recommended.")
@click.option("--room-eq", is_flag=True, default=False,
              help="Detect and attenuate room resonance frequencies from silence segments.")
@click.option("--room-eq-gain", default=-10.0, show_default=True, type=float,
              help="Attenuation applied to each resonance (dB, negative = cut).")
@click.option("--verbose", "-v", is_flag=True)
def cli(
    input_file: Path,
    output_mode: str,
    output_dir: Path | None,
    model: str,
    language: str | None,
    device: str,
    min_silence_ms: int,
    merge_gap: float,
    fillers: str | None,
    no_repetitions: bool,
    max_silence_s: float,
    no_silence_cap: bool,
    crossfade_ms: int,
    room_eq: bool,
    room_eq_gain: float,
    verbose: bool,
) -> None:
    """Detect hesitations and stutters in a webinar video and export cut regions."""
    config = AutoCutConfig(
        whisper_model=model,
        whisper_language=language,
        whisper_device=device,
        vad_min_silence_duration_ms=min_silence_ms,
        vad_max_silence_duration_s=None if no_silence_cap else max_silence_s,
        merge_gap_s=merge_gap,
        filler_words=[w.strip() for w in fillers.split(",")] if fillers else AutoCutConfig().filler_words,
        detect_repetitions=not no_repetitions,
        crossfade_ms=crossfade_ms,
        room_eq_enabled=room_eq,
        room_eq_gain_db=room_eq_gain,
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
            cut_video(input_file, result.bad_segments, result.media_info, video_path, config, result.resonant_freqs)
            console.print(f"[green]Video:[/green] {video_path}")
        except AutoCutError as e:
            console.print(Panel(f"[red]{e}[/red]", title="Video cutting error", border_style="red"))
            sys.exit(e.exit_code)
