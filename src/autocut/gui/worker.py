"""Background pipeline worker: calls each stage directly and emits log lines."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from autocut.config import AutoCutConfig
from autocut.exceptions import AutoCutError
from autocut.output.cutter import cut_video
from autocut.output.edl import write_edl, write_json
from autocut.pipeline import cache as pipeline_cache
from autocut.pipeline.audio import extract_audio
from autocut.pipeline.merger import merge_bad_segments
from autocut.pipeline.room_eq import analyze_room_resonances
from autocut.pipeline.runner import PipelineResult
from autocut.pipeline.transcriber import detect_fillers_and_repetitions
from autocut.pipeline.vad import detect_silences


class PipelineWorker(QObject):
    """QObject that runs the full AutoCut pipeline on a background QThread."""

    log_line = Signal(str)
    # (current_s, total_s) — emitted during video re-encode
    encode_progress = Signal(float, float)
    # PipelineResult passed as object to avoid Qt metatype registration
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        input_path: Path,
        output_mode: str,
        output_dir: Path,
        config: AutoCutConfig,
    ) -> None:
        """Store pipeline parameters; the worker is moved to a QThread before run() is called."""
        super().__init__()
        self._input_path = input_path
        self._output_mode = output_mode
        self._output_dir = output_dir
        self._config = config

    @Slot()
    def run(self) -> None:
        """Entry point called by QThread.started; delegates to _run and emits error on failure."""
        try:
            self._run()
        except AutoCutError as e:
            self.error.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Unexpected error: {e}")

    def _run(self) -> None:
        """Run each pipeline stage in order, emitting log lines and signals throughout."""
        self.log_line.emit("Extracting audio…")
        audio_path, media_info = extract_audio(self._input_path, self._config)
        self.log_line.emit(
            f"Audio extracted  ({media_info.duration_s:.1f}s  {media_info.fps:.2f}fps)"
        )

        cached = pipeline_cache.load(self._input_path, self._config) if self._config.use_cache else None

        if cached is not None:
            silence_segs, whisper_segs = cached
            filler_segs = [s for s in whisper_segs if s.source.value != "whisper_repetition"]
            repetition_segs = [s for s in whisper_segs if s.source.value == "whisper_repetition"]
            self.log_line.emit(
                f"Cache hit — skipped VAD + Whisper "
                f"({len(silence_segs)} silences, {len(filler_segs)} fillers, "
                f"{len(repetition_segs)} repetitions)"
            )
        else:
            self.log_line.emit("Running Voice Activity Detection…")
            silence_segs = detect_silences(audio_path, media_info.duration_s, self._config)
            self.log_line.emit(f"VAD: {len(silence_segs)} silence(s) detected")

            self.log_line.emit(f"Transcribing with Whisper ({self._config.whisper_model})…")
            filler_segs, repetition_segs = detect_fillers_and_repetitions(audio_path, self._config)
            self.log_line.emit(
                f"Whisper: {len(filler_segs)} filler(s)  {len(repetition_segs)} repetition(s)"
            )

            if self._config.use_cache:
                pipeline_cache.save(
                    self._input_path, silence_segs, filler_segs + repetition_segs, self._config
                )

        resonant_freqs: list[float] = []
        if self._config.room_eq_enabled:
            self.log_line.emit("Analysing room resonances…")
            resonant_freqs = analyze_room_resonances(audio_path, silence_segs, self._config)
            self.log_line.emit(f"Room EQ: {len(resonant_freqs)} resonance(s) identified")

        audio_path.unlink(missing_ok=True)

        all_bad = silence_segs + filler_segs + repetition_segs
        merged = merge_bad_segments(all_bad, self._config, media_info.duration_s)
        total_cut = sum(s.segment.duration for s in merged)
        self.log_line.emit(f"Found {len(merged)} region(s) to cut ({total_cut:.1f}s total)")

        stem = self._input_path.stem
        source_name = self._input_path.name

        if self._output_mode in ("edl", "both"):
            edl_path = self._output_dir / f"{stem}_cuts.edl"
            json_path = self._output_dir / f"{stem}_cuts.json"
            write_edl(merged, media_info, source_name, edl_path)
            write_json(merged, source_name, json_path)
            self.log_line.emit(f"EDL:  {edl_path}")
            self.log_line.emit(f"JSON: {json_path}")

        if self._output_mode in ("video", "both"):
            video_path = self._output_dir / f"{stem}_cleaned{self._input_path.suffix}"
            self.log_line.emit("Re-encoding video…")

            def _on_encode_progress(current_s: float, total_s: float) -> None:
                """Forward encode progress from the cutter to the encode_progress signal."""
                self.encode_progress.emit(current_s, total_s)

            cut_video(
                self._input_path,
                merged,
                media_info,
                video_path,
                self._config,
                resonant_freqs,
                progress_cb=_on_encode_progress,
            )
            self.log_line.emit(f"Video: {video_path}")

        result = PipelineResult(
            input_path=self._input_path,
            media_info=media_info,
            bad_segments=merged,
            resonant_freqs=resonant_freqs,
        )
        self.finished.emit(result)
