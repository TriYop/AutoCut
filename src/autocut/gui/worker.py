"""Background pipeline worker: calls each stage directly and emits log lines."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from autocut.config import AutoCutConfig
from autocut.exceptions import AutoCutError
from autocut.output.cutter import cut_video
from autocut.output.edl import write_edl, write_json
from autocut.pipeline.audio import extract_audio
from autocut.pipeline.merger import merge_bad_segments
from autocut.pipeline.room_eq import analyze_room_resonances
from autocut.pipeline.runner import PipelineResult
from autocut.pipeline.transcriber import detect_fillers_and_repetitions
from autocut.pipeline.vad import detect_silences


class PipelineWorker(QObject):
    log_line = Signal(str)
    finished = Signal(object)  # PipelineResult — object avoids Qt metatype registration
    error = Signal(str)

    def __init__(
        self,
        input_path: Path,
        output_mode: str,
        output_dir: Path,
        config: AutoCutConfig,
    ) -> None:
        super().__init__()
        self._input_path = input_path
        self._output_mode = output_mode
        self._output_dir = output_dir
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            self._run()
        except AutoCutError as e:
            self.error.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Unexpected error: {e}")

    def _run(self) -> None:
        self.log_line.emit("Extracting audio…")
        audio_path, media_info = extract_audio(self._input_path, self._config)
        self.log_line.emit(
            f"Audio extracted  ({media_info.duration_s:.1f}s  {media_info.fps:.2f}fps)"
        )

        self.log_line.emit("Running Voice Activity Detection…")
        silence_segs = detect_silences(audio_path, media_info.duration_s, self._config)
        self.log_line.emit(f"VAD: {len(silence_segs)} silence(s) detected")

        self.log_line.emit(f"Transcribing with Whisper ({self._config.whisper_model})…")
        filler_segs, repetition_segs = detect_fillers_and_repetitions(audio_path, self._config)
        self.log_line.emit(
            f"Whisper: {len(filler_segs)} filler(s)  {len(repetition_segs)} repetition(s)"
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
            cut_video(
                self._input_path,
                merged,
                media_info,
                video_path,
                self._config,
                resonant_freqs,
            )
            self.log_line.emit(f"Video: {video_path}")

        result = PipelineResult(
            input_path=self._input_path,
            media_info=media_info,
            bad_segments=merged,
            resonant_freqs=resonant_freqs,
        )
        self.finished.emit(result)
