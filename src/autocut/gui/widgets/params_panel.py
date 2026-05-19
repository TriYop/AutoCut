"""Parameters panel: all CLI options arranged in tabs by concern."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _CUDA_AVAILABLE = False

from autocut.config import AutoCutConfig


class ParamsPanel(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self.addTab(self._output_tab(), "Output")
        self.addTab(self._whisper_tab(), "Whisper")
        self.addTab(self._detection_tab(), "Detection")
        self.addTab(self._audio_tab(), "Audio")

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _output_tab(self) -> QWidget:
        defaults = AutoCutConfig()
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(8, 8, 8, 8)

        self.output_mode = QComboBox()
        self.output_mode.addItems(["edl", "video", "both"])
        form.addRow("Mode:", self.output_mode)

        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Same directory as input")
        browse_dir = QPushButton("…")
        browse_dir.setFixedWidth(30)
        browse_dir.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(self.output_dir)
        dir_layout.addWidget(browse_dir)
        form.addRow("Directory:", dir_row)

        self.padding_before = QDoubleSpinBox()
        self.padding_before.setRange(0.0, 1.0)
        self.padding_before.setSingleStep(0.01)
        self.padding_before.setDecimals(3)
        self.padding_before.setSuffix(" s")
        self.padding_before.setValue(defaults.padding_before_s)
        form.addRow("Padding before:", self.padding_before)

        self.padding_after = QDoubleSpinBox()
        self.padding_after.setRange(0.0, 1.0)
        self.padding_after.setSingleStep(0.01)
        self.padding_after.setDecimals(3)
        self.padding_after.setSuffix(" s")
        self.padding_after.setValue(defaults.padding_after_s)
        form.addRow("Padding after:", self.padding_after)

        self.no_cache = QCheckBox("Force re-analysis (ignore cache)")
        form.addRow("", self.no_cache)

        self.verbose = QCheckBox("Verbose output")
        form.addRow("", self.verbose)

        return tab

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.output_dir.setText(d)

    def _whisper_tab(self) -> QWidget:
        defaults = AutoCutConfig()
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(8, 8, 8, 8)

        self.model = QComboBox()
        self.model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model.setCurrentText(defaults.whisper_model)
        form.addRow("Model:", self.model)

        self.language = QLineEdit()
        self.language.setPlaceholderText("Auto-detect")
        form.addRow("Language:", self.language)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        if not _CUDA_AVAILABLE:
            cuda_item = self.device.model().item(1)
            cuda_item.setFlags(cuda_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.device.setToolTip("CUDA is not available on this system")
        self.device.setCurrentText(defaults.whisper_device)
        form.addRow("Device:", self.device)

        self.compute_type = QComboBox()
        self.compute_type.addItems(["int8", "float16", "float32"])
        self.compute_type.setCurrentText(defaults.whisper_compute_type)
        form.addRow("Compute type:", self.compute_type)

        return tab

    def _detection_tab(self) -> QWidget:
        defaults = AutoCutConfig()
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(8, 8, 8, 8)

        self.min_silence_ms = QSpinBox()
        self.min_silence_ms.setRange(100, 5000)
        self.min_silence_ms.setSingleStep(50)
        self.min_silence_ms.setSuffix(" ms")
        self.min_silence_ms.setValue(defaults.vad_min_silence_duration_ms)
        form.addRow("Min silence:", self.min_silence_ms)

        self.speech_pad_ms = QSpinBox()
        self.speech_pad_ms.setRange(0, 500)
        self.speech_pad_ms.setSingleStep(10)
        self.speech_pad_ms.setSuffix(" ms")
        self.speech_pad_ms.setValue(defaults.vad_speech_pad_ms)
        form.addRow("Speech pad:", self.speech_pad_ms)

        self.merge_gap = QDoubleSpinBox()
        self.merge_gap.setRange(0.0, 5.0)
        self.merge_gap.setSingleStep(0.05)
        self.merge_gap.setDecimals(2)
        self.merge_gap.setSuffix(" s")
        self.merge_gap.setValue(defaults.merge_gap_s)
        form.addRow("Merge gap:", self.merge_gap)

        self.max_silence_s = QDoubleSpinBox()
        self.max_silence_s.setRange(1.0, 600.0)
        self.max_silence_s.setSingleStep(5.0)
        self.max_silence_s.setDecimals(1)
        self.max_silence_s.setSuffix(" s")
        self.max_silence_s.setValue(defaults.vad_max_silence_duration_s or 30.0)
        form.addRow("Max silence:", self.max_silence_s)

        self.no_silence_cap = QCheckBox("Cut all silences (no cap — good for replays)")
        self.no_silence_cap.toggled.connect(lambda on: self.max_silence_s.setEnabled(not on))
        form.addRow("", self.no_silence_cap)

        self.fillers = QLineEdit()
        self.fillers.setPlaceholderText(", ".join(defaults.filler_words) + " (leave blank for defaults)")
        form.addRow("Filler words:", self.fillers)

        self.min_filler_duration = QDoubleSpinBox()
        self.min_filler_duration.setRange(0.05, 2.0)
        self.min_filler_duration.setSingleStep(0.05)
        self.min_filler_duration.setDecimals(2)
        self.min_filler_duration.setSuffix(" s")
        self.min_filler_duration.setValue(defaults.min_filler_duration_s)
        form.addRow("Min filler dur.:", self.min_filler_duration)

        self.no_repetitions = QCheckBox("Disable repetition detection")
        form.addRow("", self.no_repetitions)

        self.repetition_window = QSpinBox()
        self.repetition_window.setRange(2, 10)
        self.repetition_window.setSuffix(" words")
        self.repetition_window.setValue(defaults.repetition_window_words)
        self.no_repetitions.toggled.connect(lambda on: self.repetition_window.setEnabled(not on))
        form.addRow("Repetition window:", self.repetition_window)

        self.repetition_min_length = QSpinBox()
        self.repetition_min_length.setRange(1, 8)
        self.repetition_min_length.setSuffix(" chars")
        self.repetition_min_length.setValue(defaults.repetition_min_word_length)
        self.no_repetitions.toggled.connect(lambda on: self.repetition_min_length.setEnabled(not on))
        form.addRow("Min word length:", self.repetition_min_length)

        return tab

    def _audio_tab(self) -> QWidget:
        defaults = AutoCutConfig()
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(8, 8, 8, 8)

        self.crossfade_ms = QSpinBox()
        self.crossfade_ms.setRange(0, 500)
        self.crossfade_ms.setSingleStep(10)
        self.crossfade_ms.setSuffix(" ms")
        self.crossfade_ms.setSpecialValueText("Disabled")
        self.crossfade_ms.setValue(defaults.crossfade_ms)
        form.addRow("Crossfade:", self.crossfade_ms)

        self.room_eq = QCheckBox("Enable room EQ")
        form.addRow("", self.room_eq)

        self.room_eq_gain = QDoubleSpinBox()
        self.room_eq_gain.setRange(-30.0, 0.0)
        self.room_eq_gain.setSingleStep(1.0)
        self.room_eq_gain.setDecimals(1)
        self.room_eq_gain.setSuffix(" dB")
        self.room_eq_gain.setValue(defaults.room_eq_gain_db)
        self.room_eq_gain.setEnabled(False)
        form.addRow("EQ gain:", self.room_eq_gain)

        self.room_eq_threshold = QDoubleSpinBox()
        self.room_eq_threshold.setRange(1.0, 40.0)
        self.room_eq_threshold.setSingleStep(1.0)
        self.room_eq_threshold.setDecimals(1)
        self.room_eq_threshold.setSuffix(" dB")
        self.room_eq_threshold.setValue(defaults.room_eq_threshold_db)
        self.room_eq_threshold.setEnabled(False)
        form.addRow("EQ threshold:", self.room_eq_threshold)

        self.room_eq_filters = QSpinBox()
        self.room_eq_filters.setRange(1, 20)
        self.room_eq_filters.setSuffix(" filters")
        self.room_eq_filters.setValue(defaults.room_eq_max_filters)
        self.room_eq_filters.setEnabled(False)
        form.addRow("Max EQ filters:", self.room_eq_filters)

        self.room_eq_q = QDoubleSpinBox()
        self.room_eq_q.setRange(1.0, 50.0)
        self.room_eq_q.setSingleStep(0.5)
        self.room_eq_q.setDecimals(1)
        self.room_eq_q.setValue(defaults.room_eq_q_factor)
        self.room_eq_q.setEnabled(False)
        form.addRow("EQ Q factor:", self.room_eq_q)

        self.room_eq.toggled.connect(self.room_eq_gain.setEnabled)
        self.room_eq.toggled.connect(self.room_eq_threshold.setEnabled)
        self.room_eq.toggled.connect(self.room_eq_filters.setEnabled)
        self.room_eq.toggled.connect(self.room_eq_q.setEnabled)

        return tab

    # ── Public API ────────────────────────────────────────────────────────────

    def as_config(self) -> AutoCutConfig:
        """Build an AutoCutConfig from the current widget state."""
        fillers_text = self.fillers.text().strip()
        return AutoCutConfig(
            whisper_model=self.model.currentText(),
            whisper_language=self.language.text().strip() or None,
            whisper_device=self.device.currentText(),
            whisper_compute_type=self.compute_type.currentText(),
            vad_min_silence_duration_ms=self.min_silence_ms.value(),
            vad_speech_pad_ms=self.speech_pad_ms.value(),
            vad_max_silence_duration_s=None if self.no_silence_cap.isChecked() else self.max_silence_s.value(),
            merge_gap_s=self.merge_gap.value(),
            padding_before_s=self.padding_before.value(),
            padding_after_s=self.padding_after.value(),
            filler_words=[w.strip() for w in fillers_text.split(",")] if fillers_text else AutoCutConfig().filler_words,
            min_filler_duration_s=self.min_filler_duration.value(),
            detect_repetitions=not self.no_repetitions.isChecked(),
            repetition_window_words=self.repetition_window.value(),
            repetition_min_word_length=self.repetition_min_length.value(),
            use_cache=not self.no_cache.isChecked(),
            crossfade_ms=self.crossfade_ms.value(),
            room_eq_enabled=self.room_eq.isChecked(),
            room_eq_gain_db=self.room_eq_gain.value(),
            room_eq_threshold_db=self.room_eq_threshold.value(),
            room_eq_max_filters=self.room_eq_filters.value(),
            room_eq_q_factor=self.room_eq_q.value(),
        )

    @property
    def output_mode_value(self) -> str:
        return self.output_mode.currentText()

    @property
    def output_dir_value(self) -> str:
        return self.output_dir.text().strip()

    @property
    def verbose_value(self) -> bool:
        return self.verbose.isChecked()
