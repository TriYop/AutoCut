"""Parameters panel: mirrors all CLI options, grouped by concern."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autocut.config import AutoCutConfig


class ParamsPanel(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._output_group())
        layout.addWidget(self._model_group())
        layout.addWidget(self._detection_group())
        layout.addWidget(self._audio_group())
        layout.addStretch()

    # ── Output ────────────────────────────────────────────────────────────────

    def _output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)

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

        self.verbose = QCheckBox("Verbose output")
        form.addRow("", self.verbose)

        return box

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.output_dir.setText(d)

    # ── Model ─────────────────────────────────────────────────────────────────

    def _model_group(self) -> QGroupBox:
        defaults = AutoCutConfig()
        box = QGroupBox("Whisper model")
        form = QFormLayout(box)

        self.model = QComboBox()
        self.model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model.setCurrentText(defaults.whisper_model)
        form.addRow("Model:", self.model)

        self.language = QLineEdit()
        self.language.setPlaceholderText("Auto-detect")
        form.addRow("Language:", self.language)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.device.setCurrentText(defaults.whisper_device)
        form.addRow("Device:", self.device)

        return box

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detection_group(self) -> QGroupBox:
        defaults = AutoCutConfig()
        box = QGroupBox("Detection")
        form = QFormLayout(box)

        self.min_silence_ms = QSpinBox()
        self.min_silence_ms.setRange(100, 5000)
        self.min_silence_ms.setSingleStep(50)
        self.min_silence_ms.setSuffix(" ms")
        self.min_silence_ms.setValue(defaults.vad_min_silence_duration_ms)
        form.addRow("Min silence:", self.min_silence_ms)

        self.merge_gap = QDoubleSpinBox()
        self.merge_gap.setRange(0.0, 5.0)
        self.merge_gap.setSingleStep(0.05)
        self.merge_gap.setDecimals(2)
        self.merge_gap.setSuffix(" s")
        self.merge_gap.setValue(defaults.merge_gap_s)
        form.addRow("Merge gap:", self.merge_gap)

        self.fillers = QLineEdit()
        self.fillers.setPlaceholderText(", ".join(defaults.filler_words) + " (leave blank for defaults)")
        form.addRow("Filler words:", self.fillers)

        self.no_repetitions = QCheckBox("Disable repetition detection")
        form.addRow("", self.no_repetitions)

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

        return box

    # ── Audio processing ──────────────────────────────────────────────────────

    def _audio_group(self) -> QGroupBox:
        defaults = AutoCutConfig()
        box = QGroupBox("Audio processing")
        form = QFormLayout(box)

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
        self.room_eq.toggled.connect(self.room_eq_gain.setEnabled)
        form.addRow("EQ gain:", self.room_eq_gain)

        return box

    # ── Public API ────────────────────────────────────────────────────────────

    def as_config(self) -> AutoCutConfig:
        """Build an AutoCutConfig from the current widget state."""
        fillers_text = self.fillers.text().strip()
        return AutoCutConfig(
            whisper_model=self.model.currentText(),
            whisper_language=self.language.text().strip() or None,
            whisper_device=self.device.currentText(),
            vad_min_silence_duration_ms=self.min_silence_ms.value(),
            vad_max_silence_duration_s=None if self.no_silence_cap.isChecked() else self.max_silence_s.value(),
            merge_gap_s=self.merge_gap.value(),
            filler_words=[w.strip() for w in fillers_text.split(",")] if fillers_text else AutoCutConfig().filler_words,
            detect_repetitions=not self.no_repetitions.isChecked(),
            crossfade_ms=self.crossfade_ms.value(),
            room_eq_enabled=self.room_eq.isChecked(),
            room_eq_gain_db=self.room_eq_gain.value(),
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
