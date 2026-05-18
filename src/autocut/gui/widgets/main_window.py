"""Main application window."""

from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from autocut.gui.widgets.file_drop import FileDropWidget
from autocut.gui.widgets.params_panel import ParamsPanel
from autocut.gui.worker import PipelineWorker


class AutoCutMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoCut")
        self.setMinimumSize(720, 640)
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.file_drop = FileDropWidget()
        layout.addWidget(self.file_drop)

        self.params = ParamsPanel()
        layout.addWidget(self.params, stretch=1)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setEnabled(False)
        self.run_button.setFixedHeight(36)
        run_row.addStretch()
        run_row.addWidget(self.run_button)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate / busy indicator
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setPlaceholderText("Output will appear here…")
        layout.addWidget(self.log)

        self.file_drop.file_selected.connect(self._on_file_selected)
        self.run_button.clicked.connect(self._on_run)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_file_selected(self, path: str) -> None:
        self.run_button.setEnabled(bool(path))

    def _on_run(self) -> None:
        input_path = Path(self.file_drop.path)
        output_dir_str = self.params.output_dir_value
        output_dir = Path(output_dir_str) if output_dir_str else input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        self._thread = QThread()
        self._worker = PipelineWorker(
            input_path=input_path,
            output_mode=self.params.output_mode_value,
            output_dir=output_dir,
            config=self.params.as_config(),
        )
        self._worker.moveToThread(self._thread)

        # Wire up: thread start → worker runs; worker done/error → thread stops
        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self._on_pipeline_log)
        self._worker.finished.connect(self._on_pipeline_done)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")
        self.progress_bar.setVisible(True)
        self.log.clear()
        self._thread.start()

    def _on_pipeline_log(self, line: str) -> None:
        self.log.append(line)

    def _on_pipeline_done(self) -> None:
        self._reset_run_button()
        self.log.append("\nDone.")

    def _on_pipeline_error(self, message: str) -> None:
        self._reset_run_button()
        self.log.append(f"\nError: {message}")

    def _reset_run_button(self) -> None:
        self.progress_bar.setVisible(False)
        self.run_button.setText("Run")
        self.run_button.setEnabled(True)
