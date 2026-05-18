"""Main application window."""

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from autocut.gui.widgets.file_drop import FileDropWidget
from autocut.gui.widgets.params_panel import ParamsPanel


class AutoCutMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoCut")
        self.setMinimumSize(720, 640)
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

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setPlaceholderText("Output will appear here…")
        layout.addWidget(self.log)

        self.file_drop.file_selected.connect(self._on_file_selected)
        self.run_button.clicked.connect(self._on_run)

    def _on_file_selected(self, path: str) -> None:
        self.run_button.setEnabled(bool(path))

    def _on_run(self) -> None:
        # TODO: launch pipeline in a QThread, stream log lines into self.log
        self.log.append("Processing not yet implemented in GUI.")
