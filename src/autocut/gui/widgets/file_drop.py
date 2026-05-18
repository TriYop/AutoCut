"""Drag-and-drop / browse file selection widget."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)

_VIDEO_FILTER = "Video files (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)"


class FileDropWidget(QGroupBox):
    """Group box that accepts a video file via drag-and-drop or a Browse dialog.

    Emits file_selected(path) whenever the selection changes.
    """

    file_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__("Input file")
        self.setAcceptDrops(True)
        self._path: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        self._label = QLabel("Drop a video file here or click Browse…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        layout.addWidget(self._label)
        layout.addWidget(browse)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", _VIDEO_FILTER)
        if path:
            self._set_path(path)

    def _set_path(self, path: str) -> None:
        self._path = path
        self._label.setText(Path(path).name)
        self.file_selected.emit(path)

    @property
    def path(self) -> str:
        return self._path

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            self._set_path(urls[0].toLocalFile())
