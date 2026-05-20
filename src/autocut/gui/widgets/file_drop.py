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
    """Group box that accepts a video file by drag-and-drop or via a browse dialog."""

    file_selected = Signal(str)

    def __init__(self) -> None:
        """Initialise the group box, enable drop acceptance, and build the child widgets."""
        super().__init__("Input file")
        self.setAcceptDrops(True)
        self._path: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        """Create the label and Browse button and arrange them in a horizontal layout."""
        layout = QHBoxLayout(self)
        self._label = QLabel("Drop a video file here or click Browse…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        layout.addWidget(self._label)
        layout.addWidget(browse)

    def _browse(self) -> None:
        """Open a native file dialog filtered to video formats and load the chosen path."""
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", _VIDEO_FILTER)
        if path:
            self._set_path(path)

    def _set_path(self, path: str) -> None:
        """Store path, update the label to show the filename, and emit file_selected."""
        self._path = path
        self._label.setText(Path(path).name)
        self.file_selected.emit(path)

    @property
    def path(self) -> str:
        """Currently selected file path, or empty string if no file has been chosen."""
        return self._path

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events that carry at least one URL."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Load the first dropped URL as the selected file path."""
        urls = event.mimeData().urls()
        if urls:
            self._set_path(urls[0].toLocalFile())
