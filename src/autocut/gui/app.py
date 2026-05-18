"""Construct the QApplication and main window."""

from PySide6.QtWidgets import QApplication

from autocut.gui.widgets.main_window import AutoCutMainWindow


def build_app(argv: list[str]) -> tuple[QApplication, AutoCutMainWindow]:
    app = QApplication(argv)
    app.setApplicationName("AutoCut")
    app.setApplicationVersion("0.1.0")
    window = AutoCutMainWindow()
    return app, window
