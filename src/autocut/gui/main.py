"""GUI entry point — called by the `autocut-gui` console script."""

import sys


def main() -> None:
    """Entry point for the autocut-gui console script; exits non-zero on failure."""
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except ImportError:
        print(
            "The AutoCut GUI requires PySide6.\n"
            "Install it with:  pip install 'autocut[gui]'\n"
            "or:               uv sync --extra gui",
            file=sys.stderr,
        )
        sys.exit(1)

    from autocut.gui.app import build_app

    app, window = build_app(sys.argv)
    window.show()
    sys.exit(app.exec())
