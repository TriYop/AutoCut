import pytest

# Skip the entire gui test package when PySide6 is not installed so that the
# default CI run (no [gui] extra) does not fail on the import.
pytest.importorskip("PySide6", reason="GUI tests require the [gui] extra: uv sync --extra gui")
