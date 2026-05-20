"""Custom exceptions raised by the AutoCut pipeline, each carrying a process exit code."""


class AutoCutError(Exception):
    """Base exception for all AutoCut errors; carries a process exit code."""

    exit_code: int = 1


class AudioExtractionError(AutoCutError):
    """Raised when FFmpeg cannot extract or resample the audio stream."""

    exit_code = 2


class UnsupportedFormatError(AutoCutError):
    """Raised when the input file has no recognised streams or cannot be probed."""

    exit_code = 3


class ModelLoadError(AutoCutError):
    """Raised when the Whisper model fails to initialise."""

    exit_code = 4
