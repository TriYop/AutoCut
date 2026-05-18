class AutoCutError(Exception):
    exit_code: int = 1


class AudioExtractionError(AutoCutError):
    exit_code = 2


class UnsupportedFormatError(AutoCutError):
    exit_code = 3


class ModelLoadError(AutoCutError):
    exit_code = 4
