"""Audio extraction helpers: probe media properties and resample to 16 kHz WAV."""

import json
import subprocess
import tempfile
from pathlib import Path

import ffmpeg

from autocut.config import AutoCutConfig
from autocut.exceptions import AudioExtractionError, UnsupportedFormatError
from autocut.models import MediaInfo


def probe_media(input_path: Path) -> MediaInfo:
    """Return duration, frame-rate, and audio-presence for the given media file."""
    try:
        raw = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", str(input_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise UnsupportedFormatError(f"Cannot probe '{input_path}': {e.stderr}") from e

    info = json.loads(raw.stdout)
    streams = info.get("streams", [])

    has_audio = any(s["codec_type"] == "audio" for s in streams)
    if not streams:
        raise UnsupportedFormatError(f"No streams found in '{input_path}'")

    duration_s = float(info.get("format", {}).get("duration", 0))

    fps = 25.0
    for s in streams:
        if s.get("codec_type") == "video":
            r_frame_rate = s.get("r_frame_rate", "25/1")
            num, den = r_frame_rate.split("/")
            if int(den) > 0:
                fps = int(num) / int(den)
            break

    return MediaInfo(duration_s=duration_s, fps=fps, has_audio=has_audio)


def extract_audio(input_path: Path, config: AutoCutConfig) -> tuple[Path, MediaInfo]:
    """Extract a mono 16 kHz PCM WAV to a temp file; caller must delete it."""
    media_info = probe_media(input_path)

    if not media_info.has_audio:
        raise AudioExtractionError(f"'{input_path}' has no audio stream")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # noqa: SIM115
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        (
            ffmpeg
            .input(str(input_path))
            .audio
            .filter("aresample", 16000)
            .filter("aformat", sample_fmts="s16", channel_layouts="mono")
            .output(str(tmp_path), acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        tmp_path.unlink(missing_ok=True)
        stderr = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise AudioExtractionError(f"Audio extraction failed: {stderr}") from e

    return tmp_path, media_info
