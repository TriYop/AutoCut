"""Pipeline result cache: persist VAD + Whisper segments to skip re-analysis on re-runs."""

import hashlib
import json
from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment, SegmentSource

CACHE_FORMAT_VERSION = 1


def _fingerprint(path: Path) -> dict[str, object]:
    """Return a dict capturing the file's identity (path, mtime, size) for cache invalidation."""
    stat = path.stat()
    return {"path": str(path.resolve()), "mtime": stat.st_mtime, "size": stat.st_size}


def _config_hash(config: AutoCutConfig) -> str:
    """Hash detection-relevant config fields so the cache is invalidated when they change."""
    relevant: dict[str, object] = {
        "vad_min_silence_duration_ms": config.vad_min_silence_duration_ms,
        "vad_speech_pad_ms": config.vad_speech_pad_ms,
        "vad_max_silence_duration_s": config.vad_max_silence_duration_s,
        "whisper_model": config.whisper_model,
        "whisper_language": config.whisper_language,
        "whisper_compute_type": config.whisper_compute_type,
        "filler_words": sorted(config.filler_words),
        "min_filler_duration_s": config.min_filler_duration_s,
        "detect_repetitions": config.detect_repetitions,
        "repetition_window_words": config.repetition_window_words,
        "repetition_min_word_length": config.repetition_min_word_length,
    }
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def cache_path(input_path: Path) -> Path:
    """Return the sidecar cache file path for the given input."""
    return input_path.with_suffix(input_path.suffix + ".autocut-cache.json")


def _seg_to_dict(seg: BadSegment) -> dict[str, object]:
    """Serialise a BadSegment to a JSON-compatible dict."""
    return {
        "start": seg.segment.start,
        "end": seg.segment.end,
        "source": seg.source.value,
        "label": seg.label,
        "confidence": seg.confidence,
    }


def _dict_to_seg(d: dict[str, object]) -> BadSegment:
    """Deserialise a BadSegment from the JSON-compatible dict produced by _seg_to_dict."""
    return BadSegment(
        segment=Segment(float(d["start"]), float(d["end"])),
        source=SegmentSource(d["source"]),
        label=str(d["label"]),
        confidence=float(d.get("confidence", 1.0)),
    )


def load(
    input_path: Path,
    config: AutoCutConfig,
) -> tuple[list[BadSegment], list[BadSegment]] | None:
    """Return (vad_segments, whisper_segments) from cache, or None if absent/stale.

    Returns None when:
    - The cache file does not exist
    - The cache format version differs
    - The input file mtime or size has changed
    - Any detection-relevant config field has changed
    """
    cf = cache_path(input_path)
    if not cf.exists():
        return None
    try:
        data = json.loads(cf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if data.get("format_version") != CACHE_FORMAT_VERSION:
        return None
    if data.get("fingerprint") != _fingerprint(input_path):
        return None
    if data.get("config_hash") != _config_hash(config):
        return None

    vad = [_dict_to_seg(d) for d in data.get("vad_segments", [])]
    whisper = [_dict_to_seg(d) for d in data.get("whisper_segments", [])]
    return vad, whisper


def save(
    input_path: Path,
    vad_segments: list[BadSegment],
    whisper_segments: list[BadSegment],
    config: AutoCutConfig,
) -> None:
    """Write VAD and Whisper results to the sidecar cache file (best-effort)."""
    data = {
        "format_version": CACHE_FORMAT_VERSION,
        "fingerprint": _fingerprint(input_path),
        "config_hash": _config_hash(config),
        "vad_segments": [_seg_to_dict(s) for s in vad_segments],
        "whisper_segments": [_seg_to_dict(s) for s in whisper_segments],
    }
    try:
        cache_path(input_path).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
