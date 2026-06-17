# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Setup (first time)
uv sync --dev                # Install all dependencies including dev tools
uv sync --extra gui          # Also install PySide6 for the GUI

# Common commands
pytest                       # Run all tests
pytest tests/test_cache.py   # Run a single test file
ruff check src tests         # Lint the code
mypy src                     # Type check
autocut my_video.mp4         # Run the CLI
autocut-gui                  # Run the GUI
```

## Project Overview

AutoCut is a CLI and GUI tool that detects hesitations, fillers, and silence gaps in webinar videos and exports:
- **EDL** files (for import into Premiere Pro, DaVinci Resolve, etc.)
- **JSON** cut lists with timestamps and confidence scores
- **Cleaned video files** with bad regions removed

**Tech Stack:** Python 3.13, faster-whisper, Silero VAD, FFmpeg, Click (CLI), Rich (output formatting), PySide6 (GUI)

**Requirements:** Python 3.13+, FFmpeg on PATH, CUDA GPU optional

---

## Directory Structure

```
src/autocut/
├── main.py              # CLI entry point; option parsing → pipeline → output
├── config.py            # AutoCutConfig dataclass: all tunable detection parameters
├── models.py            # Domain models: Segment, BadSegment, MediaInfo, SegmentSource enum
├── exceptions.py        # Custom error types with exit codes
├── pipeline/
│   ├── runner.py        # Top-level orchestrator: audio → VAD → Whisper → merge → result
│   ├── audio.py         # FFmpeg-based audio extraction (→ 16 kHz mono WAV)
│   ├── vad.py           # Silero VAD: detect silence gaps
│   ├── transcriber.py   # Faster-whisper: detect fillers and word repetitions
│   ├── merger.py        # Merge overlapping/close segments; apply padding
│   ├── room_eq.py       # Room resonance detection (FFT analysis over silences)
│   └── cache.py         # Pipeline result caching (JSON sidecar per input file)
├── output/
│   ├── edl.py           # Write CMX 3600 EDL and JSON cut lists
│   └── cutter.py        # FFmpeg-based video re-encoding: apply cuts and optional audio effects
├── gui/
│   ├── main.py          # GUI entry point
│   ├── app.py           # QApplication lifecycle
│   ├── worker.py        # QThread-based pipeline runner (prevents UI freezing)
│   └── widgets/         # QTabWidget panels: Output, Whisper, Detection, Audio

tests/
├── test_cache.py        # Cache invalidation, load/save round-trips
├── test_cutter.py       # Video cutting, frame accuracy, crossfade
├── test_edl.py          # EDL formatting, timecode precision
├── test_merger.py       # Segment merging, padding, overlap handling
├── test_room_eq.py      # Resonance detection accuracy
├── test_transcriber.py  # Filler word and repetition detection
```

---

## Architecture & Data Flow

### Pipeline Stages (in `runner.py`)

```
Input video
  ↓
Audio Extraction (FFmpeg → mono 16 kHz WAV)
  ├─→ Silero VAD          [silence_segs]
  ├─→ Faster-Whisper      [filler_segs, repetition_segs]
  └─→ Room EQ (optional)  [resonant_freqs]
      ↓
Segment Merger (gap-fill + padding)
      ↓
Output (EDL / JSON / video)
```

**Key objects:**
- `AutoCutConfig` — all tunable parameters (detection thresholds, model sizes, padding, etc.)
- `Segment(start: float, end: float)` — a half-open time interval [start, end) in seconds
- `BadSegment` — a detected region with source (VAD/whisper_filler/whisper_repetition), label, and confidence
- `PipelineResult` — aggregated output: bad_segments list, media_info, resonant frequencies

### Caching

The pipeline caches VAD and Whisper results to `<input>.autocut-cache.json` after first run. Cache includes:
- File fingerprint (path + mtime + size)
- Config hash (detection parameters)
- Silence and Whisper segments

Cache is invalidated when the input file changes (size/mtime) or any detection parameter changes. Bypass with `--no-cache`.

### Configuration

All parameters live in `AutoCutConfig` (`config.py`). Every field is exposed via CLI flags (see README.md for full reference). Key sections:
- **VAD:** `vad_min_silence_duration_ms`, `vad_speech_pad_ms`, `vad_max_silence_duration_s`
- **Whisper:** `whisper_model`, `whisper_language`, `whisper_device`, `whisper_compute_type`
- **Filler detection:** `filler_words`, `min_filler_duration_s`
- **Repetition detection:** `detect_repetitions`, `repetition_window_words`, `repetition_min_word_length`
- **Merging:** `merge_gap_s`
- **Output padding:** `padding_before_s`, `padding_after_s`
- **Room EQ:** `room_eq_enabled`, `room_eq_threshold_db`, `room_eq_max_filters`, `room_eq_q_factor`, `room_eq_gain_db`
- **Crossfade:** `crossfade_ms` (0 = disabled; ~120 ms recommended)

---

## Code Style & Patterns

### Docstrings & Comments

- **Every function, class, and module** must have a docstring.
- **Comments** go on their own line *before* the code they describe — never trailing inline (after code on same line).
- Exception: Comments that explain WHY (non-obvious constraints, workarounds, invariants) are acceptable; comments that explain WHAT (what the code does) are redundant.

### Type Hints

- Use type hints for all function arguments and return types.
- Leverage `from __future__ import annotations` for forward refs if needed (already in use).

### File Organization

- Imports: stdlib, then third-party, then local (enforced by ruff isort rules).
- Module docstring at top.
- Then classes, functions, constants.

---

## Git Workflow

**Rule:** All changes must be made in a sub-branch; never directly on `master`.

**Merge prerequisites:**
1. CI build is stable (green).
2. Quality gates pass (SonarCloud).
3. Code review approval (for PRs).

**Merge strategy:** FF-only with squash (`git merge --ff-only --squash` or equivalent via PR).

**How to apply:**
1. Create a feature branch: `git checkout -b feature/my-feature`.
2. Make changes and commit locally.
3. Push to remote and create a PR.
4. Wait for CI + SonarCloud to pass.
5. Squash-merge once gates are green.

---

## Output Files

For an input `my_webinar.mp4`, AutoCut produces:

| File | Description |
|---|---|
| `my_webinar_cuts.edl` | CMX 3600 EDL — import into Premiere Pro, DaVinci Resolve, etc. |
| `my_webinar_cuts.json` | Machine-readable cut list with timestamps, labels, confidence scores |
| `my_webinar_cleaned.mp4` | Cleaned video (only if `--output video` or `--output both`); stream-copied by default, re-encoded only with `--crossfade-ms` or `--room-eq` |
| `my_webinar.mp4.autocut-cache.json` | Pipeline cache (auto-managed; not for manual editing) |

**JSON schema:**
```json
{
  "source": "my_webinar.mp4",
  "cuts": [
    {
      "start": 12.34,
      "end": 13.10,
      "duration": 0.76,
      "label": "euh",
      "source": "whisper_filler",
      "confidence": 0.91
    }
  ]
}
```

---

## Detection Sources

| `source` value | Detected by | What it catches |
|---|---|---|
| `vad` | Silero VAD | Silent gaps between speech turns |
| `whisper_filler` | Faster-whisper | Filler words (euh, hm, hm, donc, ben, beh, voilà, eh) and truncated words (ending with `-`) |
| `whisper_repetition` | Faster-whisper | Immediately repeated content words (sliding window check) |

---

## Running Tests

```bash
# All tests
pytest

# Single file
pytest tests/test_cache.py

# Verbose output
pytest -v

# Coverage report
pytest --cov=autocut

# Specific test function
pytest tests/test_merger.py::test_merge_overlapping
```

**Test architecture:** Tests use fixtures for common setup (e.g., mock config, sample audio). Integration tests hit real Whisper/VAD where possible; unit tests mock where needed.

---

## Common Tasks

### Add a new detection stage

1. Create a new module in `src/autocut/pipeline/`.
2. Implement the detection function (takes audio path + config, returns list of `BadSegment`).
3. Add to `runner.py`: call the function and append results to the aggregated list.
4. Update `AutoCutConfig` if new parameters are needed.
5. Add tests in `tests/test_<stage>.py`.

### Adjust a detection threshold

1. Find the parameter in `AutoCutConfig` (`config.py`).
2. Add a CLI option in `main.py` if not already present.
3. Thread it through the pipeline stage that uses it.
4. Test with real webinar footage to verify the threshold works.

### Fix a video cutting issue

1. Most re-encoding issues (frame accuracy, B-frame artifacts, crossfade clicks) go in `cutter.py`.
2. FFmpeg is called via `subprocess.Popen` with strict argument escaping.
3. Progress is reported via `-progress pipe:1` and a callback thread.
4. Check exit codes: 0 = success, non-zero = FFmpeg error (details in stderr).

---

## Troubleshooting

**Whisper model download fails:**
- Models are cached in `~/.cache/huggingface/hub/`.
- `--device cuda` requires CUDA drivers; `--device cpu` is the safe default.

**VAD is too aggressive/too lenient:**
- Adjust `--min-silence-ms` (default 700) to detect shorter/longer silences.
- `--max-silence-s` (default 30) prevents cutting Q&A breaks; set to `None` for YouTube replays where all silence is unwanted.

**Filler detection misses words:**
- Adjust `--min-filler-duration` (default 0.3 s) to catch shorter utterances.
- Add custom filler words with `--fillers euh,ah,um,ergh`.

**Cache stale:**
- Delete the `.autocut-cache.json` sidecar, or use `--no-cache` for a fresh run.

---

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic error (config validation, IO) |
| 2 | Audio extraction failed (FFmpeg) |
| 3 | Unsupported input format (no video or audio stream) |
| 4 | Whisper model load failed |

---

## Key Constants & Defaults

- **Default filler words:** `euh, hm, hmm, donc, ben, beh, voilà, eh` (in `config.py`)
- **Default VAD silence threshold:** 700 ms
- **Default Whisper model:** `small`
- **Default merge gap:** 0.2 s (merge segments closer than 200 ms)
- **Default padding:** 50 ms before and after each cut
- **Default Room EQ Q factor:** 8.0 (narrow notch)
- **Default Room EQ max filters:** 5

---

## Audio Enhancement

AutoCut includes intelligent audio cleanup features that suppress unwanted artifacts during video re-encoding.

### De-esser

Intelligent sibilant suppression using dynamic sidechain compression. Detects peaks in the sibilant range (4–9 kHz), then applies an FFmpeg `sidechaincompress` filter triggered by those frequencies. Only compresses when sibilants peak, preserving natural speech. Threshold (default 8 dB above noise floor) controls detection sensitivity. Enabled by default.

### Click/Plosive Removal

Transient suppressor using fast-attack compression. Detects sharp transients in plosive (20–500 Hz) and click (2–10 kHz) ranges, then applies FFmpeg `compand` filter with 1 ms attack and 100 ms release to gate them out without pumping artifacts. Threshold (default 12 dB above noise floor) controls detection sensitivity. Enabled by default.

### How They Work

- **Detection phase** (pipeline): Audio is analyzed via FFT to find problem frequencies
- **Application phase** (re-encode): FFmpeg applies dynamic compression filters only where needed
- Both features analyze the full audio (not just silences) to catch artifacts throughout the recording

### Configuration & Control

Both features are bypassable via CLI (`--no-deeser`, `--no-click-removal`) or GUI checkboxes in the Audio tab. See `AutoCutConfig` for tunable parameters:

- `deeser_enabled` (default True)
- `deeser_threshold_db` (default 8.0)
- `click_removal_enabled` (default True)
- `click_threshold_db` (default 12.0)

These parameters flow through the pipeline (VAD stage for detection) and are applied in the video cutter (during re-encoding with `cut_video()`).

---

## GUI Implementation Notes

The GUI (`src/autocut/gui/`) runs the pipeline in a background `QThread` (via `worker.py`) to avoid freezing the UI. Key components:

- **`main_window.py`:** Top-level `QMainWindow` with tab widget and progress/log area.
- **`params_panel.py`:** Four `QTabWidget` tabs (Output, Whisper, Detection, Audio) — synced to `AutoCutConfig` fields.
- **`file_drop.py`:** Drag-and-drop file zone + file picker button.
- **`worker.py`:** `QThread` that calls `runner.py` and emits progress/result signals.
- **`app.py`:** `QApplication` lifecycle, exception handling, resource loading.

The **CUDA device option** is auto-disabled at startup if no compatible GPU is detected.

---

## Useful References

- **README.md:** Full CLI reference, examples, output formats.
- **pyproject.toml:** Dependencies, entry points, build config.
- **.ruff.toml:** Lint rules (Python 3.13, line length 100, isort first-party rules).
- **sonar-project.properties:** SonarCloud quality gate settings.
