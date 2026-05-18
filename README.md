# AutoCut

Automatically detect and remove hesitations, fillers, and stutters from webinar videos.

AutoCut combines Voice Activity Detection (Silero VAD) with speech transcription (Whisper) to find regions worth cutting, then exports an EDL file for NLE import, a JSON summary, or a re-encoded cleaned video.

---

## Requirements

- Python 3.13+
- FFmpeg (must be on `PATH`)
- A CUDA GPU is optional but speeds up Whisper significantly

## Installation

```bash
git clone https://github.com/TriYop/AutoCut.git
cd AutoCut
uv sync          # or: pip install -e .
```

## Quick Start

```bash
# Analyse a video and write EDL + JSON cut list
autocut my_webinar.mp4

# Produce a cleaned video file directly
autocut my_webinar.mp4 --output video

# Both EDL and cleaned video in a specific directory
autocut my_webinar.mp4 --output both --output-dir ./out

# French webinar, GPU, verbose output
autocut seminar.mp4 --language fr --device cuda --verbose
```

---

## CLI Reference

```
autocut [OPTIONS] INPUT_FILE
```

### Detection options

| Option | Default | Description |
|---|---|---|
| `--model` | `small` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `--language` | auto | Language code (e.g. `fr`, `en`). Omit to auto-detect. |
| `--device` | `cpu` | Inference device: `cpu` or `cuda` |
| `--min-silence-ms` | `500` | Minimum silence duration to flag (ms) |
| `--max-silence-s` | `30` | Silences longer than this are kept (Q&A breaks, applause…) |
| `--no-silence-cap` | off | Remove the max-silence guard — cut every silence (good for YT replays) |
| `--fillers` | built-in | Comma-separated filler word list override (default: `euh,hm,hmm,donc,ben,beh,voilà,eh`) |
| `--no-repetitions` | off | Disable word-repetition detection |
| `--merge-gap` | `0.2` | Merge bad segments closer than this many seconds |

### Output options

| Option | Default | Description |
|---|---|---|
| `--output` / `-o` | `edl` | Output mode: `edl`, `video`, or `both` |
| `--output-dir` | input directory | Directory for output files |

### Audio quality options

| Option | Default | Description |
|---|---|---|
| `--crossfade-ms` | `0` | Fade-out/in at each cut point (ms). `0` = disabled. ~120 ms recommended |
| `--room-eq` | off | Detect and attenuate room resonance frequencies from silence segments |
| `--room-eq-gain` | `-10.0` | Attenuation applied to each detected resonance (dB, negative = cut) |

### Misc

| Option | Default | Description |
|---|---|---|
| `--verbose` / `-v` | off | Print each detected region with timestamp and label |

---

## Output Files

For an input file `my_webinar.mp4` the following outputs are produced:

| File | Description |
|---|---|
| `my_webinar_cuts.edl` | CMX 3600 EDL — import into Premiere Pro, DaVinci Resolve, etc. |
| `my_webinar_cuts.json` | Machine-readable cut list with timestamps, labels, and confidence scores |
| `my_webinar_cleaned.mp4` | Re-encoded video with bad regions removed (`--output video/both`) |

### JSON cut list schema

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

## Detection Pipeline

```
Input video
    │
    ▼
Audio extraction (FFmpeg → mono 16 kHz WAV)
    │
    ├─► Silero VAD ──────────────────► silence segments
    │
    ├─► faster-whisper ──────────────► filler words (euh, hm, …)
    │                                  word repetitions (sliding window)
    │
    └─► Room EQ analysis (optional) ► resonant frequencies (FFT over VAD silences)
            │
            ▼
        Segment merger (gap-fill + padding)
            │
            ▼
        EDL / JSON / video output
```

### Detection sources

| `source` value | Detected by | What it catches |
|---|---|---|
| `vad` | Silero VAD | Silent gaps between speech turns |
| `whisper_filler` | faster-whisper | Filler words, truncated words (ends with `-`) |
| `whisper_repetition` | faster-whisper | Immediately repeated content words |

---

## Configuration

All detection parameters live in `AutoCutConfig` (`src/autocut/config.py`). The CLI exposes the most useful ones; the remainder can be set programmatically:

```python
from autocut.config import AutoCutConfig
from autocut.pipeline.runner import run
from pathlib import Path
from rich.console import Console

config = AutoCutConfig(
    whisper_model="medium",
    whisper_language="fr",
    vad_min_silence_duration_ms=400,
    crossfade_ms=120,
    room_eq_enabled=True,
)

result = run(Path("my_webinar.mp4"), config, Console())
for seg in result.bad_segments:
    print(seg.segment.start, seg.segment.end, seg.label)
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic error |
| 2 | Audio extraction failed |
| 3 | Unsupported input format |
| 4 | Whisper model load failed |

---

## Development

```bash
uv sync --dev
pytest
ruff check src tests
mypy src
```
