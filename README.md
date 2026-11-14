# Transclip

Local audio/video transcription skill following a fixed 4-step workflow:
**context → transcribe → re-check → output**. No cloud, no API — everything runs
on your machine with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
on CPU. Best for support agent when clipping long form video into short form, without losing any context of audio content.  

## Install

```bash
npx skills add raiton0902/transclip
```

## Why it exists

Raw speech-to-text output is full of mis-heard words — especially domain jargon,
names, and code-switched phrases (e.g. Indonesian podcasts sprinkled with
English crypto terms). This workflow fixes that by (a) seeding the model with
user-provided context *before* transcribing and (b) re-checking low-confidence
segments on-point after the first pass.

## Workflow

| Step | What happens | Who does it |
|------|--------------|-------------|
| 1 | Ask the user for context: speakers, language, topic. Optional — proceed without it if unanswered. | Agent |
| 2 | Extract audio (ffmpeg, mono 16 kHz), split into ~11-min parts, transcribe in parallel with faster-whisper + `initial_prompt` built from the context, **then split per natural pause** (word-level gaps > 0.35 s start a new ~2–5 s line, force-break at 6 s — the same speaker keeps talking while lines still turn over). | `transcribe.py` |
| 3 | Re-check on-point: re-transcribe suspicious segments (low `avg_logprob` / high `no_speech_prob`) with tighter slices, compare, keep the better result; glossary check on context terms; clean part-boundary artifacts. | `transcribe.py` (+ agent for names/numbers) |
| 4 | Write `transcript.md`, `transcript_timestamped.txt`, `transcript.srt`, and `transkrip_report.json` (list of corrections: time → before → after). | `transcribe.py` |

## Usage

```powershell
$env:PYTHONUTF8 = "1"
& "D:\ChatGPT\skills\lemo-opuscar\.venv\Scripts\python.exe" `
  "D:\ChatGPT\skills\transkrip\transcribe.py" video.mp4 --outdir transkrip `
  --context "Speakers: Suli (host) and William (Indodax CEO). Language: Indonesian + English crypto jargon. Topic: Bitcoin, crypto." `
  --model small
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--outdir` | `transkrip/` next to the media | Output folder |
| `--context` | *(empty)* | Speakers / language / topic — becomes the `initial_prompt` |
| `--model` | `small` | `small` (fast) · `medium` · `large-v3-turbo` (most accurate, ≈8× slower on CPU) |
| `--language` | auto-detect | Force a language code, e.g. `id` |
| `--speakers` | *(empty)* | Comma-separated speaker names for labeled transcripts |
| `--workers` | 4 | Parallel transcription processes |
| `--pause-gap` | `0.35` | Silence (s) between two words that starts a new line — the natural-pause threshold |
| `--max-seg` | `6.0` | Force a line break when a segment exceeds this many seconds |
| `--keep-parts` | off | Keep intermediate part WAVs for debugging |

## Outputs

All three transcript formats are **per natural pause**: one line per short
sentence/pause (~2–5 s, average ≈3–4 s across a full video), each with its own
timestamp — the same speaker's monologue is cut into short lines, and lines do
not wait for a speaker change to turn over.

- **`transcript.md`** — clean markdown, timestamped per-pause lines.
- **`transcript_timestamped.txt`** — plain text, `[HH:MM:SS - HH:MM:SS]` per segment.
- **`transcript.srt`** — standard SubRip subtitles.
- **`transkrip_report.json`** — duration, segment count, corrections applied,
  and any context terms that never appeared (potential misspellings to review).

## Requirements

### Already satisfied on this machine

- Python with **faster-whisper ≥ 1.2.1** — `D:\ChatGPT\skills\lemo-opuscar\.venv\Scripts\python.exe`
- **ffmpeg** on PATH

### On a fresh machine

- Python 3.8+ and `pip install faster-whisper`
- ffmpeg on PATH

### Model download size & RAM (check BEFORE first run)

The model downloads automatically on first run into the HuggingFace cache
(`%USERPROFILE%\.cache\huggingface`):

| Model | Download size (disk) | RAM in use (int8) | Speed on CPU (1 min audio) |
|---|---|---|---|
| `tiny` | ~75 MB | ~0.5 GB | ~5 s |
| `base` | ~145 MB | ~0.7 GB | ~8 s |
| `small` (default) | ~484 MB | ~1.3 GB | ~20 s |
| `medium` | ~1.5 GB | ~2.7 GB | ~1 min |
| `large-v3-turbo` | ~1.6 GB | ~4.5 GB | ~3 min |

Budgeting rules:

- **Disk**: keep ≥ 2× the model's download size free (download + unpack).
- **RAM**: free RAM ≥ model RAM × 1.5, **plus ~1.5 GB per parallel worker** —
  each worker loads its own copy of the model. Example: `small` with 4 workers
  ⇒ budget ~5 GB free RAM.
- If free RAM is tight, reduce `--workers` first, then drop the model tier.
  Never exceed physical RAM — swapped-out whisper models are painfully slow.
- Quick PowerShell check: `Get-CimInstance Win32_PhysicalMemory`,
  `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory`,
  `Get-PSDrive` for disk free space.

## Tips

- The more specific the `--context`, the fewer mis-heard words. One or two
  sentences naming the speakers and the domain's key terms is ideal.
- For maximum accuracy on hard audio (heavy accents, overlapping speech), run
  `--model large-v3-turbo` — it is slow on CPU, so budget time accordingly.
- Speaker labels in the automated output are naive (name rotation). For accurate
  speaker attribution, the agent should refine the transcript using visual
  inspection or dialogue context after Step 3.
