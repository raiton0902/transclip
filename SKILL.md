---
name: transkrip
description: Transcribe audio/video locally with faster-whisper following a 4-step workflow — gather context from the user first, transcribe with a context-matched initial_prompt, run an on-point re-check pass to fix wrong words, then output transcript.md/.txt/.srt. Use when the user asks to transcribe a video or audio file, fix auto-generated captions, or produce subtitles in Indonesian (or any other language).
---

# Transkrip — 4-step workflow

Everything runs locally (faster-whisper on CPU), no cloud involved. Main script:
`D:\ChatGPT\skills\transkrip\transcribe.py`

```powershell
$env:PYTHONUTF8 = "1"
& "D:\ChatGPT\skills\lemo-opuscar\.venv\Scripts\python.exe" `
  "D:\ChatGPT\skills\transkrip\transcribe.py" <video/audio> --outdir <folder> `
  --context "Speakers: Suli (host) and William (Indodax CEO). Language: Indonesian mixed with English crypto jargon. Topic: Bitcoin, crypto, Indodax." `
  --model small
```

(faster-whisper 1.2.1 is already installed in the lemo-opuscar venv; ffmpeg is on PATH.)

## Step 1 — Ask the user for context (REQUIRED before transcribing)

Before running anything, ask the user (via ask_user_question / chat):

1. **People / speakers** — who speaks in the video and their role (host/guest).
2. **Language used** — pure Indonesian, mixed with English, regional dialect, etc.
3. **Video topic** — the domain of technical terms (crypto, tech, health, ...).

All of this is optional: **if the user doesn't know or doesn't answer, proceed
without context** (never block on it). The context is used to build the Whisper
`initial_prompt` — this is the key to keeping domain terms spelled correctly
(e.g. `captcha`, `HODL`, `XAUT`, `conviction`).

Compose the initial_prompt as one or two sentences: mention the speakers' names
plus the main domain terms.

## Step 2 — Transcription (small model, parallel on CPU)

- Default model: `small` (fast, low RAM, accurate enough for normal podcasts).
- Use `--model large-v3-turbo` if the user wants maximum accuracy (≈8× slower on CPU).
- `medium` is the middle ground.
- Audio is extracted via ffmpeg (mono 16 kHz), split into ~11-minute parts, and
  transcribed **in parallel** (one process per part, 4 CPU threads each) with
  `vad_filter=True`, `word_timestamps=True`, and the `initial_prompt` from Step 1.
- Each part's offset is added back so timestamps stay absolute relative to the
  original video.
- **Per natural pause segmentation**: the result is then split at the speaker's
  natural pauses using word-level timestamps — a silence gap between two words
  longer than `--pause-gap` (default 0.35 s) starts a new line, and a line is
  force-broken past `--max-seg` (default 6 s). A single long monologue by the
  same speaker is therefore cut into short lines of ~2–5 s, **each with its own
  timestamp** — segments do NOT wait for a speaker change to turn over.

## Step 3 — On-point re-check (fix wrong words)

Once the raw transcript exists, run a verification pass (automated in the script):

1. **Suspicious segments**: segments with low `avg_logprob` or high
   `no_speech_prob` are re-sliced and re-transcribed; the old and new results are
   compared — if they differ significantly, the re-sliced version wins (a short
   transcription with the same context prompt is usually more accurate).
2. **Glossary check**: important terms from the user's context are checked for
   occurrence; phonetically similar misspellings are reported in the corrections.
3. **Part boundaries**: artifacts at part joins (cut words / duplicates) are
   cleaned up during the merge.

For deeper corrections (person names, numbers), the agent may verify manually by
listening to / referencing the video — record every correction in the report.

## Step 4 — Output

Write 3 files to `--outdir` (default: a `transkrip/` folder next to the media),
**one line per natural pause** (~2–5 s per line, avg ≈3–4 s across a full video):

1. `transcript.md` — clean markdown, optional chapter headings, dialog lines as
   `**[MM:SS - MM:SS] Speaker**:` (speaker labels only when the user provided a
   list of people; otherwise no labels).
2. `transcript_timestamped.txt` — plain text with per-pause timestamps
   `[MM:SS - MM:SS] Speaker: ...`.
3. `transcript.srt` — standard SRT subtitles with millisecond timestamps per
   short pause, ready to import into editing software (CapCut, Premiere,
   DaVinci Resolve, ...).

Report back to the user: duration summary, segment count, and the list of
important corrections made in Step 3 (time → before → after).

## Requirements & resource check (verify BEFORE running)

The model is downloaded automatically on first run (HuggingFace cache, typically
`%USERPROFILE%\.cache\huggingface`). Check the machine can actually fit it
before starting, and tell the user what you checked:

| Model | Download size (disk) | RAM in use (int8) | Speed on CPU (1 min audio) |
|---|---|---|---|
| `tiny` | ~75 MB | ~0.5 GB | ~5 s |
| `base` | ~145 MB | ~0.7 GB | ~8 s |
| `small` (default) | ~484 MB | ~1.3 GB | ~20 s |
| `medium` | ~1.5 GB | ~2.7 GB | ~1 min |
| `large-v3-turbo` | ~1.6 GB | ~4.5 GB | ~3 min |

Rules of thumb:

- **Disk**: free space ≥ 2× the model download size (download + unpacking).
- **RAM**: free RAM ≥ the model's "RAM in use" × 1.5, **plus ~1.5 GB per
  parallel worker** (`--workers`), since each worker loads its own model copy.
  E.g. `small` with 4 workers ⇒ budget ~5 GB free RAM.
- **Never exceed the machine's physical RAM** — int8 whisper models swap badly;
  if free RAM is tight, lower `--workers` first, then drop the model size.
- Quick check from the agent: `Get-CimInstance Win32_PhysicalMemory` (installed),
  `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory` (free),
  and `Get-PSDrive` for disk space. Do this **before** the first run and report
  the numbers.
- `faster-whisper` 1.2.1 + ffmpeg are already installed on this machine (venv:
  `D:\ChatGPT\skills\lemo-opuscar\.venv\Scripts\python.exe`); a fresh machine
  needs Python 3.8+ and `pip install faster-whisper`.

If the chosen model doesn't fit, say so and offer the next smaller tier rather
than silently swapping/thrashing.

## Notes

- Don't leave temp files outside the outdir; part WAVs are deleted when done
  (the script does this automatically unless `--keep-parts` is passed).
