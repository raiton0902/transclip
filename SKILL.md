---
name: transkrip
description: Local CapCut-grade audio/video transcription with faster-whisper, MediaPipe mouth-movement lip-sync + 10 ms speech-band RMS never-early onset alignment, one-word-per-spawn captions (--word-by-word), conversational coherence & hallucination review (--apply-edits), Indonesian+English bilingual glossary auto-fix, RAM-safe worker auto-capping, batch folder mode, and word-level millisecond JSON export (.srt, .md, .txt, transcript_words.json). Use when transcribing videos/audio, generating CapCut-style word-by-word captions, fixing delayed/early subtitles, or transcribing mixed Indonesian-English speech offline.
---

# TransClip (`transkrip`) — 4-Step Audio-Visual Workflow

Everything runs 100% locally (CPU + `faster-whisper` + `mediapipe` + `opencv` + `numpy` + `ffmpeg`), zero API keys required.
Main script: `D:\ChatGPT\skills\transkrip\transcribe.py`
Local Python venv: `D:\akuvideo\.venv_transkrip\Scripts\python.exe`

```powershell
$env:PYTHONUTF8 = "1"
& "D:\akuvideo\.venv_transkrip\Scripts\python.exe" `
  "D:\ChatGPT\skills\transkrip\transcribe.py" "<video_or_folder>" --outdir "<folder>" `
  --language id --word-by-word `
  --glossary "Term1,Term2,Term3" `
  --model small
```

## Step 1 — Gather Context (Auto-Infer First, Never Leak Meta-Text)

Check any referenced project brief, folder name, or filename for:
1. **Speakers** (names)
2. **Base language** (`--language id` for Indonesian + English code-switching)
3. **Domain terms (`--glossary`)** — brand names, crypto/business/tech jargon.

Pass domain vocabulary via `--glossary` (avoid putting meta-instructions like `"Language: Indonesian"` into `--context`; `transcribe.py` also auto-strips meta-labels so Whisper never hallucinates `"Indonesia"`).

## Step 2 — RAM-Safe Transcription + Audio-Visual Lip-Sync Alignment

- **Auto-RAM guard:** Checks free physical RAM and auto-caps `--workers` to fit safely in 80% of free RAM.
- **Mouth-Movement + Never-Early Audio-Visual Gate (default ON for video):**
  - Tracks speaker lip opening (`MAR`, `jawOpen`, lip velocity) via MediaPipe FaceLandmarker + 250–3800 Hz speech-band 10 ms RMS energy.
  - Enforces a **strict never-early rule**: captions do not spawn until the speaker actually vocalizes (and moves their mouth after pauses), and disappear cleanly on silence. Falls back seamlessly to audio-only during B-roll.
- **Caption modes:**
  - `--word-by-word` — **1 word per spawn** (CapCut pop-in style, best for short-form precision).
  - `--capcut` — tight short phrases (`<= 4` words, `0.25 s` pause gap).
  - Default — natural pause lines (`0.35 s` pause gap, `6.0 s` max).

## Step 3 — Mandatory Conversational Sanity & Hallucination Review (Agent Pass)

After Pass 1 finishes, **ALWAYS read `transcript.md` and `coherence_flags` in `transkrip_report.json`** and ask:
> *"Does every sentence make sense as real human speech / conversation? Are there any gibberish non-words (e.g. `sebrenge`, `melakua`), misheard English terms (`mode` vs `moat`), or hallucinated words the speaker never said?"*

- If your environment can read audio (`read_file` on `.mp3`/`.wav`), verify any ambiguous low-confidence snippet directly against the audio.
- Write an `edits.json` file targeting `idx` from `transcript_words.json`:
  ```json
  [
    {"idx": 5, "op": "delete"},
    {"idx": 11, "op": "split", "words": ["sebenarnya", "nggak", "gitu?"]},
    {"idx": 48, "op": "replace", "text": "moat"}
  ]
  ```
- Re-run with `--apply-edits` to regenerate `.srt`, `.md`, `.txt`, and `transcript_words.json` **with zero timestamp drift**:
  ```powershell
  & "D:\akuvideo\.venv_transkrip\Scripts\python.exe" `
    "D:\ChatGPT\skills\transkrip\transcribe.py" "<video>" --outdir "<folder>" `
    --word-by-word --apply-edits "<folder>\edits.json"
  ```

## Step 4 — Output & Report

Outputs 5 files in `--outdir`:
1. `transcript.srt` — Audio-visual lip-synced subtitles (`--word-by-word`, `--capcut`, or natural pause).
2. `transcript_words.json` — Per-word `idx`, `word`, `start`, `end`, `start_ms`, `end_ms`, `prob`.
3. `transcript.md` — Clean readable markdown.
4. `transcript_timestamped.txt` — `[HH:MM:SS.mmm - HH:MM:SS.mmm]` lines.
5. `transkrip_report.json` — Duration, lip-sync stats, auto-fixes, manual edits, and `coherence_flags`.

Always report the **exact saved file paths**, lip-sync stats, and every conversational fix applied.
