# TransClip — Local CapCut-Style Video Transcription, Lip-Sync Alignment & Bilingual Subtitle Skill

**TransClip** (`transkrip`) is a 100% local, offline AI transcription skill combining `faster-whisper`, **MediaPipe mouth-movement tracking**, **10 ms speech-band RMS onset alignment**, and **conversational coherence verification**. Built for TikTok, Reels, YouTube Shorts, Remotion, CapCut, and Premiere Pro editors who need **never-early, one-word-per-spawn subtitles** and **Indonesian + English code-switching accuracy** with **$0 API costs**.

---

## Why TransClip vs. Raw Whisper?

Raw Whisper (`faster-whisper`) spawns captions too early or 150–400 ms off-beat, hallucinates prompt words (e.g., leaking `"Indonesia"` before `"Indodax"`), outputs gibberish non-words in fast bilingual speech, and can freeze 16 GB laptops. TransClip solves all five:

| Feature | Raw `faster-whisper` | **TransClip** |
|---|---|---|
| **Never-early lip-sync timing** | Attention estimate (drifts early/late) | **MediaPipe mouth-movement (MAR + `jawOpen`) + 10 ms 250–3800 Hz speech-band RMS never-early gate** |
| **Caption styles** | Long 5–10 s blocks | **`--word-by-word`** (1 word per spawn) or **`--capcut`** (≤4 words) |
| **Conversational sanity check** | Leaves gibberish (`sebrenge`, `melakua`) & prompt leaks | **Leak-free prompt + `coherence_flags` + Agent sanity review + `--apply-edits` (zero timestamp drift)** |
| **Bilingual (ID + EN)** | Language flipping / respelled English | **`--language id` + `--glossary` phonetic fuzzy-fix** |
| **Low-RAM safety** | Spawns 4 model copies → swaps/crashes | **Auto-caps `--workers` to 80% of free physical RAM** |
| **Cloud / API cost** | — | **$0 — 100% offline on CPU** |

---

## Install

```bash
# 1. Core dependencies (Python 3.9+ and ffmpeg on PATH)
pip install faster-whisper numpy

# 2. Optional (recommended) for visual mouth-movement lip-sync
pip install opencv-python-headless mediapipe

# 3. Add as an AI coding agent skill
npx skills add raiton0902/transclip
```

---

## 4-Step Workflow

1. **Context Gathering (Agent):** Reads project briefs or asks for speakers, base language, and domain terms. Meta-labels are automatically stripped before building Whisper's `initial_prompt` to prevent prompt-leak hallucinations.
2. **RAM-Safe Transcription:** Extracts 16 kHz mono WAV, checks free physical RAM, auto-caps workers, and transcribes with `faster-whisper` (`word_timestamps=True` + per-word probability).
3. **Audio-Visual Lip-Sync + Conversational Sanity Pass:**
   - **Lip-Sync Never-Early Gate:** Tracks speaker mouth opening (`MAR`, `jawOpen`, lip velocity) via MediaPipe FaceLandmarker + 250–3800 Hz speech-band RMS at 10 ms resolution so captions **never spawn before the speaker speaks**.
   - **Glossary & Non-Word Auto-Fix:** Fixes known Indonesian ASR corruptions and phonetic glossary typos (`ndodax` → `Indodax`).
   - **Conversational Coherence Review (`coherence_flags` → `--apply-edits`):** Flags low-confidence words, prefix-split hallucinations, and weird spellings in `transkrip_report.json`. The agent reviews *"Does this make sense as real conversation?"* and applies `edits.json` without shifting timestamps.
4. **Multi-Format Output:** Writes `transcript.srt`, `transcript_words.json` (`idx`, `word`, `start_ms`, `end_ms`, `prob`), `transcript.md`, `transcript_timestamped.txt`, and `transkrip_report.json`.

---

## Usage Examples

### One-Word-Per-Spawn CapCut Captions with Lip-Sync (Indonesian + English)
```bash
python transcribe.py clip.mp4 --outdir transkrip \
  --language id --word-by-word \
  --glossary "Indodax,Bitcoin,crypto,exchange,blockchain,FTX,moat"
```

### Apply Conversational Sanity Edits (Zero Timestamp Drift)
```bash
python transcribe.py clip.mp4 --outdir transkrip \
  --word-by-word --apply-edits transkrip/edits.json
```

### Batch Folder Mode
```bash
python transcribe.py ./clips_folder --language id --capcut --model small
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--word-by-word` | off | One word per spawn (`1` word per SRT cue, disappears on silence) |
| `--capcut` | off | Tight short-form cues (≤4 words, 0.25 s pause, 2.5 s max) |
| `--glossary` | *(empty)* | Comma-separated terms or path to `glossary.txt` for prompt + fuzzy-fix |
| `--language` | auto | Force base language (`id`, `en`, etc.) to prevent code-switch flipping |
| `--model` | `small` | `tiny` · `base` · `small` · `medium` · `large-v3-turbo` |
| `--no-lip-sync` | off | Disable MediaPipe mouth-movement tracking (uses audio-only gate) |
| `--no-snap-onset` | off | Disable 10 ms speech-band onset alignment |
| `--apply-edits` | *(none)* | Apply JSON edits (`replace`/`delete`/`split`/`merge`) keeping exact timestamps |
| `--workers` | auto-capped | Parallel workers (automatically capped to fit free RAM) |

---

## Frequently Asked Questions (FAQ)

**How does the mouth-movement algorithm work on B-roll or off-camera audio?**
When MediaPipe detects no speaker face in a frame (e.g., B-roll or screen recording), TransClip automatically falls back to its 250–3800 Hz speech-band never-early audio gate for those frames.

**Does TransClip require an OpenAI or Groq API key?**
No. Everything runs 100% locally on your CPU using `faster-whisper`, `mediapipe`, `opencv`, `numpy`, and `ffmpeg`.

**How does TransClip stop gibberish words and hallucinated words?**
TransClip strips meta-labels from Whisper's `initial_prompt`, records per-word probabilities, flags low-confidence or prefix-split hallucinations in `coherence_flags`, and lets the AI agent review the transcript for conversational sense and apply `--apply-edits` cleanly.

---

## License

Commercial Single-Seat License — see [LICENSE](LICENSE). Available on **agensi.io**.
