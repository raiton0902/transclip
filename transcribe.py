#!/usr/bin/env python3
"""TransClip (transkrip) — Local CapCut-grade audio/video transcription.

Pipeline:
  1. Context + clean leak-free initial_prompt + bilingual glossary
  2. RAM-safe faster-whisper transcription (word timestamps + word probabilities)
  3. Audio-Visual Lip-Sync Alignment:
     - 250-3800 Hz speech-band 10 ms RMS energy envelope
     - MediaPipe FaceLandmarker mouth-movement tracking (MAR + jawOpen + lip velocity)
     - Strict "never-early" word-level onset/offset gate
  4. Conversational Coherence & Hallucination Detection:
     - Flags low-confidence words, prompt-leak duplicates, and nonsense tokens
     - Supports --apply-edits edits.json to fix words with zero timestamp drift
  5. Outputs: transcript.srt (--word-by-word / --capcut / standard),
     transcript_words.json, transcript.md, transcript_timestamped.txt,
     transkrip_report.json.
"""
import argparse
import ctypes
import difflib
import json
import os
import re
import subprocess
import sys
import urllib.request
import wave
from concurrent.futures import ProcessPoolExecutor

PART_SECONDS = 660  # ~11 min per part
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS

MODEL_RAM_GB = {
    "tiny": 0.6,
    "base": 0.8,
    "small": 1.5,
    "medium": 3.0,
    "large-v3-turbo": 4.8,
    "large-v3": 5.2,
}

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Common Indonesian ASR phonetic corruptions (strictly non-words -> real words)
ID_NONWORD_FIXES = {
    "shukurnya": "Syukurnya",
    "syukurya": "Syukurnya",
    "kusukuri": "kusyukuri",
    "melakua": "melakukan",
    "melakukah": "melakukan",
    "sebrenge": "sebenarnya",
    "sebenerya": "sebenarnya",
    "sebrengge": "sebenarnya",
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def get_free_ram_gb():
    """Return available physical RAM in GB (cross-platform, stdlib only)."""
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / (1024 ** 3)
    elif os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    return 8.0


def safe_worker_count(requested_workers, model_name, num_parts):
    """Auto-cap parallel workers so model copies fit in free physical RAM."""
    if num_parts <= 1:
        return 1, get_free_ram_gb(), MODEL_RAM_GB.get(model_name, 2.0)
    free_gb = get_free_ram_gb()
    per_worker = MODEL_RAM_GB.get(model_name, 2.0)
    max_fit = max(1, int((free_gb * 0.80) // per_worker))
    capped = max(1, min(requested_workers, num_parts, max_fit))
    return capped, free_gb, per_worker


def media_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", path])
    return float(json.loads(out)["format"]["duration"])


def extract_audio(media, outdir):
    wav = os.path.join(outdir, "full_audio.wav")
    run(["ffmpeg", "-y", "-i", media, "-vn", "-ar", "16000", "-ac", "1", wav])
    return wav


def split_audio(wav, outdir, duration):
    n = max(1, -(-int(duration) // PART_SECONDS))
    if n == 1:
        return [wav]
    parts = []
    for i in range(n):
        p = os.path.join(outdir, f"part_{i}.wav")
        run(["ffmpeg", "-y", "-ss", str(i * PART_SECONDS), "-t", str(PART_SECONDS),
             "-i", wav, "-ar", "16000", "-ac", "1", p])
        parts.append(p)
    return parts


def build_clean_prompt(context, glossary_terms):
    """Build a leak-free Whisper initial_prompt.

    Strips meta-labels ('Speakers:', 'Language:', 'Indonesian', 'English', 'Topic:')
    that otherwise bleed into Whisper hallucinations (e.g., 'Indonesia' before 'Indodax').
    """
    terms = list(glossary_terms)
    meta_stop = {
        "speakers", "speaker", "host", "guest", "language", "indonesian",
        "indonesia", "english", "mixed", "with", "terms", "jargon", "topic",
        "bahasa", "pembicara", "topik", "dan", "and",
    }
    if context:
        # Remove meta clauses like "Language: Indonesian mixed with English..."
        cleaned = re.sub(
            r"(?i)\b(speakers?|language|bahasa|pembicara|topic|topik)\s*:\s*", "", context
        )
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']{2,}", cleaned):
            if tok.lower() not in meta_stop and tok not in terms:
                terms.append(tok)
    if not terms:
        return ""
    # Deduplicate preserving order, cap at 30 terms to stay well under 224 tokens
    seen, unique = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return ", ".join(unique[:30]) + "."


def transcribe_part(args):
    part, offset, model_name, language, initial_prompt = args
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    kw = dict(vad_filter=True, word_timestamps=True)
    if initial_prompt:
        kw["initial_prompt"] = initial_prompt
    if language:
        kw["language"] = language
    segments, _info = model.transcribe(part, **kw)
    out = []
    for s in segments:
        words = []
        for w in (s.words or []):
            txt = w.word.strip()
            if txt:
                words.append({
                    "start": round(w.start + offset, 3),
                    "end": round(w.end + offset, 3),
                    "word": txt,
                    "prob": round(float(getattr(w, "probability", 1.0)), 3),
                })
        out.append({
            "start": round(s.start + offset, 3),
            "end": round(s.end + offset, 3),
            "text": s.text.strip(),
            "avg_logprob": round(float(getattr(s, "avg_logprob", 0.0)), 3),
            "no_speech_prob": round(float(getattr(s, "no_speech_prob", 0.0)), 3),
            "words": words,
        })
    return out


def recheck_segment(args):
    """Re-transcribe one severely low-confidence slice with word timestamps filtered to bounds."""
    wav, start, end, model_name, language, initial_prompt = args
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    pad = 0.35
    kw = dict(vad_filter=True, word_timestamps=True)
    if initial_prompt:
        kw["initial_prompt"] = initial_prompt
    if language:
        kw["language"] = language
    clip_start = max(0.0, start - pad)
    segments, _ = model.transcribe(wav, clip_timestamps=[clip_start, end + pad], **kw)
    words = []
    for s in segments:
        for w in (s.words or []):
            txt = w.word.strip()
            if not txt:
                continue
            w_mid = (w.start + w.end) / 2.0
            if (start - 0.12) <= w_mid <= (end + 0.12):
                words.append({
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": txt,
                    "prob": round(float(getattr(w, "probability", 1.0)), 3),
                })
    text = " ".join(w["word"] for w in words).strip()
    return start, end, text, words


def load_speech_band_rms(wav_path, frame_ms=10):
    """Compute 250-3800 Hz speech-band RMS envelope on a 10 ms grid."""
    import numpy as np
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(sr * frame_ms / 1000))
    win = hop * 2  # 20 ms analysis window, 10 ms hop
    n_frames = max(1, (len(samples) - win) // hop + 1)
    if len(samples) < win:
        return np.array([0.0]), frame_ms / 1000.0, 0.01

    hann = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    band = (freqs >= 250.0) & (freqs <= 3800.0)

    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = samples[i * hop : i * hop + win] * hann
        spec = np.abs(np.fft.rfft(seg))
        # Parseval-scaled speech-band RMS
        rms[i] = np.sqrt(np.sum(spec[band] ** 2) / (win * win))

    # Adaptive speech gate: ignore room noise / mic hum
    noise_floor = float(np.percentile(rms, 20))
    speech_thresh = max(0.006, noise_floor * 3.2)
    return rms, frame_ms / 1000.0, speech_thresh


def ensure_face_landmarker_model():
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "transclip")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "face_landmarker.task")
    if not os.path.isfile(model_path) or os.path.getsize(model_path) < 1_000_000:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    return model_path


def extract_lip_activity(video_path, n_audio_frames, dt=0.01, log=print):
    """Track speaker mouth movement per video frame via MediaPipe FaceLandmarker.

    Returns (lip_active_10ms, face_present_10ms, stats_dict) or (None, None, None)
    if video/MediaPipe is unavailable.
    """
    import numpy as np
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        log("lip-sync skipped (install opencv-python-headless mediapipe to enable)")
        return None, None, None

    try:
        model_path = ensure_face_landmarker_model()
    except Exception as e:
        log(f"lip-sync model download failed ({e}); falling back to audio-only")
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=2,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    times, mars, jaws, has_face = [], [], [], []
    frame_idx = 0
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = int(round(frame_idx * 1000.0 / fps))
            # Downscale to 480px width for fast CPU tracking
            h, w = frame.shape[:2]
            if w > 480:
                scale = 480.0 / w
                frame = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, t_ms)

            if res.face_landmarks:
                best_mar, best_jaw = 0.0, 0.0
                for fi, lms in enumerate(res.face_landmarks):
                    # Upper inner lip (13), lower inner lip (14), corners (61, 291)
                    up, lo, lc, rc = lms[13], lms[14], lms[61], lms[291]
                    v_dist = ((up.x - lo.x) ** 2 + (up.y - lo.y) ** 2) ** 0.5
                    h_dist = max(1e-4, ((lc.x - rc.x) ** 2 + (lc.y - rc.y) ** 2) ** 0.5)
                    mar = v_dist / h_dist
                    jaw = 0.0
                    if res.face_blendshapes and fi < len(res.face_blendshapes):
                        for bs in res.face_blendshapes[fi]:
                            if bs.category_name == "jawOpen":
                                jaw = bs.score
                                break
                    if (mar + jaw) > (best_mar + best_jaw):
                        best_mar, best_jaw = mar, jaw
                times.append(t_ms / 1000.0)
                mars.append(best_mar)
                jaws.append(best_jaw)
                has_face.append(1.0)
            else:
                times.append(t_ms / 1000.0)
                mars.append(0.0)
                jaws.append(0.0)
                has_face.append(0.0)
            frame_idx += 1
    cap.release()

    if not times:
        return None, None, None

    t_arr = np.array(times, dtype=np.float32)
    mar_arr = np.array(mars, dtype=np.float32)
    jaw_arr = np.array(jaws, dtype=np.float32)
    face_arr = np.array(has_face, dtype=np.float32)

    # Lip velocity (frame-to-frame mouth motion catches bilabial plosives p/b/m)
    mar_vel = np.abs(np.diff(mar_arr, prepend=mar_arr[0]))
    jaw_vel = np.abs(np.diff(jaw_arr, prepend=jaw_arr[0]))
    vel_arr = mar_vel + 0.5 * jaw_vel

    valid = face_arr > 0.5
    face_ratio = float(np.mean(valid))
    if face_ratio < 0.05:
        log("lip-sync: no speaker face detected (<5% frames); using audio-only onset")
        return None, None, {"face_ratio": round(face_ratio, 3)}

    # Adaptive closed-mouth baseline from bottom 15% of detected face frames
    mar_closed = float(np.percentile(mar_arr[valid], 15))
    jaw_closed = float(np.percentile(jaw_arr[valid], 15))
    vel_thresh = max(0.008, float(np.percentile(vel_arr[valid], 35)))

    active_frame = (
        (mar_arr > mar_closed + 0.025)
        | (jaw_arr > max(0.035, jaw_closed + 0.025))
        | (vel_arr > vel_thresh)
    ).astype(np.float32)

    # Interpolate onto the 10 ms audio timeline
    grid = np.arange(n_audio_frames, dtype=np.float32) * dt
    lip_10ms = np.interp(grid, t_arr, active_frame) >= 0.35
    face_10ms = np.interp(grid, t_arr, face_arr) >= 0.5
    stats = {
        "frames_analyzed": frame_idx,
        "face_ratio": round(face_ratio, 3),
        "lip_active_ratio": round(float(np.mean(lip_10ms[face_10ms])) if np.any(face_10ms) else 0.0, 3),
    }
    log(f"lip-sync: analyzed {frame_idx} frames (face={face_ratio*100:.0f}%, lip_active={stats['lip_active_ratio']*100:.0f}%)")
    return lip_10ms, face_10ms, stats


def align_words_audiovisual(words, wav_path, video_path=None, use_lip_sync=True, log=print):
    """Strict never-early word-level alignment fusing 10 ms speech-band RMS + lip movement."""
    import numpy as np
    rms, dt, speech_thresh = load_speech_band_rms(wav_path, frame_ms=10)
    n_frames = len(rms)
    total_dur = n_frames * dt

    lip_10ms, face_10ms, lip_stats = None, None, None
    if use_lip_sync and video_path and os.path.splitext(video_path)[1].lower() in VIDEO_EXTS:
        lip_10ms, face_10ms, lip_stats = extract_lip_activity(video_path, n_frames, dt=dt, log=log)

    def lip_ok(idx):
        if lip_10ms is None or not face_10ms[idx]:
            return True
        # Allow +-40 ms (4 frames) visual co-articulation tolerance around audio onset
        lo = max(0, idx - 4)
        hi = min(n_frames - 1, idx + 4)
        return bool(np.any(lip_10ms[lo : hi + 1]))

    aligned = []
    shifts_ms = []
    for i, w in enumerate(words):
        orig_start = float(w["start"])
        orig_end = float(w["end"])
        prev_end = aligned[-1]["end"] if aligned else 0.0
        next_start = float(words[i + 1]["start"]) if i + 1 < len(words) else total_dur
        after_pause = (orig_start - prev_end) >= 0.12 or i == 0

        # Crisp offset: pull word end back out of trailing silence
        end_lo = max(0, int(max(prev_end, orig_start) / dt))
        end_hi = min(n_frames - 1, int(min(next_start, orig_end + 0.06) / dt))
        new_end = min(next_start, max(orig_start + 0.06, orig_end))
        offset_thresh = speech_thresh * 0.75
        last_speech_idx = end_hi
        for idx in range(end_hi, end_lo - 1, -1):
            if rms[idx] >= offset_thresh:
                last_speech_idx = idx
                new_end = min(next_start, (idx + 2) * dt)
                break

        # Anti-Pause-Stretch Rule: if Whisper stretched a single word across a long
        # silence/hesitation (> 0.90s), trace backwards from the final speech burst at
        # last_speech_idx to find where the actual spoken word begins (never early!).
        if (new_end - orig_start) > 0.90:
            silent_run = 0
            burst_start = last_speech_idx
            for idx in range(last_speech_idx, end_lo - 1, -1):
                if rms[idx] >= speech_thresh and (lip_ok(idx) if after_pause else True):
                    silent_run = 0
                    burst_start = idx
                else:
                    silent_run += 1
                    # >= 150 ms of true silence/closed-mouth (longer than voiceless consonant stops)
                    if silent_run >= 15 and (last_speech_idx - burst_start) >= 12:
                        break
            new_start = max(prev_end, burst_start * dt)
        else:
            # NEVER-EARLY rule: search strictly from orig_start forward (never pull earlier!)
            search_lo = max(0, int(max(prev_end, orig_start) / dt))
            search_hi = min(n_frames - 1, int(min(new_end - 0.04, orig_start + 0.35) / dt))
            new_start = max(prev_end, orig_start)
            found = False
            for idx in range(search_lo, search_hi + 1):
                if rms[idx] >= speech_thresh:
                    if (not after_pause) or lip_ok(idx):
                        new_start = idx * dt
                        found = True
                        break
            if not found:
                for idx in range(search_lo, search_hi + 1):
                    if rms[idx] >= speech_thresh:
                        new_start = idx * dt
                        break

        new_start = max(prev_end, min(new_start, next_start - 0.04))
        new_end = min(next_start, max(new_start + 0.06, new_end))
        shifts_ms.append(int(round((new_start - orig_start) * 1000)))

        nw = dict(w)
        nw["start"] = round(new_start, 3)
        nw["end"] = round(new_end, 3)
        aligned.append(nw)

    avg_shift = round(sum(shifts_ms) / max(1, len(shifts_ms)), 1)
    max_fwd = max(shifts_ms) if shifts_ms else 0
    log(f"audio-visual never-early align: {len(aligned)} words (avg shift +{avg_shift} ms, max +{max_fwd} ms)")
    return aligned, lip_stats


def load_glossary(glossary_arg):
    if not glossary_arg:
        return []
    if os.path.isfile(glossary_arg):
        with open(glossary_arg, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return [t.strip() for t in glossary_arg.split(",") if t.strip()]


def apply_glossary_and_nonword_fixes(words, glossary_terms):
    """Fix glossary misspellings + known Indonesian ASR non-word corruptions."""
    fixes = []
    phonetic = lambda s: s.lower().replace("ks", "x").replace("ph", "f").replace("oe", "u")
    term_map = {t.lower(): t for t in glossary_terms}
    phon_map = {phonetic(t): t for t in glossary_terms}

    for i, w in enumerate(words):
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", w["word"])
        if len(clean) < 3:
            continue
        low = clean.lower()

        # 1. Known Indonesian non-word ASR corruptions
        if low in ID_NONWORD_FIXES:
            canonical = ID_NONWORD_FIXES[low]
            if clean[0].islower():
                canonical = canonical.lower()
            fixes.append({"idx": i, "start": w["start"], "old": clean, "new": canonical, "rule": "id_nonword"})
            w["word"] = w["word"].replace(clean, canonical)
            continue

        # 2. Glossary exact/case/fuzzy match
        if not glossary_terms:
            continue
        if low in term_map:
            if clean != term_map[low]:
                w["word"] = w["word"].replace(clean, term_map[low])
            continue
        cutoff = 0.72 if len(clean) >= 5 else 0.82
        matches = difflib.get_close_matches(phonetic(clean), phon_map.keys(), n=1, cutoff=cutoff)
        if matches:
            canonical = phon_map[matches[0]]
            fixes.append({"idx": i, "start": w["start"], "old": clean, "new": canonical, "rule": "glossary"})
            w["word"] = w["word"].replace(clean, canonical)
    return fixes


def analyze_coherence(words, glossary_terms):
    """Flag words that fail conversational coherence checks for Agent Step 3b review.

    Detects:
      - Prompt-leak / prefix-split hallucinations (e.g., 'Indonesia' right before 'Indodax')
      - Low Whisper word probability (prob < 0.65)
      - Rare/garbled consonant clusters (likely non-word ASR hallucinations)
    """
    flags = []
    gloss_low = [g.lower() for g in glossary_terms]
    for i, w in enumerate(words):
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", w["word"]).lower()
        prob = float(w.get("prob", 1.0))
        reasons = []

        # Prefix-split hallucination: adjacent word shares 4+ char prefix with this word
        if i + 1 < len(words):
            nxt = re.sub(r"^[^\w]+|[^\w]+$", "", words[i + 1]["word"]).lower()
            if len(clean) >= 4 and len(nxt) >= 4 and clean[:4] == nxt[:4] and clean != nxt:
                reasons.append(f"possible_split_or_prompt_leak_before_{nxt}")
        if clean == "indonesia" and i + 1 < len(words):
            nxt = re.sub(r"^[^\w]+|[^\w]+$", "", words[i + 1]["word"]).lower()
            if any(nxt.startswith(g[:4]) or g in nxt for g in gloss_low if g.startswith("indo")):
                reasons.append("likely_prompt_leak_before_indodax")

        if prob < 0.65:
            reasons.append(f"low_confidence({prob:.2f})")

        # Rare consonant cluster heuristic (e.g., 'brng', 'srng', 'kua' artifacts)
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", clean) and clean not in gloss_low:
            reasons.append("unusual_spelling")

        if reasons:
            lo_ctx = max(0, i - 4)
            hi_ctx = min(len(words), i + 5)
            ctx_sent = " ".join(x["word"] for x in words[lo_ctx:hi_ctx])
            flags.append({
                "idx": i,
                "start": w["start"],
                "word": w["word"],
                "prob": prob,
                "reasons": reasons,
                "context": ctx_sent,
            })
    return flags


def group_words_into_segments(words, gap_threshold=0.35, max_dur=6.0, min_dur=1.0, max_words=0):
    """Group aligned words into SRT/MD cues (supports --word-by-word, --capcut, or natural pause)."""
    if not words:
        return []
    if max_words == 1:
        return [
            {"start": w["start"], "end": w["end"], "text": w["word"], "words": [w]}
            for w in words
        ]

    chunks, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        gap = w["start"] - prev["end"]
        cur_dur = w["end"] - cur[0]["start"]
        hit_words = max_words > 0 and len(cur) >= max_words
        if gap > gap_threshold or cur_dur > max_dur or hit_words:
            chunks.append(cur)
            cur = [w]
        else:
            cur.append(w)
    chunks.append(cur)

    merged = []
    for c in chunks:
        if min_dur > 0 and merged and (c[-1]["end"] - c[0]["start"]) < min_dur:
            if max_words == 0 or (len(merged[-1]) + len(c) <= max_words):
                if len(" ".join(x["word"] for x in merged[-1])) + len(" ".join(x["word"] for x in c)) < 120:
                    merged[-1].extend(c)
                    continue
        merged.append(c)

    return [
        {
            "start": c[0]["start"],
            "end": c[-1]["end"],
            "text": " ".join(x["word"] for x in c).strip(),
            "words": c,
        }
        for c in merged
    ]


def apply_manual_edits(words, edits):
    """Apply conversational sanity edits without losing audio-visual timestamps.

    Edit item schema:
      {"idx": 5, "op": "replace", "text": "sebenarnya"}
      {"idx": 5, "op": "delete"}
      {"idx": 5, "op": "split", "words": ["sebenarnya", "nggak", "gitu?"]}
      {"idx": 5, "end_idx": 6, "op": "merge", "text": "Indodax"}
    """
    by_idx = {}
    skip = set()
    for e in edits:
        idx = int(e["idx"])
        op = e.get("op", "replace")
        by_idx[idx] = e
        if op == "merge" and "end_idx" in e:
            for k in range(idx + 1, int(e["end_idx"]) + 1):
                skip.add(k)

    out = []
    for i, w in enumerate(words):
        if i in skip:
            continue
        if i not in by_idx:
            out.append(dict(w))
            continue
        e = by_idx[i]
        op = e.get("op", "replace")
        if op == "delete":
            continue
        elif op == "replace":
            nw = dict(w)
            nw["word"] = e["text"]
            nw["prob"] = 1.0
            out.append(nw)
        elif op == "merge":
            end_i = min(len(words) - 1, int(e.get("end_idx", i)))
            nw = dict(w)
            nw["word"] = e["text"]
            nw["end"] = words[end_i]["end"]
            nw["prob"] = 1.0
            out.append(nw)
        elif op == "split":
            parts = [p for p in e.get("words", []) if p.strip()]
            if not parts:
                continue
            total_chars = sum(max(1, len(p)) for p in parts)
            dur = max(0.06, float(w["end"]) - float(w["start"]))
            t = float(w["start"])
            for p in parts:
                seg_d = dur * (max(1, len(p)) / total_chars)
                out.append({
                    "word": p.strip(),
                    "start": round(t, 3),
                    "end": round(t + seg_d, 3),
                    "prob": 1.0,
                })
                t += seg_d
    return out


def fmt_ts(sec, ms_precision=False):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if ms_precision:
        ms = min(999, int(round((sec - int(sec)) * 1000)))
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_mmss(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def srt_time(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = min(999, int(round((sec - int(sec)) * 1000)))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(outdir, words, args, meta):
    """Write transcript.srt, transcript_words.json, transcript.md, transcript_timestamped.txt, report."""
    # Re-index words cleanly
    indexed_words = []
    for idx, w in enumerate(words):
        indexed_words.append({
            "idx": idx,
            "word": w["word"],
            "start": round(float(w["start"]), 3),
            "end": round(float(w["end"]), 3),
            "start_ms": int(round(float(w["start"]) * 1000)),
            "end_ms": int(round(float(w["end"]) * 1000)),
            "prob": round(float(w.get("prob", 1.0)), 3),
        })

    if args.word_by_word:
        srt_segs = group_words_into_segments(indexed_words, max_words=1, min_dur=0.0)
        mode_label = "word-by-word"
    elif args.capcut:
        max_w = args.max_words if args.max_words > 0 else 4
        srt_segs = group_words_into_segments(
            indexed_words, gap_threshold=0.25, max_dur=2.5, min_dur=0.0, max_words=max_w
        )
        mode_label = "capcut"
    else:
        srt_segs = group_words_into_segments(
            indexed_words, gap_threshold=args.pause_gap, max_dur=args.max_seg,
            min_dur=args.min_dur, max_words=args.max_words
        )
        mode_label = "natural-pause"

    # Always make transcript.md readable at natural pauses (even in word-by-word SRT mode)
    md_segs = group_words_into_segments(indexed_words, gap_threshold=0.35, max_dur=6.0, min_dur=0.8)

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    lines_md, lines_txt, srt = [], [], []
    for i, s in enumerate(md_segs, 1):
        stamp = fmt_mmss(s["start"])
        if speakers:
            speaker = speakers[(i - 1) % len(speakers)]
            lines_md.append(f"**[{stamp}] {speaker}**: {s['text']}")
        else:
            lines_md.append(f"**[{stamp}]** {s['text']}")
        lines_txt.append(f"[{fmt_ts(s['start'], ms_precision=True)} - {fmt_ts(s['end'], ms_precision=True)}] {s['text']}")

    for i, s in enumerate(srt_segs, 1):
        srt.append(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n")

    with open(os.path.join(outdir, "transcript.md"), "w", encoding="utf-8") as f:
        f.write("# Transkrip\n\n" + "\n\n".join(lines_md) + "\n")
    with open(os.path.join(outdir, "transcript_timestamped.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_txt) + "\n")
    with open(os.path.join(outdir, "transcript.srt"), "w", encoding="utf-8") as f:
        f.write("\n".join(srt) + "\n")
    with open(os.path.join(outdir, "transcript_words.json"), "w", encoding="utf-8") as f:
        json.dump(indexed_words, f, ensure_ascii=False, indent=2)

    avg_cue = sum(s["end"] - s["start"] for s in srt_segs) / max(1, len(srt_segs))
    report = dict(meta)
    report.update({
        "mode": mode_label,
        "srt_cues": len(srt_segs),
        "words": len(indexed_words),
        "avg_cue_s": round(avg_cue, 3),
    })
    with open(os.path.join(outdir, "transkrip_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return len(srt_segs), len(indexed_words)


def process_file(media_path, args):
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(media_path)), "transkrip")
    os.makedirs(outdir, exist_ok=True)
    log = lambda msg: print(f"[transkrip] {msg}", flush=True)

    # Fast path: apply conversational sanity edits to existing transcript_words.json
    if args.apply_edits:
        words_file = os.path.join(outdir, "transcript_words.json")
        if not os.path.isfile(words_file):
            raise FileNotFoundError(f"transcript_words.json not found in {outdir}")
        with open(words_file, encoding="utf-8") as f:
            words = json.load(f)
        with open(args.apply_edits, encoding="utf-8") as f:
            edits = json.load(f)
        updated = apply_manual_edits(words, edits)
        report_path = os.path.join(outdir, "transkrip_report.json")
        meta = json.load(open(report_path, encoding="utf-8")) if os.path.isfile(report_path) else {"media": media_path}
        meta["manual_edits_applied"] = len(edits)
        meta["coherence_flags"] = analyze_coherence(updated, load_glossary(args.glossary))
        cues, n_words = write_outputs(outdir, updated, args, meta)
        log(f"applied {len(edits)} edit(s) -> {cues} SRT cues, {n_words} words in {outdir}")
        return outdir

    glossary_terms = load_glossary(args.glossary)
    initial_prompt = build_clean_prompt(args.context, glossary_terms)
    log(f"clean initial_prompt: {initial_prompt or '(none)'}")

    duration = media_duration(media_path)
    log(f"media: {os.path.basename(media_path)} ({duration:.1f}s)")
    wav = extract_audio(media_path, outdir)
    parts = split_audio(wav, outdir, duration)

    workers, free_gb, per_w = safe_worker_count(args.workers, args.model, len(parts))
    if len(parts) > 1:
        log(f"RAM check: {free_gb:.1f} GB free ({per_w:.1f} GB/worker) -> workers={workers}")
    log(f"split into {len(parts)} part(s); model={args.model}; language={args.language or 'auto'}; workers={workers}")

    jobs = [(p, i * PART_SECONDS, args.model, args.language, initial_prompt)
            for i, p in enumerate(parts)]
    if workers == 1:
        results = [transcribe_part(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(transcribe_part, jobs))

    segments = []
    for r in sorted(results, key=lambda r: r[0]["start"] if r else 0):
        segments.extend(r)
    segments.sort(key=lambda s: s["start"])

    # Step 3a: Re-check severely low-confidence segments only (bounds-filtered)
    corrections = []
    suspicious = [s for s in segments if s["avg_logprob"] < -0.95 or s["no_speech_prob"] > 0.70]
    log(f"re-check: {len(suspicious)} suspicious segment(s)")
    if suspicious:
        re_jobs = [(wav, s["start"], s["end"], args.model, args.language, initial_prompt) for s in suspicious]
        re_results = [recheck_segment(j) for j in re_jobs] if workers == 1 else list(
            ProcessPoolExecutor(max_workers=workers).map(recheck_segment, re_jobs)
        )
        for (start, end, new_text, new_words), old in zip(re_results, suspicious):
            if new_text and new_text.lower() != old["text"].lower() and new_words:
                if len(new_text) >= max(4, len(old["text"]) // 2):
                    corrections.append({"start": start, "end": end, "old": old["text"], "new": new_text})
                    old["text"] = new_text
                    old["words"] = new_words

    # Flatten to word stream
    words = []
    for s in segments:
        for w in (s.get("words") or []):
            words.append(w)

    # Step 3b: Glossary & Indonesian non-word auto-fixes
    auto_fixes = apply_glossary_and_nonword_fixes(words, glossary_terms)
    if auto_fixes:
        log(f"auto-fixed {len(auto_fixes)} glossary/non-word term(s)")

    # Step 3c: Audio-Visual Lip-Sync + Strict Never-Early Word Alignment
    lip_stats = None
    if not args.no_snap_onset and words:
        words, lip_stats = align_words_audiovisual(
            words, wav, video_path=media_path, use_lip_sync=(not args.no_lip_sync), log=log
        )

    # Step 3d: Conversational Coherence & Hallucination Analysis
    coherence_flags = analyze_coherence(words, glossary_terms)
    log(f"coherence check: {len(coherence_flags)} word(s) flagged for review")

    meta = {
        "media": media_path,
        "duration_s": round(duration, 2),
        "waveform_snapped": not args.no_snap_onset,
        "lip_sync_enabled": (not args.no_lip_sync) and (lip_stats is not None),
        "lip_sync_stats": lip_stats,
        "corrections": corrections,
        "auto_fixes": auto_fixes,
        "coherence_flags": coherence_flags,
    }
    cues, n_words = write_outputs(outdir, words, args, meta)

    if not args.keep_parts:
        for p in parts:
            if p != wav and os.path.exists(p):
                os.remove(p)
        if os.path.exists(wav):
            try:
                os.remove(wav)
            except OSError:
                pass

    log(f"done -> {outdir} ({cues} SRT cues, {n_words} words)")
    return outdir


def main():
    ap = argparse.ArgumentParser(description="TransClip — Local CapCut-grade Audio-Visual Transcription")
    ap.add_argument("media", help="Video/audio file or folder (batch mode)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--context", default="", help="Domain keywords / context (meta-labels auto-stripped)")
    ap.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3-turbo")
    ap.add_argument("--language", default=None, help="Force language code, e.g. id, en")
    ap.add_argument("--speakers", default="", help="Comma-separated speaker names, optional")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--pause-gap", type=float, default=0.35)
    ap.add_argument("--max-seg", type=float, default=6.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-words", type=int, default=0, help="Max words per SRT cue (0 = unlimited)")
    ap.add_argument("--capcut", action="store_true", help="Short-form cues (<=4 words, 0.25s pause)")
    ap.add_argument("--word-by-word", action="store_true", help="One word per spawn (CapCut pop-in style)")
    ap.add_argument("--glossary", default="", help="Comma-separated terms or path to glossary.txt")
    ap.add_argument("--no-snap-onset", action="store_true", help="Disable 10ms speech-band onset alignment")
    ap.add_argument("--no-lip-sync", action="store_true", help="Disable MediaPipe mouth-movement tracking")
    ap.add_argument("--apply-edits", default=None, help="Apply edits.json to transcript_words.json without re-transcribing")
    ap.add_argument("--force", action="store_true", help="Re-transcribe in batch mode even if SRT exists")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    target = os.path.abspath(args.media)
    if os.path.isdir(target):
        files = sorted(
            os.path.join(target, fn) for fn in os.listdir(target)
            if os.path.splitext(fn)[1].lower() in MEDIA_EXTS
        )
        print(f"[transkrip] batch mode: {len(files)} media file(s) in {target}", flush=True)
        for fp in files:
            base = os.path.splitext(os.path.basename(fp))[0]
            sub_out = os.path.join(args.outdir or target, f"{base}_transkrip")
            if not args.force and os.path.exists(os.path.join(sub_out, "transcript.srt")):
                print(f"[transkrip] skip (exists): {os.path.basename(fp)}", flush=True)
                continue
            saved_outdir = args.outdir
            args.outdir = sub_out
            process_file(fp, args)
            args.outdir = saved_outdir
    else:
        process_file(target, args)


if __name__ == "__main__":
    main()
