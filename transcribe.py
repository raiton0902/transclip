#!/usr/bin/env python3
"""Transkrip skill — 4-step workflow (context -> transcribe -> re-check -> output).

Usage:
  python transcribe.py media.mp4 --outdir transkrip \
      --context "Pembicara: Suli & William. Bahasa: Indonesia. Topik: kripto." \
      --model small [--language id] [--workers 4] [--keep-parts]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

PART_SECONDS = 660  # ~11 min per part


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


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
    parts = []
    for i in range(n):
        p = os.path.join(outdir, f"part_{i}.wav")
        run(["ffmpeg", "-y", "-ss", str(i * PART_SECONDS), "-t", str(PART_SECONDS),
             "-i", wav, "-ar", "16000", "-ac", "1", p])
        parts.append(p)
    return parts


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
            words.append({"start": w.start + offset, "end": w.end + offset,
                          "word": w.word.strip()})
        out.append({
            "start": s.start + offset, "end": s.end + offset,
            "text": s.text.strip(),
            "avg_logprob": getattr(s, "avg_logprob", 0.0),
            "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
            "words": words,
        })
    return out


def split_by_pause(segments, gap_threshold=0.35, max_dur=6.0, min_dur=1.0):
    """Split at the speaker's natural pauses, regardless of who is talking.

    Uses word-level timestamps: a silence gap between two consecutive words
    longer than `gap_threshold` starts a new segment; a segment also breaks
    when it grows past `max_dur` seconds. Result: one line per natural
    pause/sentence (~2-5 s), each with its own timestamp.
    """
    out = []
    for s in segments:
        words = s.get("words") or []
        if len(words) < 2:
            out.append(s)
            continue
        chunks, cur = [], [words[0]]
        for prev, w in zip(words, words[1:]):
            gap = w["start"] - prev["end"]
            cur_dur = w["end"] - cur[0]["start"]
            if gap > gap_threshold or cur_dur > max_dur:
                chunks.append(cur)
                cur = [w]
            else:
                cur.append(w)
        chunks.append(cur)
        # avoid tiny orphan chunks: merge a chunk shorter than min_dur into
        # its neighbour (by text length, not time)
        merged_chunks = []
        for c in chunks:
            if merged_chunks and (c[-1]["end"] - c[0]["start"]) < min_dur and \
                    len(" ".join(x["word"] for x in merged_chunks[-1])) + \
                    len(" ".join(x["word"] for x in c)) < 120:
                merged_chunks[-1].extend(c)
                continue
            merged_chunks.append(c)
        for c in merged_chunks:
            out.append({
                "start": c[0]["start"], "end": c[-1]["end"],
                "text": " ".join(x["word"] for x in c).strip(),
                "avg_logprob": s["avg_logprob"],
                "no_speech_prob": s["no_speech_prob"],
                "words": c,
            })
    return out


def recheck_segment(args):
    """Step 3: re-transcribe one suspicious slice with more context."""
    wav, start, end, model_name, language, initial_prompt = args
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    pad = 1.0
    kw = dict(vad_filter=True)
    if initial_prompt:
        kw["initial_prompt"] = initial_prompt
    if language:
        kw["language"] = language
    segments, _ = model.transcribe(wav, clip_timestamps=[max(0, start - pad), end + pad], **kw)
    text = " ".join(s.text.strip() for s in segments).strip()
    return start, end, text


def fmt_ts(sec, comma=False):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((sec - int(sec)) * 1000))
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}" if comma else f"{h:02d}:{m:02d}:{s:02d}"


def fmt_mmss(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def srt_time(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--context", default="", help="Speakers / language / topic (step 1)")
    ap.add_argument("--model", default="small",
                    help="faster-whisper model: small (default) | medium | large-v3-turbo")
    ap.add_argument("--language", default=None)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--speakers", default="", help="Comma-separated speaker names, optional")
    ap.add_argument("--pause-gap", type=float, default=0.35,
                    help="Silence gap (s) between words that starts a new line (natural pause)")
    ap.add_argument("--max-seg", type=float, default=6.0,
                    help="Force a break when a segment exceeds this many seconds")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.media)), "transkrip")
    os.makedirs(outdir, exist_ok=True)
    log = lambda msg: print(f"[transkrip] {msg}", flush=True)

    # --- Step 1: context -> initial_prompt -----------------------------------
    initial_prompt = (args.context or "").strip()
    if initial_prompt:
        log(f"context: {initial_prompt}")
    else:
        log("no context given; proceeding without initial_prompt")

    # --- Step 2: extract, split, transcribe in parallel -----------------------
    duration = media_duration(args.media)
    log(f"media duration: {duration:.0f}s")
    wav = extract_audio(args.media, outdir)
    parts = split_audio(wav, outdir, duration)
    log(f"split into {len(parts)} part(s); model={args.model}; workers={args.workers}")

    jobs = [(p, i * PART_SECONDS, args.model, args.language, initial_prompt)
            for i, p in enumerate(parts)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(transcribe_part, jobs))

    segments = []
    for r in sorted(results, key=lambda r: r[0]["start"] if r else 0):
        segments.extend(r)
    segments.sort(key=lambda s: s["start"])

    # merge artifacts at part boundaries
    merged = []
    for s in segments:
        if merged and not s["text"]:
            continue
        if merged and merged[-1]["end"] > s["start"] - 0.05 and \
                len(merged[-1]["text"]) + len(s["text"]) < 12:
            merged[-1]["text"] += " " + s["text"]
            merged[-1]["end"] = s["end"]
            continue
        merged.append(s)
    segments = merged
    log(f"{len(segments)} segments after merge")

    # --- Step 2b: split per natural pause (same speaker, per sentence) --------
    segments = split_by_pause(segments, gap_threshold=args.pause_gap,
                              max_dur=args.max_seg)
    avg = sum(s["end"] - s["start"] for s in segments) / max(1, len(segments))
    log(f"{len(segments)} segments after pause-split (avg {avg:.2f}s per line)")

    # --- Step 3: re-check on-point -------------------------------------------
    corrections = []
    suspicious = [s for s in segments
                  if s["avg_logprob"] < -0.75 or s["no_speech_prob"] > 0.55]
    log(f"re-check: {len(suspicious)} suspicious segment(s)")
    if suspicious:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            re_jobs = [(wav, s["start"], s["end"], args.model, args.language, initial_prompt)
                       for s in suspicious]
            for (start, end, new_text), old in zip(ex.map(recheck_segment, re_jobs), suspicious):
                if new_text and new_text.lower() != old["text"].lower():
                    if len(new_text) >= max(4, len(old["text"]) // 2):
                        corrections.append({"start": start, "end": end,
                                            "old": old["text"], "new": new_text})
                        old["text"] = new_text

    # glossary check: words from context that got no plausible occurrence
    glossary_notes = []
    if initial_prompt:
        import re
        terms = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", initial_prompt)
                 if w.isalpha() and len(w) > 4]
        joined = " ".join(s["text"] for s in segments).lower()
        for t in terms:
            if t.lower() not in joined:
                glossary_notes.append(t)

    # --- Step 4: outputs -------------------------------------------------------
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    lines_md, lines_txt, srt = [], [], []
    for i, s in enumerate(segments, 1):
        stamp = fmt_mmss(s["start"])
        if speakers:
            speaker = speakers[i % len(speakers)]  # naive rotation; agent may refine manually
            line_md = f"**[{stamp}] {speaker}**: {s['text']}"
        else:
            line_md = f"**[{stamp}]** {s['text']}"
        lines_md.append(line_md)
        lines_txt.append(f"[{fmt_ts(s['start'])} - {fmt_ts(s['end'])}] {s['text']}")
        srt.append(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n")

    with open(os.path.join(outdir, "transcript.md"), "w", encoding="utf-8") as f:
        f.write("# Transkrip\n\n" + "\n\n".join(lines_md) + "\n")
    with open(os.path.join(outdir, "transcript_timestamped.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_txt) + "\n")
    with open(os.path.join(outdir, "transcript.srt"), "w", encoding="utf-8") as f:
        f.write("\n".join(srt) + "\n")

    report = {"duration_s": round(duration), "segments": len(segments),
              "avg_segment_s": round(avg, 2), "mode": "per natural pause",
              "corrections": corrections, "glossary_missing": glossary_notes}
    with open(os.path.join(outdir, "transkrip_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.keep_parts:
        for p in parts:
            os.remove(p)
        try:
            os.remove(wav)
        except OSError:
            pass

    log(f"done -> {outdir} (transcript.md / transcript_timestamped.txt / transcript.srt)")
    log(f"corrections applied: {len(corrections)}; glossary terms not found: {glossary_notes}")


if __name__ == "__main__":
    main()
