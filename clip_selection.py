"""
Pure helpers for the Gemini clip-selection pipeline.

Standard-library only so both main.py and gemini_worker.py can import it and
the logic stays unit-testable without the heavy video dependencies.
"""

# USD per 1M tokens (input, output incl. thinking), from ai.google.dev pricing.
MODEL_PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
}


def lookup_model_prices(model_name):
    """Longest-prefix match against MODEL_PRICES; None if unknown."""
    name = str(model_name or "").lower()
    best_key = None
    for key in MODEL_PRICES:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return MODEL_PRICES[best_key] if best_key else None


def clip_count_targets(n_windows):
    """How many clips to ask the detail pass for, given the shortlist size.

    Measured on prod 3-ago-2026: 408 of 429 jobs (95%) delivered 3 clips or
    fewer, the mode being ONE, while the prompt was free to return one per
    shortlisted window. Users who received 1-3 clips came back a second day
    0.4% of the time; those who received 4-9 came back 16.1% — so the clip
    count, not the clip quality, is what the retention curve hangs on.

    The old prompt biased hard the other way ("prefer one great clip per
    candidate window") and handed the model two unbounded licences to drop
    clips (the 2-second rule and STANDS ALONE both end in "or skip it"), with
    no floor to stop it collapsing to a single clip. This puts a floor and a
    realistic ceiling on it instead.

    ``CLIP_TARGET_MIN`` / ``CLIP_TARGET_MAX`` override both for A/B runs
    without a deploy (the reframe-testing harness drives them).
    """
    import os

    n = max(1, int(n_windows or 1))
    # Floor grows with the material: 3 windows -> 3, 5 -> 4, 10+ -> 6.
    low = max(2, min(6, n // 2 + 2))
    # Ceiling allows a rich window to yield more than one without inviting padding.
    high = min(12, max(4, n * 2))
    low = min(low, high)

    def _override(name, current):
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = _override("CLIP_TARGET_MIN", low)
    high = _override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


def trim_to_best(shorts, max_clips):
    """Cut an over-long detail-pass result down to ``max_clips`` BY SCORE.

    The detail pass hands its clips back in transcript order, batch after
    batch, so slicing the list keeps the EARLIEST clips rather than the best
    ones. On a 9-minute walkthrough that quietly threw away everything past
    minute three: the model proposed clips across the whole video, and the
    ones covering the demo, the MCP walkthrough and the close were the tail
    that got dropped. Worse, the failure scales the wrong way — the more
    generous the model is, the more of the video disappears.

    That sabotages the windowing: get_viral_clips builds scoring windows
    precisely because "a single call over the whole transcript clusters picks
    near the start", and a positional slice puts the clustering right back.

    Ranking is by ``predicted_score`` (the detail prompt already asks for it,
    and nothing else was reading it here). Ties keep transcript order, and the
    survivors come back in transcript order too, so clip numbering still runs
    front to back the way every caller downstream expects.
    """
    max_clips = max(1, int(max_clips or 1))
    if len(shorts) <= max_clips:
        return list(shorts)

    def score(item):
        try:
            return float(item[1].get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    indexed = list(enumerate(shorts))
    best = sorted(indexed, key=score, reverse=True)[:max_clips]
    return [item for _, item in sorted(best, key=lambda pair: pair[0])]


def clip_duration_bounds():
    """The clip length band (seconds) the selection prompts and word-snapping
    enforce. ``CLIP_MIN_SECONDS`` / ``CLIP_MAX_SECONDS`` override the classic
    15-60 — set per job by /api/process when the user asks for a specific
    length, or by hand for A/B runs. Values are clamped to platform-sane
    limits and re-ordered so bad input degrades instead of breaking the job.
    """
    import os

    def _read(name, default):
        try:
            return float(os.environ.get(name, ""))
        except ValueError:
            return default

    lo = _read("CLIP_MIN_SECONDS", 15.0)
    hi = _read("CLIP_MAX_SECONDS", 60.0)
    lo = min(max(lo, 5.0), 175.0)
    hi = min(max(hi, 10.0), 180.0)
    if hi < lo + 5.0:  # keep a real band: degenerate ranges starve the model
        hi = min(180.0, lo + 5.0)
    return round(lo, 3), round(hi, 3)


def compact_words(words, precision=2):
    """Round word timestamps for prompts — full float precision wastes tokens."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """
    Build scoring windows aligned to Whisper segment boundaries, so a sentence
    (and usually a viral moment) is never cut in half mid-window. Windows grow
    segment by segment to roughly window_seconds (up to 1.25x for the closing
    segment) and the next window starts ~overlap_seconds before the previous
    end, also snapped to a segment start.
    """
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Extend while the NEXT segment still fits within a tolerant cap, so the
        # window closes on a segment boundary near window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "text": " ".join(seg[2] for seg in segments[i:j + 1]),
        })
        window_index += 1

        if j >= n - 1:
            break
        # Next window starts at the first segment beginning after (end - overlap),
        # but always makes progress.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(float(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


def _get_word_text(w):
    return str(w.get("w") or w.get("word") or "").strip()


def _is_sentence_start_word(words, idx, starts, ends):
    """Returns True if the word at idx marks the start of a sentence or complete line."""
    if idx == 0:
        return True
    prev_text = _get_word_text(words[idx - 1])
    # Preceded by terminal punctuation
    if any(prev_text.endswith(p) for p in (".", "?", "!", "…")):
        return True
    # Conversational pause of 300ms or more
    gap = starts[idx] - ends[idx - 1]
    if gap >= 0.30:
        return True
    # Capitalized word following a noticeable pause of 150ms or more
    curr_text = _get_word_text(words[idx])
    if curr_text and curr_text[0].isupper() and gap >= 0.15:
        return True
    return False


def _is_sentence_end_word(words, idx, starts, ends):
    """Returns True if the word at idx marks the conclusion of a sentence or complete thought."""
    if idx == len(words) - 1:
        return True
    curr_text = _get_word_text(words[idx])
    # Contains terminal punctuation
    if any(curr_text.endswith(p) for p in (".", "?", "!", "…")):
        return True
    # Followed by conversational pause of 350ms or more
    gap = starts[idx + 1] - ends[idx]
    if gap >= 0.35:
        return True
    # Next word begins a capitalized sentence following a pause of 200ms or more
    next_text = _get_word_text(words[idx + 1])
    if next_text and next_text[0].isupper() and gap >= 0.20:
        return True
    return False


def snap_clip_to_words(start, end, words, video_duration,
                       min_duration=15.0, max_duration=60.0,
                       search_window=1.5, max_lead=0.35, max_tail=0.45,
                       audio_pre_roll=0.150):
    """
    Snap Gemini-proposed clip boundaries onto real word boundaries plus a bit
    of the surrounding silence. LLMs are bad at millisecond arithmetic; the
    word-level timestamps are ground truth, so cuts land in pauses instead of
    mid-word.

    Snaps start precisely to the first spoken word of the sentence/line with a 150ms audio pre-roll buffer
    so the opening syllable is preserved cleanly.
    Snaps end precisely to the final spoken word of the sentence/line with terminal punctuation or pause,
    guaranteeing clips never cut off mid-sentence.

    words: [{'w','s','e'}, ...] for the whole video, sorted by start.
    Returns (start, end); falls back to the input if no words are nearby or
    snapping cannot satisfy the duration bounds.
    """
    original = (round(float(start), 3), round(float(end), 3))
    if not words:
        return original

    starts = [float(w.get("s", w.get("start", 0))) for w in words]
    ends = [float(w.get("e", w.get("end", 0))) for w in words]

    # START: snap to real opening sentence and opening word
    new_start = float(start)
    candidates = [(i, s) for i, s in enumerate(starts) if abs(s - new_start) <= search_window]
    if candidates:
        best_idx, word_start = min(candidates, key=lambda pair: abs(pair[1] - new_start))

        # Search for true sentence inception: look back up to 15 words or 4.0 seconds
        sentence_start_idx = best_idx
        for back_idx in range(best_idx, max(-1, best_idx - 16), -1):
            if _is_sentence_start_word(words, back_idx, starts, ends):
                sentence_start_idx = back_idx
                break

        # If backwards search found a sentence start within allowable reach
        if abs(starts[sentence_start_idx] - new_start) <= max(search_window, 4.0):
            best_idx = sentence_start_idx
            word_start = starts[best_idx]

        # Snap start time with 150ms audio pre-roll buffer to preserve opening syllable cleanly
        if best_idx > 0:
            gap = max(0.0, word_start - ends[best_idx - 1])
            lead = audio_pre_roll if gap >= audio_pre_roll else min(audio_pre_roll, gap * 0.8)
        else:
            lead = audio_pre_roll
        new_start = max(0.0, word_start - lead)

    # END: snap to real ending sentence and ending word
    new_end = float(end)
    end_candidates = [(i, e) for i, e in enumerate(ends) if abs(e - new_end) <= search_window]
    if end_candidates:
        best_end_idx, word_end = min(end_candidates, key=lambda pair: abs(pair[1] - new_end))

        # Check if candidate is already a sentence conclusion
        chosen_end_idx = best_end_idx
        if not _is_sentence_end_word(words, best_end_idx, starts, ends):
            # 1. First search forward to complete the sentence naturally
            found_forward = False
            for fwd_idx in range(best_end_idx, min(len(words), best_end_idx + 15)):
                cand_end_t = ends[fwd_idx]
                if cand_end_t - new_start > max_duration:
                    break
                if _is_sentence_end_word(words, fwd_idx, starts, ends):
                    chosen_end_idx = fwd_idx
                    found_forward = True
                    break

            # 2. If extending forward exceeds max_duration, search backwards for prior sentence conclusion
            if not found_forward:
                for back_idx in range(best_end_idx, max(-1, best_end_idx - 15), -1):
                    cand_end_t = ends[back_idx]
                    if cand_end_t - new_start < min_duration:
                        break
                    if _is_sentence_end_word(words, back_idx, starts, ends):
                        chosen_end_idx = back_idx
                        break

        word_end = ends[chosen_end_idx]
        if chosen_end_idx + 1 < len(starts):
            gap = max(0.0, starts[chosen_end_idx + 1] - word_end)
            tail = min(max_tail, max(0.15, gap * 0.7))
        else:
            tail = max_tail
        new_end = min(float(video_duration), word_end + tail)

    # Repair duration bounds while staying on word boundaries.
    if new_end - new_start < min_duration:
        target = new_start + min_duration
        later = sorted(e for e in ends if e >= target)
        if later and later[0] - new_start <= max_duration:
            new_end = min(float(video_duration), later[0] + 0.2)
        else:
            return original
    if new_end - new_start > max_duration:
        target = new_start + max_duration
        earlier = [e for e in ends if new_start < e <= target]
        new_end = (max(earlier) + 0.2) if earlier else target
        new_end = min(new_end, new_start + max_duration, float(video_duration))

    if new_end <= new_start or new_end - new_start < min_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))
