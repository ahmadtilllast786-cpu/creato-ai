"""Real Audio, Speech, Silence, and Speaker Metadata Extractor.

Extracts real, ground-truth metadata from video clips:
1. Real Speech & Word Timestamps (Whisper / faster-whisper / Parakeet):
   - Real spoken words with start/end millisecond timestamps (startMs, endMs, s, e, w).
   - Natural opening sentence/hook identification.
   - Key emphatic or high-velocity sentences.
2. Real Silence Intervals:
   - FFmpeg silencedetect to measure exact intervals where silence > 400ms.
3. Real Face & Speaker Tracking Keyframes (MediaPipe / OpenCV):
   - Real face bounding boxes (x, y, w, h) and smoothed centers (cx, cy).
4. Persistence:
   - Saves extracted data into dedicated clip JSON files and integrates with job metadata.
"""

import os
import re
import cv2
import json
import time
import subprocess
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import ffmpeg_env
from ffmpeg_utils import (
    run_ffmpeg_command,
    ensure_file_unlocked,
    open_video_capture,
    METADATA_SCRUB,
)

MIN_SILENCE_SECONDS = 0.40
SILENCE_NOISE_DB = -30.0
FACE_SAMPLE_FPS = 4.0  # Sample 4 frames per second for lightweight, accurate tracking
ANALYSIS_WIDTH = 480


def extract_silence_intervals(
    video_path: str,
    min_silence_s: float = MIN_SILENCE_SECONDS,
    noise_db: float = SILENCE_NOISE_DB
) -> List[Dict[str, Any]]:
    """Run FFmpeg silencedetect to extract real silence intervals (>400ms).
    Returns list of dicts with start, end, duration in seconds and milliseconds.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "info",
        "-i", video_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f", "null", "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        stderr = proc.stderr or ""
    except Exception as e:
        print(f"⚠️ Silence extraction error: {e}")
        return []

    silences: List[Dict[str, Any]] = []
    current_start: Optional[float] = None

    for line in stderr.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([0-9\.]+)", line)
            if m:
                current_start = float(m.group(1))
        elif "silence_end:" in line and current_start is not None:
            m = re.search(r"silence_end:\s*([0-9\.]+)", line)
            if m:
                end = float(m.group(1))
                dur = end - current_start
                if dur >= min_silence_s:
                    silences.append({
                        "start": round(current_start, 3),
                        "end": round(end, 3),
                        "duration": round(dur, 3),
                        "startMs": int(round(current_start * 1000)),
                        "endMs": int(round(end * 1000)),
                    })
                current_start = None

    return silences


def extract_face_keyframes(
    video_path: str,
    orig_w: int,
    orig_h: int,
    duration: float,
    fps: float,
    sample_fps: float = FACE_SAMPLE_FPS
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """Sample video frames to detect real face coordinates of the speaker using MediaPipe.
    Returns:
    - keyframes: list of {t, frame, box: [x, y, w, h], center: [cx, cy]}
    - smoothed_centers: smoothed center coordinates per frame across the entire clip
    """
    default_center = (orig_w // 2, orig_h // 2)
    total_frames = max(1, int(round(duration * fps))) if duration > 0 else 1

    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.45)
    except Exception as e:
        print(f"⚠️ MediaPipe not available: {e}. Using center coordinates.")
        return [], [default_center] * total_frames

    step = max(1, int(round(fps / sample_fps)))
    keyframes: List[Dict[str, Any]] = []
    sampled_centers: List[Tuple[float, float]] = []
    frame_idx = 0
    last_cx, last_cy = float(default_center[0]), float(default_center[1])

    with open_video_capture(video_path) as cap:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                t_sec = frame_idx / fps if fps > 0 else 0.0
                h, w = frame.shape[:2]

                if w > ANALYSIS_WIDTH:
                    scale = w / float(ANALYSIS_WIDTH)
                    small_w = ANALYSIS_WIDTH
                    small_h = max(2, int(h / scale))
                    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
                else:
                    small_frame = frame

                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                best_box = None
                best_center = (last_cx, last_cy)

                if results and results.detections:
                    best_score = -1.0
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        area = bbox.width * bbox.height
                        cand_x = int(bbox.xmin * orig_w)
                        cand_y = int(bbox.ymin * orig_h)
                        cand_w = int(bbox.width * orig_w)
                        cand_h = int(bbox.height * orig_h)
                        cand_cx = cand_x + cand_w / 2.0
                        cand_cy = cand_y + cand_h / 2.0
                        dist = abs(cand_cx - last_cx) + abs(cand_cy - last_cy)
                        score = area * 1000.0 - dist * 0.1

                        if score > best_score:
                            best_score = score
                            best_box = [cand_x, cand_y, cand_w, cand_h]
                            best_center = (cand_cx, cand_cy)

                if best_box is not None:
                    last_cx, last_cy = best_center
                    keyframes.append({
                        "t": round(t_sec, 3),
                        "tMs": int(round(t_sec * 1000)),
                        "frame": frame_idx,
                        "box": best_box,
                        "center": [int(round(last_cx)), int(round(last_cy))],
                    })

                sampled_centers.append((last_cx, last_cy))

            frame_idx += 1

    detector.close()

    if not sampled_centers:
        sampled_centers = [default_center]

    # Interpolate centers across all frames and smooth with EMA (alpha=0.15)
    full_centers: List[Tuple[float, float]] = []
    for f in range(total_frames):
        s_idx = min(int(f // step), len(sampled_centers) - 1)
        full_centers.append(sampled_centers[s_idx])

    alpha = 0.15
    smoothed: List[Tuple[int, int]] = []
    sx, sy = float(full_centers[0][0]), float(full_centers[0][1])

    for cx, cy in full_centers:
        sx = alpha * cx + (1.0 - alpha) * sx
        sy = alpha * cy + (1.0 - alpha) * sy
        ix = int(round(max(0, min(orig_w, sx))))
        iy = int(round(max(0, min(orig_h, sy))))
        smoothed.append((ix, iy))

    return keyframes, smoothed


def extract_speech_and_words(
    video_path: str,
    existing_transcript: Optional[Dict[str, Any]] = None,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None
) -> Dict[str, Any]:
    """Extract real words, millisecond timestamps, opening hook, and key sentences.
    Reuses existing parent transcript if provided; otherwise runs faster-whisper on video_path.
    """
    transcript = None

    # 1. If parent transcript provided with absolute timestamps, slice the clip window
    if existing_transcript and existing_transcript.get("segments"):
        all_words = []
        clip_segments = []
        end_bound = clip_end if clip_end is not None else 1e9

        for seg in existing_transcript.get("segments", []):
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            if e > clip_start and s < end_bound:
                # Segment overlaps clip
                seg_words = []
                for w in seg.get("words", []):
                    ws = float(w.get("start", 0.0))
                    we = float(w.get("end", 0.0))
                    if we > clip_start and ws < end_bound:
                        # Relativize to clip start
                        rel_s = max(0.0, ws - clip_start)
                        rel_e = max(rel_s + 0.05, we - clip_start)
                        word_str = str(w.get("word") or w.get("w") or "").strip()
                        if word_str:
                            w_item = {
                                "w": word_str,
                                "word": word_str,
                                "s": round(rel_s, 3),
                                "e": round(rel_e, 3),
                                "startMs": int(round(rel_s * 1000)),
                                "endMs": int(round(rel_e * 1000)),
                            }
                            seg_words.append(w_item)
                            all_words.append(w_item)

                if seg_words:
                    clip_segments.append({
                        "start": round(max(0.0, s - clip_start), 3),
                        "end": round(max(0.0, e - clip_start), 3),
                        "text": seg.get("text", "").strip(),
                        "words": seg_words,
                    })

        if all_words:
            transcript = {
                "language": existing_transcript.get("language", "en"),
                "text": " ".join(s["text"] for s in clip_segments),
                "segments": clip_segments,
                "words": all_words,
            }

    # 2. If no valid words sliced, run speech transcription on the video clip directly
    if not transcript or not transcript.get("words"):
        try:
            from transcribe_backends import transcribe_media
            raw = transcribe_media(video_path)
            all_words = []
            for seg in raw.get("segments", []):
                for w in seg.get("words", []):
                    w_str = str(w.get("word", "")).strip()
                    if not w_str:
                        continue
                    s = float(w.get("start", 0.0))
                    e = float(w.get("end", 0.0))
                    all_words.append({
                        "w": w_str,
                        "word": w_str,
                        "s": round(s, 3),
                        "e": round(e, 3),
                        "startMs": int(round(s * 1000)),
                        "endMs": int(round(e * 1000)),
                    })
            transcript = {
                "language": raw.get("language", "en"),
                "text": raw.get("text", ""),
                "segments": raw.get("segments", []),
                "words": all_words,
            }
        except Exception as e:
            print(f"⚠️ Speech-to-text extraction fallback on {video_path}: {e}")
            transcript = {
                "language": "en",
                "text": "",
                "segments": [],
                "words": [],
            }

    words = transcript.get("words", [])
    segments = transcript.get("segments", [])

    # Identify Opening Hook: first cohesive statement (0.0 to ~3.5s)
    opening_hook = None
    if segments:
        first_seg = segments[0]
        opening_hook = {
            "text": first_seg.get("text", "").strip(),
            "start": first_seg.get("start", 0.0),
            "end": first_seg.get("end", 0.0),
            "startMs": int(round(first_seg.get("start", 0.0) * 1000)),
            "endMs": int(round(first_seg.get("end", 0.0) * 1000)),
        }
    elif words:
        hook_words = [w for w in words if w["s"] <= 3.5]
        if hook_words:
            opening_hook = {
                "text": " ".join(w["w"] for w in hook_words),
                "start": hook_words[0]["s"],
                "end": hook_words[-1]["e"],
                "startMs": hook_words[0]["startMs"],
                "endMs": hook_words[-1]["endMs"],
            }

    # Tag Key Sentences: high word velocity (words/sec >= 2.8) or emphatic punctuation
    key_sentences = []
    for seg in segments:
        text = seg.get("text", "").strip()
        dur = max(0.1, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        n_w = len(seg.get("words", [])) or len(text.split())
        velocity = n_w / dur
        is_emphatic = any(p in text for p in ("!", "?", "🔥", "💥")) or velocity >= 3.2
        if is_emphatic or velocity >= 2.8:
            key_sentences.append({
                "text": text,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "startMs": int(round(float(seg.get("start", 0.0)) * 1000)),
                "endMs": int(round(float(seg.get("end", 0.0)) * 1000)),
                "word_velocity": round(velocity, 2),
                "is_emphatic": is_emphatic,
            })

    return {
        "text": transcript.get("text", ""),
        "language": transcript.get("language", "en"),
        "words": words,
        "segments": segments,
        "opening_hook": opening_hook,
        "key_sentences": key_sentences,
    }


def extract_clip_metadata(
    output_dir: str,
    video_path: str,
    clip_index: int = 0,
    existing_transcript: Optional[Dict[str, Any]] = None,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None
) -> Dict[str, Any]:
    """Complete extraction pipeline for a clip:
    1. Video properties (duration, resolution, fps).
    2. Real speech & words with millisecond timestamps.
    3. Real silence intervals (>400ms) from FFmpeg.
    4. Real speaker face bounding boxes & centers from MediaPipe.
    5. Persists to disk at {clip_basename}_real_metadata.json.
    """
    ensure_file_unlocked(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found for extraction: {video_path}")

    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]

    with open_video_capture(video_path) as cap:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)

    duration = total_frames / fps if fps > 0 else 0.0

    print(f"📊 [MetadataExtractor] Extracting real data for {filename} ({duration:.1f}s)...")

    # 1. Real Speech & Words
    speech_data = extract_speech_and_words(
        video_path,
        existing_transcript=existing_transcript,
        clip_start=clip_start,
        clip_end=clip_end or duration
    )

    # 2. Real Silence Intervals
    silences = extract_silence_intervals(video_path, min_silence_s=MIN_SILENCE_SECONDS)

    # 3. Real Face Coordinates & Speaker Centers
    keyframes, smoothed_centers = extract_face_keyframes(
        video_path, orig_w, orig_h, duration, fps
    )

    metadata: Dict[str, Any] = {
        "clip_index": clip_index,
        "filename": filename,
        "duration": round(duration, 3),
        "durationMs": int(round(duration * 1000)),
        "fps": round(fps, 2),
        "width": orig_w,
        "height": orig_h,
        "words": speech_data["words"],
        "segments": speech_data["segments"],
        "text": speech_data["text"],
        "language": speech_data["language"],
        "opening_hook": speech_data["opening_hook"],
        "key_sentences": speech_data["key_sentences"],
        "silence_intervals": silences,
        "speaker_keyframes": keyframes,
        "speaker_centers": smoothed_centers,
        "extracted_at": time.time(),
        "is_extracted": True,
    }

    # Save dedicated JSON file for this clip
    meta_path = os.path.join(output_dir, f"{base_name}_real_metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"   💾 Saved real metadata: {meta_path}")
    except Exception as e:
        print(f"⚠️ Failed to write {meta_path}: {e}")

    return metadata


def load_clip_metadata(output_dir: str, filename: str) -> Optional[Dict[str, Any]]:
    """Load previously extracted metadata for a clip if present on disk."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # Check dedicated real metadata file
    meta_path = os.path.join(output_dir, f"{base_name}_real_metadata.json")
    if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Check job-level metadata.json
    job_meta_files = [f for f in os.listdir(output_dir) if f.endswith("_metadata.json") and not f.endswith("_real_metadata.json")]
    for jm in job_meta_files:
        try:
            with open(os.path.join(output_dir, jm), "r", encoding="utf-8") as f:
                data = json.load(f)
            shorts = data.get("shorts", [])
            for s in shorts:
                v_url = s.get("video_url", "")
                if os.path.basename(v_url) == os.path.basename(filename) and s.get("real_metadata"):
                    return s["real_metadata"]
        except Exception:
            pass

    return None
