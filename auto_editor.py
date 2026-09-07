"""Stage 2 On-Demand Auto Edit Engine.

Provides dynamic reframing and pacing cuts on-demand:
1. Silence & Pacing Cuts: Detects and trims dead silence (>400ms) with smooth
   audio micro-transitions (anti-pop fades) between joined cuts.
2. MediaPipe Active Speaker Tracking & Centering: Tracks the speaker's face
   dynamically and calculates a smoothed camera center (cx, cy).
3. Contextual Dynamic Zooms: Introduces subtle, punchy 1.15x–1.25x zooms
   centered on the speaker during key hook statements and punchlines, easing smoothly
   back out during pauses and demonstrations.
4. Non-Destructive Workflow: Keeps the Stage 1 base cut intact on disk, rendering
   the edited version to a dedicated output path.
5. Stability & Encoding: Strictly even dimensions (1080x1920), software fallback
   (-c:v libx264 -preset veryfast -crf 22), and full stderr capture.
"""

import ffmpeg_env
import os
import re
import cv2
import tempfile
import subprocess
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from ffmpeg_utils import (
    run_ffmpeg_command,
    ensure_file_unlocked,
    cleanup_temp_file,
    open_video_capture,
    escape_filter_value,
    METADATA_SCRUB,
)

# Silence detection defaults
MIN_SILENCE_SECONDS = 0.40  # 400ms threshold for dead silence
BREATH_MARGIN_SECONDS = 0.06  # 60ms natural breath padding around speech
SILENCE_NOISE_DB = -30.0

# Dynamic zoom defaults
DEFAULT_MAX_ZOOM = 1.20     # Centered between 1.15x and 1.25x
RISE_SECONDS = 0.25
HOLD_SECONDS = 1.50
FALL_SECONDS = 0.55

# Analysis width for fast MediaPipe face tracking
ANALYSIS_WIDTH = 480


def _ease(t: float) -> float:
    """Smoothstep easing on [0, 1]: 3t^2 - 2t^3."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def detect_silence_intervals(
    video_path: str,
    min_silence_s: float = MIN_SILENCE_SECONDS,
    noise_db: float = SILENCE_NOISE_DB
) -> List[Tuple[float, float]]:
    """Detect intervals of silence exceeding min_silence_s using FFmpeg silencedetect.
    Returns list of (silence_start, silence_end) in seconds.
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
        print(f"⚠️ Silence detection failed: {e}")
        return []

    silences: List[Tuple[float, float]] = []
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
                if end - current_start >= min_silence_s:
                    silences.append((current_start, end))
                current_start = None

    return silences


def compute_kept_segments(
    duration: float,
    silences: List[Tuple[float, float]],
    min_silence_s: float = MIN_SILENCE_SECONDS,
    breath_margin: float = BREATH_MARGIN_SECONDS
) -> List[Tuple[float, float]]:
    """Convert silence intervals into segments to keep.
    Splices out dead silence (> min_silence_s) while preserving natural breath padding.
    """
    if not silences or duration <= 0:
        return [(0.0, max(0.0, duration))]

    # Filter silences that qualify and compute trimmed gaps
    filtered_cuts = []
    for s_start, s_end in silences:
        s_start = max(0.0, min(duration, s_start))
        s_end = max(0.0, min(duration, s_end))
        if s_end - s_start >= min_silence_s:
            cut_start = s_start + breath_margin
            cut_end = s_end - breath_margin
            # Only cut if at least 100ms of dead air is removed
            if cut_end - cut_start >= 0.10:
                filtered_cuts.append((cut_start, cut_end))

    if not filtered_cuts:
        return [(0.0, duration)]

    # Merge any overlapping cuts
    merged_cuts = []
    for c_start, c_end in sorted(filtered_cuts, key=lambda x: x[0]):
        if not merged_cuts:
            merged_cuts.append([c_start, c_end])
        else:
            prev = merged_cuts[-1]
            if c_start <= prev[1]:
                prev[1] = max(prev[1], c_end)
            else:
                merged_cuts.append([c_start, c_end])

    # Invert cuts into kept segments
    kept: List[Tuple[float, float]] = []
    curr = 0.0
    for c_start, c_end in merged_cuts:
        if c_start > curr + 0.20:  # Minimum 200ms segment duration
            kept.append((round(curr, 4), round(c_start, 4)))
        curr = c_end

    if curr < duration - 0.20:
        kept.append((round(curr, 4), round(duration, 4)))

    return kept if kept else [(0.0, duration)]


def trim_silence_pacing(
    input_path: str,
    output_path: str,
    silences: List[Tuple[float, float]],
    duration: float,
    breath_margin: float = BREATH_MARGIN_SECONDS
) -> bool:
    """Splice out silence intervals > 400ms with smooth 20ms audio micro-transitions
    (afade in/out) to eliminate pops between cuts.
    """
    kept = compute_kept_segments(duration, silences, breath_margin=breath_margin)
    if len(kept) <= 1:
        # No silence trimmed
        return False

    print(f"   ✂️ Trimming {len(kept) - 1} dead silence interval(s) > 400ms")
    filter_parts = []
    concat_inputs = []
    for idx, (seg_s, seg_e) in enumerate(kept):
        filter_parts.append(
            f"[0:v]trim=start={seg_s:.4f}:end={seg_e:.4f},setpts=PTS-STARTPTS[v{idx}];"
            f"[0:a]atrim=start={seg_s:.4f}:end={seg_e:.4f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:d=0.02,afade=t=out:d=0.02[a{idx}];"
        )
        concat_inputs.append(f"[v{idx}][a{idx}]")

    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(kept)}:v=1:a=1[outv][outa]")
    filter_complex = "".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        *METADATA_SCRUB,
        output_path
    ]

    try:
        run_ffmpeg_command(cmd)
        return ensure_file_unlocked(output_path, timeout=15)
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else "Unknown FFmpeg error"
        print(f"   ❌ Silence trimming FFmpeg failed: {err_msg[:300]}")
        return False


def track_active_speaker(
    video_path: str,
    orig_w: int,
    orig_h: int,
    n_frames: int,
    fps: float,
    sample_step: int = 2
) -> List[Tuple[int, int]]:
    """MediaPipe face detection to track active speaker center (cx, cy) across frames.
    Uses exponential moving average (EMA) smoothing for tripod-like camera stability.
    """
    default_center = (orig_w // 2, orig_h // 2)
    if n_frames <= 0:
        return [default_center]

    centers: List[Tuple[float, float]] = []

    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.45)
    except Exception as e:
        print(f"⚠️ MediaPipe not available: {e}. Using center coordinates.")
        return [default_center] * n_frames

    with open_video_capture(video_path) as cap:
        last_cx, last_cy = float(default_center[0]), float(default_center[1])
        frame_idx = 0

        while frame_idx < n_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_step == 0:
                h, w = frame.shape[:2]
                # Downscale for fast face detection
                if w > ANALYSIS_WIDTH:
                    scale = w / float(ANALYSIS_WIDTH)
                    small_w = ANALYSIS_WIDTH
                    small_h = max(2, int(h / scale))
                    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
                else:
                    small_frame = frame

                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                if results and results.detections:
                    best_cand = None
                    best_score = -1.0
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        area = bbox.width * bbox.height
                        cand_cx = (bbox.xmin + bbox.width / 2.0) * orig_w
                        cand_cy = (bbox.ymin + bbox.height / 2.0) * orig_h
                        dist = abs(cand_cx - last_cx) + abs(cand_cy - last_cy)
                        # Proximity bonus + area
                        score = area * 1000.0 - dist * 0.1
                        if score > best_score:
                            best_score = score
                            best_cand = (cand_cx, cand_cy)

                    if best_cand:
                        last_cx, last_cy = best_cand

                centers.append((last_cx, last_cy))
            else:
                # Repeat last known detection for step frames
                centers.append((last_cx, last_cy))

            frame_idx += 1

    detector.close()

    # Fill remaining frames if video stream ended early
    while len(centers) < n_frames:
        centers.append(centers[-1] if centers else default_center)

    # Exponential Moving Average (EMA) smoothing for stability (tripod feel)
    alpha = 0.15
    smoothed: List[Tuple[int, int]] = []
    sx, sy = float(centers[0][0]), float(centers[0][1])

    for cx, cy in centers:
        sx = alpha * cx + (1.0 - alpha) * sx
        sy = alpha * cy + (1.0 - alpha) * sy
        # Clamp to video bounds
        ix = int(round(max(0, min(orig_w, sx))))
        iy = int(round(max(0, min(orig_h, sy))))
        smoothed.append((ix, iy))

    return smoothed


def compute_contextual_zooms(
    n_frames: int,
    fps: float,
    video_path: Optional[str] = None,
    transcript_words: Optional[List[Dict[str, Any]]] = None,
    max_zoom: float = DEFAULT_MAX_ZOOM
) -> List[float]:
    """Compute per-frame zoom factors (1.0 to max_zoom).
    Punches in on key hook statements, beats, and punchlines; smoothly eases back out
    during pauses or demonstrations.
    """
    max_zoom = min(max(max_zoom, 1.15), 1.25)
    zooms = [1.0] * max(0, n_frames)
    if not zooms or fps <= 0:
        return zooms

    duration = n_frames / fps
    span = RISE_SECONDS + HOLD_SECONDS + FALL_SECONDS

    # 1. Opening hook punch-in: first 0.5s to 2.8s
    emphasis_times = []
    if duration >= 4.0:
        emphasis_times.append(0.5)

    # 2. Audio energy emphasis beats
    if video_path and os.path.exists(video_path):
        try:
            import punch_in
            beats = punch_in.emphasis_times(video_path, duration, min_gap=6.0, prominence=0.55)
            emphasis_times.extend(beats)
        except Exception as e:
            print(f"⚠️ Audio emphasis detection error: {e}")

    # 3. Transcript-aware emphasis (exclamations, strong punchlines)
    if transcript_words:
        last_added = -10.0
        for w_obj in transcript_words:
            word_str = str(w_obj.get("w", "")).strip()
            word_s = float(w_obj.get("s", 0.0))
            if any(p in word_str for p in ("!", "?")) and (word_s - last_added >= 6.0):
                emphasis_times.append(word_s)
                last_added = word_s

    # Deduplicate and sort emphasis times
    emphasis_times = sorted(list(set([round(t, 2) for t in emphasis_times if 0.0 <= t < duration])))

    # Generate smoothstep zoom curve
    for t in emphasis_times:
        if t + span > duration + 1.0:
            continue
        start_f = max(0, int(t * fps))
        end_f = min(n_frames, int((t + span) * fps) + 1)

        for f in range(start_f, end_f):
            dt = f / fps - t
            if dt < 0:
                continue
            elif dt < RISE_SECONDS:
                amount = _ease(dt / RISE_SECONDS)
            elif dt < RISE_SECONDS + HOLD_SECONDS:
                amount = 1.0
            elif dt < span:
                amount = 1.0 - _ease((dt - RISE_SECONDS - HOLD_SECONDS) / FALL_SECONDS)
            else:
                continue

            z_val = 1.0 + (max_zoom - 1.0) * amount
            zooms[f] = max(zooms[f], z_val)

    return zooms


def generate_zoom_boxes(
    centers: List[Tuple[int, int]],
    zooms: List[float],
    orig_w: int = 1080,
    orig_h: int = 1920
) -> List[Tuple[int, int, int, int]]:
    """Calculate per-frame (crop_w, crop_h, crop_x, crop_y) centered on the speaker.
    All dimensions are strictly even.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    n = min(len(centers), len(zooms))

    for i in range(n):
        z = zooms[i]
        w = int(orig_w / z)
        h = int(orig_h / z)
        w -= w % 2
        h -= h % 2
        w = max(2, min(w, orig_w))
        h = max(2, min(h, orig_h))

        cx, cy = centers[i]
        x = int(round(cx - w / 2.0))
        y = int(round(cy - h / 2.0))

        # Clamp within video bounds
        x = max(0, min(orig_w - w, x))
        y = max(0, min(orig_h - h, y))
        x -= x % 2
        y -= y % 2

        boxes.append((w, h, x, y))

    return boxes


def generate_sendcmd_lines(
    boxes: List[Tuple[int, int, int, int]],
    fps: float,
    target: str = "crop@c"
) -> List[str]:
    """Generate deduplicated sendcmd lines for FFmpeg dynamic crop."""
    lines = []
    prev = None

    for i, box in enumerate(boxes):
        if box == prev:
            continue
        t = i / fps
        w, h, x, y = box
        pw, ph, px, py = prev if prev else (None, None, None, None)
        if w != pw:
            lines.append(f"{t:.4f} {target} w {w};")
        if h != ph:
            lines.append(f"{t:.4f} {target} h {h};")
        if x != px:
            lines.append(f"{t:.4f} {target} x {x};")
        if y != py:
            lines.append(f"{t:.4f} {target} y {y};")
        prev = box

    return lines


def auto_edit_clip(
    input_clip_path: str,
    output_clip_path: str,
    transcript_words: Optional[List[Dict[str, Any]]] = None,
    max_zoom: float = DEFAULT_MAX_ZOOM,
    min_silence_s: float = MIN_SILENCE_SECONDS,
    breath_margin: float = BREATH_MARGIN_SECONDS,
    precomputed_silences: Optional[List[Any]] = None,
    precomputed_centers: Optional[List[Tuple[int, int]]] = None
) -> Dict[str, Any]:
    """Execute the Stage 2 On-Demand Auto Edit pipeline:
    1. Silence & Pacing Cuts (>400ms dead air removed with smooth micro-transitions).
    2. MediaPipe Active Speaker Tracking & Centering.
    3. Contextual Dynamic Zooms (1.15x to 1.25x punch-in zooms on key statements/hook).
    4. Non-destructive rendering with software fallback and strictly even dimensions (1080x1920).
    """
    ensure_file_unlocked(input_clip_path)
    if not os.path.exists(input_clip_path):
        raise FileNotFoundError(f"Input clip not found: {input_clip_path}")

    print(f"\n⚡ [Stage 2] Starting Auto Edit on: {input_clip_path}")

    # Probe duration and dimensions
    with open_video_capture(input_clip_path) as cap:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)

    duration = total_frames / fps if fps > 0 else 0.0

    # Ensure strictly even base dimensions
    orig_w -= orig_w % 2
    orig_h -= orig_h % 2

    workdir = tempfile.mkdtemp(prefix="auto_edit_")
    trimmed_clip_path = os.path.join(workdir, "trimmed_pacing.mp4")
    cmd_file_path = os.path.join(workdir, "sendcmd.txt")

    silence_cuts_count = 0
    zooms_applied_count = 0

    try:
        # Step 1: Detect and trim dead silence (>400ms) with anti-pop audio micro-fades
        if precomputed_silences is not None:
            silences = [
                (float(s["start"]), float(s["end"])) if isinstance(s, dict) else (float(s[0]), float(s[1]))
                for s in precomputed_silences
            ]
        else:
            silences = detect_silence_intervals(input_clip_path, min_silence_s=min_silence_s)

        has_silence_cuts = False

        if silences:
            has_silence_cuts = trim_silence_pacing(
                input_clip_path,
                trimmed_clip_path,
                silences,
                duration,
                breath_margin=breath_margin
            )

        active_working_clip = trimmed_clip_path if has_silence_cuts else input_clip_path
        if has_silence_cuts:
            silence_cuts_count = len(compute_kept_segments(duration, silences, min_silence_s, breath_margin)) - 1
            # Re-probe working clip properties
            with open_video_capture(active_working_clip) as cap:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)
                orig_w -= orig_w % 2
                orig_h -= orig_h % 2

        # Step 2: MediaPipe active speaker tracking
        if not has_silence_cuts and precomputed_centers and len(precomputed_centers) == total_frames:
            print(f"   👤 Using precomputed speaker tracking across {total_frames} frames...")
            centers = precomputed_centers
        else:
            print(f"   👤 Tracking active speaker with MediaPipe across {total_frames} frames...")
            centers = track_active_speaker(active_working_clip, orig_w, orig_h, total_frames, fps)

        # Step 3: Compute contextual dynamic zooms (1.15x to 1.25x)
        print(f"   🔍 Computing contextual dynamic zooms (1.15x–1.25x)...")
        zooms = compute_contextual_zooms(
            total_frames,
            fps,
            video_path=active_working_clip,
            transcript_words=transcript_words,
            max_zoom=max_zoom
        )

        boxes = generate_zoom_boxes(centers, zooms, orig_w=orig_w, orig_h=orig_h)
        lines = generate_sendcmd_lines(boxes, fps)
        zooms_applied_count = len([z for z in zooms if z > 1.01])

        with open(cmd_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Step 4: Render dynamic crop natively in FFmpeg
        first_box = boxes[0] if boxes else (orig_w, orig_h, 0, 0)
        init_crop = f"w={first_box[0]}:h={first_box[1]}:x={first_box[2]}:y={first_box[3]}"

        filter_graph = (
            f"[0:v]sendcmd=f='{escape_filter_value(cmd_file_path)}',"
            f"crop@c={init_crop},"
            f"scale=1080:1920:flags=lanczos,setsar=1[v]"
        )

        render_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", active_working_clip,
            "-filter_complex", filter_graph,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            *METADATA_SCRUB,
            output_clip_path
        ]

        print(f"   🎬 Rendering auto-edited clip: {output_clip_path}")
        run_ffmpeg_command(render_cmd)
        ensure_file_unlocked(output_clip_path, timeout=15)

        print(f"   ✅ Auto Edit completed successfully!")
        return {
            "success": True,
            "output_path": output_clip_path,
            "silence_cuts": silence_cuts_count,
            "zooms_applied": zooms_applied_count > 0,
            "duration": total_frames / fps if fps > 0 else duration,
        }

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else "Unknown FFmpeg error"
        print(f"   ❌ FFmpeg Auto Edit error: {err_msg[:400]}")
        raise RuntimeError(f"Auto Edit FFmpeg failed: {err_msg[:200]}") from e

    finally:
        # Safe cleanup of temp files
        cleanup_temp_file(cmd_file_path)
        cleanup_temp_file(trimmed_clip_path)
        try:
            if os.path.exists(workdir):
                os.rmdir(workdir)
        except Exception:
            pass
