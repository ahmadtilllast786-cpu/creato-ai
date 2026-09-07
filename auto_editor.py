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
import json
import copy
import tempfile
import subprocess
import numpy as np
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Tuple, Optional, Dict, Any, Union

from ffmpeg_utils import (
    run_ffmpeg_command,
    ensure_file_unlocked,
    cleanup_temp_file,
    open_video_capture,
    escape_filter_value,
    format_ffmpeg_error,
    METADATA_SCRUB,
)

# Standard audio encoding parameters for auto edit pipeline
AUDIO_ENCODE_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]


def check_audio_stream(video_path: str) -> bool:
    """Check via ffprobe if an input/output video file contains an active audio stream."""
    if not video_path or not os.path.exists(video_path):
        return False
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json",
            video_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout)
            return len(data.get("streams", [])) > 0
    except Exception as e:
        print(f"⚠️ check_audio_stream probe error for {video_path}: {e}")
    return False


def check_audio_volume(video_path: str) -> Dict[str, float]:
    """Check audio volume levels using FFmpeg volumedetect filter."""
    if not video_path or not os.path.exists(video_path):
        return {"mean_volume": -100.0, "max_volume": -100.0}
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", video_path,
            "-af", "volumedetect",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        mean_vol = -100.0
        max_vol = -100.0
        for line in (res.stderr or "").splitlines():
            if "mean_volume:" in line:
                m = re.search(r"mean_volume:\s*(-?[0-9\.]+)\s*dB", line)
                if m:
                    mean_vol = float(m.group(1))
            elif "max_volume:" in line:
                m = re.search(r"max_volume:\s*(-?[0-9\.]+)\s*dB", line)
                if m:
                    max_vol = float(m.group(1))
        return {"mean_volume": mean_vol, "max_volume": max_vol}
    except Exception as e:
        print(f"⚠️ check_audio_volume error: {e}")
        return {"mean_volume": 0.0, "max_volume": 0.0}


def validate_output_streams(output_path: str, expect_audio: bool = True) -> Dict[str, Any]:
    """Validate with ffprobe that:
    1. An active video stream exists.
    2. An active audio stream exists if expected (has_audio=True).
    3. The output video and audio durations match within reasonable tolerance.
    4. The output audio stream is not accidentally muted/silent.
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Output file missing or empty: {output_path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if res.returncode != 0:
        err = res.stderr or "Unknown ffprobe error"
        raise RuntimeError(f"ffprobe stream validation failed on {output_path}: {err}")

    info = json.loads(res.stdout or "{}")
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_video:
        raise RuntimeError(f"Stream validation failure: No video stream present in {output_path}")
    if expect_audio and not has_audio:
        raise RuntimeError(f"Stream validation failure: Active audio stream missing in {output_path}")

    v_dur = 0.0
    a_dur = 0.0
    for s in streams:
        if s.get("codec_type") == "video":
            v_dur = float(s.get("duration") or info.get("format", {}).get("duration") or 0.0)
        elif s.get("codec_type") == "audio":
            a_dur = float(s.get("duration") or info.get("format", {}).get("duration") or 0.0)

    if has_audio and v_dur > 0 and a_dur > 0:
        diff = abs(v_dur - a_dur)
        if diff > 1.2:
            print(f"   ⚠️ Audio/Video duration mismatch: audio={a_dur:.2f}s vs video={v_dur:.2f}s (diff={diff:.2f}s)")

    if expect_audio and has_audio:
        vol = check_audio_volume(output_path)
        if vol["max_volume"] > -99.0 and vol["max_volume"] < -60.0 and vol["mean_volume"] < -75.0:
            raise RuntimeError(
                f"Stream validation failure: Output audio is completely silent "
                f"(mean_volume={vol['mean_volume']}dB, max_volume={vol['max_volume']}dB) in {output_path}"
            )

    return {
        "has_video": has_video,
        "has_audio": has_audio,
        "video_duration": v_dur,
        "audio_duration": a_dur,
    }


def remap_word_timestamps(
    words: List[Dict[str, Any]],
    kept_segments: List[Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """Recalculate and offset caption/word timestamps to match trimmed silence or pacing cuts,
    guaranteeing subtitles stay perfectly in sync with the speaker's voice.
    """
    if not words or not kept_segments:
        return words or []

    cum_starts = [0.0]
    for s, e in kept_segments[:-1]:
        cum_starts.append(cum_starts[-1] + (e - s))

    def _map_time(t: float) -> float:
        if t <= kept_segments[0][0]:
            return 0.0
        for i, (s, e) in enumerate(kept_segments):
            if s <= t <= e:
                return cum_starts[i] + (t - s)
            if i < len(kept_segments) - 1 and e < t < kept_segments[i + 1][0]:
                # Falls inside a cut-out dead silence gap: clamp to cut boundary
                return cum_starts[i] + (e - s)
        last_idx = len(kept_segments) - 1
        return cum_starts[last_idx] + (kept_segments[last_idx][1] - kept_segments[last_idx][0])

    remapped = []
    for w in words:
        w_text = str(w.get("word") or w.get("w") or "").strip()
        if not w_text:
            continue
        s_orig = float(w.get("start", w.get("s", 0.0)))
        e_orig = float(w.get("end", w.get("e", 0.0)))
        s_new = _map_time(s_orig)
        e_new = _map_time(e_orig)
        if e_new <= s_new:
            e_new = s_new + 0.10

        remapped.append({
            **w,
            "word": w_text,
            "w": w_text,
            "start": round(s_new, 3),
            "end": round(e_new, 3),
            "s": round(s_new, 3),
            "e": round(e_new, 3),
            "startMs": int(round(s_new * 1000)),
            "endMs": int(round(e_new * 1000)),
        })
    return remapped


def _hex_to_ass(hex_color: str, opacity: float = 1.0, fallback: str = "FFFFFF") -> str:
    """Convert #RRGGBB hex color to ASS &HAABBGGRR format."""
    hex_digits = str(hex_color or "").lstrip('#')
    if len(hex_digits) != 6:
        hex_digits = fallback
    alpha = round((1.0 - max(0.0, min(1.0, opacity))) * 255)
    r = hex_digits[0:2]
    g = hex_digits[2:4]
    b = hex_digits[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _hex_to_ass_inline(hex_color: str, fallback: str = "FFFFFF") -> str:
    """Convert #RRGGBB hex color to ASS inline \\c&HBBGGRR& format."""
    hex_digits = str(hex_color or "").lstrip('#')
    if len(hex_digits) != 6:
        hex_digits = fallback
    r = hex_digits[0:2]
    g = hex_digits[2:4]
    b = hex_digits[4:6]
    return f"&H{b}{g}{r}&".upper()


def generate_subtitles_ass(
    words: List[Dict[str, Any]],
    output_ass_path: str,
    max_chars: int = 16,
    max_duration: float = 1.6,
    margin_v: int = 320,
    margin_l: int = 120,
    margin_r: int = 140,
    font_size: int = 50,
    font_name: str = "Arial",
    subtitle_settings: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convert word-level transcription timestamps into an Advanced SubStation Alpha (.ass) file
    with strict invisible boundary constraints for 1080x1920 portrait shorts:
    - Anchored in the lower safe zone below the main screen window (Y in [1480, 1600])
    - Clean margin above bottom platform controls (MarginV = 320px from bottom edge)
    - Padded away from left and right screen edges (MarginL = 120px, MarginR = 140px)
    - Short punchy blocks (max 16 chars, max 2 lines) so captions never balloon upwards into the main window.
    - Preserves user subtitle styling (font, size, colors, stroke, karaoke highlights).
    """
    if not words:
        return False

    sub_cfg = subtitle_settings or {}
    font_name = sub_cfg.get("font_name") or font_name or "Impact"
    font_color = sub_cfg.get("font_color") or "#FFFFFF"
    border_color = sub_cfg.get("border_color") or "#000000"
    border_width = float(sub_cfg.get("border_width", 3.0))
    highlight_color = sub_cfg.get("highlight_color") or "#FFD700"
    style_mode = str(sub_cfg.get("style", "karaoke")).lower()
    uppercase = bool(sub_cfg.get("uppercase", True) if "uppercase" in sub_cfg else True)
    position = str(sub_cfg.get("position", "bottom")).lower()

    # Scale font size if from 288p space
    raw_fs = sub_cfg.get("font_size", font_size)
    try:
        raw_fs = int(raw_fs)
    except (ValueError, TypeError):
        raw_fs = font_size
    if raw_fs < 35:
        font_size = int(raw_fs * 2.8)
    else:
        font_size = raw_fs

    # Position alignment
    alignment = 2
    if position == "top":
        alignment = 6
        margin_v = 180
    elif position == "middle":
        alignment = 10
        margin_v = 0
    else:
        alignment = 2

    primary_colour = _hex_to_ass(font_color, 1.0)
    outline_colour = _hex_to_ass(border_color, 1.0)
    back_colour = "&H80000000"

    blocks = []
    current_block = []
    block_start = None

    for w in words:
        w_text = str(w.get("word") or w.get("w") or "").strip()
        if not w_text:
            continue
        if uppercase:
            w_text = w_text.upper()
        w_start = float(w.get("start", w.get("s", 0.0)))
        w_end = float(w.get("end", w.get("e", 0.0)))

        if not current_block:
            current_block = [{"word": w_text, "start": w_start, "end": w_end}]
            block_start = w_start
            continue

        curr_len = sum(len(x["word"]) + 1 for x in current_block)
        curr_dur = w_end - block_start

        if curr_len + len(w_text) > max_chars or curr_dur > max_duration:
            blocks.append(current_block)
            current_block = [{"word": w_text, "start": w_start, "end": w_end}]
            block_start = w_start
        else:
            current_block.append({"word": w_text, "start": w_start, "end": w_end})

    if current_block:
        blocks.append(current_block)

    if not blocks:
        return False

    def ass_time(sec: float) -> str:
        sec = max(0.0, sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        cs = int(round((sec - int(sec)) * 100))
        if cs >= 100:
            cs = 99
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, DisplaySpacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_colour},{primary_colour},{outline_colour},{back_colour},-1,0,0,0,100,100,0,0,1,{border_width:.1f},0,{alignment},{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    hi_inline = _hex_to_ass_inline(highlight_color)
    pri_inline = _hex_to_ass_inline(font_color)

    if style_mode == "karaoke":
        for block in blocks:
            for i, word in enumerate(block):
                ev_start = block[0]["start"] if i == 0 else word["start"]
                ev_end = block[i + 1]["start"] if i < len(block) - 1 else block[-1]["end"]
                if ev_end <= ev_start:
                    continue
                parts = []
                for j, other in enumerate(block):
                    txt = other["word"]
                    if j == i:
                        parts.append(f"{{\\c{hi_inline}}}{txt}{{\\c{pri_inline}}}")
                    else:
                        parts.append(txt)
                b_txt = " ".join(parts)
                w_list = b_txt.split(" ")
                if len(w_list) > 2 and len(" ".join(x["word"] for x in block)) > 13:
                    mid = len(w_list) // 2
                    b_txt = " ".join(w_list[:mid]) + "\\N" + " ".join(w_list[mid:])
                events.append(f"Dialogue: 0,{ass_time(ev_start)},{ass_time(ev_end)},Default,,0,0,0,,{b_txt}")
    else:
        for block in blocks:
            b_s = block[0]["start"]
            b_e = block[-1]["end"]
            b_txt = " ".join(x["word"] for x in block)
            w_list = b_txt.split()
            if len(w_list) > 2 and len(b_txt) > 13:
                mid = len(w_list) // 2
                b_txt = " ".join(w_list[:mid]) + "\\N" + " ".join(w_list[mid:])
            events.append(f"Dialogue: 0,{ass_time(b_s)},{ass_time(b_e)},Default,,0,0,0,,{b_txt}")

    with open(output_ass_path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n".join(events) + "\n")

    return os.path.exists(output_ass_path) and os.path.getsize(output_ass_path) > 0


def generate_subtitles_srt(
    words: List[Dict[str, Any]],
    output_srt_path: str,
    max_chars: int = 16,
    max_duration: float = 1.6
) -> bool:
    """Convert word-level transcription timestamps into a standard UTF-8 SRT file for vertical shorts."""
    if not words:
        return False

    blocks = []
    current_block = []
    block_start = None

    for w in words:
        w_text = str(w.get("word") or w.get("w") or "").strip()
        if not w_text:
            continue
        w_start = float(w.get("start", w.get("s", 0.0)))
        w_end = float(w.get("end", w.get("e", 0.0)))

        if not current_block:
            current_block = [(w_text, w_start, w_end)]
            block_start = w_start
            continue

        curr_len = sum(len(x[0]) + 1 for x in current_block)
        curr_dur = w_end - block_start

        if curr_len + len(w_text) > max_chars or curr_dur > max_duration:
            blocks.append((block_start, current_block[-1][2], " ".join(x[0] for x in current_block)))
            current_block = [(w_text, w_start, w_end)]
            block_start = w_start
        else:
            current_block.append((w_text, w_start, w_end))

    if current_block:
        blocks.append((block_start, current_block[-1][2], " ".join(x[0] for x in current_block)))

    if not blocks:
        return False

    def format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt_lines = []
    for idx, (b_start, b_end, b_text) in enumerate(blocks, 1):
        srt_lines.append(str(idx))
        srt_lines.append(f"{format_srt_time(b_start)} --> {format_srt_time(b_end)}")
        srt_lines.append(b_text)
        srt_lines.append("")

    with open(output_srt_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(srt_lines) + "\n")

    return os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0


# Silence detection defaults
MIN_SILENCE_SECONDS = 0.35  # >350ms threshold for dead silence
BREATH_MARGIN_SECONDS = 0.06  # 60ms natural breath padding around speech
SILENCE_NOISE_DB = -30.0

# Dynamic zoom defaults
DEFAULT_MAX_ZOOM = 1.20     # Centered between 1.15x and 1.25x
RISE_SECONDS = 0.25
HOLD_SECONDS = 1.40
FALL_SECONDS = 0.55

# Analysis width for fast MediaPipe face tracking
ANALYSIS_WIDTH = 480

FILLER_WORDS_DEFAULT = ["um", "uh", "uhm", "er", "ah", "like", "you know"]


@dataclass
class AutoEditConfig:
    """Configurable options for the comprehensive Auto Edit engine."""
    # 1. Dynamic Camera Movements, Reframing & Stabilization
    speaker_tracking: bool = True
    camera_stabilization: bool = True
    deadzone_ratio: float = 0.06  # 5-8% frame deadzone threshold (eliminates micro-jitter/shaking)
    ema_alpha: float = 0.12  # Exponential moving average smoothing (~0.1-0.15)
    pan_window_frames: int = 19  # Smooth pan transitions across 15-20 frames
    tilt_headroom_ratio: float = 0.33  # Golden-ratio headroom (~33% from top of crop box)
    punch_in_zooms: bool = True
    max_zoom: float = 1.20  # 1.15x - 1.25x
    zoom_cadence_mode: str = "controlled"  # 'controlled' (strictly 4-6x per minute) or 'legacy'
    min_zoom_cooldown_s: float = 8.0  # At least 8s cooldown between dynamic zoom events
    zoom_hold_s: float = 2.5  # Hold zoom for 2.0 to 3.5 seconds
    max_zooms_per_minute: int = 6
    min_zooms_per_minute: int = 4
    ken_burns_drift: bool = True  # Subtle drift (auto-muted during controlled cadence mode)
    ken_burns_rate: float = 0.0025
    multi_speaker_mode: str = "auto"  # 'auto', 'single', 'switch'
    depth_of_field_blur: bool = False

    # 2. Temporal Editing & Pacing Control
    smart_silence_trimming: bool = True
    min_silence_s: float = 0.35  # >350ms
    breath_margin_s: float = 0.06
    filler_word_cutting: bool = True
    filler_words: List[str] = field(default_factory=lambda: list(FILLER_WORDS_DEFAULT))
    speed_ramping: bool = False
    speed_ramp_factor: float = 2.0

    # 3. Canvas Positioning & Resizable Foreground
    scale_factor: float = 1.0  # 0.5x to 1.5x sizing of foreground video
    offset_x: int = 0  # Horizontal pixel offset from center
    offset_y: int = 0  # Vertical pixel offset from center
    position_x: Optional[float] = None  # Explicit normalized horizontal position (0.0 to 1.0)
    position_y: Optional[float] = None  # Explicit normalized vertical position (0.0 to 1.0)
    anchor: str = "center"  # 'center', 'top', 'bottom', 'custom'

    # 4. Visual Cuts & Transitions
    jump_cut_disguises: bool = True  # Alternate framing (wide vs medium-close) across cuts
    motion_transitions: bool = True  # Directional whip-pan / push offsets at cuts

    # 5. Color, Lighting & Visual Polish
    visual_polish: bool = True
    contrast: float = 1.05
    brightness: float = 0.01
    saturation: float = 1.06
    vignette: bool = True
    vignette_angle: float = 0.35  # Subtle cinematic edge vignette

    # 6. Audio Enhancements & Normalization
    audio_denoise: bool = True
    denoise_nr: int = 12  # afftdn noise reduction dB
    denoise_nf: int = -30  # afftdn noise floor dB
    loudnorm: bool = True
    loudnorm_i: float = -14.0  # EBU R128 integrated loudness (-14 LUFS for Shorts/Reels)
    loudnorm_tp: float = -1.0  # True peak -1.0 dBFS
    loudnorm_lra: float = 11.0
    transition_sfx: bool = True
    sfx_volume: float = 0.22
    sfx_path: Optional[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "whoosh_sfx.wav")

    # 7. Captions & Invisible Boundary Safe Zone
    burn_subtitles: bool = True
    subtitles_style: Optional[str] = None
    subtitle_settings: Optional[Dict[str, Any]] = None  # User's preserved subtitle styling
    caption_margin_v: int = 320  # Safe vertical margin from bottom edge (prevents overlap with platform UI)
    caption_margin_l: int = 120  # Safe left margin (prevents text clipping on left edge)
    caption_margin_r: int = 140  # Safe right margin (prevents overlap with right-side action buttons)
    caption_font_size: int = 50  # Balanced font size in 1080x1920 space
    caption_max_chars: int = 16  # Short punchy lines (prevents vertical ballooning into main window)


def get_auto_edit_config(
    config_data: Optional[Union[Dict[str, Any], AutoEditConfig]] = None,
    **kwargs
) -> AutoEditConfig:
    """Instantiate and merge AutoEditConfig from dictionary or object with keyword overrides."""
    if isinstance(config_data, AutoEditConfig):
        cfg = copy.deepcopy(config_data)
    else:
        cfg = AutoEditConfig()
        if config_data and isinstance(config_data, dict):
            for k, v in config_data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    # Apply legacy/explicit overrides
    for k, v in kwargs.items():
        if v is not None:
            if k == "breath_margin":
                k = "breath_margin_s"
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def detect_filler_word_intervals(
    transcript_words: Optional[List[Dict[str, Any]]],
    filler_words: Optional[List[str]] = None,
    padding_s: float = 0.04
) -> List[Tuple[float, float]]:
    """Detect filler words ('um', 'uh', 'you know', etc.) from word-level transcript.
    Returns list of (start_s, end_s) intervals to cleanly excise speech disfluencies.
    """
    if not transcript_words:
        return []

    targets = set(w.lower().strip() for w in (filler_words or FILLER_WORDS_DEFAULT))
    filler_intervals: List[Tuple[float, float]] = []

    for w_obj in transcript_words:
        w_raw = str(w_obj.get("word") or w_obj.get("w") or "").strip()
        w_clean = re.sub(r"[^\w\s]", "", w_raw).strip().lower()
        if not w_clean:
            continue

        s = float(w_obj.get("start", w_obj.get("s", 0.0)))
        e = float(w_obj.get("end", w_obj.get("e", 0.0)))
        dur = e - s

        is_filler = False
        if w_clean in {"um", "uh", "uhm", "er", "ah"}:
            is_filler = True
        elif w_clean in targets and dur < 0.65:
            is_filler = True

        if is_filler and 0.08 <= dur <= 1.20:
            cut_s = max(0.0, s - padding_s)
            cut_e = e + padding_s
            filler_intervals.append((round(cut_s, 4), round(cut_e, 4)))

    return filler_intervals


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
    silences: Optional[List[Tuple[float, float]]] = None,
    filler_intervals: Optional[List[Tuple[float, float]]] = None,
    min_silence_s: float = MIN_SILENCE_SECONDS,
    breath_margin: float = BREATH_MARGIN_SECONDS
) -> List[Tuple[float, float]]:
    """Convert silence and filler word intervals into kept segments.
    Splices out dead silence (> min_silence_s) and filler words while preserving natural breath padding.
    """
    if duration <= 0:
        return [(0.0, max(0.0, duration))]

    all_cuts: List[Tuple[float, float]] = []

    # 1. Process silence intervals
    if silences:
        for s_start, s_end in silences:
            s_start = max(0.0, min(duration, s_start))
            s_end = max(0.0, min(duration, s_end))
            if s_end - s_start >= min_silence_s:
                cut_start = s_start + breath_margin
                cut_end = s_end - breath_margin
                if cut_end - cut_start >= 0.10:
                    all_cuts.append((cut_start, cut_end))

    # 2. Process filler word intervals
    if filler_intervals:
        for f_start, f_end in filler_intervals:
            f_start = max(0.0, min(duration, f_start))
            f_end = max(0.0, min(duration, f_end))
            if f_end - f_start >= 0.08:
                all_cuts.append((f_start, f_end))

    if not all_cuts:
        return [(0.0, duration)]

    # 3. Merge overlapping cuts or cuts within 100ms
    merged_cuts: List[List[float]] = []
    for c_start, c_end in sorted(all_cuts, key=lambda x: x[0]):
        if not merged_cuts:
            merged_cuts.append([c_start, c_end])
        else:
            prev = merged_cuts[-1]
            if c_start <= prev[1] + 0.10:
                prev[1] = max(prev[1], c_end)
            else:
                merged_cuts.append([c_start, c_end])

    # 4. Invert cuts into kept segments
    kept: List[Tuple[float, float]] = []
    curr = 0.0
    for c_start, c_end in merged_cuts:
        if c_start > curr + 0.20:  # Minimum 200ms segment duration
            kept.append((round(curr, 4), round(c_start, 4)))
        curr = c_end

    if curr < duration - 0.20:
        kept.append((round(curr, 4), round(duration, 4)))

    return kept if kept else [(0.0, duration)]


def trim_pacing_segments(
    input_path: str,
    output_path: str,
    kept: List[Tuple[float, float]]
) -> bool:
    """Splice video to kept segments with smooth 20ms anti-pop audio micro-fades (afade in/out)."""
    if len(kept) <= 1:
        return False

    has_audio = check_audio_stream(input_path)
    print(f"   ✂️ Trimming pacing cuts ({len(kept) - 1} spliced cut points, audio_stream={has_audio})")

    filter_parts = []
    concat_inputs = []
    for idx, (seg_s, seg_e) in enumerate(kept):
        dur = seg_e - seg_s
        fade_d = min(0.02, max(0.001, dur / 2.0))
        fade_out_st = max(0.0, dur - fade_d)
        if has_audio:
            filter_parts.append(
                f"[0:v]trim=start={seg_s:.4f}:end={seg_e:.4f},setpts=PTS-STARTPTS[v{idx}];"
                f"[0:a]atrim=start={seg_s:.4f}:end={seg_e:.4f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={fade_d:.4f},afade=t=out:st={fade_out_st:.4f}:d={fade_d:.4f}[a{idx}];"
            )
            concat_inputs.append(f"[v{idx}][a{idx}]")
        else:
            filter_parts.append(
                f"[0:v]trim=start={seg_s:.4f}:end={seg_e:.4f},setpts=PTS-STARTPTS[v{idx}];"
            )
            concat_inputs.append(f"[v{idx}]")

    if has_audio:
        filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(kept)}:v=1:a=1[outv][outa]")
        maps = ["-map", "[outv]", "-map", "[outa]"]
        audio_args = list(AUDIO_ENCODE_ARGS)
    else:
        filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(kept)}:v=1:a=0[outv]")
        maps = ["-map", "[outv]"]
        audio_args = []

    filter_complex = "".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-filter_complex", filter_complex,
        *maps,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        *audio_args,
        *METADATA_SCRUB,
        output_path
    ]

    try:
        run_ffmpeg_command(cmd)
        ensure_file_unlocked(output_path, timeout=15)
        validate_output_streams(output_path, expect_audio=has_audio)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Pacing trimming FFmpeg failed:")
        print(format_ffmpeg_error(e, max_lines=30))
        return False
    except Exception as e:
        print(f"   ⚠️ Pacing trimming post-processing warning: {e}")
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def trim_silence_pacing(
    input_path: str,
    output_path: str,
    silences: List[Tuple[float, float]],
    duration: float,
    breath_margin: float = BREATH_MARGIN_SECONDS
) -> bool:
    """Splice out silence intervals > 350-400ms with smooth 20ms audio micro-transitions
    (afade in/out) to eliminate pops between cuts, while preserving AV sync.
    """
    kept = compute_kept_segments(duration, silences, breath_margin=breath_margin)
    if len(kept) <= 1:
        return False
    return trim_pacing_segments(input_path, output_path, kept)


def stabilize_camera_centers(
    raw_centers: List[Tuple[float, float]],
    orig_w: int,
    orig_h: int,
    deadzone_ratio: float = 0.06,
    ema_alpha: float = 0.12,
    pan_window_frames: int = 19,
) -> List[Tuple[int, int]]:
    """Stabilizes face-tracking camera centers to eliminate continuous screen shaking:
    1. Deadzone threshold (5-8% of frame size): Keeps camera stationary unless subject moves outside deadzone box.
    2. Exponential Moving Average (EMA) smoothing: alpha ~ 0.10 - 0.15.
    3. Multi-frame rolling window smoothing across 15-20 frames for silky pan transitions without jerkiness.
    """
    if not raw_centers:
        return [(orig_w // 2, orig_h // 2)]

    dz_w = max(10.0, orig_w * float(deadzone_ratio))
    dz_h = max(10.0, orig_h * float(deadzone_ratio))

    # Phase 1: Deadzone thresholding
    dz_centers: List[Tuple[float, float]] = []
    curr_target_x, curr_target_y = float(raw_centers[0][0]), float(raw_centers[0][1])

    for cx, cy in raw_centers:
        diff_x = cx - curr_target_x
        if abs(diff_x) > dz_w:
            curr_target_x = cx - float(np.sign(diff_x)) * dz_w

        diff_y = cy - curr_target_y
        if abs(diff_y) > dz_h:
            curr_target_y = cy - float(np.sign(diff_y)) * dz_h

        dz_centers.append((curr_target_x, curr_target_y))

    # Phase 2: Exponential Moving Average (EMA) smoothing
    ema_centers: List[Tuple[float, float]] = []
    sx, sy = dz_centers[0][0], dz_centers[0][1]
    alpha = max(0.01, min(1.0, float(ema_alpha)))

    for cx, cy in dz_centers:
        sx = alpha * cx + (1.0 - alpha) * sx
        sy = alpha * cy + (1.0 - alpha) * sy
        ema_centers.append((sx, sy))

    # Phase 3: Rolling window moving average (smooth pan transitions over 15-20 frames)
    w_size = max(1, int(pan_window_frames))
    half_w = w_size // 2
    n = len(ema_centers)

    all_x = [pt[0] for pt in ema_centers]
    all_y = [pt[1] for pt in ema_centers]

    if w_size > 1 and n > w_size:
        pad_x = [all_x[0]] * half_w + all_x + [all_x[-1]] * half_w
        pad_y = [all_y[0]] * half_w + all_y + [all_y[-1]] * half_w
        kernel = np.ones(w_size) / float(w_size)
        smooth_x = np.convolve(pad_x, kernel, mode="valid")[:n]
        smooth_y = np.convolve(pad_y, kernel, mode="valid")[:n]
    else:
        smooth_x = all_x
        smooth_y = all_y

    final_centers: List[Tuple[int, int]] = []
    for ix_f, iy_f in zip(smooth_x, smooth_y):
        ix = int(round(max(0, min(orig_w, ix_f))))
        iy = int(round(max(0, min(orig_h, iy_f))))
        final_centers.append((ix, iy))

    return final_centers


def track_active_speaker(
    video_path: str,
    orig_w: int,
    orig_h: int,
    n_frames: int,
    fps: float,
    sample_step: int = 2,
    multi_speaker_mode: str = "auto",
    deadzone_ratio: float = 0.06,
    ema_alpha: float = 0.12,
    pan_window_frames: int = 19,
    camera_stabilization: bool = True,
) -> List[Tuple[int, int]]:
    """MediaPipe face detection to track active speaker center (cx, cy) across frames.
    Supports single speaker tracking and multi-speaker awareness with deadzone thresholding,
    EMA smoothing, and rolling window pan transitions for rock-solid camera stabilization.
    """
    default_center = (orig_w // 2, orig_h // 2)
    if n_frames <= 0:
        return [default_center]

    centers: List[Tuple[float, float]] = []

    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.40)
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

                    # Find dominant speaker
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        area = bbox.width * bbox.height
                        cand_cx = (bbox.xmin + bbox.width / 2.0) * orig_w
                        cand_cy = (bbox.ymin + bbox.height / 2.0) * orig_h
                        dist = abs(cand_cx - last_cx) + abs(cand_cy - last_cy)

                        # Balance speaker face size with camera tracking continuity
                        if multi_speaker_mode == "switch":
                            score = area * 1200.0 - dist * 0.05
                        else:
                            score = area * 1000.0 - dist * 0.15

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

    if camera_stabilization:
        return stabilize_camera_centers(
            centers,
            orig_w,
            orig_h,
            deadzone_ratio=deadzone_ratio,
            ema_alpha=ema_alpha,
            pan_window_frames=pan_window_frames,
        )

    # Legacy basic EMA fallback
    alpha = 0.15
    smoothed: List[Tuple[int, int]] = []
    sx, sy = float(centers[0][0]), float(centers[0][1])

    for cx, cy in centers:
        sx = alpha * cx + (1.0 - alpha) * sx
        sy = alpha * cy + (1.0 - alpha) * sy
        ix = int(round(max(0, min(orig_w, sx))))
        iy = int(round(max(0, min(orig_h, sy))))
        smoothed.append((ix, iy))

    return smoothed


HIGH_EMPHASIS_KEYWORDS = {
    "stop", "never", "secret", "important", "must", "always",
    "proven", "mistake", "truth", "warning", "critical", "discover",
    "reveal", "key", "remember", "wait", "listen", "watch", "huge",
    "insane", "crazy", "why", "how", "best", "worst", "danger",
    "shocking", "exposed", "hack", "money", "win", "lose"
}


def compute_contextual_zooms(
    n_frames: int,
    fps: float,
    video_path: Optional[str] = None,
    transcript_words: Optional[List[Dict[str, Any]]] = None,
    max_zoom: float = DEFAULT_MAX_ZOOM,
    zoom_cadence_mode: str = "controlled",
    min_zoom_cooldown_s: float = 8.0,
    zoom_hold_s: float = 2.5,
    max_zooms_per_minute: int = 6,
    min_zooms_per_minute: int = 4,
    keywords: Optional[Any] = None,
) -> List[float]:
    """Compute per-frame zoom factors (1.0 to max_zoom).
    In 'controlled' mode:
    - Dynamic zooms strictly frequency-capped to 4 to 6 times per minute (~one every 10-15s).
    - Minimum 8s cooldown between successive zoom events.
    - Trigger only on high-emphasis keywords, exclamations/questions, whisper/volume velocity, or opening hook.
    - Held for 2.0-4.0s (default 2.5s) then smoothly eased back down to 1.0x with smoothstep easing.
    """
    max_zoom = min(max(max_zoom, 1.15), 1.25)
    zooms = [1.0] * max(0, n_frames)
    if not zooms or fps <= 0:
        return zooms

    duration = n_frames / fps
    if duration < 1.0:
        return zooms

    active_keywords = set(keywords) if keywords is not None else HIGH_EMPHASIS_KEYWORDS

    # 1. Candidate extraction: (timestamp, priority_score, label)
    candidates: List[Tuple[float, float, str]] = []

    # Opening hook punch-in: first 0.6s
    if duration >= 4.0:
        candidates.append((0.6, 100.0, "hook"))

    # Audio energy emphasis beats
    if video_path and os.path.exists(video_path):
        try:
            import punch_in
            beats = punch_in.emphasis_times(
                video_path,
                duration,
                min_gap=min_zoom_cooldown_s,
                prominence=0.55
            )
            for b in beats:
                if 0.5 <= b < duration - 1.5:
                    candidates.append((round(b, 2), 60.0, "audio_beat"))
        except Exception as e:
            print(f"⚠️ Audio emphasis detection error: {e}")

    # Transcript-aware keyword and punctuation emphasis
    if transcript_words:
        for w_obj in transcript_words:
            w_raw = str(w_obj.get("w") or w_obj.get("word") or "").strip()
            w_clean = re.sub(r"[^\w\s]", "", w_raw).strip().lower()
            w_start = float(w_obj.get("s", w_obj.get("start", 0.0)))

            if w_start < 0.5 or w_start >= duration - 1.5:
                continue

            score = 0.0
            label = ""
            if any(p in w_raw for p in ("!", "?")):
                score = 85.0
                label = "punct"
            elif w_clean in active_keywords:
                score = 90.0
                label = f"kw_{w_clean}"

            if score > 0.0:
                candidates.append((round(w_start, 2), score, label))

    # Sort candidates chronologically
    candidates.sort(key=lambda c: c[0])

    # Controlled cadence frequency capping
    if zoom_cadence_mode == "controlled":
        max_allowed = max(1, int(round((duration / 60.0) * max_zooms_per_minute)))
        min_target = max(1, int(round((duration / 60.0) * min_zooms_per_minute)))

        # Group candidates into cooldown clusters
        clusters: List[List[Tuple[float, float, str]]] = []
        for cand in candidates:
            if not clusters:
                clusters.append([cand])
            else:
                if cand[0] - clusters[-1][-1][0] < min_zoom_cooldown_s:
                    clusters[-1].append(cand)
                else:
                    clusters.append([cand])

        # From each cluster, pick the highest score candidate
        cluster_bests: List[Tuple[float, float, str]] = []
        for cl in clusters:
            best = max(cl, key=lambda x: x[1])
            cluster_bests.append(best)

        # Enforce cooldown on cluster bests
        selected_times: List[float] = []
        last_t = -999.0
        for t, score, lbl in cluster_bests:
            if t - last_t >= min_zoom_cooldown_s:
                selected_times.append(t)
                last_t = t
                if len(selected_times) >= max_allowed:
                    break

        # If below min_target, synthesize evenly-spaced zoom points
        if len(selected_times) < min_target:
            step = duration / (min_target + 1)
            for i in range(1, min_target + 1):
                synth_t = round(i * step, 2)
                if 1.0 <= synth_t <= duration - 2.0:
                    if all(abs(synth_t - st) >= min_zoom_cooldown_s for st in selected_times):
                        selected_times.append(synth_t)
                if len(selected_times) >= max_allowed:
                    break

        emphasis_times = sorted(selected_times[:max_allowed])
    else:
        # Legacy mode: simple deduplication
        filtered = []
        last_t = -6.0
        for t, score, lbl in candidates:
            if t - last_t >= 6.0:
                filtered.append(t)
                last_t = t
        emphasis_times = sorted(filtered)

    # 2. Render smoothstep zoom curves
    rise_s = 0.25
    hold_s = max(1.5, min(4.0, float(zoom_hold_s)))
    fall_s = 0.40
    span = rise_s + hold_s + fall_s

    for t in emphasis_times:
        start_f = max(0, int(t * fps))
        end_f = min(n_frames, int((t + span) * fps) + 1)

        for f in range(start_f, end_f):
            dt = f / fps - t
            if dt < 0:
                continue
            elif dt < rise_s:
                amount = _ease(dt / rise_s)
            elif dt < rise_s + hold_s:
                amount = 1.0
            elif dt < span:
                amount = 1.0 - _ease((dt - rise_s - hold_s) / fall_s)
            else:
                continue

            z_val = 1.0 + (max_zoom - 1.0) * amount
            zooms[f] = max(zooms[f], z_val)

    return zooms


def compute_camera_choreography(
    n_frames: int,
    fps: float,
    kept_segments: Optional[List[Tuple[float, float]]] = None,
    video_path: Optional[str] = None,
    transcript_words: Optional[List[Dict[str, Any]]] = None,
    max_zoom: float = DEFAULT_MAX_ZOOM,
    punch_in_zooms: bool = True,
    jump_cut_disguises: bool = True,
    ken_burns_drift: bool = True,
    ken_burns_rate: float = 0.0025,
    motion_transitions: bool = True,
    zoom_cadence_mode: str = "controlled",
    min_zoom_cooldown_s: float = 8.0,
    zoom_hold_s: float = 2.5,
    max_zooms_per_minute: int = 6,
    min_zooms_per_minute: int = 4,
) -> Tuple[List[float], List[int]]:
    """Compute per-frame zoom factors (1.0 to max_zoom) and horizontal motion transition offsets.
    Combines:
    1. Jump-Cut Disguises: Alternates base scale across cut boundaries (disabled in controlled mode to keep baseline steady).
    2. Ken Burns Drift: Continuous subtle slow push-in (muted in controlled mode to eliminate continuous creeping zoom).
    3. Controlled Cadence Punch-In Zooms: High-impact zoom-ins on key statements, beats, and punchlines.
    4. Motion Transitions: Directional whip-pan / push offsets at the beginning of cuts.
    """
    max_zoom = min(max(max_zoom, 1.15), 1.25)
    zooms = [1.0] * max(0, n_frames)
    x_offsets = [0] * max(0, n_frames)
    if not zooms or fps <= 0:
        return zooms, x_offsets

    duration = n_frames / fps

    # Map kept segments to timeline ranges in working clip
    seg_ranges: List[Tuple[float, float]] = []
    if kept_segments and len(kept_segments) > 1:
        curr_t = 0.0
        for seg_s, seg_e in kept_segments:
            seg_dur = seg_e - seg_s
            seg_ranges.append((curr_t, curr_t + seg_dur))
            curr_t += seg_dur
    else:
        seg_ranges = [(0.0, duration)]

    is_controlled = (zoom_cadence_mode == "controlled")
    effective_drift = False if is_controlled else ken_burns_drift

    # 1. Base Framing & Motion Transitions
    for idx, (seg_s, seg_e) in enumerate(seg_ranges):
        base_z = 1.15 if (jump_cut_disguises and len(seg_ranges) > 1 and (idx % 2 == 1)) else 1.00
        f_start = max(0, int(round(seg_s * fps)))
        f_end = min(n_frames, int(round(seg_e * fps)))

        for f in range(f_start, f_end):
            t_in_seg = (f / fps) - seg_s
            drift = min(0.05, ken_burns_rate * t_in_seg) if effective_drift else 0.0
            zooms[f] = base_z + drift

        # Motion transition whip-pan / push offset at segment cut boundary
        if motion_transitions and idx > 0:
            trans_dur = 0.18
            n_trans = int(trans_dur * fps)
            for f in range(f_start, min(f_end, f_start + n_trans)):
                dt = (f / fps) - seg_s
                progress = min(1.0, dt / trans_dur)
                dir_sign = 1 if (idx % 2 == 1) else -1
                dx = int(round(dir_sign * 35.0 * (1.0 - _ease(progress))))
                dx -= dx % 2
                x_offsets[f] = dx

    # 2. Contextual Punch-In Zooms on key hook statements, beats, and punchlines
    if punch_in_zooms:
        punch_zooms = compute_contextual_zooms(
            n_frames,
            fps,
            video_path=video_path,
            transcript_words=transcript_words,
            max_zoom=max_zoom,
            zoom_cadence_mode=zoom_cadence_mode,
            min_zoom_cooldown_s=min_zoom_cooldown_s,
            zoom_hold_s=zoom_hold_s,
            max_zooms_per_minute=max_zooms_per_minute,
            min_zooms_per_minute=min_zooms_per_minute,
        )
        for f in range(n_frames):
            pz = punch_zooms[f]
            if pz > 1.001:
                zooms[f] = max(zooms[f], pz)

    # Clamp and round zooms strictly within [1.0, max_zoom]
    zooms = [min(max_zoom, max(1.0, round(z, 4))) for z in zooms]
    return zooms, x_offsets


def generate_zoom_boxes(
    centers: List[Tuple[int, int]],
    zooms: List[float],
    orig_w: int = 1080,
    orig_h: int = 1920,
    headroom_ratio: float = 0.33,
    x_offsets: Optional[List[int]] = None
) -> List[Tuple[int, int, int, int]]:
    """Calculate per-frame (crop_w, crop_h, crop_x, crop_y) with golden-ratio headroom positioning.
    All dimensions and offsets are strictly even.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    n = min(len(centers), len(zooms))
    offsets = x_offsets if (x_offsets and len(x_offsets) == n) else [0] * n

    for i in range(n):
        z = zooms[i]
        w = int(orig_w / z)
        h = int(orig_h / z)
        w -= w % 2
        h -= h % 2
        w = max(2, min(w, orig_w))
        h = max(2, min(h, orig_h))

        cx, cy = centers[i]
        x = int(round(cx - w / 2.0)) + offsets[i]
        # Golden ratio headroom positioning:
        # Places face center at headroom_ratio (~33%) from top of crop box
        y = int(round(cy - h * headroom_ratio))

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
    target: str = "crop@c",
    deadband_px: int = 3,
) -> List[str]:
    """Generate deduplicated sendcmd lines for FFmpeg dynamic crop.
    Applies deadband threshold (deadband_px) so micro-jitter (<3px) doesn't flood FFmpeg with commands.
    """
    lines = []
    prev = None

    for i, box in enumerate(boxes):
        if prev is None:
            t = i / fps
            w, h, x, y = box
            lines.append(f"{t:.4f} {target} w {w};")
            lines.append(f"{t:.4f} {target} h {h};")
            lines.append(f"{t:.4f} {target} x {x};")
            lines.append(f"{t:.4f} {target} y {y};")
            prev = box
            continue
        w, h, x, y = box
        pw, ph, px, py = prev

        dw = abs(w - pw)
        dh = abs(h - ph)
        dx = abs(x - px)
        dy = abs(y - py)

        if dw < deadband_px and dh < deadband_px and dx < deadband_px and dy < deadband_px:
            continue

        t = i / fps
        effective_w = w if dw >= deadband_px else pw
        effective_h = h if dh >= deadband_px else ph
        effective_x = x if dx >= deadband_px else px
        effective_y = y if dy >= deadband_px else py

        if effective_w != pw:
            lines.append(f"{t:.4f} {target} w {effective_w};")
        if effective_h != ph:
            lines.append(f"{t:.4f} {target} h {effective_h};")
        if effective_x != px:
            lines.append(f"{t:.4f} {target} x {effective_x};")
        if effective_y != py:
            lines.append(f"{t:.4f} {target} y {effective_y};")

        prev = (effective_w, effective_h, effective_x, effective_y)

    return lines


def auto_edit_clip(
    input_clip_path: str,
    output_clip_path: str,
    transcript_words: Optional[List[Dict[str, Any]]] = None,
    max_zoom: Optional[float] = None,
    min_silence_s: Optional[float] = None,
    breath_margin: Optional[float] = None,
    precomputed_silences: Optional[List[Any]] = None,
    precomputed_centers: Optional[List[Tuple[int, int]]] = None,
    burn_subtitles: Optional[bool] = None,
    subtitles_style: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], AutoEditConfig]] = None,
) -> Dict[str, Any]:
    """Execute the Stage 2 On-Demand Auto Edit pipeline:
    1. Silence & Pacing Cuts: Trims dead silence (>350ms) and filler words with anti-pop audio micro-fades.
    2. MediaPipe Active Speaker Tracking: Smoothly glides crop box with golden-ratio headroom positioning.
    3. Camera Movements & Choreography:
       - Contextual dynamic punch-in zooms (1.15x–1.25x).
       - Ken Burns drift (~0.25%/s slow push-in across sustained segments).
       - Jump-cut disguises (alternating wide/medium scale across cuts).
       - Directional whip-pan / push motion transitions.
    4. Color & Visual Polish: Dynamic edge vignette and auto-contrast/saturation balancing.
    5. Audio Enhancements & Normalization:
       - Voice isolation & spectral de-noising (afftdn).
       - Transition & zoom whoosh sound effects mixing.
       - EBU R128 loudness normalization (loudnorm -14 LUFS).
    6. Subtitle Offset & Burn-In: Recalculates and offsets word timestamps to match cuts.
    7. Explicit Audio Mapping & AV Sync: Full audio-video synchronization with stream validation.
    """
    ensure_file_unlocked(input_clip_path)
    if not os.path.exists(input_clip_path):
        raise FileNotFoundError(f"Input clip not found: {input_clip_path}")

    # Build active configuration
    cfg = get_auto_edit_config(
        config,
        max_zoom=max_zoom,
        min_silence_s=min_silence_s,
        breath_margin=breath_margin,
        burn_subtitles=burn_subtitles,
        subtitles_style=subtitles_style
    )

    print(f"\n⚡ [Stage 2] Starting Auto Edit on: {input_clip_path}")
    print(f"   ⚙️ Config: max_zoom={cfg.max_zoom:.2f}, jump_cut_disguises={cfg.jump_cut_disguises}, "
          f"ken_burns={cfg.ken_burns_drift}, visual_polish={cfg.visual_polish}, loudnorm={cfg.loudnorm}")

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
    srt_sub_path = os.path.join(workdir, "temp_subs.srt")
    ass_sub_path = os.path.join(workdir, "temp_subs.ass")

    silence_cuts_count = 0
    filler_cuts_count = 0
    kept: List[Tuple[float, float]] = [(0.0, duration)]

    try:
        # Step 0: Ensure transcript words are loaded for filler cutting and subtitles
        if transcript_words is None:
            clip_dir = os.path.dirname(input_clip_path)
            clip_name = os.path.basename(input_clip_path)
            try:
                import metadata_extractor
                real_meta = metadata_extractor.load_clip_metadata(clip_dir, clip_name)
                if not real_meta:
                    import app
                    base_cut = app.get_base_cut_filename(clip_dir, clip_name)
                    real_meta = metadata_extractor.load_clip_metadata(clip_dir, base_cut)
                if real_meta and real_meta.get("words"):
                    transcript_words = real_meta["words"]
            except Exception as e:
                print(f"   ⚠️ Could not load cached transcript words: {e}")

            if transcript_words is None:
                import glob
                meta_files = glob.glob(os.path.join(clip_dir, "*_metadata.json"))
                main_meta = [f for f in meta_files if not f.endswith("_real_metadata.json")]
                if main_meta:
                    try:
                        with open(main_meta[0], "r", encoding="utf-8") as mf:
                            m_data = json.load(mf)
                            tx = m_data.get("transcript") or {}
                            all_w = []
                            for seg in tx.get("segments", []):
                                for w in seg.get("words", []):
                                    all_w.append(w)
                            if all_w:
                                transcript_words = all_w
                    except Exception as e:
                        print(f"   ⚠️ Could not read transcript from metadata JSON: {e}")

        # Step 1: Detect and trim dead silence (>350ms) and filler words
        silences: List[Tuple[float, float]] = []
        if cfg.smart_silence_trimming:
            if precomputed_silences is not None:
                silences = [
                    (float(s["start"]), float(s["end"])) if isinstance(s, dict) else (float(s[0]), float(s[1]))
                    for s in precomputed_silences
                ]
            else:
                silences = detect_silence_intervals(input_clip_path, min_silence_s=cfg.min_silence_s)

        filler_intervals: List[Tuple[float, float]] = []
        if cfg.filler_word_cutting and transcript_words:
            filler_intervals = detect_filler_word_intervals(transcript_words, cfg.filler_words)
            filler_cuts_count = len(filler_intervals)
            if filler_cuts_count > 0:
                print(f"   🗣️ Detected {filler_cuts_count} speech filler word interval(s)")

        has_pacing_cuts = False
        kept_cands = compute_kept_segments(
            duration,
            silences=silences,
            filler_intervals=filler_intervals,
            min_silence_s=cfg.min_silence_s,
            breath_margin=cfg.breath_margin_s,
        )

        if len(kept_cands) > 1:
            has_pacing_cuts = trim_pacing_segments(input_clip_path, trimmed_clip_path, kept_cands)
            if has_pacing_cuts:
                kept = kept_cands

        active_working_clip = trimmed_clip_path if has_pacing_cuts else input_clip_path
        if has_pacing_cuts:
            silence_cuts_count = max(0, len(kept) - 1 - filler_cuts_count)
            # Re-probe working clip properties
            with open_video_capture(active_working_clip) as cap:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)
                orig_w -= orig_w % 2
                orig_h -= orig_h % 2

        # Step 2: MediaPipe active speaker tracking with multi-speaker support
        if not has_pacing_cuts and precomputed_centers and len(precomputed_centers) == total_frames:
            print(f"   👤 Using precomputed speaker tracking across {total_frames} frames...")
            if cfg.camera_stabilization:
                centers = stabilize_camera_centers(
                    precomputed_centers,
                    orig_w,
                    orig_h,
                    deadzone_ratio=cfg.deadzone_ratio,
                    ema_alpha=cfg.ema_alpha,
                    pan_window_frames=cfg.pan_window_frames,
                )
            else:
                centers = precomputed_centers
        else:
            print(f"   👤 Tracking active speaker with MediaPipe across {total_frames} frames...")
            centers = track_active_speaker(
                active_working_clip,
                orig_w,
                orig_h,
                total_frames,
                fps,
                multi_speaker_mode=cfg.multi_speaker_mode,
                deadzone_ratio=cfg.deadzone_ratio,
                ema_alpha=cfg.ema_alpha,
                pan_window_frames=cfg.pan_window_frames,
                camera_stabilization=cfg.camera_stabilization,
            )

        # Step 3: Compute camera choreography (zooms, drift, jump-cut disguises, motion transitions)
        print(f"   🎥 Choreographing camera movements (punch-ins={cfg.punch_in_zooms}, "
              f"cadence_mode={cfg.zoom_cadence_mode}, stabilization={cfg.camera_stabilization})...")
        zooms, x_offsets = compute_camera_choreography(
            total_frames,
            fps,
            kept_segments=kept,
            video_path=active_working_clip,
            transcript_words=transcript_words,
            max_zoom=cfg.max_zoom,
            punch_in_zooms=cfg.punch_in_zooms,
            jump_cut_disguises=cfg.jump_cut_disguises,
            ken_burns_drift=cfg.ken_burns_drift,
            ken_burns_rate=cfg.ken_burns_rate,
            motion_transitions=cfg.motion_transitions,
            zoom_cadence_mode=cfg.zoom_cadence_mode,
            min_zoom_cooldown_s=cfg.min_zoom_cooldown_s,
            zoom_hold_s=cfg.zoom_hold_s,
            max_zooms_per_minute=cfg.max_zooms_per_minute,
            min_zooms_per_minute=cfg.min_zooms_per_minute,
        )

        boxes = generate_zoom_boxes(
            centers,
            zooms,
            orig_w=orig_w,
            orig_h=orig_h,
            headroom_ratio=cfg.tilt_headroom_ratio,
            x_offsets=x_offsets
        )
        lines = generate_sendcmd_lines(boxes, fps, deadband_px=3)
        zooms_applied_count = len([z for z in zooms if z > 1.01])

        with open(cmd_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Step 4: Subtitle Timing Recalculation & Generation (Preserving Customizer Styles)
        input_already_has_subtitles = "subtitled_" in os.path.basename(input_clip_path)
        has_subtitles = False
        if not input_already_has_subtitles and cfg.burn_subtitles and transcript_words:
            # Exclude excised filler words from displayed captions
            active_words = transcript_words
            if cfg.filler_word_cutting and filler_intervals:
                clean_words = []
                for w in transcript_words:
                    s_w = float(w.get("start", w.get("s", 0.0)))
                    e_w = float(w.get("end", w.get("e", 0.0)))
                    # Skip words falling entirely within excised filler intervals
                    if not any(f_s <= s_w and e_w <= f_e + 0.05 for f_s, f_e in filler_intervals):
                        clean_words.append(w)
                active_words = clean_words or transcript_words

            remapped_words = remap_word_timestamps(active_words, kept)
            has_subtitles = generate_subtitles_ass(
                remapped_words,
                ass_sub_path,
                max_chars=cfg.caption_max_chars,
                margin_v=cfg.caption_margin_v,
                margin_l=cfg.caption_margin_l,
                margin_r=cfg.caption_margin_r,
                font_size=cfg.caption_font_size,
                subtitle_settings=cfg.subtitle_settings,
            )
            generate_subtitles_srt(remapped_words, srt_sub_path, max_chars=cfg.caption_max_chars)
            if has_subtitles:
                print(f"   💬 Generated {len(remapped_words)} remapped caption words for burn-in (custom styling & safe boundaries preserved)")

        # Step 5: Construct FFmpeg filtergraph (canvas positioning + dynamic crop + subtitles + audio polish)
        first_box = boxes[0] if boxes else (orig_w, orig_h, 0, 0)
        init_crop = f"w={first_box[0]}:h={first_box[1]}:x={first_box[2]}:y={first_box[3]}"

        has_canvas_transform = (
            abs(float(cfg.scale_factor) - 1.0) > 0.01
            or int(cfg.offset_x) != 0
            or int(cfg.offset_y) != 0
            or str(cfg.anchor).lower() != "center"
            or cfg.position_x is not None
            or cfg.position_y is not None
        )

        if has_canvas_transform:
            scale_f = max(0.5, min(1.5, float(cfg.scale_factor)))
            fg_w = int(round(1080 * scale_f))
            fg_h = int(round(1920 * scale_f))
            fg_w -= fg_w % 2
            fg_h -= fg_h % 2

            anchor_mode = str(cfg.anchor).lower()
            if anchor_mode == "top":
                base_x = (1080 - fg_w) // 2
                base_y = int(1920 * 0.08)
            elif anchor_mode == "bottom":
                base_x = (1080 - fg_w) // 2
                base_y = int(1920 * 0.92 - fg_h)
            elif anchor_mode == "custom" or cfg.position_x is not None or cfg.position_y is not None:
                px = float(cfg.position_x) if cfg.position_x is not None else 0.5
                py = float(cfg.position_y) if cfg.position_y is not None else 0.5
                base_x = int(round(px * (1080 - fg_w)))
                base_y = int(round(py * (1920 - fg_h)))
            else:  # 'center'
                base_x = (1080 - fg_w) // 2
                base_y = (1920 - fg_h) // 2

            final_x = base_x + int(cfg.offset_x)
            final_y = base_y + int(cfg.offset_y)
            final_x -= final_x % 2
            final_y -= final_y % 2

            fg_filters = [
                f"sendcmd=f='{escape_filter_value(cmd_file_path)}'",
                f"crop@c={init_crop}",
                f"scale={fg_w}:{fg_h}:flags=lanczos",
                "setsar=1",
            ]
            bg_filter = "[in_bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5,eq=brightness=-0.08[bg]"

            post_filters = []
            if cfg.visual_polish:
                post_filters.append(
                    f"eq=contrast={cfg.contrast:.2f}:brightness={cfg.brightness:.2f}:saturation={cfg.saturation:.2f}"
                )
            if cfg.vignette:
                post_filters.append(f"vignette=a={cfg.vignette_angle:.2f}:aspect=9/16")

            post_str = f",{','.join(post_filters)}" if post_filters else ""

            base_video_filter_raw = (
                f"[0:v]split=2[in_fg][in_bg];"
                f"{bg_filter};"
                f"[in_fg]{','.join(fg_filters)}[fg];"
                f"[bg][fg]overlay=x={final_x}:y={final_y}:shortest=1,setsar=1{post_str}"
            )
        else:
            video_filters = [
                f"sendcmd=f='{escape_filter_value(cmd_file_path)}'",
                f"crop@c={init_crop}",
                "scale=1080:1920:flags=lanczos",
                "setsar=1",
            ]
            if cfg.visual_polish:
                video_filters.append(
                    f"eq=contrast={cfg.contrast:.2f}:brightness={cfg.brightness:.2f}:saturation={cfg.saturation:.2f}"
                )
            if cfg.vignette:
                video_filters.append(f"vignette=a={cfg.vignette_angle:.2f}:aspect=9/16")

            base_video_filter_raw = f"[0:v]{','.join(video_filters)}"

        filter_parts = [base_video_filter_raw]

        if not input_already_has_subtitles and cfg.burn_subtitles and has_subtitles and os.path.exists(ass_sub_path) and not (cfg.subtitles_style or subtitles_style):
            esc_ass = escape_filter_value(ass_sub_path)
            filter_parts[0] += f"[v_zoomed];[v_zoomed]ass=filename='{esc_ass}'[vout]"
        elif not input_already_has_subtitles and cfg.burn_subtitles and ((has_subtitles and os.path.exists(ass_sub_path)) or os.path.exists(srt_sub_path)):
            target_sub = srt_sub_path if os.path.exists(srt_sub_path) else ass_sub_path
            esc_sub = escape_filter_value(target_sub)
            style_str = (
                cfg.subtitles_style
                or subtitles_style
                or "FontSize=16,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Alignment=2,MarginV=48,MarginL=22,MarginR=26"
            )
            filter_parts[0] += f"[v_zoomed];[v_zoomed]subtitles=filename='{esc_sub}':force_style='{style_str}'[vout]"
        else:
            filter_parts[0] += "[vout]"

        # Step 6: Audio Processing (Voice Denoise + Whoosh SFX + Loudnorm)
        has_audio = check_audio_stream(active_working_clip)
        input_args = ["-i", active_working_clip]

        if has_audio:
            sfx_file = cfg.sfx_path if (cfg.sfx_path and os.path.exists(cfg.sfx_path)) else None
            whoosh_times: List[float] = []

            if cfg.transition_sfx and sfx_file:
                # Cuts in the active working clip
                if len(kept) > 1:
                    cum_t = 0.0
                    for seg_s, seg_e in kept[:-1]:
                        cum_t += (seg_e - seg_s)
                        if cum_t > 0.4:
                            whoosh_times.append(round(cum_t, 2))
                # Hook punch-in
                if cfg.punch_in_zooms and duration >= 4.0:
                    whoosh_times.append(0.5)

                filtered_whooshes: List[float] = []
                for wt in sorted(list(set(whoosh_times))):
                    if not filtered_whooshes or (wt - filtered_whooshes[-1] >= 3.0):
                        filtered_whooshes.append(wt)
                whoosh_times = filtered_whooshes[:4]

            if whoosh_times and sfx_file:
                input_args.extend(["-i", sfx_file])
                sfx_idx = 1
                sfx_splits = [f"[{sfx_idx}:a]asplit={len(whoosh_times)}" + "".join(f"[s{i}]" for i in range(len(whoosh_times))) + ";"]
                amix_inputs = ["[voice]"]
                for i, wt in enumerate(whoosh_times):
                    ms = int(round(max(0.0, wt - 0.08) * 1000))
                    sfx_splits.append(f"[s{i}]adelay={ms}|{ms},volume={cfg.sfx_volume:.2f}[w{i}];")
                    amix_inputs.append(f"[w{i}]")

                voice_filters = ["asetpts=PTS-STARTPTS"]
                if cfg.audio_denoise:
                    voice_filters.append(f"afftdn=nr={cfg.denoise_nr}:nf={cfg.denoise_nf}:tn=1")

                mix_filters = [f"{''.join(amix_inputs)}amix=inputs={len(amix_inputs)}:dropout_transition=0:normalize=0"]
                if cfg.loudnorm:
                    mix_filters.append(f"loudnorm=I={cfg.loudnorm_i}:TP={cfg.loudnorm_tp}:LRA={cfg.loudnorm_lra}")

                audio_filter = (
                    f"{''.join(sfx_splits)}"
                    f"[0:a]{','.join(voice_filters)}[voice];"
                    f"{','.join(mix_filters)}[aout]"
                )
                filter_parts.append(audio_filter)
            else:
                audio_filters = ["asetpts=PTS-STARTPTS"]
                if cfg.audio_denoise:
                    audio_filters.append(f"afftdn=nr={cfg.denoise_nr}:nf={cfg.denoise_nf}:tn=1")
                if cfg.loudnorm:
                    audio_filters.append(f"loudnorm=I={cfg.loudnorm_i}:TP={cfg.loudnorm_tp}:LRA={cfg.loudnorm_lra}")
                filter_parts.append(f"[0:a]{','.join(audio_filters)}[aout]")

            filter_graph = ";".join(filter_parts)
            map_args = ["-map", "[vout]", "-map", "[aout]"]
            audio_args = list(AUDIO_ENCODE_ARGS)
        else:
            filter_graph = filter_parts[0]
            map_args = ["-map", "[vout]"]
            audio_args = []

        render_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            *input_args,
            "-filter_complex", filter_graph,
            *map_args,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            *audio_args,
            *METADATA_SCRUB,
            output_clip_path
        ]

        print(f"   🎬 Rendering auto-edited clip: {output_clip_path} (audio={has_audio}, subtitles={has_subtitles})")
        try:
            run_ffmpeg_command(render_cmd)
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Initial render attempt failed:")
            print(format_ffmpeg_error(e, max_lines=30))
            if cfg.burn_subtitles and has_subtitles:
                print(f"   ⚠️ Retrying render without burned subtitles fallback...")
                filter_parts_fallback = [f"{base_video_filter_raw}[vout]"]
                if has_audio:
                    filter_parts_fallback.append(filter_parts[1])
                filter_graph_fallback = ";".join(filter_parts_fallback)
                render_cmd_fallback = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    *input_args,
                    "-filter_complex", filter_graph_fallback,
                    *map_args,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    *audio_args,
                    *METADATA_SCRUB,
                    output_clip_path
                ]
                run_ffmpeg_command(render_cmd_fallback)
                has_subtitles = False
            else:
                raise

        ensure_file_unlocked(output_clip_path, timeout=15)
        val_info = validate_output_streams(output_clip_path, expect_audio=has_audio)

        print(f"   ✅ Auto Edit completed successfully! (has_audio={val_info['has_audio']}, "
              f"dur={val_info['video_duration']:.2f}s, subtitles={has_subtitles})")
        return {
            "success": True,
            "output_path": output_clip_path,
            "silence_cuts": silence_cuts_count,
            "filler_cuts": filler_cuts_count,
            "zooms_applied": zooms_applied_count > 0,
            "has_audio": val_info["has_audio"],
            "has_subtitles": has_subtitles or input_already_has_subtitles,
            "duration": val_info["video_duration"] or (total_frames / fps if fps > 0 else duration),
            "config": asdict(cfg) if is_dataclass(cfg) else cfg,
        }

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else "Unknown FFmpeg error"
        print(f"   ❌ FFmpeg Auto Edit error: {err_msg[:400]}")
        raise RuntimeError(f"Auto Edit FFmpeg failed: {err_msg[:200]}") from e

    finally:
        # Safe cleanup of temp files
        cleanup_temp_file(cmd_file_path)
        cleanup_temp_file(trimmed_clip_path)
        cleanup_temp_file(srt_sub_path)
        cleanup_temp_file(ass_sub_path)
        try:
            if os.path.exists(workdir):
                os.rmdir(workdir)
        except Exception:
            pass
