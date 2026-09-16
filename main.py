import ffmpeg_env
import time
import math
import cv2
import subprocess
import argparse
import re
import sys
import threading
import unicodedata
import uuid
import tempfile
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import shutil
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from google.genai import types as genai_types

import gemini_worker
import hook_grounding
import layout_picker
import llm_backend
from tracking import match_box_to_track
from clip_selection import (build_transcript_windows, clip_count_targets,
                            clip_duration_bounds, get_heuristic_clips,
                            snap_clip_to_words, trim_to_best)
from ffmpeg_utils import (video_encode_args, audio_encode_args, QUALITY,
                          QUALITY_FAST, METADATA_SCRUB, safe_remove, safe_replace,
                          run_ffmpeg_command, open_video_capture, ensure_file_unlocked,
                          cleanup_temp_file, format_ffmpeg_error, escape_filter_value)
from dotenv import load_dotenv
import json
import glob

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst). In the descriptions, ALWAYS include a CTA like "Follow me and comment X and I'll send you the workflow" (especially if discussing an n8n workflow):
{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "video_description_for_tiktok": "<description for TikTok oriented to get views>",
      "video_description_for_instagram": "<description for Instagram oriented to get views>",
      "video_title_for_youtube_short": "<title for YouTube Short oriented to get views 100 chars max>",
      "viral_hook_text": "<Full complete sentence hook headline (one complete, grammatically sound sentence with 1-2 fitting emojis, 6 to 14 words). MUST BE A 100% COMPLETE THOUGHT/STATEMENT IN THE SAME LANGUAGE AS THE VIDEO TRANSCRIPT. DO NOT CUT OFF, DO NOT USE TRAILING ELLIPSIS OR '...'. Examples: 'He tried the world's most dangerous diet 🤯', 'This single mistake cost him everything 😱', 'Nobody expected what happened next 🔥'>"
    }}
  ]
}}
"""

# Load the YOLO model once (Keep for backup or scene analysis if needed)
# YOLO_MODEL_PATH lets deployments point at a pre-downloaded weights file so a
# volume mounted over the workdir doesn't trigger a re-download at startup.
model = YOLO(os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt"))

# --- MediaPipe Setup ---
# Use standard Face Detection (BlazeFace) for speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Consecutive detections a large target move must survive before the camera
# follows it (see SmoothedCameraman.update_target). Env-overridable so the
# damping can be dialled back without a deploy; 1 restores the old behaviour.
JUMP_CONFIRM_FRAMES = max(int(os.environ.get("JUMP_CONFIRM_FRAMES", "3")), 1)

# Reset the tracker and the cameraman's damping at every scene cut, so the
# first face found in the new shot is framed instantly instead of being treated
# as a suspicious "jump" from the previous shot's subject (see
# SmoothedCameraman.begin_scene). 0 restores the old behaviour.
SCENE_CUT_RESET = os.environ.get("SCENE_CUT_RESET", "1") != "0"


class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """
    def __init__(self, output_width, output_height, video_width, video_height, aspect_ratio=ASPECT_RATIO):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio

        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / aspect_ratio)
             
        # Dynamic Dead-Zone (Hysteresis): ±7% of screen dimension
        # Micro-movements within this zone keep the camera smooth and stationary
        self.deadzone_x = self.crop_width * 0.07
        self.deadzone_y = self.crop_height * 0.07

        # Exponential Moving Average (EMA) smoothing parameters
        self.smoothing_factor_x = 0.10  # Damped camera following (0.08 - 0.12)
        self.smoothing_factor_y = 0.08
        self.max_step_x = self.video_width * 0.030  # Velocity clamping: max 3.0% pan per frame
        self.max_step_y = self.video_height * 0.030
        self.target_headroom = 0.22  # Comfortable 18%-25% headroom from top

        self.current_center_x = video_width / 2.0
        self.target_center_x = video_width / 2.0
        self.current_center_y = video_height / 2.0
        self.target_center_y = video_height / 2.0

        self.jump_confirm_frames = JUMP_CONFIRM_FRAMES
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = True

    def begin_scene(self):
        """Forget previous shot's subject at a scene cut and prepare clean pre-alignment."""
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = True

    def update_target(self, face_box):
        """Update target center with center-of-mass & eye-line tracking."""
        if not face_box:
            return
        x, y, w, h = face_box
        # Target subject center of mass / eye-line upper torso
        target_x = x + w / 2.0
        target_y = y + h * 0.35

        # Vertical Headroom Alignment: Maintain 18%-25% headroom from top of crop box
        ideal_center_y = y + self.crop_height * (0.5 - self.target_headroom)
        target_cy = ideal_center_y

        if self._snap_pending:
            self._snap_pending = False
            self._pending_target = None
            self._pending_count = 0
            self.target_center_x = target_x
            self.current_center_x = target_x
            self.target_center_y = target_cy
            self.current_center_y = target_cy
            return

        # Outlier rejection for sudden detector teleport spikes (> 50% crop width)
        if abs(target_x - self.target_center_x) > self.crop_width * 0.5:
            if (self._pending_target is not None
                    and abs(target_x - self._pending_target) <= self.deadzone_x * 2.0):
                self._pending_count += 1
            else:
                self._pending_target = target_x
                self._pending_count = 1
            if self._pending_count < self.jump_confirm_frames:
                return  # Confirm outlier before jumping across frame

        self._pending_target = None
        self._pending_count = 0
        self.target_center_x = target_x
        self.target_center_y = target_cy
    
    def get_crop_box(self, force_snap=False):
        """
        Returns (x1, y1, x2, y2) for current frame with smooth EMA tracking,
        dynamic deadband stabilization, velocity clamping, and safe-zone collision avoidance.
        """
        prev_cx = self.current_center_x
        prev_cy = self.current_center_y

        if force_snap:
            self.current_center_x = self.target_center_x
            self.current_center_y = self.target_center_y
        else:
            diff_x = self.target_center_x - self.current_center_x
            diff_y = self.target_center_y - self.current_center_y

            # 1. Dynamic Dead-Zone (Hysteresis): ±7% of screen dimension
            # If subject micro-moves within deadband, keep camera stationary
            if abs(diff_x) > self.deadzone_x:
                # 2. Exponential Moving Average (EMA) Smoothing past dead-band
                step_x = (diff_x - math.copysign(self.deadzone_x, diff_x)) * self.smoothing_factor_x
                self.current_center_x += step_x

            if abs(diff_y) > self.deadzone_y:
                step_y = (diff_y - math.copysign(self.deadzone_y, diff_y)) * self.smoothing_factor_y
                self.current_center_y += step_y

        # 3. Velocity Clamping: prevent whipping during sudden fast motion
        if not force_snap:
            delta_x = self.current_center_x - prev_cx
            if abs(delta_x) > self.max_step_x:
                self.current_center_x = prev_cx + math.copysign(self.max_step_x, delta_x)
            delta_y = self.current_center_y - prev_cy
            if abs(delta_y) > self.max_step_y:
                self.current_center_y = prev_cy + math.copysign(self.max_step_y, delta_y)

        # 4. Safe-Zone Aware Auto-Reframing (Platform UI Collision Guard)
        # Safe Envelope: X in 12% - 78%, Y in 15% - 70%
        crop_x0 = self.current_center_x - self.crop_width / 2.0
        crop_y0 = self.current_center_y - self.crop_height / 2.0
        
        norm_subj_x = (self.target_center_x - crop_x0) / max(1.0, float(self.crop_width))
        norm_subj_y = (self.target_center_y - crop_y0) / max(1.0, float(self.crop_height))

        # Right-Rail Avoidance: ensure subject face never sits under right action buttons (X > 78%)
        if norm_subj_x > 0.78:
            overlap_r = (norm_subj_x - 0.78) * self.crop_width
            self.current_center_x += overlap_r
        elif norm_subj_x < 0.12:
            overlap_l = (0.12 - norm_subj_x) * self.crop_width
            self.current_center_x -= overlap_l

        # Headroom Protection: maintain comfortable headroom (Y: 18%-25%), avoid top header
        if norm_subj_y < 0.18:
            overlap_t = (0.18 - norm_subj_y) * self.crop_height
            self.current_center_y -= overlap_t
        elif norm_subj_y > 0.70:
            overlap_b = (norm_subj_y - 0.70) * self.crop_height
            self.current_center_y += overlap_b

        # 5. Soft boundary deceleration near canvas boundaries
        half_crop = self.crop_width / 2.0
        if not force_snap:
            min_cx = half_crop
            max_cx = self.video_width - half_crop
            soft_margin_x = (max_cx - min_cx) * 0.05 if max_cx > min_cx else 1.0

            dx = self.current_center_x - prev_cx
            if self.current_center_x < min_cx + soft_margin_x and dx < 0 and soft_margin_x > 0:
                t = max(0.0, (self.current_center_x - min_cx) / soft_margin_x)
                self.current_center_x = prev_cx + dx * (0.5 + 0.5 * t)
            if self.current_center_x > max_cx - soft_margin_x and dx > 0 and soft_margin_x > 0:
                t = max(0.0, (max_cx - self.current_center_x) / soft_margin_x)
                self.current_center_x = prev_cx + dx * (0.5 + 0.5 * t)

        # 6. Dynamic Scale Punch-In for Vertical Headroom Headroom Freedom
        scale = 1.0
        if self.crop_height >= self.video_height * 0.95:
            # If subject eye is low or high, scale gently up to 1.25x to provide vertical translation headroom
            subj_y = getattr(self, 'target_center_y', self.video_height / 2.0)
            ideal_norm_y = self.target_headroom
            ideal_ch = subj_y / max(0.01, ideal_norm_y)
            if ideal_ch > self.crop_height:
                scale = min(1.25, max(1.0, ideal_ch / self.crop_height))
            elif subj_y < self.crop_height * 0.20:
                scale = min(1.25, max(1.0, (self.crop_height * 0.20) / max(1.0, subj_y)))

        eff_w = self.crop_width / scale
        eff_h = self.crop_height / scale

        half_crop = eff_w / 2.0
        half_h = eff_h / 2.0
        self.current_center_x = max(half_crop, min(self.video_width - half_crop, self.current_center_x))
        self.current_center_y = max(half_h, min(self.video_height - half_h, self.current_center_y))

        x1 = int(round(self.current_center_x - half_crop))
        y1 = int(round(self.current_center_y - half_h))
        eff_w_int = int(round(eff_w))
        eff_h_int = int(round(eff_h))
        x1 = max(0, min(self.video_width - eff_w_int, x1))
        y1 = max(0, min(self.video_height - eff_h_int, y1))
        x1 -= x1 % 2
        y1 -= y1 % 2
        eff_w_int -= eff_w_int % 2
        eff_h_int -= eff_h_int % 2
        x2 = x1 + eff_w_int
        y2 = y1 + eff_h_int

        return x1, y1, x2, y2

Cameraman = SmoothedCameraman

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker
        
        # Hyperparameters
        self.stabilization_threshold = stabilization_frames # Frames needed to confirm a new speaker
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000
        
        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'box': [x,y,w,h], 'last_frame': 123}]

    def reset(self):
        """Forget every speaker at a scene cut.

        Identity, hysteresis and the switch cooldown are all about continuity
        within a shot. After a cut none of it applies: the sticky x3 bonus and
        the cooldown were holding the previous shot's speaker (returning None)
        for up to 30 frames while a new face sat unframed.
        """
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.last_switch_frame = -1000
        self.known_faces = []

    def get_target(self, face_candidates, frame_number, width, height=None):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []

        # 1. Match faces to known IDs. Use spatial overlap plus x/y and size
        # continuity, and never assign one old face to two detections in the
        # same frame. Horizontal-only matching swapped identities whenever two
        # people crossed or one detector box briefly disappeared.
        used_ids = set()
        frame_height = height if height is not None else width
        for face in face_candidates:
            box = face['box']
            best_match_id = match_box_to_track(
                box, self.known_faces, frame_number, width, frame_height,
                used_ids=used_ids, max_age=45,
            )

            # If no match, assign a new ID. The one-to-one set is updated
            # immediately so duplicate detector boxes cannot share an ID.
            if best_match_id is None:
                best_match_id = self.next_id
                self.next_id += 1

            used_ids.add(best_match_id)
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({
                'id': best_match_id,
                'box': list(box),
                'center': box[0] + box[2] / 2.0,
                'center_y': box[1] + box[3] / 2.0,
                'last_frame': frame_number,
            })

            current_candidates.append({
                'id': best_match_id,
                'box': box,
                'score': face['score']
            })

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0 # Sticky factor
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            # New person. The cooldown must hold whether or not the current
            # speaker happens to be detected in THIS frame.
            #
            # It used to fall through and switch when the active speaker was
            # missing from the candidate list — a blink, a head turn or one
            # motion-blurred frame was enough. That is precisely when the
            # cooldown is needed, so it only ever fired when it wasn't: 3 of 7
            # target switches measured on a 12s clip (25-jul-2026) jumped the
            # cooldown this way, and every jump drags the camera across frame.
            #
            # Returning None holds instead: the caller only calls
            # update_target() on a truthy box, so the camera keeps its current
            # target and finishes whatever move it was making. The hold is
            # bounded by the cooldown itself — once it expires, a speaker who
            # really did leave the shot is switched away from normally.
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                return old_cand['box'] if old_cand else None

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

# Detectors never need full-resolution frames: MediaPipe returns relative
# coords and YOLO boxes are scaled back up. Running them on a ≤640px copy cuts
# per-frame preprocessing cost hard, which is what dominates CPU-only renders.
DETECT_MAX_WIDTH = 640
# The global MediaPipe graph and YOLO model are NOT thread-safe; clips render
# in parallel, so every inference goes through this lock. Contention is small
# (a few ms per call) — the ffmpeg renders are where the parallel time goes.
DETECT_LOCK = threading.Lock()
# Synchronize initial FFmpeg video cut operations across threads to prevent
# concurrent file-access sharing violations on Windows (exit status 3199971767 / WinError 32).
CUT_LOCK = threading.Lock()
# Detect every Nth frame; SmoothedCameraman interpolates between updates. A
# two-frame default materially reduces missed faces while keeping the existing
# DETECT_STRIDE override for CPU-constrained deployments.
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "2")), 1)
# YOLO fallback (no face found) is far heavier than MediaPipe — extra throttle.
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2


def _detection_frame(frame):
    """Downscaled copy for detectors. Returns (small_frame, scale) with
    scale mapping small-frame pixel coords back to the original frame."""
    h, w = frame.shape[:2]
    if w <= DETECT_MAX_WIDTH:
        return frame, 1.0
    scale = w / DETECT_MAX_WIDTH
    small = cv2.resize(frame, (DETECT_MAX_WIDTH, max(int(h / scale), 2)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    Boxes are in ORIGINAL frame coordinates (detection runs downscaled;
    MediaPipe's relative coords make the mapping exact).
    """
    height, width, _ = frame.shape
    small, _scale = _detection_frame(frame)
    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with DETECT_LOCK:
        results = face_detection.process(rgb_frame)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        # Extract eye keypoints for gaze / focal anchoring
        eye_mid_x = float(x + w / 2.0)
        eye_mid_y = float(y + h * 0.35)
        try:
            re = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
            le = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
            if re and le:
                eye_mid_x = float((re.x + le.x) / 2.0) * width
                eye_mid_y = float((re.y + le.y) / 2.0) * height
        except Exception:
            pass

        candidates.append({
            'box': [x, y, w, h],
            'score': w * h, # Area as score
            'eye_line': [round(eye_mid_x, 2), round(eye_mid_y, 2)],
            'focal_anchor': [round(eye_mid_x, 2), round(eye_mid_y, 2)],
        })
            
    return candidates

def detect_people_yolo(frame, conf_threshold=0.25):
    """
    Detect ALL people in the scene using YOLO (class 0: person).
    Returns a list of candidate dicts:
    [
        {
            'box': [x1, y1, w, head_h],     # Head / upper torso for camera tracking
            'full_box': [x1, y1, w, h],     # Entire body bounding box
            'score': float(area * conf),
            'conf': float(conf),
            'eye_line': [float(x1 + w / 2.0), float(y1 + head_h * 0.45)],
            'focal_anchor': [float(x1 + w / 2.0), float(y1 + head_h * 0.45)],
            'type': 'person'
        }, ...
    ]
    Sorted descending by score (largest / most prominent character first).
    """
    small, scale = _detection_frame(frame)
    with DETECT_LOCK:
        results = model(small, verbose=False, classes=[0], conf=conf_threshold)

    if not results:
        return []

    people = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.5
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = [int(i * scale) for i in box.xyxy[0]]
            w = max(2, x2 - x1)
            h = max(2, y2 - y1)
            area = w * h
            head_h = max(2, int(h * 0.40))  # Head + upper chest for framing
            
            eye_cx = float(x1 + w / 2.0)
            eye_cy = float(y1 + head_h * 0.45)
            
            people.append({
                'box': [x1, y1, w, head_h],
                'full_box': [x1, y1, w, h],
                'score': float(area * (0.5 + 0.5 * conf)),
                'conf': conf,
                'eye_line': [round(eye_cx, 2), round(eye_cy, 2)],
                'focal_anchor': [round(eye_cx, 2), round(eye_cy, 2)],
                'type': 'person'
            })

    people.sort(key=lambda p: p['score'], reverse=True)
    return people


def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation, in
    ORIGINAL frame coordinates.
    """
    people = detect_people_yolo(frame)
    return people[0]['box'] if people else None


def detect_contextual_text_regions(frame):
    """
    Ultra-fast OpenCV morphological text detector for on-screen context:
    Detects lower-thirds, presentation slides, titles, banners, product labels,
    and scoreboard text that carry key contextual meaning in the video.
    Returns: list of {'box': [x, y, w, h], 'score': area, 'aspect': aspect, 'type': 'text'}
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Gradient to detect stroke edges characteristic of text and banners
    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    grad = cv2.addWeighted(grad_x, 0.7, grad_y, 0.3, 0)
    
    # 2. Morphological closing with wide horizontal kernel to connect character clusters into text lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.025)), 3))
    morph = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel_h)
    
    # 3. Otsu thresholding
    _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Line grouping
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, int(w * 0.040)), 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    min_w = int(w * 0.04)   # At least 4% of screen width
    min_h = int(h * 0.015)  # At least 1.5% of screen height
    max_h = int(h * 0.35)   # Reject full-screen blobs
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / float(max(1, ch))
        # Text characteristics: horizontal banner (aspect >= 1.2), within size bounds
        if cw >= min_w and ch >= min_h and ch <= max_h and aspect >= 1.2:
            roi_grad = grad[y:y+ch, x:x+cw]
            if np.mean(roi_grad) > 10.0:
                text_regions.append({
                    'box': [x, y, cw, ch],
                    'width': cw,
                    'height': ch,
                    'cx': x + cw / 2.0,
                    'cy': y + ch / 2.0,
                    'score': cw * ch,
                    'aspect': round(aspect, 2),
                    'type': 'text'
                })
                
    text_regions.sort(key=lambda t: t['score'], reverse=True)
    return text_regions


def is_scene_text_presentation(frames):
    """
    Checks if a scene with ZERO performers genuinely contains wide presentation slide,
    screencast, or text document elements that require the full-width text-capturing style.
    Returns True only when structured text lines / wide slide headlines are detected.
    """
    if not frames:
        return False

    slide_votes = 0
    for frame in frames:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # High-contrast horizontal edges typical of text lines
        grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        # Morphological closing along horizontal axis to group letters into words/lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(w * 0.035)), 3))
        morph = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel_h)
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Further group into text lines
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.06)), 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_lines = 0
        has_wide_headline = False
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(max(1, ch))
            # Text line characteristics: horizontal aspect ratio, bounded height
            if aspect >= 2.5 and int(h * 0.015) <= ch <= int(h * 0.20):
                if cw >= int(w * 0.15):
                    text_lines += 1
                if cw >= int(w * 0.50):
                    has_wide_headline = True

        # Pure text slides / screencasts typically feature a wide headline and multiple text lines
        if (has_wide_headline and text_lines >= 2) or text_lines >= 4:
            slide_votes += 1

    return slide_votes >= max(1, (len(frames) + 1) // 2)


def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height), interpolation=cv2.INTER_LINEAR)

    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0: start_x = 0
    background = bg_resized[:, start_x:start_x+output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Blur background: blur at quarter resolution and scale back up — visually
    # identical for a defocused backdrop, an order of magnitude cheaper than a
    # 51px Gaussian at full size.
    small_bg = cv2.resize(background, (max(output_width // 4, 2), max(output_height // 4, 2)),
                          interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small_bg, (13, 13), 0)
    background = cv2.resize(small_bg, (output_width, output_height),
                            interpolation=cv2.INTER_LINEAR)

    # 2. Foreground (Fit Width)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h), interpolation=cv2.INTER_LINEAR)

    # A source taller than the output fills the width at a height that does not
    # fit: centre-crop it instead of indexing the frame with a negative offset,
    # which raises rather than renders.
    if fg_h > output_height:
        top = (fg_h - output_height) // 2
        foreground = foreground[top:top + output_height, :]
        fg_h = output_height

    # 3. Overlay
    y_offset = (output_height - fg_h) // 2

    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, :] = foreground
    
    return final_frame

# NOTE: a "route text-heavy scenes to GENERAL" rule was tried here and removed
# on 26-jul-2026. The problem it targets is real — a screencast that happens to
# contain one face gets cropped to the face and its headlines come out cut
# mid-word — but edge density is the wrong signal for it. Measured: a
# constructed talking-head-beside-a-chart scored 0.012 while the SAME shot
# without the panels scored 0.029, because a flat panel of text has far fewer
# edges than ordinary scene detail. Canny measures visual busyness, not text.
# A real fix needs an actual text detector (MSER/EAST) validated against clips
# that contain the failure mode; this corpus has almost none.


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    strategies = []
    fps = 30.0
    try:
        with open_video_capture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
                s_f, e_f = start.get_frames(), end.get_frames()
                # Sample 5 frames spread across the scene, clamped inside it (the old
                # start+5/end-5 samples landed outside scenes shorter than ~10 frames).
                margin = min(2, max(0, (e_f - s_f - 1) // 2))
                frames_to_check = sorted(set(
                    int(round(f)) for f in np.linspace(s_f + margin, e_f - 1 - margin, 5)
                ))

                face_counts = []
                no_face_frames = []
                for f_idx in frames_to_check:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if not ret: continue

                    # Near-black frames (fades, cut-to-black) carry no faces and used
                    # to drag single-person scenes into GENERAL. Skip them.
                    if frame.mean() < 16:
                        continue

                    # Detect faces and people (solo performer, 2-3 performers, side characters)
                    candidates = detect_face_candidates(frame)
                    if not candidates:
                        people = detect_people_yolo(frame)
                        candidates = people
                    face_counts.append(len(candidates))
                    if not candidates:
                        no_face_frames.append(frame)

                # Decision Logic:
                # 1. When performers are present (solo, 2, or 3 performers):
                #    Always TRACK for full-screen 9:16 vertical crop with intelligent camera tracking.
                #    No blur bars upside and downward!
                # 2. "Text capturing video editing style" (GENERAL / blurred background) is ONLY used
                #    when the scene specifically contains genuine text elements (presentation slides,
                #    screencasts, text documents) and ZERO performers are present.
                avg_faces = sum(face_counts) / len(face_counts) if face_counts else 0
                if avg_faces >= 0.2:
                    # 1, 2, or 3 people performing -> full-screen 9:16 dynamic tracking
                    strategies.append('TRACK')
                else:
                    # 0 performers: check if scene actually contains genuine text presentation/slide elements
                    if is_scene_text_presentation(no_face_frames):
                        strategies.append('GENERAL')
                    else:
                        # Landscape / B-roll with no text -> full-screen 9:16 crop like before
                        strategies.append('TRACK')
    except Exception as e:
        print(f"   ⚠️ Could not analyze scene strategy: {e}")
        return ['TRACK'] * len(scenes)

    # Hysteresis: a short scene whose two neighbors agree on the opposite
    # strategy is almost always a sampling miss (profile face, insert shot).
    # Each TRACK<->GENERAL flip is a full on-screen layout change, so flapping
    # is worse than an occasional wrong-but-stable choice.
    max_flip_frames = int(2.0 * fps)
    for i in range(1, len(strategies) - 1):
        dur = scenes[i][1].get_frames() - scenes[i][0].get_frames()
        if (dur < max_flip_frames
                and strategies[i - 1] == strategies[i + 1] != strategies[i]):
            strategies[i] = strategies[i - 1]

    return strategies

def detect_scenes(video_path):
    import scene_detection
    return scene_detection.detect_scenes(video_path)

def get_video_resolution(video_path):
    with open_video_capture(video_path) as probe:
        return (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))


# Byte budget for the sanitized video title used as the stem of every derived
# file. Filesystems cap a name in BYTES (255 on ext4), not characters, and the
# pipeline decorates this stem: "_clip_10.mp4" (12), "subtitled_<ts>_" (21),
# "hooked_<ts>_" (18), "temp_hook_<hex8>_" (19), "autosubs_<ts>_" + ".ass" (24).
# Budgeting 120 bytes leaves room for all of them stacked (worst chain:
# subtitled_<ts>_hooked_<ts>_<stem>_clip_NN.mp4 ≈ 171 bytes) under the limit.
#
# The old cap was 100 CHARACTERS, which is 300 bytes of Bengali or Arabic — over
# the limit before any decoration. It surfaced as OSError 36 killing the hook
# endpoint in prod on 26-jul-2026.
MAX_TITLE_BYTES = 120


def truncate_bytes(text, max_bytes):
    """Trim ``text`` to a byte budget without splitting a multi-byte character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", "ignore")


def sanitize_filename(filename):
    """Remove invalid characters from filename and bound it for the filesystem."""
    # "canción" has two Unicode spellings: a precomposed ó (NFC) or an o plus a
    # combining acute (NFD). yt-dlp hands over titles in either, and the name
    # becomes the clip file, the R2 key and the URL path. Measured 24-ago-2026:
    # a key carrying the combining form is fetchable by a <video> element but a
    # fetch() of the same URL comes back 503, which broke the download button on
    # every clip with a Spanish title. Normalising here fixes the whole chain at
    # its source, and is a no-op for the ASCII names that already worked.
    filename = unicodedata.normalize('NFC', filename)
    filename = re.sub(r'[<>:"/\\|?*#]', '', filename)
    filename = filename.replace(' ', '_')
    return truncate_bytes(filename, MAX_TITLE_BYTES)


def is_youtube_url(url):
    """True for the hosts the proxy chain exists for. Anything else (a CDN
    mp4, tmpfiles/catbox, an R2 link) has no IP ban to dodge and downloads
    5-10x faster from the server's own IP than through the ISP proxies."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return True
    return host.endswith(("youtube.com", "youtu.be", "youtube-nocookie.com", "googlevideo.com"))


def plan_download_attempts(direct_first, statics, paid, have_hd, youtube=True):
    """Ordered (label, capped, proxy) download plan — pure, unit-tested.

    ``youtube=False`` (a direct file URL): the server's own IP first, then one
    static proxy as the only fallback; the paid per-GB proxy is never used.

    Cheapest bandwidth first: the server's own IP, then the flat-rate static
    ISP proxies (uncapped 1080p, free bytes), then the per-GB paid proxy
    (720p cost cap), and last the conservative fallback strategy through the
    paid proxy (or a static/direct when no paid proxy is configured).
    ``capped`` marks attempts whose bytes are billed per GB."""
    if not youtube:
        plan = [('direct', False, None)]
        if statics:
            plan.append(('static-fallback', False, statics[0]))
        return plan
    plan = []
    if direct_first:
        plan.append(('HD-direct', False, None))
    if have_hd:
        for i, s in enumerate(statics):
            plan.append((f'HD-static{i + 1}', False, s))
    if statics and paid:
        # The conservative clients (tv_embed/android) through a FREE static,
        # before any per-GB attempt: YouTube serves a fake "Video unavailable"
        # to the web/HD client from datacenter-ISP ranges on some videos while
        # the fallback clients pass on the very same IPs (verified 4-sep-2026,
        # all three statics, three countries). Costs nothing and the paid path
        # was capped to 720p anyway, so there is no quality trade.
        plan.append(('fallback-static', False, statics[0]))
    if have_hd:
        plan.append(('HD', bool(paid), paid))
    plan.append(('fallback', bool(paid),
                 paid if paid else (statics[0] if statics else None)))
    return plan


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    # SSRF guard: block non-http(s) schemes and private/loopback/metadata hosts
    # before handing the URL to yt-dlp.
    from security_utils import assert_public_url
    assert_public_url(url)
    # Throwaway hosts agents fall back to (tmpfiles.org) hand out short-lived
    # signed links; refresh through the host's page so yt-dlp gets the file.
    import file_hosts
    url = file_hosts.resolve(url)

    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = '/app/cookies.txt'
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    if cookies_env:
        print("🍪 Found YOUTUBE_COOKIES env var, creating cookies file inside container...")
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                 # Never print file CONTENT here: with a headerless cookies
                 # blob this would leak live YouTube session cookies to logs.
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ YOUTUBE_COOKIES env var not found.")
    
    # Optional HTTP proxy. Set PROXY_URL to route downloads through it; unset
    # (self-host) goes direct as before.
    _proxy = os.environ.get("PROXY_URL", "").strip() or None
    if _proxy:
        print("🌐 Using proxy for download.")

    # Flat-rate static ISP proxies (STATIC_PROXY_URLS, comma-separated), tried
    # BEFORE the per-GB proxy: dedicated IPs with unlimited traffic, so their
    # bandwidth costs nothing per job and carries no 720p cost cap. Rotated per
    # job to spread load (and YouTube's attention) across the pool. PROXY_URL
    # stays the paid last resort — with STATIC_PROXY_URLS unset the behavior is
    # byte-identical to before.
    _statics = [p.strip() for p in
                os.environ.get("STATIC_PROXY_URLS", "").split(",") if p.strip()]
    if _statics:
        import random as _random
        k = _random.randrange(len(_statics))
        _statics = _statics[k:] + _statics[:k]
        print(f"🌐 {len(_statics)} static ISP proxies configured.")

    # Two download strategies, tried in order so a break in the HD path degrades
    # gracefully instead of failing the whole job: an HD attempt first, then a
    # conservative fallback (also the only strategy for self-host).
    _bgutil_http = os.environ.get("BGUTIL_BASE_URL", "").strip()
    _bgutil_script = os.environ.get("BGUTIL_SCRIPT_PATH", "").strip()
    # Client lists live in yt_clients.py (shared with the duration probe):
    # explicit `default,mweb` because the authed defaults alone return
    # "Video unavailable" on a share of videos, from every IP, and that was
    # what fed the per-GB proxy (6-sep-2026, verified in the prod container).
    from yt_clients import hd_extractor_args, fallback_extractor_args
    hd_args = hd_extractor_args(_bgutil_http, _bgutil_script)
    fallback_args = fallback_extractor_args(_bgutil_http, _bgutil_script)

    # Cap at 720p ONLY when the bytes actually go through the PER-GB paid proxy
    # — that cap exists to control bandwidth cost, and the direct attempt and
    # the flat-rate static proxies have none.
    #
    # This is per-attempt on purpose. Deciding it once from `_proxy` capped the
    # DIRECT attempt too, so with DIRECT_FIRST=1 (which serves most downloads)
    # every YouTube source arrived at 720p and, since the reframe inherits the
    # source height, 80% of delivered clips came out 406x720 (audited 25-jul-2026).
    def _hd_fmt_for(capped):
        if capped:
            return ('bestvideo[vcodec^=avc1][height<=720][ext=mp4]+bestaudio[ext=m4a]/'
                    'bestvideo[vcodec^=avc1][height<=720]+bestaudio/'
                    'best[height<=720][ext=mp4]/best[height<=720]/best')
        return ('bestvideo[vcodec^=avc1][height<=1080][ext=mp4]+bestaudio[ext=m4a]/'
                'bestvideo[vcodec^=avc1][height<=1080]+bestaudio/'
                'best[height<=1080][ext=mp4]/best[ext=mp4]/best')

    def _base_opts(extractor_args, proxy, cookies=True):
        return {
            'quiet': False, 'verbose': True, 'no_warnings': False,
            'cookiefile': cookies_path if (cookies and cookies_path) else None,
            'proxy': proxy, 'socket_timeout': 30, 'retries': 10, 'fragment_retries': 10,
            'nocheckcertificate': True, 'cachedir': False,
            'extractor_args': extractor_args,
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ),
            },
        }

    # Wire bytes actually pulled through the (paid) proxy, summed across
    # fragments/streams. Reported to app.py via the PROXY_BYTES= line below.
    _dl_bytes = {"total": 0, "partial": 0}

    def _progress_hook(d):
        if d.get('status') == 'downloading':
            # Bytes of a fragment still in flight: a failed attempt has
            # already paid for these even though 'finished' never fires.
            _dl_bytes["partial"] = int(d.get('downloaded_bytes') or 0)
        elif d.get('status') == 'finished':
            _dl_bytes["partial"] = 0
            _dl_bytes["total"] += int(d.get('total_bytes')
                                      or d.get('total_bytes_estimate')
                                      or d.get('downloaded_bytes') or 0)

    def _attempt(extractor_args, fmt, proxy, cookies=True):
        _dl_bytes["total"] = 0
        _dl_bytes["partial"] = 0
        with yt_dlp.YoutubeDL(_base_opts(extractor_args, proxy, cookies)) as ydl:
            info = ydl.extract_info(url, download=False)
        sanitized = sanitize_filename(info.get('title', 'youtube_video'))
        expected = os.path.join(output_dir, f'{sanitized}.mp4')
        if os.path.exists(expected):
            os.remove(expected)
        dl_opts = {
            **_base_opts(extractor_args, proxy, cookies),
            'format': fmt,
            'outtmpl': os.path.join(output_dir, f'{sanitized}.%(ext)s'),
            'merge_output_format': 'mp4', 'overwrites': True,
            'progress_hooks': [_progress_hook],
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        return sanitized

    # DIRECT_FIRST=1: try the server's own IP before spending proxy bandwidth.
    # Needs cookies + a PO-token provider — without both, YouTube flags the
    # datacenter IP after the first request (verified in prod, 21-jul-2026).
    _direct_first = (os.environ.get("DIRECT_FIRST", "").strip() == "1"
                     and (_proxy or _statics) and hd_args and cookies_path)

    # A fallback attempt runs anonymously when an HD attempt (with cookies)
    # already failed on the same route: the account cookies are what narrows
    # yt-dlp to the clients that die with "Video unavailable", and the
    # anonymous defaults were measured at 1080p on the same static IP. With
    # no HD path at all (self-host without a PO token provider) the fallback
    # is the only attempt, so it keeps the cookies the operator configured.
    # Every attempt asks for the same 1080p spec: the fallback used to ask
    # for `best[ext=mp4]/best`, the best single-file format, which on
    # YouTube is the 360p progressive one even with 1080p streams listed.
    attempts = [
        (label,
         fallback_args if label.startswith('fallback') else hd_args,
         _hd_fmt_for(capped),
         proxy,
         not (label.startswith('fallback') and hd_args))
        for label, capped, proxy in plan_download_attempts(
            _direct_first, _statics, _proxy, bool(hd_args), youtube=is_youtube_url(url))
    ]
    if not is_youtube_url(url):
        print("🌐 Direct file URL: downloading from the server's own IP (no proxy).")

    sanitized_title = None
    last_err = None
    used_proxy = False
    # Every attempt, with its bytes and failure text: printed as PROXY_ROUTE=
    # below so app.py can keep a durable trail of WHY a job reached the paid
    # proxy (the container log rotates within the hour; see cloud/proxy_ledger).
    attempt_log = []
    for label, ea, fmt, proxy, cookies in attempts:
        # A 403 on the media fetch is usually transient: the googlevideo URL is
        # bound to the IP that extracted it, and the residential proxy rotates
        # its exit IP between requests. Retrying re-extracts and usually lands
        # on a consistent IP (3 of 62 downloads hit this on 22-jul-2026).
        for retry in range(2):
            try:
                print(f"📥 Download attempt: {label}" + (f" (retry {retry})" if retry else ""))
                sanitized_title = _attempt(ea, fmt, proxy, cookies)
                # Only bytes through the PER-GB proxy cost money; direct and
                # the flat-rate static proxies are free bandwidth for the
                # monthly counter's purposes.
                used_proxy = proxy is not None and proxy == _proxy
                attempt_log.append({"label": label, "ok": True,
                                    "bytes": _dl_bytes["total"] + _dl_bytes["partial"],
                                    "paid": used_proxy})
                print(f"✅ Download succeeded ({label}).")
                break
            except Exception as e:
                last_err = e
                attempt_log.append({"label": label, "ok": False,
                                    "bytes": _dl_bytes["total"] + _dl_bytes["partial"],
                                    "paid": proxy is not None and proxy == _proxy,
                                    "error": str(e)[:300]})
                print(f"⚠️  Download attempt '{label}' failed: {str(e)[:200]}")
                retryable = '403' in str(e) or 'Forbidden' in str(e)
                if not retryable or retry == 1:
                    break
                time.sleep(3)
        if sanitized_title is not None:
            break

    if sanitized_title is None:
        import sys
        error_msg = f"""
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)
❌ ================================================================= ❌
REASON: YouTube blocked the request or the download tooling is out of date.
👇 SOLUTION FOR USER: download the video manually and use the 'Upload Video' tab.
Technical Details: {str(last_err)}
"""
        print(error_msg, file=sys.stdout)
        print(error_msg, file=sys.stderr)
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(0.5)
        raise last_err

    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break

    # Paid bytes across EVERY attempt that used the per-GB proxy, failed ones
    # included: a paid attempt that died after three 10 MB fragments was
    # billed for them even though a later attempt won.
    paid_bytes = sum(int(a.get("bytes") or 0) for a in attempt_log if a.get("paid"))
    print("PROXY_ROUTE=" + json.dumps({
        "winner": attempt_log[-1]["label"] if attempt_log and attempt_log[-1].get("ok") else None,
        "paid_bytes": paid_bytes,
        "attempts": attempt_log,
    }, ensure_ascii=False))
    if paid_bytes:
        # Machine-parseable marker consumed by app.py's log reader for the
        # monthly proxy-bandwidth counter. Not shown to clients (log filter).
        print(f"PROXY_BYTES={paid_bytes}")
    print(f"✅ Video downloaded in {time.time() - step_start_time:.2f}s: {downloaded_file}")
    return downloaded_file, sanitized_title

def finalize_clip_passthrough(input_video, final_output_video):
    """Keep the clip's native framing (for horizontal/16:9 output).

    The input is the freshly encoded cut, so a stream-copy remux is enough to
    add +faststart — re-encoding here would only cost time and quality.
    """
    ensure_file_unlocked(input_video)
    cleanup_temp_file(final_output_video)
    print(f"🎬 Passthrough (native framing): {input_video}")
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-c', 'copy', *METADATA_SCRUB, '-movflags', '+faststart',
        final_output_video,
    ]
    run_ffmpeg_command(cmd, timeout=1800)
    ensure_file_unlocked(final_output_video)
    print(f"✅ Clip saved to {final_output_video}")
    return True


def get_media_duration(file_path: str) -> float:
    """Gets audio or video duration in seconds via ffprobe."""
    if not file_path or not os.path.exists(file_path):
        return 0.0
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def mix_background_audio(video_path: str, bg_audio_path: str, clip_index: int, total_clips: int, clip_duration: float, bg_volume: float = 0.18, output_path: str = None) -> str:
    """Mixes a background audio track into the video with ducked volume and intelligent non-repeating offsets.
    
    Each clip receives a distinct, non-overlapping or phase-staggered segment of the background track
    with smooth 1.0s fade-in and 1.5s fade-out, keeping the speech track loud and crisp.
    """
    if not bg_audio_path or not os.path.exists(bg_audio_path) or not os.path.exists(video_path):
        return video_path

    ensure_file_unlocked(video_path)
    output_dir = os.path.dirname(video_path)
    stem = os.path.basename(video_path)
    if not output_path:
        output_path = os.path.join(output_dir, f"bgm_{int(time.time())}_{uuid.uuid4().hex[:6]}_{stem}")

    try:
        bg_dur = get_media_duration(bg_audio_path)
        actual_clip_dur = get_media_duration(video_path) or clip_duration or 30.0

        # Intelligent offset calculation so each clip gets a different part of the audio
        if bg_dur > actual_clip_dur:
            if bg_dur >= actual_clip_dur * max(1, total_clips):
                start_offset = clip_index * actual_clip_dur
            else:
                stride = (bg_dur - actual_clip_dur) / max(1, total_clips)
                start_offset = (clip_index * max(10.0, stride)) % max(1.0, bg_dur - actual_clip_dur)
        else:
            start_offset = (clip_index * 7.5) % max(1.0, bg_dur)

        start_offset = max(0.0, min(start_offset, max(0.0, bg_dur - 2.0)))
        fade_in = min(1.0, actual_clip_dur * 0.1)
        fade_out = min(1.5, actual_clip_dur * 0.15)
        fade_out_st = max(0.0, actual_clip_dur - fade_out)

        # Check whether source video has an audio stream
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        has_voice = False
        try:
            p = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            has_voice = "audio" in p.stdout.lower()
        except Exception:
            has_voice = True

        # Volume ducking: default 0.18 (18%) low voice so speech is completely clear
        vol = max(0.05, min(0.50, float(bg_volume)))

        if has_voice:
            filter_complex = (
                f"[0:a]volume=1.0[voice];"
                f"[1:a]volume={vol:.2f},afade=t=in:st=0:d={fade_in:.1f},afade=t=out:st={fade_out_st:.1f}:d={fade_out:.1f}[bg];"
                f"[voice][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
        else:
            filter_complex = (
                f"[1:a]volume={max(0.35, vol * 2):.2f},afade=t=in:st=0:d={fade_in:.1f},afade=t=out:st={fade_out_st:.1f}:d={fade_out:.1f}[aout]"
            )

        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', f"{start_offset:.2f}",
            '-i', bg_audio_path,
            '-filter_complex', filter_complex,
            '-map', '0:v',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path
        ]

        run_ffmpeg_command(cmd)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"   🎵 Background audio mixed (clip {clip_index + 1} offset: {start_offset:.1f}s, vol: {vol:.0%})")
            return output_path
        return video_path
    except Exception as e:
        print(f"   ⚠️ Background audio mixing warning for clip {clip_index + 1}: {e}")
        return video_path


def auto_caption_clip(clip_path, transcript, clip_start, clip_end, split_ranges=None, subtitle_style=None):
    """Burns timed, word-highlight subtitles into ``clip_path``.
    Supports global preset styles and custom draggable coordinates.
    """
    if os.environ.get("AUTO_CAPTIONS", "1").strip() == "0":
        return None

    custom_config = {}
    env_config_raw = os.environ.get("AUTO_CAPTION_CONFIG")
    if env_config_raw:
        try:
            custom_config = json.loads(env_config_raw)
        except Exception:
            pass

    # Extract manual_y_offset: check custom_config first, then env var
    manual_y = custom_config.get("manual_y_offset")
    if manual_y is None and os.environ.get("AUTO_CAPTION_Y_OFFSET"):
        try:
            manual_y = float(os.environ.get("AUTO_CAPTION_Y_OFFSET"))
        except Exception:
            pass

    chosen_style = subtitle_style or custom_config.get("style") or os.environ.get("AUTO_CAPTION_STYLE") or "shorts"
    if str(chosen_style).lower() in ("none", "off", "0", "false"):
        return None
    if not transcript or not transcript.get('segments'):
        return None  # silent video: nothing to caption
    ensure_file_unlocked(clip_path)
    ass_path = None
    try:
        import subtitles as _subs
        style = _subs.get_caption_style(chosen_style, overrides=custom_config)
        output_dir = os.path.dirname(clip_path)
        stem = os.path.basename(clip_path)
        generation_id = int(time.time())
        ass_path = os.path.join(
            output_dir, f"autosubs_{generation_id}_{uuid.uuid4().hex[:8]}.ass")
        out_path = os.path.join(output_dir, f"subtitled_{generation_id}_{stem}")

        if split_ranges is None:
            import layout_ranges as _layouts
            split_ranges = _layouts.split_ranges(_layouts.read(clip_path))
        if not _subs.generate_ass(
                transcript, clip_start, clip_end, ass_path,
                split_ranges=split_ranges,
                max_chars=style.get("max_chars", 18), max_duration=style.get("max_duration", 1.5),
                alignment=style.get("alignment", "bottom"), fontsize=style.get("font_size", 40),
                font_name=style.get("font_name", "Verdana"), font_color=style.get("font_color", "#FFFFFF"),
                border_color=style.get("border_color", "#000000"), border_width=style.get("border_width", 3),
                highlight_color=style.get("highlight_color", "#FFD700"),
                bg_color=style.get("bg_color", "#000000"), bg_opacity=style.get("bg_opacity", 0.0),
                effect=style.get("effect", "none"),
                base_opacity=style.get("base_opacity", 1.0), uppercase=style.get("uppercase", False),
                manual_y_offset=manual_y):
            print("   ℹ️ No words in range — clip ships without captions.")
            return None

        _subs.burn_subtitles(
            clip_path, ass_path, out_path,
            alignment=style.get("alignment", "bottom"), fontsize=style.get("font_size", 40),
            font_name=style.get("font_name", "Verdana"), font_color=style.get("font_color", "#FFFFFF"),
            border_color=style.get("border_color", "#000000"), border_width=style.get("border_width", 3),
            bg_color=style.get("bg_color", "#000000"), bg_opacity=style.get("bg_opacity", 0.0))
        print(f"   💬 Captions burned: {os.path.basename(out_path)}")
        return out_path
    except Exception as e:
        print(f"   ⚠️ Auto-captions failed ({type(e).__name__}: {e}) — "
              f"delivering the clip without them.")
        return None
    finally:
        if ass_path:
            safe_remove(ass_path)


def auto_hook_clip(clip_path, clip):
    """Burn the clip's Gemini hook text as a DERIVED file (AUTO_HOOK=1).

    Writes ``hooked_<ts>_<clip filename>`` next to the canonical clip, exactly
    like captions write ``subtitled_<ts>_...``: the canonical stays clean, so
    the hook can later be replaced or removed by walking the prefix back
    (app.py `_strip_burned_hook`). Captions are then burned ON TOP of the
    hooked file, keeping the "captions are always the last layer" invariant.

    Returns (hooked_path, hook_config), or None when skipped or failed — a
    hook problem must never cost the user the clip itself (same fail-open
    contract as auto_caption_clip)."""
    text = (clip.get('viral_hook_text') or clip.get('video_title_for_youtube_short') or clip.get('hook') or clip.get('title') or '').strip()
    # Strip any dangling ellipsis or trailing dots/dashes to ensure clean, finished sentence
    text = re.sub(r'[\s\.\-_…]+$', '', text).strip()
    if not text:
        return None
    style = os.environ.get("AUTO_HOOK_STYLE", "yellow")
    # User request: hook should remain over video till video end by default!
    # If AUTO_HOOK_SECONDS is "0", "forever", "full", "whole", or empty, duration is None (until video ends)
    raw_seconds = os.environ.get("AUTO_HOOK_SECONDS", "0").strip().lower()
    if raw_seconds in ("0", "none", "forever", "full", "whole", ""):
        seconds = None
    else:
        try:
            seconds = float(raw_seconds)
            if seconds <= 0:
                seconds = None
        except ValueError:
            seconds = None

    pos = os.environ.get("AUTO_HOOK_POSITION", "top").strip().lower()
    try:
        from hooks import add_hook_to_video, HOOK_STYLES
        if style not in HOOK_STYLES:
            style = "classic"
        output_dir = os.path.dirname(clip_path)
        out_path = os.path.join(
            output_dir, f"hooked_{int(time.time())}_{os.path.basename(clip_path)}")
        add_hook_to_video(clip_path, text, out_path, position=pos,
                          duration=seconds, style=style)
        dur_label = "until video end" if seconds is None else f"{seconds:g}s"
        print(f"   🪝 Hook burned ({style}, {dur_label}, pos={pos}): {text}")
        return out_path, {"text": text, "style": style, "position": pos,
                          "duration_seconds": seconds}
    except Exception as e:
        print(f"   ⚠️ Auto-hook failed ({type(e).__name__}: {e}) — "
              f"delivering the clip without it.")
        return None


def render_clip(input_video, final_output_video, output_format="auto",
                force_strategy=None, crop_overrides=None):
    """Route a cut clip through the right renderer for the chosen output format.
    vertical/auto -> 9:16 reframe, square -> 1:1 reframe, horizontal -> keep.
    ``force_strategy`` (e.g. 'WIDE'/'TRACK') pins every scene's layout — the
    clip editor's whole-clip framing override. ``crop_overrides`` positions
    individual scenes by hand (the per-scene reframing editor) and wins over
    ``force_strategy`` for the scenes it names."""
    if not input_video or not os.path.exists(input_video) or os.path.getsize(input_video) == 0:
        print(f"   ⚠️ Input video {input_video} does not exist or is empty — skipping render.")
        return False
    if output_format == "horizontal":
        return finalize_clip_passthrough(input_video, final_output_video)
    aspect = 1.0 if output_format == "square" else ASPECT_RATIO
    return process_video_to_vertical(input_video, final_output_video, aspect_ratio=aspect,
                                     force_strategy=force_strategy,
                                     crop_overrides=crop_overrides)


# Watermark geometry, as fractions of the clip width/height.
WATERMARK_WIDTH_RATIO = 0.18
WATERMARK_MARGIN_RATIO = 0.04
WATERMARK_Y_RATIO = 0.90
WATERMARK_OPACITY = 0.85


def apply_watermark(video_path, position=None):
    """Burn watermark (custom uploaded from device or default OpenShorts logo) into a finished clip.
    position: 'bottom-left' | 'bottom-right' (defaults to os.environ.get('WATERMARK_POSITION', 'bottom-right'))
    """
    custom_logo = os.environ.get("WATERMARK_PATH")
    if custom_logo and os.path.exists(custom_logo) and os.path.getsize(custom_logo) > 0:
        logo_path = custom_logo
    else:
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "assets", "watermark.png")
    if not os.path.exists(logo_path):
        print(f"   ⚠️ Watermark asset missing ({logo_path}); clip kept unmarked.")
        return False

    # Scale the lockup from the clip's real width
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
            stderr=subprocess.STDOUT, timeout=60,
        ).decode().strip().split("x")
        vw, vh = int(probe[0]), int(probe[1])
    except Exception as e:
        print(f"   ⚠️ Could not probe clip for watermark ({e}); clip kept unmarked.")
        return False

    wm_w = max(70, int(vw * WATERMARK_WIDTH_RATIO))
    pos = (position or os.environ.get("WATERMARK_POSITION") or "bottom-right").strip().lower()
    if pos in ("bottom-left", "left", "bottom_left"):
        x = int(vw * WATERMARK_MARGIN_RATIO)
    else:
        x = int(vw - wm_w - vw * WATERMARK_MARGIN_RATIO)
    y = int(vh * WATERMARK_Y_RATIO)
    ensure_file_unlocked(video_path)
    filt = (
        f"[1:v]scale={wm_w}:-1,format=rgba,"
        f"colorchannelmixer=aa={WATERMARK_OPACITY}[wm];"
        f"[0:v][wm]overlay=x={x}:y={y}"
    )
    tmp_path = video_path + ".wm.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", logo_path,
           "-filter_complex", filt,
           *video_encode_args(QUALITY), "-c:a", "copy", *METADATA_SCRUB,
           "-movflags", "+faststart", tmp_path]
    try:
        run_ffmpeg_command(cmd, timeout=1800)
        ensure_file_unlocked(tmp_path)
        if safe_replace(tmp_path, video_path):
            ensure_file_unlocked(video_path)
            return True
    except Exception as e:
        print(f"   ⚠️ Watermark pass failed (clip kept unmarked): {e}")
    cleanup_temp_file(tmp_path)
    return False


def build_blurred_background_filter(out_w=1080, out_h=1920, dim=True, orig_w=None, orig_h=None, sharp=False):
    """
    Builds the optimized FFmpeg filter complex for Blurred Background Fill (1080x1920):
    1. Foreground: scaled to fit cleanly within out_w (scale=1080:-2), maintaining aspect ratio with no stretching/cropping.
    2. Background: scaled/cropped to fill out_w x out_h, with box blur (20:5) and subtle dimming.
    3. Filter: [0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];[0:v]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2
    """
    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)
    dim_str = ",eq=brightness=-0.05" if dim else ""
    sharp_flags = ":flags=lanczos+accurate_rnd,unsharp=5:5:0.6:5:5:0.0" if sharp else ""
    if orig_w and orig_h and (orig_w / float(orig_h) < out_w / float(out_h)):
        fg_scale = f"scale=-2:{out_h}{sharp_flags}"
    else:
        fg_scale = f"scale={out_w}:-2{sharp_flags}"
    return (
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},boxblur=20:5{dim_str}[bg];"
        f"[0:v]{fg_scale}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def process_video_to_vertical(input_video, final_output_video, aspect_ratio=ASPECT_RATIO,
                              force_strategy=None, crop_overrides=None):
    """
    Reframes video using Blurred Background Fill vertical layout (1080x1920):
    1. Foreground Layer (The Original Video):
       - Scale original video so its entire width fits cleanly inside the 1080px width
         (scale=1080:-2), maintaining original aspect ratio with NO stretching and NO side-cropping.
       - Position this sharp, full-width video dead-center on the screen.
    2. Background Layer (The Full-Screen Fill):
       - Scale/crop input video to fill the entire 1080x1920 canvas.
       - Apply fast box blur (and subtle dimming) so the background fills the whole screen aesthetically.
    3. FFmpeg Filter Structure:
       [0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];[0:v]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2
    """
    if not input_video or not os.path.exists(input_video) or os.path.getsize(input_video) == 0:
        print(f"   ⚠️ Input video {input_video} does not exist or is empty — skipping vertical reframe.")
        return False
    ensure_file_unlocked(input_video)

    reframe_style = os.environ.get("REFRAME_STYLE", "auto").strip().lower()
    # Dynamic Face Tracking Vertical Reframe (or manual scene crop overrides from editor UI)
    if crop_overrides or reframe_style not in ("blur_bg", "blurred", "blur"):
        try:
            import reframe_v2
            t0 = time.time()
            result = reframe_v2.render(input_video, final_output_video, aspect_ratio,
                                       force_strategy=force_strategy,
                                       crop_overrides=crop_overrides)
            print(f"   ⏱️ Dynamic face tracking reframe total: {time.time() - t0:.1f}s")
            return result
        except FileNotFoundError as fnf:
            print(f"   ⚠️ Face tracking reframe skipped: {fnf}")
            return False
        except Exception as e:
            if crop_overrides:
                raise RuntimeError(
                    f"manual framing needs crop reframe, which failed ({type(e).__name__}: {e})") from e
            print(f"   ⚠️ Face tracking reframe failed ({type(e).__name__}: {e}) — falling back to blurred background fill")

    if not input_video or not os.path.exists(input_video) or os.path.getsize(input_video) == 0:
        print(f"   ⚠️ Input video {input_video} missing or empty — skipping blurred background fallback.")
        return False

    t0 = time.time()
    print(f"🎬 Reframing with Blurred Background Fill (1080x1920): {input_video}")

    out_w = 1080
    if aspect_ratio == 1.0:
        out_h = 1080
    elif aspect_ratio and aspect_ratio != ASPECT_RATIO:
        out_h = int(round(out_w / aspect_ratio))
    else:
        out_h = 1920

    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)

    orig_w, orig_h = None, None
    try:
        orig_w, orig_h = get_video_resolution(input_video)
    except Exception:
        pass

    dim = os.environ.get("BLUR_BG_DIM", "1").strip() != "0"
    filt = build_blurred_background_filter(out_w=out_w, out_h=out_h, dim=dim,
                                          orig_w=orig_w, orig_h=orig_h, sharp=True)
    filt_graph = f"{filt}[v]"

    out_dir = os.path.dirname(os.path.abspath(final_output_video))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    staging_output = final_output_video + f".staging_{uuid.uuid4().hex[:6]}.mp4"
    if os.path.isfile(staging_output):
        cleanup_temp_file(staging_output)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_video,
        "-filter_complex", filt_graph,
        "-map", "[v]", "-map", "0:a?",
        *video_encode_args(QUALITY_FAST),
        "-c:a", "copy",
        *METADATA_SCRUB,
        "-movflags", "+faststart",
        staging_output
    ]
    try:
        run_ffmpeg_command(cmd, timeout=1800)
    except subprocess.CalledProcessError as copy_err:
        cleanup_temp_file(staging_output)
        cmd_reencode = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_video,
            "-filter_complex", filt_graph,
            "-map", "[v]", "-map", "0:a?",
            *video_encode_args(QUALITY_FAST),
            *audio_encode_args(),
            *METADATA_SCRUB,
            "-movflags", "+faststart",
            staging_output
        ]
        try:
            run_ffmpeg_command(cmd_reencode, timeout=1800)
        except subprocess.CalledProcessError as enc_err:
            print(f"   ❌ [Render Error] Failed to render {final_output_video}:")
            print(format_ffmpeg_error(enc_err, max_lines=30))
            cleanup_temp_file(staging_output)
            return False
    except Exception as e:
        print(f"   ❌ [Render Error] Unexpected error rendering {final_output_video}: {e}")
        cleanup_temp_file(staging_output)
        return False

    ensure_file_unlocked(staging_output)

    # Verify output file exists and is non-empty
    if not (os.path.exists(staging_output) and os.path.getsize(staging_output) > 0):
        print(f"   ❌ [Render Error] Output file missing or zero bytes: {staging_output}")
        cleanup_temp_file(staging_output)
        return False

    if not safe_replace(staging_output, final_output_video):
        # Fallback if safe_replace fails
        cleanup_temp_file(final_output_video)
        shutil.move(staging_output, final_output_video)
    ensure_file_unlocked(final_output_video)

    # Record layout range sidecar
    try:
        import layout_ranges
        with open_video_capture(final_output_video) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dur = total_frames / fps if fps else 0.0
        layout_ranges.write(final_output_video, [(0.0, dur, "general")])
    except Exception:
        pass

    print(f"   ✅ Blurred background clip rendered in {time.time() - t0:.2f}s -> {final_output_video}")
    return True


def _process_video_to_vertical_v1_legacy(input_video, final_output_video, aspect_ratio=ASPECT_RATIO,
                                        force_strategy=None, crop_overrides=None):

    # The v1 loop stages its work next to the final file: a silent video track
    # first, then the source audio, muxed together at the end.
    stem = os.path.splitext(final_output_video)[0]
    silent_video_path = stem + ".v1video.mp4"
    audio_track_path = stem + ".v1audio.aac"
    for stale in (silent_video_path, audio_track_path, final_output_video):
        # isfile, not exists: a caller that hands us a directory should not
        # take an EACCES here, and must never have it deleted either.
        if os.path.isfile(stale):
            cleanup_temp_file(stale)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        # Scene detection found nothing: treat the whole video as one scene.
        print("   ❌ No scenes were detected. Using full video as one scene.")
        with open_video_capture(input_video) as probe:
            span = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(span, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    original_width, original_height = get_video_resolution(input_video)
    
    # Same delivery floor as the v2 engine — a fallback render is still the clip
    # the user posts, so it must not ship sub-HD. The frame loop below already
    # resizes every cropped frame to these dims, so nothing else changes.
    from reframe_v2 import delivery_size
    OUTPUT_WIDTH, OUTPUT_HEIGHT = delivery_size(original_width, original_height,
                                                aspect_ratio)

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height, aspect_ratio=aspect_ratio)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    # Raw BGR frames stream down a pipe into ffmpeg, which encodes the silent
    # video track; the audio is muxed back in afterwards.
    encoder = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'bgr24',
         '-video_size', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}',
         '-framerate', str(fps), '-i', 'pipe:0',
         *video_encode_args(QUALITY_FAST), '-an', silent_video_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    # Per-stage wall time (server-side diagnostics; hidden from cloud logs).
    stage_seconds = {'detect': 0.0, 'write': 0.0}
    loop_started = time.time()
    frame_number = 0
    current_scene_index = 0

    with open_video_capture(input_video) as reader:
        frame_total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=frame_total, desc="   Processing", file=sys.stdout) as pbar:
            while reader.isOpened():
                ret, frame = reader.read()
                if not ret:
                    break

                # Update Scene Index
                if current_scene_index < len(scene_boundaries):
                    start_f, end_f = scene_boundaries[current_scene_index]
                    if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                        current_scene_index += 1
                
                # Determine Strategy for current frame based on scene
                current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
                
                # Apply Strategy
                if current_strategy == 'GENERAL':
                    # "Plano General" -> Blur Background + Fit Width
                    output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                    
                    # Reset cameraman/tracker so they don't drift while inactive
                    cameraman.current_center_x = original_width / 2
                    cameraman.target_center_x = original_width / 2
                    
                else:
                    # "Single Speaker" -> Track & Crop

                    # Detect every Nth frame for performance (cameraman smooths in
                    # between); the much heavier YOLO fallback gets its own stride.
                    # Snap camera on scene change to avoid panning from previous scene position
                    is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                    if is_scene_start and SCENE_CUT_RESET:
                        speaker_tracker.reset()
                        cameraman.begin_scene()

                    # Always detect on a cut, whatever the stride: the new shot's
                    # subject has to be found before the first frame is framed.
                    if frame_number % DETECT_STRIDE == 0 or (is_scene_start and SCENE_CUT_RESET):
                        t_det = time.time()
                        candidates = detect_face_candidates(frame)
                        target_box = speaker_tracker.get_target(
                            candidates, frame_number, original_width, original_height)
                        if target_box:
                            cameraman.update_target(target_box)
                        elif frame_number % YOLO_FALLBACK_STRIDE == 0 or (is_scene_start and SCENE_CUT_RESET):
                            person_box = detect_person_yolo(frame)
                            if person_box:
                                cameraman.update_target(person_box)
                        stage_seconds['detect'] += time.time() - t_det

                    x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)

                    # Crop
                    if y2 > y1 and x2 > x1:
                        cropped = frame[y1:y2, x1:x2]
                        output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    else:
                        output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

                t_wr = time.time()
                encoder.stdin.write(output_frame.tobytes())
                stage_seconds['write'] += time.time() - t_wr
                frame_number += 1
                pbar.update(1)
    
    loop_total = time.time() - loop_started
    other = loop_total - stage_seconds['detect'] - stage_seconds['write']
    print(f"\n   ⏱️ Frame loop: {loop_total:.1f}s total — "
          f"detect {stage_seconds['detect']:.1f}s, "
          f"encode-wait {stage_seconds['write']:.1f}s, "
          f"decode+render {other:.1f}s ({frame_number} frames)")

    encoder.stdin.close()
    encode_log = encoder.stderr.read().decode()
    encoder.wait()

    if encoder.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", encode_log)
        return False

    print("\n   🔊 Step 5: Extracting audio...")
    try:
        run_ffmpeg_command(
            ['ffmpeg', '-y', '-i', input_video, '-vn', '-c:a', 'copy', audio_track_path],
            timeout=1800)
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")

    print("\n   ✨ Step 6: Merging...")
    mux = ['ffmpeg', '-y', '-i', silent_video_path]
    if os.path.exists(audio_track_path):
        mux += ['-i', audio_track_path]
    mux += ['-c', 'copy', *METADATA_SCRUB, '-movflags', '+faststart', final_output_video]
    try:
        run_ffmpeg_command(mux, timeout=1800)
        ensure_file_unlocked(final_output_video)
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode() if e.stderr else "")
        return False

    for leftover in (silent_video_path, audio_track_path):
        cleanup_temp_file(leftover)

    return True

# --- Transcript checkpoint (survive a redeploy without paying twice) ---------
# A job interrupted by a container restart is re-run from its resume manifest
# (app.py) with the SAME output directory. Transcription is the slow, paid part
# of the pipeline that ran before the interruption, so the finished transcript
# is left in the job directory and picked up by the re-run instead of
# transcribing again. Same shape and same validation as --transcript.
TRANSCRIPT_CHECKPOINT = ".transcript_checkpoint.json"


def _checkpoint_source_key(input_video, duration):
    """What ties a checkpoint to ONE source. Not the path: a resumed cloud job
    re-downloads to the same name, and the CLI may be pointed at a different
    file in a directory where an earlier run died. Name plus duration is what
    both the server and a careful CLI user keep stable across the two runs."""
    return {"name": os.path.basename(input_video), "duration": round(float(duration), 1)}


def save_transcript_checkpoint(output_dir, transcript, input_video, duration):
    """Best effort: a failure here must never fail the job."""
    try:
        payload = {"source": _checkpoint_source_key(input_video, duration),
                   "transcript": transcript}
        with open(os.path.join(output_dir, TRANSCRIPT_CHECKPOINT), "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"⚠️ Could not save transcript checkpoint: {e}")


def load_transcript_checkpoint(output_dir, input_video, duration):
    """The transcript an interrupted run left behind for THIS source, or None.

    A checkpoint for a different source (a CLI run that died, then a new video
    processed in the same directory) is ignored, not reused."""
    path = os.path.join(output_dir, TRANSCRIPT_CHECKPOINT)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        source = payload.get("source") or {}
        expected = _checkpoint_source_key(input_video, duration)
        if source.get("name") != expected["name"] \
                or abs(float(source.get("duration", -1)) - expected["duration"]) > 0.5:
            print("⏭️ Transcript checkpoint belongs to another source — ignoring it.")
            return None
        transcript = payload.get("transcript") or {}
        if not transcript.get("segments"):
            raise ValueError("checkpoint has no segments")
        return transcript
    except Exception as e:
        print(f"⚠️ Ignoring unusable transcript checkpoint ({e}).")
        return None


def clear_transcript_checkpoint(output_dir):
    safe_remove(os.path.join(output_dir, TRANSCRIPT_CHECKPOINT))


def transcribe_video(video_path):
    print("🎙️  Transcribing video...")
    from transcribe_backends import transcribe_media

    transcript = transcribe_media(video_path)

    print(f"   Detected language '{transcript['language']}', "
          f"{len(transcript['segments'])} segments")
    for segment in transcript['segments']:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    return transcript

def _run_gemini_stage(client, model_name, prompt, schema, creative=False):
    """One schema-enforced model call with transient-error backoff.
    Returns (parsed_dict, cost_analysis).

    With an OpenAI-compatible server configured (``llm_backend.active()``)
    the call goes there instead of Gemini and ``client`` is unused; the
    retry policy is shared because a local server has the same failure
    shapes (connection refused while the model loads, a truncated body,
    a 5xx from a busy vLLM)."""
    use_local = llm_backend.active()
    config = None if use_local else genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.7 if creative else 0.1,
    )
    active_model = model_name
    candidate_fallbacks = [model_name, 'gemini-2.5-flash', 'gemini-2.0-flash']
    seen = set()
    candidate_fallbacks = [m for m in candidate_fallbacks if m and not (m in seen or seen.add(m))]

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if use_local:
                return llm_backend.generate_json(prompt, schema, model=model_name)
            if attempt > 1 and len(candidate_fallbacks) >= attempt:
                active_model = candidate_fallbacks[attempt - 1]
            response = client.models.generate_content(model=active_model, contents=prompt, config=config)
            # Policy blocks are deterministic — retrying only burns quota and
            # time, and the user deserves the real reason instead of a generic
            # "empty response" (prod 23-jul: PROHIBITED_CONTENT on every try).
            gemini_worker.raise_if_blocked(response)
            # Parsing lives inside the retry loop on purpose: Gemini sometimes
            # returns 200 with an empty body, which raises here rather than at
            # the call. Retrying that recovered every occurrence seen in prod
            # (22-jul-2026) — the same payload succeeds on the next attempt.
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
            else:
                parsed = gemini_worker._parse_json_response_text(
                    gemini_worker._get_response_text(response))
            return parsed, gemini_worker._calculate_cost_analysis(response, active_model)
        except gemini_worker.GeminiBlockedError:
            raise  # deterministic policy block — never retry
        except Exception as e:
            msg = str(e)
            transient = any(tok in msg for tok in (
                '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED',
                '500', 'INTERNAL', 'overloaded', 'Deadline',
                'empty response body', 'did not contain a JSON object',
                'Failed to parse Gemini JSON response',
                # OpenAI-compatible servers: model still loading, busy, or a
                # small model that skipped a required field this time.
                'ConnectError', 'ReadTimeout', 'RemoteProtocolError', '502', '504',
                'validation error'))
            if attempt == max_attempts or not transient:
                raise
            wait = 2 * (2 ** (attempt - 1))
            who = "LLM server" if use_local else "Gemini"
            print(f"⚠️ {who} transient error (attempt {attempt}/{max_attempts}), retrying in {wait}s: {msg[:150]}")
            time.sleep(wait)


def _run_stage_split(client, model_name, items, build_prompt, schema, key, costs, label, creative=False):
    """Run a Gemini stage over ``items``; on a policy block, bisect."""
    if not items:
        return []
    prompt = build_prompt(items)
    try:
        parsed, cost = _run_gemini_stage(client, model_name, prompt, schema, creative=creative)
        if cost:
            costs.append(cost)
        return list(parsed.get(key) or [])
    except gemini_worker.GeminiBlockedError as e:
        if len(items) == 1:
            print(f"   🚫 {label}: Gemini blocked window {items[0].get('id')} on its own; skipping it ({e})")
            return []
        mid = len(items) // 2
        print(f"   🚫 {label}: Gemini blocked a batch of {len(items)}; retrying as {mid} + {len(items) - mid}")
        return (_run_stage_split(client, model_name, items[:mid], build_prompt, schema, key, costs, label, creative=creative)
                + _run_stage_split(client, model_name, items[mid:], build_prompt, schema, key, costs, label, creative=creative))


def score_batch_size():
    """Transcript windows per scoring call: ``LLM_SCORE_BATCH`` if set, else
    8 for Gemini (1M context) and 3 for an OpenAI-compatible server."""
    raw = os.environ.get("LLM_SCORE_BATCH", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 3 if llm_backend.active() else 8


def get_viral_clips(transcript_result, video_duration, creative=False):
    """Two-pass clip selection: score transcript windows, then detail the best.

    Windowing gives even coverage on long videos (a single call over the whole
    transcript clusters picks near the start), and the cheap scoring pass keeps
    the expensive detail reasoning focused on the shortlist. Cuts are snapped to
    word boundaries so clips don't start/end mid-word.
    """
    language = str(transcript_result.get('language') or 'unknown')
    if llm_backend.active():
        # Self-hosted text model: no Google key needed for this stage.
        client = None
        model_name = llm_backend.model_name()
        print(f"\U0001f916  Analyzing with local LLM at {llm_backend.base_url()} (2-pass: score → detail)...")
    else:
        print("\U0001f916  Analyzing with Gemini (2-pass: score → detail)...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ Notice: GEMINI_API_KEY not found in environment. Seamlessly generating clips via intelligent transcript analysis.")
            min_secs, max_secs = clip_duration_bounds()
            return get_heuristic_clips(transcript_result, video_duration, min_secs=min_secs, max_secs=max_secs)
        client = genai.Client(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL") or 'gemini-2.5-flash'
    print(f"\U0001f916  Model: {model_name} | language: {language}")

    # Full word list — ground truth for snapping cut points.
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})

    try:
        # Scoring windows must be able to CONTAIN a max-length clip (the detail
        # prompt keeps clips inside their candidate window), so scale them with
        # the requested band — a user asking for 60-90s clips on the default
        # 90s windows would get clips squeezed against the window walls.
        min_secs, max_secs = clip_duration_bounds()
        windows = build_transcript_windows(
            transcript_result, video_duration,
            window_seconds=max(90, int(max_secs * 1.5)))
        print(f"   Built {len(windows)} scoring window(s).")
        costs = []

        # --- Pass 1: score windows in batches, keep the highest-scoring ---
        scored = []
        # Local models usually run with a 4-8k context (Ollama defaults to
        # 4096 unless OLLAMA_CONTEXT_LENGTH says otherwise) and 8 windows of
        # transcript do not fit; a silently truncated prompt scores garbage.
        SCORE_BATCH = score_batch_size()
        def _payload(ws):
            return [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]} for w in ws]

        def _score_prompt(ws):
            return gemini_worker.SCORE_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                windows_json=json.dumps(_payload(ws), ensure_ascii=False))

        for b in range(0, len(windows), SCORE_BATCH):
            scored.extend(_run_stage_split(
                client, model_name, windows[b:b + SCORE_BATCH], _score_prompt,
                gemini_worker.ScoreResponse, "windows", costs, "score", creative=creative))

        # Shortlist the top windows; scale with duration so long videos surface
        # more candidates without exploding the detail call.
        scored.sort(key=lambda w: w.get("score", 0), reverse=True)
        target = max(3, min(10, int(video_duration // 90) + 2))
        by_id = {w["id"]: w for w in windows}
        shortlist = [by_id[w["id"]] for w in scored[:target] if w.get("id") in by_id]
        if not shortlist:
            shortlist = windows[:target]  # scoring returned nothing usable
        print(f"   Shortlisted {len(shortlist)} window(s) for detail.")

        # --- Pass 2: detailed clip extraction on the shortlist ---
        min_clips, max_clips = clip_count_targets(len(shortlist))

        def _detail_prompt(ws):
            # A split batch keeps the full clip-count band: a short list can
            # still hold the best clips, and the model returns fewer anyway.
            return gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                min_clips=min_clips, max_clips=max_clips,
                min_secs=min_secs, max_secs=max_secs,
                windows_json=json.dumps(_payload(ws), ensure_ascii=False))

        shorts = _run_stage_split(client, model_name, shortlist, _detail_prompt,
                                  gemini_worker.DetailResponse, "shorts", costs, "detail", creative=creative)
        if len(shorts) > max_clips:
            # By score, never by position: the results arrive in transcript
            # order, so slicing kept the earliest clips and silently dropped
            # the back half of the video. See trim_to_best.
            dropped = len(shorts) - max_clips
            shorts = trim_to_best(shorts, max_clips)
            print(f"   Kept the {max_clips} best-scoring clip(s) of "
                  f"{max_clips + dropped}.")
        # Snap each proposed clip onto real word boundaries (+ a bit of silence).
        for s in shorts:
            ns, ne = snap_clip_to_words(s.get("start", 0), s.get("end", 0), words, video_duration,
                                        min_duration=min_secs, max_duration=max_secs)
            s["start"], s["end"] = ns, ne

        # Aggregate cost across both passes.
        cost_analysis = None
        if costs:
            cost_analysis = {
                "input_tokens": sum(c.get("input_tokens", 0) for c in costs),
                "output_tokens": sum(c.get("output_tokens", 0) for c in costs),
                "total_cost": sum(c.get("total_cost", 0) for c in costs),
                "model": model_name,
            }
            print(f"\U0001f4b0 Total cost ({model_name}, 2-pass, {len(costs)} calls): ${cost_analysis['total_cost']:.6f}")

        if not shorts:
            print("⚠️ 2-pass returned no clips.")
            return None

        result = {"shorts": shorts}
        if cost_analysis:
            result["cost_analysis"] = cost_analysis
        return result
    except gemini_worker.GeminiBlockedError as e:
        # Content-policy rejection: propagate so the job fails with the real
        # reason instead of a generic "no clips found".
        print(f"🚫 {e}")
        raise
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        print("⚠️ Seamlessly falling back to intelligent transcript analysis...")
        min_secs, max_secs = clip_duration_bounds()
        return get_heuristic_clips(transcript_result, video_duration, min_secs=min_secs, max_secs=max_secs)


# --- Speech too sparse to clip by transcript -------------------------------
# The vision path used to fire only on a missing audio TRACK. A nursery-rhyme
# video or a dashcam drive has audio, so it went through transcription, came
# back as one segment ("Uh uh"), produced one scoring window and Gemini
# returned no clips — three failed jobs on 25-aug-2026, one user twice. Speech
# is ~120-160 words/min; below these floors there is nothing to clip by words.
MIN_SPEECH_WORDS_PER_MIN = float(os.environ.get("MIN_SPEECH_WORDS_PER_MIN", "5"))
MIN_SPEECH_WORDS = int(os.environ.get("MIN_SPEECH_WORDS", "8"))
MIN_TOTAL_SPEECH_WORDS = int(os.environ.get("MIN_TOTAL_SPEECH_WORDS", "30"))


def speech_is_sparse(transcript, duration):
    """True when the transcript is too thin to drive clip selection."""
    words = sum(len((seg.get("text") or "").split())
                for seg in (transcript or {}).get("segments", []))
    if words >= MIN_TOTAL_SPEECH_WORDS:
        # A long video with 30+ spoken words across multiple sentences has plenty
        # of dialogue to select clips from. It should never be treated as "silent".
        return False
    minutes = max(float(duration or 0) / 60.0, 1e-6)
    return words < MIN_SPEECH_WORDS or words / minutes < MIN_SPEECH_WORDS_PER_MIN


def _create_visual_proxy(video_path):
    """Creates a fast, highly-compressed 1-fps 480p proxy for Gemini Vision.
    Gemini Vision only samples at 1 fps anyway, so sending a full-res multi-GB
    video wastes bandwidth and causes timeouts/503 errors."""
    proxy_path = os.path.join(tempfile.gettempdir(), f"gemini_proxy_{uuid.uuid4().hex[:8]}.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vf", "scale=480:-2,fps=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
        "-an", proxy_path
    ]
    try:
        run_ffmpeg_command(cmd, timeout=600)
        if os.path.exists(proxy_path) and os.path.getsize(proxy_path) > 0:
            return proxy_path
    except Exception as e:
        print(f"⚠️ Proxy creation failed ({e}) — uploading original file.")
        cleanup_temp_file(proxy_path)
    return video_path


def get_visual_clips(video_path, video_duration, language="en"):
    """Clip a SILENT video by vision: Gemini watches the footage and picks the
    most engaging visual moments (no transcript). Returns the same
    {"shorts", "cost_analysis"} shape as get_viral_clips, or None."""
    print("🎥  Silent video — analyzing with Gemini vision (no transcript)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        if llm_backend.active():
            print("❌ This video has no usable speech, so it has to be clipped by "
                  "watching it, and that needs Gemini (a text-only LLM server "
                  "cannot see the footage). Add a GEMINI_API_KEY for silent videos.")
        else:
            print("❌ Error: GEMINI_API_KEY not found.")
        return None
    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-2.5-flash'
    
    upload_file_path = _create_visual_proxy(video_path)
    is_temp_proxy = (upload_file_path != video_path)
    print(f"🎥  Model: {model_name} | uploading {os.path.basename(upload_file_path)}…")

    file_upload = None
    try:
        file_upload = client.files.upload(file=upload_file_path)
        deadline = time.time() + 300
        while True:
            info = client.files.get(name=file_upload.name)
            state = str(getattr(getattr(info, "state", info), "name", "")).upper()
            if state == "ACTIVE":
                break
            if state == "FAILED":
                print("❌ Gemini could not process the video.")
                return None
            if time.time() > deadline:
                print("❌ Gemini video processing timed out.")
                return None
            time.sleep(2)

        # The vision path has no scoring windows to derive a count from, so the
        # env targets (user request) apply directly over the classic 3-15.
        def _env_int(name, default):
            try:
                return max(1, int(os.environ.get(name, "")))
            except ValueError:
                return default
        v_min_clips = _env_int("CLIP_TARGET_MIN", 3)
        v_max_clips = max(v_min_clips, _env_int("CLIP_TARGET_MAX", 15))
        v_min_secs, v_max_secs = clip_duration_bounds()
        prompt = gemini_worker.VISUAL_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language,
            min_clips=v_min_clips, max_clips=v_max_clips,
            min_secs=v_min_secs, max_secs=v_max_secs)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.VisualResponse,
        )
        response = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=[file_upload, prompt], config=config)
                gemini_worker.raise_if_blocked(response)
                break
            except gemini_worker.GeminiBlockedError:
                raise
            except Exception as e:
                msg = str(e)
                transient = any(tok in msg for tok in (
                    '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED',
                    '500', 'INTERNAL', 'overloaded', 'Deadline'))
                if attempt == max_attempts or not transient:
                    print(f"❌ Gemini vision error: {e}")
                    return None
                wait = 2 ** attempt
                print(f"⚠️ Gemini vision busy ({e}) — retrying in {wait}s...")
                time.sleep(wait)

        if response is None:
            return None
        parsed = json.loads(response.text)
        shorts = parsed.get("shorts") or []
        # Clamp to the real duration; drop anything degenerate.
        clean = []
        for s in shorts:
            s["start"] = max(0.0, float(s.get("start", 0)))
            s["end"] = min(float(video_duration), float(s.get("end", 0)))
            if s["end"] - s["start"] >= 1.0:
                clean.append(s)
        if not clean:
            print("⚠️ Vision pass returned no usable clips.")
            return None

        cost = gemini_worker._calculate_cost_analysis(response, model_name)
        if cost:
            print(f"💰 Vision cost ({model_name}): ${cost.get('total_cost', 0):.6f}")
        result = {"shorts": clean}
        if cost:
            result["cost_analysis"] = cost
        return result
    except gemini_worker.GeminiBlockedError as e:
        print(f"🚫 {e}")
        raise
    except Exception as e:
        print(f"❌ Gemini vision error: {e}")
        return None
    finally:
        if file_upload is not None:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass
        if is_temp_proxy:
            cleanup_temp_file(upload_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--format', type=str, default="auto", choices=["auto", "vertical", "horizontal", "square"],
                        help="Output aspect: vertical/auto (9:16), horizontal (keep 16:9), square (1:1).")
    parser.add_argument('--transcript', type=str,
                        help="Path to a precomputed transcript JSON (transcribe_media shape); skips transcription.")
    parser.add_argument('--bg-audio', type=str, default=None,
                        help="Path to background audio file to mix softly into generated clips.")
    parser.add_argument('--bg-audio-volume', type=float, default=0.18,
                        help="Background audio volume (default 0.18 for low voice / ducked music).")
    parser.add_argument('--subtitle-style', type=str, default="shorts",
                        help="Subtitle style preset: shorts, tiktok, reels, beast, gold, neon, cyber, minimal, classic, boxed, or none.")
    parser.add_argument('--fresh', action='store_true',
                        help="Force fresh clip analysis, bypassing cached metadata.")

    args = parser.parse_args()
    output_format = args.format

    script_start_time = time.time()
    print("🎬 Video processing pipeline initialized.", flush=True)
    if args.url:
        print(f"📥 Preparing to download YouTube video: {args.url}", flush=True)
    else:
        print(f"📁 Source video selected: {os.path.basename(args.input)}", flush=True)
    
    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path
    
    # 1. Get Input Video
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."
        
        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        
        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # Layout choice is per SOURCE video, not per clip: one upload and one call
    # instead of one per clip, and the answer is a property of the material
    # ("this is a screencast"), which does not change between its own clips.
    # It runs before any render so the modules are switched on in time.
    if layout_picker.ENABLED:
        try:
            with open_video_capture(input_video) as _cap:
                _fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
                _duration = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / _fps
                _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            from reframe_v2 import source_already_fits  # imports main back
            # A source already shot vertical has no width to reorganise, and
            # the render passes it through whatever the model says. Asking
            # anyway costs a Gemini call per upload to be ignored.
            if _w and _h and source_already_fits(_w, _h, ASPECT_RATIO):
                print(f"   ↕️  Source is {_w}x{_h} — already vertical, no layout to pick.")
            else:
                layout_picker.pick_and_apply(input_video, _duration)
        except Exception as e:
            print(f"⚠️ Layout choice skipped ({e}) — using the default layout.")

    # 2. Decision: Analyze clips or process whole?
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        # --output is documented as "directory or file". When it names a
        # directory we still need a filename: passing the directory through
        # ends up in os.remove() on it further down and dies with EACCES.
        output_file = args.output
        if (not output_file or os.path.isdir(output_file)
                or output_file.endswith(("/", os.sep))):
            output_file = os.path.join(output_dir, f"{video_title}_vertical.mp4")
        render_clip(input_video, output_file, output_format)
    else:
        # Get duration (needed by both the transcript and the vision path).
        with open_video_capture(input_video) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps else 0

        # 3. Transcribe — unless the video has no audio, in which case fall back
        # to Gemini vision (picks clips from the imagery instead of the speech).
        from transcribe_backends import NoAudioError
        transcript = None
        # Module handover (issue #68): another module already transcribed this
        # exact source with the same backend, so reuse its output. Any problem
        # with the file falls back to transcribing normally rather than failing.
        if args.transcript:
            try:
                with open(args.transcript, 'r') as f:
                    transcript = json.load(f)
                if not transcript.get('segments'):
                    raise ValueError("transcript has no segments")
                print(f"⏩ Reusing precomputed transcript "
                      f"({len(transcript['segments'])} segments) — skipping transcription.")
            except Exception as e:
                print(f"⚠️ Could not use precomputed transcript ({e}) — transcribing normally.")
                transcript = None
        if transcript is None:
            transcript = load_transcript_checkpoint(output_dir, input_video, duration)
            if transcript is not None:
                print(f"♻️ Reusing the transcript from the interrupted run "
                      f"({len(transcript['segments'])} segments) — skipping transcription.")
        if transcript is None:
            try:
                transcript = transcribe_video(input_video)
                save_transcript_checkpoint(output_dir, transcript, input_video, duration)
            except NoAudioError as e:
                print(f"🔇 {e} — switching to visual analysis.")

        # Music-only or wordless footage transcribes to a handful of words.
        # Clip it by what is on screen instead, like a video with no audio.
        raw_transcript = transcript
        if transcript is not None and speech_is_sparse(transcript, duration):
            n_words = sum(len((sg.get("text") or "").split()) for sg in transcript["segments"])
            print(f"🔇 Only {n_words} word(s) of speech in {duration:.0f}s — "
                  f"switching to visual analysis.")
            transcript = None

        # 4. Gemini Analysis (or reuse existing metadata if already generated)
        metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
        clips_data = None
        bypass_cache = args.fresh or os.environ.get("FORCE_FRESH_CLIPS") == "1"
        if not bypass_cache and os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            try:
                with open(metadata_file, 'r') as f:
                    cached = json.load(f)
                if cached.get('shorts'):
                    print(f"♻️ Reusing existing clip metadata ({len(cached['shorts'])} clips) — skipping LLM analysis.")
                    clips_data = cached
                    if transcript is None and cached.get('transcript'):
                        transcript = cached['transcript']
            except Exception:
                clips_data = None
        elif bypass_cache:
            print("🔄 Fresh clip generation requested — analyzing new viral moments with AI...")

        if clips_data is None:
            print("🤖 Analyzing transcript with AI to identify viral moments...", flush=True)
            if transcript is not None:
                clips_data = get_viral_clips(transcript, duration, creative=bypass_cache)
            else:
                clips_data = get_visual_clips(input_video, duration)
                # Fall back to transcript if visual analysis returned no clips but speech exists
                if (not clips_data or 'shorts' not in clips_data) and raw_transcript and raw_transcript.get('segments'):
                    print("⚠️ Vision pass returned no clips — falling back to transcript speech segments.")
                    clips_data = get_viral_clips(raw_transcript, duration, creative=bypass_cache)
                    transcript = raw_transcript

        if not clips_data or 'shorts' not in clips_data:
            if (transcript and transcript.get('segments')) or (raw_transcript and raw_transcript.get('segments')):
                print("⚠️ Falling back to transcript heuristic analysis for clip generation.")
                min_secs, max_secs = clip_duration_bounds()
                clips_data = get_heuristic_clips(transcript or raw_transcript, duration, min_secs=min_secs, max_secs=max_secs)
                if transcript is None:
                    transcript = raw_transcript

        if not clips_data or 'shorts' not in clips_data:
            raise RuntimeError(
                "Clip detection failed — unable to identify usable moments in this video.")
        else:
            print(f"🔥 Found {len(clips_data['shorts'])} clips!")

            # Save metadata. Silent videos have no transcript → no subtitles,
            # which is correct (there's no speech to caption).
            clips_data['transcript'] = transcript or {"language": "none", "segments": []}
            # The clip editor's re-render path needs to find the source video
            # again and reproduce the render settings, so record both. The
            # basename is enough — the file sits in the job dir (URL jobs with
            # --keep-original) or in uploads/ (upload jobs).
            clips_data['source_video'] = os.path.basename(input_video)
            clips_data['output_format'] = output_format
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(clips_data, f, indent=2)
            print(f"   Saved metadata to {metadata_file}")

            # Purge lingering dead temp files / uncompleted staging from prior interrupted runs
            for dead_pattern in ("temp_*.mp4", "*.staging_*.mp4", "*.wm.mp4", "seg_*.mp4"):
                for dead_f in glob.glob(os.path.join(output_dir, dead_pattern)):
                    try:
                        cleanup_temp_file(dead_f)
                    except Exception:
                        pass

            if bypass_cache:
                # Fresh run: purge old clip results so stale previous results are never recycled
                for old_f in glob.glob(os.path.join(output_dir, f"*{video_title}_clip_*.mp4")):
                    try:
                        cleanup_temp_file(old_f)
                    except Exception:
                        pass

            # 5. Process clips in parallel: each worker cuts + renders one
            # clip. Renders are mostly ffmpeg subprocesses (parallelize well);
            # detector inference is serialized internally via DETECT_LOCK.
            def _process_one_clip(i, clip):
                start = clip['start']
                end = clip['end']
                print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")

                # Ensure target output directory exists before any rendering begins
                os.makedirs(output_dir, exist_ok=True)

                clip_filename = f"{video_title}_clip_{i+1}.mp4"
                clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                clip_final_path = os.path.join(output_dir, clip_filename)

                os.makedirs(os.path.dirname(os.path.abspath(clip_temp_path)), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.abspath(clip_final_path)), exist_ok=True)

                # Cache re-use check: if clip already rendered in an earlier session/run, skip re-cut/reframe
                if not bypass_cache:
                    existing_pattern = os.path.join(output_dir, f"*{video_title}_clip_{i+1}.mp4")
                    existing_matches = [
                        f for f in glob.glob(existing_pattern)
                        if os.path.isfile(f) and os.path.getsize(f) > 0 and not os.path.basename(f).startswith("temp_")
                    ]
                    if existing_matches:
                        sub_cands = [f for f in existing_matches if os.path.basename(f).startswith("subtitled_")]
                        hook_cands = [f for f in existing_matches if os.path.basename(f).startswith("hooked_")]
                        raw_cands = [f for f in existing_matches if os.path.basename(f) == clip_filename]
                        existing_clip = (
                            max(sub_cands, key=os.path.getmtime) if sub_cands
                            else max(hook_cands, key=os.path.getmtime) if hook_cands
                            else raw_cands[0] if raw_cands else None
                        )
                        if existing_clip:
                            print(f"   ♻️ Clip {i+1} already rendered ({os.path.basename(existing_clip)}) — skipping re-render.")
                            if not clip.get('is_extracted'):
                                try:
                                    import metadata_extractor
                                    clip['real_metadata'] = metadata_extractor.extract_clip_metadata(
                                        output_dir, clip_final_path if os.path.exists(clip_final_path) else existing_clip,
                                        clip_index=i, existing_transcript=transcript, clip_start=start, clip_end=end
                                    )
                                    clip['is_extracted'] = True
                                except Exception as meta_err:
                                    print(f"   ⚠️ Metadata extraction warning for clip {i+1}: {meta_err}")
                            print(f"CLIP_READY {i} {os.path.basename(existing_clip)}")
                            return True

                try:
                    # ffmpeg cut — re-encoding for precision on strict seconds.
                    # Initial cut is serialized across workers to prevent concurrent
                    # read conflicts on input_video on Windows.
                    cut_command = [
                        'ffmpeg', '-y',
                        '-ss', str(start),
                        '-to', str(end),
                        '-i', input_video,
                        *video_encode_args(QUALITY_FAST),
                        *audio_encode_args(),
                        clip_temp_path
                    ]
                    with CUT_LOCK:
                        try:
                            run_ffmpeg_command(cut_command)
                        except subprocess.CalledProcessError as cut_err:
                            print(f"   ❌ Cut failed for clip {i+1}:")
                            print(format_ffmpeg_error(cut_err, max_lines=30))
                            return False

                    if not ensure_file_unlocked(clip_temp_path, timeout=15) or not (os.path.exists(clip_temp_path) and os.path.getsize(clip_temp_path) > 0):
                        print(f"   ❌ Cut file {clip_temp_path} is locked, missing or 0 bytes!")
                        return False

                    try:
                        print(f"🎥 Tracking subjects & rendering vertical video for clip {i+1}...", flush=True)
                        success = render_clip(clip_temp_path, clip_final_path, output_format)
                    except Exception as render_err:
                        print(f"   ❌ Render failed for clip {i+1}: {render_err}")
                        if isinstance(render_err, subprocess.CalledProcessError):
                            print(format_ffmpeg_error(render_err, max_lines=30))
                        success = False

                    if not success or not (os.path.exists(clip_final_path) and os.path.getsize(clip_final_path) > 0):
                        print(f"   ❌ Final rendered clip {clip_final_path} missing or 0 bytes!")
                        return False

                    ensure_file_unlocked(clip_final_path, timeout=15)

                    deliver_path = clip_final_path

                    if os.environ.get("WATERMARK") == "1":
                        try:
                            wm_pos = os.environ.get("WATERMARK_POSITION", "bottom-right")
                            if apply_watermark(clip_final_path, position=wm_pos):
                                ensure_file_unlocked(clip_final_path, timeout=15)
                        except Exception as wm_err:
                            print(f"   ⚠️ Watermark pass warning for clip {i+1}: {wm_err}")

                    try:
                        import layout_ranges as _layouts
                        clip['layout_ranges'] = _layouts.read(clip_final_path)
                    except Exception as lr_err:
                        print(f"   ⚠️ Layout ranges read warning: {lr_err}")
                        clip['layout_ranges'] = []

                    try:
                        if hook_grounding.wanted(clip.get('layout_ranges', []), end - start):
                            hook_grounding.reground(clip_final_path, clip, transcript, start, end)
                    except Exception as hg_err:
                        print(f"   ⚠️ Hook grounding warning for clip {i+1}: {hg_err}")

                    if os.environ.get("AUTO_HOOK") == "1":
                        try:
                            hooked = auto_hook_clip(clip_final_path, clip)
                            if hooked and os.path.exists(hooked[0]) and os.path.getsize(hooked[0]) > 0:
                                deliver_path, clip['auto_hook'] = hooked
                                ensure_file_unlocked(deliver_path, timeout=15)
                        except Exception as hook_err:
                            print(f"   ⚠️ Auto-hook warning for clip {i+1}: {hook_err}")

                    # Mix background audio if provided (different part for every clip with low voice)
                    bg_audio_file = getattr(args, 'bg_audio', None) or os.environ.get("BG_AUDIO_PATH")
                    bg_audio_vol = float(getattr(args, 'bg_audio_volume', None) or os.environ.get("BG_AUDIO_VOLUME", "0.18"))
                    if bg_audio_file and os.path.exists(bg_audio_file):
                        try:
                            total_count = len(clips)
                            mixed_clip = mix_background_audio(
                                deliver_path, bg_audio_file, clip_index=i,
                                total_clips=total_count, clip_duration=end - start,
                                bg_volume=bg_audio_vol
                            )
                            if mixed_clip and os.path.exists(mixed_clip) and os.path.getsize(mixed_clip) > 0:
                                deliver_path = mixed_clip
                                ensure_file_unlocked(deliver_path, timeout=15)
                        except Exception as bg_err:
                            print(f"   ⚠️ Background audio mixing error for clip {i+1}: {bg_err}")

                    captioned = None
                    try:
                        import layout_ranges as _layouts
                        sub_style = getattr(args, 'subtitle_style', None) or os.environ.get("AUTO_CAPTION_STYLE") or "shorts"
                        captioned = auto_caption_clip(
                            deliver_path, transcript, start, end,
                            split_ranges=_layouts.split_ranges(clip.get('layout_ranges', [])),
                            subtitle_style=sub_style)
                        if captioned and (not os.path.exists(captioned) or os.path.getsize(captioned) == 0):
                            print(f"   ⚠️ Captions file {captioned} missing or 0 bytes — using uncaptioned base")
                            captioned = None
                    except Exception as cap_err:
                        print(f"   ⚠️ Auto-captions warning for clip {i+1}: {cap_err} — using uncaptioned base")
                        captioned = None

                    final_delivery = captioned or deliver_path
                    if not (os.path.exists(final_delivery) and os.path.getsize(final_delivery) > 0):
                        print(f"   ❌ Final delivery file {final_delivery} missing or 0 bytes!")
                        return False

                    print(f"   ✅ Clip {i+1} ready: {final_delivery}")

                    # Real Audio, Speech, Silence, and Speaker Metadata Extraction pass (strictly once)
                    if not clip.get('is_extracted'):
                        try:
                            import metadata_extractor
                            clip_meta = metadata_extractor.extract_clip_metadata(
                                output_dir, clip_final_path, clip_index=i,
                                existing_transcript=transcript,
                                clip_start=start, clip_end=end
                            )
                            clip['real_metadata'] = clip_meta
                            clip['is_extracted'] = True
                        except Exception as meta_err:
                            print(f"   ⚠️ Metadata extraction warning for clip {i+1}: {meta_err}")

                    print(f"CLIP_READY {i} "
                          f"{os.path.basename(captioned or deliver_path)}")
                    return True

                except Exception as clip_err:
                    print(f"   ❌ Error processing clip {i+1}: {clip_err}")
                    if isinstance(clip_err, subprocess.CalledProcessError):
                        print(format_ffmpeg_error(clip_err, max_lines=30))
                    return False
                finally:
                    cleanup_temp_file(clip_temp_path)

            clip_workers = max(int(os.environ.get("CLIP_WORKERS", "3")), 1)
            shorts = clips_data['shorts']
            successful_clips = 0
            with ThreadPoolExecutor(max_workers=min(clip_workers, len(shorts))) as pool:
                futures = {pool.submit(_process_one_clip, i, clip): i
                           for i, clip in enumerate(shorts)}
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        res = future.result()
                        if res:
                            successful_clips += 1
                        else:
                            print(f"   ⚠️ Clip {i+1} did not produce a valid output file on disk.")
                    except Exception as e:
                        print(f"   ❌ Clip {i+1} worker thread error: {type(e).__name__}: {e}")
                        if isinstance(e, subprocess.CalledProcessError):
                            print(format_ffmpeg_error(e, max_lines=30))


            if successful_clips == 0 and len(shorts) > 0:
                print(f"❌ All {len(shorts)} clips failed during rendering!")
                sys.exit(1)

            # Persist per-clip render results added by the workers (auto_hook)
            # so the editor can see what is already burned into each clip.
            if any('auto_hook' in c or 'hook_grounding' in c for c in shorts):
                with open(metadata_file, 'w') as f:
                    json.dump(clips_data, f, indent=2)

            # Clean up any leftover temporary clip cut files
            if output_dir and os.path.exists(output_dir):
                for leftover_temp in glob.glob(os.path.join(output_dir, "temp_*_clip_*.mp4")):
                    cleanup_temp_file(leftover_temp)

    # Clean up original if requested
    if args.url and not args.keep_original:
        if cleanup_temp_file(input_video):
            print(f"🗑️  Cleaned up downloaded video.")
    # The job finished: a later run in this directory must transcribe afresh.
    if not args.skip_analysis:
        clear_transcript_checkpoint(output_dir)

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
