"""Rule of Thirds / 3-Zone Dynamic Framing Engine.

Segments 16:9 source footage into three vertical zones:
  - Left Zone   [0, W/3)
  - Center Zone [W/3, 2W/3]
  - Right Zone  (2W/3, W]

Features:
  1. Default framing focuses on the Center Zone (primary subject), filling the
     vertical 9:16 (1080x1920) canvas cleanly without black borders.
  2. Multi-participant tracking across Left, Center, and Right zones.
  3. Dynamic active-speaker switching with a strict 2.0s minimum hold timer to
     prevent camera jitter and rapid oscillation.
  4. Rapid dialogue monitor that detects back-and-forth exchanges and triggers
     split-stack layout or clean speech-boundary cuts.
  5. Contextual "Special Action" detection (pointing, unboxing, showing objects,
     gestures, physical tasks) shifting focus to the action zone for 2-5s
     before smoothly returning to the primary speaker.
  6. Cinematic camera motion with 15-25 frame EMA smoothing and instant cuts
     at scene boundaries.
"""
from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class Zone(IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


@dataclass
class ThreeZoneConfig:
    """Hyperparameters for 3-Zone Dynamic Framing."""
    # Zone division as fractions of source width: widened central focus band (20% to 80%)
    left_boundary: float = 0.20
    right_boundary: float = 0.80

    # Minimum time to hold any zone before allowing an intra-scene camera move
    min_hold_seconds: float = 2.0

    # Deadzone ratio for studio tripod lock (fraction of orig_w): ±8% screen width
    # When subject is within deadzone, camera is 100% frozen stationary (zero shaking)
    deadzone_ratio: float = 0.08  # ±8% of screen width (~154px on 1920w)

    # Maximum pan velocity as fraction of frame width per frame (2.5% = ~48px/frame on 1920w)
    # Prevents camera whipping / sudden jolts during fast subject motion
    max_pan_velocity_ratio: float = 0.025

    # Sticky speaker bonus to prevent jitter / jumping between people in a group
    sticky_speaker_bonus: float = 2.5

    # Switch mode: 'cut' (discrete 1-frame hard snap), 'pan' (cubic ease pan), or 'ema'
    switch_mode: str = "ema"
    pan_frames: int = 15  # 12-18 frame cubic pan

    # Voice corroboration & speech energy thresholds
    speech_energy_min_duration: float = 0.0   # 0.0 in base config; >1.2s in DirectorConfig
    transient_noise_max_duration: float = 1.0  # Ignored transient noise (<1.0s)

    # Action detection & hold timing
    action_min_sustained_seconds: float = 0.0  # 0.0 in base config; >1.5s in DirectorConfig
    action_hold_seconds_min: float = 2.0
    action_hold_seconds_max: float = 4.0
    action_hold_seconds_default: float = 3.0
    action_motion_threshold: float = 12.0  # Mean pixel delta in action zone

    # Camera smoothing
    ema_smoothing_frames: int = 25  # EMA window → alpha ≈ 0.077 (damping 0.06–0.10)
    snap_distance_ratio: float = 0.45  # Snap cut if distance exceeds 45% of width

    # Rapid conversation threshold
    rapid_turn_threshold_seconds: float = 2.5
    rapid_turn_count_trigger: int = 3  # 3 quick switches within window triggers rapid dialogue


@dataclass
class DirectorConfig(ThreeZoneConfig):
    """Hyperparameters tailored for the Director Multi-Camera Engine.

    Operates like a professional TV studio director:
      - Widened Center Scanning Zone (20% to 80%, ±30% from center).
      - Mandatory 2.5 to 3.5s dwell time.
      - ±15% horizontal deadzone with strict anchor lock (Zero pan).
      - Discrete 3-camera shot switching via hard jump cuts (Camera A: Center, B: Left, C: Right).
      - Voice-corroborated side participant tracking (>1.2s speech energy).
      - Action override for sustained motion (>1.5s).
    """
    left_boundary: float = 0.20    # 20% left boundary
    right_boundary: float = 0.80   # 80% right boundary
    min_hold_seconds: float = 3.0  # Enforce 3.0s minimum dwell time (2.5 to 3.5s)
    deadzone_ratio: float = 0.08   # ±8% center deadzone
    switch_mode: str = "cut"       # Discrete hard jump cuts
    pan_frames: int = 15
    speech_energy_min_duration: float = 1.2  # Floor taken >1.2s
    transient_noise_max_duration: float = 1.0 # Ignored transient noises (<1.0s)
    action_min_sustained_seconds: float = 1.5 # Action override motion (>1.5s)
    action_hold_seconds_default: float = 3.0
    action_hold_seconds_min: float = 2.0
    action_hold_seconds_max: float = 4.0


def get_zone_for_x(x: float, width: float, config: Optional[ThreeZoneConfig] = None) -> Zone:
    """Classify an x coordinate into LEFT, CENTER, or RIGHT zone."""
    cfg = config or ThreeZoneConfig()
    norm_x = x / float(width) if width > 0 else 0.5
    if norm_x < cfg.left_boundary:
        return Zone.LEFT
    elif norm_x > cfg.right_boundary:
        return Zone.RIGHT
    return Zone.CENTER


def get_zone_nominal_center(zone: Zone, width: float, config: Optional[ThreeZoneConfig] = None) -> float:
    """Nominal horizontal center for each zone."""
    cfg = config or ThreeZoneConfig()
    if zone == Zone.LEFT:
        return (cfg.left_boundary / 2.0) * width
    elif zone == Zone.RIGHT:
        return (cfg.right_boundary + (1.0 - cfg.right_boundary) / 2.0) * width
    return width / 2.0


def calculate_crop_dimensions(orig_w: int, orig_h: int,
                              aspect_ratio: float = 9.0 / 16.0) -> Tuple[int, int]:
    """Calculate vertical 9:16 crop width and height in source space."""
    crop_h = orig_h
    crop_w = int(round(crop_h * aspect_ratio))
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(round(crop_w / aspect_ratio))

    crop_w -= (crop_w % 2)
    crop_h -= (crop_h % 2)
    return max(2, crop_w), max(2, crop_h)


class ActionDetector:
    """Detects physical tasks, gestures, and object presentations using motion energy."""
    def __init__(self, config: Optional[ThreeZoneConfig] = None):
        self.config = config or ThreeZoneConfig()
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.active_action_zone: Optional[Zone] = None
        self.action_start_frame: int = -1000
        self.action_duration_frames: int = 0
        self.sustained_motion_frames: Dict[Zone, int] = {
            Zone.LEFT: 0, Zone.CENTER: 0, Zone.RIGHT: 0
        }

    def reset(self):
        self.prev_frame_gray = None
        self.active_action_zone = None
        self.action_start_frame = -1000
        self.action_duration_frames = 0
        self.sustained_motion_frames = {
            Zone.LEFT: 0, Zone.CENTER: 0, Zone.RIGHT: 0
        }

    def analyze_frame_motion(self, frame_bgr_or_gray: np.ndarray,
                             frame_idx: int, fps: float,
                             face_boxes: Optional[List[List[int]]] = None,
                             orig_w: Optional[int] = None,
                             orig_h: Optional[int] = None) -> Optional[Tuple[Zone, float]]:
        """Compute motion energy per zone, masking face areas to highlight hand/task actions.

        Returns (detected_zone, motion_intensity) if an action is detected, else None.
        """
        if frame_bgr_or_gray is None or frame_bgr_or_gray.size == 0:
            return None

        # Convert to grayscale if needed
        if frame_bgr_or_gray.ndim == 3:
            gray = (0.299 * frame_bgr_or_gray[:, :, 2] +
                    0.587 * frame_bgr_or_gray[:, :, 1] +
                    0.114 * frame_bgr_or_gray[:, :, 0]).astype(np.uint8)
        else:
            gray = frame_bgr_or_gray

        h, w = gray.shape[:2]
        if self.prev_frame_gray is None or self.prev_frame_gray.shape != gray.shape:
            self.prev_frame_gray = gray.copy()
            return None

        # Compute absolute frame difference
        diff = np.abs(gray.astype(np.int16) - self.prev_frame_gray.astype(np.int16)).astype(np.uint8)
        self.prev_frame_gray = gray.copy()

        # Mask out face boxes to isolate hand movements, objects, and gestures
        mask = np.ones((h, w), dtype=np.uint8)
        if face_boxes:
            # Check coordinate scaling: if face boxes are in source resolution (e.g. 1920x1080)
            max_coord = max(b[0] + b[2] for b in face_boxes) if face_boxes else 0
            if orig_w and orig_w > w:
                scale_x = w / float(orig_w)
                scale_y = (h / float(orig_h)) if orig_h else scale_x
            elif max_coord > w * 1.1:
                scale_x = w / float(max_coord)
                scale_y = scale_x
            else:
                scale_x = 1.0
                scale_y = 1.0

            for box in face_boxes:
                bx, by, bw, bh = box
                sbx = bx * scale_x
                sby = by * scale_y
                sbw = bw * scale_x
                sbh = bh * scale_y

                # Expand mask to cover head, face, neck, and upper chest movement
                pad_x = sbw * 0.35
                pad_y = sbh * 0.35
                bx0 = max(0, min(int(sbx - pad_x), w - 1))
                by0 = max(0, min(int(sby - pad_y), h - 1))
                bx1 = max(bx0 + 1, min(int(sbx + sbw + pad_x), w))
                by1 = max(by0 + 1, min(int(sby + sbh + pad_y * 1.5), h))
                mask[by0:by1, bx0:bx1] = 0

        diff_masked = diff * mask

        # Segment diff into the three vertical zones
        w_left = int(w * self.config.left_boundary)
        w_right = int(w * self.config.right_boundary)

        # Focus especially on middle and lower 70% of frame where hands and objects appear
        y_start = int(h * 0.25)

        zone_diffs = {
            Zone.LEFT: diff_masked[y_start:, :w_left],
            Zone.CENTER: diff_masked[y_start:, w_left:w_right],
            Zone.RIGHT: diff_masked[y_start:, w_right:]
        }

        # Calculate mean motion in each zone
        zone_energy = {}
        for z, patch in zone_diffs.items():
            zone_energy[z] = float(np.mean(patch)) if patch.size > 0 else 0.0

        # Check if an existing action hold is still active
        if self.active_action_zone is not None:
            elapsed = frame_idx - self.action_start_frame
            if elapsed < self.action_duration_frames:
                return self.active_action_zone, zone_energy.get(self.active_action_zone, 0.0)
            else:
                self.active_action_zone = None
                self.sustained_motion_frames = {Zone.LEFT: 0, Zone.CENTER: 0, Zone.RIGHT: 0}

        # Find maximum motion zone
        max_zone = max(zone_energy, key=zone_energy.get)
        max_motion = zone_energy[max_zone]

        if max_motion >= self.config.action_motion_threshold:
            self.sustained_motion_frames[max_zone] += 1
            for z in (Zone.LEFT, Zone.CENTER, Zone.RIGHT):
                if z != max_zone:
                    self.sustained_motion_frames[z] = max(0, self.sustained_motion_frames[z] - 1)

            sustained_s = self.sustained_motion_frames[max_zone] / max(1.0, float(fps))
            is_instant_strong = max_motion >= (self.config.action_motion_threshold * 2.5)
            is_sustained = (sustained_s >= self.config.action_min_sustained_seconds) or (self.config.action_min_sustained_seconds <= 0.0)

            if is_instant_strong or is_sustained:
                duration_s = min(
                    self.config.action_hold_seconds_max,
                    max(self.config.action_hold_seconds_min,
                        self.config.action_hold_seconds_default)
                )
                self.active_action_zone = max_zone
                self.action_start_frame = frame_idx
                self.action_duration_frames = int(round(duration_s * fps))
                return max_zone, max_motion
        else:
            for z in (Zone.LEFT, Zone.CENTER, Zone.RIGHT):
                self.sustained_motion_frames[z] = max(0, self.sustained_motion_frames[z] - 1)

        return None


class ConversationalTurnMonitor:
    """Monitors speaker turn-taking to detect rapid dialogue exchanges."""
    def __init__(self, config: Optional[ThreeZoneConfig] = None):
        self.config = config or ThreeZoneConfig()
        self.switch_history: List[Tuple[int, Zone]] = []  # (frame_idx, Zone)

    def reset(self):
        self.switch_history.clear()

    def record_speaker_switch(self, frame_idx: int, to_zone: Zone, fps: float) -> bool:
        """Record a switch to a new zone. Returns True if rapid conversation is detected."""
        self.switch_history.append((frame_idx, to_zone))

        # Trim old entries outside 3x turn threshold window
        cutoff_frames = int(round(self.config.rapid_turn_threshold_seconds * 3.0 * fps))
        self.switch_history = [
            (f, z) for f, z in self.switch_history if frame_idx - f <= cutoff_frames
        ]

        if len(self.switch_history) < self.config.rapid_turn_count_trigger:
            return False

        # Check if consecutive switches happened rapidly across different zones
        rapid_count = 0
        threshold_frames = int(round(self.config.rapid_turn_threshold_seconds * fps))
        for i in range(1, len(self.switch_history)):
            prev_f, prev_z = self.switch_history[i - 1]
            curr_f, curr_z = self.switch_history[i]
            if prev_z != curr_z and (curr_f - prev_f) <= threshold_frames:
                rapid_count += 1

        return rapid_count >= (self.config.rapid_turn_count_trigger - 1)


class ThreeZoneFramingEngine:
    """Dynamic Rule-of-Thirds Camera Engine for Vertical 9:16 Reframing.

    Coordinates:
      - 3-Zone division (Left, Center, Right)
      - Center Zone default focus (Camera A)
      - Multi-participant speaker tracking
      - Dwell time enforcement (minimum hold duration)
      - Studio Tripod Anchor-Lock & 20% center deadzone (Zero pan)
      - Discrete 3-camera switching (hard cuts or cubic ease pans)
      - Voice-corroborated side participant tracking
      - Contextual 2-4s action/task reframe with direct return to speaker
      - Rapid dialogue detection for split layout or speech cuts
    """
    def __init__(self, orig_w: int, orig_h: int, fps: float = 30.0,
                 aspect_ratio: float = 9.0 / 16.0,
                 config: Optional[ThreeZoneConfig] = None):
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.fps = max(1.0, float(fps))
        self.aspect_ratio = aspect_ratio
        self.config = config or ThreeZoneConfig()

        self.crop_w, self.crop_h = calculate_crop_dimensions(orig_w, orig_h, aspect_ratio)
        self.max_x = max(0, orig_w - self.crop_w)

        # State machine
        self.current_center_x: float = orig_w / 2.0
        self.target_center_x: float = orig_w / 2.0
        self.anchor_center_x: float = orig_w / 2.0
        self.current_center_y: float = orig_h / 2.0
        self.target_center_y: float = orig_h / 2.0
        self.anchor_center_y: float = orig_h / 2.0
        self.committed_zone: Zone = Zone.CENTER
        self.last_zone_switch_frame: int = -10000
        self.last_anchor_switch_frame: int = -10000

        # Jump confirmation & group stability state
        self.pending_anchor_x: Optional[float] = None
        self.pending_anchor_count: int = 0
        self.group_active_frames: int = 0

        # Discrete shot / pan mechanics
        self.pan_start_frame: int = -10000
        self.pan_start_x: float = orig_w / 2.0
        self.pan_target_x: float = orig_w / 2.0
        self.is_panning: bool = False

        # Side participant voice corroboration tracking
        self.side_speech_frames: Dict[Zone, int] = {Zone.LEFT: 0, Zone.RIGHT: 0}

        # Subsystems
        self.action_detector = ActionDetector(self.config)
        self.turn_monitor = ConversationalTurnMonitor(self.config)

        # EMA smoothing factor: alpha = 2 / (N + 1)
        n = max(5, self.config.ema_smoothing_frames)
        self.ema_alpha = 2.0 / (n + 1.0)

        # Rapid dialogue flag for current scene
        self.rapid_dialogue_active: bool = False

    def reset(self, new_center: Optional[float] = None, new_center_y: Optional[float] = None):
        """Reset camera state for a new scene cut."""
        center = self.orig_w / 2.0 if new_center is None else new_center
        center_y = self.orig_h / 2.0 if new_center_y is None else new_center_y
        self.current_center_x = center
        self.target_center_x = center
        self.anchor_center_x = center
        self.current_center_y = center_y
        self.target_center_y = center_y
        self.anchor_center_y = center_y
        self.committed_zone = get_zone_for_x(center, self.orig_w, self.config)
        self.last_zone_switch_frame = -10000
        self.last_anchor_switch_frame = -10000
        self.pending_anchor_x = None
        self.pending_anchor_count = 0
        self.group_active_frames = 0
        self.pan_start_frame = -10000
        self.pan_start_x = center
        self.pan_target_x = center
        self.is_panning = False
        self.side_speech_frames = {Zone.LEFT: 0, Zone.RIGHT: 0}
        self.action_detector.reset()
        self.turn_monitor.reset()
        self.rapid_dialogue_active = False

    def snap_to(self, center_x: float, center_y: Optional[float] = None):
        """Instant cut/snap to target center without panning."""
        clamped = self._clamp_center(center_x)
        self.current_center_x = clamped
        self.target_center_x = clamped
        self.anchor_center_x = clamped
        self.last_anchor_switch_frame = -10000
        self.pending_anchor_x = None
        self.pending_anchor_count = 0
        if center_y is not None:
            clamped_y = self._clamp_center_y(center_y)
            self.current_center_y = clamped_y
            self.target_center_y = clamped_y
            self.anchor_center_y = clamped_y
        self.committed_zone = get_zone_for_x(clamped, self.orig_w, self.config)
        self.is_panning = False

    def _clamp_center(self, cx: float) -> float:
        """Clamp center x so crop box stays strictly within [0, orig_w]."""
        half_w = self.crop_w / 2.0
        return max(half_w, min(cx, self.orig_w - half_w))

    def _clamp_center_y(self, cy: float) -> float:
        """Clamp center y so crop box stays strictly within [0, orig_h]."""
        half_h = self.crop_h / 2.0
        return max(half_h, min(cy, self.orig_h - half_h))

    def _apply_soft_boundary(self, cx: float, prev_cx: float) -> float:
        """Soft-edge deceleration: smoothly slow the camera as it approaches
        the crop boundary instead of hard-clamping (which creates a jarring
        'stuck-at-edge' feel).  A cosine ramp in the outermost 5% of travel
        range tapers velocity to zero, then the final safety clamp catches
        any overshoot.
        """
        half_w = self.crop_w / 2.0
        min_cx = half_w
        max_cx = self.orig_w - half_w
        if max_cx <= min_cx:
            return max(min_cx, min(cx, max_cx))

        margin = (max_cx - min_cx) * 0.05  # 5% soft deceleration zone
        if margin < 1.0:
            return max(min_cx, min(cx, max_cx))

        delta = cx - prev_cx
        # Approaching left edge
        if cx < min_cx + margin and delta < 0:
            t = max(0.0, (cx - min_cx) / margin)  # 0 at edge, 1 at margin start
            cx = prev_cx + delta * (0.5 + 0.5 * t)  # decelerate
        # Approaching right edge
        if cx > max_cx - margin and delta > 0:
            t = max(0.0, (max_cx - cx) / margin)
            cx = prev_cx + delta * (0.5 + 0.5 * t)

        return max(min_cx, min(cx, max_cx))  # final safety clamp

    def _apply_soft_boundary_y(self, cy: float, prev_cy: float) -> float:
        """Soft-edge deceleration for vertical axis."""
        half_h = self.crop_h / 2.0
        min_cy = half_h
        max_cy = self.orig_h - half_h
        if max_cy <= min_cy:
            return max(min_cy, min(cy, max_cy))

        margin = (max_cy - min_cy) * 0.05
        if margin < 1.0:
            return max(min_cy, min(cy, max_cy))

        delta = cy - prev_cy
        if cy < min_cy + margin and delta < 0:
            t = max(0.0, (cy - min_cy) / margin)
            cy = prev_cy + delta * (0.5 + 0.5 * t)
        if cy > max_cy - margin and delta > 0:
            t = max(0.0, (max_cy - cy) / margin)
            cy = prev_cy + delta * (0.5 + 0.5 * t)

        return max(min_cy, min(cy, max_cy))

    def update_frame(self, frame_idx: int,
                     face_candidates: Optional[List[Dict[str, Any]]] = None,
                     active_speaker_idx: Optional[int] = None,
                     frame_image: Optional[np.ndarray] = None,
                     force_snap: bool = False,
                     speech_durations: Optional[Dict[Zone, float]] = None) -> Tuple[int, int, int, int]:
        """Process one frame and return crop box: (x1, y1, crop_w, crop_h)."""
        if force_snap:
            if face_candidates:
                # Target primary face or center
                fc = face_candidates[0]['box']
                target_cx = fc[0] + fc[2] / 2.0
                target_cy = fc[1] + fc[3] / 2.0
                self.snap_to(target_cx, target_cy)
            else:
                self.snap_to(self.orig_w / 2.0, self.orig_h / 2.0)
            self.last_anchor_switch_frame = frame_idx
            self.last_zone_switch_frame = frame_idx
            self.pending_anchor_x = None
            self.pending_anchor_count = 0
            return self.get_current_crop_box()

        # Update side participant speech tracking
        if speech_durations:
            for z in (Zone.LEFT, Zone.RIGHT):
                if z in speech_durations:
                    self.side_speech_frames[z] = int(round(speech_durations[z] * self.fps))
        elif face_candidates and active_speaker_idx is not None and 0 <= active_speaker_idx < len(face_candidates):
            spk_cand = face_candidates[active_speaker_idx]
            spk_cx = spk_cand['box'][0] + spk_cand['box'][2] / 2.0
            spk_zone = get_zone_for_x(spk_cx, self.orig_w, self.config)
            for z in (Zone.LEFT, Zone.RIGHT):
                if spk_zone == z:
                    self.side_speech_frames[z] += 1
                else:
                    self.side_speech_frames[z] = max(0, self.side_speech_frames[z] - 1)
        else:
            for z in (Zone.LEFT, Zone.RIGHT):
                self.side_speech_frames[z] = max(0, self.side_speech_frames[z] - 1)

        # Universal Dwell Time: mandatory hold before allowing any camera anchor change
        min_hold_frames = int(round(self.config.min_hold_seconds * self.fps))
        in_dwell_period = (frame_idx - self.last_anchor_switch_frame) < min_hold_frames

        # Extended horizontal deadzone: ±15% of screen width (~288px on 1920w)
        deadzone_px = self.orig_w * self.config.deadzone_ratio

        # 1. Action / Task Detection
        action_zone = None
        if frame_image is not None:
            face_boxes = [c['box'] for c in face_candidates] if face_candidates else None
            action_res = self.action_detector.analyze_frame_motion(
                frame_image, frame_idx, self.fps, face_boxes=face_boxes,
                orig_w=self.orig_w, orig_h=self.orig_h
            )
            if action_res is not None:
                action_zone, _intensity = action_res

        # 2. Determine Candidate Target Center & Zone
        desired_zone = self.committed_zone
        desired_center = self.anchor_center_x
        is_action_override = False

        if action_zone is not None:
            # Action event takes priority for its duration (2 to 4 seconds)
            is_action_override = True
            desired_zone = action_zone
            desired_center = get_zone_nominal_center(action_zone, self.orig_w, self.config)
        elif face_candidates and len(face_candidates) > 0:
            target_cand = None
            if active_speaker_idx is not None and 0 <= active_speaker_idx < len(face_candidates):
                target_cand = face_candidates[active_speaker_idx]
            else:
                def _score_cand(cand):
                    raw_s = cand.get('score', cand['box'][2] * cand['box'][3])
                    c_x = cand['box'][0] + cand['box'][2] / 2.0
                    if abs(c_x - self.anchor_center_x) <= deadzone_px:
                        return raw_s * self.config.sticky_speaker_bonus
                    return raw_s

                target_cand = max(face_candidates, key=_score_cand)

            box = target_cand['box']
            face_cx = box[0] + box[2] / 2.0

            # Vertical Headroom Alignment (20%–30% from top of 9:16 canvas, avoiding head-chopping)
            top_face_y = min(c['box'][1] for c in face_candidates)
            target_headroom = 0.25  # Nominal 25% headroom
            ideal_center_y = top_face_y + self.crop_h * (0.5 - target_headroom)
            self.target_center_y = self._clamp_center_y(ideal_center_y)

            # Multi-person group hysteresis:
            # Require multi-person balance to persist across consecutive frames before shifting
            # to group midpoint, preventing oscillation when a second face flickers.
            is_balanced_group = False
            if len(face_candidates) > 1 and active_speaker_idx is None:
                sorted_cands = sorted(
                    face_candidates,
                    key=lambda c: c.get('score', c['box'][2] * c['box'][3]),
                    reverse=True
                )
                s0 = sorted_cands[0].get('score', sorted_cands[0]['box'][2] * sorted_cands[0]['box'][3])
                s1 = sorted_cands[1].get('score', sorted_cands[1]['box'][2] * sorted_cands[1]['box'][3])
                if s1 > 0.40 * s0:
                    is_balanced_group = True

            if is_balanced_group:
                self.group_active_frames += 1
            else:
                self.group_active_frames = max(0, self.group_active_frames - 1)

            if self.group_active_frames >= 4:
                all_min_x = min(c['box'][0] for c in face_candidates)
                all_max_x = max(c['box'][0] + c['box'][2] for c in face_candidates)
                group_span = all_max_x - all_min_x
                group_mid = (all_min_x + all_max_x) / 2.0
                if group_span > self.crop_w * 0.90:
                    self.rapid_dialogue_active = True
                face_cx = group_mid

            desired_center = face_cx
            desired_zone = get_zone_for_x(face_cx, self.orig_w, self.config)
        else:
            # No face detected and no active action: return to Center if not already there
            if self.committed_zone != Zone.CENTER:
                desired_zone = Zone.CENTER
                desired_center = get_zone_nominal_center(Zone.CENTER, self.orig_w, self.config)
            else:
                desired_center = self.anchor_center_x
                desired_zone = self.committed_zone

        # 3. Anchor & Dwell Decision with Strict Tripod Lock
        dist_from_anchor = abs(desired_center - self.anchor_center_x)

        if dist_from_anchor <= deadzone_px or in_dwell_period:
            # Within deadzone OR still within mandatory dwell time:
            # Camera MUST REMAIN STRICTLY LOCKED (Zero pan / Tripod Lock)
            self.pending_anchor_x = None
            self.pending_anchor_count = 0
            self.target_center_x = self.anchor_center_x
            if not self.is_panning:
                self.current_center_x = self.anchor_center_x
        else:
            # Outside deadzone AND past dwell time:
            if (self.pending_anchor_x is not None
                    and abs(desired_center - self.pending_anchor_x) <= deadzone_px):
                self.pending_anchor_count += 1
            else:
                self.pending_anchor_x = desired_center
                self.pending_anchor_count = 1

            confirm_threshold = getattr(self.config, 'jump_confirm_frames', 1)
            if self.pending_anchor_count >= confirm_threshold:
                # Confirmed move! Check voice corroboration for side zones if not an action override
                is_corroborated = True
                candidate_zone = get_zone_for_x(self.pending_anchor_x, self.orig_w, self.config)
                if not is_action_override and candidate_zone in (Zone.LEFT, Zone.RIGHT) and self.config.speech_energy_min_duration > 0.0:
                    dur_s = self.side_speech_frames[candidate_zone] / self.fps
                    if dur_s < self.config.speech_energy_min_duration:
                        is_corroborated = False

                if is_corroborated:
                    self.committed_zone = candidate_zone
                    self.last_anchor_switch_frame = frame_idx
                    self.last_zone_switch_frame = frame_idx
                    self.anchor_center_x = self._clamp_center(self.pending_anchor_x)
                    self.target_center_x = self.anchor_center_x
                    self.pending_anchor_x = None
                    self.pending_anchor_count = 0

                    if self.config.switch_mode == "cut":
                        # Discrete studio camera hard cut (1-frame snap)
                        self.current_center_x = self.target_center_x
                        self.is_panning = False
                    elif self.config.switch_mode == "pan":
                        # 12-18 frame intentional cubic pan
                        self.pan_start_frame = frame_idx
                        self.pan_start_x = self.current_center_x
                        self.pan_target_x = self.target_center_x
                        self.is_panning = True

                    # Check for rapid conversational turn-taking
                    is_rapid = self.turn_monitor.record_speaker_switch(
                        frame_idx, self.committed_zone, self.fps
                    )
                    if is_rapid:
                        self.rapid_dialogue_active = True
                else:
                    self.target_center_x = self.anchor_center_x
                    if not self.is_panning:
                        self.current_center_x = self.anchor_center_x
            else:
                # Still confirming move, hold anchor
                self.target_center_x = self.anchor_center_x
                if not self.is_panning:
                    self.current_center_x = self.anchor_center_x

        # 4. Motion Execution & Stabilization
        prev_cx = self.current_center_x
        max_step = self.orig_w * self.config.max_pan_velocity_ratio

        if self.is_panning:
            pan_frames = max(1, self.config.pan_frames)
            elapsed = frame_idx - self.pan_start_frame + 1
            if elapsed >= pan_frames:
                self.is_panning = False
                self.current_center_x = self.pan_target_x
            else:
                t = min(1.0, max(0.0, elapsed / float(pan_frames)))
                cubic_t = t * t * (3.0 - 2.0 * t)
                self.current_center_x = self.pan_start_x + (self.pan_target_x - self.pan_start_x) * cubic_t
        elif self.config.switch_mode == "cut":
            # In cut mode, once anchored, camera is 100% frozen stationary at anchor
            self.current_center_x = self.anchor_center_x
        else:
            dist = abs(self.target_center_x - self.current_center_x)
            if dist < 1.0:
                self.current_center_x = self.target_center_x
            elif dist > (self.orig_w * self.config.snap_distance_ratio):
                self.current_center_x = self.target_center_x
            else:
                self.current_center_x = (
                    self.ema_alpha * self.target_center_x +
                    (1.0 - self.ema_alpha) * self.current_center_x
                )

        # Clamp per-frame X velocity to max_pan_velocity_ratio (prevent whipping)
        delta_x = self.current_center_x - prev_cx
        if abs(delta_x) > max_step:
            self.current_center_x = prev_cx + math.copysign(max_step, delta_x)

        # Apply soft boundary deceleration (smooth edge approach, no jarring stop)
        self.current_center_x = self._apply_soft_boundary(self.current_center_x, prev_cx)

        # Y-axis: EMA + velocity clamp + soft boundary
        prev_cy = self.current_center_y
        max_step_y = self.orig_h * self.config.max_pan_velocity_ratio

        dist_y = abs(self.target_center_y - self.current_center_y)
        if dist_y < 1.0 or force_snap:
            self.current_center_y = self.target_center_y
        else:
            self.current_center_y = (
                self.ema_alpha * self.target_center_y +
                (1.0 - self.ema_alpha) * self.current_center_y
            )

        # Clamp per-frame Y velocity
        delta_y = self.current_center_y - prev_cy
        if abs(delta_y) > max_step_y:
            self.current_center_y = prev_cy + math.copysign(max_step_y, delta_y)

        # Apply soft boundary deceleration for Y
        self.current_center_y = self._apply_soft_boundary_y(self.current_center_y, prev_cy)

        return self.get_current_crop_box()

    def get_current_crop_box(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, crop_w, crop_h) with strict canvas bounds clamping."""
        clamped_cx = self._clamp_center(self.current_center_x)
        x1 = int(round(clamped_cx - self.crop_w / 2.0))
        x1 = max(0, min(x1, self.max_x))
        x1 -= x1 % 2

        clamped_cy = self._clamp_center_y(self.current_center_y)
        y1 = int(round(clamped_cy - self.crop_h / 2.0))
        y1 = max(0, min(y1, self.orig_h - self.crop_h))
        y1 -= y1 % 2
        return x1, y1, self.crop_w, self.crop_h


class DirectorMultiCameraEngine(ThreeZoneFramingEngine):
    """Professional Director Multi-Camera Engine.

    Treats the 3 zones as discrete studio cameras:
      - Camera A: Center / Host (default focus)
      - Camera B: Left / Guest
      - Camera C: Right / Guest

    Features:
      1. Shot Stabilization: Anchor-lock with 20% center deadzone (0.0px camera movement).
      2. Dwell Time Enforcement: 2.5 to 3.5s minimum shot duration (default 3.0s).
      3. 3-Zone Discrete Shot Switching: Clean hard jump cuts (or 12-18 frame cubic pans).
      4. Strict Speaker Recognition: Voice-corroborated side participant tracking (>1.2s speech).
      5. Action Override: Sustained motion (>1.5s) triggers action shot held 2-4s before cutting back.
    """
    def __init__(self, orig_w: int, orig_h: int, fps: float = 30.0,
                 aspect_ratio: float = 9.0 / 16.0,
                 config: Optional[ThreeZoneConfig] = None):
        cfg = config or DirectorConfig()
        super().__init__(orig_w, orig_h, fps=fps, aspect_ratio=aspect_ratio, config=cfg)


def calculate_3zone_trajectory(
    video_path: str,
    scenes_boundaries: List[Tuple[int, int]],
    fps: float,
    orig_w: int,
    orig_h: int,
    config: Optional[ThreeZoneConfig] = None
) -> Tuple[List[int], Dict[int, str]]:
    """Compute per-frame crop x positions and detected layout recommendations.

    Recommends 'SPLIT' layout for scenes exhibiting rapid back-and-forth dialogue.
    Returns:
      (xs, layout_recommendations)
    """
    import cv2
    from ffmpeg_utils import open_video_capture
    import main as m

    cfg = config or ThreeZoneConfig()
    engine = ThreeZoneFramingEngine(orig_w, orig_h, fps=fps, config=cfg)

    small_w = min(640, orig_w)
    small_w -= (small_w % 2)
    small_h = max(2, int(orig_h * small_w / orig_w))
    small_h -= (small_h % 2)
    scale = orig_w / float(small_w)

    xs: List[int] = []
    layout_recommendations: Dict[int, str] = {}

    current_scene_idx = 0
    total_scenes = len(scenes_boundaries)

    with open_video_capture(video_path) as cap:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Handle scene boundaries
            if current_scene_idx < total_scenes:
                start_f, end_f = scenes_boundaries[current_scene_idx]
                if frame_idx >= end_f and current_scene_idx < total_scenes - 1:
                    # Check if previous scene had rapid dialogue
                    if engine.rapid_dialogue_active:
                        layout_recommendations[current_scene_idx] = 'SPLIT'
                    current_scene_idx += 1
                    engine.reset()

            is_scene_start = (
                current_scene_idx < total_scenes and
                frame_idx == scenes_boundaries[current_scene_idx][0]
            )

            if is_scene_start:
                engine.reset()

            # Downsample for fast analysis
            small_frame = cv2.resize(frame, (small_w, small_h))

            # Detect faces on detect stride or scene start
            candidates = []
            if frame_idx % m.DETECT_STRIDE == 0 or is_scene_start:
                raw_cands = m.detect_face_candidates(small_frame)
                for cand in raw_cands:
                    bx, by, bw, bh = cand['box']
                    scaled_box = [
                        int(bx * scale),
                        int(by * scale),
                        int(bw * scale),
                        int(bh * scale)
                    ]
                    candidates.append({
                        'box': scaled_box,
                        'score': cand.get('score', scaled_box[2] * scaled_box[3])
                    })

            x1, _y1, _cw, _ch = engine.update_frame(
                frame_idx=frame_idx,
                face_candidates=candidates if candidates else None,
                frame_image=small_frame,
                force_snap=is_scene_start
            )
            xs.append(x1)
            frame_idx += 1

    # Check last scene for rapid dialogue
    if engine.rapid_dialogue_active and current_scene_idx < total_scenes:
        layout_recommendations[current_scene_idx] = 'SPLIT'

    return xs, layout_recommendations
