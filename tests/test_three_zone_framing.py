"""Tests for Rule of Thirds / 3-Zone Dynamic Framing Engine."""
import pytest
import numpy as np

import three_zone_framing as tzf
from three_zone_framing import (
    Zone,
    ThreeZoneConfig,
    DirectorConfig,
    ThreeZoneFramingEngine,
    DirectorMultiCameraEngine,
    ActionDetector,
    ConversationalTurnMonitor,
    get_zone_for_x,
    get_zone_nominal_center,
    calculate_crop_dimensions,
)


class TestZoneGeometry:
    """Validate 3-zone division according to the Rule of Thirds."""

    def test_zone_classification_1080p(self):
        w = 1920
        # Left Zone: [0, 640)
        assert get_zone_for_x(100, w) == Zone.LEFT
        assert get_zone_for_x(600, w) == Zone.LEFT

        # Center Zone: [640, 1280]
        assert get_zone_for_x(650, w) == Zone.CENTER
        assert get_zone_for_x(960, w) == Zone.CENTER
        assert get_zone_for_x(1200, w) == Zone.CENTER

        # Right Zone: (1280, 1920]
        assert get_zone_for_x(1300, w) == Zone.RIGHT
        assert get_zone_for_x(1800, w) == Zone.RIGHT

    def test_zone_nominal_centers(self):
        w = 1920
        assert get_zone_nominal_center(Zone.LEFT, w) == 320.0
        assert get_zone_nominal_center(Zone.CENTER, w) == 960.0
        assert get_zone_nominal_center(Zone.RIGHT, w) == 1600.0

    def test_crop_dimensions_9_16(self):
        # 1920x1080 landscape input -> 9:16 vertical crop
        crop_w, crop_h = calculate_crop_dimensions(1920, 1080, aspect_ratio=9 / 16)
        assert crop_h == 1080
        # 1080 * 9 / 16 = 607.5 -> 608 (even)
        assert crop_w == 608
        assert crop_w % 2 == 0 and crop_h % 2 == 0
        assert crop_w < 1920

    def test_crop_dimensions_already_vertical(self):
        # 1080x1920 vertical input -> keeps full dimensions
        crop_w, crop_h = calculate_crop_dimensions(1080, 1920, aspect_ratio=9 / 16)
        assert crop_w == 1080
        assert crop_h == 1920


class TestDefaultFocusAndHold:
    """Validate Center Zone default focus and minimum hold time."""

    def test_default_focus_is_center(self):
        engine = ThreeZoneFramingEngine(1920, 1080, fps=30.0)
        # Without any faces or action, camera starts and stays centered
        x1, y1, cw, ch = engine.update_frame(0)
        center_x = x1 + cw / 2.0
        assert engine.committed_zone == Zone.CENTER
        assert abs(center_x - 960.0) <= 2.0
        assert y1 == 0
        assert cw == 608
        assert ch == 1080

    def test_minimum_hold_time_enforced(self):
        cfg = ThreeZoneConfig(min_hold_seconds=2.0)
        fps = 30.0
        engine = ThreeZoneFramingEngine(1920, 1080, fps=fps, config=cfg)

        # Subject in LEFT zone (x=200) locks on
        left_cand = [{'box': [150, 200, 100, 100], 'score': 10000}]
        engine.update_frame(0, face_candidates=left_cand)
        assert engine.committed_zone == Zone.LEFT

        # Right zone subject appears at frame 10 (0.33s later, < 2.0s hold)
        right_cand = [{'box': [1550, 200, 100, 100], 'score': 20000}]
        engine.update_frame(10, face_candidates=right_cand)
        # MUST hold LEFT zone because 2.0s (60 frames) have not elapsed
        assert engine.committed_zone == Zone.LEFT

        # Advance past 60 frames (e.g. frame 65 = > 2.0s)
        engine.update_frame(65, face_candidates=right_cand)
        # Now it is allowed to switch to RIGHT zone
        assert engine.committed_zone == Zone.RIGHT


class TestActiveSpeakerSwitching:
    """Validate switching between participants in different zones."""

    def test_switching_to_active_speaker(self):
        engine = ThreeZoneFramingEngine(1920, 1080, fps=30.0)

        cands = [
            {'box': [250, 200, 100, 100], 'score': 5000},   # Left
            {'box': [1550, 200, 100, 100], 'score': 5000}   # Right
        ]

        # Active speaker is index 0 (Left)
        engine.update_frame(0, face_candidates=cands, active_speaker_idx=0)
        assert engine.committed_zone == Zone.LEFT

        # Later past hold duration (e.g. frame 70), active speaker is index 1 (Right)
        engine.update_frame(70, face_candidates=cands, active_speaker_idx=1)
        assert engine.committed_zone == Zone.RIGHT


class TestSpecialActionDetection:
    """Validate contextual action/task detection (pointing, unboxing, showing objects)."""

    def test_action_triggers_focus_shift_and_holds_2_to_5s(self):
        cfg = ThreeZoneConfig(
            action_motion_threshold=10.0,
            action_hold_seconds_default=3.0,
            min_hold_seconds=0.1
        )
        fps = 30.0
        engine = ThreeZoneFramingEngine(1920, 1080, fps=fps, config=cfg)

        # Baseline frame
        f0 = np.zeros((270, 480, 3), dtype=np.uint8)
        engine.update_frame(0, frame_image=f0)

        # Create localized motion in RIGHT zone (hands/object interaction)
        f1 = f0.copy()
        # Right zone is x > 320 in 480-wide frame, y > 70
        f1[80:200, 350:450] = 200  # Strong motion difference

        # Update at frame 1: action should be detected in RIGHT zone
        engine.update_frame(1, frame_image=f1)
        assert engine.committed_zone == Zone.RIGHT

        # Subsequent quiet frames should maintain action hold for ~3.0s (90 frames)
        f_quiet = f1.copy()
        engine.update_frame(30, frame_image=f_quiet)
        assert engine.committed_zone == Zone.RIGHT

        # After 4.0s (120 frames), action expires and camera returns to center or primary face
        engine.update_frame(150, frame_image=f_quiet)
        assert engine.committed_zone == Zone.CENTER


class TestRapidDialogueDetection:
    """Validate conversational turn monitoring for rapid exchanges."""

    def test_rapid_back_and_forth_triggers_flag(self):
        cfg = ThreeZoneConfig(
            rapid_turn_threshold_seconds=2.0,
            rapid_turn_count_trigger=3
        )
        monitor = ConversationalTurnMonitor(cfg)
        fps = 30.0

        # Speaker switch at t=0 (frame 0) to LEFT
        r1 = monitor.record_speaker_switch(0, Zone.LEFT, fps)
        assert not r1

        # Speaker switch at t=1.0s (frame 30) to RIGHT
        r2 = monitor.record_speaker_switch(30, Zone.RIGHT, fps)
        assert not r2

        # Speaker switch at t=2.0s (frame 60) back to LEFT
        r3 = monitor.record_speaker_switch(60, Zone.LEFT, fps)
        # 3 rapid switches across zones -> rapid dialogue detected!
        assert r3 is True


class TestCinematicSmoothing:
    """Validate 15-25 frame EMA smoothing and instant snap on cuts."""

    def test_ema_smoothing_prevents_jump(self):
        cfg = ThreeZoneConfig(ema_smoothing_frames=20, min_hold_seconds=0.0)
        engine = ThreeZoneFramingEngine(1920, 1080, fps=30.0, config=cfg)

        engine.snap_to(960.0)  # Start dead-center
        init_crop = engine.get_current_crop_box()[0]

        # Target suddenly moves to x=400 (smooth pan, not snap)
        cands = [{'box': [350, 200, 100, 100], 'score': 10000}]
        x1, _, _, _ = engine.update_frame(1, face_candidates=cands)

        # Smooth EMA should move gradually towards 400, not jump all the way
        assert x1 < init_crop
        # Target crop for 400 is max(0, 400 - 304) = 96
        # Initial crop was 960 - 304 = 656
        # In 1 frame, with alpha ~ 0.095, x moves only ~50px
        assert x1 > 500

    def test_force_snap_cuts_instantly(self):
        engine = ThreeZoneFramingEngine(1920, 1080, fps=30.0)
        engine.snap_to(960.0)

        cands = [{'box': [350, 200, 100, 100], 'score': 10000}]
        x1, _, _, _ = engine.update_frame(1, face_candidates=cands, force_snap=True)

        # With force_snap=True, it jumps directly to the target box center (400 - 304 = 96)
        assert x1 == 96


class TestCanvasFillAndBounds:
    """Validate edge-to-edge 9:16 delivery (1080x1920) without black bars."""

    def test_crop_bounds_never_exceed_frame(self):
        engine = ThreeZoneFramingEngine(1920, 1080, fps=30.0)

        # Extremely far left
        engine.snap_to(-500.0)
        x1, y1, cw, ch = engine.get_current_crop_box()
        assert x1 == 0
        assert x1 + cw <= 1920
        assert y1 >= 0 and y1 + ch <= 1080

        # Extremely far right
        engine.snap_to(5000.0)
        x1, y1, cw, ch = engine.get_current_crop_box()
        assert x1 + cw == 1920
        assert x1 >= 0


class TestStudioTripodStability:
    """Verify camera does NOT move or shake while person is talking (face centered, 0 movement)."""

    def test_camera_remains_frozen_while_person_talks_and_nods(self):
        engine = ThreeZoneFramingEngine(1920, 1080, fps=30.0)
        engine.snap_to(960.0)  # Face starts dead center
        initial_x = engine.get_current_crop_box()[0]

        # Person is talking: micro-movements of head/lips (e.g. 960 -> 955 -> 966 -> 958)
        # All within deadzone (~120px)
        for frame in range(1, 45):
            jitter_cx = 960.0 + 15.0 * np.sin(frame * 0.3)
            cands = [{'box': [int(jitter_cx - 50), 200, 100, 100], 'score': 10000}]
            x1, _, _, _ = engine.update_frame(frame, face_candidates=cands)
            # Camera MUST REMAIN STRICTLY FROZEN: 0 movement
            assert x1 == initial_x, f"Camera moved at frame {frame}: {x1} != {initial_x}"


class TestGroupFramingDecisions:
    """Verify intelligent decisions in group scenes (sticky speaker focus, no shaking/flapping)."""

    def test_group_sticky_speaker_prevents_flapping(self):
        engine = ThreeZoneFramingEngine(1920, 1080, fps=30.0)

        # Two people in group: Speaker A at x=400, Speaker B at x=600
        # Speaker A has slightly higher initial score
        group_cands = [
            {'box': [350, 200, 100, 100], 'score': 10000},  # Person A (x=400)
            {'box': [550, 200, 100, 100], 'score': 9500},   # Person B (x=600)
        ]

        engine.update_frame(0, face_candidates=group_cands, force_snap=True)
        locked_crop = engine.get_current_crop_box()[0]

        # Person B's detection fluctuates higher temporarily (e.g. B leans forward)
        fluctuating_cands = [
            {'box': [350, 200, 100, 100], 'score': 10000},  # Person A
            {'box': [550, 200, 100, 100], 'score': 11000},  # Person B temporarily slightly higher
        ]

        # With sticky speaker bonus, camera must NOT jump to Person B
        for f in range(1, 30):
            x1, _, _, _ = engine.update_frame(f, face_candidates=fluctuating_cands)
            assert x1 == locked_crop, f"Camera flapped to secondary speaker at frame {f}!"


class TestDirectorMultiCameraEngine:
    """Validate Director Multi-Camera Engine rules:
    1. Shot Stabilization: Strict Anchor Lock & 20% center deadzone (Zero pan).
    2. Dwell Time Enforcement: Mandatory 2.5 - 3.5s minimum shot duration.
    3. 3-Zone Discrete Shot Switching: Clean hard jump cuts (Camera A: Center, B: Left, C: Right).
    4. Voice-corroborated side participant tracking (>1.2s speech energy).
    5. Action override: sustained motion (>1.5s) held 2-4s before cutting back.
    """

    def test_discrete_hard_jump_cut_on_zone_switch(self):
        engine = DirectorMultiCameraEngine(1920, 1080, fps=30.0)
        # Starts on Camera A (Center, x=960)
        x1, _, cw, _ = engine.update_frame(0)
        assert x1 == 656  # 960 - 304

        # Subject takes floor in Camera B (Left, x=300) with corroborated speech > 1.2s (40 frames)
        cands = [{'box': [250, 200, 100, 100], 'score': 10000}]
        # Speech energy passes corroborated duration
        speech = {Zone.LEFT: 1.5}
        # First frame after dwell time (> 3.0s, e.g. frame 95)
        x_cut, _, _, _ = engine.update_frame(95, face_candidates=cands, speech_durations=speech)

        # Must be an INSTANT 1-frame hard jump cut to target (300 - 304 = 0 clamped to 0)
        assert engine.committed_zone == Zone.LEFT
        assert x_cut == 0  # Instant cut, NOT wandering or floating

    def test_dwell_time_enforces_minimum_3s_hold(self):
        engine = DirectorMultiCameraEngine(1920, 1080, fps=30.0)
        # Face in LEFT zone starts locked at frame 0
        left_cands = [{'box': [250, 200, 100, 100], 'score': 10000}]
        engine.update_frame(0, face_candidates=left_cands, speech_durations={Zone.LEFT: 2.0})
        assert engine.committed_zone == Zone.LEFT
        left_crop = engine.get_current_crop_box()[0]

        # Right participant tries to take floor at frame 60 (2.0s later, < 3.0s dwell time)
        right_cands = [{'box': [1550, 200, 100, 100], 'score': 10000}]
        x1, _, _, _ = engine.update_frame(60, face_candidates=right_cands, speech_durations={Zone.RIGHT: 2.0})

        # MUST hold LEFT zone because 3.0s (90 frames) have not elapsed
        assert engine.committed_zone == Zone.LEFT
        assert x1 == left_crop

        # Past 3.0s dwell time (frame 95 = 3.17s)
        x2, _, _, _ = engine.update_frame(95, face_candidates=right_cands, speech_durations={Zone.RIGHT: 2.0})
        assert engine.committed_zone == Zone.RIGHT
        # Instant cut to Right zone
        assert x2 > 1200

    def test_voice_corroboration_filters_transient_noise(self):
        engine = DirectorMultiCameraEngine(1920, 1080, fps=30.0)
        # Host on Camera A (Center)
        host_cands = [
            {'box': [910, 200, 100, 100], 'score': 10000},  # Host (Center)
            {'box': [250, 200, 100, 100], 'score': 9000},   # Guest (Left)
        ]
        engine.update_frame(0, face_candidates=host_cands, active_speaker_idx=0)
        assert engine.committed_zone == Zone.CENTER
        center_crop = engine.get_current_crop_box()[0]

        # Guest on Left makes a brief noise (cough/laughter: 0.5s duration < 1.0s) at frame 100
        engine.update_frame(100, face_candidates=host_cands, active_speaker_idx=1,
                            speech_durations={Zone.LEFT: 0.5})
        # Camera MUST REMAIN locked on Host in Center (noise ignored)
        assert engine.committed_zone == Zone.CENTER
        assert engine.get_current_crop_box()[0] == center_crop

        # Guest takes the floor and speaks for >1.2s (e.g. 1.4s) at frame 120
        engine.update_frame(120, face_candidates=host_cands, active_speaker_idx=1,
                            speech_durations={Zone.LEFT: 1.4})
        # Now camera cuts to Guest on Left
        assert engine.committed_zone == Zone.LEFT

    def test_action_override_sustained_motion_and_return(self):
        engine = DirectorMultiCameraEngine(1920, 1080, fps=30.0)
        # Starts on Center
        f0 = np.zeros((270, 480, 3), dtype=np.uint8)
        engine.update_frame(0, frame_image=f0)
        assert engine.committed_zone == Zone.CENTER

        # Sustained physical action in RIGHT zone (hands/object interaction)
        f_action = f0.copy()
        f_action[80:200, 350:450] = 200

        # Feed motion for 1.6s (> 1.5s sustained threshold, 48 frames at 30fps)
        for f in range(1, 50):
            engine.update_frame(f, frame_image=f_action)

        # Action cut should have triggered on RIGHT zone
        assert engine.committed_zone == Zone.RIGHT

        # Quiet frame during hold (hold duration: 3.0s = 90 frames)
        f_quiet = f_action.copy()
        engine.update_frame(80, frame_image=f_quiet)
        assert engine.committed_zone == Zone.RIGHT

        # After action hold expires (> 140 frames from action start), cuts back to Center
        engine.update_frame(200, frame_image=f_quiet)
        assert engine.committed_zone == Zone.CENTER


