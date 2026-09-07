import os
import tempfile
import numpy as np
import pytest
import auto_editor
from auto_editor import (
    _ease,
    AutoEditConfig,
    get_auto_edit_config,
    detect_filler_word_intervals,
    compute_kept_segments,
    compute_contextual_zooms,
    compute_camera_choreography,
    generate_zoom_boxes,
    generate_sendcmd_lines,
    remap_word_timestamps,
    generate_subtitles_srt,
    generate_subtitles_ass,
    check_audio_stream,
    check_audio_volume,
    trim_pacing_segments,
    trim_silence_pacing,
    validate_output_streams,
    MIN_SILENCE_SECONDS,
    BREATH_MARGIN_SECONDS,
)


class TestEase:
    def test_ease_boundaries(self):
        assert _ease(0.0) == 0.0
        assert _ease(1.0) == 1.0
        assert _ease(-0.5) == 0.0
        assert _ease(1.5) == 1.0

    def test_ease_monotonic(self):
        vals = [_ease(i / 10.0) for i in range(11)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]


class TestSilencePacingSegments:
    def test_no_silence_keeps_full_duration(self):
        kept = compute_kept_segments(30.0, [])
        assert kept == [(0.0, 30.0)]

    def test_short_silence_below_threshold_is_ignored(self):
        # A 0.25s pause is below 0.40s threshold -> should not be cut
        silences = [(5.0, 5.25)]
        kept = compute_kept_segments(30.0, silences, min_silence_s=0.40)
        assert kept == [(0.0, 30.0)]

    def test_silence_above_threshold_is_trimmed_with_breath_margin(self):
        # 1.0s silence at 10.0 -> 11.0
        # Cut should be (10.0 + 0.06, 11.0 - 0.06) = (10.06, 10.94)
        silences = [(10.0, 11.0)]
        kept = compute_kept_segments(30.0, silences, min_silence_s=0.40, breath_margin=0.06)
        assert len(kept) == 2
        assert abs(kept[0][0] - 0.0) < 0.001
        assert abs(kept[0][1] - 10.06) < 0.001
        assert abs(kept[1][0] - 10.94) < 0.001
        assert abs(kept[1][1] - 30.0) < 0.001

    def test_multiple_silences_trimmed(self):
        silences = [(5.0, 6.0), (15.0, 16.5)]
        kept = compute_kept_segments(30.0, silences, min_silence_s=0.40, breath_margin=0.06)
        assert len(kept) == 3


class TestContextualZooms:
    def test_zooms_bounded_and_within_range(self):
        fps = 30.0
        n_frames = 300  # 10s video
        zooms = compute_contextual_zooms(n_frames, fps, max_zoom=1.20)
        assert len(zooms) == n_frames
        assert all(1.0 <= z <= 1.25 for z in zooms)

    def test_opening_hook_punches_in(self):
        fps = 30.0
        n_frames = 300  # 10s video
        zooms = compute_contextual_zooms(n_frames, fps, max_zoom=1.20)
        # First frame should be 1.0
        assert abs(zooms[0] - 1.0) < 0.01
        # Around frame 40 (1.33s), zoom should have punched in
        assert max(zooms[20:60]) >= 1.15


class TestZoomBoxesStrictlyEven:
    def test_all_boxes_have_strictly_even_dimensions_and_valid_bounds(self):
        orig_w = 1080
        orig_h = 1920
        centers = [(540, 960), (600, 900), (450, 1000)] * 30
        zooms = [1.0] * 30 + [1.18] * 30 + [1.22] * 30

        boxes = generate_zoom_boxes(centers, zooms, orig_w=orig_w, orig_h=orig_h)
        assert len(boxes) == len(centers)

        for w, h, x, y in boxes:
            # Strictly even dimensions
            assert w % 2 == 0, f"w={w} is not even"
            assert h % 2 == 0, f"h={h} is not even"
            assert x % 2 == 0, f"x={x} is not even"
            assert y % 2 == 0, f"y={y} is not even"
            # Bounds check
            assert 0 <= x <= orig_w - w
            assert 0 <= y <= orig_h - h
            assert w <= orig_w
            assert h <= orig_h


class TestSendcmdLines:
    def test_sendcmd_deduplication(self):
        fps = 30.0
        boxes = [(1080, 1920, 0, 0)] * 10 + [(900, 1600, 90, 160)] * 10
        lines = generate_sendcmd_lines(boxes, fps)
        # Should only emit changes at t=0 and t=10/30=0.3333
        assert len(lines) == 8  # 4 params at t=0, 4 params at t=0.3333
        assert any("crop@c w 1080;" in line for line in lines)
        assert any("crop@c w 900;" in line for line in lines)


class TestRemapWordTimestamps:
    def test_no_cuts_leaves_timestamps_unchanged(self):
        words = [
            {"word": "Bonjour", "start": 1.0, "end": 2.0},
            {"word": "monde", "start": 2.5, "end": 3.5},
        ]
        kept = [(0.0, 10.0)]
        remapped = remap_word_timestamps(words, kept)
        assert len(remapped) == 2
        assert remapped[0]["start"] == 1.0
        assert remapped[0]["end"] == 2.0
        assert remapped[1]["start"] == 2.5
        assert remapped[1]["end"] == 3.5

    def test_silence_cut_shifts_later_words_accurately(self):
        # 3-second silence between 4.0s and 7.0s is cut out
        words = [
            {"word": "Before", "start": 1.0, "end": 2.0},
            {"word": "After", "start": 8.0, "end": 9.0},
        ]
        kept = [(0.0, 4.0), (7.0, 15.0)]
        remapped = remap_word_timestamps(words, kept)
        assert len(remapped) == 2
        # Before cut: untouched
        assert remapped[0]["start"] == 1.0
        assert remapped[0]["end"] == 2.0
        # After cut: 8.0 - (7.0 - 4.0) = 5.0
        assert remapped[1]["start"] == 5.0
        assert remapped[1]["end"] == 6.0

    def test_words_in_cut_gap_clamped_safely(self):
        words = [
            {"word": "Gap", "start": 5.0, "end": 6.0},
        ]
        kept = [(0.0, 4.0), (7.0, 15.0)]
        remapped = remap_word_timestamps(words, kept)
        assert len(remapped) == 1
        assert remapped[0]["start"] == 4.0
        assert remapped[0]["end"] == 4.10


class TestGenerateSubtitlesSrt:
    def test_generates_valid_srt_file(self, tmp_path):
        words = [
            {"word": "Welcome", "start": 0.5, "end": 1.2},
            {"word": "to", "start": 1.3, "end": 1.5},
            {"word": "our", "start": 1.5, "end": 1.8},
            {"word": "channel", "start": 1.8, "end": 2.5},
        ]
        srt_file = str(tmp_path / "test.srt")
        ok = generate_subtitles_srt(words, srt_file, max_chars=20)
        assert ok is True
        with open(srt_file, "r", encoding="utf-8-sig") as f:
            content = f.read()
        assert "1\n00:00:00,500 --> " in content
        assert "Welcome to our" in content or "channel" in content

    def test_empty_words_returns_false(self, tmp_path):
        srt_file = str(tmp_path / "empty.srt")
        assert generate_subtitles_srt([], srt_file) is False


class TestGenerateSubtitlesAss:
    def test_generates_valid_ass_with_invisible_safe_boundaries(self, tmp_path):
        words = [
            {"word": "Welcome", "start": 0.5, "end": 1.2},
            {"word": "to", "start": 1.3, "end": 1.5},
            {"word": "our", "start": 1.5, "end": 1.8},
            {"word": "channel", "start": 1.8, "end": 2.5},
        ]
        ass_file = str(tmp_path / "test.ass")
        ok = generate_subtitles_ass(words, ass_file, margin_v=320, margin_l=120, margin_r=140, font_size=50)
        assert ok is True
        with open(ass_file, "r", encoding="utf-8-sig") as f:
            content = f.read()

        # Check resolution definition is 1080x1920
        assert "PlayResX: 1080" in content
        assert "PlayResY: 1920" in content

        # Check invisible safe boundaries in style line
        style_lines = [l for l in content.splitlines() if l.startswith("Style: Default")]
        assert len(style_lines) == 1
        assert ",2,120,140,320,1" in style_lines[0]

        # Check event dialogue lines
        assert "Dialogue: 0,0:00:00.50," in content

    def test_empty_words_returns_false(self, tmp_path):
        ass_file = str(tmp_path / "empty.ass")
        assert generate_subtitles_ass([], ass_file) is False

    def test_config_defaults_include_safe_boundaries(self):
        cfg = AutoEditConfig()
        assert cfg.caption_margin_v == 320
        assert cfg.caption_margin_l == 120
        assert cfg.caption_margin_r == 140
        assert cfg.caption_font_size == 50
        assert cfg.caption_max_chars == 16


class TestAudioAndStreamValidation:
    def test_check_audio_stream_handles_nonexistent_gracefully(self):
        assert check_audio_stream("nonexistent_video.mp4") is False

    def test_validate_output_streams_raises_on_missing_file(self):
        with pytest.raises(RuntimeError):
            validate_output_streams("missing_output.mp4")

    def test_check_audio_volume_returns_sensible_levels(self, tmp_path):
        import subprocess
        # Generate a 2-second audio sine tone video
        test_video = str(tmp_path / "tone.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=2",
            "-f", "lavfi", "-i", "sine=f=1000:d=2",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "128k",
            test_video
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        vol = check_audio_volume(test_video)
        assert vol["max_volume"] > -25.0
        assert vol["mean_volume"] > -40.0

    def test_trim_silence_pacing_preserves_audio_volume(self, tmp_path):
        import subprocess
        # Generate a 5-second video with tone and test that trim_silence_pacing does not mute audio
        src_video = str(tmp_path / "src_tone.mp4")
        out_video = str(tmp_path / "trimmed_tone.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=5",
            "-f", "lavfi", "-i", "sine=f=1000:d=5",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "128k",
            src_video
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        # Simulate a silence interval in the middle
        silences = [(1.5, 2.5)]
        trimmed = trim_silence_pacing(src_video, out_video, silences, duration=5.0)
        assert trimmed is True
        vol = check_audio_volume(out_video)
        # Verify that audio is NOT muted or near -90dB
        assert vol["max_volume"] > -25.0
        assert vol["mean_volume"] > -40.0


class TestAutoEditConfig:
    def test_default_config(self):
        cfg = get_auto_edit_config()
        assert cfg.max_zoom == 1.15
        assert cfg.speaker_tracking is True
        assert cfg.tilt_headroom_ratio == 0.33
        assert cfg.jump_cut_disguises is True
        assert cfg.ken_burns_drift is True
        assert cfg.visual_polish is True
        assert cfg.loudnorm is True
        assert cfg.loudnorm_i == -14.0

    def test_override_config_from_dict_and_kwargs(self):
        cfg = get_auto_edit_config({"max_zoom": 1.18, "visual_polish": False}, max_zoom=1.22)
        assert cfg.max_zoom == 1.22
        assert cfg.visual_polish is False


class TestFillerWordDetection:
    def test_detects_common_fillers(self):
        transcript_words = [
            {"word": "So", "start": 0.5, "end": 0.8},
            {"word": "um,", "start": 0.9, "end": 1.2},
            {"word": "today", "start": 1.3, "end": 1.8},
            {"word": "uh", "start": 2.0, "end": 2.3},
            {"word": "we", "start": 2.4, "end": 2.6},
        ]
        fillers = detect_filler_word_intervals(transcript_words)
        assert len(fillers) == 2
        # Check that filler word intervals cover "um" and "uh"
        assert fillers[0][0] <= 0.9 and fillers[0][1] >= 1.2
        assert fillers[1][0] <= 2.0 and fillers[1][1] >= 2.3

    def test_empty_transcript_returns_empty(self):
        assert detect_filler_word_intervals([]) == []


class TestFillerWordPacingCuts:
    def test_compute_kept_segments_with_silences_and_fillers(self):
        # 20-second video
        silences = [(5.0, 6.0)]
        filler_intervals = [(10.0, 10.4)]
        kept = compute_kept_segments(20.0, silences=silences, filler_intervals=filler_intervals)
        assert len(kept) == 3
        # First segment: 0.0 to ~5.06
        assert kept[0][0] == 0.0
        assert abs(kept[0][1] - 5.06) < 0.01
        # Second segment: ~5.94 to 10.0
        assert abs(kept[1][0] - 5.94) < 0.01
        assert abs(kept[1][1] - 10.0) < 0.01
        # Third segment: 10.4 to 20.0
        assert abs(kept[2][0] - 10.4) < 0.01
        assert abs(kept[2][1] - 20.0) < 0.01


class TestCameraChoreography:
    def test_camera_choreography_combines_jump_cut_disguises_and_drift(self):
        fps = 30.0
        n_frames = 600  # 20s
        kept = [(0.0, 6.0), (7.0, 13.0), (14.0, 22.0)]
        zooms, x_offsets = compute_camera_choreography(
            n_frames,
            fps,
            kept_segments=kept,
            max_zoom=1.22,
            jump_cut_disguises=True,
            ken_burns_drift=True,
            motion_transitions=True,
        )
        assert len(zooms) == n_frames
        assert len(x_offsets) == n_frames

        # Jump cut disguise: Segment 0 is base 1.0, Segment 1 is base 1.15
        # Frame at 4.0s (in segment 0)
        f_seg0 = int(4.0 * fps)
        # Frame at 8.0s (in segment 1)
        f_seg1 = int(8.0 * fps)
        assert zooms[f_seg0] < 1.10
        assert zooms[f_seg1] >= 1.15

        # Motion transition whip-pan: should have non-zero even offset at cut
        f_cut = int(6.0 * fps)
        assert x_offsets[f_cut] != 0
        assert x_offsets[f_cut] % 2 == 0

    def test_generate_zoom_boxes_with_golden_ratio_headroom(self):
        orig_w = 1080
        orig_h = 1920
        n_frames = 100
        centers = [(540, 650)] * n_frames
        zooms = [1.18] * n_frames
        x_offsets = [16] * n_frames

        boxes = generate_zoom_boxes(
            centers,
            zooms,
            orig_w=orig_w,
            orig_h=orig_h,
            headroom_ratio=0.33,
            x_offsets=x_offsets,
        )
        assert len(boxes) == n_frames
        for w, h, x, y in boxes:
            assert w % 2 == 0
            assert h % 2 == 0
            assert x % 2 == 0
            assert y % 2 == 0
            assert 0 <= x <= orig_w - w
            assert 0 <= y <= orig_h - h
            # Headroom check: face at 650 should result in y positioned based on ~33% headroom
            expected_y = max(0, min(orig_h - h, int(round(650 - h * 0.33))))
            expected_y -= expected_y % 2
            assert abs(y - expected_y) <= 2


class TestCameraStabilization:
    """Test Camera Stabilization: Deadzone threshold, EMA smoothing, and multi-frame pan smoothing."""

    def test_deadzone_locks_micro_jitter(self):
        """Random sub-deadzone fluctuations (<6% of 1080 = 64.8px) must NOT move the camera center."""
        orig_w, orig_h = 1080, 1920
        base_cx, base_cy = 540.0, 960.0
        # 60 frames of micro-jitter within +/- 25px
        np.random.seed(42)
        raw_centers = [
            (base_cx + float(np.random.uniform(-25.0, 25.0)),
             base_cy + float(np.random.uniform(-35.0, 35.0)))
            for _ in range(60)
        ]
        smoothed = auto_editor.stabilize_camera_centers(
            raw_centers,
            orig_w,
            orig_h,
            deadzone_ratio=0.06,
            ema_alpha=0.12,
            pan_window_frames=19
        )
        assert len(smoothed) == 60
        # Since jitter never exceeds deadzone (~64.8px), target remains at raw_centers[0]
        # and smoothed output should stay virtually constant (<= 1px fluctuation)
        all_x = [pt[0] for pt in smoothed]
        all_y = [pt[1] for pt in smoothed]
        assert max(all_x) - min(all_x) <= 2
        assert max(all_y) - min(all_y) <= 2

    def test_smooth_pan_transition_across_frames(self):
        """Subject moving significantly outside deadzone initiates a smooth multi-frame pan."""
        orig_w, orig_h = 1080, 1920
        # Subject at 300px for 30 frames, then smoothly moves to 800px over 30 frames, then stays at 800px
        raw_centers = [(300.0, 960.0)] * 30 + [(300.0 + (800.0 - 300.0) * (i / 30.0), 960.0) for i in range(30)] + [(800.0, 960.0)] * 30
        smoothed = auto_editor.stabilize_camera_centers(
            raw_centers,
            orig_w,
            orig_h,
            deadzone_ratio=0.06,
            ema_alpha=0.12,
            pan_window_frames=19
        )
        assert len(smoothed) == len(raw_centers)
        # Frame-to-frame step should be gradual (no sudden jump > 30px per frame)
        for i in range(1, len(smoothed)):
            step_x = abs(smoothed[i][0] - smoothed[i - 1][0])
            assert step_x <= 25, f"Jerk detected at frame {i}: step={step_x}px"

    def test_sendcmd_deadband_suppresses_micro_updates(self):
        """Changes below deadband_px (<3px) do not emit sendcmd commands."""
        fps = 30.0
        # Initial box, followed by 5 frames with 1px changes, followed by 1 frame with 20px change
        boxes = [
            (1080, 1920, 0, 0),
            (1080, 1920, 1, 0),  # dx = 1 < 3 -> suppressed
            (1080, 1920, 2, 0),  # dx = 2 < 3 -> suppressed
            (1080, 1920, 20, 0), # dx = 20 >= 3 -> emitted
        ]
        lines = auto_editor.generate_sendcmd_lines(boxes, fps, deadband_px=3)
        # Initial frame (4 lines) + frame 3 with 20px change (1 line for x)
        assert len(lines) == 5
        assert any(line.endswith(" x 20;") for line in lines)
        assert not any(line.endswith(" x 1;") for line in lines)
        assert not any(line.endswith(" x 2;") for line in lines)


class TestControlledZoomCadence:
    """Test Controlled Zoom Cadence: 4-6 zooms per minute, >=8s cooldown, keyword triggering, and steady baseline."""

    def test_controlled_zoom_frequency_cap_60s(self):
        """A 60-second clip must produce between 4 and 6 dynamic zooms."""
        fps = 30.0
        n_frames = int(60.0 * fps)
        # Lots of exclamation marks every 3 seconds
        words = [
            {"word": f"Statement {i}!", "start": float(i * 3), "end": float(i * 3 + 1)}
            for i in range(1, 19)
        ]
        zooms = auto_editor.compute_contextual_zooms(
            n_frames,
            fps,
            transcript_words=words,
            max_zoom=1.20,
            zoom_cadence_mode="controlled",
            min_zoom_cooldown_s=8.0,
            max_zooms_per_minute=6,
            min_zooms_per_minute=4,
        )
        assert len(zooms) == n_frames

        # Count individual zoom pulses (where zoom factor rises above 1.05)
        pulse_count = 0
        in_pulse = False
        for z in zooms:
            if z > 1.05 and not in_pulse:
                pulse_count += 1
                in_pulse = True
            elif z <= 1.01 and in_pulse:
                in_pulse = False

        assert 4 <= pulse_count <= 6, f"Expected 4-6 zoom pulses in 60s, got {pulse_count}"

    def test_controlled_zoom_keyword_triggering(self):
        """Marker words ('stop', 'never', 'secret', 'important') trigger zoom events."""
        fps = 30.0
        n_frames = int(30.0 * fps)
        words = [
            {"word": "Hey", "start": 1.0, "end": 1.3},
            {"word": "stop", "start": 10.0, "end": 10.5},     # High-emphasis keyword (outside hook cooldown)
            {"word": "talking", "start": 10.6, "end": 11.0},
            {"word": "the", "start": 19.8, "end": 20.0},
            {"word": "secret", "start": 20.1, "end": 20.6},   # High-emphasis keyword (cooldown >= 8s)
            {"word": "is", "start": 20.7, "end": 21.0},
        ]
        zooms = auto_editor.compute_contextual_zooms(
            n_frames,
            fps,
            transcript_words=words,
            max_zoom=1.22,
            zoom_cadence_mode="controlled",
            min_zoom_cooldown_s=8.0,
            zoom_hold_s=2.5,
        )
        # Check zoom around word 'stop' (t=10.0s)
        f_stop = int(10.5 * fps)
        assert zooms[f_stop] > 1.15, f"Expected zoom on 'stop' keyword, got {zooms[f_stop]}"

        # Check zoom around word 'secret' (t=20.1s)
        f_secret = int(20.6 * fps)
        assert zooms[f_secret] > 1.15, f"Expected zoom on 'secret' keyword, got {zooms[f_secret]}"

    def test_controlled_zoom_minimum_cooldown(self):
        """Successive zooms must respect the minimum 8.0s cooldown."""
        fps = 30.0
        n_frames = int(30.0 * fps)
        # Two exclamation words only 3 seconds apart
        words = [
            {"word": "Wow!", "start": 5.0, "end": 5.5},
            {"word": "Look!", "start": 8.0, "end": 8.5},   # 3s later: must NOT trigger a second zoom!
            {"word": "Amazing!", "start": 16.0, "end": 16.5} # 11s later: triggers second zoom!
        ]
        zooms = auto_editor.compute_contextual_zooms(
            n_frames,
            fps,
            transcript_words=words,
            max_zoom=1.20,
            zoom_cadence_mode="controlled",
            min_zoom_cooldown_s=8.0,
        )
        # Pulse at ~5.0-8.0s is active, but second word at 8.0s should not start a new pulse at 8.0s
        # Frame at 11.5s should be back to 1.0 (between pulses)
        f_cooldown = int(11.5 * fps)
        assert zooms[f_cooldown] <= 1.02

    def test_zoom_eases_back_to_one(self):
        """Zoom pulse eases back down to 1.0x cleanly after hold duration."""
        fps = 30.0
        n_frames = int(15.0 * fps)
        words = [{"word": "Never", "start": 2.0, "end": 2.5}]
        zooms = auto_editor.compute_contextual_zooms(
            n_frames,
            fps,
            transcript_words=words,
            max_zoom=1.20,
            zoom_cadence_mode="controlled",
            zoom_hold_s=2.5,
        )
        # At t=0, baseline is 1.0
        assert zooms[0] == 1.0
        # At peak (t=3.0s), zoom is ~1.20
        assert zooms[int(3.0 * fps)] >= 1.18
        # At t=6.0s (after rise 0.25s + hold 2.5s + fall 0.40s = 3.15s -> ~5.2s), zoom is back to 1.0
        assert zooms[int(6.0 * fps)] == 1.0

    def test_steady_baseline_framing_no_drift(self):
        """In controlled mode, continuous creeping Ken Burns drift is disabled."""
        fps = 30.0
        n_frames = int(20.0 * fps)
        zooms, _ = auto_editor.compute_camera_choreography(
            n_frames,
            fps,
            punch_in_zooms=False,
            zoom_cadence_mode="controlled",
            ken_burns_drift=True,
        )
        # Baseline framing should remain exactly 1.0 across the entire timeline
        assert all(z == 1.0 for z in zooms)


class TestCanvasPositioningAndForegroundResize:
    """Test Subject Scaling (0.5x to 1.5x) and Canvas Positioning (offsets & anchors)."""

    def test_config_canvas_parameters(self):
        cfg = auto_editor.get_auto_edit_config(
            scale_factor=0.85,
            offset_x=20,
            offset_y=-30,
            anchor="bottom"
        )
        assert cfg.scale_factor == 0.85
        assert cfg.offset_x == 20
        assert cfg.offset_y == -30
        assert cfg.anchor == "bottom"

    def test_generate_subtitles_ass_preserves_custom_settings(self):
        """Preserves custom font name, size, colors, stroke, and karaoke style."""
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            words = [
                {"word": "Hello", "start": 0.5, "end": 1.0},
                {"word": "world", "start": 1.1, "end": 1.6},
            ]
            custom_settings = {
                "font_name": "Montserrat",
                "font_size": 42,
                "font_color": "#00FFCC",
                "border_color": "#FF0055",
                "border_width": 4.5,
                "highlight_color": "#FFE500",
                "style": "karaoke",
                "position": "bottom",
            }
            res = auto_editor.generate_subtitles_ass(
                words,
                ass_path,
                subtitle_settings=custom_settings
            )
            assert res is True
            with open(ass_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            assert "Montserrat" in content
            assert "PlayResX: 1080" in content
            assert "PlayResY: 1920" in content
            assert "\\c&H" in content  # Karaoke color tags present
            # Safe bottom margin (MarginV >= 320)
            for line in content.splitlines():
                if line.startswith("Style:"):
                    parts = line.split(",")
                    assert int(parts[21]) >= 300  # MarginV safe zone
                    assert int(parts[19]) >= 100  # MarginL safe zone
                    assert int(parts[20]) >= 120  # MarginR safe zone
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)
