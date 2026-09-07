"""Unit tests for Stage 2 Auto Editor."""

import pytest
from auto_editor import (
    _ease,
    compute_kept_segments,
    compute_contextual_zooms,
    generate_zoom_boxes,
    generate_sendcmd_lines,
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
