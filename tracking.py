"""Pure helpers for stable, multi-subject video tracking.

The render pipeline deliberately keeps detector calls behind the existing
MediaPipe/YOLO gates. This module contains the small pieces of state-free
tracking math that can be tested without loading either model, which makes it
safe to improve the camera without turning the test suite into a GPU job.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Box = Sequence[float]


def box_iou(a: Box, b: Box) -> float:
    """Return intersection-over-union for ``[x, y, width, height]`` boxes."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1 = float(a[0]), float(a[1])
    ax2, ay2 = ax1 + max(0.0, float(a[2])), ay1 + max(0.0, float(a[3]))
    bx1, by1 = float(b[0]), float(b[1])
    bx2, by2 = bx1 + max(0.0, float(b[2])), by1 + max(0.0, float(b[3]))

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _centre(box: Box) -> Tuple[float, float]:
    return float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0


def match_box_to_track(
    box: Box,
    tracks: Iterable[Dict],
    frame_number: int,
    frame_width: float,
    frame_height: float,
    used_ids: Optional[Iterable[int]] = None,
    max_age: int = 45,
) -> Optional[int]:
    """Choose the existing track that best explains a detection.

    The old tracker matched only horizontal centre distance. That lets two
    people swap IDs when they cross or when one face briefly disappears. The
    cost below combines normalised x/y motion, IoU, and relative size while
    enforcing a per-frame one-to-one assignment. ``None`` means the caller
    should create a new track.
    """
    if not box or frame_width <= 0 or frame_height <= 0:
        return None

    used = {int(value) for value in (used_ids or ())}
    cx, cy = _centre(box)
    area = max(1.0, float(box[2]) * float(box[3]))
    candidates: List[Tuple[float, int]] = []

    for track in tracks:
        try:
            track_id = int(track["id"])
            last_frame = int(track.get("last_frame", frame_number))
            previous = track.get("box")
            if track_id in used or not previous or frame_number - last_frame > max_age:
                continue
            pcx, pcy = _centre(previous)
            dx = abs(cx - pcx) / float(frame_width)
            dy = abs(cy - pcy) / float(frame_height)
            previous_area = max(1.0, float(previous[2]) * float(previous[3]))
            size_delta = abs(math.log(area / previous_area))
            overlap = box_iou(box, previous)

            # A face may move quickly, but a candidate that is both far away
            # and non-overlapping is almost certainly a different person.
            if overlap < 0.01 and (dx > 0.24 or dy > 0.28):
                continue

            # IoU is strongest when available; centre continuity and size keep
            # the match useful during small occlusions or detector box wobble.
            cost = (0.48 * dx) + (0.22 * dy) + (0.22 * (1.0 - overlap)) + (0.08 * min(size_delta, 2.0))
            candidates.append((cost, track_id))
        except (KeyError, TypeError, ValueError, IndexError):
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    # Keep the gate conservative: a new detection is preferable to assigning a
    # face to a stale subject and making the camera teleport across the frame.
    return candidates[0][1] if candidates[0][0] <= 0.34 else None


def scan_boundaries(width: int, zone_count: int = 3) -> List[int]:
    """Return evenly spaced scan-line boundaries for a frame width.

    The director still makes LEFT/CENTER/RIGHT editorial decisions, while the
    motion detector may inspect 3–7 narrower bands. This makes action/context
    detection flexible without changing the public three-camera semantics.
    """
    width = max(1, int(width))
    zone_count = max(3, min(7, int(zone_count)))
    return [round(width * i / zone_count) for i in range(zone_count + 1)]


def stabilize_crop_path(
    xs: Sequence[Optional[int]],
    scene_boundaries: Sequence[Tuple[int, int]],
    strategies: Sequence[str],
    orig_width: int,
    crop_width: int,
    window: int = 5,
    max_step_ratio: float = 0.02,
    outlier_ratio: float = 0.06,
) -> List[Optional[int]]:
    """Remove isolated detector spikes and cap per-frame crop movement.

    Each scene is filtered independently, so intentional scene-cut snaps are
    preserved. Non-TRACK layouts and ``None`` entries are returned untouched.
    """
    output = list(xs)
    if not output or orig_width <= 0:
        return output

    width = max(1, int(window))
    if width % 2 == 0:
        width += 1
    radius = width // 2
    max_step = max(1.0, float(orig_width) * max(0.0, float(max_step_ratio)))
    outlier_distance = max(4.0, float(orig_width) * max(0.0, float(outlier_ratio)))
    max_x = max(0, int(orig_width) - max(0, int(crop_width)))

    for scene_index, (raw_start, raw_end) in enumerate(scene_boundaries):
        strategy = str(strategies[scene_index]).upper() if scene_index < len(strategies) else "TRACK"
        if strategy != "TRACK":
            continue
        start = max(0, int(raw_start))
        end = min(len(output), max(start, int(raw_end)))
        values = [output[i] for i in range(start, end)]
        if len(values) < 2 or any(value is None for value in values):
            continue

        # A median window removes one-frame detector glitches without blurring
        # a genuine sustained move.
        filtered = [int(value) for value in values]
        for i, value in enumerate(filtered):
            lo, hi = max(0, i - radius), min(len(filtered), i + radius + 1)
            neighbourhood = filtered[lo:hi]
            middle = float(median(neighbourhood))
            if abs(value - middle) > outlier_distance:
                filtered[i] = int(round(middle))

        # A second pass guarantees a hard upper bound on movement speed. The
        # first frame remains the detector's scene-start snap by design.
        for i in range(1, len(filtered)):
            previous = filtered[i - 1]
            delta = filtered[i] - previous
            if abs(delta) > max_step:
                filtered[i] = int(round(previous + math.copysign(max_step, delta)))
            filtered[i] = max(0, min(max_x, filtered[i]))

        output[start:end] = filtered

    return output
