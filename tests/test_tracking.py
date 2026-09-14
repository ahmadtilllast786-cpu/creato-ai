"""Pure regression tests for Creato's multi-face tracking foundation."""

from tracking import box_iou, match_box_to_track, scan_boundaries, stabilize_crop_path


def test_box_iou_for_overlapping_and_disjoint_boxes():
    assert box_iou([0, 0, 10, 10], [5, 0, 10, 10]) == 1 / 3
    assert box_iou([0, 0, 10, 10], [20, 0, 10, 10]) == 0.0


def test_match_uses_y_and_iou_and_is_one_to_one():
    tracks = [
        {"id": 7, "box": [100, 100, 80, 80], "last_frame": 10},
        {"id": 9, "box": [500, 100, 80, 80], "last_frame": 10},
    ]
    used = set()
    assert match_box_to_track([104, 103, 80, 80], tracks, 11, 1000, 600, used) == 7
    used.add(7)
    assert match_box_to_track([504, 103, 80, 80], tracks, 11, 1000, 600, used) == 9
    assert match_box_to_track([104, 103, 80, 80], tracks, 11, 1000, 600, {7}) is None


def test_match_does_not_resurrect_stale_track():
    tracks = [{"id": 2, "box": [100, 100, 80, 80], "last_frame": 1}]
    assert match_box_to_track([105, 105, 80, 80], tracks, 50, 1000, 600, max_age=20) is None


def test_scan_boundaries_clamp_to_three_through_seven_bands():
    assert scan_boundaries(100, 1) == [0, 33, 67, 100]
    assert len(scan_boundaries(100, 99)) == 8
    assert scan_boundaries(1, 5)[0] == 0
    assert scan_boundaries(1, 5)[-1] == 1


def test_stabilize_removes_isolated_spike_and_caps_motion():
    raw = [100, 102, 500, 104, 106]
    stable = stabilize_crop_path(
        raw, [(0, len(raw))], ["TRACK"], 1000, 500,
        window=5, max_step_ratio=0.02, outlier_ratio=0.06,
    )
    assert stable[2] != 500
    assert all(abs(b - a) <= 20 for a, b in zip(stable, stable[1:]))


def test_stabilize_keeps_scene_cut_snap_and_static_layouts_untouched():
    raw = [100, 105, 900, 905, None]
    stable = stabilize_crop_path(
        raw, [(0, 2), (2, 4), (4, 5)], ["TRACK", "TRACK", "GENERAL"],
        1000, 500, max_step_ratio=0.01,
    )
    assert stable[2] == 900  # first frame of the next scene is an intentional snap
    assert stable[4] is None
