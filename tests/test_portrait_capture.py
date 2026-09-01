import numpy as np

from face.face_detector import select_best_face_frame
from face.face_landmarks import (
    assess_face_quality,
    average_landmark_sets,
    groups_from_points,
    hands_overlap_face,
    landmark_spread,
)
from face.face_renderer import draw_face_portrait_from_points


def _frontal_face(width=640, height=480, scale=1.0, shift=(0, 0)):
    pts = np.zeros((478, 2), dtype=np.float32)
    cx, cy = width * 0.5 + shift[0], height * 0.45 + shift[1]
    rx, ry = 90 * scale, 120 * scale

    oval = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]
    for i, idx in enumerate(oval):
        angle = -np.pi / 2 + (2 * np.pi * i / len(oval))
        pts[idx] = [cx + rx * np.cos(angle), cy + ry * np.sin(angle)]

    pts[33] = [cx - 35 * scale, cy - 15 * scale]
    pts[133] = [cx - 15 * scale, cy - 15 * scale]
    pts[362] = [cx + 15 * scale, cy - 15 * scale]
    pts[263] = [cx + 35 * scale, cy - 15 * scale]
    pts[1] = [cx, cy + 20 * scale]
    pts[61] = [cx - 22 * scale, cy + 45 * scale]
    pts[291] = [cx + 22 * scale, cy + 45 * scale]
    pts[10] = [cx, cy - ry]
    pts[152] = [cx, cy + ry]
    pts[234] = [cx - rx, cy]
    pts[454] = [cx + rx, cy]
    return pts


def test_select_best_face_frame_prefers_stable_high_confidence_sample():
    samples = [
        {"confidence": 0.62, "landmark_variance": 9.5},
        {"confidence": 0.93, "landmark_variance": 2.1},
        {"confidence": 0.88, "landmark_variance": 3.9},
    ]
    best = select_best_face_frame(samples)
    assert best == samples[1]


def test_select_best_face_frame_returns_none_when_no_samples():
    assert select_best_face_frame([]) is None


def test_average_landmark_sets_uses_real_geometry():
    a = _frontal_face(shift=(0, 0))
    b = _frontal_face(shift=(10, 4))
    averaged = average_landmark_sets([a, b])
    assert averaged is not None
    np.testing.assert_allclose(averaged[1], (a[1] + b[1]) / 2, atol=0.01)


def test_assess_face_quality_rejects_points_outside_frame():
    pts = _frontal_face()
    pts[33] = [-20, 40]
    assert assess_face_quality(pts, (480, 640, 3)) == "partially_occluded"


def test_assess_face_quality_accepts_frontal_face():
    pts = _frontal_face()
    assert assess_face_quality(pts, (480, 640, 3)) == "ok"


def test_groups_from_points_use_mesh_indices():
    pts = _frontal_face()
    data = groups_from_points(pts)
    assert len(data["face_oval"]) == 36
    assert len(data["left_eye"]) >= 4
    assert data["bbox"] is not None


def test_two_faces_draw_on_separate_sides():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    left = _frontal_face(shift=(-140, 0))
    right = _frontal_face(shift=(140, 0))
    draw_face_portrait_from_points(canvas, left, thickness=2)
    draw_face_portrait_from_points(canvas, right, thickness=2)
    assert canvas[:, :300].sum() > 0
    assert canvas[:, 340:].sum() > 0


def test_hands_overlap_face_detects_palm_on_bbox():
    class FakeHand:
        def __init__(self, palmx, palmy):
            self.landmarks = [(10, 10)] * 21
            self.landmarks[0] = (palmx, palmy)
            self.landmarks[9] = (palmx, palmy)

    bbox = (280, 160, 360, 260)
    assert hands_overlap_face([FakeHand(320, 200)], bbox) is True
    assert hands_overlap_face([FakeHand(10, 10)], bbox) is False


def test_landmark_spread_is_finite_for_face():
    spread = landmark_spread(_frontal_face())
    assert spread > 0
    assert spread < 500
