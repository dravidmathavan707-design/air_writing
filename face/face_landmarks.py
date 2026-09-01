"""
Semantic Face Mesh groups and portrait-capture helpers.

Landmark indices follow MediaPipe Face Mesh / Face Landmarker
(468 points, or 478 with iris refinement).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Face oval in contour order (forehead → right → chin → left)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

LEFT_EYEBROW_UPPER = [70, 63, 105, 66, 107]
LEFT_EYEBROW_LOWER = [46, 53, 52, 65, 55]
RIGHT_EYEBROW_UPPER = [300, 293, 334, 296, 336]
RIGHT_EYEBROW_LOWER = [276, 283, 282, 295, 285]
LEFT_EYEBROW = LEFT_EYEBROW_UPPER + list(reversed(LEFT_EYEBROW_LOWER))
RIGHT_EYEBROW = RIGHT_EYEBROW_UPPER + list(reversed(RIGHT_EYEBROW_LOWER))

NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]
NOSE_TIP = [1]
NOSTRILS = [98, 97, 2, 326, 327]
NOSE_WINGS = [48, 64, 98, 327, 294, 278]

LIPS_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]
LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LEFT_UPPER_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133]
LEFT_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_UPPER_LID = [362, 398, 384, 385, 386, 387, 388, 466, 263]
RIGHT_LOWER_LID = [362, 382, 381, 380, 374, 373, 390, 249, 263]

LEFT_IRIS_RING = [468, 469, 470, 471, 472]
RIGHT_IRIS_RING = [473, 474, 475, 476, 477]

LIPS_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

NOSE_LEFT_ALA = [48, 64, 98, 97, 2]
NOSE_RIGHT_ALA = [278, 294, 327, 326, 2]
JAWLINE = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
]
LEFT_EAR = [234, 127, 162, 21]
RIGHT_EAR = [454, 356, 389, 251]
PHILTRUM = [164, 167, 393, 0]
GLABELLA = [9, 8, 168]


NOSE_TIP_IDX = 1
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454
FOREHEAD = 10
CHIN = 152
LEFT_IRIS = 468
RIGHT_IRIS = 473

KEY_LANDMARKS = [
    33, 133, 362, 263, 1, 61, 291, 10, 152, 234, 454,
]

GROUP_INDICES: Dict[str, List[int]] = {
    "face_oval": FACE_OVAL,
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "left_eyebrow": LEFT_EYEBROW,
    "right_eyebrow": RIGHT_EYEBROW,
    "left_eyebrow_upper": LEFT_EYEBROW_UPPER,
    "right_eyebrow_upper": RIGHT_EYEBROW_UPPER,
    "nose_bridge": NOSE_BRIDGE,
    "nose_tip": NOSE_TIP,
    "nostrils": NOSTRILS,
    "nose_wings": NOSE_WINGS,
    "lips_outer": LIPS_OUTER,
    "lips_inner": LIPS_INNER,
    "lips_upper": LIPS_UPPER,
    "lips_lower": LIPS_LOWER,
    "left_upper_lid": LEFT_UPPER_LID,
    "left_lower_lid": LEFT_LOWER_LID,
    "right_upper_lid": RIGHT_UPPER_LID,
    "right_lower_lid": RIGHT_LOWER_LID,
    "left_iris": LEFT_IRIS_RING,
    "right_iris": RIGHT_IRIS_RING,
    "nose_left_ala": NOSE_LEFT_ALA,
    "nose_right_ala": NOSE_RIGHT_ALA,
    "jawline": JAWLINE,
    "left_ear": LEFT_EAR,
    "right_ear": RIGHT_EAR,
    "philtrum": PHILTRUM,
    "glabella": GLABELLA,
}


def _as_array(points: Sequence) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)


def get_group(points: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    valid = [i for i in indices if 0 <= i < len(points)]
    if not valid:
        return np.zeros((0, 2), dtype=np.float32)
    return _as_array(points)[valid]


def groups_from_points(points: Sequence) -> Dict:
    pts = _as_array(points)
    data = {name: get_group(pts, idxs) for name, idxs in GROUP_INDICES.items()}
    data["all_landmarks"] = pts
    data["confidence"] = 1.0

    if len(data["face_oval"]) > 0:
        x_min, y_min = data["face_oval"].min(axis=0)
        x_max, y_max = data["face_oval"].max(axis=0)
        data["bbox"] = (int(x_min), int(y_min), int(x_max), int(y_max))
    else:
        data["bbox"] = bbox_from_points(pts)

    return data


def bbox_from_points(
    points: Sequence,
    padding: float = 0.0,
) -> Optional[Tuple[int, int, int, int]]:
    pts = _as_array(points)
    if len(pts) < 4:
        return None
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    if padding:
        w = x_max - x_min
        h = y_max - y_min
        x_min -= w * padding
        x_max += w * padding
        y_min -= h * padding
        y_max += h * padding
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def average_landmark_sets(samples: Sequence[Sequence]) -> Optional[np.ndarray]:
    arrays = []
    for sample in samples:
        pts = _as_array(sample)
        if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 468:
            continue
        arrays.append(pts[:, :2])
    if not arrays:
        return None
    min_len = min(len(arr) for arr in arrays)
    stacked = np.stack([arr[:min_len] for arr in arrays], axis=0)
    if len(stacked) >= 4:
        median = np.median(stacked, axis=0)
        distances = np.linalg.norm(stacked.reshape(len(stacked), -1) - median.reshape(1, -1), axis=1)
        keep = distances <= np.quantile(distances, 0.75)
        if keep.any():
            stacked = stacked[keep]
    return stacked.mean(axis=0).astype(np.float32)


def hands_overlap_face(hands: Sequence, bbox: Optional[Tuple[int, int, int, int]]) -> bool:
    """True when a palm or wrist sits on the face (gesture covering the mesh)."""
    if bbox is None or not hands:
        return False
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * 0.08
    pad_y = (y2 - y1) * 0.08
    x1, y1, x2, y2 = x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y
    for hand in hands:
        landmarks = getattr(hand, "landmarks", None) or hand
        if landmarks is None or len(landmarks) < 10:
            continue
        for idx in (0, 9):
            x, y = float(landmarks[idx][0]), float(landmarks[idx][1])
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
    return False


def landmark_spread(points: Sequence) -> float:
    pts = _as_array(points)
    if len(pts) < 2:
        return 999.0
    center = pts.mean(axis=0)
    return float(np.sqrt(((pts - center) ** 2).sum(axis=1).mean()))


def _visibility_of(visibilities: Optional[Sequence[float]], index: int) -> float:
    if visibilities is None or index >= len(visibilities):
        return 1.0
    value = visibilities[index]
    if value is None:
        return 1.0
    return float(value)


def assess_face_quality(
    points: Sequence,
    frame_shape: Tuple[int, int, int] | Tuple[int, int],
    visibilities: Optional[Sequence[float]] = None,
    margin: int = 8,
    min_visibility: float = 0.45,
) -> str:
    """
    Returns one of: "ok", "not_detected", "partially_occluded".
    """
    pts = _as_array(points)
    if len(pts) < 468:
        return "not_detected"

    height, width = frame_shape[:2]
    core = (33, 133, 362, 263, 1)
    for idx in core:
        if idx >= len(pts):
            return "partially_occluded"
        x, y = float(pts[idx][0]), float(pts[idx][1])
        if x < margin or y < margin or x >= width - margin or y >= height - margin:
            return "partially_occluded"
        if _visibility_of(visibilities, idx) < min_visibility:
            return "partially_occluded"

    left_eye = pts[33]
    right_eye = pts[263]
    left_edge = pts[LEFT_FACE_EDGE]
    right_edge = pts[RIGHT_FACE_EDGE]
    forehead = pts[FOREHEAD]
    chin = pts[CHIN]

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    face_width = float(np.linalg.norm(right_edge - left_edge))
    face_height = float(np.linalg.norm(chin - forehead))

    if face_width < 40 or face_height < 50:
        return "partially_occluded"
    if eye_dist < face_width * 0.18:
        return "partially_occluded"
    if face_width > face_height * 1.8:
        return "partially_occluded"

    return "ok"


def face_roll_degrees(points: Sequence) -> float:
    pts = _as_array(points)
    if len(pts) <= RIGHT_FACE_EDGE:
        return 0.0
    left = pts[LEFT_FACE_EDGE]
    right = pts[RIGHT_FACE_EDGE]
    delta = right - left
    return float(np.degrees(np.arctan2(delta[1], delta[0])))
