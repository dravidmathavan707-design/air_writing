"""
Face drawing from official MediaPipe Face Mesh connections.

Each feature is traced between real landmark pairs (the same graph
MediaPipe uses), so eye spacing, jaw width, nose, and lips follow
the person instead of a cartoon template.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
)

from face.hair_detector import draw_hair_contour

Color = Tuple[int, int, int]
BROW_IDXS = (70, 63, 105, 66, 107, 300, 293, 334, 296, 336)


def _xy(points: np.ndarray, index: int) -> Optional[Tuple[int, int]]:
    if index >= len(points):
        return None
    return (int(round(float(points[index][0]))), int(round(float(points[index][1]))))


def _ordered_chain(connections: Iterable[Tuple[int, int]]) -> List[int]:
    adj: Dict[int, List[int]] = {}
    for a, b in connections:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if not adj:
        return []
    start = min(adj, key=lambda node: len(adj[node]))
    chain = [start]
    prev = None
    for _ in range(len(adj) + 2):
        neighbors = [n for n in adj[chain[-1]] if n != prev]
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt == start:
            break
        prev = chain[-1]
        chain.append(nxt)
    return chain


def _draw_connections(
    canvas: np.ndarray,
    points: np.ndarray,
    connections: Set[Tuple[int, int]],
    color: Color,
    thickness: int,
) -> None:
    for a, b in connections:
        pa = _xy(points, a)
        pb = _xy(points, b)
        if pa is None or pb is None:
            continue
        cv2.line(canvas, pa, pb, color, max(1, thickness), cv2.LINE_AA)


def _draw_smooth_chain(
    canvas: np.ndarray,
    points: np.ndarray,
    connections: Set[Tuple[int, int]],
    color: Color,
    thickness: int,
    closed: bool = False,
) -> None:
    chain = _ordered_chain(connections)
    pts = []
    for idx in chain:
        xy = _xy(points, idx)
        if xy is not None:
            pts.append(xy)
    if len(pts) < 2:
        _draw_connections(canvas, points, connections, color, thickness)
        return
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        canvas,
        [arr],
        isClosed=closed,
        color=color,
        thickness=max(1, thickness),
        lineType=cv2.LINE_AA,
    )


def clip_hair_to_crown(
    contour: Optional[np.ndarray],
    landmarks: np.ndarray,
) -> Optional[np.ndarray]:
    """Keep hair outline above the eyebrows so beard is not drawn as hair."""
    if contour is None or len(contour) < 3 or landmarks is None or len(landmarks) < 107:
        return None
    brow_y = min(float(landmarks[i][1]) for i in BROW_IDXS if i < len(landmarks))
    cutoff = brow_y + 6.0
    pts = contour.reshape(-1, 2).astype(np.float32)
    keep = pts[pts[:, 1] <= cutoff]
    if len(keep) < 6:
        return None
    return keep.reshape(-1, 1, 2).astype(np.int32)


def _draw_iris(
    canvas: np.ndarray,
    points: np.ndarray,
    connections: Set[Tuple[int, int]],
    center_idx: int,
    color: Color,
) -> None:
    _draw_connections(canvas, points, connections, color, 1)
    center = _xy(points, center_idx)
    if center is None:
        return
    xs, ys = [], []
    for a, b in connections:
        for idx in (a, b):
            xy = _xy(points, idx)
            if xy is not None:
                xs.append(xy[0])
                ys.append(xy[1])
    if not xs:
        cv2.circle(canvas, center, 2, color, -1, cv2.LINE_AA)
        return
    radius = max(2, int(0.35 * max(max(xs) - min(xs), max(ys) - min(ys))))
    cv2.circle(canvas, center, radius, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, center, max(1, radius // 2), color, -1, cv2.LINE_AA)


def draw_face_portrait(
    canvas: np.ndarray,
    face_data: Optional[Dict],
    color: Color = (0, 255, 255),
    thickness: int = 2,
    hair_contour: Optional[np.ndarray] = None,
) -> None:
    if face_data is None:
        return
    points = face_data.get("all_landmarks")
    if points is None or len(points) < 468:
        return
    points = np.asarray(points, dtype=np.float32)

    hair = clip_hair_to_crown(hair_contour, points)
    if hair is not None:
        draw_hair_contour(canvas, hair, color=color, thickness=max(2, thickness + 1))

    oval_thick = max(2, thickness + 1)
    _draw_smooth_chain(canvas, points, FACEMESH_FACE_OVAL, color, oval_thick, closed=True)
    _draw_connections(canvas, points, FACEMESH_LEFT_EYEBROW, color, thickness)
    _draw_connections(canvas, points, FACEMESH_RIGHT_EYEBROW, color, thickness)
    _draw_connections(canvas, points, FACEMESH_LEFT_EYE, color, thickness)
    _draw_connections(canvas, points, FACEMESH_RIGHT_EYE, color, thickness)
    _draw_connections(canvas, points, FACEMESH_NOSE, color, max(1, thickness - 1))
    _draw_connections(canvas, points, FACEMESH_LIPS, color, thickness)

    if len(points) >= 478:
        left_center = _xy(points, 473)
        right_center = _xy(points, 468)
        left_eye = _xy(points, 362)
        right_eye = _xy(points, 33)
        if left_center and left_eye and abs(left_center[0] - left_eye[0]) < 80:
            _draw_iris(canvas, points, FACEMESH_LEFT_IRIS, 473, color)
        if right_center and right_eye and abs(right_center[0] - right_eye[0]) < 80:
            _draw_iris(canvas, points, FACEMESH_RIGHT_IRIS, 468, color)


def draw_face_portrait_from_points(
    canvas: np.ndarray,
    points: Sequence,
    color: Color = (0, 255, 255),
    thickness: int = 2,
    hair_contour: Optional[np.ndarray] = None,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    draw_face_portrait(
        canvas,
        {"all_landmarks": pts},
        color=color,
        thickness=thickness,
        hair_contour=hair_contour,
    )
