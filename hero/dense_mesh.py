"""Dense Face Mesh control points + stable UVs for Snapchat-style warps."""

from __future__ import annotations

import math

import cv2
import numpy as np

from face.face_landmarks import (
    FACE_OVAL,
    GLABELLA,
    JAWLINE,
    LEFT_EYE,
    LEFT_EYEBROW_UPPER,
    LIPS_OUTER,
    NOSE_BRIDGE,
    NOSE_WINGS,
    RIGHT_EYE,
    RIGHT_EYEBROW_UPPER,
)

TEX = 512

_CHEEKS = [
    50, 101, 118, 205, 36, 123, 147, 187, 203, 206, 207,
    280, 330, 347, 425, 266, 352, 376, 411, 423, 426, 427,
    116, 117, 119, 345, 346, 348, 212, 432,
]

_INTERIOR = {
    1: (0.00, 0.14),
    2: (0.00, 0.22),
    4: (0.00, 0.04),
    5: (0.00, -0.04),
    6: (0.00, -0.16),
    8: (0.00, -0.30),
    9: (0.00, -0.36),
    13: (0.00, 0.42),
    14: (0.00, 0.46),
    17: (0.00, 0.52),
    18: (0.00, 0.48),
    33: (-0.34, -0.08),
    133: (-0.14, -0.08),
    159: (-0.24, -0.12),
    145: (-0.24, -0.02),
    362: (0.14, -0.08),
    263: (0.34, -0.08),
    386: (0.24, -0.12),
    374: (0.24, -0.02),
    61: (-0.20, 0.36),
    291: (0.20, 0.36),
    0: (0.00, 0.32),
    70: (-0.30, -0.42),
    63: (-0.22, -0.46),
    105: (-0.12, -0.48),
    66: (-0.04, -0.46),
    107: (0.04, -0.44),
    336: (-0.04, -0.44),
    296: (0.04, -0.46),
    334: (0.12, -0.48),
    293: (0.22, -0.46),
    300: (0.30, -0.42),
    168: (0.00, -0.22),
    197: (0.00, -0.10),
    195: (0.00, -0.02),
    48: (-0.16, 0.16),
    64: (-0.12, 0.12),
    98: (-0.10, 0.20),
    327: (0.10, 0.20),
    294: (0.12, 0.12),
    278: (0.16, 0.16),
    50: (-0.46, 0.10),
    101: (-0.38, 0.00),
    118: (-0.50, -0.06),
    205: (-0.40, 0.28),
    36: (-0.42, -0.18),
    123: (-0.52, 0.08),
    147: (-0.48, 0.32),
    187: (-0.36, 0.22),
    203: (-0.28, 0.30),
    206: (-0.34, 0.18),
    207: (-0.30, 0.26),
    280: (0.46, 0.10),
    330: (0.38, 0.00),
    347: (0.50, -0.06),
    425: (0.40, 0.28),
    266: (0.42, -0.18),
    352: (0.52, 0.08),
    376: (0.48, 0.32),
    411: (0.36, 0.22),
    423: (0.28, 0.30),
    426: (0.34, 0.18),
    427: (0.30, 0.26),
    116: (-0.56, -0.02),
    117: (-0.54, -0.12),
    119: (-0.48, -0.22),
    345: (0.56, -0.02),
    346: (0.54, -0.12),
    348: (0.48, -0.22),
    212: (-0.44, 0.40),
    432: (0.44, 0.40),
    200: (0.00, 0.58),
    199: (0.00, 0.62),
    175: (0.00, 0.70),
    163: (-0.30, -0.04),
    154: (-0.18, -0.02),
    157: (-0.18, -0.12),
    161: (-0.30, -0.12),
    382: (0.18, -0.02),
    373: (0.30, -0.04),
    388: (0.30, -0.12),
    385: (0.18, -0.12),
}

MESH_IDX = tuple(
    idx
    for idx in dict.fromkeys(
        FACE_OVAL
        + JAWLINE
        + LEFT_EYEBROW_UPPER
        + RIGHT_EYEBROW_UPPER
        + LEFT_EYE[::2]
        + RIGHT_EYE[::2]
        + NOSE_BRIDGE
        + NOSE_WINGS
        + LIPS_OUTER[::2]
        + GLABELLA
        + _CHEEKS
        + [1, 2, 4, 5, 8, 13, 14, 17, 18, 200, 199, 175]
    )
    if idx in FACE_OVAL or idx in _INTERIOR
)


def _uv_points() -> np.ndarray:
    cx = cy = TEX * 0.5
    rx, ry = TEX * 0.39, TEX * 0.46
    uvs = []
    for idx in MESH_IDX:
        if idx in FACE_OVAL:
            i = FACE_OVAL.index(idx)
            ang = -math.pi / 2 + (2 * math.pi * i) / len(FACE_OVAL)
            uvs.append((cx + math.cos(ang) * rx, cy + math.sin(ang) * ry))
        elif idx in _INTERIOR:
            nx, ny = _INTERIOR[idx]
            uvs.append((cx + nx * rx, cy + ny * ry))
    pts = np.array(uvs, dtype=np.float32)
    return np.clip(pts, 4, TEX - 5)


def _triangles(uv: np.ndarray) -> tuple:
    subdiv = cv2.Subdiv2D((0, 0, TEX, TEX))
    for x, y in uv:
        subdiv.insert((float(x), float(y)))
    uniq = []
    seen = set()
    for t in subdiv.getTriangleList():
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        ids = []
        ok = True
        for x, y in pts:
            if not (1 <= x < TEX - 1 and 1 <= y < TEX - 1):
                ok = False
                break
            d = (uv - np.array([x, y], dtype=np.float32)) ** 2
            ids.append(int(np.argmin(d.sum(axis=1))))
        if not ok or len(set(ids)) < 3:
            continue
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tuple(ids))
    return tuple(uniq)


UV = _uv_points()
_ALL_TRIS = _triangles(UV)


def _keep_largest(tris, uv, keep: int = 88):
    if len(tris) <= keep:
        return tris
    areas = []
    for tri in tris:
        d = uv[list(tri)]
        areas.append(
            abs(
                (d[1, 0] - d[0, 0]) * (d[2, 1] - d[0, 1])
                - (d[2, 0] - d[0, 0]) * (d[1, 1] - d[0, 1])
            )
        )
    order = np.argsort(np.asarray(areas))[::-1][:keep]
    return tuple(tris[int(i)] for i in order)


TRIS = _keep_largest(_ALL_TRIS, UV)
OVAL_SET = set(FACE_OVAL)
FOREHEAD_SET = {10, 338, 297, 67, 109, 103, 54, 21, 162, 127}
GROW = np.array(
    [1.18 if i in FOREHEAD_SET else 1.10 if i in OVAL_SET else 1.015 for i in MESH_IDX],
    dtype=np.float32,
)
MESH_NP = np.fromiter(MESH_IDX, dtype=np.int32)
