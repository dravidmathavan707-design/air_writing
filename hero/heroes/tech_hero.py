"""Original armored visor with circuit plates (not a licensed Iron Man helmet)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from hero.face_mapper import HeroFace

NAME = "tech"
FILL = (48, 38, 28)
EDGE = (40, 210, 255)
LENS = (20, 180, 255)
ACCENT = (30, 160, 255)
GLOW = (20, 200, 255)


def decorate(layer: np.ndarray, face: HeroFace) -> None:
    if face.left_eye is None or face.right_eye is None:
        return
    lx = int(face.left_eye[:, 0].mean())
    ly = int(face.left_eye[:, 1].mean())
    rx = int(face.right_eye[:, 0].mean())
    ry = int(face.right_eye[:, 1].mean())
    cv2.line(layer, (lx, ly), (rx, ry), EDGE, 3, cv2.LINE_AA)
    brow_y = int(min(ly, ry) - face.scale * 0.18)
    cv2.line(layer, (lx - 8, brow_y), (rx + 8, brow_y), ACCENT, 2, cv2.LINE_AA)
    jaw = (int(face.chin[0]), int(face.chin[1] - 6))
    cv2.line(layer, (lx, ly + 10), jaw, EDGE, 1, cv2.LINE_AA)
    cv2.line(layer, (rx, ry + 10), jaw, EDGE, 1, cv2.LINE_AA)
    for i in range(6):
        ang = face.angle + i * math.pi / 3
        px = int(face.center[0] + math.cos(ang) * face.scale * 0.42)
        py = int(face.center[1] + math.sin(ang) * face.scale * 0.42)
        cv2.circle(layer, (px, py), 4, ACCENT, 1, cv2.LINE_AA)
