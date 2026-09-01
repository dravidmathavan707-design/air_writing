"""Original wrap-style visor with a forehead band and eye slits."""

from __future__ import annotations

import cv2
import numpy as np

from hero.face_mapper import HeroFace

NAME = "ninja"
FILL = (28, 28, 28)
EDGE = (90, 90, 90)
LENS = (40, 220, 255)
ACCENT = (200, 200, 200)
GLOW = (180, 180, 180)


def decorate(layer: np.ndarray, face: HeroFace) -> None:
    if face.left_eye is None or face.right_eye is None:
        return
    ly = int(min(face.left_eye[:, 1].min(), face.right_eye[:, 1].min()) - 10)
    hy = ly + 14
    x0 = int(face.bbox[0])
    x1 = int(face.bbox[0] + face.bbox[2])
    cv2.rectangle(layer, (x0, ly), (x1, hy), ACCENT, -1)
    cv2.rectangle(layer, (x0, ly), (x1, hy), EDGE, 1)
    knot = (x1 + 8, (ly + hy) // 2)
    cv2.line(layer, (x1, ly + 4), knot, ACCENT, 2, cv2.LINE_AA)
    cv2.line(layer, (x1, hy - 4), knot, ACCENT, 2, cv2.LINE_AA)
