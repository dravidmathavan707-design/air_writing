"""Original web-themed visor (inspired by comic webbing, not a licensed costume)."""

from __future__ import annotations

import cv2
import numpy as np

from hero.face_mapper import HeroFace

NAME = "web"
FILL = (42, 18, 78)
EDGE = (50, 50, 230)
LENS = (245, 245, 255)
ACCENT = (70, 70, 255)
GLOW = (80, 80, 255)


def decorate(layer: np.ndarray, face: HeroFace) -> None:
    if face.oval is None:
        return
    oval = face.oval.astype(np.int32)
    nose = (int(face.nose[0]), int(face.nose[1]))
    for pt in oval[::2]:
        cv2.line(layer, nose, (int(pt[0]), int(pt[1])), ACCENT, 1, cv2.LINE_AA)
    cx, cy = int(face.center[0]), int(face.center[1])
    for t in (0.35, 0.55, 0.78):
        ring = (face.oval * t + np.array(face.center) * (1 - t)).astype(np.int32)
        cv2.polylines(layer, [ring.reshape(-1, 1, 2)], True, EDGE, 1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), 3, LENS, -1, cv2.LINE_AA)
