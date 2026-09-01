"""Original cosmic visor with star-field seams."""

from __future__ import annotations

import cv2
import numpy as np

from hero.face_mapper import HeroFace

NAME = "cosmic"
FILL = (70, 18, 48)
EDGE = (255, 120, 220)
LENS = (255, 200, 80)
ACCENT = (255, 160, 90)
GLOW = (255, 90, 200)


def decorate(layer: np.ndarray, face: HeroFace) -> None:
    if face.oval is None:
        return
    rng = np.random.default_rng(7)
    x0, y0, w, h = face.bbox
    for _ in range(18):
        px = int(rng.uniform(x0, x0 + max(w, 1)))
        py = int(rng.uniform(y0, y0 + max(h, 1)))
        cv2.circle(layer, (px, py), 1, LENS, -1, cv2.LINE_AA)
    cx, cy = int(face.center[0]), int(face.center[1] - face.scale * 0.35)
    axes = (int(face.scale * 0.28), int(face.scale * 0.12))
    cv2.ellipse(layer, (cx, cy), axes, np.degrees(face.angle), 200, 340, ACCENT, 2, cv2.LINE_AA)
    cv2.circle(layer, (int(face.forehead[0]), int(face.forehead[1])), 5, EDGE, 2, cv2.LINE_AA)
