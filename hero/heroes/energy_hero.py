"""Original energy visor with lightning seams."""

from __future__ import annotations

import cv2
import numpy as np

from hero.face_mapper import HeroFace

NAME = "energy"
FILL = (70, 40, 10)
EDGE = (40, 255, 255)
LENS = (0, 255, 255)
ACCENT = (80, 255, 255)
GLOW = (30, 255, 255)


def decorate(layer: np.ndarray, face: HeroFace) -> None:
    fh = (int(face.forehead[0]), int(face.forehead[1]))
    nose = (int(face.nose[0]), int(face.nose[1]))
    chin = (int(face.chin[0]), int(face.chin[1]))
    cv2.line(layer, fh, nose, ACCENT, 2, cv2.LINE_AA)
    cv2.line(layer, nose, chin, EDGE, 1, cv2.LINE_AA)
    if face.left_eye is not None and face.right_eye is not None:
        le = face.left_eye.mean(axis=0).astype(int)
        re = face.right_eye.mean(axis=0).astype(int)
        mid = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2 - int(face.scale * 0.08))
        cv2.line(layer, tuple(le), mid, LENS, 2, cv2.LINE_AA)
        cv2.line(layer, tuple(re), mid, LENS, 2, cv2.LINE_AA)
        cv2.line(layer, mid, fh, ACCENT, 2, cv2.LINE_AA)
    if face.oval is not None:
        for pt in face.oval[::4]:
            jitter = (int(pt[0]), int(pt[1]))
            cv2.line(layer, nose, jitter, EDGE, 1, cv2.LINE_AA)
