"""Fit an original hero visor to Face Mesh landmarks (not a static PNG)."""

from __future__ import annotations

import cv2
import numpy as np

from hero.face_mapper import HeroFace
from hero.heroes import STYLES
from hero.transformation import ASSEMBLE


def _poly(points: np.ndarray) -> np.ndarray:
    return points.astype(np.int32).reshape(-1, 1, 2)


def render_mask(layer: np.ndarray, face: HeroFace, style_name: str, alpha: float, stage: str) -> None:
    if not face.present or alpha <= 0.01 or face.oval is None:
        return
    style = STYLES.get(style_name, STYLES["tech"])
    fill = tuple(int(c * alpha) for c in style.FILL)
    edge = tuple(int(c * min(1.0, alpha + 0.15)) for c in style.EDGE)
    lens = tuple(int(c * alpha) for c in style.LENS)

    overlay = np.zeros_like(layer)
    if getattr(style, "SHAPE_MASK", False):
        style.decorate(overlay, face, alpha)
    else:
        cv2.fillPoly(overlay, [_poly(face.oval)], fill)
        cv2.polylines(overlay, [_poly(face.oval)], True, edge, 2, cv2.LINE_AA)
        style.decorate(overlay, face)

        if face.left_eye is not None:
            cv2.fillPoly(overlay, [_poly(face.left_eye)], (0, 0, 0))
            cv2.polylines(overlay, [_poly(face.left_eye)], True, lens, 2, cv2.LINE_AA)
        if face.right_eye is not None:
            cv2.fillPoly(overlay, [_poly(face.right_eye)], (0, 0, 0))
            cv2.polylines(overlay, [_poly(face.right_eye)], True, lens, 2, cv2.LINE_AA)

        if face.lips is not None and len(face.lips) >= 6:
            lip_h = float(face.lips[:, 1].max() - face.lips[:, 1].min())
            if lip_h > face.scale * 0.11:
                cv2.fillPoly(overlay, [_poly(face.lips)], (0, 0, 0))
                cv2.polylines(overlay, [_poly(face.lips)], True, edge, 1, cv2.LINE_AA)

    if stage == ASSEMBLE:
        t = max(0.05, min(1.0, alpha))
        clip = np.zeros(overlay.shape[:2], dtype=np.uint8)
        radius = int(face.scale * (0.25 + 1.7 * t))
        cv2.circle(clip, (int(face.center[0]), int(face.center[1])), radius, 255, -1)
        overlay[clip == 0] = 0

    cv2.add(layer, overlay, dst=layer)
