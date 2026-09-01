from __future__ import annotations

import cv2
import numpy as np


def apply_glow(
    frame: np.ndarray,
    effect_layer: np.ndarray,
    sigma: float = 10.0,
    glow_strength: float = 0.82,
    core_strength: float = 1.15,
) -> np.ndarray:
    """Composite neon VFX onto the camera frame with a dark pocket so it reads on white walls."""
    if effect_layer is None or effect_layer.size == 0:
        return frame
    gray = cv2.cvtColor(effect_layer, cv2.COLOR_BGR2GRAY)
    presence = cv2.GaussianBlur(gray, (0, 0), max(1.0, float(sigma) * 0.7))
    weight = np.clip(presence.astype(np.float32) / 130.0, 0.0, 1.0)[..., None]
    base = frame.astype(np.float32) * (1.0 - 0.58 * weight)
    blurred = cv2.GaussianBlur(effect_layer, (0, 0), max(1.0, float(sigma))).astype(np.float32)
    out = base + blurred * glow_strength + effect_layer.astype(np.float32) * core_strength
    return np.clip(out, 0, 255).astype(np.uint8)
