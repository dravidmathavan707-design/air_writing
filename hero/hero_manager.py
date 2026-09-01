"""Live AR hero visor: landmarks → pose → mask + scan/charge VFX."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import cv2
import numpy as np

from animation.glow import apply_glow
from hero.face_mapper import FaceMapperPool
from hero.hero_effects import HeroEffects
from hero.heroes import HERO_ORDER, STYLES
from hero.mask_renderer import render_mask
from hero.transformation import HERO, IDLE, SCAN, Transformation


class HeroManager:
    def __init__(self):
        self.style = "iron"
        self.pool = FaceMapperPool()
        self.xform = Transformation()
        self.fx = HeroEffects()
        self._last = time.perf_counter()
        self.face_count = 0

    def set_hero(self, name: str) -> None:
        if name not in STYLES:
            return
        self.style = name
        self.pool.reset()
        self.xform.begin()
        self.fx.reset()

    def reset(self) -> None:
        self.pool.reset()
        self.xform.reset()
        self.fx.reset()
        self.face_count = 0
        self._last = time.perf_counter()

    def status_text(self, has_face: bool, count: int | None = None) -> str:
        label = self.style.upper()
        n = self.face_count if count is None else count
        if not has_face or n <= 0:
            return f"{label} - SHOW YOUR FACE"
        who = "1 FACE" if n == 1 else f"{n} FACES"
        stage = self.xform.stage
        if stage == SCAN:
            return f"{label} - SCANNING {who}"
        if stage == "charge":
            return f"{label} - ENERGY BUILDUP ({who})"
        if stage == "assemble":
            return f"{label} - MASK FORMING ({who})"
        if stage == HERO:
            return f"{label} LOCKED ON {who}"
        return f"{label} - {who} READY"

    def render(self, frame: np.ndarray, faces: Optional[Sequence] = None) -> np.ndarray:
        now = time.perf_counter()
        dt = min(0.05, max(0.001, now - self._last))
        self._last = now

        mapped = self.pool.map_all(_all_points(faces))
        self.face_count = len(mapped)
        self.xform.update(dt, self.face_count > 0)
        style = STYLES[self.style]
        self.fx.update(dt, mapped, self.xform.stage, style.GLOW)

        if getattr(style, "SNAP_BLEND", False) and hasattr(style, "composite"):
            composed = frame
            for hero_face in mapped:
                composed = style.composite(composed, hero_face, self.xform.mask_alpha)
            fx_layer = np.zeros_like(composed)
            self.fx.draw(fx_layer, mapped, self.xform.stage, self.xform.scan_t, style.GLOW)
            if self.xform.stage == IDLE and not mapped:
                return composed
            return apply_glow(composed, fx_layer, sigma=5.0, glow_strength=0.38, core_strength=0.85)

        dimmed = cv2.convertScaleAbs(frame, alpha=0.78, beta=-8)
        layer = np.zeros_like(dimmed)
        for hero_face in mapped:
            render_mask(layer, hero_face, self.style, self.xform.mask_alpha, self.xform.stage)
        self.fx.draw(layer, mapped, self.xform.stage, self.xform.scan_t, style.GLOW)
        if self.xform.stage == IDLE and not mapped:
            return dimmed
        return apply_glow(dimmed, layer, sigma=8.0, glow_strength=0.72, core_strength=1.05)


def _all_points(faces: Optional[Sequence]):
    out = []
    if not faces:
        return out
    for face in faces:
        pts = face.get("points") if isinstance(face, dict) else None
        if pts is not None and len(pts) >= 400:
            out.append(pts)
    return out
