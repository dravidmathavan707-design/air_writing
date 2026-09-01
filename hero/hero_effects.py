"""Particles and scan-line VFX for hero transformation."""

from __future__ import annotations

import math

import cv2
import numpy as np

from animation.particle_system import Particle, ParticleSystem
from hero.face_mapper import HeroFace
from hero.transformation import HERO, SCAN


class HeroEffects:
    def __init__(self):
        self.sparks = ParticleSystem(220)

    def reset(self) -> None:
        self.sparks.clear()

    def update(self, dt: float, faces, stage: str, color) -> None:
        if isinstance(faces, HeroFace):
            faces = [faces] if faces.present else []
        faces = [f for f in (faces or []) if getattr(f, "present", False)]
        if not faces:
            self.sparks.update(dt)
            return
        rng = np.random.default_rng()
        per = max(1, 4 // max(len(faces), 1))
        if stage == HERO:
            per = 1
        for face in faces:
            cx, cy = face.center
            for _ in range(per):
                ang = float(rng.uniform(0, 2 * math.pi))
                rad = float(rng.uniform(face.scale * 0.2, face.scale * 1.1))
                self.sparks.emit(
                    Particle(
                        x=cx + math.cos(ang) * rad,
                        y=cy + math.sin(ang) * rad,
                        vx=math.cos(ang) * 40,
                        vy=math.sin(ang) * 40,
                        radius=float(rng.uniform(1.2, 3.0)),
                        life=float(rng.uniform(0.12, 0.4)),
                        max_life=0.4,
                        color=color,
                    )
                )
        self.sparks.update(dt)

    def draw(self, layer: np.ndarray, faces, stage: str, scan_t: float, color) -> None:
        if isinstance(faces, HeroFace):
            faces = [faces] if faces.present else []
        for face in faces or []:
            if not getattr(face, "present", False):
                continue
            if stage == SCAN:
                x, y, w, h = face.bbox
                line_y = int(y + h * scan_t)
                cv2.line(layer, (x, line_y), (x + w, line_y), color, 2, cv2.LINE_AA)
                cv2.rectangle(layer, (x, y), (x + w, y + h), color, 1)
        for particle in self.sparks.particles:
            fade = max(0.0, particle.life / particle.max_life)
            cv2.circle(
                layer,
                (int(particle.x), int(particle.y)),
                max(1, int(particle.radius)),
                tuple(int(c * fade) for c in particle.color),
                -1,
                cv2.LINE_AA,
            )
