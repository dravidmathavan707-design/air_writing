from __future__ import annotations

import math

import cv2
import numpy as np

from animation.hand_pose import AnimationPose
from animation.particle_system import Particle, ParticleSystem


class MagicPortal:
    def __init__(self):
        self.sparks = ParticleSystem(160)
        self.time = 0.0

    def reset(self):
        self.sparks.clear()
        self.time = 0.0

    def update(self, dt: float, pose: AnimationPose, charge: float, state: str) -> None:
        self.time += dt
        if not pose.present:
            self.sparks.update(dt)
            return
        cx, cy = pose.palm
        rng = np.random.default_rng()
        radius = pose.scale * (0.45 + 0.55 * max(charge, 0.35))
        for _ in range(2):
            ang = float(rng.uniform(0, 2 * math.pi))
            self.sparks.emit(
                Particle(
                    x=cx + math.cos(ang) * radius,
                    y=cy + math.sin(ang) * radius,
                    radius=float(rng.uniform(1.4, 3.0)),
                    life=0.45,
                    max_life=0.45,
                    angle=ang,
                    orbit_radius=4.0,
                    spin=2.8,
                    color=(80, 255, 90) if rng.random() > 0.45 else (255, 220, 40),
                )
            )
        self.sparks.update(dt)

    def draw(self, layer: np.ndarray, pose: AnimationPose, charge: float, state: str) -> None:
        if not pose.present:
            return
        cx, cy = int(pose.palm[0]), int(pose.palm[1])
        radius = int(pose.scale * (0.45 + 0.55 * max(charge, 0.35)))
        rot = pose.angle + self.time
        colors = ((40, 255, 70), (255, 200, 40), (255, 80, 200))
        for i, (scale, spin) in enumerate(((1.05, 1.0), (0.78, -1.35), (0.52, 1.9), (0.28, -2.2))):
            r = max(8, int(radius * scale))
            color = colors[i % 3]
            cv2.ellipse(
                layer,
                (cx, cy),
                (r, int(r * 0.88)),
                math.degrees(rot * spin),
                0,
                360,
                (8, 40, 10),
                6,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                layer,
                (cx, cy),
                (r, int(r * 0.88)),
                math.degrees(rot * spin),
                0,
                360,
                color,
                3,
                cv2.LINE_AA,
            )
            for k in range(8):
                ang = rot * spin + k * (math.pi / 4)
                px = int(cx + math.cos(ang) * r)
                py = int(cy + math.sin(ang) * r)
                cv2.drawMarker(layer, (px, py), color, cv2.MARKER_STAR, 12, 2, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), 10, (10, 40, 8), -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), 6, (80, 255, 120), -1, cv2.LINE_AA)
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
