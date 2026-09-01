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
                    color=(20, 40, 255),
                )
            )
        self.sparks.update(dt)

    def draw(self, layer: np.ndarray, pose: AnimationPose, charge: float, state: str) -> None:
        if not pose.present:
            return
        cx, cy = int(pose.palm[0]), int(pose.palm[1])
        radius = int(pose.scale * (0.45 + 0.55 * max(charge, 0.35)))
        rot = pose.angle + self.time
        colors = ((0, 50, 255), (40, 0, 255), (0, 230, 255))
        for i, (scale, spin) in enumerate(((1.0, 1.0), (0.72, -1.3), (0.42, 1.8))):
            r = max(8, int(radius * scale))
            cv2.ellipse(
                layer,
                (cx, cy),
                (r, int(r * 0.92)),
                math.degrees(rot * spin),
                0,
                360,
                colors[i],
                4,
                cv2.LINE_AA,
            )
            for k in range(6):
                ang = rot * spin + k * (math.pi / 3)
                px = int(cx + math.cos(ang) * r)
                py = int(cy + math.sin(ang) * r)
                cv2.drawMarker(layer, (px, py), colors[i], cv2.MARKER_DIAMOND, 10, 2, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), 8, (10, 10, 40), -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), 5, (0, 220, 255), -1, cv2.LINE_AA)
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
