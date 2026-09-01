from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from animation.hand_pose import AnimationPose
from animation.particle_system import Particle, ParticleSystem


class EnergySphere:
    def __init__(self):
        self.orbit = ParticleSystem(220)
        self.time = 0.0

    def reset(self):
        self.orbit.clear()
        self.time = 0.0

    def update(self, dt: float, pose: AnimationPose, charge: float, state: str) -> None:
        self.time += dt
        if not pose.present:
            self.orbit.update(dt)
            return
        cx, cy = pose.index_tip
        intensity = 0.2 + 0.8 * charge
        if state in ("charging", "ready", "active") or pose.is_pointing:
            rng = np.random.default_rng()
            for layer, spin, count in ((18, 3.2, 2), (32, -2.4, 2), (48, 4.6, 1)):
                for _ in range(count):
                    ang = float(rng.uniform(0, 2 * math.pi))
                    self.orbit.emit(
                        Particle(
                            x=cx + math.cos(ang) * layer * intensity,
                            y=cy + math.sin(ang) * layer * intensity,
                            vx=0,
                            vy=0,
                            radius=float(rng.uniform(1.4, 3.2)),
                            life=float(rng.uniform(0.25, 0.55)),
                            max_life=0.55,
                            angle=ang,
                            orbit_radius=float(layer * intensity * 0.08),
                            spin=spin,
                            color=(255, 90, 220) if layer < 30 else (255, 40, 160),
                        )
                    )
        self.orbit.update(dt)

    def draw(self, layer: np.ndarray, pose: AnimationPose, charge: float, state: str) -> None:
        if not pose.present and state not in ("releasing",):
            self._draw_particles(layer)
            return
        cx, cy = int(pose.index_tip[0]), int(pose.index_tip[1])
        radius = int(14 + 58 * charge)
        core = (255, 210, 255)
        ring = (255, 50, 200)
        glow = (255, 120, 40)
        ink = (70, 0, 40)
        pulse = 1.0 + 0.08 * math.sin(self.time * 8)
        r = max(8, int(radius * pulse))
        cv2.circle(layer, (cx, cy), r + 6, ink, 5, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), max(5, r // 4), core, -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), r, glow, 3, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), int(r * 0.68), ring, 2, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), int(r * 0.38), core, 2, cv2.LINE_AA)
        for i, spin in enumerate((self.time * 2.6, -self.time * 1.9, self.time * 3.4)):
            for k in range(10):
                ang = spin + k * (math.pi / 5)
                dist = r * (0.42 + 0.2 * i)
                px = int(cx + math.cos(ang) * dist)
                py = int(cy + math.sin(ang) * dist)
                cv2.circle(layer, (px, py), 2 + (i == 0), ring if i else core, -1, cv2.LINE_AA)
        self._draw_particles(layer)

    def _draw_particles(self, layer: np.ndarray) -> None:
        for particle in self.orbit.particles:
            fade = max(0.0, particle.life / particle.max_life)
            color = tuple(int(c * fade) for c in particle.color)
            cv2.circle(
                layer,
                (int(particle.x), int(particle.y)),
                max(1, int(particle.radius)),
                color,
                -1,
                cv2.LINE_AA,
            )
