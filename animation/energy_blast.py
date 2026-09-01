from __future__ import annotations

import math

import cv2
import numpy as np

from animation.hand_pose import AnimationPose
from animation.particle_system import ParticleSystem


class EnergyBlast:
    def __init__(self):
        self.burst = ParticleSystem(280)
        self.age = 0.0
        self.active = False
        self.origin = (0.0, 0.0)
        self.power = 1.0

    def reset(self):
        self.burst.clear()
        self.age = 0.0
        self.active = False

    def trigger(self, origin, power: float = 1.0) -> None:
        self.origin = (float(origin[0]), float(origin[1]))
        self.power = max(0.35, min(1.0, power))
        self.age = 0.0
        self.active = True
        self.burst.clear()
        self.burst.emit_burst(
            self.origin[0],
            self.origin[1],
            count=int(40 + 50 * self.power),
            speed=220 * self.power,
            life=0.85,
            color=(10, 200, 255),
        )
        self.burst.emit_burst(
            self.origin[0],
            self.origin[1],
            count=18,
            speed=140 * self.power,
            life=0.6,
            color=(20, 40, 255),
        )

    def update(self, dt: float, pose: AnimationPose, charge: float, state: str) -> None:
        if not self.active:
            return
        self.age += dt
        self.burst.update(dt)
        if self.age > 1.05:
            self.active = False
            self.burst.clear()

    def draw(self, layer: np.ndarray, pose: AnimationPose, charge: float, state: str) -> None:
        if not self.active:
            return
        ox, oy = int(self.origin[0]), int(self.origin[1])
        fade = max(0.0, 1.0 - self.age / 1.05)
        shock = int((40 + 220 * self.age) * self.power)
        cv2.circle(layer, (ox, oy), shock + 4, (10, 20, 60), 5, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), shock, (0, int(210 * fade), 255), 4, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), int(shock * 0.55), (20, 40, int(255 * fade)), 2, cv2.LINE_AA)
        for k in range(8):
            ang = k * math.pi / 4
            length = int((50 + 180 * self.age) * self.power)
            end = (int(ox + math.cos(ang) * length), int(oy + math.sin(ang) * length))
            cv2.line(layer, (ox, oy), end, (10, 20, 50), 4, cv2.LINE_AA)
            cv2.line(layer, (ox, oy), end, (0, int(230 * fade), 255), 2, cv2.LINE_AA)
        for particle in self.burst.particles:
            life = max(0.0, particle.life / particle.max_life)
            cv2.circle(
                layer,
                (int(particle.x), int(particle.y)),
                max(1, int(particle.radius)),
                tuple(int(c * life) for c in particle.color),
                -1,
                cv2.LINE_AA,
            )
