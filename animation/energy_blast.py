from __future__ import annotations

import math

import cv2
import numpy as np

from animation.hand_pose import AnimationPose
from animation.particle_system import ParticleSystem

# Distinct full-power releases. Chidori is handled separately.
STYLES = {
    "nova": {
        "ink": (60, 0, 40),
        "a": (255, 80, 220),
        "b": (255, 40, 160),
        "c": (255, 210, 255),
    },
    "rift": {
        "ink": (8, 40, 10),
        "a": (40, 255, 80),
        "b": (255, 210, 40),
        "c": (120, 255, 180),
    },
    "cut": {
        "ink": (8, 30, 10),
        "a": (40, 255, 90),
        "b": (40, 230, 255),
        "c": (0, 180, 255),
    },
    "fire": {
        "ink": (0, 20, 40),
        "a": (0, 160, 255),
        "b": (0, 255, 255),
        "c": (0, 40, 255),
    },
}


class EnergyBlast:
    def __init__(self):
        self.burst = ParticleSystem(280)
        self.age = 0.0
        self.active = False
        self.origin = (0.0, 0.0)
        self.power = 1.0
        self.style = "fire"

    def reset(self):
        self.burst.clear()
        self.age = 0.0
        self.active = False
        self.style = "fire"

    def trigger(self, origin, power: float = 1.0, style: str = "fire") -> None:
        self.origin = (float(origin[0]), float(origin[1]))
        self.power = max(0.35, min(1.0, power))
        self.age = 0.0
        self.active = True
        self.style = style if style in STYLES else "fire"
        palette = STYLES[self.style]
        self.burst.clear()
        self.burst.emit_burst(
            self.origin[0],
            self.origin[1],
            count=int(48 + 60 * self.power),
            speed=260 * self.power,
            life=0.95,
            color=palette["a"],
        )
        self.burst.emit_burst(
            self.origin[0],
            self.origin[1],
            count=22,
            speed=160 * self.power,
            life=0.7,
            color=palette["b"],
        )
        self.burst.emit_burst(
            self.origin[0],
            self.origin[1],
            count=16,
            speed=100 * self.power,
            life=0.55,
            color=palette["c"],
        )

    def update(self, dt: float, pose: AnimationPose, charge: float, state: str) -> None:
        if not self.active:
            return
        self.age += dt
        self.burst.update(dt)
        if self.age > 1.15:
            self.active = False
            self.burst.clear()

    def draw(self, layer: np.ndarray, pose: AnimationPose, charge: float, state: str) -> None:
        if not self.active:
            return
        ox, oy = int(self.origin[0]), int(self.origin[1])
        fade = max(0.0, 1.0 - self.age / 1.15)
        palette = STYLES[self.style]
        if self.style == "nova":
            self._draw_nova(layer, ox, oy, fade, palette)
        elif self.style == "rift":
            self._draw_rift(layer, ox, oy, fade, palette)
        elif self.style == "cut":
            self._draw_cut(layer, ox, oy, fade, palette)
        else:
            self._draw_fire(layer, ox, oy, fade, palette)
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

    def _draw_nova(self, layer, ox, oy, fade, pal) -> None:
        shock = int((50 + 260 * self.age) * self.power)
        cv2.circle(layer, (ox, oy), shock + 8, pal["ink"], 7, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), shock, pal["a"], 4, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), int(shock * 0.62), pal["b"], 3, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), int(shock * 0.28), pal["c"], -1, cv2.LINE_AA)
        petals = 10
        reach = int((70 + 210 * self.age) * self.power)
        for k in range(petals):
            ang = k * (2 * math.pi / petals) + self.age * 3
            tip = (int(ox + math.cos(ang) * reach), int(oy + math.sin(ang) * reach))
            left = (
                int(ox + math.cos(ang - 0.22) * reach * 0.45),
                int(oy + math.sin(ang - 0.22) * reach * 0.45),
            )
            right = (
                int(ox + math.cos(ang + 0.22) * reach * 0.45),
                int(oy + math.sin(ang + 0.22) * reach * 0.45),
            )
            cv2.fillConvexPoly(layer, np.array([tip, left, right], dtype=np.int32), pal["b"])
            cv2.line(layer, (ox, oy), tip, pal["c"], 1, cv2.LINE_AA)

    def _draw_rift(self, layer, ox, oy, fade, pal) -> None:
        shock = int((40 + 240 * self.age) * self.power)
        for i, (scale, spin) in enumerate(((1.0, 1.2), (0.72, -1.6), (0.44, 2.1))):
            r = max(12, int(shock * scale))
            color = pal["a"] if i == 0 else pal["b"] if i == 1 else pal["c"]
            cv2.ellipse(
                layer,
                (ox, oy),
                (r, int(r * 0.55)),
                math.degrees(self.age * 220 * spin),
                0,
                360,
                pal["ink"],
                6,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                layer,
                (ox, oy),
                (r, int(r * 0.55)),
                math.degrees(self.age * 220 * spin),
                0,
                360,
                color,
                3,
                cv2.LINE_AA,
            )
        height = int((80 + 260 * self.age) * self.power)
        cv2.line(layer, (ox, oy - height), (ox, oy + height), pal["ink"], 7, cv2.LINE_AA)
        cv2.line(layer, (ox, oy - height), (ox, oy + height), pal["c"], 2, cv2.LINE_AA)
        for k in range(10):
            ang = k * (math.pi / 5) + self.age * 4
            end = (
                int(ox + math.cos(ang) * shock),
                int(oy + math.sin(ang) * shock),
            )
            cv2.drawMarker(layer, end, pal["b"], cv2.MARKER_STAR, 14, 2, cv2.LINE_AA)

    def _draw_cut(self, layer, ox, oy, fade, pal) -> None:
        reach = int((90 + 280 * self.age) * self.power)
        for i in range(8):
            ang = i * (math.pi / 4) + self.age * 5
            end = (int(ox + math.cos(ang) * reach), int(oy + math.sin(ang) * reach))
            cv2.line(layer, (ox, oy), end, pal["ink"], 7, cv2.LINE_AA)
            cv2.line(layer, (ox, oy), end, pal["a"] if i % 2 == 0 else pal["b"], 3, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), int(reach * 0.22), pal["c"], 3, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), int(reach * 0.12), pal["a"], -1, cv2.LINE_AA)

    def _draw_fire(self, layer, ox, oy, fade, pal) -> None:
        shock = int((40 + 240 * self.age) * self.power)
        cv2.circle(layer, (ox, oy), shock + 6, pal["ink"], 6, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), shock, pal["a"], 4, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), int(shock * 0.62), pal["b"], 3, cv2.LINE_AA)
        cv2.circle(layer, (ox, oy), int(shock * 0.28), pal["c"], 2, cv2.LINE_AA)
        for k in range(14):
            ang = k * math.pi / 7 + self.age * 2.4
            length = int((60 + 220 * self.age) * self.power)
            end = (int(ox + math.cos(ang) * length), int(oy + math.sin(ang) * length))
            cv2.line(layer, (ox, oy), end, pal["ink"], 5, cv2.LINE_AA)
            cv2.line(layer, (ox, oy), end, pal["b"] if k % 2 else pal["a"], 2, cv2.LINE_AA)
