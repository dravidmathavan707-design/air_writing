"""
Wind-release spinning shuriken: a compressed spiral core with rotating
cutting blades. Replaces the old wind-spiral effect.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from animation.hand_pose import AnimationPose
from animation.particle_system import Particle, ParticleSystem

CORE = (20, 255, 90)
WIND = (40, 230, 255)
BLADE = (0, 160, 255)
EDGE = (0, 50, 20)
INK = (10, 25, 8)


class WindSpiral:
    """Rasenshuriken-style wind sphere that follows the index fingertip."""

    def __init__(self):
        self.shards = ParticleSystem(260)
        self.time = 0.0

    def reset(self):
        self.shards.clear()
        self.time = 0.0

    def update(self, dt: float, pose: AnimationPose, charge: float, state: str) -> None:
        self.time += dt
        if not pose.present:
            self.shards.update(dt)
            return
        cx, cy = pose.index_tip
        rng = np.random.default_rng()
        power = 0.2 + 0.8 * max(charge, 0.15)
        spin = 10.0 + 18.0 * charge
        count = 6 if state == "idle" else 12
        reach = 14 + 44 + 110 * power + pose.scale * 0.22
        for _ in range(count):
            ang = float(rng.uniform(0, 2 * math.pi))
            rad = float(rng.uniform(10, reach))
            self.shards.emit(
                Particle(
                    x=cx + math.cos(ang) * rad,
                    y=cy + math.sin(ang) * rad,
                    vx=math.cos(ang + math.pi / 2) * spin * 5,
                    vy=math.sin(ang + math.pi / 2) * spin * 5,
                    radius=float(rng.uniform(1.4, 3.6)),
                    life=float(rng.uniform(0.14, 0.4)),
                    max_life=0.4,
                    angle=ang,
                    orbit_radius=rad * 0.14,
                    spin=spin if rng.random() > 0.5 else -spin,
                    color=WIND if rng.random() > 0.4 else CORE,
                )
            )
        self.shards.update(dt)

    def draw(self, layer: np.ndarray, pose: AnimationPose, charge: float, state: str) -> None:
        if not pose.present and state not in ("releasing",):
            self._draw_shards(layer)
            return
        cx, cy = int(pose.index_tip[0]), int(pose.index_tip[1])
        power = max(charge, 0.18 if pose.is_pointing else charge)
        if state == "releasing":
            power = max(power, 0.85)
        radius = int(28 + pose.scale * 0.45 + 88 * power)
        spin = self.time * (7.0 + 14.0 * power)

        self._draw_core(layer, cx, cy, radius, spin, power)
        self._draw_blades(layer, cx, cy, radius, spin, power)
        self._draw_wind_shells(layer, cx, cy, radius, spin, power)
        if state == "releasing":
            self._draw_cut_burst(layer, cx, cy, radius, spin)
        self._draw_shards(layer)

    def _draw_core(self, layer, cx, cy, radius, spin, power):
        cv2.circle(layer, (cx, cy), max(10, int(radius * 0.22)), INK, -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), max(7, int(radius * 0.20)), CORE, -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), max(12, int(radius * 0.38)), WIND, 4, cv2.LINE_AA)
        for speed, scale in ((1.0, 0.22), (-1.35, 0.30), (1.8, 0.40)):
            pts = []
            for i in range(36):
                t = i / 35.0
                ang = spin * speed + t * 4 * math.pi
                rad = radius * scale * (0.4 + t)
                pts.append((int(cx + math.cos(ang) * rad), int(cy + math.sin(ang) * rad)))
            for a, b in zip(pts, pts[1:]):
                cv2.line(layer, a, b, CORE, 1, cv2.LINE_AA)

    def _draw_blades(self, layer, cx, cy, radius, spin, power):
        blade_len = int(radius * (1.28 + 0.62 * power))
        blade_w = max(10, int(16 + 22 * power))
        for i in range(4):
            ang = spin * 1.15 + i * (math.pi / 2)
            perp = ang + math.pi / 2
            tip = (
                int(cx + math.cos(ang) * blade_len),
                int(cy + math.sin(ang) * blade_len),
            )
            left = (
                int(cx + math.cos(ang) * radius * 0.25 + math.cos(perp) * blade_w),
                int(cy + math.sin(ang) * radius * 0.25 + math.sin(perp) * blade_w),
            )
            right = (
                int(cx + math.cos(ang) * radius * 0.25 - math.cos(perp) * blade_w),
                int(cy + math.sin(ang) * radius * 0.25 - math.sin(perp) * blade_w),
            )
            curve = []
            for t in np.linspace(0, 1, 8):
                t = float(t)
                # Slightly hooked cutting edge.
                hook = math.sin(t * math.pi) * 0.22
                px = (1 - t) * (1 - t) * left[0] + 2 * (1 - t) * t * (tip[0] + math.cos(perp) * blade_w * hook) + t * t * right[0]
                py = (1 - t) * (1 - t) * left[1] + 2 * (1 - t) * t * (tip[1] + math.sin(perp) * blade_w * hook) + t * t * right[1]
                curve.append((int(px), int(py)))
            poly = np.array([left, tip, right], dtype=np.int32)
            cv2.fillConvexPoly(layer, poly, INK, lineType=cv2.LINE_AA)
            inner = np.array(
                [
                    (
                        int(left[0] * 0.15 + cx * 0.85),
                        int(left[1] * 0.15 + cy * 0.85),
                    ),
                    tip,
                    (
                        int(right[0] * 0.15 + cx * 0.85),
                        int(right[1] * 0.15 + cy * 0.85),
                    ),
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(layer, inner, BLADE, lineType=cv2.LINE_AA)
            cv2.polylines(layer, [np.array(curve, dtype=np.int32)], False, EDGE, 3, cv2.LINE_AA)
            cv2.line(layer, (cx, cy), tip, WIND, 3, cv2.LINE_AA)

    def _draw_wind_shells(self, layer, cx, cy, radius, spin, power):
        for i, (scale, thick) in enumerate(((0.92, 2), (1.18, 3), (1.42, 2), (1.62, 1))):
            r = max(10, int(radius * scale))
            tilt = math.degrees(spin * (1.2 if i % 2 == 0 else -0.9))
            cv2.ellipse(
                layer,
                (cx, cy),
                (r, int(r * 0.78)),
                tilt,
                0,
                360,
                WIND if i else EDGE,
                thick,
                cv2.LINE_AA,
            )

    def _draw_cut_burst(self, layer, cx, cy, radius, spin):
        reach = int(radius * 2.85)
        for i in range(16):
            ang = spin + i * (math.pi / 8)
            end = (int(cx + math.cos(ang) * reach), int(cy + math.sin(ang) * reach))
            cv2.line(layer, (cx, cy), end, BLADE, 3, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), int(radius * 1.75), EDGE, 3, cv2.LINE_AA)

    def _draw_shards(self, layer: np.ndarray) -> None:
        for particle in self.shards.particles:
            fade = max(0.0, particle.life / particle.max_life)
            cv2.circle(
                layer,
                (int(particle.x), int(particle.y)),
                max(1, int(particle.radius)),
                tuple(int(c * fade) for c in particle.color),
                -1,
                cv2.LINE_AA,
            )


Rasenshuriken = WindSpiral
