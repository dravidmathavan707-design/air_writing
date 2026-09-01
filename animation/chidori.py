"""
Chidori-style lightning: a dense charge in the hand, then senbon,
spears, and stream attacks along the pointing direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from animation.hand_pose import AnimationPose
from animation.particle_system import Particle, ParticleSystem

WHITE = (255, 250, 230)
ARC = (255, 210, 40)
CORE = (255, 120, 10)
GLOW = (255, 40, 0)
INK = (40, 8, 0)


def _jitter(rng, scale: float = 1.0) -> float:
    return float(rng.uniform(-scale, scale))


@dataclass
class Bolt:
    x: float
    y: float
    vx: float
    vy: float
    kind: str
    life: float
    max_life: float
    length: float
    width: int


class Chidori:
    def __init__(self):
        self.sparks = ParticleSystem(240)
        self.bolts: List[Bolt] = []
        self.time = 0.0
        self.firing = False

    def reset(self):
        self.sparks.clear()
        self.bolts.clear()
        self.time = 0.0
        self.firing = False

    def release(self, pose: AnimationPose, charge: float) -> None:
        cx, cy = pose.index_tip
        ang = pose.angle
        power = max(0.4, min(1.0, charge))
        rng = np.random.default_rng()
        self.firing = True
        # Long spear
        self.bolts.append(
            Bolt(
                x=cx,
                y=cy,
                vx=math.cos(ang) * 980 * power,
                vy=math.sin(ang) * 980 * power,
                kind="spear",
                life=0.55,
                max_life=0.55,
                length=70 + 90 * power,
                width=3,
            )
        )
        # Senbon volley
        for _ in range(int(10 + 14 * power)):
            spread = float(rng.uniform(-0.28, 0.28))
            a = ang + spread
            spd = float(rng.uniform(640, 1100) * power)
            self.bolts.append(
                Bolt(
                    x=cx + _jitter(rng, 8),
                    y=cy + _jitter(rng, 8),
                    vx=math.cos(a) * spd,
                    vy=math.sin(a) * spd,
                    kind="senbon",
                    life=float(rng.uniform(0.28, 0.5)),
                    max_life=0.5,
                    length=float(rng.uniform(18, 36)),
                    width=1,
                )
            )
        # Stream ribbons
        for k in range(4):
            a = ang + (k - 1.5) * 0.08
            self.bolts.append(
                Bolt(
                    x=cx,
                    y=cy,
                    vx=math.cos(a) * 720 * power,
                    vy=math.sin(a) * 720 * power,
                    kind="stream",
                    life=0.42,
                    max_life=0.42,
                    length=110 + 40 * power,
                    width=2,
                )
            )

    def update(self, dt: float, pose: AnimationPose, charge: float, state: str) -> None:
        self.time += dt
        rng = np.random.default_rng()
        if pose.present and (charge > 0.08 or pose.is_pointing or state in ("charging", "ready")):
            cx, cy = pose.index_tip
            for _ in range(int(3 + 10 * charge)):
                ang = float(rng.uniform(0, 2 * math.pi))
                dist = float(rng.uniform(4, 18 + 28 * charge))
                self.sparks.emit(
                    Particle(
                        x=cx + math.cos(ang) * dist,
                        y=cy + math.sin(ang) * dist,
                        vx=math.cos(ang) * 40,
                        vy=math.sin(ang) * 40,
                        radius=float(rng.uniform(0.8, 2.2)),
                        life=float(rng.uniform(0.05, 0.18)),
                        max_life=0.18,
                        color=ARC if rng.random() > 0.35 else WHITE,
                    )
                )
        self.sparks.update(dt)
        alive = []
        for bolt in self.bolts:
            bolt.life -= dt
            if bolt.life <= 0:
                continue
            bolt.x += bolt.vx * dt
            bolt.y += bolt.vy * dt
            alive.append(bolt)
        self.bolts = alive
        self.firing = len(self.bolts) > 0

    def draw(self, layer: np.ndarray, pose: AnimationPose, charge: float, state: str) -> None:
        rng = np.random.default_rng(int(self.time * 60) % 10000)
        if pose.present and charge > 0.05:
            self._draw_hand_blade(layer, pose, charge, rng)
            if pose.is_pointing or state in ("charging", "ready", "releasing"):
                self._draw_stream(layer, pose, charge, rng)
        self._draw_bolts(layer, rng)
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

    def _draw_hand_blade(self, layer, pose: AnimationPose, charge: float, rng) -> None:
        px, py = int(pose.palm[0]), int(pose.palm[1])
        tx, ty = int(pose.index_tip[0]), int(pose.index_tip[1])
        radius = int(10 + 26 * charge)
        flicker = 1 if rng.random() > 0.25 else 0
        cv2.circle(layer, (tx, ty), radius + 3, INK, -1, cv2.LINE_AA)
        cv2.circle(layer, (tx, ty), radius + flicker, CORE, -1, cv2.LINE_AA)
        cv2.circle(layer, (tx, ty), max(4, radius // 3), WHITE, -1, cv2.LINE_AA)
        for _ in range(int(6 + 10 * charge)):
            ang = float(rng.uniform(0, 2 * math.pi))
            length = float(rng.uniform(radius * 0.6, radius * 1.8))
            mid = (
                int(tx + math.cos(ang) * length * 0.45),
                int(ty + math.sin(ang) * length * 0.45),
            )
            end = (
                int(tx + math.cos(ang) * length),
                int(ty + math.sin(ang) * length),
            )
            cv2.line(layer, (tx, ty), mid, WHITE, 2, cv2.LINE_AA)
            cv2.line(layer, mid, end, ARC, 2, cv2.LINE_AA)
        cv2.line(layer, (px, py), (tx, ty), INK, 5, cv2.LINE_AA)
        cv2.line(layer, (px, py), (tx, ty), CORE, 3, cv2.LINE_AA)

    def _draw_stream(self, layer, pose: AnimationPose, charge: float, rng) -> None:
        tx, ty = pose.index_tip
        ang = pose.angle
        length = 90 + 220 * charge
        segs = 10
        prev = (int(tx), int(ty))
        for i in range(1, segs + 1):
            t = i / segs
            wobble = math.sin(self.time * 40 + i) * (6 + 10 * charge)
            px = tx + math.cos(ang) * length * t + math.cos(ang + math.pi / 2) * wobble
            py = ty + math.sin(ang) * length * t + math.sin(ang + math.pi / 2) * wobble
            pt = (int(px), int(py))
            cv2.line(layer, prev, pt, INK, 5, cv2.LINE_AA)
            cv2.line(layer, prev, pt, WHITE if i % 2 == 0 else ARC, 2, cv2.LINE_AA)
            prev = pt
        # Spear silhouette along the stream
        tip = (
            int(tx + math.cos(ang) * length),
            int(ty + math.sin(ang) * length),
        )
        cv2.line(layer, (int(tx), int(ty)), tip, GLOW, 3, cv2.LINE_AA)

    def _draw_bolts(self, layer, rng) -> None:
        for bolt in self.bolts:
            fade = max(0.0, bolt.life / bolt.max_life)
            speed = math.hypot(bolt.vx, bolt.vy) + 1e-5
            dx, dy = bolt.vx / speed, bolt.vy / speed
            start = (int(bolt.x), int(bolt.y))
            end = (int(bolt.x + dx * bolt.length), int(bolt.y + dy * bolt.length))
            color = WHITE if bolt.kind == "senbon" else ARC
            if bolt.kind == "spear":
                cv2.line(layer, start, end, INK, bolt.width + 5, cv2.LINE_AA)
                cv2.line(layer, start, end, CORE, bolt.width + 2, cv2.LINE_AA)
                cv2.line(layer, start, end, WHITE, 1, cv2.LINE_AA)
                # spear head
                nx, ny = -dy, dx
                head = end
                left = (int(end[0] - dx * 16 + nx * 7), int(end[1] - dy * 16 + ny * 7))
                right = (int(end[0] - dx * 16 - nx * 7), int(end[1] - dy * 16 - ny * 7))
                cv2.fillConvexPoly(layer, np.array([head, left, right], dtype=np.int32), WHITE)
            elif bolt.kind == "stream":
                wobble = int(rng.integers(-8, 9))
                mid = (
                    int((start[0] + end[0]) / 2 - dy * wobble),
                    int((start[1] + end[1]) / 2 + dx * wobble),
                )
                cv2.line(layer, start, mid, INK, bolt.width + 3, cv2.LINE_AA)
                cv2.line(layer, start, mid, ARC, bolt.width, cv2.LINE_AA)
                cv2.line(layer, mid, end, WHITE, 2, cv2.LINE_AA)
            else:
                cv2.line(layer, start, end, INK, bolt.width + 3, cv2.LINE_AA)
                cv2.line(layer, start, end, tuple(int(c * fade) for c in color), bolt.width + 1, cv2.LINE_AA)
