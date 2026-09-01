from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 2.0
    life: float = 1.0
    max_life: float = 1.0
    angle: float = 0.0
    speed: float = 0.0
    orbit_radius: float = 0.0
    spin: float = 0.0
    color: Tuple[int, int, int] = (80, 220, 255)


class ParticleSystem:
    def __init__(self, max_particles: int = 400):
        self.max_particles = max_particles
        self.particles: List[Particle] = []

    def emit(self, particle: Particle) -> None:
        if len(self.particles) >= self.max_particles:
            self.particles.pop(0)
        self.particles.append(particle)

    def emit_burst(
        self,
        x: float,
        y: float,
        count: int,
        speed: float = 80.0,
        life: float = 0.7,
        color: Tuple[int, int, int] = (80, 220, 255),
    ) -> None:
        rng = np.random.default_rng()
        for _ in range(count):
            ang = float(rng.uniform(0, 2 * np.pi))
            spd = float(rng.uniform(0.35, 1.0) * speed)
            self.emit(
                Particle(
                    x=x,
                    y=y,
                    vx=np.cos(ang) * spd,
                    vy=np.sin(ang) * spd,
                    radius=float(rng.uniform(1.2, 3.8)),
                    life=life,
                    max_life=life,
                    angle=ang,
                    speed=spd,
                    color=color,
                )
            )

    def update(self, dt: float) -> None:
        alive = []
        for particle in self.particles:
            particle.life -= dt
            if particle.life <= 0:
                continue
            if particle.orbit_radius > 0:
                particle.angle += particle.spin * dt
                particle.x += np.cos(particle.angle) * particle.orbit_radius * dt * 8
                particle.y += np.sin(particle.angle) * particle.orbit_radius * dt * 8
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vx *= 0.98
            particle.vy *= 0.98
            alive.append(particle)
        self.particles = alive

    def clear(self) -> None:
        self.particles.clear()

    def __len__(self) -> int:
        return len(self.particles)
