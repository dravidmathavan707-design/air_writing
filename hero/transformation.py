"""Hero transformation stages: scan, charge, assemble, hero."""

from __future__ import annotations

SCAN = "scan"
CHARGE = "charge"
ASSEMBLE = "assemble"
HERO = "hero"
IDLE = "idle"


class Transformation:
    def __init__(self):
        self.stage = IDLE
        self.time = 0.0
        self.scan_s = 0.38
        self.charge_s = 0.22
        self.assemble_s = 0.28

    def reset(self) -> None:
        self.stage = IDLE
        self.time = 0.0

    def begin(self) -> None:
        self.stage = SCAN
        self.time = 0.0

    def update(self, dt: float, has_face: bool) -> None:
        if not has_face:
            if self.stage != HERO:
                self.stage = IDLE
                self.time = 0.0
            return
        if self.stage == IDLE:
            self.begin()
            return
        if self.stage == HERO:
            self.time += dt
            return
        self.time += dt
        if self.stage == SCAN and self.time >= self.scan_s:
            self.stage = CHARGE
            self.time = 0.0
        elif self.stage == CHARGE and self.time >= self.charge_s:
            self.stage = ASSEMBLE
            self.time = 0.0
        elif self.stage == ASSEMBLE and self.time >= self.assemble_s:
            self.stage = HERO
            self.time = 0.0

    @property
    def mask_alpha(self) -> float:
        if self.stage == ASSEMBLE:
            return min(1.0, self.time / max(self.assemble_s, 0.01))
        if self.stage == HERO:
            return 1.0
        if self.stage == CHARGE:
            return min(0.55, self.time / max(self.charge_s, 0.01) * 0.55)
        return 0.0

    @property
    def scan_t(self) -> float:
        if self.stage != SCAN:
            return 1.0
        return min(1.0, self.time / max(self.scan_s, 0.01))
