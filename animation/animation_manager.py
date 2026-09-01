from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from animation.chidori import Chidori
from animation.energy_blast import EnergyBlast
from animation.energy_sphere import EnergySphere
from animation.glow import apply_glow
from animation.hand_pose import AnimationPose, pose_from_hand, smooth_point
from animation.magic_portal import MagicPortal
from animation.wind_spiral import WindSpiral

IDLE = "idle"
CHARGING = "charging"
READY = "ready"
RELEASING = "releasing"
COOLDOWN = "cooldown"

EFFECTS = (
    "energy_sphere",
    "wind_spiral",
    "magic_portal",
    "energy_blast",
    "chidori",
)


class AnimationManager:
    def __init__(self, charge_seconds: float = 1.6, cooldown_seconds: float = 1.0):
        self.charge_seconds = charge_seconds
        self.cooldown_seconds = cooldown_seconds
        self.active_effect = "energy_sphere"
        self.state = IDLE
        self.charge = 0.0
        self.cooldown_left = 0.0
        self.sphere = EnergySphere()
        self.wind = WindSpiral()
        self.portal = MagicPortal()
        self.blast = EnergyBlast()
        self.chidori = Chidori()
        self.smooth_tip = None
        self.smooth_palm = None
        self._last_time = time.perf_counter()

    def set_effect(self, name: str) -> None:
        if name not in EFFECTS:
            return
        if name != self.active_effect:
            self.reset()
            self.active_effect = name

    def reset(self) -> None:
        self.state = IDLE
        self.charge = 0.0
        self.cooldown_left = 0.0
        self.sphere.reset()
        self.wind.reset()
        self.portal.reset()
        self.blast.reset()
        self.chidori.reset()
        self.smooth_tip = None
        self.smooth_palm = None

    def update(self, hand) -> AnimationPose:
        now = time.perf_counter()
        dt = min(0.05, max(0.001, now - self._last_time))
        self._last_time = now
        pose = pose_from_hand(hand)
        if pose.present:
            self.smooth_tip = smooth_point(self.smooth_tip, pose.index_tip, 0.32)
            self.smooth_palm = smooth_point(self.smooth_palm, pose.palm, 0.28)
            pose.index_tip = self.smooth_tip
            pose.palm = self.smooth_palm
        self._update_state(dt, pose)
        self._active().update(dt, pose, self.charge, self.state)
        if self.active_effect not in ("energy_blast", "chidori"):
            self.blast.update(dt, pose, self.charge, self.state)
        return pose

    def _update_state(self, dt: float, pose: AnimationPose) -> None:
        if self.state == COOLDOWN:
            self.cooldown_left -= dt
            if self.cooldown_left <= 0:
                self.state = IDLE
                self.charge = 0.0
            return

        if self.state == RELEASING:
            still_going = self.chidori.firing if self.active_effect == "chidori" else self.blast.active
            if not still_going:
                self.state = COOLDOWN
                self.cooldown_left = self.cooldown_seconds
                self.charge = 0.0
            return

        if not pose.present:
            if self.state in (CHARGING, READY) and self.charge > 0:
                self.charge = max(0.0, self.charge - dt * 0.35)
            return

        if pose.is_fist and self.state in (IDLE, CHARGING, READY):
            self.state = CHARGING
            self.charge = min(1.0, self.charge + dt / self.charge_seconds)
            if self.charge >= 1.0:
                self.state = READY
                self.charge = 1.0
            return

        if pose.is_open and self.state in (CHARGING, READY) and self.charge >= 0.35:
            if self.active_effect == "chidori":
                self.chidori.release(pose, self.charge)
            else:
                origin = pose.palm if self.active_effect == "magic_portal" else pose.index_tip
                style = {
                    "energy_sphere": "nova",
                    "magic_portal": "rift",
                    "wind_spiral": "cut",
                    "energy_blast": "fire",
                }.get(self.active_effect, "fire")
                self.blast.trigger(origin, self.charge, style=style)
            self.state = RELEASING
            return

        if self.state == CHARGING and not pose.is_fist:
            self.charge = max(0.0, self.charge - dt * 0.5)
            if self.charge <= 0.02:
                self.state = IDLE
                self.charge = 0.0

    def _active(self):
        if self.active_effect == "wind_spiral":
            return self.wind
        if self.active_effect == "magic_portal":
            return self.portal
        if self.active_effect == "energy_blast":
            return self.blast
        if self.active_effect == "chidori":
            return self.chidori
        return self.sphere

    def render(self, frame: np.ndarray, pose: AnimationPose) -> np.ndarray:
        dimmed = cv2.convertScaleAbs(frame, alpha=0.46, beta=-22)
        layer = np.zeros_like(frame)
        self._active().draw(layer, pose, self.charge, self.state)
        if self.active_effect not in ("energy_blast", "chidori") and (
            self.blast.active or self.state == RELEASING
        ):
            self.blast.draw(layer, pose, self.charge, self.state)
        return apply_glow(dimmed, layer, sigma=10.0, glow_strength=0.95, core_strength=1.18)

    def status_text(self) -> str:
        labels = {
            IDLE: "IDLE - FIST TO CHARGE",
            CHARGING: f"CHARGING {int(self.charge * 100)}%",
            READY: "READY - OPEN PALM TO RELEASE",
            RELEASING: "RELEASE",
            COOLDOWN: "COOLDOWN",
        }
        names = {
            "energy_sphere": "ENERGY SPHERE",
            "wind_spiral": "RASENSHURIKEN",
            "magic_portal": "MAGIC PORTAL",
            "energy_blast": "ENERGY BLAST",
            "chidori": "CHIDORI",
        }
        effect = names.get(self.active_effect, self.active_effect.replace("_", " ").upper())
        return f"{effect} | {labels.get(self.state, self.state.upper())}"
