"""Map Face Mesh landmarks into a stable hero-mask pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from face.face_landmarks import (
    CHIN,
    FACE_OVAL,
    FOREHEAD,
    LEFT_EYE,
    LIPS_OUTER,
    NOSE_TIP_IDX,
    RIGHT_EYE,
    get_group,
)


@dataclass
class HeroFace:
    present: bool = False
    oval: Optional[np.ndarray] = None
    left_eye: Optional[np.ndarray] = None
    right_eye: Optional[np.ndarray] = None
    lips: Optional[np.ndarray] = None
    center: tuple = (0.0, 0.0)
    nose: tuple = (0.0, 0.0)
    forehead: tuple = (0.0, 0.0)
    chin: tuple = (0.0, 0.0)
    scale: float = 80.0
    angle: float = 0.0
    bbox: tuple = (0, 0, 0, 0)
    points: Optional[np.ndarray] = None


class FaceMapper:
    def __init__(self, smooth: float = 0.35):
        self.smooth = smooth
        self._prev: Optional[HeroFace] = None

    def reset(self) -> None:
        self._prev = None

    def map(self, points) -> HeroFace:
        if points is None or len(points) < 400:
            self._prev = None
            return HeroFace()
        pts = np.asarray(points, dtype=np.float32)
        oval = get_group(pts, FACE_OVAL)
        left = get_group(pts, LEFT_EYE)
        right = get_group(pts, RIGHT_EYE)
        lips = get_group(pts, LIPS_OUTER)
        nose = tuple(pts[NOSE_TIP_IDX]) if len(pts) > NOSE_TIP_IDX else (0.0, 0.0)
        forehead = tuple(pts[FOREHEAD]) if len(pts) > FOREHEAD else nose
        chin = tuple(pts[CHIN]) if len(pts) > CHIN else nose
        if len(oval) < 8 or len(left) < 4 or len(right) < 4:
            return HeroFace()
        lc = left.mean(axis=0)
        rc = right.mean(axis=0)
        center = ((lc[0] + rc[0]) * 0.5, (lc[1] + rc[1]) * 0.5)
        dx, dy = rc[0] - lc[0], rc[1] - lc[1]
        angle = float(np.arctan2(dy, dx))
        scale = float(np.hypot(dx, dy) + np.hypot(chin[0] - forehead[0], chin[1] - forehead[1]) * 0.45)
        x0, y0 = oval.min(axis=0)
        x1, y1 = oval.max(axis=0)
        face = HeroFace(
            present=True,
            oval=oval,
            left_eye=left,
            right_eye=right,
            lips=lips,
            center=center,
            nose=nose,
            forehead=forehead,
            chin=chin,
            scale=max(40.0, scale),
            angle=angle,
            bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            points=pts,
        )
        if self._prev and self._prev.present:
            a = self.smooth
            face.center = _mix2(self._prev.center, face.center, a)
            face.scale = self._prev.scale * (1 - a) + face.scale * a
            face.angle = self._prev.angle * (1 - a) + face.angle * a
            if self._prev.points is not None and len(self._prev.points) == len(pts):
                face.points = self._prev.points * (1.0 - a) + pts * a
                face.oval = get_group(face.points, FACE_OVAL)
                face.left_eye = get_group(face.points, LEFT_EYE)
                face.right_eye = get_group(face.points, RIGHT_EYE)
                face.lips = get_group(face.points, LIPS_OUTER)
        self._prev = face
        return face


def _mix2(a, b, t):
    return (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)


class FaceMapperPool:
    """Keep one smoother per on-screen face so every person gets a mask."""

    def __init__(self, smooth: float = 0.35, max_miss: int = 12, match_px: float = 160.0):
        self.smooth = smooth
        self.max_miss = max_miss
        self.match_px = match_px
        self.slots: list = []

    def reset(self) -> None:
        self.slots = []

    def map_all(self, point_lists) -> list:
        detections = []
        for pts in point_lists or []:
            if pts is None or len(pts) < 400:
                continue
            arr = np.asarray(pts, dtype=np.float32)
            detections.append((_center_of(arr), arr))
        used = set()
        mapped = []
        for slot in self.slots:
            slot["miss"] += 1
        for center, arr in detections:
            best_i = -1
            best_d = self.match_px
            for i, slot in enumerate(self.slots):
                if i in used:
                    continue
                d = float(np.hypot(center[0] - slot["center"][0], center[1] - slot["center"][1]))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i < 0:
                mapper = FaceMapper(smooth=self.smooth)
                face = mapper.map(arr)
                self.slots.append({"mapper": mapper, "center": face.center, "miss": 0})
                mapped.append(face)
            else:
                used.add(best_i)
                slot = self.slots[best_i]
                slot["miss"] = 0
                face = slot["mapper"].map(arr)
                slot["center"] = face.center
                mapped.append(face)
        self.slots = [s for s in self.slots if s["miss"] <= self.max_miss]
        return [f for f in mapped if f.present]


def _center_of(pts: np.ndarray) -> tuple:
    if len(pts) > 263:
        left, right = pts[33], pts[263]
        return ((float(left[0]) + float(right[0])) * 0.5, (float(left[1]) + float(right[1])) * 0.5)
    mid = pts.mean(axis=0)
    return (float(mid[0]), float(mid[1]))
