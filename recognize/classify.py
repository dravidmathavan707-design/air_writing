"""Classify one ink blob as a shape, symbol, letter, or digit."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from recognize.blobs import Blob, occupancy_grid

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
HOLE_HINTS = {
    0: set("CEFGHIJKLMNSTUVWXYZ12357"),
    1: set("ADOPQR0469"),
    2: set("B8"),
}


def _templates() -> Dict[str, List[np.ndarray]]:
    bank: Dict[str, List[np.ndarray]] = {}
    fonts = (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    )
    for ch in LETTERS + DIGITS:
        grids = []
        for font in fonts:
            canvas = np.zeros((120, 120), dtype=np.uint8)
            cv2.putText(canvas, ch, (18, 90), font, 2.2, 255, 4, cv2.LINE_AA)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thick = cv2.dilate(canvas, kernel, iterations=1)
            grids.append(occupancy_grid(canvas))
            grids.append(occupancy_grid(thick))
        bank[ch] = grids
    return bank


_TEMPLATE_BANK: Optional[Dict[str, List[np.ndarray]]] = None


def template_bank() -> Dict[str, List[np.ndarray]]:
    global _TEMPLATE_BANK
    if _TEMPLATE_BANK is None:
        _TEMPLATE_BANK = _templates()
    return _TEMPLATE_BANK


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    fa, fb = a.ravel(), b.ravel()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-6)
    return float(np.dot(fa, fb) / denom)


def classify_geometry(blob: Blob) -> Optional[Tuple[str, str, float]]:
    """Return (kind, label, score) for a strong geometric match."""
    cnt = blob.contour
    peri = cv2.arcLength(cnt, True)
    if peri < 8:
        return None
    area = max(blob.area, 1.0)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    verts = len(approx)
    _x, _y, w, h = blob.bbox
    aspect = w / max(h, 1)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity = area / hull_area
    circularity = 4 * math.pi * area / (peri * peri)
    extent = area / max(w * h, 1)
    mask = blob.mask

    if circularity >= 0.72 and verts >= 6:
        label = "CIRCLE" if 0.75 <= aspect <= 1.28 else "ELLIPSE"
        return "shape", label, min(0.97, 0.75 + 0.25 * circularity)

    if verts == 3:
        return "shape", "TRIANGLE", 0.9

    if verts == 4:
        box = cv2.minAreaRect(cnt)
        bw, bh = box[1]
        if min(bw, bh) <= 1:
            return "shape", "LINE", 0.8
        ratio = max(bw, bh) / max(min(bw, bh), 1)
        ang = abs(box[2])
        diamond = 28 < ang < 62 and 0.7 < aspect < 1.4
        if diamond and ratio < 1.55:
            return "shape", "DIAMOND", 0.86
        if ratio < 1.22 and 0.78 < aspect < 1.28:
            return "shape", "SQUARE", 0.9
        if ratio > 3.8:
            return "shape", "LINE", 0.84
        return "shape", "RECTANGLE", 0.86

    if verts == 5 and solidity > 0.8 and circularity < 0.78:
        return "shape", "PENTAGON", 0.8

    if verts == 6 and solidity > 0.8 and circularity < 0.78:
        return "shape", "HEXAGON", 0.8

    equals = _equals_score(mask)
    if equals >= 0.78:
        return "symbol", "EQUALS", equals

    minus = aspect > 2.6 and h < 28 and solidity > 0.7 and verts <= 6
    if minus:
        return "symbol", "MINUS", 0.86

    plus = _plus_score(mask, circularity, aspect)
    if plus >= 0.78:
        return "symbol", "PLUS", plus

    if _star_score(cnt, solidity, verts) >= 0.8:
        return "shape", "STAR", 0.84

    heart = _heart_score(cnt, solidity, circularity, aspect)
    if heart >= 0.76:
        return "symbol", "HEART", heart

    arrow = _arrow_score(cnt, solidity, aspect, verts)
    if arrow >= 0.78:
        return "shape", "ARROW", arrow

    if aspect > 4.2 and extent > 0.35:
        return "shape", "LINE", 0.8
    return None


def _plus_score(mask: np.ndarray, circularity: float, aspect: float) -> float:
    if mask.size == 0 or circularity > 0.62 or not (0.55 <= aspect <= 1.7):
        return 0.0
    h, w = mask.shape
    ink = mask > 0
    if ink.mean() < 0.08:
        return 0.0
    corners = [
        ink[: h // 3, : w // 3],
        ink[: h // 3, 2 * w // 3 :],
        ink[2 * h // 3 :, : w // 3],
        ink[2 * h // 3 :, 2 * w // 3 :],
    ]
    arms = [
        ink[: h // 3, w // 3 : 2 * w // 3],
        ink[2 * h // 3 :, w // 3 : 2 * w // 3],
        ink[h // 3 : 2 * h // 3, : w // 3],
        ink[h // 3 : 2 * h // 3, 2 * w // 3 :],
    ]
    center = ink[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    corner_mean = float(np.mean([q.mean() for q in corners]))
    arm_vals = [float(q.mean()) for q in arms]
    center_mean = float(center.mean())
    if corner_mean > 0.18 or center_mean < 0.22:
        return 0.0
    if min(arm_vals) < 0.16:
        return 0.0
    return max(0.0, min(1.0, 0.3 * center_mean + 0.5 * float(np.mean(arm_vals)) + 0.3 * (1.0 - corner_mean)))


def _equals_score(mask: np.ndarray) -> float:
    h, w = mask.shape
    if w < h * 1.15:
        return 0.0
    ink = (mask > 0).astype(np.uint8)
    rows = ink.mean(axis=1)
    peaks = []
    on = False
    start = 0
    for i, v in enumerate(rows):
        if v > 0.12 and not on:
            on = True
            start = i
        elif v <= 0.12 and on:
            on = False
            peaks.append((start, i))
    if on:
        peaks.append((start, len(rows)))
    if len(peaks) != 2:
        return 0.0
    (a0, a1), (b0, b1) = peaks
    gap = b0 - a1
    if gap < 2:
        return 0.0
    return 0.88


def _star_score(cnt, solidity: float, verts: int) -> float:
    if solidity > 0.58 or verts < 8:
        return 0.0
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 5:
        return 0.0
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return 0.0
    deep = 0
    for row in defects:
        _s, _e, _f, depth = row[0]
        if depth / 256.0 > 10:
            deep += 1
    if deep >= 5:
        return 0.84
    return 0.0


def _heart_score(cnt, solidity: float, circularity: float, aspect: float) -> float:
    if not (0.55 <= aspect <= 1.25):
        return 0.0
    if not (0.62 <= solidity <= 0.9):
        return 0.0
    if circularity > 0.82:
        return 0.0
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None:
        return 0.0
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None or len(defects) < 1:
        return 0.0
    pts = cnt.reshape(-1, 2)
    top = pts[np.argmin(pts[:, 1])]
    # cleft near the top center
    cx = float(np.mean(pts[:, 0]))
    near_top = pts[pts[:, 1] < np.percentile(pts[:, 1], 35)]
    if len(near_top) < 4:
        return 0.0
    dip = float(near_top[np.argmin(np.abs(near_top[:, 0] - cx))][1] - pts[:, 1].min())
    if dip > 4:
        return 0.8
    return 0.0


def _arrow_score(cnt, solidity: float, aspect: float, verts: int) -> float:
    if verts < 5 or verts > 9:
        return 0.0
    if solidity > 0.92:
        return 0.0
    pts = cnt.reshape(-1, 2).astype(np.float32)
    cx, cy = pts.mean(axis=0)
    dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    tip = pts[int(np.argmax(dist))]
    far = float(dist.max())
    med = float(np.median(dist))
    if far > med * 1.55 and 0.9 < aspect < 3.2:
        return 0.82
    return 0.0


def classify_character(blob: Blob) -> Optional[Tuple[str, str, float]]:
    grid = occupancy_grid(blob.mask)
    if float(grid.mean()) < 0.04:
        return None
    holes = min(2, max(0, blob.holes))
    allowed = HOLE_HINTS.get(holes, set(LETTERS + DIGITS))
    bank = template_bank()
    best_ch = None
    best = 0.0
    second = 0.0
    for ch, grids in bank.items():
        if ch not in allowed and ch not in "B8P":
            continue
        score = max(cosine(grid, tmpl) for tmpl in grids)
        if ch not in allowed:
            score *= 0.88
        if score > best:
            second = best
            best = score
            best_ch = ch
        elif score > second:
            second = score
    if best_ch is None or best < 0.62:
        return None
    if best - second < 0.02 and best < 0.78:
        return None
    kind = "digit" if best_ch in DIGITS else "letter"
    return kind, best_ch, float(best)


def classify_blob(blob: Blob) -> Optional[Tuple[str, str, float]]:
    geom = classify_geometry(blob)
    char = classify_character(blob)
    if geom and geom[0] in ("shape", "object") and geom[1] in (
        "CIRCLE",
        "ELLIPSE",
        "TRIANGLE",
        "SQUARE",
        "RECTANGLE",
        "DIAMOND",
        "ARROW",
        "PENTAGON",
        "HEXAGON",
        "LINE",
    ):
        if char is None or geom[1] in ("CIRCLE", "ELLIPSE", "TRIANGLE", "SQUARE", "RECTANGLE") or geom[2] >= char[2] + 0.08:
            return geom
    if char and char[2] >= 0.72:
        return char
    if geom:
        return geom
    return char
