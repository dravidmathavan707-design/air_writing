"""Turn an air-drawing canvas into connected ink blobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass
class Blob:
    contour: np.ndarray
    bbox: BBox
    area: float
    holes: int
    mask: np.ndarray
    cx: float
    cy: float


def canvas_to_binary(canvas: np.ndarray) -> np.ndarray:
    if canvas.ndim == 3:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)


def _holes_for(index: int, hierarchy: np.ndarray) -> int:
    child = int(hierarchy[0][index][2])
    count = 0
    while child != -1:
        count += 1
        child = int(hierarchy[0][child][0])
    return count


def extract_blobs(binary: np.ndarray, min_area: float = 90.0) -> List[Blob]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []

    blobs: List[Blob] = []
    h, w = binary.shape[:2]
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = binary[y0:y1, x0:x1].copy()
        M = cv2.moments(contour)
        cx = (M["m10"] / M["m00"]) if M["m00"] else x + bw / 2
        cy = (M["m01"] / M["m00"]) if M["m00"] else y + bh / 2
        blobs.append(
            Blob(
                contour=contour,
                bbox=(x, y, bw, bh),
                area=area,
                holes=_holes_for(i, hierarchy),
                mask=crop,
                cx=float(cx),
                cy=float(cy),
            )
        )
    return _merge_nearby(blobs, binary)


def _box_gap(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    dx = max(0, max(ax, bx) - min(ax2, bx2))
    dy = max(0, max(ay, by) - min(ay2, by2))
    return float(np.hypot(dx, dy))


def _union_box(a: BBox, b: BBox) -> BBox:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0


def _merge_nearby(blobs: List[Blob], binary: np.ndarray) -> List[Blob]:
    """Join dots with stems (i/j) and stacked bars (=)."""
    if len(blobs) < 2:
        return blobs
    used = [False] * len(blobs)
    merged: List[Blob] = []
    for i, left in enumerate(blobs):
        if used[i]:
            continue
        group = [left]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(blobs):
                if used[j]:
                    continue
                if _should_merge(group, other):
                    group.append(other)
                    used[j] = True
                    changed = True
        merged.append(_combine(group, binary) if len(group) > 1 else left)
    return merged


def _should_merge(group: List[Blob], other: Blob) -> bool:
    gx = [b.bbox[0] for b in group] + [b.bbox[0] + b.bbox[2] for b in group]
    gy = [b.bbox[1] for b in group] + [b.bbox[1] + b.bbox[3] for b in group]
    gbox = (min(gx), min(gy), max(gx) - min(gx), max(gy) - min(gy))
    gap = _box_gap(gbox, other.bbox)
    gw, gh = max(gbox[2], 1), max(gbox[3], 1)
    ow, oh = max(other.bbox[2], 1), max(other.bbox[3], 1)
    x_overlap = min(gbox[0] + gbox[2], other.bbox[0] + other.bbox[2]) - max(gbox[0], other.bbox[0])
    y_overlap = min(gbox[1] + gbox[3], other.bbox[1] + other.bbox[3]) - max(gbox[1], other.bbox[1])
    dotted = x_overlap > 0.25 * min(gw, ow) and gap < max(18.0, 0.55 * max(gh, oh))
    stacked = y_overlap < 0 and gap < 22 and abs(gw - ow) < 0.45 * max(gw, ow)
    close = gap < 10 and (x_overlap > 0 or y_overlap > 0)
    return dotted or stacked or close


def _combine(group: List[Blob], binary: np.ndarray) -> Blob:
    box = group[0].bbox
    for blob in group[1:]:
        box = _union_box(box, blob.bbox)
    x, y, w, h = box
    pad = 4
    h_img, w_img = binary.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    mask = binary[y0:y1, x0:x1].copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    main = max(contours, key=cv2.contourArea) if contours else group[0].contour
    holes = 0
    if hierarchy is not None:
        for i, _cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                holes = max(holes, _holes_for(i, hierarchy))
    area = float(sum(b.area for b in group))
    return Blob(
        contour=main,
        bbox=(x, y, w, h),
        area=area,
        holes=holes,
        mask=mask,
        cx=x + w / 2,
        cy=y + h / 2,
    )


def occupancy_grid(mask: np.ndarray, size: int = 20) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.float32)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)
