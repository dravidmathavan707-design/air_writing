"""Detect letters, digits, shapes, symbols, objects, and words on a drawing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from recognize.blobs import Blob, canvas_to_binary, extract_blobs
from recognize.classify import classify_blob

KIND_COLOR = {
    "letter": (40, 230, 80),
    "digit": (20, 200, 255),
    "word": (20, 210, 255),
    "number": (20, 200, 255),
    "shape": (180, 80, 255),
    "symbol": (0, 220, 255),
    "object": (40, 60, 255),
}


@dataclass
class Detection:
    kind: str
    label: str
    score: float
    bbox: Tuple[int, int, int, int]


class DrawingRecognizer:
    def __init__(self, min_interval: float = 0.35):
        self.min_interval = min_interval
        self.detections: List[Detection] = []
        self._last_hash = 0
        self._last_time = 0.0

    def reset(self) -> None:
        self.detections = []
        self._last_hash = 0

    def update(self, canvas: np.ndarray, now: float, force: bool = False) -> List[Detection]:
        digest = int(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).sum()) if canvas.ndim == 3 else int(canvas.sum())
        if not force and digest == self._last_hash and now - self._last_time < 1.0:
            return self.detections
        if not force and now - self._last_time < self.min_interval and digest == self._last_hash:
            return self.detections
        self._last_hash = digest
        self._last_time = now
        self.detections = recognize_drawing(canvas)
        return self.detections

    def summary(self) -> str:
        if not self.detections:
            return "NO INK TO READ"
        parts = []
        for det in self.detections:
            if det.kind in ("word", "number"):
                parts.append(det.label)
            elif det.kind in ("letter", "digit"):
                continue
            else:
                parts.append(det.label)
        letters = [d.label for d in self.detections if d.kind in ("letter", "digit")]
        if not parts and letters:
            parts = letters
        text = ", ".join(parts[:8])
        return f"FOUND {len(self.detections)}: {text}" if text else "SCANNING DRAWING"


def recognize_drawing(canvas: np.ndarray) -> List[Detection]:
    binary = canvas_to_binary(canvas)
    if int(cv2.countNonZero(binary)) < 80:
        return []
    blobs = extract_blobs(binary)
    detections: List[Detection] = []
    for blob in blobs:
        result = classify_blob(blob)
        if result is None:
            continue
        kind, label, score = result
        detections.append(Detection(kind, label, score, blob.bbox))
    detections = _add_objects(detections, blobs)
    detections.extend(_group_text(detections))
    detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
    return detections


def _center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _add_objects(detections: List[Detection], blobs: Sequence[Blob]) -> List[Detection]:
    used = set()
    extra: List[Detection] = []

    def take(*indexes: int) -> None:
        used.update(indexes)

    shapes = [(i, d) for i, d in enumerate(detections) if d.kind == "shape"]
    triangles = [(i, d) for i, d in shapes if d.label == "TRIANGLE"]
    rects = [(i, d) for i, d in shapes if d.label in ("RECTANGLE", "SQUARE")]
    circles = [(i, d) for i, d in shapes if d.label in ("CIRCLE", "ELLIPSE")]
    lines = [(i, d) for i, d in shapes if d.label == "LINE"]

    for ti, tri in triangles:
        for ri, rect in rects:
            if ti in used or ri in used:
                continue
            tx, ty, tw, th = tri.bbox
            rx, ry, rw, rh = rect.bbox
            x_overlap = min(tx + tw, rx + rw) - max(tx, rx)
            stacked = ty + th <= ry + rh * 0.35 and abs((ty + th) - ry) < max(24, th * 0.6)
            if x_overlap > 0.35 * min(tw, rw) and (stacked or (ty < ry and ty + th > ry - 20)):
                box = (
                    min(tx, rx),
                    min(ty, ry),
                    max(tx + tw, rx + rw) - min(tx, rx),
                    max(ty + th, ry + rh) - min(ty, ry),
                )
                extra.append(Detection("object", "HOUSE", 0.88, box))
                take(ti, ri)

    for ci, circ in circles:
        if ci in used:
            continue
        cx, cy = _center(circ.bbox)
        nearby_lines = 0
        line_ids = []
        for li, line in lines:
            lx, ly = _center(line.bbox)
            dist = np.hypot(cx - lx, cy - ly)
            if dist < max(circ.bbox[2], circ.bbox[3]) * 1.35 + 30:
                nearby_lines += 1
                line_ids.append(li)
        if nearby_lines >= 3:
            xs = [circ.bbox[0]] + [detections[i].bbox[0] for i in line_ids]
            ys = [circ.bbox[1]] + [detections[i].bbox[1] for i in line_ids]
            x2 = [circ.bbox[0] + circ.bbox[2]] + [detections[i].bbox[0] + detections[i].bbox[2] for i in line_ids]
            y2 = [circ.bbox[1] + circ.bbox[3]] + [detections[i].bbox[1] + detections[i].bbox[3] for i in line_ids]
            extra.append(
                Detection(
                    "object",
                    "SUN",
                    0.84,
                    (min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys)),
                )
            )
            take(ci, *line_ids)

    for ci, circ in circles:
        if ci in used:
            continue
        for ri, rect in rects:
            if ri in used:
                continue
            rx, ry, rw, rh = rect.bbox
            tall = rh > rw * 1.25
            cx, cy = _center(circ.bbox)
            if tall and circ.bbox[1] + circ.bbox[3] <= ry + 18 and abs(cx - (rx + rw / 2)) < rw * 0.8:
                extra.append(
                    Detection(
                        "object",
                        "TREE",
                        0.8,
                        (
                            min(circ.bbox[0], rx),
                            min(circ.bbox[1], ry),
                            max(circ.bbox[0] + circ.bbox[2], rx + rw) - min(circ.bbox[0], rx),
                            max(circ.bbox[1] + circ.bbox[3], ry + rh) - min(circ.bbox[1], ry),
                        ),
                    )
                )
                take(ci, ri)

    for blob in blobs:
        peri = cv2.arcLength(blob.contour, True)
        approx = cv2.approxPolyDP(blob.contour, 0.04 * peri, True)
        pts = approx.reshape(-1, 2)
        if blob.holes >= 2 and len(pts) >= 5:
            peak = pts[np.argmin(pts[:, 1])]
            bottom = pts[pts[:, 1] >= np.percentile(pts[:, 1], 70)]
            if len(bottom) >= 2:
                span = float(bottom[:, 0].max() - bottom[:, 0].min())
                peaked = abs(peak[0] - blob.cx) < 0.28 * max(blob.bbox[2], 1)
                wide_base = span > 0.55 * blob.bbox[2]
                if peaked and wide_base and blob.bbox[3] > blob.bbox[2] * 0.7:
                    extra.append(Detection("object", "HOUSE", 0.82, blob.bbox))

    for blob in blobs:
        if blob.holes < 2 or blob.bbox[2] < 50 or blob.bbox[3] < 50:
            continue
        aspect = blob.bbox[2] / max(blob.bbox[3], 1)
        if not (0.75 <= aspect <= 1.35):
            continue
        taken = any(
            (d.kind in ("letter", "digit") and d.label in ("B", "8") and d.bbox == blob.bbox)
            or (d.kind == "object" and d.label == "HOUSE")
            for d in detections
        )
        if not taken:
            extra.append(Detection("object", "SMILEY", 0.78, blob.bbox))

    kept = [d for i, d in enumerate(detections) if i not in used]
    return kept + extra


def _group_text(detections: Sequence[Detection]) -> List[Detection]:
    chars = [d for d in detections if d.kind in ("letter", "digit")]
    chars = sorted(chars, key=lambda d: (round(d.bbox[1] / 28.0), d.bbox[0]))
    groups: List[List[Detection]] = []
    current: List[Detection] = []
    for item in chars:
        if not current:
            current = [item]
            continue
        prev = current[-1]
        same_line = abs((item.bbox[1] + item.bbox[3] / 2) - (prev.bbox[1] + prev.bbox[3] / 2)) < max(
            18, 0.55 * max(item.bbox[3], prev.bbox[3])
        )
        gap = item.bbox[0] - (prev.bbox[0] + prev.bbox[2])
        close = gap < max(28, 0.9 * max(item.bbox[2], prev.bbox[2]))
        if same_line and close and gap > -12:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    if current:
        groups.append(current)

    extra: List[Detection] = []
    for group in groups:
        if len(group) < 2:
            continue
        label = "".join(d.label for d in group)
        x0 = min(d.bbox[0] for d in group)
        y0 = min(d.bbox[1] for d in group)
        x1 = max(d.bbox[0] + d.bbox[2] for d in group)
        y1 = max(d.bbox[1] + d.bbox[3] for d in group)
        score = float(np.mean([d.score for d in group]))
        if all(d.kind == "digit" for d in group):
            extra.append(Detection("number", label, score, (x0, y0, x1 - x0, y1 - y0)))
        elif any(d.kind == "letter" for d in group):
            extra.append(Detection("word", label, score, (x0, y0, x1 - x0, y1 - y0)))
    return extra


def _contained(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return ix >= ox - 4 and iy >= oy - 4 and ix + iw <= ox + ow + 4 and iy + ih <= oy + oh + 4


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> None:
    covers = [d for d in detections if d.kind in ("word", "number", "object")]
    for det in detections:
        if det.kind in ("letter", "digit") and any(_contained(det.bbox, other.bbox) for other in covers):
            continue
        x, y, w, h = det.bbox
        color = KIND_COLOR.get(det.kind, (20, 210, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (8, 6, 4), 4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        caption = f"{det.kind.upper()}: {det.label}"
        tx, ty = x, max(24, y - 8)
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(frame, (tx, ty - th - 8), (tx + tw + 12, ty + 6), (12, 8, 6), -1)
        cv2.putText(frame, caption, (tx + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (8, 6, 4), 4, cv2.LINE_AA)
        cv2.putText(
            frame,
            caption,
            (tx + 5, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
