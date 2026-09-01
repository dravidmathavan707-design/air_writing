"""Armored gold/crimson faceplate — dense mesh warp + Snapchat-style blend."""

from __future__ import annotations

import cv2
import numpy as np

from hero.dense_mesh import GROW, MESH_IDX, MESH_NP, OVAL_SET, TEX, TRIS, UV
from hero.face_mapper import HeroFace
from hero.shape_warp import warp_texture

NAME = "iron"
FILL = (28, 150, 210)
EDGE = (30, 200, 240)
LENS = (60, 255, 255)
ACCENT = (18, 28, 170)
GLOW = (40, 240, 255)
SHAPE_MASK = True
SNAP_BLEND = True

_TEXTURE = None
_TEX_ALPHA = None
_TEX_RGBA = None


def _noise(h: int, w: int, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 1, (h, 1)).astype(np.float32)
    grain = np.repeat(grain, w, axis=1)
    return cv2.GaussianBlur(grain, (1, 31), 0)


def _fill(img, pts, color):
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)


def _stroke(img, pts, color, thickness=2, closed=True):
    cv2.polylines(img, [np.array(pts, np.int32)], closed, color, thickness, cv2.LINE_AA)


def _texture():
    global _TEXTURE, _TEX_ALPHA, _TEX_RGBA
    if _TEX_RGBA is not None:
        return _TEX_RGBA

    s = TEX
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    cx = cy = s * 0.5
    nx = (xx - cx) / (s * 0.38)
    ny = (yy - cy) / (s * 0.46)
    r = np.sqrt(nx * nx * 0.92 + ny * ny)

    gold = np.array([32, 178, 236], dtype=np.float32)
    gold_hi = np.array([140, 236, 255], dtype=np.float32)
    gold_lo = np.array([8, 78, 118], dtype=np.float32)
    crimson = np.array([18, 22, 186], dtype=np.float32)
    crimson_d = np.array([10, 10, 78], dtype=np.float32)
    steel = np.array([28, 32, 36], dtype=np.float32)
    seam = (12, 62, 92)

    cheek = np.clip(np.abs(nx) * 1.35 - 0.18 - ny * 0.12, 0, 1)[..., None]
    cheek = np.power(cheek, 0.85)
    faceplate = np.clip(1.15 - np.abs(nx) * 1.55 - np.maximum(ny - 0.15, 0) * 0.25, 0, 1)[..., None]
    jaw = np.clip((ny - 0.28) * 2.4, 0, 1)[..., None]
    brow = np.clip((-ny - 0.08) * 1.8, 0, 1)[..., None]
    ao = np.clip(1.0 - np.maximum(r - 0.68, 0) * 4.2, 0, 1)[..., None]
    cavity = np.exp(-((np.abs(nx) - 0.28) ** 2 * 28 + (ny + 0.06) ** 2 * 55))[..., None]

    base = gold * (0.42 + 0.58 * faceplate)
    base = base * (1.0 - cheek * 0.92) + crimson * cheek
    base = base * (1.0 - jaw * 0.22) + gold * jaw * 0.35 + gold_lo * jaw * 0.4
    base = base * (1.0 - brow * 0.18) + gold_lo * brow * 0.35
    base = base * (1.0 - cavity * 0.55) + steel * cavity * 0.35
    brush = _noise(s, s)
    base = base + brush[..., None] * 9.0
    spec = np.exp(-((nx + 0.22) ** 2 * 7.5 + (ny + 0.42) ** 2 * 12))[..., None]
    rim = np.exp(-((r - 0.78) ** 2 * 40))[..., None]
    base = base + gold_hi * spec * 0.62 + gold_hi * rim * 0.18
    base = base * (0.62 + 0.38 * ao)
    img = np.clip(base, 0, 255).astype(np.uint8)

    # Gold faceplate catch-lights and crimson cheek plates.
    _fill(img, [(256, 22), (118, 168), (148, 178), (256, 78), (364, 178), (394, 168)], (16, 92, 132))
    _stroke(img, [(256, 28), (132, 164), (256, 84), (380, 164)], (48, 170, 210), 2, False)
    _fill(img, [(40, 120), (150, 96), (168, 210), (72, 268), (36, 210)], (14, 16, 150))
    _fill(img, [(472, 120), (362, 96), (344, 210), (440, 268), (476, 210)], (14, 16, 150))
    _stroke(img, [(150, 96), (168, 210), (72, 268)], (40, 150, 190), 2, False)
    _stroke(img, [(362, 96), (344, 210), (440, 268)], (40, 150, 190), 2, False)

    # Nose / brow ridge between the visors.
    _fill(img, [(244, 88), (268, 88), (282, 250), (256, 278), (230, 250)], (108, 214, 248))
    cv2.line(img, (256, 90), (256, 268), seam, 2, cv2.LINE_AA)
    cv2.line(img, (248, 200), (256, 268), seam, 1, cv2.LINE_AA)
    cv2.line(img, (264, 200), (256, 268), seam, 1, cv2.LINE_AA)

    # Cheek vents.
    for pts in (
        [(86, 268), (150, 286), (144, 304), (80, 284)],
        [(426, 268), (362, 286), (368, 304), (432, 284)],
        [(92, 292), (156, 310), (150, 322), (86, 304)],
        [(420, 292), (356, 310), (362, 322), (426, 304)],
    ):
        _fill(img, pts, tuple(int(c) for c in crimson_d))
        _stroke(img, pts, seam, 1)

    # Iconic angled visor slits, aligned to Face Mesh eye UVs (~y 220-252).
    left_eye = [(132, 214), (228, 226), (226, 250), (128, 256)]
    right_eye = [(284, 226), (380, 214), (384, 256), (286, 250)]
    _fill(img, left_eye, (4, 6, 8))
    _fill(img, right_eye, (4, 6, 8))
    inner_l = [(142, 222), (220, 232), (218, 246), (140, 248)]
    inner_r = [(292, 232), (370, 222), (372, 248), (294, 246)]
    glow = np.zeros_like(img)
    _fill(glow, inner_l, (80, 255, 255))
    _fill(glow, inner_r, (80, 255, 255))
    glow = cv2.GaussianBlur(glow, (0, 0), 5.0)
    img = cv2.addWeighted(img, 1.0, glow, 1.05, 0)
    _fill(img, inner_l, (230, 255, 255))
    _fill(img, inner_r, (230, 255, 255))
    core_l = [(168, 228), (210, 234), (208, 242), (166, 244)]
    core_r = [(302, 234), (344, 228), (346, 244), (304, 242)]
    _fill(img, core_l, (255, 255, 255))
    _fill(img, core_r, (255, 255, 255))
    _stroke(img, left_eye, (20, 190, 230), 2)
    _stroke(img, right_eye, (20, 190, 230), 2)

    # Mouth slot and chin plate.
    _fill(img, [(208, 328), (304, 328), (294, 348), (218, 348)], tuple(int(c) for c in steel))
    cv2.line(img, (222, 338), (290, 338), (70, 78, 84), 2, cv2.LINE_AA)
    _fill(img, [(168, 352), (256, 478), (344, 352), (308, 372), (256, 432), (204, 372)], (22, 110, 150))
    _stroke(img, [(176, 356), (256, 460), (336, 356)], seam, 2, False)
    _stroke(img, [(204, 372), (256, 432), (308, 372)], (60, 190, 220), 2, False)

    # Forehead gold cap over the chevron.
    _fill(img, [(256, 18), (186, 92), (256, 58), (326, 92)], (96, 210, 245))

    alpha = np.clip(1.18 - r * 0.88, 0, 1)
    alpha = np.power(np.clip(alpha, 0, 1), 0.62)
    alpha = np.clip(alpha * (0.35 + 0.65 * ao[..., 0]), 0, 1)
    _TEXTURE = img
    _TEX_ALPHA = (alpha * 255).astype(np.uint8)
    _TEX_RGBA = np.dstack([_TEXTURE, _TEX_ALPHA])
    return _TEX_RGBA


def _texture():
    global _TEXTURE, _TEX_ALPHA, _TEX_RGBA
    if _TEX_RGBA is not None:
        return _TEX_RGBA

    s = TEX
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    cx = cy = s * 0.5
    nx = (xx - cx) / (s * 0.38)
    ny = (yy - cy) / (s * 0.46)
    r = np.sqrt(nx * nx + ny * ny)

    gold = np.array([48, 188, 226], dtype=np.float32)
    gold_hi = np.array([110, 230, 255], dtype=np.float32)
    gold_lo = np.array([18, 96, 128], dtype=np.float32)
    crimson = np.array([28, 36, 168], dtype=np.float32)
    steel = np.array([38, 42, 46], dtype=np.float32)

    cheek = np.clip(np.abs(nx) * 1.15 - 0.22 - ny * 0.08, 0, 1)[..., None]
    center = np.clip(1.0 - r * 0.85, 0, 1)[..., None]
    jaw = np.clip((ny - 0.35) * 2.2, 0, 1)[..., None]
    brow = np.clip((-ny - 0.15) * 1.6, 0, 1)[..., None]
    ao = np.clip(1.0 - np.maximum(r - 0.72, 0) * 3.5, 0, 1)[..., None]

    base = gold * (0.35 + 0.65 * center)
    base = base * (1.0 - cheek * 0.82) + crimson * cheek
    base = base * (1.0 - jaw * 0.35) + gold_lo * jaw * 0.55 + base * (1 - jaw * 0.2)
    base = base * (1.0 - brow * 0.25) + gold_lo * brow * 0.4
    brush = _noise(s, s)
    base = base + brush[..., None] * 7.0
    spec = np.exp(-((nx + 0.18) ** 2 * 8 + (ny + 0.35) ** 2 * 14))[..., None]
    base = base + gold_hi * spec * 0.55
    base = base * (0.55 + 0.45 * ao)

    img = np.clip(base, 0, 255).astype(np.uint8)

    def poly(pts, color, closed=True, thickness=-1):
        arr = np.array(pts, np.int32)
        if thickness < 0:
            cv2.fillConvexPoly(img, arr, color)
        else:
            cv2.polylines(img, [arr], closed, color, thickness, cv2.LINE_AA)

    chevron = [(256, 36), (168, 148), (196, 156), (256, 88), (316, 156), (344, 148)]
    poly(chevron, (22, 88, 118))
    cv2.polylines(img, [np.array(chevron, np.int32)], True, (40, 160, 200), 2, cv2.LINE_AA)

    ridge = [(244, 92), (268, 92), (280, 268), (256, 292), (232, 268)]
    poly(ridge, (96, 214, 246))
    cv2.line(img, (256, 96), (256, 280), (24, 90, 120), 2, cv2.LINE_AA)

    for x0, x1 in ((118, 148), (364, 394)):
        cv2.line(img, (x0, 210), (x1, 248), (16, 18, 80), 3, cv2.LINE_AA)
        cv2.line(img, (x0, 228), (x1, 266), (16, 18, 80), 3, cv2.LINE_AA)

    left_eye = np.array([[96, 186], [214, 194], [210, 226], [102, 232]], np.int32)
    right_eye = np.array([[298, 194], [416, 186], [410, 232], [302, 226]], np.int32)
    cv2.fillConvexPoly(img, left_eye, (6, 8, 10))
    cv2.fillConvexPoly(img, right_eye, (6, 8, 10))
    inner_l = np.array([[108, 196], [202, 202], [198, 220], [112, 222]], np.int32)
    inner_r = np.array([[310, 202], [404, 196], [400, 222], [314, 220]], np.int32)
    glow = np.zeros_like(img)
    cv2.fillConvexPoly(glow, inner_l, (50, 255, 255))
    cv2.fillConvexPoly(glow, inner_r, (50, 255, 255))
    glow = cv2.GaussianBlur(glow, (0, 0), 3.5)
    img = cv2.addWeighted(img, 1.0, glow, 0.85, 0)
    cv2.fillConvexPoly(img, inner_l, (90, 255, 255))
    cv2.fillConvexPoly(img, inner_r, (90, 255, 255))
    cv2.polylines(img, [left_eye, right_eye], True, (30, 200, 230), 2, cv2.LINE_AA)

    mouth = np.array([[214, 338], [298, 338], [292, 354], [220, 354]], np.int32)
    cv2.fillConvexPoly(img, mouth, tuple(int(c) for c in steel))
    cv2.line(img, (226, 346), (286, 346), (88, 92, 96), 1, cv2.LINE_AA)

    chin = np.array([[176, 360], [256, 468], [336, 360], [308, 378], [256, 430], [204, 378]], np.int32)
    cv2.polylines(img, [chin], True, (30, 70, 90), 2, cv2.LINE_AA)

    alpha = np.clip(1.15 - r * 0.92, 0, 1)
    alpha = np.power(alpha, 0.75)
    _TEXTURE = img
    _TEX_ALPHA = (alpha * 255).astype(np.uint8)
    _TEX_RGBA = np.dstack([_TEXTURE, _TEX_ALPHA])
    return _TEX_RGBA


def _dest_points(face: HeroFace) -> np.ndarray | None:
    pts = face.points
    if pts is None or len(pts) <= int(MESH_NP.max()):
        return None
    center = np.asarray(face.center, dtype=np.float32)
    p = pts[MESH_NP]
    return center + (p - center) * GROW[:, None]


def _face_roi(face: HeroFace, shape):
    h, w = shape[:2]
    x, y, bw, bh = face.bbox
    pad = int(max(24, face.scale * 0.5))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _render_roi(face: HeroFace, shape):
    dest = _dest_points(face)
    box = _face_roi(face, shape)
    if dest is None or box is None:
        return None
    x0, y0, x1, y1 = box
    local = dest - np.array([x0, y0], dtype=np.float32)
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    warp_texture(rgba, _texture(), UV, local, TRIS)
    color = rgba[..., :3]
    alpha = rgba[..., 3]
    oval_i = [i for i, idx in enumerate(MESH_IDX) if idx in OVAL_SET]
    if oval_i:
        hull = local[oval_i].astype(np.int32)
        matte = np.zeros(alpha.shape, dtype=np.uint8)
        cv2.fillConvexPoly(matte, cv2.convexHull(hull), 255)
        k = max(7, min(11, int(face.scale * 0.12) | 1))
        matte = cv2.GaussianBlur(matte, (k, k), 0)
        alpha = cv2.min(alpha, matte)
    return x0, y0, x1, y1, color, alpha


def _render_overlay(shape, face: HeroFace):
    pack = _render_roi(face, shape)
    if pack is None:
        return None, None
    x0, y0, x1, y1, color, alpha = pack
    full_c = np.zeros(shape, dtype=np.uint8)
    full_a = np.zeros(shape[:2], dtype=np.uint8)
    full_c[y0:y1, x0:x1] = color
    full_a[y0:y1, x0:x1] = alpha
    return full_c, full_a


def decorate(layer: np.ndarray, face: HeroFace, alpha: float = 1.0) -> None:
    color, matte = _render_overlay(layer.shape, face)
    if color is None:
        return
    a = (matte.astype(np.float32) / 255.0) * float(alpha)
    layer[:] = np.clip(layer.astype(np.float32) + color.astype(np.float32) * a[..., None], 0, 255).astype(
        np.uint8
    )


def composite(frame: np.ndarray, face: HeroFace, alpha: float) -> np.ndarray:
    if alpha <= 0.01:
        return frame
    pack = _render_roi(face, frame.shape)
    if pack is None:
        return frame
    x0, y0, x1, y1, color, matte = pack
    a = (matte.astype(np.float32) * (float(alpha) / 255.0))[..., None]
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    luma = float(np.mean(gray)) / 255.0
    light = 0.78 + 0.38 * luma
    lit = color.astype(np.float32) * light
    yaw = (face.nose[0] - face.center[0]) / max(face.scale, 1.0)
    xs = np.arange(x1 - x0, dtype=np.float32)
    side = np.clip((xs - (face.center[0] - x0)) / max(face.scale * 0.85, 1.0), -1.0, 1.0)
    lit *= (1.0 - 0.22 * np.clip(yaw * 2.2, -1.0, 1.0) * side)[None, :, None]
    b, g, _r = cv2.split(color)
    visor = np.clip((g.astype(np.float32) + b.astype(np.float32)) * 0.5 - _r.astype(np.float32) * 0.65, 0, 255)
    visor = cv2.GaussianBlur(visor, (0, 0), 3.2)
    bloom = np.dstack([visor, visor, visor * 0.92])
    lit = np.clip(lit + bloom * 0.55, 0, 255)
    spec_c = (
        int(face.center[0] - x0 - yaw * face.scale * 0.28),
        int(face.center[1] - y0 - face.scale * 0.32),
    )
    spec = np.zeros_like(roi)
    cv2.ellipse(
        spec,
        spec_c,
        (max(10, int(face.scale * 0.20)), max(6, int(face.scale * 0.09))),
        np.degrees(face.angle),
        0,
        360,
        (160, 230, 255),
        -1,
        cv2.LINE_AA,
    )
    spec = cv2.GaussianBlur(spec, (0, 0), 6)
    lit = np.clip(lit + spec.astype(np.float32) * a * 0.28, 0, 255)
    blended = roi.astype(np.float32) * (1.0 - a) + lit * a
    frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame
