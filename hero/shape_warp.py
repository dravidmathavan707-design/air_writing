"""Warp a canonical face texture onto live Face Mesh triangles (shape-mask)."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def warp_texture(
    overlay: np.ndarray,
    texture: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    triangles: Sequence[tuple],
) -> None:
    h, w = overlay.shape[:2]
    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)
    th, tw = texture.shape[:2]
    for tri in triangles:
        i0, i1, i2 = tri
        s = src_pts[[i0, i1, i2]]
        d = dst_pts[[i0, i1, i2]]
        if not np.isfinite(d).all():
            continue
        area = abs(
            (d[1, 0] - d[0, 0]) * (d[2, 1] - d[0, 1])
            - (d[2, 0] - d[0, 0]) * (d[1, 1] - d[0, 1])
        )
        if area < 16:
            continue
        _warp_triangle(overlay, texture, s, d, w, h, tw, th)


def _warp_triangle(overlay, texture, src, dst, w, h, tw, th):
    x, y, bw, bh = cv2.boundingRect(dst)
    bw += 1
    bh += 1
    if bw < 2 or bh < 2:
        return
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    x = max(0, x)
    y = max(0, y)
    bw = x2 - x
    bh = y2 - y
    if bw < 2 or bh < 2:
        return

    sx, sy, sw, sh = cv2.boundingRect(src)
    sx = int(np.clip(sx, 0, tw - 2))
    sy = int(np.clip(sy, 0, th - 2))
    sw = int(np.clip(sw, 2, tw - sx))
    sh = int(np.clip(sh, 2, th - sy))

    src_off = src - (sx, sy)
    dst_off = dst - (x, y)
    try:
        matrix = cv2.getAffineTransform(
            np.asarray(src_off, dtype=np.float32),
            np.asarray(dst_off, dtype=np.float32),
        )
    except cv2.error:
        return
    patch = cv2.warpAffine(
        texture[sy : sy + sh, sx : sx + sw],
        matrix,
        (bw, bh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    mask = np.zeros((bh, bw), dtype=np.uint8)
    pts = np.round(dst_off).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, bw - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, bh - 1)
    cv2.fillConvexPoly(mask, pts, 255)
    if mask.max() == 0:
        return
    roi = overlay[y : y + bh, x : x + bw]
    hh = min(roi.shape[0], patch.shape[0], mask.shape[0])
    ww = min(roi.shape[1], patch.shape[1], mask.shape[1])
    if hh < 1 or ww < 1:
        return
    sel = mask[:hh, :ww] > 0
    if not np.any(sel):
        return
    dst = overlay[y : y + hh, x : x + ww]
    src = patch[:hh, :ww]
    if dst.ndim == 3:
        dst[sel] = src[sel]
    else:
        dst[sel] = src[sel]
