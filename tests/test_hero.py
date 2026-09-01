import numpy as np

from hero.face_mapper import FaceMapper
from hero.hero_manager import HeroManager
from hero.mask_renderer import render_mask
from hero.transformation import ASSEMBLE, CHARGE, HERO, SCAN, Transformation


def _mesh(width=640.0, height=480.0):
    pts = np.zeros((478, 2), dtype=np.float32)
    cx, cy = width * 0.5, height * 0.45
    for i in range(478):
        ang = i * 0.13
        pts[i] = (cx + np.cos(ang) * 70, cy + np.sin(ang) * 90)
    pts[10] = (cx, cy - 95)
    pts[9] = (cx, cy - 70)
    pts[152] = (cx, cy + 110)
    pts[234] = (cx - 80, cy)
    pts[454] = (cx + 80, cy)
    pts[1] = (cx, cy + 10)
    pts[33] = (cx - 28, cy - 8)
    pts[133] = (cx - 12, cy - 8)
    pts[362] = (cx + 12, cy - 8)
    pts[263] = (cx + 28, cy - 8)
    for idx in (7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246):
        pts[idx] = (cx - 20 + (idx % 5) * 3, cy - 6 + (idx % 3))
    for idx in (382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398):
        pts[idx] = (cx + 20 + (idx % 5) * 3, cy - 6 + (idx % 3))
    oval = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]
    for i, idx in enumerate(oval):
        ang = i / len(oval) * 2 * np.pi - np.pi / 2
        pts[idx] = (cx + np.cos(ang) * 85, cy + np.sin(ang) * 110)
    lips = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ]
    for i, idx in enumerate(lips):
        ang = i / len(lips) * 2 * np.pi
        pts[idx] = (cx + np.cos(ang) * 22, cy + 38 + np.sin(ang) * 8)
    return pts


def test_mapper_finds_pose():
    face = FaceMapper(smooth=1.0).map(_mesh())
    assert face.present
    assert face.scale > 40
    assert face.oval is not None and len(face.oval) > 8


def test_transformation_reaches_hero():
    xform = Transformation()
    xform.scan_s = xform.charge_s = xform.assemble_s = 0.1
    xform.begin()
    assert xform.stage == SCAN
    xform.update(0.12, True)
    assert xform.stage == CHARGE
    xform.update(0.12, True)
    assert xform.stage == ASSEMBLE
    xform.update(0.12, True)
    assert xform.stage == HERO
    assert xform.mask_alpha == 1.0


def test_manager_renders_without_crash():
    manager = HeroManager()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = manager.render(frame, [])
    assert out.shape == frame.shape
    faces = [{"points": _mesh(320, 240), "bbox": (80, 40, 240, 200)}]
    out2 = manager.render(frame, faces)
    assert out2.shape == frame.shape
    left = _mesh(320, 240)
    right = _mesh(320, 240)
    left[:, 0] -= 70
    right[:, 0] += 70
    two = [
        {"points": left, "bbox": (20, 40, 140, 200)},
        {"points": right, "bbox": (160, 40, 280, 200)},
    ]
    out3 = manager.render(frame, two)
    assert out3.shape == frame.shape
    assert manager.face_count == 2
    manager.set_hero("web")
    assert manager.style == "web"


def test_mask_draws_pixels():
    from hero.face_mapper import FaceMapper

    layer = np.zeros((240, 320, 3), dtype=np.uint8)
    face = FaceMapper(smooth=1.0).map(_mesh(320, 240))
    render_mask(layer, face, "tech", 1.0, HERO)
    assert layer.sum() > 0
    iron = np.zeros((240, 320, 3), dtype=np.uint8)
    render_mask(iron, face, "iron", 1.0, HERO)
    assert iron.sum() > 0
