"""Hit-tests for on-screen HUD buttons."""

import cv2

import main


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def _click(x, y):
    main.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)


def _reset():
    main.app_mode = main.MODE_FINGER
    main.pending_camera_switch = False
    main.animation_manager.reset()
    main.hero_manager.reset()
    main.hero_manager.style = "iron"


def _no_overlap(boxes):
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            overlap = not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)
            assert not overlap, (a, b)


def test_visible_buttons_do_not_overlap():
    _no_overlap([item[1] for item in main.MODE_BUTTONS] + [main.CAM_BUTTON, main.FULL_BUTTON])
    _no_overlap([item[1] for item in main.HERO_BUTTONS])
    _no_overlap([item[1] for item in main.EFFECT_BUTTONS])


def test_mode_buttons_switch_mode():
    _reset()
    for mode, box, _label in main.MODE_BUTTONS:
        x, y = _center(box)
        _click(x, y)
        assert main.app_mode == mode
        # 1px inside the far corner still counts
        _click(box[2], box[3])
        assert main.app_mode == mode


def test_animation_effect_buttons():
    _reset()
    _click(*_center(main.MODE_BUTTONS[2][1]))
    assert main.app_mode == main.MODE_ANIMATION
    for name, box, _label in main.EFFECT_BUTTONS:
        _click(*_center(box))
        assert main.animation_manager.active_effect == name


def test_hero_style_buttons():
    _reset()
    _click(*_center(main.MODE_BUTTONS[4][1]))
    assert main.app_mode == main.MODE_HERO
    for name, box, _label in main.HERO_BUTTONS:
        _click(*_center(box))
        assert main.hero_manager.style == name


def test_cam_and_full_buttons():
    _reset()
    _click(*_center(main.CAM_BUTTON))
    assert main.pending_camera_switch is True
    main.pending_camera_switch = False
    before = main.fullscreen
    _click(*_center(main.FULL_BUTTON))
    assert main.fullscreen is not before
    main.fullscreen = False


def test_effect_clicks_ignored_outside_animation():
    _reset()
    main.animation_manager.set_effect("chidori")
    _click(*_center(main.EFFECT_BUTTONS[0][1]))
    assert main.app_mode == main.MODE_FINGER
    assert main.animation_manager.active_effect == "chidori"
