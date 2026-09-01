from hand.hand_detector import HandPoint, is_index_extended, pointing_score


def _pt(x, y):
    return (float(x), float(y))


def make_hand(pose="point"):
    """Build a 21-point hand in pixel space.

    pose: point | fist | open | peace | sideways_point
    """
    wrist = _pt(200, 360)
    points = [_pt(0, 0)] * 21
    points[0] = wrist

    # Thumb (loosely out of the way)
    points[1], points[2], points[3], points[4] = (
        _pt(170, 330),
        _pt(150, 300),
        _pt(140, 275),
        _pt(130, 255),
    )

    def finger(mcp, pip, dip, tip, mcp_xy, direction, extended):
        mx, my = mcp_xy
        dx, dy = direction
        points[mcp] = _pt(mx, my)
        if extended:
            points[pip] = _pt(mx + dx * 28, my + dy * 28)
            points[dip] = _pt(mx + dx * 52, my + dy * 52)
            points[tip] = _pt(mx + dx * 78, my + dy * 78)
        else:
            points[pip] = _pt(mx + dx * 10, my + dy * 10)
            points[dip] = _pt(mx - dx * 6, my - dy * 4)
            points[tip] = _pt(mx - dx * 18, my + 8)

    up = (0.05, -1.0)
    left = (-1.0, 0.05)

    if pose == "point":
        finger(5, 6, 7, 8, (185, 300), up, True)
        finger(9, 10, 11, 12, (200, 300), up, False)
        finger(13, 14, 15, 16, (215, 302), up, False)
        finger(17, 18, 19, 20, (230, 305), up, False)
    elif pose == "sideways_point":
        finger(5, 6, 7, 8, (185, 300), left, True)
        finger(9, 10, 11, 12, (200, 300), left, False)
        finger(13, 14, 15, 16, (215, 302), left, False)
        finger(17, 18, 19, 20, (230, 305), left, False)
    elif pose == "open":
        finger(5, 6, 7, 8, (185, 300), up, True)
        finger(9, 10, 11, 12, (200, 300), up, True)
        finger(13, 14, 15, 16, (215, 302), up, True)
        finger(17, 18, 19, 20, (230, 305), up, True)
    elif pose == "peace":
        finger(5, 6, 7, 8, (185, 300), up, True)
        finger(9, 10, 11, 12, (200, 300), up, True)
        finger(13, 14, 15, 16, (215, 302), up, False)
        finger(17, 18, 19, 20, (230, 305), up, False)
    else:  # fist
        finger(5, 6, 7, 8, (185, 300), up, False)
        finger(9, 10, 11, 12, (200, 300), up, False)
        finger(13, 14, 15, 16, (215, 302), up, False)
        finger(17, 18, 19, 20, (230, 305), up, False)

    return points


def test_pointing_index_is_detected_up_and_sideways():
    assert is_index_extended(make_hand("point"))
    assert is_index_extended(make_hand("sideways_point"))
    assert pointing_score(make_hand("point")) > pointing_score(make_hand("fist"))


def test_fist_and_open_palm_do_not_draw():
    assert not is_index_extended(make_hand("fist"))
    assert not is_index_extended(make_hand("open"))


def test_peace_sign_does_not_start_drawing():
    assert not is_index_extended(make_hand("peace"))


def test_draw_gate_needs_stable_frames():
    from hand.hand_detector import HandDetector

    detector = HandDetector.__new__(HandDetector)
    detector.enter_score = 0.52
    detector.exit_score = 0.38
    detector.enter_frames = 2
    detector.exit_frames = 4
    detector.lost_frames = 6
    detector._drawing = False
    detector._streak = 0
    detector._lost = 0
    detector._locked_label = None

    pointing = HandPoint(0, "Right", make_hand("point"), (0, 0), score=pointing_score(make_hand("point")))
    fist = HandPoint(0, "Right", make_hand("fist"), (0, 0), score=pointing_score(make_hand("fist")))

    assert detector.update_draw_state(pointing) is False
    assert detector.update_draw_state(pointing) is True
    assert detector.update_draw_state(fist) is True
    assert detector.update_draw_state(fist) is True
    assert detector.update_draw_state(fist) is True
    assert detector.update_draw_state(fist) is False


def test_brief_hand_loss_keeps_stroke_alive():
    from hand.hand_detector import HandDetector

    detector = HandDetector.__new__(HandDetector)
    detector.enter_score = 0.52
    detector.exit_score = 0.38
    detector.enter_frames = 1
    detector.exit_frames = 4
    detector.lost_frames = 6
    detector._drawing = False
    detector._streak = 0
    detector._lost = 0
    detector._locked_label = None

    pointing = HandPoint(0, "Right", make_hand("point"), (0, 0), score=pointing_score(make_hand("point")))
    assert detector.update_draw_state(pointing) is True
    assert detector.update_draw_state(None) is True
    assert detector.update_draw_state(None) is True
    for _ in range(6):
        detector.update_draw_state(None)
    assert detector.update_draw_state(None) is False


def test_camera_facing_point_still_scores_with_z():
    points = make_hand("point")
    foreshortened = []
    for i, point in enumerate(points):
        if i in (6, 7, 8):
            foreshortened.append((point[0], point[1] + (8 - i) * 4, -40.0 - i * 8))
        else:
            foreshortened.append((point[0], point[1], 0.0))
    assert pointing_score(foreshortened) > pointing_score(make_hand("fist"))

