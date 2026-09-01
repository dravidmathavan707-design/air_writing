from body.body_detector import (
    FaceGestureCycle,
    any_hand_is_fist,
    any_hand_is_open,
    gesture_commits_face_draw,
    gesture_starts_face_draw,
    hand_is_fist,
    hand_is_open,
)


def make_hand(wrist=(150, 260), palm=(170, 220), open_fingers=False):
    points = [wrist] + [(150, 235)] * 4

    if open_fingers:
        index_mcp = (170, 220)
        index_pip = (175, 190)
        index_tip = (185, 150)

        middle_mcp = (185, 220)
        middle_pip = (195, 185)
        middle_tip = (210, 140)

        ring_mcp = (200, 220)
        ring_pip = (215, 190)
        ring_tip = (230, 160)

        pinky_mcp = (215, 220)
        pinky_pip = (228, 200)
        pinky_tip = (245, 175)
    else:
        index_mcp = (170, 220)
        index_pip = (170, 210)
        index_tip = (175, 210)

        middle_mcp = (185, 220)
        middle_pip = (185, 210)
        middle_tip = (190, 210)

        ring_mcp = (200, 220)
        ring_pip = (200, 210)
        ring_tip = (205, 210)

        pinky_mcp = (215, 220)
        pinky_pip = (216, 212)
        pinky_tip = (220, 214)

    hand = [
        wrist,
        (145, 235),
        (155, 235),
        (165, 235),
        (175, 235),
        index_mcp,
        index_pip,
        (180, 200),
        index_tip,
        middle_mcp,
        middle_pip,
        (200, 200),
        middle_tip,
        ring_mcp,
        ring_pip,
        (220, 200),
        ring_tip,
        pinky_mcp,
        pinky_pip,
        (235, 200),
        pinky_tip,
    ]
    return hand


class FakeHand:
    def __init__(self, open_fingers):
        self.landmarks = make_hand(open_fingers=open_fingers)


def test_open_hand_starts_face_scan():
    opened = make_hand(open_fingers=True)
    closed = make_hand(open_fingers=False)
    assert gesture_starts_face_draw(opened, None) is True
    assert gesture_starts_face_draw(None, closed) is False
    assert any_hand_is_open([FakeHand(True)]) is True
    assert any_hand_is_open([FakeHand(False)]) is False


def test_after_draw_requires_close_then_open_before_next_scan():
    cycle = FaceGestureCycle(hold_frames=2)

    assert cycle.update(True, False) is None
    assert cycle.update(True, False) == "scan"
    assert cycle.update(True, False) == "scan"
    assert cycle.update(False, True) is None
    assert cycle.update(False, True) == "draw"
    assert cycle.state == FaceGestureCycle.AWAIT_CLOSE

    # Opening immediately must not start a new scan.
    assert cycle.update(True, False) is None
    assert cycle.state == FaceGestureCycle.AWAIT_CLOSE

    assert cycle.update(False, True) is None
    assert cycle.update(False, True) == "need_open"
    assert cycle.update(True, False) is None
    assert cycle.update(True, False) == "ready"
    assert cycle.state == FaceGestureCycle.IDLE

    assert cycle.update(True, False) is None
    assert cycle.update(True, False) == "scan"


def test_closed_hand_commits_face_draw():
    opened = make_hand(open_fingers=True)
    closed = make_hand(open_fingers=False)
    assert gesture_commits_face_draw(closed, None) is True
    assert gesture_commits_face_draw(opened, None) is False
    assert any_hand_is_fist([FakeHand(False)]) is True
    assert hand_is_open(opened)
    assert hand_is_fist(closed)
