from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp

Point = Tuple[int, int]


def distance(point_a: Point, point_b: Point) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def hand_is_open(landmarks: Optional[Sequence[Point]]) -> bool:
    if not landmarks or len(landmarks) < 21:
        return False

    wrist = landmarks[0]
    palm_center = landmarks[9]
    palm_distance = distance(wrist, palm_center)
    if palm_distance == 0:
        return False

    finger_tips = [
        landmarks[4],
        landmarks[8],
        landmarks[12],
        landmarks[16],
        landmarks[20],
    ]
    average_tip_distance = sum(
        distance(palm_center, tip) for tip in finger_tips
    ) / len(finger_tips)

    return average_tip_distance > palm_distance * 0.9


def hand_is_fist(landmarks: Optional[Sequence[Point]]) -> bool:
    if not landmarks or len(landmarks) < 21:
        return False

    wrist = landmarks[0]
    palm_center = landmarks[9]
    palm_distance = distance(wrist, palm_center)
    if palm_distance == 0:
        return False

    finger_tips = [
        landmarks[4],
        landmarks[8],
        landmarks[12],
        landmarks[16],
        landmarks[20],
    ]
    average_tip_distance = sum(
        distance(palm_center, tip) for tip in finger_tips
    ) / len(finger_tips)

    return average_tip_distance < palm_distance * 0.55


def any_hand_is_open(hands: Optional[Sequence]) -> bool:
    if not hands:
        return False
    for hand in hands:
        landmarks = getattr(hand, "landmarks", hand)
        if hand_is_open(landmarks):
            return True
    return False


def any_hand_is_fist(hands: Optional[Sequence]) -> bool:
    if not hands:
        return False
    for hand in hands:
        landmarks = getattr(hand, "landmarks", hand)
        if hand_is_fist(landmarks):
            return True
    return False


def gesture_starts_face_draw(
    left_hand: Optional[Sequence[Point]],
    right_hand: Optional[Sequence[Point]],
) -> bool:
    """Open palm on either hand (scan faces)."""
    return hand_is_open(left_hand) or hand_is_open(right_hand)


def gesture_commits_face_draw(
    left_hand: Optional[Sequence[Point]],
    right_hand: Optional[Sequence[Point]],
) -> bool:
    """Closed fist on either hand (draw the counted faces)."""
    return hand_is_fist(left_hand) or hand_is_fist(right_hand)


def gesture_resets_face_draw(
    left_hand: Optional[Sequence[Point]],
    right_hand: Optional[Sequence[Point]],
) -> bool:
    return False


class FaceGestureCycle:
    """
    Open palm → scan faces
    Close fist → draw those faces
    Then close again (or keep fist) → open palm → ready for a new scan

    After a draw, opening the hand does not immediately rescan.
    """

    IDLE = "idle"
    SCANNING = "scanning"
    AWAIT_CLOSE = "await_close"
    AWAIT_OPEN = "await_open"

    def __init__(self, hold_frames: int = 3):
        self.hold_frames = hold_frames
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.open_hold = 0
        self.close_hold = 0

    def update(self, hand_open: bool, hand_closed: bool) -> Optional[str]:
        if self.state == self.IDLE:
            return self._idle(hand_open)
        if self.state == self.SCANNING:
            return self._scanning(hand_open, hand_closed)
        if self.state == self.AWAIT_CLOSE:
            return self._await_close(hand_closed)
        return self._await_open(hand_open)

    def _idle(self, hand_open: bool) -> Optional[str]:
        if hand_open:
            self.open_hold += 1
            if self.open_hold >= self.hold_frames:
                self.state = self.SCANNING
                self.open_hold = 0
                return "scan"
            return None
        self.open_hold = 0
        return None

    def _scanning(self, hand_open: bool, hand_closed: bool) -> Optional[str]:
        if hand_open:
            self.close_hold = 0
            return "scan"
        if hand_closed:
            self.close_hold += 1
            if self.close_hold >= self.hold_frames:
                self.state = self.AWAIT_CLOSE
                self.close_hold = 0
                return "draw"
            return None
        self.close_hold = 0
        return None

    def _await_close(self, hand_closed: bool) -> Optional[str]:
        if hand_closed:
            self.close_hold += 1
            if self.close_hold >= self.hold_frames:
                self.state = self.AWAIT_OPEN
                self.close_hold = 0
                return "need_open"
            return None
        self.close_hold = 0
        return None

    def _await_open(self, hand_open: bool) -> Optional[str]:
        if hand_open:
            self.open_hold += 1
            if self.open_hold >= self.hold_frames:
                self.state = self.IDLE
                self.open_hold = 0
                return "ready"
            return None
        self.open_hold = 0
        return None


def pair_hand_landmarks(hands: Sequence) -> Tuple[Optional[Sequence], Optional[Sequence]]:
    """Map MediaPipe hands to left/right by label, then by x position."""
    if not hands:
        return None, None
    left = None
    right = None
    unlabeled = []
    for hand in hands:
        label = str(getattr(hand, "label", "")).lower()
        landmarks = getattr(hand, "landmarks", hand)
        if label.startswith("left"):
            left = landmarks
        elif label.startswith("right"):
            right = landmarks
        else:
            unlabeled.append(landmarks)
    leftover = [item for item in unlabeled if item is not None]
    if left is None and leftover:
        left = leftover.pop(0)
    if right is None and leftover:
        right = leftover.pop(0)
    if (left is None or right is None) and len(hands) >= 2:
        ordered = sorted(
            hands,
            key=lambda hand: float(getattr(hand, "landmarks", hand)[0][0]),
        )
        left = getattr(ordered[0], "landmarks", ordered[0])
        right = getattr(ordered[1], "landmarks", ordered[1])
    return left, right


class BodyDetector:

    def __init__(
        self,
        detection_confidence=0.5,
        tracking_confidence=0.5,
    ):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min(detection_confidence, 0.4),
            min_tracking_confidence=min(tracking_confidence, 0.4),
        )

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        return results

    def get_right_wrist(
        self,
        results,
        frame_shape,
    ):

        if not results.pose_landmarks:
            return None

        landmark = (
            results.pose_landmarks.landmark[
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            ]
        )

        if landmark.visibility < 0.20:
            return None

        height, width = frame_shape[:2]

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        x = max(
            0,
            min(width - 1, x)
        )

        y = max(
            0,
            min(height - 1, y)
        )

        return (x, y)

    def close(self):

        self.pose.close()
