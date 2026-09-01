from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp

from face.landmark_smoother import OneEuroFilter


Point = Tuple[float, ...]

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass
class HandPoint:
    index: int
    label: str
    landmarks: List[Point]
    index_tip: Tuple[int, int]
    score: float = 0.0
    tip_xy: Tuple[float, float] = (0.0, 0.0)


def _xyz(point: Sequence[float]) -> Tuple[float, float, float]:
    z = float(point[2]) if len(point) > 2 else 0.0
    return float(point[0]), float(point[1]), z


def _distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    ax, ay, az = _xyz(point_a)
    bx, by, bz = _xyz(point_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _joint_angle(point_a, point_b, point_c) -> float:
    ax, ay, az = _xyz(point_a)
    bx, by, bz = _xyz(point_b)
    cx, cy, cz = _xyz(point_c)
    vector_a = (ax - bx, ay - by, az - bz)
    vector_c = (cx - bx, cy - by, cz - bz)
    length_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2 + vector_a[2] ** 2)
    length_c = math.sqrt(vector_c[0] ** 2 + vector_c[1] ** 2 + vector_c[2] ** 2)
    if length_a == 0 or length_c == 0:
        return 0.0
    cosine = (
        vector_a[0] * vector_c[0]
        + vector_a[1] * vector_c[1]
        + vector_a[2] * vector_c[2]
    ) / (length_a * length_c)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def finger_straightness(
    landmarks: Sequence[Point], mcp: int, pip: int, dip: int, tip: int
) -> float:
    """1.0 = fully extended, 0.0 = curled. Uses 3D when z is present."""
    if len(landmarks) <= tip:
        return 0.0
    chain = (
        _distance(landmarks[mcp], landmarks[pip])
        + _distance(landmarks[pip], landmarks[dip])
        + _distance(landmarks[dip], landmarks[tip])
    )
    if chain < 1.0:
        return 0.0
    straight = _distance(landmarks[mcp], landmarks[tip]) / chain
    return max(0.0, min(1.0, (straight - 0.38) / 0.56))


def pointing_score(landmarks: Sequence[Point]) -> float:
    """
    High when the index is extended and the other fingers are curled.
    Works for up, sideways, and toward-camera pointing.
    """
    if landmarks is None or len(landmarks) < 21:
        return 0.0

    index = finger_straightness(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    middle = finger_straightness(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
    ring = finger_straightness(landmarks, RING_MCP, RING_PIP, RING_DIP, RING_TIP)
    pinky = finger_straightness(landmarks, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)

    wrist = landmarks[WRIST]
    palm = max(_distance(wrist, landmarks[MIDDLE_MCP]), 1.0)
    index_reach = _distance(wrist, landmarks[INDEX_TIP]) / palm
    middle_reach = _distance(wrist, landmarks[MIDDLE_TIP]) / palm
    reach_advantage = max(0.0, min(1.0, (index_reach - middle_reach + 0.10) / 0.42))

    pip_angle = _joint_angle(
        landmarks[INDEX_MCP], landmarks[INDEX_PIP], landmarks[INDEX_TIP]
    )
    dip_angle = _joint_angle(
        landmarks[INDEX_PIP], landmarks[INDEX_DIP], landmarks[INDEX_TIP]
    )
    angle_ok = max(
        0.0,
        min(1.0, ((pip_angle - 118.0) / 48.0 + (dip_angle - 115.0) / 50.0) / 2.0),
    )

    isolation = max(
        0.0,
        min(1.0, _distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) / (palm * 0.85) - 0.12),
    )
    others_curled = ((1.0 - middle) + (1.0 - ring) + (1.0 - pinky)) / 3.0

    score = (
        0.36 * index
        + 0.24 * others_curled
        + 0.16 * reach_advantage
        + 0.12 * angle_ok
        + 0.12 * isolation
    )
    if index < 0.38:
        score *= 0.30
    if middle > 0.70 and ring > 0.70:
        score *= 0.22
    if middle > 0.68 and index > 0.55:
        score *= 0.55
    return max(0.0, min(1.0, score))


def is_index_extended(landmarks: Sequence[Point], min_score: float = 0.52) -> bool:
    return pointing_score(landmarks) >= min_score


class TipTracker:
    """One-Euro fingertip smoother with jump rejection."""

    def __init__(self):
        self._x = OneEuroFilter(min_cutoff=1.4, beta=0.012, d_cutoff=1.0)
        self._y = OneEuroFilter(min_cutoff=1.4, beta=0.012, d_cutoff=1.0)
        self.last: Optional[Tuple[float, float]] = None

    def update(self, xy: Sequence[float], timestamp: Optional[float] = None) -> Tuple[int, int]:
        t = time.perf_counter() if timestamp is None else timestamp
        x = self._x(t, float(xy[0]))
        y = self._y(t, float(xy[1]))
        if self.last is not None and _distance((x, y), self.last) > 220:
            self.reset()
            x, y = float(xy[0]), float(xy[1])
            self._x(t, x)
            self._y(t, y)
        self.last = (x, y)
        return (int(round(x)), int(round(y)))

    def reset(self):
        self._x.reset()
        self._y.reset()
        self.last = None


class HandDetector:
    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.45,
        tracking_confidence: float = 0.40,
        enter_score: float = 0.52,
        exit_score: float = 0.38,
        enter_frames: int = 2,
        exit_frames: int = 4,
        lost_frames: int = 6,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.enter_score = enter_score
        self.exit_score = exit_score
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.lost_frames = lost_frames
        self._drawing = False
        self._streak = 0
        self._lost = 0
        self._locked_label: Optional[str] = None
        self.tip_tracker = TipTracker()

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return results

    def extract(self, results, frame_shape):
        height, width = frame_shape[:2]
        hands = []
        if not results or not results.multi_hand_landmarks:
            return hands

        handedness = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            points = []
            for landmark in hand_landmarks.landmark:
                points.append(
                    (landmark.x * width, landmark.y * height, landmark.z * width)
                )

            label = "Hand"
            if i < len(handedness):
                label = handedness[i].classification[0].label

            tip = points[INDEX_TIP]
            hands.append(
                HandPoint(
                    index=i,
                    label=label,
                    landmarks=points,
                    index_tip=(int(round(tip[0])), int(round(tip[1]))),
                    score=pointing_score(points),
                    tip_xy=(tip[0], tip[1]),
                )
            )
        return hands

    def select_drawing_hand(self, hands: Sequence[HandPoint]) -> Optional[HandPoint]:
        if not hands:
            return None
        best = max(hands, key=lambda hand: hand.score)
        if self._locked_label is None:
            self._locked_label = best.label
            return best

        locked = next((hand for hand in hands if hand.label == self._locked_label), None)
        if locked is not None and (
            locked.score + 0.10 >= best.score or locked.score >= self.exit_score
        ):
            return locked

        self._locked_label = best.label
        return best

    def update_draw_state(self, hand: Optional[HandPoint]) -> bool:
        if hand is None:
            self._lost += 1
            if self._lost >= self.lost_frames:
                self._drawing = False
                self._streak = 0
                self._locked_label = None
            return self._drawing

        self._lost = 0
        score = float(hand.score)
        want = score >= (self.exit_score if self._drawing else self.enter_score)
        if want == self._drawing:
            self._streak = 0
            return self._drawing

        self._streak += 1
        needed = self.exit_frames if self._drawing else self.enter_frames
        if self._streak >= needed:
            self._drawing = want
            self._streak = 0
        return self._drawing

    def reset_draw_state(self):
        self._drawing = False
        self._streak = 0
        self._lost = 0
        self._locked_label = None
        self.tip_tracker.reset()

    def draw_landmarks(self, frame, results):
        if not results or not results.multi_hand_landmarks:
            return
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
            )

    def is_index_extended(self, hand, min_score: float = 0.52) -> bool:
        if hand is None or not getattr(hand, "landmarks", None):
            return False
        if getattr(hand, "score", None) is not None:
            return hand.score >= min_score
        return is_index_extended(hand.landmarks, min_score=min_score)

    def close(self):
        self.hands.close()
