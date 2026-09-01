from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from body.body_detector import hand_is_fist, hand_is_open
from hand.hand_detector import pointing_score


@dataclass
class AnimationPose:
    present: bool = False
    index_tip: Tuple[float, float] = (0.0, 0.0)
    palm: Tuple[float, float] = (0.0, 0.0)
    wrist: Tuple[float, float] = (0.0, 0.0)
    is_fist: bool = False
    is_open: bool = False
    is_pointing: bool = False
    angle: float = 0.0
    scale: float = 80.0


def pose_from_hand(hand) -> AnimationPose:
    if hand is None or not getattr(hand, "landmarks", None) or len(hand.landmarks) < 21:
        return AnimationPose()
    lm = hand.landmarks
    wrist = (float(lm[0][0]), float(lm[0][1]))
    palm = (float(lm[9][0]), float(lm[9][1]))
    tip = hand.tip_xy if getattr(hand, "tip_xy", None) else (
        float(lm[8][0]),
        float(lm[8][1]),
    )
    dx = tip[0] - wrist[0]
    dy = tip[1] - wrist[1]
    angle = float(np.arctan2(dy, dx))
    scale = float(np.hypot(palm[0] - wrist[0], palm[1] - wrist[1]) * 2.4)
    return AnimationPose(
        present=True,
        index_tip=tip,
        palm=palm,
        wrist=wrist,
        is_fist=hand_is_fist(lm),
        is_open=hand_is_open(lm),
        is_pointing=pointing_score(lm) >= 0.52,
        angle=angle,
        scale=max(40.0, min(220.0, scale)),
    )


def smooth_point(
    previous: Optional[Tuple[float, float]],
    current: Tuple[float, float],
    amount: float = 0.28,
) -> Tuple[float, float]:
    if previous is None:
        return current
    return (
        previous[0] * (1.0 - amount) + current[0] * amount,
        previous[1] * (1.0 - amount) + current[1] * amount,
    )
