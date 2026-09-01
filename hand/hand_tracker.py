from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


Point = Tuple[int, int]


@dataclass
class TrackedHand:
    index: int
    label: str
    landmarks: List[Point]
    index_tip: Point


class HandTracker:
    def extract(self, results, frame_shape) -> List[TrackedHand]:
        h, w = frame_shape[:2]
        output: List[TrackedHand] = []

        if not results.multi_hand_landmarks:
            return output

        handedness = results.multi_handedness or []

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            points = [
                (
                    max(0, min(w - 1, int(lm.x * w))),
                    max(0, min(h - 1, int(lm.y * h))),
                )
                for lm in hand_landmarks.landmark
            ]

            label = "Hand"
            if i < len(handedness):
                label = handedness[i].classification[0].label

            output.append(
                TrackedHand(
                    index=i,
                    label=label,
                    landmarks=points,
                    index_tip=points[8],
                )
            )

        return output

    @staticmethod
    def finger_extended(points, tip, pip) -> bool:
        return points[tip][1] < points[pip][1]
