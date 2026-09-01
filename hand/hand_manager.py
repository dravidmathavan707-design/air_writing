from __future__ import annotations

from dataclasses import dataclass
from typing import List

from hand.hand_tracker import TrackedHand


@dataclass
class HandState:
    hand_count: int = 0
    hands: List[TrackedHand] | None = None


class HandManager:
    def update(self, hands: List[TrackedHand]) -> HandState:
        return HandState(
            hand_count=len(hands),
            hands=hands,
        )
