import math
from typing import Optional

import numpy as np


class OneEuroFilter:
    """A lightweight 1€ filter for a scalar signal."""

    def __init__(
        self,
        min_cutoff: float = 0.05,
        beta: float = 80.0,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def __call__(self, t: float, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            return float(x)

        t_delta = float(t) - float(self.t_prev)
        if t_delta <= 0:
            return float(self.x_prev)

        dx = (float(x) - float(self.x_prev)) / t_delta
        a_d = self._smoothing_factor(t_delta, self.d_cutoff)
        dx_hat = self._exp_smooth(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_delta, cutoff)
        x_hat = self._exp_smooth(a, float(x), float(self.x_prev))

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat

    @staticmethod
    def _smoothing_factor(t_delta: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_delta
        return r / (r + 1)

    @staticmethod
    def _exp_smooth(a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class LandmarkSmoother:
    """Independent One Euro smoothers for each landmark x/y coordinate."""

    def __init__(
        self,
        num_landmarks: int = 478,
        min_cutoff: float = 0.05,
        beta: float = 80.0,
        d_cutoff: float = 1.0,
    ):
        self.filters = [
            [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(2)]
            for _ in range(num_landmarks)
        ]

    def smooth(self, landmarks: np.ndarray, timestamp: float) -> np.ndarray:
        if landmarks.ndim != 2 or landmarks.shape[1] < 2:
            return landmarks.copy()

        out = landmarks.copy().astype(np.float32)
        for i in range(len(self.filters)):
            if i >= len(out):
                break
            out[i, 0] = self.filters[i][0](timestamp, float(out[i, 0]))
            out[i, 1] = self.filters[i][1](timestamp, float(out[i, 1]))
        return out

    def reset(self):
        for pair in self.filters:
            for smoother in pair:
                smoother.reset()
