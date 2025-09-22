import numpy as np
import math


class OneEuroFilter:
    """One Euro Filter for smooth signal filtering"""

    def __init__(self, te=1.0 / 25.0, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.x_prev = None
        self.dx_prev = 0.0
        self.te = te
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / self.te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) / self.te
        dx_hat = self._exponential_smoothing(dx, self.dx_prev, self._alpha(self.dcutoff))

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        x_hat = self._exponential_smoothing(x, self.x_prev, self._alpha(cutoff))

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

    def _exponential_smoothing(self, x_cur, x_prev, alpha):
        return alpha * x_cur + (1 - alpha) * x_prev


class LandmarkSmoother:
    """Smooth landmarks using One Euro Filter"""

    def __init__(self, num_points=68, fps=25, beta=0.007):
        te = 1.0 / fps
        self.filters = [
            [OneEuroFilter(te=te, beta=beta), OneEuroFilter(te=te, beta=beta)]
            for _ in range(num_points)
        ]

    def smooth(self, landmarks):
        smoothed = np.zeros_like(landmarks, dtype=np.float32)
        for i in range(len(landmarks)):
            smoothed[i, 0] = self.filters[i][0].filter(landmarks[i, 0])
            smoothed[i, 1] = self.filters[i][1].filter(landmarks[i, 1])
        return smoothed.astype(np.int32)