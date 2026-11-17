"""Small signal constructors shared by tests."""

from __future__ import annotations

import numpy as np


def sine(freq: float, sfreq: float, duration: float, phase: float = 0.0, amplitude: float = 1.0) -> np.ndarray:
    times = np.arange(int(round(duration * sfreq))) / sfreq
    return amplitude * np.sin(2 * np.pi * freq * times + phase)
