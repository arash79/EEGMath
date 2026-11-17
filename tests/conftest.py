"""Shared synthetic recordings. Matplotlib is forced off-screen."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from eegmath.recording import EEG
from synthetic import sine


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(23)


@pytest.fixture
def alpha_eeg() -> EEG:
    sfreq = 250.0
    data = np.vstack(
        [
            sine(10.0, sfreq, 8.0),
            sine(10.0, sfreq, 8.0, phase=np.pi / 2),
            sine(4.0, sfreq, 8.0, amplitude=0.2),
        ]
    )
    return EEG(
        data,
        sfreq,
        ("Fz", "Cz", "Pz"),
        units="uV",
        subject_id="S1",
        condition="eyes-closed",
        recording_id="rec-1",
        reference="average",
    )
