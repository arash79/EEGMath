"""Canonical EEG frequency bands.

These edges are a common clinical partition, not a universal standard
and not a claim about any one guideline. Delta, theta, and alpha are
0.5–4, 4–8, and 8–13 Hz. Beta is 13–30 Hz and gamma is 30–45 Hz. The
30 Hz cut is this partition's split between beta and gamma. Gamma stops
at 45 Hz so that a 50 Hz mains line, and the transition band of a 45 Hz
low-pass, are not absorbed into the band. Pass an explicit mapping to any
function that accepts ``bands`` when a study uses different edges.

``partition_masks`` assigns frequency bins. Intervals are half-open
``[low, high)``. A band that reaches the highest upper edge in the set
includes that edge, so the top bin is not dropped. A bin that sits on a
boundary between two bands belongs to the higher band. The order of the
mapping does not choose which edge is closed. Band power does not use
that mask. It integrates the density over the requested interval, with
the endpoints interpolated, and a shared boundary is only a point.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Analysis limits used when a method needs one broadband range.
DEFAULT_FREQ_MIN = 0.5
DEFAULT_FREQ_MAX = 45.0


def band_mask(
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    *,
    closed: bool = False,
) -> np.ndarray:
    """Boolean mask of frequency bins belonging to ``[fmin, fmax)`` or ``[fmin, fmax]``."""
    freqs = np.asarray(freqs, dtype=float)
    if fmax < fmin:
        raise ValueError(f"fmax ({fmax}) is below fmin ({fmin}).")
    if closed:
        return (freqs >= fmin) & (freqs <= fmax)
    return (freqs >= fmin) & (freqs < fmax)


def partition_masks(
    freqs: np.ndarray,
    bands: Mapping[str, tuple[float, float]] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Half-open masks. The uppermost cutoff in the set is closed.

    Every other upper edge stays open, whatever the order of ``bands``.
    A boundary bin therefore belongs to the higher band. Bands that share
    that highest cutoff each include it.
    """
    bands = dict(DEFAULT_BANDS if bands is None else bands)
    items = list(bands.items())
    finite_upper = [float(high) for _, (_, high) in items if np.isfinite(high)]
    top = max(finite_upper) if finite_upper else None
    masks: list[tuple[str, np.ndarray]] = []
    for name, (fmin, fmax) in items:
        closed = top is not None and np.isfinite(fmax) and abs(float(fmax) - top) <= 1e-9
        masks.append((name, band_mask(freqs, fmin, fmax, closed=closed)))
    return masks
