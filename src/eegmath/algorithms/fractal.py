"""Fractal-dimension and detrended-fluctuation estimators.

Higuchi fractal dimension follows Higuchi (1988): mean curve length ``L(k)``
across ``k`` subsampled series, with dimension equal to the slope of
``log L(k)`` against ``log(1/k)``.

Petrosian fractal dimension follows Petrosian (1995), using the number of
sign changes in the first difference.

Katz fractal dimension follows the amplitude-only form used for sampled
waveforms: ``log10(n) / (log10(d/L) + log10(n))`` with ``n = L / a``,
where successive distances are absolute amplitude differences. This is
not the planar curve with a time axis, and it is not bounded by 2.
A straight line is 1. A rapidly oscillating trace can exceed 2. When
the farthest point from the first sample equals the mean step, the
denominator is zero and the value is infinite.

Higuchi scales whose curve length is zero are omitted. A regression
that included ``log(0)`` would not be finite. The slope is then the
slope of the remaining scales only.

Detrended fluctuation analysis follows Peng et al. (1994). Window lengths
grow geometrically from 4 samples to 10% of the series. The reported
exponent is the slope of ``log F(n)`` against ``log n`` after linear
detrending inside each window. White noise is near 0.5; integrated white
noise (Brownian motion) is near 1.5.
"""

from __future__ import annotations

import numpy as np

from eegmath.algorithms.time_domain import nonzero_sign_changes
from eegmath.errors import ParameterError


def higuchi_fd(x: np.ndarray, *, kmax: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    kmax = int(kmax)
    if kmax < 2:
        raise ParameterError("Higuchi kmax must be at least 2.")
    n_times = x.size
    if n_times < kmax + 2 or not np.isfinite(x).all():
        return np.nan
    lengths = np.empty(kmax, dtype=float)
    for k in range(1, kmax + 1):
        curve = []
        for m in range(k):
            n_max = (n_times - m - 1) // k
            if n_max < 1:
                continue
            idx = m + np.arange(n_max + 1) * k
            ll = float(np.abs(np.diff(x[idx])).sum())
            ll = (ll / k) * ((n_times - 1) / (k * n_max))
            curve.append(ll)
        lengths[k - 1] = float(np.mean(curve)) if curve else np.nan
    valid = np.isfinite(lengths) & (lengths > 0)
    if int(valid.sum()) < 2:
        return np.nan
    ks = np.arange(1, kmax + 1, dtype=float)
    slope = np.polyfit(np.log(1.0 / ks[valid]), np.log(lengths[valid]), 1)[0]
    return float(slope)


def petrosian_fd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3 or not np.isfinite(x).all():
        return np.nan
    # A derivative that only touches zero has not changed sign.
    n_delta = nonzero_sign_changes(np.diff(x))
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))


def katz_fd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3 or not np.isfinite(x).all():
        return np.nan
    distances = np.abs(np.diff(x))
    length = float(np.sum(distances))
    if length == 0:
        return 1.0
    average = float(np.mean(distances))
    extent = float(np.max(np.abs(x - x[0])))
    if extent == 0 or average == 0:
        return 1.0
    n = length / average
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.log10(n) / (np.log10(extent / length) + np.log10(n))
    if not np.isfinite(value):
        return float(np.inf)
    return float(value)


def detrended_fluctuation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 16 or not np.isfinite(x).all():
        return np.nan
    min_n = 4.0
    max_n = 0.1 * n
    if max_n <= min_n:
        return np.nan
    factor = 1.2
    max_i = int(np.floor(np.log(max_n / min_n) / np.log(factor)))
    scales = [int(min_n)]
    for i in range(max_i + 1):
        value = int(np.floor(min_n * (factor**i)))
        if value > scales[-1]:
            scales.append(value)
    walk = np.cumsum(x - np.mean(x))
    log_n: list[float] = []
    log_f: list[float] = []
    for scale in scales:
        n_windows = n // scale
        if n_windows < 1:
            continue
        windows = walk[: n_windows * scale].reshape(n_windows, scale)
        t = np.arange(scale, dtype=float)
        t_centered = t - t.mean()
        denom = float(np.dot(t_centered, t_centered))
        if denom <= 0:
            continue
        centered = windows - windows.mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            slopes = centered @ t_centered / denom
        if not np.isfinite(slopes).all():
            continue
        trend = windows.mean(axis=1, keepdims=True) + slopes[:, None] * t_centered
        residual = windows - trend
        if not np.isfinite(residual).all():
            continue
        fluctuation = float(np.sqrt(np.mean(np.mean(residual**2, axis=1))))
        if fluctuation > 0:
            log_n.append(np.log(scale))
            log_f.append(np.log(fluctuation))
    if len(log_n) < 2:
        return np.nan
    return float(np.polyfit(log_n, log_f, 1)[0])
