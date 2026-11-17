"""Time-domain descriptors.

Hjorth mobility is the square root of ``var(dx/dt) / var(x)`` with the
first difference scaled by the sampling frequency. For a pure sinusoid
of frequency ``f`` the continuous-time value is ``2 π f`` radians per
second. The one-sample difference equals ``2 * sfreq * sin(π f / sfreq)``,
which approaches ``2 π f`` when the sampling rate is high relative to
``f``. Complexity is dimensionless. It equals 1 for a pure sinusoid in
continuous time. On a finite epoch the second difference is shorter
than the series, so the value sits slightly above 1 and moves closer
as the epoch gets longer. Activity is the population variance (``ddof=0``).

Line length is the mean absolute successive difference per second
(amplitude units per second). Zero-crossing rate is crossings per
second of the span from the first sample to the last, ``(n - 1) / sfreq``.
An exact zero has no sign. A touch that returns to the same side is
not a crossing.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

from eegmath.algorithms.entropy import lempel_ziv_complexity
from eegmath.errors import ParameterError


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        return np.nan
    return float(np.sqrt(np.mean(np.square(x))))


def line_length(x: np.ndarray, sfreq: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2 or not np.isfinite(x).all() or not np.isfinite(sfreq) or sfreq <= 0:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))) * sfreq)


def nonzero_sign_changes(x: np.ndarray) -> int:
    """Sign changes after exact zeros have been ignored.

    ``np.signbit`` treats ``+0`` as positive, so a touch from below
    (negative, zero, negative) was counted as two crossings and a touch
    from above as none. Zeros are neither sign. The samples on either
    side of a run of zeros are compared directly.
    """
    signs = np.sign(np.asarray(x, dtype=float))
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def zero_crossing_rate(x: np.ndarray, sfreq: float) -> float:
    """Sign changes per second. Exact zeros are not a sign."""
    x = np.asarray(x, dtype=float)
    if x.size < 2 or not np.isfinite(x).all() or sfreq <= 0:
        return np.nan
    # The crossings lie on the span from the first sample to the last,
    # which is (n - 1) / sfreq, the same time base as line length.
    return float(nonzero_sign_changes(x) * sfreq / (x.size - 1))


def peak_to_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        return np.nan
    return float(np.ptp(x))


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3 or not np.isfinite(x).all():
        return np.nan
    return float(stats.skew(x, bias=True))


def kurtosis(x: np.ndarray) -> float:
    """Fisher kurtosis (normal distribution → 0), biased sample moment."""
    x = np.asarray(x, dtype=float)
    if x.size < 4 or not np.isfinite(x).all():
        return np.nan
    return float(stats.kurtosis(x, fisher=True, bias=True))


def hjorth_parameters(x: np.ndarray, sfreq: float) -> tuple[float, float, float]:
    """Return ``(activity, mobility_rad_per_s, complexity)``."""
    x = np.asarray(x, dtype=float)
    if x.size < 3 or not np.isfinite(x).all() or not np.isfinite(sfreq) or sfreq <= 0:
        return (np.nan, np.nan, np.nan)
    dx = np.diff(x) * sfreq
    ddx = np.diff(dx) * sfreq
    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))
    if var_x <= 0 or var_dx <= 0:
        mobility = 0.0 if var_x > 0 and var_dx == 0 else np.nan
        complexity = np.nan
        return (var_x, mobility, complexity)
    mobility = float(np.sqrt(var_dx / var_x))
    complexity = float(np.sqrt(var_ddx / var_dx) / mobility)
    return (var_x, mobility, complexity)


def burst_suppression_ratio(
    x: np.ndarray,
    sfreq: float,
    *,
    threshold: float = 5e-6,
    min_duration: float = 0.5,
) -> float:
    """Fraction of time spent in sustained low-amplitude intervals.

    A sample is below threshold when ``abs(x) < threshold``. A run of
    such samples counts only when it is strictly longer than
    ``min_duration`` seconds. Briefer dips, including the instants at
    which a larger oscillation crosses the threshold, do not count.
    The defaults are 5 µV (``5e-6`` when ``x`` is in volts) and 0.5 s.
    That pair is the human-scalp convention associated with Rampil
    (1998): an interval longer than 0.5 s during which the voltage
    stays within approximately ±5 µV, as restated by Muhlhofer et al.
    (2017). The ratio is the fraction of this array, not of a fixed
    monitor window.

    Vijn and Sneyd (1998) used a different detector, on the time
    derivative, inside a ±15 µV window, with a minimum of 100 ms.
    This function does not implement that detector.

    ``threshold`` is in the same unit as ``x``. The ratio is not
    meaningful after z-scoring or any other rescaling that removes the
    voltage. An array no longer than ``min_duration`` cannot contain a
    qualifying run, so the ratio is missing. ``min_duration=0`` returns
    the fraction of samples below threshold, without a duration rule.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).all() or not np.isfinite(sfreq) or sfreq <= 0:
        return np.nan
    if threshold < 0 or not np.isfinite(min_duration) or min_duration < 0:
        raise ParameterError("threshold and min_duration must be non-negative and finite.")
    below = np.abs(x) < threshold
    if min_duration == 0:
        return float(np.mean(below))
    if x.size / float(sfreq) <= min_duration:
        return np.nan
    if not np.any(below):
        return 0.0
    padded = np.concatenate((np.array([False]), below, np.array([False])))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    keep = np.zeros(x.size, dtype=bool)
    for start, end in zip(starts, ends):
        if (end - start) / float(sfreq) > min_duration:
            keep[start:end] = True
    return float(np.mean(keep))


def _bandpass(x: np.ndarray, sfreq: float, flo: float, fhi: float) -> np.ndarray:
    nyq = sfreq / 2.0
    if flo <= 0 or fhi <= flo or fhi >= nyq:
        raise ParameterError(
            f"Band-pass edges ({flo}, {fhi}) Hz are not inside (0, {nyq}) Hz at sfreq={sfreq}."
        )
    sos = signal.butter(4, [flo, fhi], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, x)


def envelope_lzc(
    x: np.ndarray,
    sfreq: float,
    *,
    flo: float = 1.0,
    fhi: float = 40.0,
) -> float:
    """Lempel–Ziv complexity of the Hilbert amplitude, split at its mean.

    The signal is band-passed, the analytic amplitude is taken, and that
    amplitude is set to 1 where it exceeds its mean. Schartner et al.
    (2015) binarise the Hilbert amplitude at that mean. The number
    returned here is then the single-channel Zhang normalisation of the
    binary sequence. It is not their multichannel complexity, which
    concatenates channels and divides by the complexity of a shuffled
    copy. The default 1–40 Hz band-pass belongs to this feature.
    Schartner reported broadband signals and, separately, band-limited
    ones, and did not fix this interval.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 16 or not np.isfinite(x).all():
        return np.nan
    try:
        filtered = _bandpass(x, sfreq, flo, fhi)
    except ParameterError:
        raise
    except ValueError:
        return np.nan
    envelope = np.abs(signal.hilbert(filtered))
    binary = (envelope > np.mean(envelope)).astype(np.uint8)
    return lempel_ziv_complexity(binary, normalize=True)
