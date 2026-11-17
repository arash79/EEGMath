"""Welch spectra and spectral summaries.

Power spectral density is estimated with Welch's method (Welch, 1967):
Hann windows, 50% overlap, density scaling (amplitude²/Hz). The default
segment length is 2 seconds, so the frequency resolution does not change
silently when the sampling rate changes. ``detrend="constant"`` removes
the segment mean. Linear detrend is available but attenuates the lowest
frequencies inside a short segment, so it is not the default.

Band power is the trapezoidal integral of the density. Relative power
uses the sum of the partition integrals as its denominator, so the
relative values of a complete partition sum to 1. A zero denominator
is undefined: relative power is missing, and a ratio with a positive
numerator is infinite. No constant is added to a denominator. Spectral
edge frequency, mean frequency, and median frequency are computed from
the same trapezoidal cumulative integral over ``[freq_min, freq_max]``.
The cumulative integral starts at zero on the lower edge of the
requested interval, after the density has been linearly interpolated
onto that edge, so a percentile that falls inside the first step is
interpolated there. Median frequency and SEF50 are the same quantity
and median frequency is always computed, whether or not 50 is in the
requested edge list. Adjacent bands share only the boundary frequency.
That point has no width, so the integrals of a contiguous partition
add up to the integral of the union and to ``total_power`` when the
partition covers the analysis interval. A band that does not overlap
the estimated spectrum is omitted from the relative-power denominator.
A band whose bins are all equal has no peak frequency. Peak frequency
uses the bins assigned to the band; the upper boundary bin belongs to
the next band.

Spectral entropy is the Shannon entropy of the Welch bin masses inside
the analysis band, divided by ``log2(n_bins)``. The result lies in
``[0, 1]``. Inouye et al. (1991) defined the unnormalised entropy of
the normalised spectrum and reported it in bits. The division by
``log2(n_bins)`` is not part of that index. It keeps the value from
growing when the number of Welch bins changes. Bins are uniformly
spaced, so normalising the density values is equivalent to normalising
probability mass.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import signal

from eegmath.bands import DEFAULT_BANDS, DEFAULT_FREQ_MAX, DEFAULT_FREQ_MIN, partition_masks
from eegmath.errors import ParameterError


def _trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    # NumPy 2 renamed trapz. The package still declares support for 1.26.
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x, axis=-1)


def _divide(numerator, denominator) -> np.ndarray:
    """Elementwise ratio. Zero divided by zero is missing; a positive
    numerator over zero is infinite. No floor is added to the denominator.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


def _power_db(power) -> np.ndarray:
    """``10 log10(power)``. Non-positive power is missing, not a floored stand-in."""
    values = np.asarray(power, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        converted = 10.0 * np.log10(values)
    return np.where(values > 0, converted, np.nan)


def default_n_per_seg(sfreq: float, n_times: int, segment_sec: float = 2.0) -> int:
    """Segment length in samples. Falls back to the series length when it is shorter."""
    if sfreq <= 0 or n_times < 4:
        raise ParameterError("A spectrum needs a positive sampling rate and at least 4 samples.")
    requested = int(round(segment_sec * sfreq))
    requested = max(8, requested)
    return int(min(requested, n_times))


def welch_psd(
    data: np.ndarray,
    sfreq: float,
    *,
    n_per_seg: int | None = None,
    overlap: float = 0.5,
    detrend: str = "constant",
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD.

    ``data`` may be ``(n_times,)``, ``(n_channels, n_times)``, or
    ``(n_epochs, n_channels, n_times)``. The spectrum is taken along the
    last axis. Epoched data are not concatenated: each epoch has its own
    spectrum.
    """
    data = np.asarray(data, dtype=float)
    if data.shape[-1] < 4:
        raise ParameterError(f"Cannot estimate a spectrum from {data.shape[-1]} samples.")
    if not 0 <= overlap < 1:
        raise ParameterError("overlap must be in [0, 1).")
    n_per = default_n_per_seg(sfreq, data.shape[-1]) if n_per_seg is None else int(n_per_seg)
    n_per = min(n_per, data.shape[-1])
    if n_per < 4:
        raise ParameterError("n_per_seg must be at least 4.")
    noverlap = int(round(overlap * n_per))
    noverlap = min(noverlap, n_per - 1)
    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        window=window,
        nperseg=n_per,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
        axis=-1,
    )
    return freqs, psd


def _trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _trapezoid(y, x)


def _interval_samples(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Density samples on ``[fmin, fmax]``, with the endpoints interpolated.

    The returned grid always starts at the lower limit and ends at the
    upper limit, clipped to the frequencies that were actually estimated.
    Interior bins keep their original values. A band narrower than the
    bin spacing is the trapezoid between its two interpolated endpoints.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if freqs.ndim != 1 or freqs.size < 2:
        return None, None
    if not np.all(np.diff(freqs) > 0):
        return None, None
    if not np.isfinite(fmin) or not np.isfinite(fmax) or float(fmax) <= float(fmin):
        return None, None
    lo = max(float(fmin), float(freqs[0]))
    hi = min(float(fmax), float(freqs[-1]))
    if hi <= lo:
        return None, None
    tolerance = max(1e-9, 1e-8 * (hi - lo))

    def at(freq: float) -> np.ndarray:
        index = int(np.searchsorted(freqs, freq, side="left"))
        if index < freqs.size and abs(float(freqs[index]) - freq) <= tolerance:
            return psd[..., index]
        if index > 0 and abs(float(freqs[index - 1]) - freq) <= tolerance:
            return psd[..., index - 1]
        if index <= 0 or index >= freqs.size:
            return psd[..., min(max(index, 0), freqs.size - 1)]
        left_f = float(freqs[index - 1])
        right_f = float(freqs[index])
        weight = (freq - left_f) / (right_f - left_f)
        return (1.0 - weight) * psd[..., index - 1] + weight * psd[..., index]

    interior = (freqs > lo + tolerance) & (freqs < hi - tolerance)
    grid = np.concatenate(([lo], freqs[interior], [hi]))
    left = np.asarray(at(lo), dtype=float)[..., None]
    right = np.asarray(at(hi), dtype=float)[..., None]
    values = np.concatenate([left, psd[..., interior], right], axis=-1)
    return grid, values


def band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    *,
    closed: bool = True,
) -> np.ndarray:
    """Integrate ``psd`` over ``[fmin, fmax]``.

    Endpoints that fall between bins are linearly interpolated. The
    integral of a density does not change if the endpoint itself is
    excluded, so ``closed`` is accepted and does not select a different
    value. Adjacent calls that meet at one frequency do not count that
    frequency twice. Returns missing when the interval does not overlap
    the estimated spectrum.
    """
    del closed
    _grid, values = _interval_samples(freqs, psd, fmin, fmax)
    if values is None:
        return np.full(np.shape(psd)[:-1], np.nan)
    return _trapz(values, _grid)


def spectral_edge_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    percentile: float,
    *,
    fmin: float = DEFAULT_FREQ_MIN,
    fmax: float = DEFAULT_FREQ_MAX,
) -> np.ndarray:
    """Frequency at which the cumulative band integral reaches ``percentile`` percent."""
    if not 0 < percentile <= 100:
        raise ParameterError("SEF percentile must be in (0, 100].")
    f, p = _interval_samples(freqs, psd, fmin, fmax)
    if f is None:
        return np.full(np.shape(psd)[:-1], np.nan)
    increments = (p[..., :-1] + p[..., 1:]) * 0.5 * np.diff(f)
    total = np.sum(increments, axis=-1)
    cumulative = np.cumsum(increments, axis=-1)
    cumulative = np.concatenate(
        [np.zeros(cumulative.shape[:-1] + (1,), dtype=float), cumulative],
        axis=-1,
    )
    target = (percentile / 100.0) * total
    out = np.empty(np.shape(total), dtype=float)
    flat_c = np.reshape(cumulative, (-1, cumulative.shape[-1]))
    flat_t = np.broadcast_to(np.asarray(target, dtype=float), np.shape(total)).reshape(-1)
    flat_out = out.reshape(-1)
    for i, (curve, goal) in enumerate(zip(flat_c, flat_t)):
        flat_out[i] = _edge_from_cumulative(curve, goal, f)
    return out


def _edge_from_cumulative(curve: np.ndarray, goal: float, edges: np.ndarray) -> float:
    """Lowest frequency at which the cumulative integral reaches ``goal``.

    ``curve[0]`` is zero at ``edges[0]``. A flat step means a bin with no
    power; the edge stays at the start of that step.
    """
    if not np.isfinite(goal) or goal <= 0 or not np.isfinite(curve).all():
        return np.nan
    if curve.size != edges.size or curve.size < 2:
        return np.nan
    goal = float(min(goal, curve[-1]))
    if goal <= 0:
        return np.nan
    steps = np.diff(curve)
    if np.any(steps < -1e-9 * max(float(np.max(np.abs(curve))), 1.0)):
        return np.nan
    index = int(np.searchsorted(curve, goal, side="left"))
    index = min(max(index, 1), curve.size - 1)
    left = float(curve[index - 1])
    right = float(curve[index])
    if right <= left:
        return float(edges[index - 1])
    fraction = (goal - left) / (right - left)
    return float(edges[index - 1] + fraction * (edges[index] - edges[index - 1]))


def _spectral_entropy(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    p = psd[..., mask]
    if p.shape[-1] < 2:
        return np.full(psd.shape[:-1], np.nan)
    mass = np.sum(p, axis=-1)
    positive = mass > 0
    if not np.any(positive):
        return np.full(mass.shape, np.nan)
    prob = np.divide(p, mass[..., None], out=np.zeros_like(p), where=positive[..., None])
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(prob > 0, np.log2(prob), 0.0)
    values = -np.sum(prob * logp, axis=-1) / np.log2(p.shape[-1])
    return np.where(positive, values, np.nan)


def _mean_frequency(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    f, p = _interval_samples(freqs, psd, fmin, fmax)
    if f is None:
        return np.full(np.shape(psd)[:-1], np.nan)
    numerator = _trapz(p * f, f)
    denominator = _trapz(p, f)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


RATIO_DEFINITIONS: dict[str, tuple[str, str]] = {
    "delta_alpha_ratio": ("delta", "alpha"),
    "theta_alpha_ratio": ("theta", "alpha"),
    "delta_theta_ratio": ("delta", "theta"),
    "beta_alpha_ratio": ("beta", "alpha"),
    "beta_delta_ratio": ("beta", "delta"),
}


def is_derived_spectral_column(name: str) -> bool:
    """True for columns that are not linear in the spectrum.

    Averaging these across epochs or channels is not the same operation as
    recomputing them from averaged band power.
    """
    return name in RATIO_DEFINITIONS or name in {"slow_fast_ratio", "alpha_dominance"} or name.endswith(
        ("_relative", "_power_db")
    )


def power_ratios(band_power_map: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ratios of absolute band powers.

    A missing band produces NaN. A positive numerator over a zero
    denominator is infinite. Both zero is NaN. Nothing is added to the
    denominator.
    """

    def get(name: str):
        if name not in band_power_map:
            return np.nan
        return band_power_map[name]

    out: dict[str, np.ndarray] = {}
    for name, (num, den) in RATIO_DEFINITIONS.items():
        out[name] = _divide(get(num), get(den))
    if {"delta", "theta", "alpha", "beta"} <= set(band_power_map):
        slow = np.asarray(band_power_map["delta"], dtype=float) + np.asarray(band_power_map["theta"], dtype=float)
        fast = np.asarray(band_power_map["alpha"], dtype=float) + np.asarray(band_power_map["beta"], dtype=float)
        out["slow_fast_ratio"] = _divide(slow, fast)
        out["alpha_dominance"] = _divide(band_power_map["alpha"], slow + fast)
    return out


def spectral_feature_set(
    data: np.ndarray,
    sfreq: float,
    *,
    bands: Mapping[str, tuple[float, float]] | None = None,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    n_per_seg: int | None = None,
    segment_sec: float = 2.0,
    overlap: float = 0.5,
    detrend: str = "constant",
    sef_percentiles: tuple[float, ...] = (50, 75, 90, 95),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Spectral features and the PSD used to compute them.

    Returned arrays have the prefix shape of ``data`` (everything except
    the time axis). ``freq_max`` is limited to the Nyquist frequency.
    Named bands are not clipped to ``[freq_min, freq_max]``. Those limits
    define total power, the spectral edge, the mean frequency, and the
    spectral entropy. A band that extends past them still contributes its
    own integral, so the band powers equal ``total_power`` only when the
    bands partition that interval.
    """
    bands = dict(DEFAULT_BANDS if bands is None else bands)
    nyquist = sfreq / 2.0
    freq_max = min(float(freq_max), nyquist)
    if freq_max <= freq_min:
        raise ParameterError(
            f"Analysis band {freq_min}–{freq_max} Hz is empty at sfreq={sfreq} Hz."
        )
    if n_per_seg is None:
        n_per_seg = default_n_per_seg(sfreq, data.shape[-1], segment_sec)
    freqs, psd = welch_psd(
        data, sfreq, n_per_seg=n_per_seg, overlap=overlap, detrend=detrend
    )
    features: dict[str, np.ndarray] = {}
    absolute: dict[str, np.ndarray] = {}
    masks = dict(partition_masks(freqs, bands))
    nan = np.full(psd.shape[:-1], np.nan)
    for name, (band_min, band_max) in bands.items():
        power = band_power(freqs, psd, band_min, band_max)
        if not np.any(np.isfinite(np.atleast_1d(power))):
            features[f"{name}_power"] = nan.copy()
            features[f"{name}_relative"] = nan.copy()
            features[f"{name}_power_db"] = nan.copy()
            features[f"{name}_peak_freq"] = nan.copy()
            continue
        absolute[name] = power
        features[f"{name}_power"] = power
        mask = masks[name]
        if int(np.count_nonzero(mask)) < 2:
            features[f"{name}_peak_freq"] = nan.copy()
            continue
        band = psd[..., mask]
        maximum = np.max(band, axis=-1, keepdims=True)
        n_at_max = np.sum(band == maximum, axis=-1)
        maximum = np.squeeze(maximum, axis=-1)
        peak = np.argmax(band, axis=-1)
        peak_freq = np.asarray(freqs[mask][peak], dtype=float)
        undefined = (maximum <= 0) | (n_at_max == band.shape[-1])
        features[f"{name}_peak_freq"] = np.where(undefined, np.nan, peak_freq)
    if absolute:
        outside = [
            f"{name} ({band_min:g}–{band_max:g} Hz)"
            for name, (band_min, band_max) in bands.items()
            if name in absolute
            and (band_min < float(freq_min) - 1e-9 or band_max > float(freq_max) + 1e-9)
        ]
        if outside:
            warnings.warn(
                "These named bands extend outside the analysis interval "
                f"{float(freq_min):g}–{float(freq_max):g} Hz and still enter the "
                f"relative-power denominator: {', '.join(outside)}. freq_max "
                "limits total_power, the spectral edge, the mean frequency, and "
                "the spectral entropy. It does not remove a named band from a "
                "relative power.",
                UserWarning,
                stacklevel=2,
            )
        denom = sum(absolute.values())
        for name, power in absolute.items():
            features[f"{name}_relative"] = _divide(power, denom)
            features[f"{name}_power_db"] = _power_db(power)
        features.update(power_ratios(absolute))
    total = band_power(freqs, psd, freq_min, freq_max)
    if np.any(np.isfinite(np.atleast_1d(total))):
        features["total_power"] = total
        features["mean_freq"] = _mean_frequency(freqs, psd, freq_min, freq_max)
        features["spectral_entropy"] = _spectral_entropy(freqs, psd, freq_min, freq_max)
        for pct in sef_percentiles:
            edge = spectral_edge_frequency(freqs, psd, pct, fmin=freq_min, fmax=freq_max)
            features[f"sef{int(pct) if float(pct).is_integer() else pct}"] = edge
        if "sef50" in features:
            features["median_freq"] = features["sef50"]
        else:
            features["median_freq"] = spectral_edge_frequency(
                freqs, psd, 50, fmin=freq_min, fmax=freq_max
            )
    return features, {"freqs": freqs, "psd": psd}


def warn_if_ratios_left_missing(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Warn when aggregation could not rebuild a ratio column.

    The columns are left missing on purpose. Averaging the ratios would
    be a different quantity, and rebuilding them needs the absolute
    ``{band}_power`` columns of the partition.
    """
    stuck = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if values.size and not np.isfinite(values).any():
            stuck.append(str(column))
    if not stuck:
        return
    warnings.warn(
        f"These columns are missing after aggregation: {stuck}. "
        "Relative power, decibels, and band ratios are rebuilt from absolute "
        "{band}_power columns. They are not the average of the ratios. "
        "Extract the absolute band powers as well, then aggregate again.",
        UserWarning,
        stacklevel=3,
    )


def recompute_power_ratios(df: pd.DataFrame, bands: Mapping[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Replace ratio, relative, and dB columns using absolute ``{band}_power`` columns.

    Use this after averaging absolute power across channels or epochs.
    Averaging ratios is not the same operation as the ratio of averages.
    """
    bands = dict(DEFAULT_BANDS if bands is None else bands)
    present = {name: df[f"{name}_power"] for name in bands if f"{name}_power" in df.columns}
    if not present:
        return df.copy()
    out = df.copy()
    # A relative power is a share of the whole partition. Rebuilding it from
    # the subset of band columns that happen to be in the table turns a
    # single retained band into 1.
    if set(present) == set(bands):
        denom = sum(present.values())
        for name, power in present.items():
            rel = f"{name}_relative"
            if rel in out.columns:
                out[rel] = _divide(power, denom)
    for name, power in present.items():
        db = f"{name}_power_db"
        if db in out.columns:
            out[db] = _power_db(power)
    derived = power_ratios(present)
    for name, values in derived.items():
        if name not in out.columns:
            continue
        if name in RATIO_DEFINITIONS:
            numerator, denominator = RATIO_DEFINITIONS[name]
            if numerator not in present or denominator not in present:
                continue
        elif name in {"slow_fast_ratio", "alpha_dominance"} and not {"delta", "theta", "alpha", "beta"} <= set(present):
            continue
        out[name] = values
    return out


def aperiodic_fit(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    fmin: float = 2.0,
    fmax: float = 40.0,
    reject_peaks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary least-squares slope of log power versus log frequency.

    Returns ``(exponent, offset)`` where the model is
    ``log10(psd) = offset - exponent * log10(freq)``. This is the
    aperiodic component of a spectrum in the sense of a straight line in
    log-log coordinates. It is not the FOOOF peak-plus-aperiodic model
    (Donoghue et al., 2020). When ``reject_peaks`` is true, bins whose
    residual exceeds the mean residual by one residual standard deviation
    are removed and the line is refit once. That trims the upper tail of
    the residuals. It does not identify a peak, and a narrow peak that
    does not stand above that tail after the first fit stays in the line.

    ``psd`` may include a leading channel or epoch axis. The fit is
    performed independently on each spectrum.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    mask = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0) & np.isfinite(freqs)
    if int(mask.sum()) < 4:
        shape = psd.shape[:-1]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    f = freqs[mask]
    p = psd[..., mask]
    logf = np.log10(f)
    flat = p.reshape(-1, p.shape[-1])
    exponent = np.empty(flat.shape[0])
    offset = np.empty(flat.shape[0])
    for i, row in enumerate(flat):
        valid = np.isfinite(row) & (row > 0)
        if int(valid.sum()) < 4:
            exponent[i] = np.nan
            offset[i] = np.nan
            continue
        x = logf[valid]
        y = np.log10(row[valid])
        slope, intercept = np.polyfit(x, y, 1)
        if reject_peaks:
            resid = y - (slope * x + intercept)
            keep = resid <= (np.mean(resid) + np.std(resid))
            if int(keep.sum()) >= 4 and int(keep.sum()) < int(valid.sum()):
                slope, intercept = np.polyfit(x[keep], y[keep], 1)
        exponent[i] = -slope
        offset[i] = intercept
    shape = p.shape[:-1]
    return exponent.reshape(shape), offset.reshape(shape)
