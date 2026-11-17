"""Matplotlib figures. Each function returns ``(figure, axes)`` and does not show them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegmath.recording import EEG


def _axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_timeseries(eeg: EEG, *, picks: list[str] | None = None, duration: float | None = 5.0, ax=None):
    """Plot a short stretch of one epoch or a continuous recording."""
    fig, ax = _axes(ax, (8, 4))
    data = eeg.data[0] if eeg.epoched else eeg.data
    names = list(eeg.ch_names if picks is None else picks)
    idx = eeg.channel_index(names)
    n = data.shape[-1] if duration is None else min(data.shape[-1], int(duration * eeg.sfreq))
    times = eeg.tmin + np.arange(n) / eeg.sfreq
    sample = data[:, :n]
    finite = sample[np.isfinite(sample)]
    spread = float(np.std(finite)) if finite.size else np.nan
    offset = spread * 4 if np.isfinite(spread) and spread > 0 else 1.0
    for i, channel in enumerate(idx):
        ax.plot(times, data[channel, :n] + i * offset, lw=0.8)
    ax.set_yticks([i * offset for i in range(len(idx))], names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (offset for display)")
    ax.set_title(eeg.recording_id or "EEG")
    return fig, ax


def plot_psd(freqs: np.ndarray, psd: np.ndarray, *, ch_names: list[str] | None = None, fmax: float = 45.0, ax=None):
    """Plot power spectral density in dB relative to 1 unit squared per Hz.

    A non-positive density is drawn at ``10 log10(1e-30)`` so the line
    stays on the axes. Feature columns do not use that display floor.
    """
    fig, ax = _axes(ax, (7, 4))
    psd = np.atleast_2d(psd)
    mask = freqs <= fmax
    for i, row in enumerate(psd):
        label = None if ch_names is None else ch_names[i]
        ax.plot(freqs[mask], 10 * np.log10(row[mask] + 1e-30), lw=1.0, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re 1 unit²/Hz)")
    if ch_names is not None and len(ch_names) <= 12:
        ax.legend(fontsize=8, frameon=False)
    return fig, ax


def plot_forest(results: pd.DataFrame, *, estimate: str = "effect", low: str = "ci_low", high: str = "ci_high", label: str = "feature", ax=None):
    """Point estimates with intervals. Rows are drawn in the given order.

    A missing endpoint is omitted. It is not replaced by the point
    estimate, which would look like an interval of width zero.
    """
    fig, ax = _axes(ax, (7, max(3, 0.35 * len(results))))
    y = np.arange(len(results))
    est = results[estimate].to_numpy(dtype=float)
    lo = results[low].to_numpy(dtype=float) if low in results.columns else np.full(len(results), np.nan)
    hi = results[high].to_numpy(dtype=float) if high in results.columns else np.full(len(results), np.nan)
    lower = np.where(np.isfinite(est) & np.isfinite(lo) & (lo <= est), est - lo, np.nan)
    upper = np.where(np.isfinite(est) & np.isfinite(hi) & (hi >= est), hi - est, np.nan)
    xerr = np.vstack([lower, upper])
    ax.errorbar(est, y, xerr=xerr, fmt="o", color="0.2", ecolor="0.45", capsize=2)
    ax.axvline(0.0, color="0.7", lw=0.8)
    ax.set_yticks(y, results[label].astype(str))
    ax.set_xlabel(estimate)
    ax.invert_yaxis()
    return fig, ax
