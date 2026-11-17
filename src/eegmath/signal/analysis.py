"""Signal-domain analyses that do not require a feature table.

Time-frequency power uses Morlet wavelets through MNE. ERP measures
and intertrial phase clustering require epochs that share an event
alignment. The modulation index is Tort's normalised Kullback-Leibler
divergence of the phase-amplitude distribution (Tort et al., 2010).
The augmented Dickey-Fuller test reports evidence against a unit root;
a small p-value is not a general proof of stationarity. The KPSS test
has the complementary null: the series is stationary. Its p-value is
taken from a published table and is censored at the ends of that table.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats

from eegmath.algorithms.time_domain import _bandpass
from eegmath.errors import DataError, ParameterError
from eegmath.recording import EEG

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise ImportError("Signal analyses that use Morlet wavelets require mne.") from exc


def _morlet_transform(data: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: float, output: str):
    """Call MNE and turn an over-long wavelet into ``ParameterError``."""
    try:
        return mne.time_frequency.tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output=output,
            verbose=False,
        )
    except ValueError as exc:
        text = str(exc)
        if "longer than the signal" in text:
            raise ParameterError(
                f"{text} Use a longer epoch, a higher frequency, or a smaller n_cycles."
            ) from exc
        raise


def morlet_power(
    eeg: EEG,
    freqs: np.ndarray,
    *,
    n_cycles: float = 7.0,
    average_epochs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Morlet time-frequency power.

    Returns ``(power, times)``. ``power`` has shape
    ``(n_epochs, n_channels, n_freqs, n_times)`` unless
    ``average_epochs=True``, in which case the epoch axis is averaged
    away and the shape is ``(n_channels, n_freqs, n_times)``. A continuous
    recording is treated as one epoch. The array is MNE's squared Morlet
    transform after the wavelet is scaled to unit norm times √2
    (Tallon-Baudry). It is not Welch band power and it is not in µV².
    A pure 10 µV sine has Welch variance 50. The same sine, 2 s at
    250 Hz, has a midpoint above 500 for a 3-cycle wavelet. Compare
    values only inside one normalisation. Do not multiply the array by
    ``1e12``. The default wavelet has
    seven cycles. A 4 Hz wavelet of that length does not fit in a 2 s
    epoch, and the call raises :class:`~eegmath.errors.ParameterError`
    rather than returning a truncated transform.
    """
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size < 1:
        raise ParameterError("freqs must be a one-dimensional grid.")
    if np.any(freqs <= 0) or np.any(freqs >= eeg.sfreq / 2):
        raise ParameterError("Wavelet frequencies must lie inside (0, Nyquist).")
    data = eeg.data[None, ...] if not eeg.epoched else eeg.data
    power = _morlet_transform(data, eeg.sfreq, freqs, n_cycles, "power")
    if average_epochs:
        power = power.mean(axis=0)
    return power, eeg.times


def _require_aligned_epochs(eeg: EEG, what: str) -> None:
    if not eeg.epoched or eeg.n_epochs < 2:
        raise DataError(
            f"{what} needs at least two epochs that share a clock. "
            "The function does not test whether that clock is an external event. "
            "A window onset is a clock."
        )


def _wavelet_frequencies(eeg: EEG, freqs: np.ndarray) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size < 1:
        raise ParameterError("freqs must be a one-dimensional grid.")
    if np.any(freqs <= 0) or np.any(freqs >= eeg.sfreq / 2):
        raise ParameterError("Wavelet frequencies must lie inside (0, Nyquist).")
    return freqs


def _unit_wavelets(eeg: EEG, freqs: np.ndarray, n_cycles: float) -> np.ndarray:
    """Unit-length Morlet coefficients, shape ``(n_epochs, n_channels, n_freqs, n_times)``."""
    complex_tfr = _morlet_transform(eeg.data, eeg.sfreq, freqs, n_cycles, "complex")
    magnitude = np.abs(complex_tfr)
    return np.divide(complex_tfr, magnitude, out=np.zeros_like(complex_tfr), where=magnitude > 0)


def intertrial_phase_clustering(
    eeg: EEG,
    freqs: np.ndarray,
    *,
    n_cycles: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude of the mean unit-length wavelet phase across epochs.

    This is intertrial phase clustering. The epochs must share a clock.
    That clock may be an external event, or the onset of windows cut
    from a continuous series. If a periodic component completes an
    integer number of cycles in every window, every window is the same
    epoch and the clustering is 1. The function does not inspect an
    event table. It only requires at least two epochs. The default
    wavelet has seven cycles. At 10 Hz its support is longer than a
    1 s epoch, and that call raises rather than returning a phase
    statistic. Returns ``(itpc, times)``. ``itpc`` has shape
    ``(n_channels, n_freqs, n_times)`` and lies in ``[0, 1]``. With few
    epochs the value is pulled above zero even when the phases are
    unrelated. :func:`intertrial_phase_consistency` removes that bias.
    """
    _require_aligned_epochs(eeg, "Intertrial phase clustering")
    freqs = _wavelet_frequencies(eeg, freqs)
    unit = _unit_wavelets(eeg, freqs, n_cycles)
    return np.abs(np.mean(unit, axis=0)), eeg.times


def intertrial_phase_consistency(
    eeg: EEG,
    freqs: np.ndarray,
    *,
    n_cycles: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise phase consistency of wavelet phase across epochs.

    Vinck, van Wingerden, Womelsdorf, Fries, and Pennartz (2010) define
    pairwise phase consistency so that a small number of observations
    does not pull the estimate upward. For ``n`` epochs and intertrial
    phase clustering ``P`` the value is ``(n P² − 1) / (n − 1)``. It
    estimates squared phase locking, lies in ``[-1, 1]``, and can be
    negative. Its lower bound is ``-1/(n-1)``, reached when the mean
    unit vector is zero. Two epochs in exact antiphase therefore give
    ``-1``. The epochs must share a clock, for the same reason as
    intertrial phase clustering. A window onset is such a clock.
    Returns ``(ppc, times)`` with shape ``(n_channels, n_freqs, n_times)``.
    """
    _require_aligned_epochs(eeg, "Intertrial phase consistency")
    freqs = _wavelet_frequencies(eeg, freqs)
    unit = _unit_wavelets(eeg, freqs, n_cycles)
    n = unit.shape[0]
    squared = np.abs(np.mean(unit, axis=0)) ** 2
    ppc = (n * squared - 1.0) / (n - 1.0)
    return np.clip(ppc, -1.0, 1.0), eeg.times


def average_epochs(eeg: EEG) -> tuple[np.ndarray, np.ndarray]:
    """Mean across epochs. Shape ``(n_channels, n_times)``."""
    if not eeg.epoched:
        raise DataError("average_epochs requires epoched data. Window a continuous recording first.")
    return eeg.data.mean(axis=0), eeg.times


def peak_measure(
    values: np.ndarray,
    times: np.ndarray,
    tmin: float,
    tmax: float,
    *,
    mode: str = "abs",
) -> dict[str, float]:
    """Latency and amplitude of a peak inside a time window."""
    if mode not in {"abs", "pos", "neg"}:
        raise ParameterError("mode must be 'abs', 'pos', or 'neg'.")
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (times >= tmin) & (times <= tmax)
    if int(mask.sum()) < 1:
        raise ParameterError(f"No samples in [{tmin}, {tmax}].")
    segment = values[mask]
    window = times[mask]
    if not np.isfinite(segment).any():
        raise ParameterError(f"No finite samples in [{tmin}, {tmax}].")
    if mode == "pos":
        index = int(np.nanargmax(segment))
    elif mode == "neg":
        index = int(np.nanargmin(segment))
    else:
        index = int(np.nanargmax(np.abs(segment)))
    return {"latency": float(window[index]), "amplitude": float(segment[index]), "mode": mode}


def modulation_index(
    x: np.ndarray,
    sfreq: float,
    *,
    phase_band: tuple[float, float] = (0.5, 4.0),
    amplitude_band: tuple[float, float] = (8.0, 13.0),
    n_bins: int = 18,
) -> float:
    """Tort modulation index for one channel.

    The index is the Kullback–Leibler divergence of the phase-binned
    mean amplitude from a uniform distribution, divided by ``log(n_bins)``
    (Tort, Komorowski, Eichenbaum, and Kopell, 2010). A phase bin with
    no samples has no mean amplitude. The index is then missing. An
    illegal band raises :class:`~eegmath.errors.ParameterError` rather
    than returning a missing value.
    """
    x = np.asarray(x, dtype=float)
    if n_bins < 4:
        raise ParameterError("n_bins must be at least 4.")
    if x.size < int(sfreq) or not np.isfinite(x).all():
        return np.nan
    try:
        phase = np.angle(signal.hilbert(_bandpass(x, sfreq, *phase_band)))
        amplitude = np.abs(signal.hilbert(_bandpass(x, sfreq, *amplitude_band)))
    except ParameterError:
        raise
    except ValueError:
        return np.nan
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    which = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(which, minlength=n_bins)
    # Tort's bin mean is undefined when a bin contains no samples. The
    # reference MATLAB implementation then returns a non-finite index.
    # Substituting zero invents an amplitude and pushes the divergence
    # toward a more peaked distribution.
    if np.any(counts == 0):
        return np.nan
    means = np.array([amplitude[which == i].mean() for i in range(n_bins)])
    total = float(means.sum())
    if total <= 0 or not np.isfinite(total):
        return np.nan
    distribution = means / total
    positive = distribution > 0
    kl = float(np.sum(distribution[positive] * np.log(distribution[positive] * n_bins)))
    return kl / np.log(n_bins)


def modulation_index_channels(eeg: EEG, **kwargs) -> pd.DataFrame:
    """Modulation index for each channel.

    Epoched data contribute one index per epoch. The returned value is
    the mean of those finite indices. Averaging the voltage first would
    cancel phase that is not locked to the epoch clock, and the coupling
    of that average is a different quantity.
    """
    if eeg.epoched:
        values = []
        for channel in range(eeg.n_channels):
            scores = [
                modulation_index(eeg.data[epoch, channel], eeg.sfreq, **kwargs)
                for epoch in range(eeg.n_epochs)
            ]
            finite = [score for score in scores if np.isfinite(score)]
            values.append(float(np.mean(finite)) if finite else np.nan)
        note = "mean of within-epoch modulation indices"
    else:
        values = [modulation_index(eeg.data[i], eeg.sfreq, **kwargs) for i in range(eeg.n_channels)]
        note = "computed on the continuous recording"
    return pd.DataFrame({"channel": list(eeg.ch_names), "modulation_index": values, "note": note})


def adf_stationarity(x: np.ndarray, *, autolag: str = "AIC") -> dict:
    """Augmented Dickey-Fuller test. The null hypothesis is a unit root.

    A rank-deficient regression returns a missing p-value. That includes a
    pure sinusoid, whether or not the installed statsmodels warns about it.
    A statistic larger than one million is treated as that failure. A
    completed test is much smaller.
    """
    from statsmodels.tsa.stattools import adfuller

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 16:
        return {"statistic": np.nan, "p_value": np.nan, "n_obs": int(x.size), "note": "series too short"}
    # Dropping an interior gap and concatenating the sides would invent
    # a sample-to-sample step. The test needs equally spaced observations.
    if not np.isfinite(x).all():
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_obs": int(x.size),
            "note": "series contains non-finite samples; they were not removed",
        }
    if float(np.std(x)) == 0:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_obs": int(x.size),
            "null": "unit root",
            "note": "series is constant; a unit-root regression is not defined",
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore",
            message="adfuller currently returns",
            category=FutureWarning,
        )
        try:
            stat, p_value, usedlag, nobs, critical, _icbest = adfuller(x, autolag=autolag)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "n_obs": int(x.size),
                "null": "unit root",
                "note": f"The test was not computed ({exc}).",
            }
    singular = [
        item
        for item in caught
        if "rank" in str(item.message).lower() or item.category.__name__ == "SingularMatrixWarning"
    ]
    # A completed Dickey-Fuller statistic is a small number. A pure sine can
    # make the regression singular. statsmodels 0.15 warns. 0.14.6 returns a
    # statistic near 1e13 and a p-value of 0 without that warning. Either
    # sign means the p-value was not computed.
    if singular or not np.isfinite(stat) or abs(float(stat)) > 1e6:
        detail = f" {singular[0].message}" if singular else ""
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_obs": int(x.size),
            "null": "unit root",
            "note": (
                "The unit-root regression was rank-deficient, so no p-value is reported."
                + detail
            ),
        }
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "used_lag": int(usedlag),
        "n_obs": int(nobs),
        "critical_values": {key: float(val) for key, val in critical.items()},
        "null": "unit root",
        "note": "Rejecting a unit root is not a general demonstration of stationarity.",
    }


def kpss_stationarity(x: np.ndarray, *, regression: str = "c") -> dict:
    """Kwiatkowski–Phillips–Schmidt–Shin stationarity test.

    The null is level stationarity when ``regression="c"`` and trend
    stationarity when ``regression="ct"`` (Kwiatkowski, Phillips, Schmidt,
    and Shin, 1992). This is the complement of the augmented Dickey–Fuller
    null, which is a unit root. A small p-value is evidence against
    stationarity. statsmodels interpolates the p-value from the published
    table and returns a boundary value, 0.01 or 0.1, when the statistic
    falls outside that table. Those two numbers are bounds, not precise
    tail probabilities. A constant series and a non-finite sample are
    missing. Interior gaps are not dropped and concatenated.
    """
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import kpss

    if regression not in {"c", "ct"}:
        raise ParameterError("regression must be 'c' or 'ct'.")
    x = np.asarray(x, dtype=float).reshape(-1)
    null = "level stationarity" if regression == "c" else "trend stationarity"
    if x.size < 16:
        return {"statistic": np.nan, "p_value": np.nan, "n_obs": int(x.size), "null": null, "note": "series too short"}
    if not np.isfinite(x).all():
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_obs": int(x.size),
            "null": null,
            "note": "series contains non-finite samples; they were not removed",
        }
    if float(np.std(x)) == 0:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_obs": int(x.size),
            "null": null,
            "note": "series is constant; the stationarity regression is not defined",
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        try:
            stat, p_value, lags, _critical = kpss(x, regression=regression, nlags="auto")
        except ValueError as exc:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "n_obs": int(x.size),
                "null": null,
                "note": f"The test was not computed ({exc}).",
            }
    note = (
        "The null is stationarity, not a unit root. "
        "Rejecting it is not by itself a demonstration of a unit root."
    )
    for item in caught:
        if issubclass(item.category, InterpolationWarning):
            note += " " + str(item.message).strip()
            break
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "lags": int(lags),
        "n_obs": int(x.size),
        "regression": regression,
        "null": null,
        "note": note,
    }


def _series_for_dependence(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ParameterError("Autocorrelation expects a one-dimensional series.")
    return x


def _dependence_lags(n: int, n_lags: int | None) -> int:
    if n_lags is None:
        if n < 4:
            return 1
        cap = int(np.floor(10.0 * np.log10(n)))
        return int(max(1, min(cap, n // 2 - 1)))
    if int(n_lags) < 1:
        raise ParameterError("n_lags must be at least 1.")
    return int(n_lags)


def _missing_dependence(n: int, n_lags: int, key: str, note: str) -> dict:
    return {
        "lag": np.arange(n_lags + 1),
        key: np.full(n_lags + 1, np.nan),
        "n_obs": n,
        "note": note,
    }


def autocorrelation(x: np.ndarray, *, n_lags: int | None = None, adjusted: bool = False) -> dict:
    """Sample autocorrelation of one equally spaced series.

    The default divides the lag product by ``n`` (the biased estimator).
    ``adjusted=True`` divides lag ``k`` by ``n - k``. Lag 0 is 1 when the
    series has positive variance. A constant series, a non-finite sample,
    or a series shorter than 8 samples is missing. Epochs are not joined.
    No white-noise test and no confidence interval is attached: a Bartlett
    interval answers a modelling question this function does not ask.

    The population autocorrelation of an autoregressive series of order 1
    with coefficient ``phi`` is ``phi`` to the power of the lag. The
    function estimates that curve; it does not fit ``phi``. The numerical
    work is :func:`statsmodels.tsa.stattools.acf`. That function cites
    Parzen (1963) and Brockwell and Davis; those publications were not
    opened for this wrapper.
    """
    from statsmodels.tsa.stattools import acf

    x = _series_for_dependence(x)
    n = int(x.size)
    lags = _dependence_lags(n, n_lags)
    if lags >= n:
        raise ParameterError("n_lags must be shorter than the series.")
    undefined = "The autocorrelation is missing. The series is too short, constant, or non-finite."
    if n < 8 or not np.isfinite(x).all() or float(np.std(x)) == 0:
        out = _missing_dependence(n, lags, "acf", undefined)
        out["adjusted"] = bool(adjusted)
        return out
    values = np.asarray(acf(x, adjusted=bool(adjusted), nlags=lags, fft=True), dtype=float)
    if adjusted:
        note = "Unbiased sample autocorrelation: lag k is divided by n-k. This is not a white-noise test."
    else:
        note = "Biased sample autocorrelation: every lag is divided by n. This is not a white-noise test."
    return {"lag": np.arange(values.size), "acf": values, "n_obs": n, "adjusted": bool(adjusted), "note": note}


def partial_autocorrelation(x: np.ndarray, *, n_lags: int | None = None) -> dict:
    """Yule–Walker partial autocorrelation.

    The autocovariance is divided by ``n`` at every lag (statsmodels
    method ``ywmle``). Dividing lag ``k`` by ``n - k`` is the adjusted
    form. statsmodels documents that form as having larger error than
    the unadjusted Yule–Walker estimator in its own comparison; that
    note does not cite a paper, and the comparison was not re-run here.
    Lag 0 is 1. For a long autoregressive series of order 1, lag 1
    estimates the coefficient and later lags stay near zero. A requested
    lag at or above ``n/2`` is outside that recursion and raises
    :class:`~eegmath.errors.ParameterError`.
    """
    from statsmodels.tsa.stattools import pacf

    x = _series_for_dependence(x)
    n = int(x.size)
    lags = _dependence_lags(n, n_lags)
    if lags >= n / 2:
        raise ParameterError("partial autocorrelation n_lags must be below half the series length.")
    undefined = "The partial autocorrelation is missing. The series is too short, constant, or non-finite."
    if n < 8 or not np.isfinite(x).all() or float(np.std(x)) == 0:
        return _missing_dependence(n, lags, "pacf", undefined)
    values = np.asarray(pacf(x, nlags=lags, method="ywmle"), dtype=float)
    return {
        "lag": np.arange(values.size),
        "pacf": values,
        "n_obs": n,
        "method": "ywmle",
        "note": "Yule-Walker partial autocorrelation with every lag divided by n. This is not an order-selection test.",
    }


def autocorrelation_table(
    eeg: EEG,
    *,
    epoch: int = 0,
    n_lags: int | None = None,
    adjusted: bool = False,
) -> pd.DataFrame:
    """Autocorrelation of each channel. Epoched input uses one epoch."""
    data = eeg.data[epoch] if eeg.epoched else eeg.data
    rows = []
    for name, channel in zip(eeg.ch_names, data):
        result = autocorrelation(channel, n_lags=n_lags, adjusted=adjusted)
        for lag, value in zip(result["lag"], result["acf"]):
            rows.append(
                {
                    "channel": name,
                    "epoch": epoch if eeg.epoched else pd.NA,
                    "lag": int(lag),
                    "acf": float(value),
                    "n_obs": result["n_obs"],
                }
            )
    return pd.DataFrame(rows)


def stationarity_table(eeg: EEG, *, epoch: int = 0) -> pd.DataFrame:
    """ADF test per channel. Epoched input uses one epoch, not a concatenation."""
    data = eeg.data[epoch] if eeg.epoched else eeg.data
    rows = []
    for name, channel in zip(eeg.ch_names, data):
        row = adf_stationarity(channel)
        row["channel"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def band_power_timecourse(
    eeg: EEG,
    fmin: float,
    fmax: float,
    *,
    window_sec: float = 1.0,
    step_sec: float | None = None,
) -> pd.DataFrame:
    """Windowed band power. Windows stay inside each epoch."""
    from eegmath.spectral.psd import band_power, welch_psd

    step = window_sec / 2 if step_sec is None else step_sec
    width = int(round(window_sec * eeg.sfreq))
    hop = max(1, int(round(step * eeg.sfreq)))
    if width < 8 or width > eeg.n_times:
        raise ParameterError("window_sec does not fit inside the recording.")
    source = eeg.data if eeg.epoched else eeg.data[None, ...]
    rows = []
    for epoch in range(source.shape[0]):
        starts = range(0, eeg.n_times - width + 1, hop)
        for start in starts:
            segment = source[epoch, :, start : start + width]
            freqs, psd = welch_psd(segment, eeg.sfreq, n_per_seg=min(width, max(8, width // 2)))
            power = band_power(freqs, psd, fmin, fmax, closed=True)
            onset = eeg.tmin + start / eeg.sfreq
            for channel, value in zip(eeg.ch_names, power):
                rows.append(
                    {
                        "epoch": epoch if eeg.epoched else pd.NA,
                        "channel": channel,
                        "onset": onset,
                        "power": float(value),
                    }
                )
    return pd.DataFrame(rows)
