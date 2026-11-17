"""Sensor-space spectral connectivity.

Estimators are accumulated from periodic Hann-windowed FFT segments,
the same taper Welch's method uses. Epoched
input is not concatenated: each epoch contributes its own segments, and
the cross-spectrum is the average across those segments. This avoids
treating an epoch boundary as a real sample-to-sample transition.

Metrics
-------
coherence
    Magnitude-squared coherence. Sensitive to volume conduction and
    positively biased when the number of segments is small.
pli
    Phase-lag index, the absolute mean sign of the imaginary
    cross-spectrum (Stam, Nolte, and Daffertshofer, 2007).
wpli
    Debiased squared weighted phase-lag index (Vinck et al., 2011).
    The sample estimate lies in ``[-1, 1]`` and can be negative: it is a
    pairwise average of products of imaginary cross-spectra, and that
    average is an estimate of a square, not a square of an estimate.
    Negative values are kept. Clipping them to zero would put back the
    small-sample positive bias the debiasing step removes. A negative
    value is not a lead or a lag, and it is not an Onnela weight.
    Fewer than two segments leaves the pairwise estimator undefined, so
    those entries are missing rather than zero. A denominator that
    vanishes because every imaginary part is zero is reported as zero:
    there is then no phase-lag weight.
plv
    Phase-locking value, the magnitude of the mean unit-length
    cross-spectrum (Lachaux et al., 1999).
ppc
    Pairwise phase consistency (Vinck, van Wingerden, Womelsdorf, Fries,
    and Pennartz, 2010). For ``n`` segments with phase-locking value
    ``P`` it is ``(n P² − 1) / (n − 1)``, the average pairwise cosine of
    the segment phase differences. The population value equals the
    squared phase-locking value, and the sample value is not pulled
    upward by a small segment count. It lies in ``[-1, 1]`` and can be
    negative. These samples are Fourier segments, the same segments
    used for the phase-locking value, not spike phases.
imcoh
    Imaginary part of coherency (Nolte et al., 2004). The cross-spectrum
    is ``X_i conj(X_j)``, so a positive ``imcoh[i, j]`` means channel
    ``i`` leads channel ``j``. The band matrix is the mean of the signed
    values, so opposite lags inside one band cancel. Node-feature columns
    instead average the absolute value across frequency before the node
    mean. The column says how large the lag is, not which channel leads.

All of these are sensor-space statistics. They are not evidence of
anatomical connectivity. Zero-lag mixing from a common reference or from
volume conduction inflates coherence and PLV and is exactly what PLI,
wPLI, and imaginary coherency are designed to reduce. A small segment
count leaves a positive bias in coherence, phase-locking value, and
PLI. Pairwise phase consistency is the unbiased companion of the
phase-locking value. Values are withheld when fewer than
``min_segments`` segments are available.
"""

from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np
from scipy import signal, stats

from eegmath.bands import DEFAULT_BANDS, DEFAULT_FREQ_MAX, DEFAULT_FREQ_MIN, partition_masks
from eegmath.errors import ParameterError
from eegmath.spectral.psd import default_n_per_seg

_EPS = 1e-20
# Bins this far below a channel's peak autospectrum are numerical nulls.
# A periodic Hann transform of a pure tone is exactly zero two bins away
# from the tone; treating that 0/0 ratio as zero coupling pulls a band
# average toward zero.
_RELATIVE_POWER_FLOOR = 1e-12
METRICS = ("coherence", "pli", "wpli", "plv", "ppc", "imcoh")


def _iter_epochs(data: np.ndarray) -> list[np.ndarray]:
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        return [data[None, :]]
    if data.ndim == 2:
        return [data]
    if data.ndim == 3:
        return [data[i] for i in range(data.shape[0])]
    raise ParameterError("Connectivity data must be 1D, 2D, or 3D.")


def _detrend_segment(segment: np.ndarray, detrend: str) -> np.ndarray:
    if detrend == "constant":
        return segment - segment.mean(axis=1, keepdims=True)
    if detrend == "linear":
        return signal.detrend(segment, axis=-1, type="linear")
    if detrend in {"none", "None"}:
        return segment
    raise ParameterError("detrend must be 'constant', 'linear', or 'none'.")


def _accumulate(
    data: np.ndarray,
    sfreq: float,
    n_per_seg: int,
    overlap: float,
    metrics: tuple[str, ...],
    detrend: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    epochs = _iter_epochs(data)
    n_channels = epochs[0].shape[0]
    noverlap = min(int(round(overlap * n_per_seg)), n_per_seg - 1)
    step = n_per_seg - noverlap
    # Periodic Hann, the same window Welch's method uses. The symmetric
    # NumPy window puts zeros at both endpoints and is a different taper.
    window = signal.windows.hann(n_per_seg, sym=False)
    freqs = np.fft.rfftfreq(n_per_seg, d=1.0 / sfreq)
    n_freq = freqs.size
    want = set(metrics)
    sxy = np.zeros((n_channels, n_channels, n_freq), dtype=np.complex128)
    sxx = np.zeros((n_channels, n_freq), dtype=np.float64)
    sign_sum = np.zeros((n_channels, n_channels, n_freq)) if "pli" in want else None
    im_sum = np.zeros((n_channels, n_channels, n_freq)) if "wpli" in want else None
    abs_im_sum = np.zeros((n_channels, n_channels, n_freq)) if "wpli" in want else None
    sq_im_sum = np.zeros((n_channels, n_channels, n_freq)) if "wpli" in want else None
    plv_sum = (
        np.zeros((n_channels, n_channels, n_freq), dtype=np.complex128)
        if "plv" in want or "ppc" in want
        else None
    )
    n_used = 0
    for epoch in epochs:
        if epoch.shape[0] != n_channels:
            raise ParameterError("Every epoch must have the same number of channels.")
        n_times = epoch.shape[1]
        if n_times < n_per_seg:
            continue
        for start in range(0, n_times - n_per_seg + 1, step):
            segment = _detrend_segment(epoch[:, start : start + n_per_seg], detrend)
            spectrum = np.fft.rfft(segment * window, axis=-1)
            cross = spectrum[:, None, :] * np.conj(spectrum[None, :, :])
            sxy += cross
            sxx += np.abs(spectrum) ** 2
            if sign_sum is not None or im_sum is not None or plv_sum is not None:
                imag = cross.imag
                # A relative floor keeps floating-point noise from flipping the
                # sign of a true zero-lag pair. Real phase lags are far larger.
                imag_floor = 1e-8 * (np.abs(cross) + 1e-30)
                imag = np.where(np.abs(imag) <= imag_floor, 0.0, imag)
            if sign_sum is not None:
                sign_sum += np.sign(imag)
            if im_sum is not None:
                im_sum += imag
                abs_im_sum += np.abs(imag)
                sq_im_sum += imag**2
            if plv_sum is not None:
                plv_sum += cross / (np.abs(cross) + _EPS)
            n_used += 1
    matrices: dict[str, np.ndarray] = {}
    if n_used == 0:
        nan = np.full((n_channels, n_channels, n_freq), np.nan)
        return freqs, {metric: nan for metric in metrics}, 0
    sxy /= n_used
    sxx /= n_used
    denom = sxx[:, None, :] * sxx[None, :, :]
    if "coherence" in want:
        matrices["coherence"] = (np.abs(sxy) ** 2) / (denom + _EPS)
    if "imcoh" in want:
        matrices["imcoh"] = sxy.imag / (np.sqrt(denom) + _EPS)
    if "pli" in want:
        matrices["pli"] = np.abs(sign_sum / n_used)
    if "wpli" in want:
        # Vinck et al. 2011, the debiased estimator of squared wPLI.
        # FieldTrip returns NaN when only one segment is available,
        # because the sums run over pairs of segments.
        if n_used < 2:
            matrices["wpli"] = np.full((n_channels, n_channels, n_freq), np.nan)
        else:
            num = im_sum**2 - sq_im_sum
            den = abs_im_sum**2 - sq_im_sum
            wpli = np.zeros_like(num)
            usable = np.abs(den) > _EPS
            with np.errstate(divide="ignore", invalid="ignore"):
                wpli[usable] = num[usable] / den[usable]
            # Float noise can step just outside the closed interval of a
            # weighted average of ±1. Values inside [-1, 0) stay there.
            matrices["wpli"] = np.clip(wpli, -1.0, 1.0)
    if "plv" in want:
        matrices["plv"] = np.abs(plv_sum / n_used)
    if "ppc" in want:
        # Vinck et al. 2010, eq. 14: the mean pairwise cosine of the
        # segment phase differences. Equivalent form (N * PLV^2 - 1) / (N - 1).
        if n_used < 2:
            matrices["ppc"] = np.full((n_channels, n_channels, n_freq), np.nan)
        else:
            magnitude_sq = np.abs(plv_sum) ** 2
            ppc = (magnitude_sq - n_used) / (n_used * (n_used - 1.0))
            matrices["ppc"] = np.clip(ppc, -1.0, 1.0)
    peak = np.max(sxx, axis=-1)
    floor = _RELATIVE_POWER_FLOOR * np.maximum(peak, 0.0)
    present = sxx > floor[:, None]
    pair = present[:, None, :] & present[None, :, :]
    for name, matrix in list(matrices.items()):
        masked = np.asarray(matrix, dtype=float).copy()
        masked[~pair] = np.nan
        matrices[name] = masked
    return freqs, matrices, n_used


def _band_mean(matrix: np.ndarray, mask: np.ndarray, *, absolute: bool = False) -> np.ndarray:
    if int(np.count_nonzero(mask)) == 0:
        return np.full(matrix.shape[:2], np.nan)
    block = np.abs(matrix[:, :, mask]) if absolute else matrix[:, :, mask]
    if not np.isfinite(block).any():
        return np.full(matrix.shape[:2], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(block, axis=-1)


def _zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    if out.ndim != 2:
        return out
    np.fill_diagonal(out, 0.0)
    return out


def _off_diagonal(matrix: np.ndarray, *, absolute: bool = False) -> np.ndarray:
    values = np.abs(matrix) if absolute else np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ParameterError("A node summary expects a square connectivity matrix.")
    out = values.copy()
    np.fill_diagonal(out, np.nan)
    return out


def node_mean(matrix: np.ndarray, *, absolute: bool = False) -> np.ndarray:
    """Mean coupling of each node with the other nodes.

    Pairs that are missing stay out of the mean. One undefined pair does
    not erase the remaining pairs. A node with no finite pair is missing.
    """
    values = _off_diagonal(matrix, absolute=absolute)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=1)


def node_strength(matrix: np.ndarray, *, absolute: bool = False) -> np.ndarray:
    """Sum of finite coupling weights from each node, excluding the diagonal.

    A node whose every pair is missing is missing. Missing pairs are not
    entered as zeros, and a single missing pair does not erase the sum.
    """
    values = _off_diagonal(matrix, absolute=absolute)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        total = np.nansum(values, axis=1).astype(float)
    total[~np.isfinite(values).any(axis=1)] = np.nan
    return total


def clustering_coefficient(matrix: np.ndarray) -> np.ndarray:
    """Onnela weighted clustering for weights already in ``[0, 1]``.

    The geometric mean is not invariant to dividing every weight by the
    largest edge, so this function does not rescale. A weight above 1,
    or a weight below zero by more than a rounding tolerance, raises
    :class:`~eegmath.errors.ParameterError`. A non-finite weight is not
    an edge: it does not add a neighbour, and a missing link between two
    neighbours adds nothing to the triangle sum, as a weight of zero
    would. A triangle of weight ``w`` has coefficient ``w``.
    """
    raw = _zero_diagonal(np.asarray(matrix, dtype=float))
    finite = np.isfinite(raw)
    if np.any(finite & (raw < -1e-8)):
        raise ParameterError(
            "Onnela clustering needs non-negative weights in [0, 1]. "
            "Apply abs() yourself when the sign is not part of the weight."
        )
    if np.any(finite & (raw > 1.0 + 1e-6)):
        raise ParameterError(
            "Onnela clustering expects weights in [0, 1]. "
            "Divide by the largest weight explicitly if that normalisation is intended."
        )
    w = np.where(finite, np.clip(raw, 0.0, 1.0), 0.0)
    if not np.any(w > 0):
        return np.zeros(w.shape[0])
    n = w.shape[0]
    coeff = np.zeros(n)
    for i in range(n):
        neighbours = np.flatnonzero(w[i] > 0)
        degree = neighbours.size
        if degree < 2:
            continue
        weights = w[i, neighbours]
        sub = w[np.ix_(neighbours, neighbours)]
        triangles = np.sum((weights[:, None] * weights[None, :] * sub) ** (1.0 / 3.0))
        coeff[i] = triangles / (degree * (degree - 1))
    return coeff


def _cross_spectra(
    data: np.ndarray,
    sfreq: float,
    *,
    bands: Mapping[str, tuple[float, float]] | None,
    metrics: tuple[str, ...],
    n_per_seg: int | None,
    segment_sec: float,
    overlap: float,
    detrend: str,
) -> tuple[np.ndarray, dict[str, tuple[float, float]], np.ndarray, dict[str, np.ndarray], int, dict]:
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ParameterError(f"Unknown connectivity metrics {unknown}. Choose from {METRICS}.")
    resolved = dict(DEFAULT_BANDS if bands is None else bands)
    sample = _iter_epochs(np.asarray(data, dtype=float))[0]
    n_per = (
        default_n_per_seg(sfreq, sample.shape[-1], segment_sec)
        if n_per_seg is None
        else int(n_per_seg)
    )
    n_per = min(n_per, sample.shape[-1])
    freqs, full, n_used = _accumulate(data, sfreq, n_per, overlap, tuple(metrics), detrend)
    meta: dict[str, np.ndarray | int] = {"freqs": freqs, "n_segments": n_used, "n_per_seg": n_per}
    return sample, resolved, freqs, full, n_used, meta


def _withheld_matrices(
    metrics: tuple[str, ...],
    bands: Mapping[str, tuple[float, float]],
    n_channels: int,
) -> dict[str, dict[str, np.ndarray]]:
    nan = np.full((n_channels, n_channels), np.nan)
    out: dict[str, dict[str, np.ndarray]] = {}
    labels = list(bands) + ["broadband"]
    for metric in metrics:
        out[metric] = {name: nan.copy() for name in labels}
    return out


def _band_matrices(
    freqs: np.ndarray,
    cubes: dict[str, np.ndarray],
    bands: Mapping[str, tuple[float, float]],
    sfreq: float,
    freq_min: float,
    freq_max: float,
    *,
    absolute_metrics: tuple[str, ...] = (),
) -> dict[str, dict[str, np.ndarray]]:
    absolute = set(absolute_metrics)
    out: dict[str, dict[str, np.ndarray]] = {}
    broadband = (freqs >= freq_min) & (freqs <= min(freq_max, sfreq / 2.0))
    for metric, cube in cubes.items():
        use_abs = metric in absolute
        out[metric] = {}
        for name, mask in partition_masks(freqs, bands):
            out[metric][name] = _band_mean(cube, mask, absolute=use_abs)
        out[metric]["broadband"] = _band_mean(cube, broadband, absolute=use_abs)
    return out


def _warn_if_too_few_segments(
    n_used: int,
    min_segments: int,
    segment_sec: float,
    overlap: float,
) -> None:
    if n_used >= min_segments:
        return
    warnings.warn(
        f"Connectivity is missing because {n_used} Fourier segments were accumulated "
        f"and {min_segments} are required. Segments are {segment_sec:g} s with overlap "
        f"{overlap:g}. The values are NaN, not zero coupling. At the default 2 s "
        "segments and 50% overlap, 20 s of continuous data yields 19 segments and "
        "21 s yields 20.",
        UserWarning,
        stacklevel=3,
    )


def connectivity_matrices(
    data: np.ndarray,
    sfreq: float,
    *,
    bands: Mapping[str, tuple[float, float]] | None = None,
    metrics: tuple[str, ...] = METRICS,
    n_per_seg: int | None = None,
    segment_sec: float = 2.0,
    overlap: float = 0.5,
    min_segments: int = 20,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    detrend: str = "constant",
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray | int]]:
    """Band-averaged connectivity matrices.

    Keys of the returned mapping are metric names. Each value maps a band
    name, plus ``"broadband"``, to a square matrix. The band value is the
    mean over bins in which both channels have autospectral power above
    ``1e-12`` of that channel's peak. Bins below that floor are numerical
    nulls and are omitted rather than entered as zero coupling.
    """
    sample, resolved, freqs, full, n_used, meta = _cross_spectra(
        data,
        sfreq,
        bands=bands,
        metrics=metrics,
        n_per_seg=n_per_seg,
        segment_sec=segment_sec,
        overlap=overlap,
        detrend=detrend,
    )
    if n_used < min_segments:
        meta["sufficient_segments"] = False
        _warn_if_too_few_segments(n_used, min_segments, segment_sec, overlap)
        return _withheld_matrices(tuple(metrics), resolved, sample.shape[0]), meta
    meta["sufficient_segments"] = True
    return _band_matrices(freqs, full, resolved, sfreq, freq_min, freq_max), meta


def connectivity_node_features(
    data: np.ndarray,
    sfreq: float,
    **kwargs,
) -> tuple[dict[str, np.ndarray], dict]:
    """Per-channel mean coupling for each requested metric and band.

    Coherence, PLI, and phase-locking value are non-negative. Debiased
    wPLI and pairwise phase consistency can be negative, and that sign
    is kept when the values are averaged across frequency.
    Imaginary coherency is averaged as an absolute value across frequency,
    then the node mean is taken. The signed band matrix is still returned
    under ``meta["matrices"]``. A missing pair is omitted from the node mean.
    """
    known = {
        "bands",
        "metrics",
        "n_per_seg",
        "segment_sec",
        "overlap",
        "min_segments",
        "freq_min",
        "freq_max",
        "detrend",
    }
    unknown = sorted(set(kwargs) - known)
    if unknown:
        raise ParameterError(f"Unknown connectivity arguments {unknown}.")
    sample, resolved, freqs, full, n_used, meta = _cross_spectra(
        data,
        sfreq,
        bands=kwargs.get("bands"),
        metrics=kwargs.get("metrics", METRICS),
        n_per_seg=kwargs.get("n_per_seg"),
        segment_sec=kwargs.get("segment_sec", 2.0),
        overlap=kwargs.get("overlap", 0.5),
        detrend=kwargs.get("detrend", "constant"),
    )
    freq_min = kwargs.get("freq_min", DEFAULT_FREQ_MIN)
    freq_max = kwargs.get("freq_max", DEFAULT_FREQ_MAX)
    min_segments = kwargs.get("min_segments", 20)
    metrics = tuple(kwargs.get("metrics", METRICS))
    if n_used < min_segments:
        matrices = _withheld_matrices(metrics, resolved, sample.shape[0])
        meta["sufficient_segments"] = False
        _warn_if_too_few_segments(
            n_used,
            min_segments,
            float(kwargs.get("segment_sec", 2.0)),
            float(kwargs.get("overlap", 0.5)),
        )
    else:
        meta["sufficient_segments"] = True
        matrices = _band_matrices(
            freqs,
            full,
            resolved,
            sfreq,
            freq_min,
            freq_max,
            absolute_metrics=("imcoh",) if "imcoh" in full else (),
        )
    # The returned matrices stay signed. Node columns for imaginary
    # coherency use the magnitude average computed above; rebuild the
    # signed matrices for the metadata when that average was requested.
    if meta["sufficient_segments"] and "imcoh" in full:
        signed = _band_matrices(freqs, full, resolved, sfreq, freq_min, freq_max)
    else:
        signed = matrices
    features: dict[str, np.ndarray] = {}
    for metric, band_matrices in matrices.items():
        for band, matrix in band_matrices.items():
            column = f"{band}_{metric}" if band != "broadband" else f"global_{metric}"
            features[column] = node_mean(matrix)
    return features, {**meta, "matrices": signed}


def region_connectivity(
    matrix: np.ndarray,
    ch_names: list[str] | tuple[str, ...],
    region_a: list[str] | tuple[str, ...],
    region_b: list[str] | tuple[str, ...],
) -> float:
    """Mean of the submatrix linking two disjoint channel sets.

    Channels absent from ``ch_names`` are ignored. The diagonal is not
    included. Returns NaN when either set is empty after that intersection.
    """
    names = list(ch_names)
    index = {name: i for i, name in enumerate(names)}
    a_names = [name for name in region_a if name in index]
    b_names = [name for name in region_b if name in index]
    overlap = sorted(set(a_names) & set(b_names))
    if overlap:
        raise ParameterError(f"Regions overlap on {overlap}. Pass disjoint channel sets.")
    a = [index[name] for name in a_names]
    b = [index[name] for name in b_names]
    if not a or not b:
        return float("nan")
    block = np.asarray(matrix, dtype=float)[np.ix_(a, b)]
    if not np.isfinite(block).any():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(block))


def _correlation_matrix(epoch: np.ndarray, method: str) -> np.ndarray:
    n_channels = epoch.shape[0]
    if n_channels == 1:
        varied = bool(np.isfinite(epoch[0]).all() and np.std(epoch[0]) > 0)
        return np.array([[1.0 if varied else np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "pearson":
            matrix = np.corrcoef(epoch)
        else:
            # Rank first, then Pearson. scipy.spearmanr returns a scalar
            # for two variables and a matrix for more, which is easy to
            # mis-shape. A non-finite channel stays missing.
            ranked = np.vstack(
                [
                    stats.rankdata(row) if np.isfinite(row).all() else np.full(row.size, np.nan)
                    for row in epoch
                ]
            )
            matrix = np.corrcoef(ranked)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    if matrix.shape != (n_channels, n_channels):
        raise ParameterError("The correlation routine did not return one coefficient per channel pair.")
    return matrix


def sensor_correlation(data: np.ndarray, *, method: str = "pearson") -> np.ndarray:
    """Time-domain correlation between channels.

    ``method`` is ``pearson`` or ``spearman``. Epoched input is not
    concatenated. Each epoch has its own matrix, and epochs are combined
    by the mean of Fisher z values. A constant channel has missing
    correlations. This is not a spectral connectivity measure, and a
    shared reference or a common source raises it.
    """
    if method not in {"pearson", "spearman"}:
        raise ParameterError("method must be 'pearson' or 'spearman'.")
    epochs = _iter_epochs(data)
    n_channels = epochs[0].shape[0]
    if any(epoch.shape[0] != n_channels for epoch in epochs):
        raise ParameterError("Every epoch must have the same number of channels.")
    stack = np.stack([_correlation_matrix(epoch, method) for epoch in epochs], axis=0)
    finite = np.isfinite(stack)
    clipped = np.clip(stack, -1.0 + 1e-12, 1.0 - 1e-12)
    z = np.full(stack.shape, np.nan, dtype=float)
    z[finite] = np.arctanh(clipped[finite])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_z = np.nanmean(z, axis=0)
    out = np.tanh(mean_z)
    for index in range(n_channels):
        out[index, index] = 1.0 if np.isfinite(stack[:, index, index]).any() else np.nan
    return out
