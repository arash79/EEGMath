"""Feature names, families, and the public extractor.

Each feature is a named column. Families are only a way to request a
group of columns. A feature can be computed without computing the rest
of its family, except that spectral columns share one Welch spectrum and
connectivity columns share one cross-spectrum.

Segment features are computed inside each epoch. They are not averaged
unless ``reduce_epochs=True``. Recording features use the whole
recording: connectivity pools FFT segments across epochs without
concatenating across epoch boundaries, and ``rms_cv`` is the coefficient
of variation of RMS across epochs or across internal windows.

Epochs are labelled in the ``epoch`` column. They are repeated
observations of one recording. Statistical routines must not treat those
rows as subjects.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from eegmath.algorithms.entropy import (
    approximate_entropy,
    lempel_ziv_complexity,
    multiscale_sample_entropy,
    permutation_entropy,
    sample_entropy,
    svd_entropy,
)
from eegmath.algorithms.fractal import detrended_fluctuation, higuchi_fd, katz_fd, petrosian_fd
from eegmath.algorithms.time_domain import (
    burst_suppression_ratio,
    envelope_lzc,
    hjorth_parameters,
    kurtosis,
    line_length,
    peak_to_peak,
    rms,
    skewness,
    zero_crossing_rate,
)
from eegmath.bands import DEFAULT_BANDS
from eegmath.connectivity.metrics import METRICS, connectivity_node_features
from eegmath.errors import ParameterError
from eegmath.recording import EEG, EEGCollection, attach_identity
from eegmath.spectral.psd import (
    RATIO_DEFINITIONS,
    aperiodic_fit,
    is_derived_spectral_column,
    recompute_power_ratios,
    spectral_feature_set,
    warn_if_ratios_left_missing,
)

TIME_FEATURES: tuple[str, ...] = (
    "rms",
    "line_length",
    "zero_crossing_rate",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
)
ENTROPY_FEATURES: tuple[str, ...] = (
    "lzc",
    "approximate_entropy",
    "sample_entropy",
    "permutation_entropy",
    "svd_entropy",
    "mse_mean",
    "mse_scale1",
    "mse_scale5",
    "mse_scale10",
)
COMPLEXITY_FEATURES: tuple[str, ...] = ("higuchi_fd", "petrosian_fd", "katz_fd", "dfa")
DYNAMICS_FEATURES: tuple[str, ...] = ("burst_suppression_ratio", "envelope_lzc")
VARIABILITY_FEATURES: tuple[str, ...] = ("rms_cv",)
APERIODIC_FEATURES: tuple[str, ...] = ("aperiodic_exponent", "aperiodic_offset")

ALIASES: dict[str, str] = {
    "pe": "permutation_entropy",
    "apen": "approximate_entropy",
    "sampen": "sample_entropy",
    "bsr": "burst_suppression_ratio",
}

SEGMENT_FAMILIES = {
    "time": TIME_FEATURES,
    "entropy": ENTROPY_FEATURES,
    "complexity": COMPLEXITY_FEATURES,
    "dynamics": DYNAMICS_FEATURES,
}


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters shared by a feature extraction call.

    ``None`` for a band mapping means :data:`eegmath.bands.DEFAULT_BANDS`.
    """

    bands: dict[str, tuple[float, float]] | None = None
    freq_min: float = 0.5
    freq_max: float = 45.0
    segment_sec: float = 2.0
    n_per_seg: int | None = None
    overlap: float = 0.5
    detrend: str = "constant"
    sef_percentiles: tuple[float, ...] = (50, 75, 90, 95)
    pe_order: int = 3
    pe_delay: int = 1
    entropy_order: int = 2
    # Each entry becomes a column ``mse_scale{n}``. The defaults are the
    # catalog names mse_scale1, mse_scale5, and mse_scale10.
    mse_scales: tuple[int, ...] = (1, 5, 10)
    mse_r_mode: str = "original"
    # Costa, Goldberger, and Peng (2002) illustrated 0.15. The default
    # matches sample entropy. ``mse_mean`` does not drop a non-finite scale.
    mse_r_fraction: float = 0.2
    higuchi_kmax: int = 10
    connectivity_min_segments: int = 20
    connectivity_metrics: tuple[str, ...] = METRICS
    bsr_threshold_uv: float = 5.0
    bsr_min_duration: float = 0.5
    envelope_band: tuple[float, float] = (1.0, 40.0)
    variability_window_sec: float = 2.0

    def resolved_bands(self) -> dict[str, tuple[float, float]]:
        return dict(DEFAULT_BANDS if self.bands is None else self.bands)


def spectral_names(config: FeatureConfig | None = None) -> tuple[str, ...]:
    config = config or FeatureConfig()
    names: list[str] = []
    for band in config.resolved_bands():
        names.extend(
            [f"{band}_power", f"{band}_relative", f"{band}_power_db", f"{band}_peak_freq"]
        )
    bandset = set(config.resolved_bands())
    for name, (numerator, denominator) in RATIO_DEFINITIONS.items():
        if numerator in bandset and denominator in bandset:
            names.append(name)
    if {"delta", "theta", "alpha", "beta"} <= bandset:
        names.extend(["slow_fast_ratio", "alpha_dominance"])
    names.extend(["total_power", "mean_freq", "median_freq", "spectral_entropy"])
    for pct in config.sef_percentiles:
        label = int(pct) if float(pct).is_integer() else pct
        names.append(f"sef{label}")
    names.extend(APERIODIC_FEATURES)
    # median_freq duplicates sef50 when 50 is requested. Keep both names.
    return tuple(dict.fromkeys(names))


def connectivity_names(config: FeatureConfig | None = None) -> tuple[str, ...]:
    config = config or FeatureConfig()
    names: list[str] = []
    for band in list(config.resolved_bands()) + ["global"]:
        for metric in config.connectivity_metrics:
            prefix = band if band != "global" else "global"
            names.append(f"{prefix}_{metric}")
    return tuple(names)


def entropy_names(config: FeatureConfig | None = None) -> tuple[str, ...]:
    """Entropy columns, including one ``mse_scale{n}`` per configured scale."""
    config = config or FeatureConfig()
    names = [
        "lzc",
        "approximate_entropy",
        "sample_entropy",
        "permutation_entropy",
        "svd_entropy",
        "mse_mean",
    ]
    for scale in config.mse_scales:
        names.append(f"mse_scale{int(scale)}")
    return tuple(dict.fromkeys(names))


def family_names(family: str, config: FeatureConfig | None = None) -> tuple[str, ...]:
    if family == "entropy":
        return entropy_names(config)
    if family in SEGMENT_FAMILIES:
        return SEGMENT_FAMILIES[family]
    if family == "spectral":
        return spectral_names(config)
    if family == "connectivity":
        return connectivity_names(config)
    if family == "variability":
        return VARIABILITY_FEATURES
    raise ParameterError(
        f"Unknown feature family '{family}'. "
        "Choose from time, spectral, entropy, complexity, dynamics, connectivity, variability."
    )


def all_feature_names(config: FeatureConfig | None = None) -> tuple[str, ...]:
    config = config or FeatureConfig()
    names: list[str] = []
    for family in ("time", "spectral", "entropy", "complexity", "dynamics", "connectivity", "variability"):
        names.extend(family_names(family, config))
    return tuple(dict.fromkeys(names))


def feature_quantity(name: str) -> str:
    """What a feature column contains.

    The ``units`` column of a feature row is the recording unit for every
    feature on that row. This string is the quantity of one column. A wide
    table cannot carry a different unit on each value, so the catalog is
    the place that distinction lives.
    """
    if name.endswith("_relative") or name == "burst_suppression_ratio":
        return "proportion"
    if name.endswith("_power_db"):
        return "decibels"
    if name.endswith("_power") or name in {"total_power", "hjorth_activity"}:
        return "recording unit squared"
    if name.endswith("_peak_freq") or name in {"mean_freq", "median_freq"} or name.startswith("sef"):
        return "hertz"
    if name in RATIO_DEFINITIONS or name in {"slow_fast_ratio", "alpha_dominance"}:
        return "power ratio"
    if name == "spectral_entropy":
        return "normalized entropy"
    if name == "aperiodic_exponent":
        return "log-log slope"
    if name == "aperiodic_offset":
        return "log10 power density"
    if name in {"rms", "peak_to_peak"}:
        return "recording unit"
    if name == "line_length":
        return "recording unit per second"
    if name == "zero_crossing_rate":
        return "per second"
    if name in {"skewness", "kurtosis"}:
        return "dimensionless moment"
    if name == "hjorth_mobility":
        return "radians per second"
    if name == "hjorth_complexity":
        return "dimensionless"
    if name == "rms_cv":
        return "coefficient of variation"
    return "dimensionless"


def list_features(family: str | None = None, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Table of feature names, families, scope, and quantity.

    ``quantity`` says what the number is. ``proportion`` and ``decibels``
    are not in the recording unit. ``recording unit`` and
    ``recording unit squared`` follow ``units`` on the feature row: volts
    for a file, microvolts only when the array was loaded that way.
    """
    config = config or FeatureConfig()
    families = (
        (family,)
        if family
        else ("time", "spectral", "entropy", "complexity", "dynamics", "connectivity", "variability")
    )
    rows = []
    for fam in families:
        scope = "recording" if fam in {"connectivity", "variability"} else "segment"
        for name in family_names(fam, config):
            rows.append(
                {
                    "feature": name,
                    "family": fam,
                    "scope": scope,
                    "quantity": feature_quantity(name),
                }
            )
    return pd.DataFrame(rows)


def _with_spectral_dependencies(names: Sequence[str], config: FeatureConfig) -> tuple[str, ...]:
    """Keep every absolute band power required to rebuild a requested ratio.

    Relative power uses the configured partition, not the columns the caller
    asked to see. Those extra power columns are computed and then dropped.
    """
    catalog = set(spectral_names(config))
    requested = [name for name in names if name in catalog]
    if not requested:
        return tuple(names)
    bands = config.resolved_bands()
    extra: list[str] = []
    if any(name.endswith("_relative") or name.endswith("_power_db") for name in requested):
        extra.extend(f"{band}_power" for band in bands)
    for name, (numerator, denominator) in RATIO_DEFINITIONS.items():
        if name in requested:
            extra.extend((f"{numerator}_power", f"{denominator}_power"))
    if "slow_fast_ratio" in requested or "alpha_dominance" in requested:
        extra.extend(f"{band}_power" for band in ("delta", "theta", "alpha", "beta") if band in bands)
    return tuple(dict.fromkeys((*names, *extra)))


def _keep_requested(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    identity = {
        "recording_id",
        "subject_id",
        "session_id",
        "condition",
        "sfreq",
        "units",
        "reference",
        "duration_sec",
        "highpass_hz",
        "lowpass_hz",
        "channel",
        "epoch",
        "scope",
        "aggregation",
        "onset",
        "event_code",
    }
    wanted = set(names)
    columns = [column for column in frame.columns if column in identity or column in wanted]
    return frame.loc[:, columns]


def resolve_features(features: str | Sequence[str], config: FeatureConfig | None = None) -> tuple[str, ...]:
    config = config or FeatureConfig()
    if isinstance(features, str):
        if features == "all":
            return all_feature_names(config)
        if features in SEGMENT_FAMILIES or features in {"spectral", "connectivity", "variability"}:
            return family_names(features, config)
        features = [features]
    known = set(all_feature_names(config))
    resolved: list[str] = []
    unknown: list[str] = []
    for name in features:
        if name in SEGMENT_FAMILIES or name in {"spectral", "connectivity", "variability", "all"}:
            resolved.extend(resolve_features(name, config))
            continue
        canonical = ALIASES.get(name, name)
        if canonical not in known:
            unknown.append(name)
        else:
            resolved.append(canonical)
    if unknown:
        raise ParameterError(
            f"Unknown features {unknown}. Call list_features() for the catalog."
        )
    return tuple(dict.fromkeys(resolved))


def _channel_map(data: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn(channel)`` over the last-but-one axis. ``data`` is 2D."""
    out = np.empty(data.shape[0], dtype=float)
    for i in range(data.shape[0]):
        channel = data[i]
        if channel.size == 0 or not np.isfinite(channel).all():
            out[i] = np.nan
            continue
        try:
            out[i] = float(fn(channel))
        except ParameterError:
            raise
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            out[i] = np.nan
    return out


def _time_block(data: np.ndarray, sfreq: float, names: Sequence[str]) -> dict[str, np.ndarray]:
    wanted = set(names)
    out: dict[str, np.ndarray] = {}
    if "rms" in wanted:
        out["rms"] = _channel_map(data, rms)
    if "line_length" in wanted:
        out["line_length"] = _channel_map(data, lambda x: line_length(x, sfreq))
    if "zero_crossing_rate" in wanted:
        out["zero_crossing_rate"] = _channel_map(data, lambda x: zero_crossing_rate(x, sfreq))
    if "peak_to_peak" in wanted:
        out["peak_to_peak"] = _channel_map(data, peak_to_peak)
    if "skewness" in wanted:
        out["skewness"] = _channel_map(data, skewness)
    if "kurtosis" in wanted:
        out["kurtosis"] = _channel_map(data, kurtosis)
    if {"hjorth_activity", "hjorth_mobility", "hjorth_complexity"} & wanted:
        params = [_hjorth_or_nan(data[i], sfreq) for i in range(data.shape[0])]
        stacked = np.asarray(params, dtype=float)
        out["hjorth_activity"] = stacked[:, 0]
        out["hjorth_mobility"] = stacked[:, 1]
        out["hjorth_complexity"] = stacked[:, 2]
    return {name: out[name] for name in names if name in out}


def _hjorth_or_nan(x: np.ndarray, sfreq: float) -> tuple[float, float, float]:
    if not np.isfinite(x).all():
        return (np.nan, np.nan, np.nan)
    return hjorth_parameters(x, sfreq)


def _entropy_block(data: np.ndarray, names: Sequence[str], config: FeatureConfig) -> dict[str, np.ndarray]:
    wanted = set(names)
    out: dict[str, np.ndarray] = {}

    def col(fn) -> np.ndarray:
        return _channel_map(data, fn)

    if "lzc" in wanted:
        out["lzc"] = col(lambda x: lempel_ziv_complexity((x > np.median(x)).astype(np.uint8), normalize=True))
    if "approximate_entropy" in wanted:
        out["approximate_entropy"] = col(lambda x: approximate_entropy(x, order=config.entropy_order))
    if "sample_entropy" in wanted:
        out["sample_entropy"] = col(lambda x: sample_entropy(x, order=config.entropy_order))
    if "permutation_entropy" in wanted:
        out["permutation_entropy"] = col(
            lambda x: permutation_entropy(x, order=config.pe_order, delay=config.pe_delay, normalize=True)
        )
    if "svd_entropy" in wanted:
        out["svd_entropy"] = col(
            lambda x: svd_entropy(x, order=config.pe_order, delay=config.pe_delay, normalize=True)
        )
    mse_names = [name for name in names if name == "mse_mean" or name.startswith("mse_scale")]
    if mse_names:
        scales = tuple(sorted(set(config.mse_scales) | _requested_scales(mse_names)))

        def one(x: np.ndarray) -> dict[int, float]:
            return multiscale_sample_entropy(
                x,
                scales,
                order=config.entropy_order,
                r_mode=config.mse_r_mode,
                r_fraction=config.mse_r_fraction,
            )

        curves = [one(data[i]) if np.isfinite(data[i]).all() else {} for i in range(data.shape[0])]
        if "mse_mean" in wanted:
            # A missing or infinite scale is part of the curve. Dropping it
            # would report the mean of the scales that happened to be finite.
            means = []
            for curve in curves:
                vals = [curve.get(scale, np.nan) for scale in scales]
                means.append(float(np.mean(vals)) if vals else np.nan)
            out["mse_mean"] = np.asarray(means, dtype=float)
        for scale in scales:
            column = f"mse_scale{int(scale)}"
            if column in wanted:
                out[column] = np.asarray(
                    [curve.get(scale, curve.get(int(scale), np.nan)) for curve in curves],
                    dtype=float,
                )
    return {name: out[name] for name in names if name in out}


def _requested_scales(names: Sequence[str]) -> set[int]:
    scales: set[int] = set()
    for name in names:
        if name.startswith("mse_scale"):
            scales.add(int(name.replace("mse_scale", "")))
    return scales


def _complexity_block(data: np.ndarray, names: Sequence[str], config: FeatureConfig) -> dict[str, np.ndarray]:
    wanted = set(names)
    out: dict[str, np.ndarray] = {}
    if "higuchi_fd" in wanted:
        out["higuchi_fd"] = _channel_map(data, lambda x: higuchi_fd(x, kmax=config.higuchi_kmax))
    if "petrosian_fd" in wanted:
        out["petrosian_fd"] = _channel_map(data, petrosian_fd)
    if "katz_fd" in wanted:
        out["katz_fd"] = _channel_map(data, katz_fd)
    if "dfa" in wanted:
        out["dfa"] = _channel_map(data, detrended_fluctuation)
    return {name: out[name] for name in names if name in out}


def _dynamics_block(
    data: np.ndarray,
    sfreq: float,
    names: Sequence[str],
    config: FeatureConfig,
    units: str,
    provenance: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    wanted = set(names)
    out: dict[str, np.ndarray] = {}
    threshold = config.bsr_threshold_uv * (1e-6 if units == "V" else 1.0)
    zscored = units == "z" or any(str(note).startswith("zscore") for note in provenance)
    if "burst_suppression_ratio" in wanted:
        if zscored:
            out["burst_suppression_ratio"] = np.full(data.shape[0], np.nan)
        else:
            out["burst_suppression_ratio"] = _channel_map(
                data,
                lambda x: burst_suppression_ratio(
                    x, sfreq, threshold=threshold, min_duration=config.bsr_min_duration
                ),
            )
    if "envelope_lzc" in wanted:
        flo, fhi = config.envelope_band
        out["envelope_lzc"] = _channel_map(data, lambda x: envelope_lzc(x, sfreq, flo=flo, fhi=fhi))
    return {name: out[name] for name in names if name in out}


def _spectral_block(data: np.ndarray, sfreq: float, names: Sequence[str], config: FeatureConfig) -> dict[str, np.ndarray]:
    features, psd_meta = spectral_feature_set(
        data,
        sfreq,
        bands=config.resolved_bands(),
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        n_per_seg=config.n_per_seg,
        segment_sec=config.segment_sec,
        overlap=config.overlap,
        detrend=config.detrend,
        sef_percentiles=config.sef_percentiles,
    )
    if any(name in names for name in APERIODIC_FEATURES):
        exponent, offset = aperiodic_fit(
            psd_meta["freqs"],
            psd_meta["psd"],
            fmin=max(2.0, config.freq_min),
            fmax=min(40.0, config.freq_max, sfreq / 2.0),
        )
        features["aperiodic_exponent"] = exponent
        features["aperiodic_offset"] = offset
    return {name: features[name] for name in names if name in features}


def _segment_features(
    data: np.ndarray,
    sfreq: float,
    names: Sequence[str],
    config: FeatureConfig,
    units: str,
    provenance: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """Features of one continuous segment. ``data`` is (n_channels, n_times)."""
    out: dict[str, np.ndarray] = {}
    groups = {
        "time": [n for n in names if n in TIME_FEATURES],
        "entropy": [n for n in names if n in ENTROPY_FEATURES or n.startswith("mse_scale")],
        "complexity": [n for n in names if n in COMPLEXITY_FEATURES],
        "dynamics": [n for n in names if n in DYNAMICS_FEATURES],
        "spectral": [n for n in names if n in set(spectral_names(config))],
    }
    if groups["time"]:
        out.update(_time_block(data, sfreq, groups["time"]))
    if groups["spectral"]:
        out.update(_spectral_block(data, sfreq, groups["spectral"], config))
    if groups["entropy"]:
        out.update(_entropy_block(data, groups["entropy"], config))
    if groups["complexity"]:
        out.update(_complexity_block(data, groups["complexity"], config))
    if groups["dynamics"]:
        out.update(_dynamics_block(data, sfreq, groups["dynamics"], config, units, provenance))
    return out


def _as_rows(values: Mapping[str, np.ndarray], ch_names: Sequence[str], names: Sequence[str]) -> pd.DataFrame:
    frame = {"channel": list(ch_names)}
    n = len(ch_names)
    for name in names:
        column = values.get(name)
        if column is None:
            frame[name] = np.full(n, np.nan)
        else:
            frame[name] = np.asarray(column, dtype=float).reshape(-1)
    return pd.DataFrame(frame)


def _rms_cv(eeg: EEG, config: FeatureConfig) -> np.ndarray:
    if eeg.epoched:
        if eeg.n_epochs < 2:
            return np.full(eeg.n_channels, np.nan)
        rms_values = np.stack(
            [np.sqrt(np.mean(np.square(eeg.data[i]), axis=-1)) for i in range(eeg.n_epochs)]
        )
    else:
        width = int(round(config.variability_window_sec * eeg.sfreq))
        if width < 4 or eeg.n_times < 4 * width:
            return np.full(eeg.n_channels, np.nan)
        n_win = eeg.n_times // width
        clipped = eeg.data[:, : n_win * width].reshape(eeg.n_channels, n_win, width)
        rms_values = np.sqrt(np.mean(np.square(clipped), axis=-1)).T
    mean = np.mean(rms_values, axis=0)
    std = np.std(rms_values, axis=0, ddof=1)
    return np.divide(std, mean, out=np.full(mean.shape, np.nan), where=np.abs(mean) > 0)


def _recording_features(eeg: EEG, names: Sequence[str], config: FeatureConfig) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    conn = [name for name in names if name in set(connectivity_names(config))]
    if conn:
        node, _meta = connectivity_node_features(
            eeg.data,
            eeg.sfreq,
            bands=config.resolved_bands(),
            metrics=config.connectivity_metrics,
            n_per_seg=config.n_per_seg,
            segment_sec=config.segment_sec,
            overlap=config.overlap,
            min_segments=config.connectivity_min_segments,
            freq_min=config.freq_min,
            freq_max=config.freq_max,
            detrend=config.detrend,
        )
        out.update({name: node[name] for name in conn if name in node})
    if "rms_cv" in names:
        out["rms_cv"] = _rms_cv(eeg, config)
    return out


def _average_preserving_spectral_columns(
    frame: pd.DataFrame,
    id_cols: list[str],
    value_cols: list[str],
    bands: dict[str, tuple[float, float]] | None,
) -> pd.DataFrame:
    """Average segment-level columns, then rebuild spectral transforms of power.

    Peak frequency, spectral edge, entropy, and the other segment features
    are averaged as features: the result is the mean of the epoch values,
    not the feature of the mean spectrum. Relative power, decibels, and
    band ratios are not averaged. They are restored from the averaged
    absolute powers when the configured partition is present, and left
    missing when it is not.
    """
    derived = [column for column in value_cols if is_derived_spectral_column(column)]
    plain = [column for column in value_cols if column not in derived]
    if plain:
        reduced = frame.groupby(id_cols, dropna=False, as_index=False)[plain].mean()
    else:
        reduced = frame[id_cols].drop_duplicates().reset_index(drop=True)
    for column in derived:
        reduced[column] = np.nan
    reduced = recompute_power_ratios(reduced, bands=bands)
    warn_if_ratios_left_missing(reduced, derived)
    return reduced


def _reduce_epoch_frame(frame: pd.DataFrame, bands: dict[str, tuple[float, float]]) -> pd.DataFrame:
    id_cols = [
        c
        for c in (
            "recording_id",
            "subject_id",
            "session_id",
            "condition",
            "sfreq",
            "units",
            "reference",
            "duration_sec",
            "highpass_hz",
            "lowpass_hz",
            "channel",
        )
        if c in frame.columns
    ]
    value_cols = [c for c in frame.columns if c not in id_cols + ["epoch", "scope", "aggregation"]]
    reduced = _average_preserving_spectral_columns(frame, id_cols, value_cols, bands)
    reduced["scope"] = "recording"
    reduced["aggregation"] = "mean_of_epochs"
    return reduced


def extract_features(
    eeg: EEG | EEGCollection,
    features: str | Sequence[str] = "spectral",
    *,
    config: FeatureConfig | None = None,
    reduce_epochs: bool = False,
) -> pd.DataFrame:
    """Extract named features.

    Parameters
    ----------
    features
        ``"all"``, a family name, a feature name, or a sequence of those.
    reduce_epochs
        If true and the input is epoched, average segment features across
        epochs and recompute spectral ratios from the averaged band powers.
        Recording-level features are joined onto that one-row-per-channel table.
        If false, epoch rows and recording-level rows are stacked and
        distinguished by ``scope``.

    Every row carries ``duration_sec`` and the header limits
    ``highpass_hz`` and ``lowpass_hz``. A missing header limit is missing,
    not zero. Those columns describe the recording that was extracted.
    They are not a filter applied by this function. ``units`` remains the
    recording unit on every column. The quantity of each column is in
    :func:`list_features`.
    """
    if isinstance(eeg, EEGCollection):
        frames = [
            extract_features(rec, features, config=config, reduce_epochs=reduce_epochs) for rec in eeg
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not isinstance(eeg, EEG):
        raise ParameterError("extract_features expects an EEG or EEGCollection. Use load() first.")
    config = config or FeatureConfig()
    names = resolve_features(features, config)
    compute_names = _with_spectral_dependencies(names, config)
    segment = [name for name in compute_names if name not in set(connectivity_names(config)) | set(VARIABILITY_FEATURES)]
    recording = [name for name in names if name not in segment]
    dimensionless = eeg.units == "z" or any(str(note).startswith("zscore") for note in eeg.provenance)
    if "burst_suppression_ratio" in names and dimensionless:
        warnings.warn(
            "Burst-suppression ratio needs an amplitude threshold, so it is missing after z-scoring. "
            "Compute it before z-scoring, or skip the z-score.",
            UserWarning,
            stacklevel=2,
        )
    frames: list[pd.DataFrame] = []
    if segment:
        if eeg.epoched and not reduce_epochs:
            blocks = []
            onsets = eeg.metadata.get("window_onsets")
            event_codes = eeg.metadata.get("event_codes")
            for index, epoch in eeg.iter_epochs():
                values = _segment_features(epoch.data, eeg.sfreq, segment, config, eeg.units, eeg.provenance)
                block = _as_rows(values, eeg.ch_names, segment)
                block["epoch"] = index
                block["scope"] = "epoch"
                if onsets is not None and index < len(onsets):
                    block["onset"] = float(onsets[index])
                if event_codes is not None and index < len(event_codes):
                    block["event_code"] = int(event_codes[index])
                blocks.append(block)
            frames.append(pd.concat(blocks, ignore_index=True))
        elif eeg.epoched and reduce_epochs:
            blocks = []
            for index, epoch in eeg.iter_epochs():
                values = _segment_features(epoch.data, eeg.sfreq, segment, config, eeg.units, eeg.provenance)
                block = _as_rows(values, eeg.ch_names, segment)
                block["epoch"] = index
                blocks.append(attach_identity(block, eeg))
            frames.append(
                _reduce_epoch_frame(pd.concat(blocks, ignore_index=True), config.resolved_bands())
            )
        else:
            values = _segment_features(eeg.data, eeg.sfreq, segment, config, eeg.units, eeg.provenance)
            block = _as_rows(values, eeg.ch_names, segment)
            block["epoch"] = pd.NA
            block["scope"] = "recording"
            frames.append(block)
    if recording:
        values = _recording_features(eeg, recording, config)
        block = _as_rows(values, eeg.ch_names, recording)
        block["epoch"] = pd.NA
        block["scope"] = "recording"
        frames.append(block)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 2 and reduce_epochs:
        left, right = frames
        merged = left.merge(right.drop(columns=["epoch", "scope"], errors="ignore"), on="channel", how="outer")
        merged["scope"] = "recording"
        out = attach_identity(merged, eeg)
        return _keep_requested(out, names)
    out = pd.concat(frames, ignore_index=True, sort=False)
    return _keep_requested(attach_identity(out, eeg), names)


_MISSING_KEY = "\0eegmath-missing-key"


def _merge_missing_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Left-join even when a key is missing.

    Pandas does not match missing values to each other. A recording loaded
    without an id would otherwise lose the recording-level columns.
    """
    if not keys:
        return left
    merged_left = left.copy()
    merged_right = right.copy()
    for key in keys:
        merged_left[key] = merged_left[key].astype(object).where(merged_left[key].notna(), _MISSING_KEY)
        merged_right[key] = merged_right[key].astype(object).where(merged_right[key].notna(), _MISSING_KEY)
    merged = merged_left.merge(merged_right, on=keys, how="left")
    for key in keys:
        merged.loc[merged[key] == _MISSING_KEY, key] = np.nan
    return merged


def aggregate_epochs(
    features: pd.DataFrame,
    *,
    bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Average epoch-scoped rows within channel and recording.

    Recording-scoped rows are joined onto that average. A column that is
    finite only on recording rows is taken from those rows; it is not the
    mean of the empty cells on the epoch rows. Relative power, decibels,
    and band ratios are not averaged. They are rebuilt from averaged
    absolute power when ``bands`` (defaulting to the canonical partition)
    is complete in the table, and left missing otherwise. Missing
    recording identifiers still match.
    """
    if "scope" not in features.columns or not (features["scope"] == "epoch").any():
        return features.copy()
    epoch_rows = features.loc[features["scope"] == "epoch"].copy()
    recording_rows = features.loc[features["scope"] == "recording"].copy()
    id_cols = [
        c
        for c in (
            "recording_id",
            "subject_id",
            "session_id",
            "condition",
            "sfreq",
            "units",
            "reference",
            "duration_sec",
            "highpass_hz",
            "lowpass_hz",
            "channel",
        )
        if c in epoch_rows.columns
    ]
    value_cols = [
        c
        for c in epoch_rows.columns
        if c not in id_cols + ["epoch", "scope", "aggregation", "onset", "event_code"]
    ]
    supplied = [
        c
        for c in value_cols
        if c in recording_rows.columns and recording_rows[c].notna().any()
    ]
    average_cols = [c for c in value_cols if c not in supplied]
    if average_cols:
        reduced = _average_preserving_spectral_columns(epoch_rows, id_cols, average_cols, bands)
    else:
        reduced = epoch_rows[id_cols].drop_duplicates().reset_index(drop=True)
    reduced["scope"] = "recording"
    reduced["aggregation"] = "mean_of_epochs"
    if not supplied:
        return reduced
    keys = [
        c
        for c in ("recording_id", "subject_id", "session_id", "condition", "channel")
        if c in reduced.columns and c in recording_rows.columns
    ]
    addon = recording_rows.groupby(keys, dropna=False, as_index=False)[supplied].mean()
    return _merge_missing_keys(reduced, addon, keys)
