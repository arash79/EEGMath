"""Preprocessing operations.

Nothing in feature extraction calls these functions. Each function returns
a new :class:`~eegmath.recording.EEG` and appends a provenance note.
Filtering uses MNE-Python's zero-phase FIR design. Re-applying a filter is
not harmless, and data that were already filtered by a study should be
left alone unless the caller has a reason to filter again.

Short epochs are a poor place for a long FIR. MNE warns when the
designed kernel is longer than the signal and still applies it. That
warning is raised here, because the result would not have the requested
frequency response.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from eegmath.errors import DataError, DependencyError, ParameterError
from eegmath.recording import EEG, EEGCollection

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise ImportError("eegmath preprocessing requires mne.") from exc


def _types(eeg: EEG) -> list[str]:
    if eeg.ch_types is None:
        return ["eeg"] * eeg.n_channels
    return list(eeg.ch_types)


def _finite_limit(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _merge_limits(
    eeg: EEG,
    *,
    highpass: float | None = None,
    lowpass: float | None = None,
) -> tuple[float | None, float | None]:
    """Most restrictive cutoff after one more filter.

    MNE's ``Info`` no longer accepts a direct assignment of these fields.
    The limits are therefore kept on the recording and updated here.
    A new high-pass can only raise the recorded cutoff. A new low-pass,
    including the Nyquist frequency after resampling, can only lower it.
    """
    hp = _finite_limit(eeg.metadata.get("highpass"))
    lp = _finite_limit(eeg.metadata.get("lowpass"))
    if highpass is not None:
        hp = float(highpass) if hp is None else max(hp, float(highpass))
    if lowpass is not None:
        lp = float(lowpass) if lp is None else min(lp, float(lowpass))
    return hp, lp


def _warn_if_dimensionless(eeg: EEG, operation: str) -> None:
    if eeg.units != "z":
        return
    warnings.warn(
        f"This {operation} is being applied to z-scored data. The result is not "
        "the z-score of the same operation applied to the voltages.",
        UserWarning,
        stacklevel=3,
    )


def _apply_positions(info, names: list[str], ch_pos) -> None:
    """Restore finite head coordinates. Unknown rows are left unset."""
    if ch_pos is None:
        return
    pos = {}
    for name, row in zip(names, ch_pos):
        xyz = np.asarray(row, dtype=float)
        if xyz.shape == (3,) and np.isfinite(xyz).all():
            pos[name] = xyz
    if not pos:
        return
    montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head")
    info.set_montage(montage, match_case=True, on_missing="ignore", verbose=False)


def to_mne(eeg: EEG):
    """Convert to an MNE Raw or EpochsArray.

    Voltage recordings are converted to volts. Z-scored recordings are
    copied through without rescaling: a dimensionless series is not a
    voltage, and multiplying it would invent one.     Filter limits stay on
    the recording. Finite ``ch_pos`` rows are restored as a head-coordinate
    montage. Current MNE releases reject ``info['highpass'] = ...``,
    and a loaded file almost always has a stored cutoff, so writing that
    field here would make every later filter, reference, and resample fail.
    """
    if eeg.units == "z":
        data, names, sfreq, types = eeg.data, list(eeg.ch_names), eeg.sfreq, _types(eeg)
        tmin, epoched, n_epochs = eeg.tmin, eeg.epoched, eeg.n_epochs
    else:
        volts = eeg.to_volts()
        data, names, sfreq, types = volts.data, list(volts.ch_names), volts.sfreq, _types(volts)
        tmin, epoched, n_epochs = volts.tmin, volts.epoched, volts.n_epochs
    info = mne.create_info(names, sfreq, ch_types=types, verbose=False)
    _apply_positions(info, names, eeg.ch_pos)
    if epoched:
        events = np.column_stack(
            [
                np.arange(n_epochs, dtype=int),
                np.zeros(n_epochs, dtype=int),
                np.ones(n_epochs, dtype=int),
            ]
        )
        return mne.EpochsArray(data.copy(), info, events=events, tmin=tmin, verbose=False)
    return mne.io.RawArray(data.copy(), info, verbose=False)


def _pick_indices(info, picks) -> np.ndarray:
    """Channel indices with the same rules as ``Raw.notch_filter``.

    ``mne.filter.notch_filter`` accepts indices, not channel-type strings.
    Resolving them here lets one call serve continuous and epoched data.
    ``EpochsArray`` has no ``notch_filter`` method.
    """
    from mne._fiff.pick import _picks_to_idx

    return _picks_to_idx(info, picks, exclude=(), none="data_or_ica")


def _raise_if_filter_exceeds_signal(caught, *, remedy: str) -> None:
    """Refuse an FIR MNE would apply after warning that it does not fit."""
    overlong = None
    others = []
    for item in caught:
        text = str(item.message)
        if "longer than the signal" in text and "distortion is likely" in text:
            overlong = text
        else:
            others.append(item)
    for item in others:
        warnings.warn(item.message, item.category, stacklevel=3)
    if overlong is not None:
        raise ParameterError(
            "The FIR is longer than this recording, so it would not have the "
            f"requested frequency response. {remedy}"
        )


def _from_mne_data(
    eeg: EEG,
    inst,
    note: str,
    *,
    limits: tuple[float | None, float | None],
    data: np.ndarray | None = None,
    **metadata,
) -> EEG:
    if data is None:
        data = inst.get_data()
    if eeg.units == "uV":
        data = data * 1e6
    meta = dict(eeg.metadata)
    meta.update(metadata)
    meta["highpass"], meta["lowpass"] = limits
    reference = eeg.reference
    if "reference" in metadata:
        reference = metadata["reference"]
    return eeg.spawn(note, data=data, metadata=meta, reference=reference, sfreq=float(inst.info["sfreq"]))


def bandpass(
    eeg: EEG,
    l_freq: float | None = 0.5,
    h_freq: float | None = 45.0,
    *,
    fir_window: str = "hamming",
    picks: str | list[str] = "eeg",
) -> EEG:
    """Zero-phase FIR band-pass. Either cutoff may be None for a one-sided filter.

    ``picks="eeg"`` leaves ECG, EOG, EMG, and stimulus channels unchanged
    when their types are known. Channels with no recorded type are treated
    as EEG. Pass ``picks="all"`` to filter every channel.

    The kernel length follows MNE's Hamming FIR design. At 250 Hz the
    default 0.5 Hz high-pass produces a kernel of 1651 samples, about
    6.6 s. A shorter series, including one epoch of an epoched
    recording, raises rather than returning a filter whose response is
    not the one that was requested. A 10 s series is long enough for
    that default.
    """
    if l_freq is None and h_freq is None:
        raise ParameterError("bandpass needs l_freq, h_freq, or both.")
    nyq = eeg.sfreq / 2.0
    if h_freq is not None and h_freq >= nyq:
        raise ParameterError(f"h_freq={h_freq} Hz is not below Nyquist ({nyq} Hz).")
    if l_freq is not None and l_freq <= 0:
        raise ParameterError("l_freq must be positive or None.")
    _warn_if_dimensionless(eeg, "filter")
    inst = to_mne(eeg)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inst.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks=picks,
                method="fir",
                fir_window=fir_window,
                phase="zero",
                verbose=False,
            )
    except ValueError as exc:
        raise ParameterError(
            "MNE could not design this filter for the current recording length. "
            "Short epochs often cannot support a low high-pass cutoff. "
            f"Original error: {exc}"
        ) from exc
    _raise_if_filter_exceeds_signal(
        caught,
        remedy="Use a longer segment, or a higher high-pass cutoff, which shortens the kernel.",
    )
    return _from_mne_data(
        eeg,
        inst,
        f"bandpass:{l_freq}-{h_freq}Hz",
        limits=_merge_limits(eeg, highpass=l_freq, lowpass=h_freq),
    )


def notch(
    eeg: EEG,
    freqs: float | Sequence[float] = 50.0,
    *,
    picks: str | list[str] = "eeg",
    trans_bandwidth: float = 1.0,
) -> EEG:
    """Zero-phase FIR notch at one or more line frequencies.

    The default ``picks="eeg"`` matches :func:`bandpass`. Pass
    ``picks="all"`` to notch every channel. Epoched recordings are
    filtered one epoch at a time. The kernel does not cross epoch
    boundaries.

    ``trans_bandwidth`` is the full width, in hertz, passed to
    MNE-Python. MNE places half of it on each side of the stop band and
    sets the Hamming ``firwin`` length to 3.3 divided by that half-width.
    The default of 1 Hz is therefore a kernel of about 6.6 s. A shorter
    recording, or a shorter epoch, raises rather than returning a
    distorted notch. Widen ``trans_bandwidth`` only when that wider
    transition is acceptable.
    """
    _warn_if_dimensionless(eeg, "notch")
    if not np.isfinite(trans_bandwidth) or trans_bandwidth <= 0:
        raise ParameterError(f"trans_bandwidth must be a positive finite number, got {trans_bandwidth!r}.")
    requested = (float(freqs),) if isinstance(freqs, (int, float)) else tuple(float(f) for f in freqs)
    nyq = eeg.sfreq / 2.0
    if any(line <= 0 or line >= nyq for line in requested):
        raise ParameterError(f"Notch frequencies {requested} must lie inside (0, {nyq}) Hz.")
    lowpass = _finite_limit(eeg.metadata.get("lowpass"))
    skipped = [line for line in requested if lowpass is not None and line > lowpass]
    lines = tuple(line for line in requested if line not in skipped)
    if not lines:
        return eeg.spawn(
            f"notch_skipped:{tuple(skipped)}>lowpass {lowpass}",
            metadata={**eeg.metadata, "notch_skipped": skipped},
        )
    inst = to_mne(eeg)
    data = inst.get_data(copy=True) if eeg.epoched else np.array(inst.get_data(), copy=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filtered = mne.filter.notch_filter(
                data,
                float(eeg.sfreq),
                list(lines),
                picks=_pick_indices(inst.info, picks),
                method="fir",
                trans_bandwidth=float(trans_bandwidth),
                verbose=False,
            )
    except ValueError as exc:
        raise ParameterError(
            "MNE could not design this notch for the current recording length. "
            f"Original error: {exc}"
        ) from exc
    _raise_if_filter_exceeds_signal(
        caught,
        remedy="Use a longer segment, or a wider trans_bandwidth, which shortens the kernel.",
    )
    note = f"notch:{list(lines)}"
    extra: dict = {}
    if skipped:
        note += f";skipped:{skipped}>lowpass {lowpass}"
        extra["notch_skipped"] = skipped
    return _from_mne_data(eeg, inst, note, limits=_merge_limits(eeg), data=filtered, **extra)


def rereference(eeg: EEG, ref: str = "average") -> EEG:
    """Re-reference EEG channels.

    ``ref="average"`` uses the average of channels whose type is ``eeg``
    and needs at least two of them. The average of one channel is zero,
    so that case is refused rather than written back as a flat trace.
    A channel name uses that electrode. Non-EEG channels are left out of
    the reference projection when types are known. Z-scored data are
    refused: each channel has already been divided by its own standard
    deviation, so the average of those channels is not an average reference.
    Average reference on exactly two EEG channels makes one the negative
    of the other. The call warns. Coherence, phase-locking value, and
    pairwise phase consistency on that pair are then 1, and the lag
    measures are 0. That is the reference, not coupling between the sites.
    """
    if eeg.units == "z":
        raise ParameterError(
            "Re-referencing needs voltages. Z-scored channels were divided by "
            "their own standard deviations, so their average is not an average reference."
        )
    if ref == "average":
        n_eeg = sum(kind == "eeg" for kind in _types(eeg))
        if n_eeg < 2:
            raise ParameterError(
                "Average reference needs at least two EEG channels. "
                f"This recording has {n_eeg}. The average of one channel is "
                "that channel minus itself, which is a zero signal, so the "
                "reference was not applied."
            )
        if n_eeg == 2:
            warnings.warn(
                "Average reference on exactly two EEG channels makes one "
                "channel the negative of the other. Coherence, phase-locking "
                "value, and pairwise phase consistency on that pair are then "
                "1, and the lag measures are 0. That is the reference, not "
                "coupling between the two sites.",
                UserWarning,
                stacklevel=2,
            )
    inst = to_mne(eeg)
    if ref == "average":
        inst.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
        label = "average"
    else:
        match = [name for name in eeg.ch_names if name.lower() == ref.lower()]
        if len(match) != 1:
            raise ParameterError(
                f"Reference {ref!r} matched {match or 'no channels'}. "
                "Channel names are case-sensitive except for this one lookup."
            )
        inst.set_eeg_reference(ref_channels=[match[0]], projection=False, verbose=False)
        label = match[0]
    return _from_mne_data(
        eeg, inst, f"rereference:{label}", limits=_merge_limits(eeg), reference=label
    )


def resample(eeg: EEG, sfreq: float) -> EEG:
    """Resample with MNE's Fourier method. Epochs are resampled separately."""
    _warn_if_dimensionless(eeg, "resampling")
    if sfreq <= 0:
        raise ParameterError("sfreq must be positive.")
    if np.isclose(sfreq, eeg.sfreq):
        return eeg.spawn("resample:unchanged")
    inst = to_mne(eeg)
    inst.resample(sfreq, verbose=False)
    # Upsampling does not create energy above the original Nyquist rate.
    # Downsampling removes energy above the new one. The recorded low-pass
    # is the tighter of those two rates and any cutoff already stored.
    representable = min(float(eeg.sfreq), float(sfreq)) / 2.0
    return _from_mne_data(
        eeg,
        inst,
        f"resample:{eeg.sfreq}->{sfreq}",
        limits=_merge_limits(eeg, lowpass=representable),
    )


def detrend(eeg: EEG, type: str = "linear") -> EEG:
    """Detrend each channel, and each epoch, along time.

    ``constant`` subtracts the mean. ``linear`` subtracts the least-squares
    line along a centered time index, so the slope is independent of the
    mean. A series shorter than two samples has no slope.
    """
    _warn_if_dimensionless(eeg, "detrend")
    if type not in {"linear", "constant"}:
        raise ParameterError("detrend type must be 'linear' or 'constant'.")
    data = np.asarray(eeg.data, dtype=float)
    centered = data - np.mean(data, axis=-1, keepdims=True)
    if type == "constant" or data.shape[-1] < 2:
        return eeg.spawn(f"detrend:{type}", data=centered)
    t = np.arange(data.shape[-1], dtype=float)
    t -= t.mean()
    slope = np.tensordot(centered, t, axes=([-1], [0])) / float(np.dot(t, t))
    return eeg.spawn(f"detrend:{type}", data=centered - slope[..., None] * t)


def baseline_correct(eeg: EEG, tmin: float, tmax: float) -> EEG:
    """Subtract the mean amplitude inside ``[tmin, tmax]`` seconds.

    Times are relative to ``eeg.tmin`` (zero for a continuous recording
    that starts at the beginning of the array).
    """
    _warn_if_dimensionless(eeg, "baseline correction")
    if tmax < tmin:
        raise ParameterError("baseline tmax is before tmin.")
    times = eeg.times
    mask = (times >= tmin) & (times <= tmax)
    if int(mask.sum()) < 1:
        raise ParameterError(
            f"No samples fall inside baseline [{tmin}, {tmax}] s. "
            f"This recording spans {times[0]:.4g} to {times[-1]:.4g} s."
        )
    baseline = np.mean(eeg.data[..., mask], axis=-1, keepdims=True)
    return eeg.spawn(f"baseline:{tmin}:{tmax}", data=eeg.data - baseline)


def zscore(eeg: EEG) -> EEG:
    """Z-score each channel along time. A constant channel becomes NaN.

    Epoched recordings are scaled inside each epoch. Epochs are not
    concatenated to form one mean and standard deviation. The result
    has ``units="z"``. It is dimensionless, so voltage conversion,
    re-referencing, and a microvolt threshold do not apply to it.
    A later filter, notch, resample, detrend, or baseline correction does
    not undo the z-score, and it is not the same recording as applying
    that operation before the z-score.
    """
    mean = np.mean(eeg.data, axis=-1, keepdims=True)
    std = np.std(eeg.data, axis=-1, keepdims=True, ddof=0)
    scaled = np.divide(eeg.data - mean, std, out=np.full_like(eeg.data, np.nan), where=std > 0)
    return eeg.spawn("zscore:time", data=scaled, units="z")


def window(eeg: EEG, window_sec: float, overlap: float = 0.0) -> EEG:
    """Cut a recording into fixed windows. The result is epoched.

    Windows are taken inside each existing epoch. Epochs are not
    concatenated, because a boundary is not a continuous sample.
    ``metadata["window_onsets"]`` gives each window's start in seconds
    on the pre-window time axis. Annotations are left on that same axis;
    they are not retimed onto the individual windows. When the recording
    is already epoched and ``event_codes`` or ``event_times`` has one
    entry per epoch, each window inherits the code and time of its parent
    epoch. The window onset is not substituted for that event time.
    """
    if window_sec <= 0:
        raise ParameterError("window_sec must be positive.")
    if not 0 <= overlap < 1:
        raise ParameterError("overlap must be in [0, 1).")
    width = int(round(window_sec * eeg.sfreq))
    step = max(1, int(round(width * (1.0 - overlap))))
    if width < 4:
        raise ParameterError("window_sec is shorter than 4 samples.")
    if width > eeg.n_times:
        raise DataError(
            f"window_sec={window_sec} s ({width} samples) is longer than the "
            f"recording ({eeg.duration:.4g} s)."
        )
    starts = list(range(0, eeg.n_times - width + 1, step))
    onsets = [float(eeg.tmin + start / eeg.sfreq) for start in starts]
    if eeg.epoched:
        pieces = []
        parents = []
        window_onsets = []
        for epoch in range(eeg.n_epochs):
            for start, onset in zip(starts, onsets):
                pieces.append(eeg.data[epoch, :, start : start + width])
                parents.append(epoch)
                window_onsets.append(onset)
        data = np.stack(pieces, axis=0)
        meta = {**eeg.metadata, "parent_epoch": parents, "window_onsets": window_onsets}
        # Indexing the original event list by the new window number assigns
        # the second epoch's code to the second window of the first epoch.
        codes = eeg.metadata.get("event_codes")
        if codes is not None and len(tuple(codes)) == eeg.n_epochs:
            meta["event_codes"] = tuple(codes[epoch] for epoch in parents)
        event_times = eeg.metadata.get("event_times")
        if event_times is not None and len(tuple(event_times)) == eeg.n_epochs:
            meta["event_times"] = tuple(event_times[epoch] for epoch in parents)
    else:
        data = np.stack([eeg.data[:, start : start + width] for start in starts], axis=0)
        meta = {**eeg.metadata, "window_onsets": onsets}
    if eeg.annotations is not None:
        meta["annotations_timebase"] = "before windowing; annotations are not split across windows"
    return eeg.spawn(
        f"window:{window_sec}s overlap={overlap}",
        data=data,
        metadata=meta,
        tmin=0.0,
    )


def drop_channels(eeg: EEG, names: Sequence[str], *, missing: str = "ignore") -> EEG:
    """Drop channels by name.

    Unknown names are ignored. :meth:`eegmath.recording.EEG.drop_channels`
    raises on an unknown name unless ``missing="ignore"`` is passed.
    The two defaults are different on purpose: a named electrode that is
    absent from one file should not stop a directory, and a direct call
    on one recording should not drop a typo quietly.
    """
    return eeg.drop_channels(names, missing=missing)


def pick_channels(eeg: EEG, names: Sequence[str]) -> EEG:
    return eeg.pick_channels(names)


def detect_bad_channels(
    eeg: EEG,
    z_thresh: float = 5.0,
    variance_ratio: float = 10.0,
) -> list[str]:
    """Channels whose log-variance is a robust z-score outlier.

    This is a screening rule, not a decision to interpolate. Interpolation
    needs electrode positions and is intentionally not implied by detection.
    A channel with any non-finite sample is flagged. Leaving it in the
    median would make ``numpy.median`` undefined for every channel, and
    the variance of the samples that remain is not the variance of the
    whole channel.

    The scale is 1.4826 times the median absolute deviation of the
    log-variance. A standard normal variable has median absolute
    deviation ``Φ^{-1}(0.75) ≈ 0.67449``, and the reciprocal is
    approximately 1.4826, so the scale estimates a normal standard
    deviation. Channels with non-finite variance are left out of the
    median. When more than half the remaining channels share one
    variance, the deviation is zero and a robust z-score cannot move.
    In that case a channel is flagged when its variance is at least
    ``variance_ratio`` times the median variance of the finite channels,
    or at most the reciprocal, or is not finite.
    """
    data = eeg.data.reshape(-1, eeg.n_channels, eeg.n_times) if eeg.epoched else eeg.data[None, ...]
    incomplete = ~np.isfinite(data).all(axis=(0, -1))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        variance = np.nanvar(data, axis=-1).mean(axis=0)
    logged = np.where(np.isfinite(variance) & ~incomplete, np.log(variance + 1e-30), np.nan)
    center = float(np.nanmedian(logged))
    if np.isfinite(center):
        scale = 1.4826 * float(np.nanmedian(np.abs(logged - center)))
    else:
        scale = np.nan
    flagged = incomplete | ~np.isfinite(variance)
    if not np.isfinite(scale) or scale == 0:
        reference = variance[~incomplete & np.isfinite(variance)]
        median_var = float(np.median(reference)) if reference.size else np.nan
        for index, var in enumerate(variance):
            if flagged[index]:
                continue
            if not np.isfinite(var) or var <= 0 or not np.isfinite(median_var) or median_var <= 0:
                flagged[index] = True
                continue
            ratio = float(var / median_var)
            if ratio >= variance_ratio or ratio <= 1.0 / variance_ratio:
                flagged[index] = True
        return [eeg.ch_names[i] for i in np.flatnonzero(flagged)]
    z = (logged - center) / scale
    flagged |= np.isfinite(z) & (np.abs(z) > z_thresh)
    return [eeg.ch_names[i] for i in np.flatnonzero(flagged)]


def qc_summary(eeg: EEG) -> dict:
    """Scalar quality summary. It does not modify the recording."""
    data = eeg.data
    finite = np.isfinite(data)
    channel_std = np.nanstd(np.where(finite, data, np.nan), axis=-1)
    if eeg.epoched:
        channel_std = np.nanmean(channel_std, axis=0)
    flat = [eeg.ch_names[i] for i, value in enumerate(channel_std) if not np.isfinite(value) or value == 0]
    median_std = float(np.nanmedian(channel_std)) if channel_std.size else np.nan
    if eeg.units == "z":
        hint = "dimensionless_zscore"
    elif eeg.units == "V" and np.isfinite(median_std):
        hint = "consistent_with_volts" if median_std < 1e-2 else "larger_than_typical_scalp_volts"
    elif eeg.units == "uV" and np.isfinite(median_std):
        hint = "consistent_with_microvolts" if 0.1 <= median_std <= 500 else "check_amplitude_units"
    else:
        hint = "unknown"
    summary = eeg.describe()
    summary.update(
        {
            "median_channel_std": median_std,
            "amplitude_scale_hint": hint,
            "flat_channels": flat,
            "highpass_hz": eeg.metadata.get("highpass"),
            "lowpass_hz": eeg.metadata.get("lowpass"),
            "bad_channel_screen": detect_bad_channels(eeg),
            "marked_bad_channels": list(eeg.metadata.get("bad_channels", ())),
        }
    )
    return summary


def qc_table(eeg: EEG | EEGCollection) -> "pd.DataFrame":
    import pandas as pd

    recordings = eeg.recordings if isinstance(eeg, EEGCollection) else (eeg,)
    return pd.DataFrame([qc_summary(rec) for rec in recordings])


STEPS = {
    "bandpass": bandpass,
    "notch": notch,
    "rereference": rereference,
    "resample": resample,
    "detrend": detrend,
    "baseline": baseline_correct,
    "zscore": zscore,
    "window": window,
    "drop_channels": drop_channels,
    "pick_channels": pick_channels,
}


def apply_steps(eeg: EEG, steps: Sequence[tuple[str, dict]]) -> EEG:
    """Apply named preprocessing steps. ``steps`` is empty by default at every caller."""
    out = eeg
    for name, kwargs in steps:
        if name not in STEPS:
            raise ParameterError(f"Unknown preprocessing step '{name}'. Known steps: {sorted(STEPS)}.")
        out = STEPS[name](out, **kwargs)
    return out


def map_collection(collection: EEGCollection, fn, *, n_jobs: int = 1) -> list:
    """Apply ``fn`` to each recording.

    ``n_jobs=1`` is sequential. Other values require the optional ``joblib``
    package and do not share mutable state between recordings.
    """
    if n_jobs == 1:
        return [fn(rec) for rec in collection]
    try:
        from joblib import Parallel, delayed
    except ImportError as exc:
        raise DependencyError(
            "n_jobs other than 1 requires joblib. Install the parallel extra "
            "with pip install 'eegmath[parallel]', or keep n_jobs=1."
        ) from exc
    return Parallel(n_jobs=n_jobs)(delayed(fn)(rec) for rec in collection)
