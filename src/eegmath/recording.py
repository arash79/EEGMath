"""Recording objects and collections.

A recording is the contract between loading, preprocessing, feature
extraction, and signal analysis. Functions do not assume a file format,
a sampling frequency, a reference, or an experimental condition. Epoched
data stay epoched: epochs are repeated observations of one recording, not
independent subjects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from eegmath.errors import DataError


def _as_float_array(data: np.ndarray) -> np.ndarray:
    # Copy so later in-place work, including MNE filters, cannot alter the
    # array the caller still holds.
    arr = np.array(data, dtype=np.float64, copy=True)
    if arr.ndim not in (2, 3):
        raise DataError(
            "EEG data must have shape (n_channels, n_times) or "
            f"(n_epochs, n_channels, n_times); received shape {arr.shape}."
        )
    return arr


@dataclass(frozen=True)
class EEG:
    """Continuous or epoched multichannel EEG.

    Parameters
    ----------
    data
        Continuous shape ``(n_channels, n_times)`` or epoched shape
        ``(n_epochs, n_channels, n_times)``.
    sfreq
        Sampling frequency in hertz.
    ch_names
        Channel names. Length must equal ``n_channels``.
    units
        Amplitude unit of ``data``. ``"V"`` matches MNE-Python. ``"uV"``
        is microvolts. ``"z"`` is a dimensionless z-score and has no
        voltage scale.
    ch_pos
        Head coordinates in metres, shape ``(n_channels, 3)``. A row of
        non-finite values is an unknown location. Positions are not used
        by feature or connectivity calculations.
    """

    data: np.ndarray
    sfreq: float
    ch_names: tuple[str, ...]
    units: str = "V"
    ch_types: tuple[str, ...] | None = None
    ch_pos: tuple[tuple[float, float, float], ...] | None = None
    reference: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    condition: str | None = None
    recording_id: str | None = None
    tmin: float = 0.0
    annotations: pd.DataFrame | None = None
    provenance: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = _as_float_array(self.data)
        if not np.isfinite(self.sfreq) or float(self.sfreq) <= 0:
            raise DataError(f"sfreq must be a positive finite number, got {self.sfreq!r}.")
        names = tuple(str(n) for n in self.ch_names)
        n_channels = data.shape[-2]
        if len(names) != n_channels:
            raise DataError(f"{len(names)} channel names were given for {n_channels} channels.")
        if len(set(names)) != len(names):
            raise DataError(f"Channel names must be unique; duplicates: {_duplicates(names)}.")
        if self.units not in {"V", "uV", "z"}:
            raise DataError("units must be 'V', 'uV', or 'z'.")
        types = None if self.ch_types is None else tuple(str(t) for t in self.ch_types)
        if types is not None and len(types) != n_channels:
            raise DataError("ch_types must have one entry per channel.")
        positions = _channel_positions(self.ch_pos, n_channels)
        annotations = self.annotations
        if annotations is not None:
            if not isinstance(annotations, pd.DataFrame):
                raise DataError("annotations must be a pandas DataFrame or None.")
            annotations = annotations.copy()
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sfreq", float(self.sfreq))
        object.__setattr__(self, "ch_names", names)
        object.__setattr__(self, "ch_types", types)
        object.__setattr__(self, "ch_pos", positions)
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "tmin", float(self.tmin))

    @property
    def epoched(self) -> bool:
        return self.data.ndim == 3

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[-2])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[-1])

    @property
    def n_epochs(self) -> int:
        """Epoch count. A continuous recording returns 1 so a single pass can use the same loop.

        :meth:`describe` reports 0 for that case. The zero means "not epoched",
        which is a different question from how many times the loop runs.
        """
        return int(self.data.shape[0]) if self.epoched else 1

    @property
    def duration(self) -> float:
        """Duration of one continuous recording or one epoch, in seconds."""
        return self.n_times / self.sfreq

    @property
    def times(self) -> np.ndarray:
        return self.tmin + np.arange(self.n_times, dtype=float) / self.sfreq

    def copy(self) -> EEG:
        return self.spawn("copy", data=self.data.copy(), metadata=dict(self.metadata))

    def spawn(self, note: str, **changes: Any) -> EEG:
        """Return a new recording with ``note`` appended to the provenance."""
        metadata = dict(self.metadata)
        if "metadata" in changes:
            metadata = dict(changes.pop("metadata"))
        annotations = self.annotations
        if "annotations" not in changes and annotations is not None:
            annotations = annotations.copy()
        changes.setdefault("annotations", annotations)
        return replace(
            self,
            provenance=self.provenance + (note,),
            metadata=metadata,
            **changes,
        )

    def channel_index(self, names: Sequence[str]) -> np.ndarray:
        missing = [name for name in names if name not in self.ch_names]
        if missing:
            raise KeyError(
                f"Channels not in this recording: {missing}. "
                f"Available examples: {list(self.ch_names[:8])}."
            )
        lookup = {name: i for i, name in enumerate(self.ch_names)}
        return np.array([lookup[name] for name in names], dtype=int)

    def pick_channels(self, names: Sequence[str]) -> EEG:
        idx = self.channel_index(names)
        types = None if self.ch_types is None else tuple(self.ch_types[i] for i in idx)
        positions = None if self.ch_pos is None else tuple(self.ch_pos[int(i)] for i in idx)
        data = self.data[..., idx, :]
        return self.spawn(
            f"pick_channels:{list(names)}",
            data=data,
            ch_names=tuple(names),
            ch_types=types,
            ch_pos=positions,
        )

    def drop_channels(self, names: Sequence[str], *, missing: str = "raise") -> EEG:
        """Drop channels by name.

        An unknown name raises. :func:`eegmath.preprocess.drop_channels`
        defaults to ``missing="ignore"`` and otherwise calls this method.
        """
        unknown = [name for name in names if name not in self.ch_names]
        if unknown and missing == "raise":
            raise KeyError(f"Cannot drop unknown channels: {unknown}.")
        if unknown and missing != "ignore":
            raise ValueError("missing must be 'raise' or 'ignore'.")
        keep = [name for name in self.ch_names if name not in set(names)]
        if not keep:
            raise DataError("drop_channels would remove every channel.")
        idx = self.channel_index(keep)
        types = None if self.ch_types is None else tuple(self.ch_types[i] for i in idx)
        positions = None if self.ch_pos is None else tuple(self.ch_pos[int(i)] for i in idx)
        return self.spawn(
            f"drop_channels:{list(names)}",
            data=self.data[..., idx, :],
            ch_names=tuple(keep),
            ch_types=types,
            ch_pos=positions,
        )

    def to_volts(self) -> EEG:
        if self.units == "z":
            raise DataError(
                "Z-scored data are dimensionless. Multiplying them into volts "
                "would invent a voltage scale, so the conversion was not applied."
            )
        if self.units == "V":
            return self
        return self.spawn("units:uV_to_V", data=self.data * 1e-6, units="V")

    def to_microvolts(self) -> EEG:
        if self.units == "z":
            raise DataError(
                "Z-scored data are dimensionless. Multiplying them into microvolts "
                "would invent a voltage scale, so the conversion was not applied."
            )
        if self.units == "uV":
            return self
        return self.spawn("units:V_to_uV", data=self.data * 1e6, units="uV")

    def scale_to(self, units: str) -> EEG:
        if units == "V":
            return self.to_volts()
        if units == "uV":
            return self.to_microvolts()
        raise DataError("units must be 'V' or 'uV'.")

    def iter_epochs(self) -> Iterator[tuple[int, EEG]]:
        """Yield ``(epoch_index, continuous EEG)`` copies of each epoch."""
        if not self.epoched:
            yield 0, self
            return
        for i in range(self.n_epochs):
            meta = dict(self.metadata)
            meta["epoch"] = i
            yield i, self.spawn(f"epoch:{i}", data=self.data[i], metadata=meta)

    def describe(self) -> dict[str, Any]:
        finite = np.isfinite(self.data)
        return {
            "recording_id": self.recording_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "condition": self.condition,
            "epoched": self.epoched,
            "n_epochs": self.n_epochs if self.epoched else 0,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "sfreq": self.sfreq,
            "duration_sec": self.duration,
            "units": self.units,
            "reference": self.reference,
            "n_positioned_channels": 0
            if self.ch_pos is None
            else int(sum(np.isfinite(row).all() for row in self.ch_pos)),
            "tmin": self.tmin,
            "fraction_nonfinite": float(1.0 - finite.mean()) if self.data.size else 1.0,
            "provenance": list(self.provenance),
        }


def _channel_positions(value: Any, n_channels: int) -> tuple[tuple[float, float, float], ...] | None:
    """Store one head-coordinate triple per channel, in metres.

    An all-missing array means the recording has no positions. A single
    missing row stays missing so that a channel without a location does
    not inherit a neighbour's.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n_channels, 3):
        raise DataError(f"ch_pos must have shape {(n_channels, 3)}; received {arr.shape}.")
    if not np.isfinite(arr).any():
        return None
    return tuple(tuple(float(coord) for coord in row) for row in arr)


def _duplicates(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for name in names:
        if name in seen and name not in dup:
            dup.append(name)
        seen.add(name)
    return dup


@dataclass(frozen=True)
class EEGCollection:
    """An ordered collection of recordings.

    Iteration order is the order supplied by the caller. The collection
    does not require shared sampling rates, channel sets, or conditions.
    """

    recordings: tuple[EEG, ...]

    def __post_init__(self) -> None:
        recs = tuple(self.recordings)
        if not recs:
            raise DataError("EEGCollection requires at least one recording.")
        if not all(isinstance(r, EEG) for r in recs):
            raise DataError("EEGCollection recordings must be EEG objects.")
        object.__setattr__(self, "recordings", recs)

    def __iter__(self) -> Iterator[EEG]:
        return iter(self.recordings)

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int) -> EEG:
        return self.recordings[index]

    @classmethod
    def from_sequence(cls, recordings: Sequence[EEG]) -> EEGCollection:
        return cls(tuple(recordings))

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame([r.describe() for r in self.recordings])

    def compatibility(self) -> dict[str, Any]:
        """Report mismatches without changing the recordings."""
        sfreqs = sorted({round(r.sfreq, 6) for r in self.recordings})
        units = sorted({r.units for r in self.recordings})
        refs = sorted({r.reference for r in self.recordings}, key=lambda v: (v is None, str(v)))
        channel_sets = [tuple(r.ch_names) for r in self.recordings]
        intersection: set[str] = set(channel_sets[0])
        for names in channel_sets[1:]:
            intersection &= set(names)
        return {
            "n_recordings": len(self.recordings),
            "sfrequencies_hz": sfreqs,
            "same_sfreq": len(sfreqs) == 1,
            "units": units,
            "same_units": len(units) == 1,
            "references": [None if r is None else str(r) for r in refs],
            "identical_channel_sets": len(set(channel_sets)) == 1,
            "n_shared_channels": len(intersection),
            "shared_channels": sorted(intersection),
            "subjects": sorted({r.subject_id for r in self.recordings}),
            "conditions": sorted({r.condition for r in self.recordings}, key=lambda v: (v is None, str(v))),
            "n_epoched": sum(r.epoched for r in self.recordings),
        }

    def require(self, *, same_sfreq: bool = False, same_channels: bool = False, same_units: bool = False) -> dict[str, Any]:
        """Raise :class:`DataError` when a requested compatibility check fails."""
        report = self.compatibility()
        problems: list[str] = []
        if same_sfreq and not report["same_sfreq"]:
            problems.append(f"sampling frequencies differ: {report['sfrequencies_hz']}")
        if same_units and not report["same_units"]:
            problems.append(f"amplitude units differ: {report['units']}")
        if same_channels and not report["identical_channel_sets"]:
            problems.append("channel names or channel order are not identical across recordings")
        if problems:
            raise DataError(
                "This collection cannot be treated as homogeneous. "
                + "; ".join(problems)
                + ". Resample, convert units, or pick a shared channel set explicitly."
            )
        return report


def align_channels(collection: EEGCollection, *, how: str = "intersection") -> EEGCollection:
    """Pick one channel set for every recording.

    ``how="intersection"`` keeps names present in every recording, in the
    order of the first recording. Recordings are not resampled and are not
    re-referenced.
    """
    if how != "intersection":
        raise ValueError("Only how='intersection' is implemented. Pass an explicit channel list to pick_channels.")
    report = collection.compatibility()
    shared = report["shared_channels"]
    if not shared:
        raise DataError("These recordings have no channel names in common.")
    order = [name for name in collection.recordings[0].ch_names if name in set(shared)]
    aligned = tuple(rec.pick_channels(order) for rec in collection.recordings)
    return EEGCollection(aligned)


def _optional_hertz(value: Any) -> float:
    """A filter limit from a file header, or NaN when the header has none.

    Zero is kept. A header that says 0 Hz is not the same fact as a
    missing header, and a later test can group on that difference.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    return number


def recording_table_keys() -> tuple[str, ...]:
    return (
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
    )


def attach_identity(df: pd.DataFrame, eeg: EEG) -> pd.DataFrame:
    """Insert recording identity columns. Existing columns are left unchanged.

    ``units`` is the recording unit, copied onto every feature on the row.
    It is not the unit of each column: a relative power is a proportion,
    and an absolute power is the square of this unit. ``duration_sec`` is
    the length of the continuous recording or of one epoch.
    ``highpass_hz`` and ``lowpass_hz`` are the limits stored on the
    recording. They are missing when the header did not carry a limit.
    Extraction does not apply those filters.
    """
    out = df.copy()
    identity: Mapping[str, Any] = {
        "recording_id": eeg.recording_id,
        "subject_id": eeg.subject_id,
        "session_id": eeg.session_id,
        "condition": eeg.condition,
        "sfreq": eeg.sfreq,
        "units": eeg.units,
        "reference": eeg.reference,
        "duration_sec": eeg.duration,
        "highpass_hz": _optional_hertz(eeg.metadata.get("highpass")),
        "lowpass_hz": _optional_hertz(eeg.metadata.get("lowpass")),
    }
    for key, value in identity.items():
        if key not in out.columns:
            out[key] = value
    return out
