"""Readers for arrays, MNE objects, EEGLAB, EDF/BDF/BrainVision, and BIDS.

Loading does not filter, re-reference, or epoch beyond what the file
already contains. EEGLAB epoched sets are read as epochs. A directory
that contains ``dataset_description.json`` is read as BIDS; any other
directory is a list of EEG files. Filenames contribute subject, session,
and task labels only when they use BIDS entities (``sub-``, ``ses-``,
``task-``, ``run-``). Other names stay in ``recording_id``.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from eegmath.errors import DataError, ParameterError
from eegmath.recording import EEG, EEGCollection

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise ImportError("eegmath loading requires mne.") from exc

_EEG_SUFFIXES = {".set", ".edf", ".bdf", ".fif", ".vhdr"}
_ENTITY = re.compile(r"(?:^|_)(sub|ses|task|run|acq)-([^_]+)")
_FILE_IDENTITY = {"subject_id", "session_id", "condition", "recording_id", "reference"}
_BIDS_SELECTION = {"subjects", "tasks", "sessions"}
_BIDS_MISSING = {"", "n/a", "na", "none"}


def _positions_from_mne(inst) -> np.ndarray | None:
    """Head coordinates from MNE, in metres.

    MNE stores an unknown location as zeros or as non-finite values.
    Those rows stay missing. A file in which every location is unknown
    returns ``None`` rather than a block of zeros at the head origin.
    """
    locs = np.array([ch["loc"][:3] for ch in inst.info["chs"]], dtype=float)
    if locs.size == 0:
        return None
    missing = ~np.isfinite(locs).all(axis=1) | (np.linalg.norm(np.nan_to_num(locs, nan=0.0), axis=1) == 0.0)
    if bool(missing.all()):
        return None
    locs = locs.copy()
    locs[missing] = np.nan
    return locs


def _reject_unknown(kwargs: dict[str, Any], allowed: set[str], message: str) -> None:
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ParameterError(message.format(unknown=unknown))


def _optional_bids_text(value: Any) -> str | None:
    """Return a BIDS free-text field, or None when it is missing or ``n/a``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    if text.lower() in _BIDS_MISSING:
        return None
    return text


def from_mne(inst, *, source: str | None = None, **identity: Any) -> EEG:
    """Build an :class:`EEG` from an MNE Raw or Epochs instance.

    MNE data are in volts. ``units`` is not accepted here: relabelling a
    volt array as microvolts would rescale every later threshold by
    ``1e6``. Convert with :meth:`~eegmath.recording.EEG.to_microvolts`
    after loading, or pass ``units`` only to :func:`load_array`.
    ``reference`` records a reference already present in the file. It
    does not re-reference. Channel types, head coordinates, filter limits
    stored on ``info``, and annotations are copied. The MNE object is not
    modified. Coordinates are the first three entries of each channel
    ``loc`` vector, which MNE keeps in head coordinates and metres.
    """
    data = inst.get_data()
    annotations = None
    if getattr(inst, "annotations", None) is not None and len(inst.annotations):
        annotations = pd.DataFrame(
            {
                "onset": np.asarray(inst.annotations.onset, dtype=float),
                "duration": np.asarray(inst.annotations.duration, dtype=float),
                "description": [str(item) for item in inst.annotations.description],
            }
        )
    metadata: dict[str, Any] = {
        "highpass": inst.info.get("highpass"),
        "lowpass": inst.info.get("lowpass"),
        "source": source,
        "line_freq": inst.info.get("line_freq"),
    }
    if inst.info.get("description"):
        metadata["reader_note"] = inst.info.get("description")
    if isinstance(inst, mne.BaseEpochs):
        metadata["n_epochs_file"] = len(inst)
        events = np.asarray(inst.events)
        if events.size:
            # Sample indices belong to the sampling rate at load time.
            # Store seconds so a later resample does not reinterpret them.
            metadata["event_times"] = tuple(float(sample) / float(inst.info["sfreq"]) for sample in events[:, 0])
            metadata["event_codes"] = tuple(int(code) for code in events[:, 2])
        if getattr(inst, "event_id", None):
            metadata["event_id"] = {str(key): int(value) for key, value in inst.event_id.items()}
        tmin = float(inst.tmin)
        epoched = True
    else:
        tmin = 0.0
        epoched = False
    _reject_unknown(
        identity,
        _FILE_IDENTITY,
        "Unknown arguments {unknown} for an MNE or file recording. "
        "Accepted identity fields are subject_id, session_id, condition, "
        "recording_id, and reference. units is not one of them: these "
        "readers return volts.",
    )
    recording_id = identity.get("recording_id")
    if recording_id is None and source:
        recording_id = Path(source).stem
    return EEG(
        data=data,
        sfreq=float(inst.info["sfreq"]),
        ch_names=tuple(inst.ch_names),
        units="V",
        ch_types=tuple(inst.get_channel_types()),
        ch_pos=_positions_from_mne(inst),
        subject_id=identity.get("subject_id"),
        session_id=identity.get("session_id"),
        condition=identity.get("condition"),
        recording_id=recording_id,
        tmin=tmin,
        annotations=annotations,
        provenance=(f"load:{source}" if source else "load:mne",),
        metadata=metadata,
        reference=identity.get("reference"),
    )


def load_array(
    data: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> EEG:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if ch_names is None:
        ch_names = tuple(f"CH{i + 1}" for i in range(arr.shape[-2]))
    return EEG(data=arr, sfreq=sfreq, ch_names=tuple(ch_names), provenance=("load:array",), **kwargs)


def _warning_note(caught) -> str | None:
    messages = []
    for item in caught:
        text = str(item.message).strip().replace("\n", " ")
        if text and text not in messages:
            messages.append(text)
    if not messages:
        return None
    return " | ".join(messages)[:500]


def _read_eeglab(path: Path):
    errors: list[Exception] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            epochs = mne.read_epochs_eeglab(path, verbose=False)
            if len(epochs) >= 1:
                epochs.info["description"] = _warning_note(caught)
                return epochs
        except Exception as exc:
            errors.append(exc)
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
            raw.info["description"] = _warning_note(caught)
            return raw
        except Exception as exc:
            errors.append(exc)
    hint = ""
    message = " ".join(str(exc).lower() for exc in errors)
    if "v7.3" in message or "pymatreader" in message or "h5py" in message:
        hint = (
            " This file is MATLAB v7.3. eegmath depends on pymatreader and h5py "
            "for that format; check that both are installed in this environment."
        )
    raise DataError(f"Could not read EEGLAB file {path.name}: {errors[-1]}.{hint}") from errors[-1]


_BIDS_CHANNEL_TYPES = {
    "EEG": "eeg",
    "EOG": "eog",
    "ECG": "ecg",
    "EKG": "ecg",
    "EMG": "emg",
    "MISC": "misc",
    "STIM": "stim",
    "TRIG": "stim",
    "REF": "misc",
    "MEG": "mag",
}


def _bids_stem(name: str) -> str:
    """Filename stem with only a trailing ``_eeg`` suffix removed.

    ``task-eegrest_eeg`` keeps the task ``eegrest``. Replacing every
    ``_eeg`` would turn that task into ``rest``.
    """
    suffix = Path(name).suffix
    stem = name[: -len(suffix)] if suffix else name
    if stem.endswith("_eeg"):
        stem = stem[: -len("_eeg")]
    return stem


def _channels_table(path: Path) -> Path | None:
    """BIDS sidecar ``*_channels.tsv`` next to an ``*_eeg`` recording."""
    candidate = path.with_name(_bids_stem(path.name) + "_channels.tsv")
    return candidate if candidate.exists() else None


def apply_channel_table(eeg: EEG, table: str | Path) -> EEG:
    """Set channel types from a BIDS ``channels.tsv``.

    The amplitude array is not rescaled. MNE readers already return volts
    for EDF, BDF, and BrainVision, even when the sidecar says microvolts.
    A ``status`` of ``bad`` is recorded in ``metadata["bad_channels"]``
    and the channel is kept. Dropping it is a separate call.
    """
    frame = pd.read_csv(table, sep="\t", encoding="utf-8-sig")
    if "name" not in frame.columns or "type" not in frame.columns:
        return eeg
    types = list(eeg.ch_types or ("eeg",) * eeg.n_channels)
    lookup = {str(row["name"]): str(row["type"]).upper() for _, row in frame.iterrows()}
    for index, name in enumerate(eeg.ch_names):
        kind = lookup.get(name)
        if kind in _BIDS_CHANNEL_TYPES:
            types[index] = _BIDS_CHANNEL_TYPES[kind]
    metadata = dict(eeg.metadata)
    if "status" in frame.columns:
        status = {
            str(row["name"]): str(row["status"]).strip().lower()
            for _, row in frame.iterrows()
            if pd.notna(row["status"])
        }
        # Kept, not dropped. A variance screen is a different decision.
        metadata["bad_channels"] = tuple(name for name in eeg.ch_names if status.get(name) == "bad")
    return eeg.spawn(
        f"channel_types:{Path(table).name}",
        ch_types=tuple(types),
        metadata=metadata,
    )


def _apply_bids_json(eeg: EEG, path: Path) -> EEG:
    """Copy reference, line frequency, and placement from a BIDS ``eeg.json``.

    ``n/a`` stays missing. A reference string is a label for the montage
    already stored in the file. This function does not re-reference and
    does not filter. An explicit ``reference`` already set on ``eeg`` is
    left as it is.
    """
    sidecar = path.with_name(_bids_stem(path.name) + "_eeg.json")
    if not sidecar.exists():
        return eeg
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Could not read BIDS sidecar {sidecar.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataError(f"BIDS sidecar {sidecar.name} is not a JSON object.")
    metadata = dict(eeg.metadata)
    reference = eeg.reference
    claimed = _optional_bids_text(payload.get("EEGReference"))
    if claimed is not None and reference is None:
        reference = claimed
    line = payload.get("PowerLineFrequency")
    if metadata.get("line_freq") in (None, "") and isinstance(line, (int, float)) and not isinstance(line, bool):
        metadata["line_freq"] = float(line)
    scheme = _optional_bids_text(payload.get("EEGPlacementScheme"))
    if scheme is not None:
        metadata["eeg_placement"] = scheme
    return eeg.spawn(f"bids_json:{sidecar.name}", metadata=metadata, reference=reference)


def load_file(path: str | Path, **identity: Any) -> EEG:
    """Load one EEG file. The suffix selects the reader."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No file or directory at {path}. "
            f"Supported EEG suffixes: {', '.join(sorted(_EEG_SUFFIXES))}. "
            "A BIDS dataset is a directory that contains dataset_description.json."
        )
    suffix = path.suffix.lower()
    if suffix == ".set":
        inst = _read_eeglab(path)
    elif suffix == ".edf":
        inst = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif suffix == ".bdf":
        inst = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif suffix == ".vhdr":
        inst = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    elif suffix == ".fif":
        name = path.name.lower()
        if name.endswith(("-epo.fif", "_epo.fif")):
            inst = mne.read_epochs(path, preload=True, verbose=False)
        else:
            inst = mne.io.read_raw_fif(path, preload=True, verbose=False)
    else:
        raise DataError(
            f"Unsupported EEG suffix '{suffix}'. Supported: {sorted(_EEG_SUFFIXES)}."
        )
    parsed = _bids_entities(path.stem)
    merged = {**parsed, **{k: v for k, v in identity.items() if v is not None}}
    recording = from_mne(inst, source=str(path), **merged)
    table = _channels_table(path)
    if table is not None:
        recording = apply_channel_table(recording, table)
    return _apply_bids_json(recording, path)


def _bids_entities(stem: str) -> dict[str, str | None]:
    found = {key: value for key, value in _ENTITY.findall(stem)}
    return {
        "subject_id": found.get("sub"),
        "session_id": found.get("ses"),
        "condition": found.get("task"),
        "recording_id": stem,
    }


def _bids_paths(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.glob("sub-*/*eeg/*_eeg.*")
        if path.suffix.lower() in _EEG_SUFFIXES and "derivatives" not in path.parts
    )
    files.extend(
        path
        for path in root.glob("sub-*/ses-*/eeg/*_eeg.*")
        if path.suffix.lower() in _EEG_SUFFIXES and path not in files and "derivatives" not in path.parts
    )
    return files


def _iter_bids(
    root: Path,
    *,
    subjects: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
):
    description = root / "dataset_description.json"
    if not description.exists():
        raise DataError(f"{root} has no dataset_description.json, so it is not a BIDS dataset.")
    participants: dict[str, dict] = {}
    table = root / "participants.tsv"
    if table.exists():
        frame = pd.read_csv(table, sep="\t", encoding="utf-8-sig")
        if "participant_id" in frame.columns:
            participants = frame.set_index("participant_id").to_dict(orient="index")
    dataset_meta = json.loads(description.read_text())
    subjects = _label_sequence(subjects, "subjects")
    tasks = _label_sequence(tasks, "tasks")
    sessions = _label_sequence(sessions, "sessions")
    subject_filter = _normalize_ids(subjects, "sub-")
    task_filter = set(tasks) if tasks is not None else None
    session_filter = _normalize_ids(sessions, "ses-")
    files = _bids_paths(root)
    if not files:
        raise DataError(f"No BIDS EEG files (*_eeg.set/edf/bdf/vhdr/fif) were found under {root}.")
    available: dict[str, list[str]] = {"subjects": [], "tasks": [], "sessions": []}
    seen: dict[str, set[str]] = {key: set() for key in available}
    catalogued: list[dict[str, str | None]] = []
    for path in files:
        entities = _bids_entities(_bids_stem(path.name))
        catalogued.append(entities)
        for key, label in (
            ("subjects", entities.get("subject_id")),
            ("tasks", entities.get("condition")),
            ("sessions", entities.get("session_id")),
        ):
            if label and label not in seen[key]:
                seen[key].add(str(label))
                available[key].append(str(label))
    matched = False
    for path, entities in zip(files, catalogued):
        subject = entities.get("subject_id")
        if subject_filter is not None and subject not in subject_filter:
            continue
        if task_filter is not None and entities.get("condition") not in task_filter:
            continue
        if session_filter is not None and entities.get("session_id") not in session_filter:
            continue
        eeg = load_file(path)
        participant_key = f"sub-{subject}" if subject else None
        meta = dict(eeg.metadata)
        meta["bids_dataset"] = dataset_meta.get("Name")
        meta["bids_root_name"] = root.name
        if participant_key and participant_key in participants:
            meta["participant"] = participants[participant_key]
        matched = True
        yield eeg.spawn("bids_metadata", metadata=meta)
    if not matched:
        requested = []
        if subjects is not None:
            requested.append(f"subjects={list(subjects)}")
        if tasks is not None:
            requested.append(f"tasks={list(tasks)}")
        if sessions is not None:
            requested.append(f"sessions={list(sessions)}")

        def _preview(labels: list[str]) -> str:
            if not labels:
                return "none"
            shown = ", ".join(labels[:8])
            extra = "" if len(labels) <= 8 else f", and {len(labels) - 8} more"
            return shown + extra

        raise DataError(
            "BIDS selection matched no recordings. "
            f"Requested {', '.join(requested)}. "
            f"Subject labels here include {_preview(available['subjects'])}. "
            f"Task labels include {_preview(available['tasks'])}. "
            f"Session labels include {_preview(available['sessions'])}. "
            "Pass each label without its entity prefix: tasks=['rest'] matches "
            "task-rest, and tasks=['task-rest'] matches nothing."
        )


def load_bids(
    root: str | Path,
    *,
    subjects: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
) -> EEGCollection:
    """Load a BIDS EEG dataset.

    Files under ``derivatives/`` are skipped. ``participants.tsv`` is
    copied into each recording's metadata under ``participant``. Selection
    arguments match BIDS entity labels without the entity prefix
    (``["01"]`` or ``["sub-01"]`` are both accepted for subjects).
    Each argument is a sequence of labels. A bare string is rejected,
    because a string is a sequence of characters.
    """
    return EEGCollection.from_sequence(
        _iter_bids(Path(root), subjects=subjects, tasks=tasks, sessions=sessions)
    )


def _label_sequence(values: Sequence[str] | None, name: str) -> Sequence[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        examples = {"subjects": "['01']", "tasks": "['rest']", "sessions": "['01']"}
        example = examples.get(name, "['01']")
        raise ParameterError(
            f"{name} must be a sequence of labels, for example {example}. "
            f"A bare string is a sequence of characters, so {name}={values!r} "
            "does not select that one label."
        )
    return values


def _normalize_ids(values: Sequence[str] | None, prefix: str) -> set[str] | None:
    if values is None:
        return None
    out = set()
    for value in values:
        text = str(value)
        out.add(text[len(prefix) :] if text.startswith(prefix) else text)
    return out


def _directory_files(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in _EEG_SUFFIXES and "derivatives" not in item.parts
    )


def load_directory(path: str | Path) -> EEGCollection:
    """Load every supported EEG file under a non-BIDS directory."""
    path = Path(path)
    files = _directory_files(path)
    if not files:
        raise DataError(f"No supported EEG files were found under {path}.")
    return EEGCollection.from_sequence(load_file(item) for item in files)


def iter_recordings(source, **kwargs):
    """Yield recordings one at a time.

    A directory is not read into memory as a collection first. ``load``
    still returns a collection when the caller asks for one, which does
    hold every recording. Use this iterator, or :func:`eegmath.feature_table`,
    when the directory is larger than you want resident at once.
    """
    if isinstance(source, EEG):
        yield source
        return
    if isinstance(source, EEGCollection):
        yield from source.recordings
        return
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from iter_recordings(item, **kwargs)
        return
    if isinstance(source, np.ndarray):
        yield load(source, **kwargs)
        return
    module = type(source).__module__
    if module.startswith("mne"):
        yield from_mne(source, **kwargs)
        return
    path = Path(source)
    if path.is_dir() and (path / "dataset_description.json").exists():
        _reject_unknown(
            kwargs,
            _BIDS_SELECTION,
            "A BIDS directory accepts subjects, tasks, and sessions. "
            "It does not apply {unknown} to every recording. Set identity "
            "on one recording with spawn(), or pass it to load_file().",
        )
        selection = {key: kwargs[key] for key in _BIDS_SELECTION if key in kwargs}
        yield from _iter_bids(path, **selection)
        return
    if path.is_dir():
        _reject_unknown(
            kwargs,
            set(),
            "A directory of files does not apply {unknown} to every recording. "
            "Load one file, or set identity with spawn() after loading.",
        )
        files = _directory_files(path)
        if not files:
            raise DataError(f"No supported EEG files were found under {path}.")
        for item in files:
            yield load_file(item)
        return
    yield load_file(path, **{k: v for k, v in kwargs.items() if v is not None})


def load(source, **kwargs) -> EEG | EEGCollection:
    """Load an array, file, directory, BIDS root, MNE object, or list of any of those.

    Keyword arguments are forwarded to :func:`load_array` or :func:`load_file`.
    A BIDS directory accepts ``subjects``, ``tasks``, and ``sessions``.
    A directory of files accepts no identity arguments. Unknown arguments
    raise :class:`~eegmath.errors.ParameterError` instead of being ignored.
    """
    if isinstance(source, EEG):
        return source
    if isinstance(source, EEGCollection):
        return source
    if isinstance(source, (list, tuple)):
        loaded = [load(item, **kwargs) for item in source]
        flat: list[EEG] = []
        for item in loaded:
            if isinstance(item, EEGCollection):
                flat.extend(item.recordings)
            else:
                flat.append(item)
        return EEGCollection.from_sequence(flat)
    if isinstance(source, np.ndarray):
        if "sfreq" not in kwargs:
            raise DataError("Loading an array requires sfreq.")
        return load_array(source, **kwargs)
    module = type(source).__module__
    if module.startswith("mne"):
        return from_mne(source, **kwargs)
    path = Path(source)
    if path.is_dir():
        if (path / "dataset_description.json").exists():
            _reject_unknown(
                kwargs,
                _BIDS_SELECTION,
                "A BIDS directory accepts subjects, tasks, and sessions. "
                "It does not apply {unknown} to every recording.",
            )
            selection = {key: kwargs[key] for key in _BIDS_SELECTION if key in kwargs}
            return load_bids(path, **selection)
        _reject_unknown(
            kwargs,
            set(),
            "A directory of files does not apply {unknown} to every recording. "
            "Load one file, or set identity with spawn() after loading.",
        )
        return load_directory(path)
    return load_file(path, **kwargs)
