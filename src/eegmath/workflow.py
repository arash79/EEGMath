"""Optional composition of loading, preprocessing, and feature extraction.

Preprocessing steps are applied only when the caller supplies them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from eegmath.errors import ParameterError
from eegmath.features.extract import FeatureConfig, extract_features
from eegmath.io.load import iter_recordings
from eegmath.preprocess.operations import apply_steps, window
from eegmath.recording import EEG


def _prepare(
    recording: EEG,
    preprocess: Sequence[tuple[str, dict]] | None,
    window_sec: float | None,
    window_overlap: float,
) -> EEG:
    out = apply_steps(recording, preprocess) if preprocess else recording
    if window_sec is not None:
        out = window(out, window_sec, window_overlap)
    return out


def _bids_root(source) -> bool:
    if isinstance(source, (list, tuple)):
        return bool(source) and all(_bids_root(item) for item in source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.is_dir() and (path / "dataset_description.json").is_file()
    return False


def feature_table(
    source,
    features: str | Sequence[str] = "spectral",
    *,
    preprocess: Sequence[tuple[str, dict]] | None = None,
    window_sec: float | None = None,
    window_overlap: float = 0.0,
    reduce_epochs: bool = False,
    config: FeatureConfig | None = None,
    subjects: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load ``source`` and extract features.

    ``preprocess`` is a sequence of ``(step_name, kwargs)`` applied to
    each recording before features are computed. The default is to leave
    the recording as it was loaded. ``window_sec`` likewise defaults to
    using the loaded time axis rather than cutting new epochs.

    ``subjects``, ``tasks``, and ``sessions`` are forwarded to the BIDS
    loader. They select recordings in a directory that contains
    ``dataset_description.json``. On an array, an MNE object, a file, or
    a plain directory they raise :class:`~eegmath.errors.ParameterError`
    instead of being ignored. Pass a list, such as ``["01"]`` or
    ``["sub-01"]``. A bare string is rejected, because it would be read
    character by character. The same arguments are accepted by
    :func:`eegmath.load`.
    """
    selection = {
        key: value
        for key, value in {
            "subjects": subjects,
            "tasks": tasks,
            "sessions": sessions,
        }.items()
        if value is not None
    }
    if selection and not _bids_root(source):
        missing_path = isinstance(source, (str, Path)) and not Path(source).exists()
        if not missing_path:
            raise ParameterError(
                "subjects, tasks, and sessions select recordings inside a BIDS "
                "directory, one that contains dataset_description.json. "
                "load() accepts the same arguments. They are not a filter on "
                "an array, a file, a plain directory, or a table that is already loaded."
            )
    frames = []
    for recording in iter_recordings(source, **selection):
        prepared = _prepare(recording, preprocess, window_sec, window_overlap)
        frames.append(extract_features(prepared, features, config=config, reduce_epochs=reduce_epochs))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
