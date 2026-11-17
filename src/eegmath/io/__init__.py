"""Data ingestion."""

from eegmath.io.load import from_mne, iter_recordings, load, load_array, load_bids, load_directory, load_file

__all__ = [
    "from_mne",
    "iter_recordings",
    "load",
    "load_array",
    "load_bids",
    "load_directory",
    "load_file",
]
