"""Smoke tests against the two public validation datasets, when they are present.

These checks confirm ingestion and a few numerical contracts. They are
not the scientific notebooks. A checkout without ``external_data`` skips
them; that absence is not a passing validation of the recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegmath import extract_features, load
from eegmath.recording import EEGCollection

ROOT = Path(__file__).resolve().parents[1] / "external_data"
CAMBRIDGE = ROOT / "cambridge_propofol" / "extracted" / "Sedation-RestingState"
NEMAR = ROOT / "nemar_nm000109"

pytestmark = pytest.mark.skipif(
    not CAMBRIDGE.exists() or not NEMAR.exists(),
    reason="Public validation recordings are stored outside the package and are not in this checkout.",
)


def test_cambridge_epochs_load_without_reprocessing_and_keep_volt_scale():
    path = sorted(CAMBRIDGE.glob("*.set"))[0]
    eeg = load(path)
    assert eeg.epoched
    assert eeg.sfreq == 250.0
    assert eeg.n_channels == 91
    assert eeg.n_times == 2500
    assert eeg.units == "V"
    assert eeg.provenance == (f"load:{path}",) or eeg.provenance[0].startswith("load:")
    assert not any("bandpass" in note or "rereference" in note for note in eeg.provenance)
    channel_std = float(np.std(eeg.data))
    assert 1e-7 < channel_std < 1e-3
    # A spatial mean much smaller than a typical channel is the signature of
    # an average reference already present in the file.
    spatial_mean = float(np.std(eeg.data.mean(axis=1)))
    assert spatial_mean < 0.05 * channel_std
    picked = eeg.pick_channels(["O1", "O2", "Oz"])
    table = extract_features(picked, ["alpha_relative", "alpha_peak_freq", "rms"], reduce_epochs=True)
    assert len(table) == 3
    assert table["alpha_relative"].between(0, 1).all()
    # A single retained relative column must still be a share of the full
    # partition. Exactly 1 would mean every other band was dropped from the denominator.
    assert (table["alpha_relative"] < 0.99).all()
    assert np.isfinite(table["alpha_peak_freq"]).all()


def test_nemar_bids_labels_conditions_and_the_ecg_channel():
    collection = load(NEMAR, subjects=["00"])
    assert isinstance(collection, EEGCollection)
    assert len(collection) == 2
    conditions = {rec.condition for rec in collection}
    assert conditions == {"rest", "mentalArithmetic"}
    for rec in collection:
        assert rec.sfreq == 500.0
        assert rec.units == "V"
        assert rec.subject_id == "00"
        assert not rec.epoched
        assert rec.ch_types is not None
        assert rec.ch_types[rec.ch_names.index("ECG")] == "ecg"
        assert rec.metadata.get("highpass") == 0.5
        assert rec.metadata.get("lowpass") == 45.0
        assert rec.reference is None
        assert rec.metadata.get("bad_channels") == ()
        assert "10/20" in str(rec.metadata.get("eeg_placement"))
        assert 1e-6 < float(np.std(rec.data)) < 1e-2
        assert int(rec.metadata["participant"]["age"]) == 21
    eeg = next(rec for rec in collection if rec.condition == "rest").drop_channels(["ECG"])
    table = extract_features(eeg.pick_channels(["Cz"]), ["alpha_relative", "total_power"])
    assert len(table) == 1
    assert 0 <= float(table["alpha_relative"].iloc[0]) <= 1
