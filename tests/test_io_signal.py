"""Loading, preprocessing, signal analyses, and stage composition."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from eegmath.errors import DataError, DesignError, ParameterError
from eegmath.io.load import load
from eegmath.preprocess import (
    apply_steps,
    bandpass,
    detect_bad_channels,
    detrend,
    notch,
    qc_summary,
    rereference,
    resample,
    to_mne,
    window,
    zscore,
)
from eegmath.recording import EEG, EEGCollection, align_channels
from eegmath.signal import (
    adf_stationarity,
    average_epochs,
    intertrial_phase_clustering,
    kpss_stationarity,
    modulation_index,
    morlet_power,
    peak_measure,
)
from eegmath.viz import plot_psd, plot_timeseries
from eegmath.workflow import feature_table
from synthetic import sine


def _raw(path: Path, subject: str, task: str) -> None:
    info = mne.create_info(["Fz", "Cz"], 100.0, ch_types="eeg", verbose=False)
    data = np.vstack([sine(10.0, 100.0, 2.0), sine(6.0, 100.0, 2.0)]) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.save(path, overwrite=True, verbose=False)


def test_load_array_mne_fif_and_bids(tmp_path: Path):
    array = load(np.vstack([sine(8.0, 100.0, 1.0)]), sfreq=100.0, ch_names=["Oz"], units="uV")
    assert array.n_channels == 1 and array.units == "uV"
    with pytest.raises(DataError):
        load(np.zeros((2, 10)))
    info = mne.create_info(["Fz"], 100.0, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(sine(10.0, 100.0, 1.0)[None, :] * 1e-6, info, verbose=False)
    loaded = load(raw)
    assert loaded.units == "V"
    assert loaded.sfreq == 100.0

    bids = tmp_path / "dataset"
    eeg_dir = bids / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (bids / "dataset_description.json").write_text('{"Name": "Synthetic EEG", "BIDSVersion": "1.8.0"}\n')
    (bids / "participants.tsv").write_text("\ufeffparticipant_id\tage\nsub-01\t31\n")
    _raw(eeg_dir / "sub-01_task-rest_eeg.fif", "01", "rest")
    other = bids / "sub-02" / "eeg"
    other.mkdir(parents=True)
    _raw(other / "sub-02_task-calc_eeg.fif", "02", "calc")
    collection = load(bids)
    assert isinstance(collection, EEGCollection)
    assert len(collection) == 2
    by_subject = {rec.subject_id: rec for rec in collection}
    assert set(by_subject) == {"01", "02"}
    assert by_subject["01"].condition == "rest"
    assert by_subject["01"].metadata["participant"]["age"] == 31
    selected = load(bids, tasks=["calc"])
    assert len(selected) == 1 and selected[0].condition == "calc"
    one = load(eeg_dir / "sub-01_task-rest_eeg.fif")
    assert one.subject_id == "01"
    directory = load([eeg_dir / "sub-01_task-rest_eeg.fif", other / "sub-02_task-calc_eeg.fif"])
    assert len(directory) == 2


def test_collection_does_not_silently_align_mismatches():
    a = EEG(np.ones((2, 20)), 100.0, ("A", "B"), units="V", subject_id="1")
    b = EEG(np.ones((3, 40)), 200.0, ("B", "C", "A"), units="uV", subject_id="2")
    collection = EEGCollection.from_sequence([a, b])
    report = collection.compatibility()
    assert report["same_sfreq"] is False
    assert report["same_units"] is False
    with pytest.raises(DataError):
        collection.require(same_sfreq=True, same_units=True)
    aligned = align_channels(collection)
    assert aligned[0].ch_names == aligned[1].ch_names == ("A", "B")
    assert aligned[0].sfreq != aligned[1].sfreq


def test_preprocessing_changes_only_what_was_requested():
    sfreq = 200.0
    duration = 8.0
    slow = sine(1.0, sfreq, duration)
    fast = sine(20.0, sfreq, duration)
    eeg = EEG(np.vstack([slow + fast, slow + fast]), sfreq, ("C3", "C4"), units="V")
    original = eeg.provenance
    filtered = bandpass(eeg, 8.0, 30.0)
    assert original == ()
    assert filtered.provenance[-1].startswith("bandpass")
    # The 20 Hz component remains; the 1 Hz component does not.
    residual = filtered.data[0]
    assert np.corrcoef(residual, fast)[0, 1] > 0.9
    assert abs(np.corrcoef(residual, slow)[0, 1]) < 0.3

    line = sine(50.0, sfreq, duration) + 0.2 * sine(10.0, sfreq, duration)
    line_eeg = EEG(line[None, :], sfreq, ("Cz",), units="V")
    before = line_eeg.data.copy()
    notched = notch(line_eeg)
    assert np.allclose(line_eeg.data, before)
    assert np.std(notched.data) < 0.5 * np.std(before)
    skipped = notch(EEG(line[None, :], sfreq, ("Cz",), units="V", metadata={"lowpass": 40.0}))
    assert "notch_skipped" in skipped.metadata
    assert np.allclose(skipped.data, line)
    # A line above the recorded low-pass is left out. A line below it is applied.
    partial = notch(
        EEG(line[None, :], sfreq, ("Cz",), units="V", metadata={"highpass": 0.5, "lowpass": 40.0}),
        (30.0, 50.0),
    )
    assert partial.metadata["notch_skipped"] == [50.0]
    assert partial.metadata["highpass"] == 0.5
    assert partial.metadata["lowpass"] == 40.0
    assert "notch:[30.0]" in partial.provenance[-1]
    assert np.isfinite(partial.data).all()

    levels = np.array([1.0, 2.0, 3.0])[:, None] * np.ones(int(sfreq))
    referenced = rereference(EEG(levels, sfreq, ("A", "B", "C"), units="V"))
    assert referenced.reference == "average"
    assert np.allclose(referenced.data.mean(axis=0), 0.0, atol=1e-8)

    resampled = resample(eeg, 100.0)
    assert resampled.sfreq == 100.0
    assert resampled.duration == pytest.approx(eeg.duration, abs=0.02)

    windows = window(eeg, 1.0, 0.5)
    assert windows.epoched and windows.n_epochs == 15
    assert windows.metadata["window_onsets"][0] == pytest.approx(0.0)
    assert windows.metadata["window_onsets"][-1] == pytest.approx(7.0)
    assert len(windows.metadata["window_onsets"]) == 15
    parent = window(EEG(np.stack([eeg.data, eeg.data]), sfreq, eeg.ch_names, units="V"), 2.0)
    assert parent.n_epochs == 8
    assert parent.metadata["parent_epoch"][:4] == [0, 0, 0, 0]

    flat = zscore(EEG(np.vstack([sine(5, sfreq, 2), np.ones(int(2 * sfreq))]), sfreq, ("A", "B"), units="uV"))
    assert np.isnan(flat.data[1]).all()
    assert zscore(EEG(sine(5, sfreq, 2)[None, :], sfreq, ("A",), units="uV")).data.mean() == pytest.approx(0, abs=1e-8)

    rng = np.random.default_rng(0)
    noise = rng.normal(scale=1e-6, size=(8, 500))
    noise = np.vstack([noise, noise[0] * 50])
    names = [f"CH{i}" for i in range(8)] + ["LOUD"]
    assert "LOUD" in detect_bad_channels(EEG(noise, 100.0, tuple(names), units="V"))
    identical = np.vstack([sine(8, 100, 2)] * 6)
    identical[5] *= 40
    found = detect_bad_channels(EEG(identical, 100.0, tuple(f"E{i}" for i in range(6)), units="uV"))
    assert found == ["E5"]
    summary = qc_summary(EEG(noise, 100.0, tuple(names), units="V"))
    assert summary["amplitude_scale_hint"] == "consistent_with_volts"
    assert apply_steps(eeg, []) is eeg


def _coef(series: np.ndarray, reference: np.ndarray) -> float:
    """Least-squares coefficient of ``reference`` in ``series``."""
    return float(np.dot(series, reference) / np.dot(reference, reference))


def test_notch_filters_each_epoch_and_keeps_the_voltage_scale():
    sfreq = 200.0
    duration = 8.0
    line = sine(50.0, sfreq, duration) + 0.2 * sine(10.0, sfreq, duration)
    ecg = sine(1.2, sfreq, duration)
    ten = sine(10.0, sfreq, duration)
    fifty = sine(50.0, sfreq, duration)
    one = np.stack([line, line.copy(), ecg])
    eeg = EEG(
        np.stack([one, one * 0.5]),
        sfreq,
        ("C3", "C4", "ECG"),
        units="V",
        ch_types=("eeg", "eeg", "ecg"),
        tmin=-0.2,
    )
    before = eeg.data.copy()
    notched = notch(eeg)
    assert np.allclose(eeg.data, before)
    assert notched.tmin == -0.2 and notched.n_epochs == 2
    assert notched.provenance[-1] == "notch:[50.0]"
    # A 6.6 s kernel has a long transient on an 8 s epoch, so correlation
    # with the whole epoch is a weak check. The tone coefficients are not.
    for epoch, scale in ((0, 1.0), (1, 0.5)):
        assert _coef(notched.data[epoch, 0], ten) == pytest.approx(0.2 * scale, abs=1e-3)
        assert abs(_coef(notched.data[epoch, 0], fifty)) < 0.05 * scale
        assert np.allclose(notched.data[epoch, 2], before[epoch, 2])
    separate = np.stack([fifty, ten])[:, None, :]
    separated = notch(EEG(separate, sfreq, ("Cz",), units="V"))
    assert np.std(separated.data[0, 0]) < 0.2 * np.std(fifty)
    assert _coef(separated.data[1, 0], ten) == pytest.approx(1.0, abs=1e-3)
    volts = notch(EEG(line[None, :], sfreq, ("Cz",), units="V"))
    microvolts = notch(EEG((line * 1e6)[None, :], sfreq, ("Cz",), units="uV"))
    assert microvolts.data == pytest.approx(volts.data * 1e6, rel=1e-6, abs=1e-6)
    short = EEG(eeg.data[:, :, :800], sfreq, eeg.ch_names, units="V", ch_types=eeg.ch_types)
    with pytest.raises(ParameterError, match="longer than"):
        notch(short)
    widened = notch(short, trans_bandwidth=4.0)
    assert abs(_coef(widened.data[0, 0], fifty[:800])) < 0.05
    assert _coef(widened.data[0, 0], ten[:800]) == pytest.approx(0.2, abs=1e-3)
    with pytest.raises(ParameterError, match="trans_bandwidth"):
        notch(eeg, trans_bandwidth=0.0)


def test_bandpass_refuses_a_kernel_longer_than_the_signal():
    short = EEG(sine(10.0, 200.0, 1.0)[None, :], 200.0, ("Cz",), units="V")
    with pytest.raises(ParameterError, match="longer than"):
        bandpass(short, 1.0, 40.0)
    fitted = bandpass(EEG(sine(10.0, 200.0, 8.0)[None, :], 200.0, ("Cz",), units="V"), 1.0, 40.0)
    assert np.isfinite(fitted.data).all()
    assert fitted.provenance[-1] == "bandpass:1.0-40.0Hz"


def test_bad_channel_screen_is_not_collapsed_by_one_missing_channel():
    names = tuple(f"E{i}" for i in range(6))
    identical = np.vstack([sine(8.0, 100.0, 2.0)] * 6)
    identical[5] *= 40.0
    assert detect_bad_channels(EEG(identical, 100.0, names, units="uV")) == ["E5"]
    missing = np.vstack([identical, np.full(identical.shape[1], np.nan)])
    assert detect_bad_channels(EEG(missing, 100.0, names + ("NAN",), units="uV")) == ["E5", "NAN"]
    partial = identical.copy()
    partial[0, : partial.shape[1] // 2] = np.nan
    partial_flags = detect_bad_channels(EEG(partial, 100.0, names, units="uV"))
    assert partial_flags[0] == "E0"
    assert "E5" in partial_flags
    intact = np.vstack([sine(8.0, 100.0, 2.0)] * 5)
    only_missing = np.vstack([intact, np.full(intact.shape[1], np.nan)])
    assert detect_bad_channels(EEG(only_missing, 100.0, tuple(f"C{i}" for i in range(5)) + ("NAN",), units="uV")) == ["NAN"]
    # Two channels do not make a flat trace a robust z-score outlier.
    pair = np.vstack([sine(8.0, 100.0, 2.0), np.zeros(int(2.0 * 100.0))])
    assert detect_bad_channels(EEG(pair, 100.0, ("A", "B"), units="uV")) == []


def test_signal_methods_and_optional_workflow_do_not_preprocess(alpha_eeg):
    power, times = morlet_power(alpha_eeg, np.array([4.0, 10.0, 20.0]), n_cycles=5)
    # Channel 0 is a 10 Hz sine, so that wavelet dominates the other two.
    mean_power = power[0, 0].mean(axis=-1)
    assert mean_power[1] > mean_power[0]
    assert mean_power[1] > mean_power[2]
    assert len(times) == alpha_eeg.n_times

    evoked, evoked_times = average_epochs(
        EEG(np.stack([alpha_eeg.data, alpha_eeg.data]), alpha_eeg.sfreq, alpha_eeg.ch_names, units="uV")
    )
    peak = peak_measure(evoked[0], evoked_times, 0.0, 0.2, mode="pos")
    assert 0.0 <= peak["latency"] <= 0.2

    times = np.arange(int(20.0 * 200.0)) / 200.0
    slow = np.sin(2 * np.pi * 2.0 * times)
    carrier = np.sin(2 * np.pi * 40.0 * times)
    # The slow oscillation has to be present in the voltage. A product of a
    # carrier and a modulator has sidebands, not a phase-giving 2 Hz rhythm.
    coupled = slow + (1.0 + 0.9 * slow) * carrier
    coupled_mi = modulation_index(coupled, 200.0, phase_band=(1.0, 4.0), amplitude_band=(30.0, 45.0))
    uncoupled_mi = modulation_index(carrier, 200.0, phase_band=(1.0, 4.0), amplitude_band=(30.0, 45.0))
    assert coupled_mi > 0.05
    assert uncoupled_mi < coupled_mi
    white = np.random.default_rng(1).normal(size=500)
    assert adf_stationarity(white)["p_value"] < 0.05
    walk = np.cumsum(white)
    assert adf_stationarity(walk)["p_value"] > 0.05
    gappy = white.copy()
    gappy[250] = np.nan
    gap = adf_stationarity(gappy)
    assert np.isnan(gap["p_value"])
    assert gap["n_obs"] == 500
    assert "non-finite" in gap["note"]

    table = feature_table(alpha_eeg, ["alpha_relative", "alpha_peak_freq"])
    assert table["alpha_peak_freq"].iloc[0] == pytest.approx(10.0, abs=0.5)
    assert "bandpass" not in " ".join(alpha_eeg.provenance)
    figure, _ax = plot_timeseries(alpha_eeg, duration=1.0)
    _psd_fig, _ax = plot_psd(*__import__("eegmath.spectral", fromlist=["welch_psd"]).welch_psd(alpha_eeg.data, alpha_eeg.sfreq), ch_names=list(alpha_eeg.ch_names))
    figure.clf()
    _psd_fig.clf()


def test_bids_channel_table_marks_ecg_and_reference_ignores_it(tmp_path: Path):
    from eegmath.io.load import apply_channel_table

    table = tmp_path / "channels.tsv"
    table.write_text("name\ttype\tunits\nA\tEEG\tuV\nB\tEEG\tuV\nECG\tECG\tuV\n")
    data = np.array([1.0, 3.0, 100.0])[:, None] * np.ones(50)
    eeg = apply_channel_table(EEG(data, 100.0, ("A", "B", "ECG"), units="V"), table)
    assert eeg.ch_types == ("eeg", "eeg", "ecg")
    with pytest.warns(UserWarning, match="exactly two EEG channels"):
        referenced = rereference(eeg)
    assert np.allclose(referenced.data[:2].mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(referenced.data[2], 100.0)


def test_public_source_does_not_name_a_clinical_protocol():
    root = Path(__file__).resolve().parents[1] / "src" / "eegmath"
    banned = ("anesthesia", "anaesthesia", "propofol", "rousable", "streamlit", "dashboard")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            if word in text:
                offenders.append(f"{path.name}:{word}")
    assert offenders == []


def test_epoched_modulation_index_is_the_mean_of_epochs():
    from eegmath.signal.analysis import modulation_index_channels

    sfreq = 200.0
    times = np.arange(int(4.0 * sfreq)) / sfreq
    slow = np.sin(2 * np.pi * 2.0 * times)
    slow_shifted = np.sin(2 * np.pi * 2.0 * times + np.pi)
    carrier = np.sin(2 * np.pi * 40.0 * times)
    first = slow + (1.0 + 0.9 * slow) * carrier
    second = slow_shifted + (1.0 + 0.9 * slow_shifted) * carrier
    bands = {"phase_band": (1.0, 4.0), "amplitude_band": (30.0, 45.0)}
    eeg = EEG(np.stack([first, second])[:, None, :], sfreq, ("Cz",), units="uV")
    table = modulation_index_channels(eeg, **bands)
    expected = np.mean([modulation_index(first, sfreq, **bands), modulation_index(second, sfreq, **bands)])
    evoked = modulation_index(np.mean(np.stack([first, second]), axis=0), sfreq, **bands)
    assert table.loc[0, "modulation_index"] == pytest.approx(expected)
    assert table.loc[0, "modulation_index"] > evoked + 0.02
    assert table.loc[0, "note"] == "mean of within-epoch modulation indices"


def test_peak_measure_rejects_a_window_of_missing_values():
    with pytest.raises(ParameterError, match="finite"):
        peak_measure(np.array([np.nan, np.nan]), np.array([0.0, 0.1]), 0.0, 0.2)


def test_bids_stem_keeps_eeg_inside_the_task_name():
    from eegmath.io.load import _bids_entities, _bids_stem

    stem = _bids_stem("sub-01_task-eegrest_eeg.edf")
    entities = _bids_entities(stem)
    assert entities["condition"] == "eegrest"
    assert entities["subject_id"] == "01"


def test_intertrial_phase_consistency_removes_the_two_epoch_bias():
    from eegmath.signal import intertrial_phase_consistency

    sfreq = 200.0
    tone = sine(10.0, sfreq, 2.0)
    locked = EEG(np.stack([tone, tone])[:, None, :], sfreq, ("Cz",), units="uV")
    freqs = np.array([10.0])
    itpc, times = intertrial_phase_clustering(locked, freqs, n_cycles=4.0)
    ppc, _times = intertrial_phase_consistency(locked, freqs, n_cycles=4.0)
    middle = slice(times.size // 4, 3 * times.size // 4)
    # (n P^2 - 1) / (n - 1) with n = 2.
    expected = 2.0 * np.square(itpc) - 1.0
    assert ppc[0, 0, middle] == pytest.approx(expected[0, 0, middle], abs=1e-6)
    assert ppc[0, 0, middle].mean() > 0.9
    opposite = sine(10.0, sfreq, 2.0, phase=np.pi)
    anti = EEG(np.stack([tone, opposite])[:, None, :], sfreq, ("Cz",), units="uV")
    anti_ppc, _times = intertrial_phase_consistency(anti, freqs, n_cycles=4.0)
    anti_itpc, _times = intertrial_phase_clustering(anti, freqs, n_cycles=4.0)
    assert anti_itpc[0, 0, middle].mean() < 0.2
    assert anti_ppc[0, 0, middle].mean() < -0.5


def test_modulation_index_is_missing_when_a_phase_bin_is_empty():
    from eegmath.signal.analysis import modulation_index

    series = np.random.default_rng(2).normal(size=500)
    assert np.isnan(modulation_index(series, 250.0, n_bins=400))
    assert np.isfinite(modulation_index(series, 250.0, n_bins=18))
    with pytest.raises(ParameterError, match="Band-pass"):
        modulation_index(series, 100.0, phase_band=(0.5, 80.0))


def test_file_identity_does_not_relabel_units_or_apply_to_a_directory(tmp_path: Path):
    info = mne.create_info(["Cz"], 100.0, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(np.full((1, 200), 1e-6), info, verbose=False)
    path = tmp_path / "sub-01_task-rest_eeg.fif"
    raw.save(path, overwrite=True, verbose=False)
    with pytest.raises(ParameterError, match="units"):
        load(path, units="uV")
    labelled = load(path, reference="linked-mastoid")
    assert labelled.units == "V"
    assert labelled.reference == "linked-mastoid"
    assert np.allclose(labelled.data, 1e-6)
    (tmp_path / "sub-01_task-rest_eeg.json").write_text(
        '{"EEGReference": "n/a", "PowerLineFrequency": 50, "EEGPlacementScheme": "10-20"}',
        encoding="utf-8",
    )
    (tmp_path / "sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\tunits\tstatus\nCz\tEEG\tuV\tbad\n",
        encoding="utf-8",
    )
    described = load(path)
    assert described.reference is None
    assert described.metadata["line_freq"] == 50.0
    assert described.metadata["eeg_placement"] == "10-20"
    assert described.metadata["bad_channels"] == ("Cz",)
    assert described.ch_names == ("Cz",)
    with pytest.raises(ParameterError, match="directory"):
        load(tmp_path, condition="rest")


def test_windowing_a_periodic_tone_can_lock_its_phase():
    from eegmath.signal import intertrial_phase_consistency

    sfreq = 200.0
    continuous = EEG(sine(10.0, sfreq, 16.0)[None, :], sfreq, ("Cz",), units="uV")
    fitted = window(continuous, 1.0)
    itpc, times = intertrial_phase_clustering(fitted, np.array([10.0]), n_cycles=4.0)
    ppc, _times = intertrial_phase_consistency(fitted, np.array([10.0]), n_cycles=4.0)
    middle = slice(times.size // 4, 3 * times.size // 4)
    assert fitted.n_epochs == 16
    assert itpc[0, 0, middle].mean() == pytest.approx(1.0, abs=1e-6)
    assert ppc[0, 0, middle].mean() == pytest.approx(1.0, abs=1e-6)

    offset = window(continuous, 1.05)
    itpc, times = intertrial_phase_clustering(offset, np.array([10.0]), n_cycles=4.0)
    ppc, _times = intertrial_phase_consistency(offset, np.array([10.0]), n_cycles=4.0)
    middle = slice(times.size // 4, 3 * times.size // 4)
    # Half-cycle steps and an odd epoch count leave a remainder of 1/N.
    assert offset.n_epochs == 15
    assert itpc[0, 0, middle].mean() == pytest.approx(1.0 / 15.0, abs=1e-6)
    assert ppc[0, 0, middle].mean() == pytest.approx(-1.0 / 15.0, abs=1e-6)

    tone = sine(10.0, sfreq, 2.0)
    opposite = sine(10.0, sfreq, 2.0, phase=np.pi)
    split = np.stack([tone if i % 2 == 0 else opposite for i in range(8)])
    anti = EEG(split[:, None, :], sfreq, ("Cz",), units="uV")
    itpc, times = intertrial_phase_clustering(anti, np.array([10.0]), n_cycles=4.0)
    ppc, _times = intertrial_phase_consistency(anti, np.array([10.0]), n_cycles=4.0)
    middle = slice(times.size // 4, 3 * times.size // 4)
    assert itpc[0, 0, middle].mean() == pytest.approx(0.0, abs=1e-8)
    assert ppc[0, 0, middle].mean() == pytest.approx(-1.0 / 7.0, abs=1e-8)

    # Seven cycles at 10 Hz is 223 samples, longer than a 1 s epoch.
    with pytest.raises(ValueError, match="longer than the signal"):
        intertrial_phase_clustering(fitted, np.array([10.0]))


def test_intertrial_phase_clustering_detects_a_shared_phase():
    sfreq = 200.0
    tone = sine(10.0, sfreq, 2.0)
    locked = EEG(np.stack([tone, tone])[:, None, :], sfreq, ("Cz",), units="uV")
    itpc, times = intertrial_phase_clustering(locked, np.array([10.0]), n_cycles=4.0)
    middle = slice(times.size // 4, 3 * times.size // 4)
    assert itpc[0, 0, middle].mean() > 0.9
    opposite = sine(10.0, sfreq, 2.0, phase=np.pi)
    anti = EEG(np.stack([tone, opposite])[:, None, :], sfreq, ("Cz",), units="uV")
    anti_itpc, _times = intertrial_phase_clustering(anti, np.array([10.0]), n_cycles=4.0)
    assert anti_itpc[0, 0, middle].mean() < 0.2
    with pytest.raises(DataError, match="epochs"):
        intertrial_phase_clustering(EEG(tone[None, :], sfreq, ("Cz",), units="uV"), np.array([10.0]))


def test_preprocessing_accepts_filter_limits_stored_by_mne():
    from eegmath.io.load import from_mne

    info = mne.create_info(["Cz"], 200.0, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(sine(10.0, 200.0, 16.0)[None, :], info, verbose=False)
    loaded = from_mne(raw)
    assert loaded.metadata["highpass"] == 0.0
    filtered = bandpass(loaded, 1.0, 40.0)
    assert filtered.metadata["highpass"] == pytest.approx(1.0)
    assert filtered.metadata["lowpass"] == pytest.approx(40.0)
    # An existing stricter high-pass is not replaced by a lower requested cutoff.
    seeded = loaded.spawn("limits", metadata={**loaded.metadata, "highpass": 2.0, "lowpass": 45.0})
    again = bandpass(seeded, 0.5, 30.0)
    assert again.metadata["highpass"] == pytest.approx(2.0)
    assert again.metadata["lowpass"] == pytest.approx(30.0)
    with pytest.warns(UserWarning, match="exactly two EEG channels"):
        referred = rereference(EEG(np.vstack([sine(10, 200, 2), sine(10, 200, 2)]), 200.0, ("A", "B"), units="V", metadata={"highpass": 0.5, "lowpass": 40.0}))
    assert referred.metadata["highpass"] == pytest.approx(0.5)
    assert referred.metadata["lowpass"] == pytest.approx(40.0)
    slower = resample(seeded, 80.0)
    assert slower.metadata["highpass"] == pytest.approx(2.0)
    assert slower.metadata["lowpass"] == pytest.approx(40.0)
    # Upsampling cannot raise the low-pass to the new Nyquist frequency.
    bare = EEG(np.zeros((1, 200)), 100.0, ("Cz",), units="V")
    up = resample(bare, 200.0)
    assert up.metadata["lowpass"] == pytest.approx(50.0)
    assert up.metadata["highpass"] is None


def test_bandpass_leaves_a_typed_ecg_channel_unchanged():
    sfreq = 200.0
    heart = sine(1.0, sfreq, 4.0)
    mixed = sine(10.0, sfreq, 4.0) + heart
    eeg = EEG(np.vstack([mixed, heart]), sfreq, ("Cz", "ECG"), units="V", ch_types=("eeg", "ecg"))
    filtered = bandpass(eeg, 8.0, 30.0)
    assert np.allclose(filtered.data[1], heart)
    assert abs(np.corrcoef(filtered.data[0], heart)[0, 1]) < 0.3


def test_linear_detrend_matches_an_independent_polynomial_fit():
    import warnings

    from scipy import signal

    sfreq = 200.0
    index = np.arange(int(sfreq * 4.0), dtype=float)
    series = np.sin(2 * np.pi * 10.0 * index / sfreq) + 0.25 * index
    other = 2.0 * series + 0.1
    eeg = EEG(np.stack([series, other]), sfreq, ("Cz", "Pz"), units="uV")
    residual = detrend(eeg).data
    for original, cleaned in zip((series, other), residual):
        line = np.polynomial.Polynomial.fit(index, original, deg=1)(index)
        assert cleaned == pytest.approx(original - line, abs=1e-12)
        assert abs(np.corrcoef(cleaned, index)[0, 1]) < 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scipy_residual = signal.detrend(np.stack([series, other]), axis=-1, type="linear")
    assert residual == pytest.approx(scipy_residual, abs=1e-10)
    constant = detrend(eeg, type="constant")
    assert constant.data.mean(axis=-1) == pytest.approx([0.0, 0.0], abs=1e-12)
    epoched = EEG(np.stack([eeg.data, eeg.data + 3.0]), sfreq, eeg.ch_names, units="uV")
    epoched_residual = detrend(epoched).data
    assert epoched_residual.shape == (2, 2, series.size)
    assert epoched_residual[1] == pytest.approx(epoched_residual[0], abs=1e-12)


def test_windows_of_epochs_keep_the_parent_event_code():
    sfreq = 100.0
    epoch = sine(10.0, sfreq, 4.0)
    data = np.stack([epoch, 0.5 * epoch])[:, None, :]
    eeg = EEG(
        data,
        sfreq,
        ("Cz",),
        units="uV",
        metadata={"event_codes": (10, 20), "event_times": (0.0, 1.5)},
    )
    windows = window(eeg, 2.0)
    assert windows.metadata["parent_epoch"] == [0, 0, 1, 1]
    assert windows.metadata["event_codes"] == (10, 10, 20, 20)
    assert windows.metadata["event_times"] == (0.0, 0.0, 1.5, 1.5)
    from eegmath.features import extract_features

    table = extract_features(windows, ["rms"])
    assert list(table["event_code"]) == [10, 10, 20, 20]
    assert list(table["onset"]) == pytest.approx([0.0, 2.0, 0.0, 2.0])


def test_zscore_drops_the_voltage_unit_and_blocks_a_false_rescaling():
    sfreq = 200.0
    raw = EEG(sine(10.0, sfreq, 4.0)[None, :] * 50.0, sfreq, ("Cz",), units="uV")
    scaled = zscore(raw)
    assert scaled.units == "z"
    assert np.std(scaled.data) == pytest.approx(1.0)
    assert qc_summary(scaled)["amplitude_scale_hint"] == "dimensionless_zscore"
    with pytest.raises(DataError, match="dimensionless"):
        scaled.to_microvolts()
    with pytest.raises(ParameterError, match="Re-referencing"):
        rereference(scaled)
    with pytest.warns(UserWarning, match="z-scored"):
        filtered = bandpass(scaled, 1.0, 30.0)
    assert filtered.units == "z"
    assert np.std(filtered.data) < 5.0
    from eegmath.preprocess import baseline_correct, detrend, resample

    with pytest.warns(UserWarning, match="z-scored"):
        assert resample(scaled, 100.0).units == "z"
    with pytest.warns(UserWarning, match="z-scored"):
        assert detrend(scaled).units == "z"
    with pytest.warns(UserWarning, match="z-scored"):
        assert baseline_correct(scaled, 0.0, 0.2).units == "z"


def test_constant_series_does_not_crash_the_unit_root_test():
    from eegmath.signal import stationarity_table

    result = adf_stationarity(np.ones(40))
    assert np.isnan(result["p_value"])
    assert result["n_obs"] == 40
    assert "constant" in result["note"]
    eeg = EEG(np.vstack([np.ones(80), np.random.default_rng(0).normal(size=80)]), 100.0, ("flat", "noise"), units="uV")
    table = stationarity_table(eeg)
    assert np.isnan(table.loc[table["channel"] == "flat", "p_value"]).all()
    assert np.isfinite(table.loc[table["channel"] == "noise", "p_value"]).all()


def test_autoregressive_autocorrelation_follows_the_coefficient():
    from eegmath.signal import autocorrelation, partial_autocorrelation

    rng = np.random.default_rng(0)
    n = 20000
    phi = 0.7
    noise = rng.normal(size=n + 500)
    series = np.zeros(n + 500)
    for t in range(1, series.size):
        series[t] = phi * series[t - 1] + noise[t]
    series = series[500:]
    direct = []
    centered = series - series.mean()
    c0 = float(np.dot(centered, centered) / series.size)
    for lag in range(0, 4):
        if lag == 0:
            direct.append(1.0)
        else:
            direct.append(float(np.dot(centered[:-lag], centered[lag:]) / series.size / c0))
    acf = autocorrelation(series, n_lags=3)
    assert acf["acf"] == pytest.approx(direct, abs=1e-12)
    assert acf["acf"][1] == pytest.approx(phi, abs=0.02)
    assert acf["acf"][2] == pytest.approx(phi**2, abs=0.03)
    pacf = partial_autocorrelation(series, n_lags=3)
    assert pacf["pacf"][1] == pytest.approx(phi, abs=0.02)
    assert abs(pacf["pacf"][2]) < 0.05
    assert abs(pacf["pacf"][3]) < 0.05
    missing = autocorrelation(np.ones(20), n_lags=4)
    assert np.isnan(missing["acf"]).all()
    with pytest.raises(ParameterError, match="one-dimensional"):
        autocorrelation(np.ones((2, 20)))


def test_zscored_burst_suppression_ratio_is_missing():
    from eegmath.features.extract import extract_features

    scaled = zscore(EEG(sine(10.0, 200.0, 4.0)[None, :], 200.0, ("Cz",), units="uV"))
    with pytest.warns(UserWarning, match="z-scoring"):
        table = extract_features(scaled, ["bsr"])
    assert np.isnan(table["burst_suppression_ratio"].iloc[0])


def test_forest_and_timeseries_accept_missing_limits_and_flat_traces():
    import matplotlib.pyplot as plt

    from eegmath.viz import plot_forest

    forest = pd.DataFrame({"feature": ["alpha"], "effect": [1.0], "ci_low": [2.0], "ci_high": [np.nan]})
    fig, _ax = plot_forest(forest)
    plt.close(fig)
    flat = EEG(np.full((1, 20), np.nan), 100.0, ("Cz",), units="uV")
    fig, _ax = plot_timeseries(flat, duration=None)
    plt.close(fig)


def test_annotations_are_copied_and_windows_keep_their_onsets():
    from eegmath.features.extract import extract_features

    annotations = pd.DataFrame({"onset": [1.0], "duration": [0.0], "description": ["mark"]})
    eeg = EEG(sine(10.0, 100.0, 8.0)[None, :], 100.0, ("Cz",), units="uV", annotations=annotations)
    annotations.loc[0, "onset"] = 99.0
    assert eeg.annotations.loc[0, "onset"] == pytest.approx(1.0)
    windowed = window(eeg, 2.0)
    table = extract_features(windowed, ["rms"])
    assert table["onset"].tolist() == pytest.approx(windowed.metadata["window_onsets"])
    assert table["scope"].eq("epoch").all()


def test_epoch_event_codes_are_kept_from_mne():
    from eegmath.features.extract import extract_features
    from eegmath.io.load import from_mne

    info = mne.create_info(["Cz"], 100.0, ch_types="eeg", verbose=False)
    events = np.array([[10, 0, 1], [40, 0, 2], [70, 0, 1]])
    epochs = mne.EpochsArray(np.zeros((3, 1, 20)), info, events=events, tmin=-0.1, verbose=False)
    eeg = from_mne(epochs)
    assert eeg.metadata["event_codes"] == (1, 2, 1)
    assert eeg.metadata["event_times"] == pytest.approx((0.1, 0.4, 0.7))
    table = extract_features(eeg, ["rms"])
    assert table["event_code"].tolist() == [1, 2, 1]


def test_channel_positions_survive_loading_and_channel_selection():
    from eegmath.io.load import from_mne

    rng = np.random.default_rng(0)
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(rng.normal(scale=1e-6, size=(3, 200)), info, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    loaded = from_mne(raw)
    assert loaded.describe()["n_positioned_channels"] == 3
    cz = loaded.pick_channels(["Cz", "Pz"])
    assert cz.ch_names == ("Cz", "Pz")
    assert np.allclose(cz.ch_pos[0], loaded.ch_pos[1])
    dropped = loaded.drop_channels(["Fz"])
    assert dropped.ch_names == ("Cz", "Pz")
    assert np.allclose(dropped.ch_pos[0], loaded.ch_pos[1])
    restored = to_mne(loaded)
    for index in range(3):
        assert np.allclose(restored.info["chs"][index]["loc"][:3], raw.info["chs"][index]["loc"][:3], atol=1e-6)
    bare = EEG(np.zeros((1, 8)), 100.0, ("Cz",), units="uV")
    assert bare.ch_pos is None
    assert from_mne(mne.io.RawArray(np.zeros((1, 8)), mne.create_info(["Cz"], 100.0, verbose=False), verbose=False)).ch_pos is None
    mixed = mne.io.RawArray(rng.normal(scale=1e-6, size=(2, 50)), mne.create_info(["Cz", "Pz"], 100.0, ch_types="eeg", verbose=False), verbose=False)
    mixed.set_montage("standard_1020", verbose=False)
    mixed.info["chs"][1]["loc"][:] = 0.0
    partial = from_mne(mixed)
    assert np.isfinite(partial.ch_pos[0]).all()
    assert not np.isfinite(partial.ch_pos[1]).all()
    assert partial.describe()["n_positioned_channels"] == 1
    again = from_mne(to_mne(partial))
    assert np.allclose(again.ch_pos[0], partial.ch_pos[0], atol=1e-6)
    assert not np.isfinite(again.ch_pos[1]).all()


def test_average_reference_refuses_a_single_eeg_channel():
    with pytest.raises(ParameterError, match="at least two EEG"):
        rereference(EEG(np.ones((1, 50)), 100.0, ("Cz",), units="V"))
    mixed = EEG(
        np.vstack([np.ones(50), np.zeros(50)]),
        100.0,
        ("Cz", "ECG"),
        units="V",
        ch_types=("eeg", "ecg"),
    )
    with pytest.raises(ParameterError, match="at least two EEG"):
        rereference(mixed)


def test_kpss_null_is_stationarity_and_a_random_walk_rejects_it():
    rng = np.random.default_rng(0)
    white = rng.normal(size=800)
    walk = np.cumsum(white)
    level = kpss_stationarity(white)
    trend = kpss_stationarity(walk)
    assert level["null"] == "level stationarity"
    assert level["p_value"] > 0.05
    assert trend["p_value"] <= 0.05
    assert trend["null"] == "level stationarity"
    constant = kpss_stationarity(np.ones(40))
    assert np.isnan(constant["p_value"])
    assert "constant" in constant["note"]
    with pytest.raises(ParameterError):
        kpss_stationarity(white, regression="trend")
