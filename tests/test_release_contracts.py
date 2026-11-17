"""Release contracts that a user can trip over without a bad formula.

These checks lock the behaviour added so that a missing label, a ratio
that cannot be rebuilt, a short connectivity estimate, and a rank-deficient
unit-root fit are not reported as ordinary results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mne

from eegmath import extract_features, list_features
from eegmath.errors import DataError, DesignError, ParameterError
from eegmath.features import FeatureConfig
from eegmath.features.regions import aggregate_regions
from eegmath.io import load
from eegmath.preprocess import rereference
from eegmath.recording import EEG
from eegmath.signal import adf_stationarity, morlet_power
from eegmath.stats import compare_groups
from eegmath.workflow import feature_table
from synthetic import sine


def _paired(condition_a: str = "task") -> pd.DataFrame:
    rows = []
    for index in range(4):
        rows.append({"subject": f"s{index}", "condition": "rest", "alpha_relative": 0.5})
        rows.append({"subject": f"s{index}", "condition": condition_a, "alpha_relative": 0.2})
    return pd.DataFrame(rows)


def test_absent_group_label_is_not_a_negative_result():
    with pytest.raises(DesignError, match="no rows labelled"):
        compare_groups(
            _paired(),
            ["alpha_relative"],
            "condition",
            "Task",
            "rest",
            subject="subject",
            test="wilcoxon",
        )


def test_one_pair_does_not_return_a_completed_wilcoxon():
    frame = _paired().loc[lambda table: table["subject"] == "s0"]
    with pytest.raises(DesignError, match="at least two pairs"):
        compare_groups(
            frame,
            ["alpha_relative"],
            "condition",
            "task",
            "rest",
            subject="subject",
            test="wilcoxon",
        )


def test_region_aggregation_warns_when_absolute_power_is_absent():
    frame = pd.DataFrame(
        {
            "recording_id": ["a", "a"],
            "channel": ["O1", "O2"],
            "alpha_relative": [0.4, 0.6],
            "rms": [10.0, 12.0],
        }
    )
    with pytest.warns(UserWarning, match="absolute"):
        out = aggregate_regions(frame, {"occipital": ["O1", "O2"]})
    assert np.isnan(out.loc[0, "alpha_relative"])
    assert out.loc[0, "rms"] == pytest.approx(11.0)


def test_pure_sine_does_not_report_a_unit_root_pvalue():
    result = adf_stationarity(sine(10.0, 250.0, 8.0))
    assert np.isnan(result["p_value"])
    assert "rank" in result["note"].lower() or "not computed" in result["note"].lower()


def test_empty_table_is_not_an_absent_label():
    frame = _paired().iloc[0:0]
    with pytest.raises(DesignError, match="table has no rows"):
        compare_groups(
            frame,
            ["alpha_relative"],
            "condition",
            "task",
            "rest",
            subject="subject",
            test="wilcoxon",
        )


def test_region_aggregation_names_a_channel_that_is_absent():
    frame = pd.DataFrame(
        {
            "channel": ["O1", "O2"],
            "alpha_power": [2.0, 4.0],
            "delta_power": [1.0, 1.0],
            "theta_power": [1.0, 1.0],
            "beta_power": [1.0, 1.0],
            "gamma_power": [1.0, 1.0],
        }
    )
    with pytest.warns(UserWarning, match="Oz"):
        out = aggregate_regions(frame, {"occipital": ["O1", "O2", "Oz"]})
    assert out.loc[0, "n_channels"] == 2
    assert out.loc[0, "alpha_power"] == pytest.approx(3.0)


def test_feature_table_keeps_bids_selection_on_the_loader(tmp_path):
    eeg = EEG(np.ones((1, 32)), 32.0, ("Cz",), units="uV")
    with pytest.raises(ParameterError, match="dataset_description.json"):
        feature_table(eeg, ["rms"], subjects=["01"])
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text('{"Name": "empty"}', encoding="utf-8")
    with pytest.raises(DataError, match="No BIDS EEG"):
        feature_table(root, ["rms"], subjects=["01"])


def test_morlet_power_is_not_a_welch_band_power():
    sfreq = 250.0
    times = np.arange(0.0, 2.0, 1.0 / sfreq)
    wave = 10.0 * np.sin(2 * np.pi * 10.0 * times)
    eeg = EEG(wave[None, :], sfreq, ("Cz",), units="uV")
    power, _times = morlet_power(eeg, np.array([10.0]), n_cycles=3.0)
    midpoint = float(power[0, 0, 0, power.shape[-1] // 2])
    assert midpoint > 500.0
    with pytest.raises(ParameterError, match="longer than the signal"):
        morlet_power(eeg, np.array([4.0]), n_cycles=7.0)


def test_fetch_script_reads_eegmath_data(tmp_path, monkeypatch):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_validation_data.py"
    spec = importlib.util.spec_from_file_location("fetch_validation_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setenv("EEGMATH_DATA", str(tmp_path))
    assert module.data_root() == tmp_path


def test_bids_selection_rejects_a_bare_string(tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text('{"Name": "empty"}', encoding="utf-8")
    with pytest.raises(ParameterError, match="sequence of labels"):
        load(root, subjects="01")
    with pytest.raises(ParameterError, match="sequence of labels"):
        feature_table(root, ["rms"], tasks="rest")
    with pytest.raises(ParameterError, match="sequence of labels"):
        feature_table(root, ["rms"], sessions="01")


def test_rejected_test_warns_when_the_mean_interval_includes_zero():
    rows = []
    for index, shift in enumerate([-0.001] * 7 + [-100.0]):
        rows.append({"subject": f"s{index}", "condition": "rest", "alpha": 0.0})
        rows.append({"subject": f"s{index}", "condition": "task", "alpha": float(shift)})
    with pytest.warns(UserWarning, match="interval includes 0"):
        result = compare_groups(
            pd.DataFrame(rows),
            ["alpha"],
            "condition",
            "task",
            "rest",
            subject="subject",
            test="wilcoxon",
        )
    row = result.iloc[0]
    assert bool(row["significant"])
    assert row["ci_low"] < 0 < row["ci_high"]
    assert row["p_value"] < 0.05


def test_uncorrected_interval_warns_when_the_flag_stays_false():
    rows = []
    for index, shift in enumerate([-0.20, -0.18, -0.22, -0.19, -0.21, -0.17]):
        rows.append({"subject": f"s{index}", "condition": "rest", "alpha": 0.4, "rms": 0.0})
        rows.append({"subject": f"s{index}", "condition": "task", "alpha": 0.4 + shift, "rms": 1.0 if index % 2 == 0 else -1.0})
    with pytest.warns(UserWarning, match="interval excludes 0"):
        result = compare_groups(
            pd.DataFrame(rows),
            ["alpha", "rms"],
            "condition",
            "task",
            "rest",
            subject="subject",
            test="wilcoxon",
        )
    alpha = result.loc[result["feature"] == "alpha"].iloc[0]
    assert bool(alpha["significant"]) is False
    assert alpha["ci_high"] < 0
    assert alpha["p_value"] < 0.05
    assert alpha["p_corrected"] > 0.05
    alone = compare_groups(
        pd.DataFrame(rows).drop(columns="rms"),
        ["alpha"],
        "condition",
        "task",
        "rest",
        subject="subject",
        test="wilcoxon",
    )
    assert bool(alone.loc[0, "significant"]) is True
    assert alone.loc[0, "ci_low"] == pytest.approx(alpha["ci_low"])
    assert alone.loc[0, "ci_high"] == pytest.approx(alpha["ci_high"])


def test_zero_paired_differences_do_not_surface_the_scipy_warning():
    rows = []
    for index in range(8):
        rows.append({"subject": f"s{index}", "condition": "rest", "alpha": 1.0})
        rows.append({"subject": f"s{index}", "condition": "task", "alpha": 1.0})
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compare_groups(
            pd.DataFrame(rows),
            ["alpha"],
            "condition",
            "task",
            "rest",
            subject="subject",
            test="wilcoxon",
        )
    assert not any(item.category is RuntimeWarning for item in caught)
    assert any("zero width" in str(item.message) for item in caught)
    row = result.iloc[0]
    assert row["ci_low"] == pytest.approx(0.0)
    assert row["ci_high"] == pytest.approx(0.0)
    assert bool(row["significant"]) is False
    assert row["p_value"] == pytest.approx(1.0)
    assert np.isnan(row["effect"])


def test_feature_rows_keep_duration_and_header_limits():
    times = np.arange(0.0, 4.0, 0.01)
    wave = np.sin(2 * np.pi * 10.0 * times)
    eeg = EEG(
        np.vstack([wave, wave]),
        100.0,
        ("O1", "O2"),
        units="uV",
        metadata={"highpass": 0.5, "lowpass": 40.0},
        subject_id="s01",
        condition="rest",
    )
    table = extract_features(eeg, ["rms", "alpha_relative"])
    assert table["duration_sec"].eq(4.0).all()
    assert table["highpass_hz"].eq(0.5).all()
    assert table["lowpass_hz"].eq(40.0).all()
    assert set(table["units"]) == {"uV"}
    bare = extract_features(EEG(wave[None, :], 100.0, ("Cz",), units="uV"), ["rms"])
    assert np.isnan(bare.loc[0, "highpass_hz"])
    assert np.isnan(bare.loc[0, "lowpass_hz"])
    identity = {
        "recording_id",
        "subject_id",
        "session_id",
        "condition",
        "channel",
        "sfreq",
        "units",
        "reference",
        "duration_sec",
        "highpass_hz",
        "lowpass_hz",
        "rms",
    }
    regional = aggregate_regions(table.loc[:, [column for column in table.columns if column in identity]], {"occipital": ["O1", "O2"]})
    assert regional.loc[0, "highpass_hz"] == pytest.approx(0.5)
    assert regional.loc[0, "n_channels"] == 2
    catalog = list_features().set_index("feature")["quantity"]
    assert catalog["alpha_relative"] == "proportion"
    assert catalog["rms"] == "recording unit"
    assert catalog["alpha_power"] == "recording unit squared"


def test_relative_power_warns_when_a_named_band_remains_outside_freq_max():
    times = np.arange(0.0, 8.0, 1.0 / 250.0)
    eeg = EEG(np.sin(2 * np.pi * 10.0 * times)[None, :], 250.0, ("Cz",), units="uV")
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default = extract_features(eeg, ["alpha_relative"])
    assert not any("relative-power denominator" in str(item.message) for item in caught)
    assert 0 < float(default["alpha_relative"].iloc[0]) <= 1
    with pytest.warns(UserWarning, match="relative-power denominator"):
        limited = extract_features(eeg, ["alpha_relative"], config=FeatureConfig(freq_max=20))
    assert limited["alpha_relative"].iloc[0] == pytest.approx(default["alpha_relative"].iloc[0])


def test_two_channel_average_reference_warns():
    eeg = EEG(np.vstack([sine(10.0, 100.0, 2.0), sine(6.0, 100.0, 2.0)]), 100.0, ("Oz", "Fz"), units="uV")
    with pytest.warns(UserWarning, match="exactly two EEG channels"):
        referenced = rereference(eeg, "average")
    assert referenced.data.shape == eeg.data.shape
    three = EEG(np.ones((3, 200)), 100.0, ("A", "B", "C"), units="uV")
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rereference(three, "average")
    assert not any("exactly two EEG channels" in str(item.message) for item in caught)


def test_bids_miss_names_the_labels_that_exist(tmp_path):
    root = tmp_path / "bids"
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "Synthetic", "BIDSVersion": "1.8.0"}\n')
    info = mne.create_info(["Cz"], 100.0, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(sine(10.0, 100.0, 1.0)[None, :] * 1e-6, info, verbose=False)
    raw.save(eeg_dir / "sub-01_task-rest_eeg.fif", overwrite=True, verbose=False)
    with pytest.raises(ParameterError, match=r"tasks.*\['rest'\]"):
        load(root, tasks="rest")
    with pytest.raises(DataError, match="rest") as raised:
        load(root, tasks=["task-rest"])
    message = str(raised.value)
    assert "task-rest" in message
    assert "Task labels include rest" in message


def test_synthetic_paired_example_rejects_task_alpha():
    from paired_alpha import main

    result = main()
    alpha = result.loc[result["feature"] == "alpha_relative"].iloc[0]
    assert alpha["n_a"] == 8
    assert alpha["mean_difference_a_minus_b"] < 0
    assert alpha["p_value"] < 0.05
