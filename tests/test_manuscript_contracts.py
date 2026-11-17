"""Contracts stated in the methodological manuscript.

These checks do not read the public archives. They lock the identities
the paper uses: catalog size, the default feature request, ratio
rebuild, the experimental unit, and the paired sign concordance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegmath import aggregate_epochs, extract_features, list_features
from eegmath.errors import DesignError
from eegmath.features.regions import aggregate_regions
from eegmath.recording import EEG
from eegmath.stats import compare_groups, prediction_probability


def test_catalog_count_and_spectral_edge_alias():
    catalog = list_features()
    assert len(catalog) == 98
    assert set(catalog["family"]) == {
        "time",
        "spectral",
        "entropy",
        "complexity",
        "dynamics",
        "connectivity",
        "variability",
    }
    assert (catalog["family"] == "spectral").sum() == 37
    assert (catalog["family"] == "connectivity").sum() == 36
    assert "burst_suppression_ratio" in set(catalog["feature"])
    assert "envelope_lzc" in set(catalog["feature"])


def test_default_extract_features_is_the_spectral_family():
    time = np.arange(0, 4.0, 1.0 / 250.0)
    data = np.sin(2 * np.pi * 10.0 * time)[None, :]
    table = extract_features(EEG(data, 250.0, ["Oz"]))
    spectral = set(list_features().loc[list_features()["family"] == "spectral", "feature"])
    assert set(table.columns) >= spectral
    assert "lzc" not in table.columns
    assert table["alpha_relative"].iloc[0] == pytest.approx(1.0)
    assert table["median_freq"].iloc[0] == pytest.approx(table["sef50"].iloc[0])


def test_region_aggregation_rebuilds_relative_power():
    frame = pd.DataFrame(
        {
            "channel": ["A", "B"],
            "alpha_power": [9.0, 1.0],
            "delta_power": [1.0, 1.0],
            "theta_power": [1.0, 1.0],
            "beta_power": [1.0, 1.0],
            "gamma_power": [1.0, 1.0],
            "alpha_relative": [9.0 / 13.0, 1.0 / 5.0],
        }
    )
    aggregated = aggregate_regions(frame, {"pair": ["A", "B"]})
    assert aggregated["alpha_relative"].iloc[0] == pytest.approx(5.0 / 9.0)
    assert frame["alpha_relative"].mean() == pytest.approx(0.4461538461538461)


def test_compare_groups_refuses_repeated_rows_and_channel_mixtures():
    repeated = pd.DataFrame(
        {
            "subject": ["s0", "s0", "s1", "s1"],
            "group": ["a", "a", "b", "b"],
            "y": [0.0, 0.1, 1.0, 1.1],
        }
    )
    with pytest.raises(DesignError):
        compare_groups(repeated, ["y"], "group", "a", "b", subject="subject", unit="row")
    channels = repeated.assign(channel=["Cz", "Oz", "Cz", "Oz"])
    with pytest.raises(DesignError):
        compare_groups(channels, ["y"], "group", "a", "b", subject="subject", unit="subject")


def test_paired_sign_concordance_is_not_the_wilcoxon_test():
    rows = []
    for index, difference in enumerate([0.1] * 8 + [-5.0]):
        rows.append({"subject": f"s{index}", "condition": "low", "y": 0.0})
        rows.append({"subject": f"s{index}", "condition": "high", "y": difference})
    frame = pd.DataFrame(rows)
    paired = prediction_probability(frame, "y", "condition", "subject", "low", "high")
    ranked = compare_groups(
        frame, ["y"], "condition", "low", "high", subject="subject", test="wilcoxon", unit="subject"
    )
    assert paired.strategy == "paired"
    assert paired.pk == pytest.approx(8.0 / 9.0)
    assert paired.p_value == pytest.approx(0.0390625)
    assert float(ranked["p_value"].iloc[0]) == pytest.approx(0.1484375)


def test_epoch_aggregation_is_exported():
    assert callable(aggregate_epochs)
