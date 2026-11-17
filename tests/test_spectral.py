"""Spectra, connectivity, and the feature table contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegmath.errors import ParameterError
from eegmath.connectivity.metrics import (
    clustering_coefficient,
    connectivity_matrices,
    node_strength,
    region_connectivity,
)
from eegmath.features.extract import extract_features, list_features, resolve_features
from eegmath.features.regions import aggregate_regions
from eegmath.recording import EEG
from eegmath.spectral.psd import (
    aperiodic_fit,
    spectral_edge_frequency,
    spectral_feature_set,
    welch_psd,
)
from synthetic import sine


def test_relative_band_powers_sum_to_one_and_locate_a_10hz_peak():
    sfreq = 250.0
    data = np.vstack([sine(10.0, sfreq, 10.0), sine(10.0, sfreq, 10.0)])
    features, _meta = spectral_feature_set(data, sfreq)
    relative = np.vstack(
        [features[f"{band}_relative"] for band in ("delta", "theta", "alpha", "beta", "gamma")]
    )
    assert relative.sum(axis=0) == pytest.approx(1.0, abs=1e-6)
    assert features["alpha_relative"][0] > 0.8
    assert features["alpha_peak_freq"][0] == pytest.approx(10.0, abs=0.5)
    assert features["median_freq"][0] == pytest.approx(features["sef50"][0])
    assert features["sef50"][0] <= features["sef75"][0] <= features["sef90"][0] <= features["sef95"][0]


def test_spectral_edge_of_a_flat_density_is_the_midpoint():
    freqs = np.arange(0.5, 45.0 + 1e-9, 0.5)
    psd = np.ones_like(freqs)
    sef50 = spectral_edge_frequency(freqs, psd, 50, fmin=0.5, fmax=45.0)
    sef90 = spectral_edge_frequency(freqs, psd, 90, fmin=0.5, fmax=45.0)
    assert float(sef50) == pytest.approx((0.5 + 45.0) / 2, abs=0.5)
    assert float(sef50) < float(sef90)


def test_aperiodic_line_recovers_an_inverse_square_spectrum():
    freqs = np.linspace(2.0, 40.0, 120)
    psd = 10 ** (0.5 - 2.0 * np.log10(freqs))
    peaked = psd.copy()
    peaked[40] *= 30
    exponent, offset = aperiodic_fit(peaked, freqs) if False else aperiodic_fit(freqs, peaked)
    assert float(exponent) == pytest.approx(2.0, abs=0.15)
    assert float(offset) == pytest.approx(0.5, abs=0.15)


def test_welch_does_not_concatenate_epochs():
    sfreq = 100.0
    quiet = 1e-3 * sine(10.0, sfreq, 4.0)
    loud = sine(10.0, sfreq, 4.0)
    data = np.stack([quiet[None, :], loud[None, :]])
    _freqs, psd = welch_psd(data, sfreq)
    assert psd.shape[0] == 2
    assert psd[1].sum() > 100 * psd[0].sum()


def test_identical_channels_are_coherent_and_have_no_phase_lag():
    sfreq = 200.0
    channel = sine(10.0, sfreq, 20.0)
    data = np.vstack([channel, channel, channel])
    matrices, meta = connectivity_matrices(
        data, sfreq, metrics=("coherence", "pli", "plv", "imcoh"), min_segments=5, segment_sec=1.0
    )
    assert meta["sufficient_segments"] is True
    alpha = matrices["coherence"]["alpha"]
    assert alpha[0, 1] > 0.95
    assert matrices["pli"]["alpha"][0, 1] < 0.15
    assert matrices["plv"]["alpha"][0, 1] > 0.95
    assert abs(matrices["imcoh"]["alpha"][0, 1]) < 0.15


def test_quadrature_pair_has_a_phase_lag_index_near_one():
    sfreq = 200.0
    data = np.vstack(
        [
            sine(10.0, sfreq, 20.0),
            sine(10.0, sfreq, 20.0, phase=np.pi / 2),
        ]
    )
    matrices, _meta = connectivity_matrices(
        data, sfreq, metrics=("pli", "wpli", "imcoh", "plv"), min_segments=5, segment_sec=1.0
    )
    assert matrices["pli"]["alpha"][0, 1] > 0.8
    assert matrices["wpli"]["alpha"][0, 1] > 0.8
    assert abs(matrices["imcoh"]["alpha"][0, 1]) > 0.5
    assert matrices["plv"]["alpha"][0, 1] > 0.8


def test_positive_imaginary_coherency_means_the_row_channel_leads():
    sfreq = 200.0
    times = np.arange(int(sfreq * 20.0)) / sfreq
    lead = np.sin(2 * np.pi * 10.0 * times)
    lag = np.sin(2 * np.pi * 10.0 * (times - 0.01))
    matrices, _meta = connectivity_matrices(
        np.vstack([lead, lag]),
        sfreq,
        metrics=("imcoh",),
        min_segments=5,
        segment_sec=2.0,
    )
    assert matrices["imcoh"]["alpha"][0, 1] > 0.3
    assert matrices["imcoh"]["alpha"][1, 0] < -0.3


def test_debiased_wpli_keeps_a_negative_estimate_and_one_segment_is_missing():
    sfreq = 200.0
    times = np.arange(int(sfreq * 4.0)) / sfreq
    lead = np.sin(2 * np.pi * 10.0 * times)
    positive = np.sin(2 * np.pi * 10.0 * times - 1.0)
    negative = np.sin(2 * np.pi * 10.0 * times + 1.0)
    paired = np.stack([np.vstack([lead, positive]), np.vstack([lead, negative])])
    matrices, meta = connectivity_matrices(
        paired, sfreq, metrics=("wpli", "pli"), min_segments=1, segment_sec=4.0
    )
    assert meta["n_segments"] == 2
    # Equal and opposite lags: the pairwise product is -1, not a clipped 0.
    assert matrices["wpli"]["alpha"][0, 1] < -0.5
    assert abs(matrices["pli"]["alpha"][0, 1]) < 0.2
    single = np.vstack([lead[: int(sfreq * 2.0)], positive[: int(sfreq * 2.0)]])
    one_matrix, one = connectivity_matrices(
        single, sfreq, metrics=("wpli", "coherence"), min_segments=1, segment_sec=2.0
    )
    assert one["n_segments"] == 1 and one["sufficient_segments"] is True
    assert np.isnan(one_matrix["wpli"]["alpha"]).all()
    assert one_matrix["coherence"]["alpha"][0, 1] > 0.9


def test_too_few_segments_are_withheld_rather_than_reported_as_zero():
    sfreq = 100.0
    data = np.vstack([sine(10.0, sfreq, 3.0), sine(10.0, sfreq, 3.0, phase=1.0)])
    with pytest.warns(UserWarning, match="Fourier segments"):
        matrices, meta = connectivity_matrices(data, sfreq, min_segments=20, segment_sec=2.0)
    assert meta["sufficient_segments"] is False
    assert np.isnan(matrices["coherence"]["alpha"]).all()


def test_clustering_of_a_triangle_is_one():
    weight = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    assert clustering_coefficient(weight) == pytest.approx([1.0, 1.0, 1.0])
    assert node_strength(weight) == pytest.approx([2.0, 2.0, 2.0])


def test_region_connectivity_averages_only_the_requested_block():
    matrix = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.4], [0.8, 0.4, 0.0]])
    value = region_connectivity(matrix, ["A", "B", "C"], ["A"], ["C"])
    assert value == pytest.approx(0.8)
    assert np.isnan(region_connectivity(matrix, ["A", "B", "C"], ["A"], ["missing"]))


def test_feature_catalog_matches_extracted_columns_and_drops_study_regions():
    catalog = list_features()
    assert "frontoparietal" not in " ".join(catalog["feature"])
    assert set(catalog["family"]) >= {"time", "spectral", "entropy", "connectivity"}
    sfreq = 128.0
    data = np.vstack([sine(10.0, sfreq, 4.0), sine(6.0, sfreq, 4.0)])
    eeg = EEG(data, sfreq, ("C3", "C4"), units="uV", subject_id="S", condition="rest")
    spectral = extract_features(eeg, "spectral")
    expected = resolve_features("spectral")
    assert list(expected) == [c for c in spectral.columns if c in set(expected)]
    for name in expected:
        assert name in spectral.columns
    assert spectral["alpha_relative"].iloc[0] > 0.5
    custom = extract_features(eeg, ["rms", "pe", "bsr"])
    assert {"rms", "permutation_entropy", "burst_suppression_ratio"} <= set(custom.columns)
    # A 1 uV sine is below the 5 uV suppression threshold.
    assert custom["burst_suppression_ratio"].iloc[0] > 0.9


def test_epoch_rows_are_not_copies_of_recording_features():
    sfreq = 100.0
    epoch = sine(10.0, sfreq, 4.0)
    data = np.stack([np.vstack([epoch, epoch]), np.vstack([epoch, -epoch])])
    eeg = EEG(data, sfreq, ("A", "B"), units="uV", subject_id="S1", recording_id="R")
    table = extract_features(eeg, ["rms", "rms_cv"], config=None)
    epoch_rows = table.loc[table["scope"] == "epoch"]
    recording_rows = table.loc[table["scope"] == "recording"]
    assert len(epoch_rows) == 4
    assert recording_rows["rms"].isna().all()
    assert epoch_rows["rms_cv"].isna().all()
    assert recording_rows["rms_cv"].notna().all()
    reduced = extract_features(eeg, ["alpha_power", "alpha_relative"], reduce_epochs=True)
    assert reduced["scope"].eq("recording").all()
    assert len(reduced) == 2


def test_partial_relative_request_uses_the_full_partition_after_averaging():
    sfreq = 200.0
    mixed = sine(10.0, sfreq, 8.0) + 0.8 * sine(4.0, sfreq, 8.0)
    data = np.stack([mixed, mixed * 0.5])[:, None, :]
    eeg = EEG(data, sfreq, ("Oz",), units="uV", subject_id="S", condition="rest")
    full = extract_features(eeg, "spectral", reduce_epochs=True)
    partial = extract_features(
        eeg, ["alpha_relative", "theta_alpha_ratio", "alpha_power"], reduce_epochs=True
    )
    assert "delta_power" not in partial.columns
    assert partial.loc[0, "alpha_relative"] == pytest.approx(full.loc[0, "alpha_relative"], rel=1e-6)
    assert partial.loc[0, "theta_alpha_ratio"] == pytest.approx(full.loc[0, "theta_alpha_ratio"], rel=1e-6)
    assert partial.loc[0, "alpha_relative"] < 0.95


def test_recompute_does_not_turn_a_single_retained_band_into_one():
    from eegmath.spectral.psd import recompute_power_ratios

    frame = pd.DataFrame({"alpha_power": [2.0], "alpha_relative": [0.25], "theta_alpha_ratio": [0.5]})
    out = recompute_power_ratios(frame)
    assert out.loc[0, "alpha_relative"] == pytest.approx(0.25)
    assert out.loc[0, "theta_alpha_ratio"] == pytest.approx(0.5)


def test_region_mean_recomputes_ratios_from_power():
    sfreq = 200.0
    data = np.vstack([sine(10.0, sfreq, 6.0), sine(4.0, sfreq, 6.0)])
    eeg = EEG(data, sfreq, ("F3", "F4"), units="uV")
    table = extract_features(eeg, "spectral")
    grouped = aggregate_regions(table, {"frontal": ["F3", "F4"]})
    assert len(grouped) == 1
    power = grouped.iloc[0]
    ratio = power["alpha_power"] / (power["delta_power"] + power["theta_power"] + power["alpha_power"] + power["beta_power"] + power["gamma_power"])
    assert power["alpha_relative"] == pytest.approx(ratio, rel=1e-6)
    assert power["n_channels"] == 2


def test_region_channel_count_follows_the_electrodes_present_in_each_recording():
    frame = pd.DataFrame(
        {
            "recording_id": ["A", "A", "B"],
            "channel": ["O1", "O2", "O1"],
            "alpha_power": [1.0, 3.0, 10.0],
        }
    )
    out = aggregate_regions(frame, {"occipital": ["O1", "O2"]}).set_index("recording_id")
    assert out.loc["A", "n_channels"] == 2
    assert out.loc["B", "n_channels"] == 1
    assert out.loc["A", "alpha_power"] == pytest.approx(2.0)
    assert out.loc["B", "alpha_power"] == pytest.approx(10.0)


def test_coarse_partition_accepts_the_older_temporal_names():
    from eegmath.features.regions import COARSE_10_20

    frame = pd.DataFrame(
        {
            "channel": ["T3", "T4", "T7", "T5", "P7", "O1"],
            "alpha_power": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        }
    )
    with pytest.warns(UserWarning, match="skipped channel"):
        out = aggregate_regions(frame, COARSE_10_20).set_index("region")
    assert out.loc["temporal", "n_channels"] == 3
    assert out.loc["temporal", "alpha_power"] == pytest.approx((1.0 + 3.0 + 5.0) / 3)
    assert out.loc["parietal", "n_channels"] == 2
    assert out.loc["parietal", "alpha_power"] == pytest.approx(8.0)
    assert "T5" not in COARSE_10_20["temporal"]
    assert "T3" not in COARSE_10_20["parietal"]


def test_custom_bands_do_not_invent_missing_ratios():
    sfreq = 200.0
    eeg = EEG(sine(10.0, sfreq, 6.0)[None, :], sfreq, ("Cz",), units="uV")
    from eegmath.features.extract import FeatureConfig

    config = FeatureConfig(bands={"alpha": (8.0, 13.0), "beta": (13.0, 30.0)})
    names = resolve_features("spectral", config)
    assert "delta_alpha_ratio" not in names
    assert "beta_alpha_ratio" in names
    table = extract_features(eeg, "spectral", config=config)
    assert "beta_alpha_ratio" in table.columns
    assert table["alpha_relative"].iloc[0] + table["beta_relative"].iloc[0] == pytest.approx(1.0, abs=1e-5)


def test_constant_and_very_short_signals_stay_defined():
    eeg = EEG(np.ones((2, 64)), 64.0, ("A", "B"), units="uV")
    with pytest.warns(UserWarning, match="relative-power denominator"):
        table = extract_features(eeg, ["rms", "spectral"])
    assert np.isfinite(table["rms"]).all()
    # Eight samples can still produce a spectrum. Fewer than four cannot.
    short = EEG(np.ones((1, 8)), 32.0, ("A",), units="V")
    short_table = extract_features(short, ["rms"])
    assert short_table["rms"].iloc[0] == pytest.approx(1.0)
    with pytest.raises(Exception):
        extract_features(EEG(np.ones((1, 3)), 32.0, ("A",), units="V"), "spectral")


def test_spectral_edge_interpolates_inside_the_first_bin():
    freqs = np.arange(0.5, 45.0 + 1e-9, 0.5)
    psd = np.ones_like(freqs)
    edge = spectral_edge_frequency(freqs, psd, 1)
    assert float(edge) == pytest.approx(0.5 + 0.01 * (45.0 - 0.5), abs=1e-6)
    assert float(spectral_edge_frequency(freqs, psd, 50)) == pytest.approx(0.5 + 0.5 * (45.0 - 0.5), abs=1e-6)
    multi = spectral_edge_frequency(freqs, np.ones((2, freqs.size)), 1)
    assert multi.shape == (2,)
    assert multi[0] == pytest.approx(multi[1])


def test_aggregate_epochs_keeps_recording_variability_without_an_id():
    from eegmath.features.extract import aggregate_epochs

    sfreq = 200.0
    tone = sine(10.0, sfreq, 4.0)
    data = np.stack([tone, 2.0 * tone])[:, None, :]
    eeg = EEG(data, sfreq, ("Cz",), units="uV", subject_id="S", condition="rest")
    table = extract_features(eeg, ["rms", "rms_cv"])
    assert table["recording_id"].isna().all()
    reduced = aggregate_epochs(table)
    assert len(reduced) == 1
    epoch_rms = table.loc[table["scope"] == "epoch", "rms"]
    assert reduced["rms"].iloc[0] == pytest.approx(epoch_rms.mean())
    assert reduced["rms_cv"].iloc[0] == pytest.approx(
        table.loc[table["scope"] == "recording", "rms_cv"].iloc[0]
    )
    assert reduced["rms_cv"].iloc[0] == pytest.approx(float(np.sqrt(2) / 3), rel=1e-3)


def test_mse_scales_name_the_columns_that_are_computed():
    from eegmath.features.extract import FeatureConfig

    config = FeatureConfig(mse_scales=(1, 2))
    names = resolve_features("entropy", config)
    assert "mse_scale2" in names
    assert "mse_scale5" not in names
    assert resolve_features(["mse_scale2"], config) == ("mse_scale2",)
    rng = np.random.default_rng(0)
    eeg = EEG(rng.normal(size=(1, 400)), 100.0, ("Cz",), units="uV")
    table = extract_features(eeg, ["mse_scale2"], config=config)
    assert np.isfinite(table["mse_scale2"]).all()


def test_mse_fraction_and_mean_keep_every_configured_scale():
    from eegmath.features.extract import FeatureConfig

    rng = np.random.default_rng(0)
    eeg = EEG(rng.normal(size=(1, 400)), 100.0, ("Cz",), units="uV")
    wide = extract_features(eeg, ["mse_scale1"], config=FeatureConfig(mse_r_fraction=0.2))
    tight = extract_features(eeg, ["mse_scale1"], config=FeatureConfig(mse_r_fraction=0.15))
    assert np.isfinite(wide["mse_scale1"]).all()
    assert wide["mse_scale1"].iloc[0] != pytest.approx(tight["mse_scale1"].iloc[0])
    # Scale 10 of an 80-sample series has no m+1 matches, so sample entropy
    # is infinite. The mean must not fall back to the finite scales.
    short = EEG(rng.normal(size=(1, 80)), 100.0, ("Cz",), units="uV")
    table = extract_features(short, ["mse_mean", "mse_scale1", "mse_scale10"])
    assert np.isfinite(table["mse_scale1"].iloc[0])
    assert np.isposinf(table["mse_scale10"].iloc[0])
    assert np.isposinf(table["mse_mean"].iloc[0])


def test_pairwise_phase_consistency_matches_the_pairwise_cosine():
    from scipy import signal

    from eegmath.connectivity.metrics import _accumulate

    sfreq = 200.0
    times = np.arange(int(sfreq * 20.0)) / sfreq
    lead = np.sin(2 * np.pi * 10.0 * times)
    lag = np.sin(2 * np.pi * 10.0 * times + 0.8)
    data = np.vstack([lead, lag])
    n_per = int(sfreq)
    freqs, cubes, n_used = _accumulate(data, sfreq, n_per, 0.5, ("plv", "ppc"), "constant")
    k = int(np.argmin(np.abs(freqs - 10.0)))
    plv = float(cubes["plv"][0, 1, k])
    ppc = float(cubes["ppc"][0, 1, k])
    assert ppc == pytest.approx((n_used * plv**2 - 1.0) / (n_used - 1.0))
    assert ppc > 0.95
    # Independent reconstruction of Vinck et al. 2010, equation 14.
    window = signal.windows.hann(n_per, sym=False)
    phases = []
    step = n_per // 2
    for start in range(0, lead.size - n_per + 1, step):
        left = lead[start : start + n_per]
        right = lag[start : start + n_per]
        left = left - left.mean()
        right = right - right.mean()
        left_f = np.fft.rfft(left * window)
        right_f = np.fft.rfft(right * window)
        phases.append(np.angle(left_f[k] * np.conj(right_f[k])))
    phases = np.asarray(phases)
    pairs = []
    for i in range(phases.size):
        for j in range(i + 1, phases.size):
            pairs.append(np.cos(phases[i] - phases[j]))
    assert ppc == pytest.approx(float(np.mean(pairs)), abs=1e-10)
    matrices, _meta = connectivity_matrices(
        data, sfreq, metrics=("ppc",), min_segments=5, segment_sec=1.0
    )
    assert matrices["ppc"]["alpha"][0, 1] > 0.9
    # Two opposite phase differences cancel in the phase-locking value
    # and give pairwise phase consistency of -1. The negative value is kept.
    tone = np.sin(2 * np.pi * 10.0 * times[: int(sfreq)])
    opposite = np.stack([np.vstack([tone, tone]), np.vstack([tone, -tone])])
    _freqs, opposite_cubes, n_opposite = _accumulate(
        opposite, sfreq, int(sfreq), 0.0, ("plv", "ppc"), "constant"
    )
    k_opposite = int(np.argmin(np.abs(_freqs - 10.0)))
    assert n_opposite == 2
    assert opposite_cubes["plv"][0, 1, k_opposite] == pytest.approx(0.0, abs=1e-8)
    assert opposite_cubes["ppc"][0, 1, k_opposite] == pytest.approx(-1.0, abs=1e-8)


def test_zero_power_ratios_are_not_replaced_by_a_huge_finite_number():
    from eegmath.spectral.psd import power_ratios

    out = power_ratios({"delta": np.array([2.0]), "alpha": np.array([0.0])})
    assert np.isinf(out["delta_alpha_ratio"][0])
    assert np.isnan(out["theta_alpha_ratio"][0])
    flat, _meta = spectral_feature_set(np.ones(512), 128.0)
    assert np.isnan(flat["alpha_peak_freq"])
    assert np.isnan(flat["spectral_entropy"])


def test_median_frequency_does_not_depend_on_the_edge_list():
    sfreq = 200.0
    signal = sine(10.0, sfreq, 8.0)
    features, _meta = spectral_feature_set(signal, sfreq, sef_percentiles=(95,))
    assert "sef50" not in features
    assert np.isfinite(features["median_freq"])
    again, _meta = spectral_feature_set(signal, sfreq, sef_percentiles=(50, 95))
    assert float(np.squeeze(features["median_freq"])) == pytest.approx(
        float(np.squeeze(again["median_freq"]))
    )


def test_epoch_average_rebuilds_ratios_from_power_not_from_the_ratios():
    from eegmath.features.extract import FeatureConfig, aggregate_epochs

    sfreq = 200.0
    alpha = sine(10.0, sfreq, 4.0, amplitude=1.0)
    beta = sine(20.0, sfreq, 4.0, amplitude=3.0)
    data = np.stack([alpha, beta])[:, None, :]
    config = FeatureConfig(bands={"alpha": (8.0, 13.0), "beta": (13.0, 30.0)})
    eeg = EEG(data, sfreq, ("Cz",), units="uV", recording_id="r")
    reduced = extract_features(
        eeg, ["alpha_relative", "alpha_power", "beta_power"], config=config, reduce_epochs=True
    )
    separate = extract_features(
        eeg, ["alpha_relative", "alpha_power", "beta_power"], config=config
    )
    mean_of_ratios = separate.loc[separate["scope"] == "epoch", "alpha_relative"].mean()
    rebuilt = reduced["alpha_power"].iloc[0] / (reduced["alpha_power"].iloc[0] + reduced["beta_power"].iloc[0])
    assert reduced["alpha_relative"].iloc[0] == pytest.approx(rebuilt)
    assert abs(reduced["alpha_relative"].iloc[0] - mean_of_ratios) > 0.05
    ratio_only = separate.loc[separate["scope"] == "epoch", ["recording_id", "channel", "epoch", "scope", "alpha_relative"]]
    with pytest.warns(UserWarning, match="missing after aggregation"):
        aggregated = aggregate_epochs(ratio_only, bands=config.resolved_bands())
    assert np.isnan(aggregated["alpha_relative"].iloc[0])


def test_dead_channel_does_not_erase_the_other_nodes():
    from eegmath.connectivity.metrics import connectivity_node_features

    sfreq = 200.0
    tone = sine(10.0, sfreq, 20.0)
    data = np.vstack([tone, tone, np.zeros_like(tone)])
    node, _meta = connectivity_node_features(
        data, sfreq, metrics=("coherence",), min_segments=5, segment_sec=1.0
    )
    assert node["global_coherence"][0] > 0.9
    assert node["global_coherence"][1] > 0.9
    assert np.isnan(node["global_coherence"][2])


def test_imaginary_coherency_node_feature_keeps_opposite_lags():
    from eegmath.connectivity.metrics import connectivity_node_features

    sfreq = 200.0
    lead = sine(10.0, sfreq, 30.0) + sine(20.0, sfreq, 30.0)
    lag = sine(10.0, sfreq, 30.0, phase=np.pi / 2) + sine(20.0, sfreq, 30.0, phase=-np.pi / 2)
    data = np.vstack([lead, lag])
    matrices, _meta = connectivity_matrices(
        data, sfreq, metrics=("imcoh",), min_segments=5, segment_sec=1.0
    )
    node, meta = connectivity_node_features(
        data, sfreq, metrics=("imcoh",), min_segments=5, segment_sec=1.0
    )
    assert abs(matrices["imcoh"]["broadband"][0, 1]) < 0.2
    assert node["global_imcoh"][0] > 0.8
    assert abs(meta["matrices"]["imcoh"]["broadband"][0, 1]) < 0.2


def test_sensor_correlation_of_a_quadrature_pair_is_zero():
    from eegmath.connectivity import sensor_correlation

    sfreq = 200.0
    tone = sine(10.0, sfreq, 5.0)
    data = np.vstack([tone, sine(10.0, sfreq, 5.0, phase=np.pi / 2), tone, np.ones_like(tone)])
    corr = sensor_correlation(data)
    assert corr[0, 2] == pytest.approx(1.0, abs=1e-6)
    assert corr[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(corr[3, 0])
    assert np.isnan(corr[3, 3])
    ranked = sensor_correlation(np.vstack([tone, 2.0 * tone]), method="spearman")
    assert ranked[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_connectivity_detrend_is_explicit():
    from eegmath.connectivity.metrics import connectivity_node_features

    sfreq = 200.0
    data = np.vstack([sine(10.0, sfreq, 8.0), sine(10.0, sfreq, 8.0)])
    node, _meta = connectivity_node_features(
        data, sfreq, metrics=("coherence",), detrend="linear", min_segments=4, segment_sec=1.0
    )
    assert node["global_coherence"][0] > 0.9
    with pytest.raises(ParameterError, match="detrend"):
        connectivity_node_features(data, sfreq, detrend="nope")


def test_overlapping_regions_are_rejected():
    with pytest.raises(ParameterError, match="overlap"):
        region_connectivity(np.eye(3), ["A", "B", "C"], ["A", "B"], ["B", "C"])


def test_partition_masks_follow_frequency_rather_than_mapping_order():
    from eegmath.bands import partition_masks

    freqs = np.array([8.0, 13.0, 30.0, 45.0])
    alpha_last = {"beta": (13.0, 30.0), "alpha": (8.0, 13.0)}
    masks = dict(partition_masks(freqs, alpha_last))
    assert freqs[masks["alpha"]].tolist() == [8.0]
    assert freqs[masks["beta"]].tolist() == [13.0, 30.0]
    alpha_first = {"alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
    masks = dict(partition_masks(freqs, alpha_first))
    assert 13.0 not in freqs[masks["alpha"]]
    assert 30.0 not in freqs[masks["beta"]]
    assert freqs[masks["gamma"]].tolist() == [30.0, 45.0]
    tone = sine(13.0, 100.0, 8.0)
    for bands in (alpha_last, {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}):
        features, _meta = spectral_feature_set(tone, 100.0, bands=bands, freq_min=0.5, freq_max=30.0)
        assert float(features["beta_peak_freq"]) == pytest.approx(13.0)
        assert float(features["alpha_peak_freq"]) != pytest.approx(13.0)


def test_band_integrals_partition_the_spectrum_including_off_grid_edges():
    from eegmath.spectral.psd import _mean_frequency, band_power

    freqs = np.arange(0.0, 50.0 + 1e-9, 0.5)
    flat = np.ones_like(freqs)
    pieces = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
    covered = sum(float(band_power(freqs, flat, low, high)) for low, high in pieces)
    assert covered == pytest.approx(float(band_power(freqs, flat, 0.5, 45.0)))
    assert covered == pytest.approx(44.5)
    assert float(band_power(freqs, flat, 0.5, 4.0)) + float(band_power(freqs, flat, 4.0, 8.0)) == pytest.approx(
        float(band_power(freqs, flat, 0.5, 8.0))
    )
    coarse = np.arange(0.0, 40.0 + 1e-9, 1.0)
    assert float(band_power(coarse, np.ones_like(coarse), 8.2, 12.2)) == pytest.approx(4.0)
    assert float(band_power(coarse, np.ones_like(coarse), 10.1, 10.4)) == pytest.approx(0.3)
    assert float(_mean_frequency(freqs, flat, 0.5, 45.0)) == pytest.approx(22.75)
    ramp_f = np.arange(0.0, 21.0)
    assert float(band_power(ramp_f, ramp_f, 0.5, 10.5)) == pytest.approx((10.5**2 - 0.5**2) / 2)


def test_bands_above_nyquist_drop_out_of_the_partition():
    rng = np.random.default_rng(1)
    with pytest.warns(UserWarning, match="relative-power denominator"):
        features, _meta = spectral_feature_set(rng.normal(size=50 * 20)[None, :], 50.0)
    assert not np.isfinite(features["gamma_power"][0])
    covered = sum(float(features[f"{name}_power"][0]) for name in ("delta", "theta", "alpha", "beta"))
    assert covered == pytest.approx(float(features["total_power"][0]), rel=1e-6)
    with pytest.warns(UserWarning, match="relative-power denominator"):
        partial = spectral_feature_set(rng.normal(size=64 * 16)[None, :], 64.0)[0]
    covered = sum(float(partial[f"{name}_power"][0]) for name in ("delta", "theta", "alpha", "beta", "gamma"))
    assert np.isfinite(partial["gamma_power"][0])
    assert covered == pytest.approx(float(partial["total_power"][0]), rel=1e-6)


def test_default_partition_sums_to_total_power_for_broadband_noise():
    rng = np.random.default_rng(0)
    features, _meta = spectral_feature_set(rng.normal(size=256 * 16)[None, :], 256.0)
    names = ("delta", "theta", "alpha", "beta", "gamma")
    covered = sum(float(features[f"{name}_power"][0]) for name in names)
    assert covered == pytest.approx(float(features["total_power"][0]), rel=1e-8)
    assert sum(float(features[f"{name}_relative"][0]) for name in names) == pytest.approx(1.0)


def test_clustering_keeps_weights_that_are_already_in_the_unit_interval():
    half = np.full((3, 3), 0.5)
    np.fill_diagonal(half, 0.0)
    assert clustering_coefficient(half) == pytest.approx([0.5, 0.5, 0.5])
    with pytest.raises(ParameterError, match="0, 1"):
        clustering_coefficient(half * 4)
    with pytest.raises(ParameterError, match="non-negative"):
        clustering_coefficient(-half)
