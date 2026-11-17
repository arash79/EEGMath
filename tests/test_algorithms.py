"""Analytical and reference checks for signal descriptors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eegmath.errors import ParameterError
from eegmath.algorithms.entropy import (
    approximate_entropy,
    lempel_ziv_complexity,
    multiscale_sample_entropy,
    permutation_entropy,
    sample_entropy,
    svd_entropy,
)
from scipy import signal as scipy_signal
from eegmath.algorithms.fractal import detrended_fluctuation, higuchi_fd, katz_fd, petrosian_fd
from eegmath.algorithms.time_domain import (
    burst_suppression_ratio,
    envelope_lzc,
    hjorth_parameters,
    line_length,
    rms,
    zero_crossing_rate,
)

from synthetic import sine


def test_permutation_entropy_of_a_monotone_series_is_zero():
    assert permutation_entropy(np.arange(40), order=3, normalize=True) == 0.0


def test_permutation_entropy_matches_the_ordinal_distribution():
    series = np.array([4, 7, 9, 10, 6, 11, 3], dtype=float)
    # Order 2: four ascending pairs and two descending pairs.
    ascending = 4 / 6
    descending = 2 / 6
    expected = -(ascending * math.log2(ascending) + descending * math.log2(descending))
    assert permutation_entropy(series, order=2, normalize=True) == pytest.approx(expected)


def test_lzc_known_strings():
    assert np.isnan(lempel_ziv_complexity("", normalize=False))
    assert lempel_ziv_complexity("1001111011000010", normalize=False) == 6
    assert lempel_ziv_complexity("1001111011000010", normalize=True) == pytest.approx(1.5)
    binary = np.array([1, 0] * 5)
    assert lempel_ziv_complexity(binary, normalize=False) == 3
    assert lempel_ziv_complexity(binary.astype(float), normalize=False) == 3
    signed = np.array([-2, -1, -2, 4], dtype=float)
    assert lempel_ziv_complexity(signed, normalize=False) == lempel_ziv_complexity(signed.astype(int), normalize=False)
    with pytest.raises(ParameterError, match="discrete"):
        lempel_ziv_complexity(np.array([0.2, 0.8, 0.2, 1.5]))


def test_sample_and_approximate_entropy_are_small_for_a_sine():
    x = sine(1.0, 100.0, 30.0)
    sampen = sample_entropy(x)
    apen = approximate_entropy(x)
    assert 0 < sampen < 0.5
    assert 0 < apen < 0.5


def test_sample_entropy_of_a_constant_is_zero():
    assert sample_entropy(np.ones(80)) == 0.0


def test_entropy_rejects_bad_parameters_and_nonfinite_input():
    with pytest.raises(ParameterError):
        permutation_entropy(np.arange(10), order=1)
    assert np.isnan(sample_entropy(np.array([1.0, np.nan, 2.0])))
    assert np.isnan(svd_entropy(np.array([1.0])))


def test_multiscale_entropy_fixes_radius_from_the_original_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    fixed = multiscale_sample_entropy(x, (1, 2), r_mode="original")
    assert set(fixed) == {1, 2}
    assert np.isfinite(fixed[1])
    with pytest.raises(ParameterError):
        multiscale_sample_entropy(x, (1,), r_mode="per-scale")


def test_straight_line_fractal_dimensions_are_one():
    line = np.arange(1000, dtype=float)
    assert higuchi_fd(line) == pytest.approx(1.0, abs=1e-6)
    assert petrosian_fd(line) == pytest.approx(1.0, abs=1e-6)
    assert katz_fd(line) == pytest.approx(1.0, abs=1e-6)


def test_dfa_orders_white_noise_below_its_integral():
    rng = np.random.default_rng(4)
    white = rng.normal(size=2000)
    brown = np.cumsum(white)
    alpha_white = detrended_fluctuation(white)
    alpha_brown = detrended_fluctuation(brown)
    assert 0.3 < alpha_white < 0.8
    assert 1.2 < alpha_brown < 1.8
    assert alpha_brown > alpha_white + 0.5


def test_hjorth_mobility_of_a_sine_is_two_pi_f():
    sfreq = 100.0
    freq = 1.0
    activity, mobility, complexity = hjorth_parameters(sine(freq, sfreq, 20.0), sfreq)
    # Discrete derivative: 2 * sfreq * sin(pi * f / sfreq), which is 2*pi*f
    # up to the usual one-sample lag. Complexity of a pure sine is 1.
    # The one-sample difference is shorter than the series, so the value sits
    # within a fraction of a percent of 2*sfreq*sin(pi*f/sfreq) and of 2*pi*f.
    assert activity == pytest.approx(0.5, abs=1e-6)
    assert mobility == pytest.approx(2 * math.pi * freq, rel=1e-3)
    assert complexity == pytest.approx(1.0, abs=1e-3)


def test_line_length_and_zero_crossings_have_physical_units():
    sfreq = 200.0
    ramp = np.arange(int(sfreq)) / sfreq
    assert line_length(ramp, sfreq) == pytest.approx(1.0)
    assert np.isnan(line_length(ramp, 0.0))
    assert zero_crossing_rate(sine(10.0, sfreq, 5.0, phase=0.3), sfreq) == pytest.approx(20.0, abs=0.5)
    # Three crossings on the span from the first sample to the last.
    assert zero_crossing_rate(np.array([1.0, -1.0, 1.0, -1.0]), 4.0) == pytest.approx(4.0)
    # Exact zero is not a sign. A touch that returns is not a crossing,
    # from either side. A crossing through zero is one crossing.
    assert zero_crossing_rate(np.array([-1.0, 0.0, -1.0]), 2.0) == 0.0
    assert zero_crossing_rate(np.array([1.0, 0.0, 1.0]), 2.0) == 0.0
    assert zero_crossing_rate(np.array([1.0, 0.0, -1.0]), 2.0) == pytest.approx(1.0)
    assert rms(np.array([3.0, -3.0])) == pytest.approx(3.0)


def test_burst_suppression_ratio_counts_only_sustained_intervals():
    sfreq = 200.0
    # One second of silence is longer than the 0.5 s minimum.
    assert burst_suppression_ratio(np.zeros(int(sfreq)), sfreq, threshold=5e-6) == 1.0
    # Start at the crest so the loud segment does not continue a preceding silence.
    loud = sine(10.0, sfreq, 4.0, amplitude=50e-6, phase=np.pi / 2)
    # Zero crossings of a 50 µV sine fall below 5 µV, but only for a few milliseconds.
    assert burst_suppression_ratio(loud, sfreq, threshold=5e-6) == 0.0
    quiet = sine(10.0, sfreq, 4.0, amplitude=1e-6)
    assert burst_suppression_ratio(quiet, sfreq, threshold=5e-6) == 1.0
    # Exactly 0.5 s is not longer than the minimum. One extra sample is.
    exact = np.zeros(int(0.5 * sfreq))
    assert np.isnan(burst_suppression_ratio(exact, sfreq, threshold=5e-6))
    held = np.concatenate([np.zeros(int(0.5 * sfreq) + 1), loud[: int(sfreq)]])
    assert burst_suppression_ratio(held, sfreq, threshold=5e-6) == pytest.approx(
        (0.5 * sfreq + 1) / held.size
    )
    mixed = np.concatenate([np.zeros(int(sfreq)), loud[: int(sfreq)]])
    assert burst_suppression_ratio(mixed, sfreq, threshold=5e-6) == pytest.approx(0.5)
    # A sample sitting on the threshold is not below it.
    assert burst_suppression_ratio(np.full(int(sfreq), 5e-6), sfreq, threshold=5e-6) == 0.0
    instantaneous = burst_suppression_ratio(loud, sfreq, threshold=5e-6, min_duration=0)
    assert instantaneous == pytest.approx(float(np.mean(np.abs(loud) < 5e-6)))


def test_petrosian_counts_every_change_of_derivative_sign():
    series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    n = series.size
    n_delta = 5
    expected = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))
    assert petrosian_fd(series) == pytest.approx(expected)
    assert petrosian_fd(np.arange(50, dtype=float)) == pytest.approx(1.0)
    # Derivative 1, 0, -1 changes sign once. The zero sample is not a second change.
    touched = np.array([0.0, 1.0, 1.0, 0.0])
    n_touch = touched.size
    n_delta_touch = 1
    touched_expected = np.log10(n_touch) / (
        np.log10(n_touch) + np.log10(n_touch / (n_touch + 0.4 * n_delta_touch))
    )
    assert petrosian_fd(touched) == pytest.approx(touched_expected)


def test_feature_extraction_does_not_hide_an_illegal_parameter():
    from eegmath.features import FeatureConfig, extract_features
    from eegmath.recording import EEG

    series = sine(10.0, 200.0, 4.0)
    eeg = EEG(np.vstack([series, series]), 200.0, ("A", "B"), units="uV")
    with pytest.raises(ParameterError, match="Band-pass"):
        extract_features(eeg, ["envelope_lzc"], config=FeatureConfig(envelope_band=(1.0, 200.0)))
    with pytest.raises(ParameterError, match="kmax"):
        extract_features(eeg, ["higuchi_fd"], config=FeatureConfig(higuchi_kmax=1))


def test_envelope_lzc_binarises_the_hilbert_amplitude_at_its_mean():
    sfreq = 200.0
    times = np.arange(int(sfreq * 8.0)) / sfreq
    series = np.where(times < 4.0, 1.0, 5.0) * np.sin(2 * np.pi * 10.0 * times)
    sos = scipy_signal.butter(4, [1.0, 40.0], btype="bandpass", fs=sfreq, output="sos")
    envelope = np.abs(scipy_signal.hilbert(scipy_signal.sosfiltfilt(sos, series)))
    at_mean = (envelope > np.mean(envelope)).astype(np.uint8)
    at_median = (envelope > np.median(envelope)).astype(np.uint8)
    assert not np.array_equal(at_mean, at_median)
    assert envelope_lzc(series, sfreq) == pytest.approx(lempel_ziv_complexity(at_mean))
    with pytest.raises(ParameterError, match="Band-pass"):
        envelope_lzc(series, sfreq, fhi=sfreq)


def test_katz_dimension_is_infinite_when_the_denominator_vanishes():
    import warnings

    alternating = np.array([1.0, 2.0] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isposinf(katz_fd(alternating))


def test_higuchi_omits_scales_whose_curve_length_is_zero():
    # Even scales of a period-2 series are constant, so their length is zero.
    # The slope is computed from the scales that remain.
    value = higuchi_fd(np.array([1.0, 2.0] * 80), kmax=6)
    assert np.isfinite(value)
    assert value > 1.5


def test_hjorth_complexity_of_a_short_sine_is_only_near_one():
    sfreq = 100.0
    _activity, _mobility, complexity = hjorth_parameters(sine(10.0, sfreq, 0.5), sfreq)
    assert 1.0 < complexity < 1.05


def test_antropy_agreement_when_installed():
    antropy = pytest.importorskip("antropy")
    series = np.array([4, 7, 9, 10, 6, 11, 3], dtype=float)
    assert permutation_entropy(series, order=3, normalize=True) == pytest.approx(
        antropy.perm_entropy(series, order=3, normalize=True), abs=1e-10
    )
    sine_wave = sine(1.0, 100.0, 30.0)
    assert sample_entropy(sine_wave) == pytest.approx(antropy.sample_entropy(sine_wave), abs=1e-6)
    assert approximate_entropy(sine_wave) == pytest.approx(antropy.app_entropy(sine_wave), abs=1e-6)
    assert svd_entropy(sine_wave, normalize=True) == pytest.approx(
        antropy.svd_entropy(sine_wave, normalize=True), abs=1e-6
    )
    mobility, complexity = antropy.hjorth_params(sine_wave, sf=100.0)
    _, ours, our_complexity = hjorth_parameters(sine_wave, 100.0)
    assert ours == pytest.approx(float(mobility), rel=1e-6)
    assert our_complexity == pytest.approx(float(complexity), rel=1e-6)
    line = np.arange(800, dtype=float)
    assert higuchi_fd(line) == pytest.approx(antropy.higuchi_fd(line), abs=1e-6)
    assert petrosian_fd(line) == pytest.approx(float(antropy.petrosian_fd(line)), abs=1e-8)
    assert katz_fd(line) == pytest.approx(float(antropy.katz_fd(line)), abs=1e-8)
    rng = np.random.default_rng(9)
    white = rng.normal(size=1500)
    assert detrended_fluctuation(white) == pytest.approx(antropy.detrended_fluctuation(white), abs=0.03)
