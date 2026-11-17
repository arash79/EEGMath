"""Statistical procedures, including dependence and multiple comparisons."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from eegmath.errors import DesignError, ParameterError
from eegmath.stats import (
    bic_bayes_factor,
    bootstrap_ci,
    cluster_permutation_1d,
    cohens_d,
    compare_groups,
    correlation,
    design_summary,
    fdr_bh,
    hedges_g,
    holm,
    mean_difference_posterior,
    partial_correlation,
    permutation_test,
    pk_from_arrays,
    prediction_probability,
    rank_biserial,
    repeated_measures,
    subject_means,
)
from scipy import stats


def _two_groups(n: int = 12, effect: float = 1.5) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for i in range(n):
        rows.append({"subject": f"S{i}", "condition": "a", "score": rng.normal(0.0, 0.4)})
        rows.append({"subject": f"S{i}", "condition": "b", "score": rng.normal(effect, 0.4)})
    return pd.DataFrame(rows)


def test_benjamini_hochberg_and_holm_known_values():
    _reject, adjusted = fdr_bh([0.001, 0.04, 0.2])
    assert adjusted == pytest.approx([0.003, 0.06, 0.2])
    _reject, holm_adjusted = holm([0.01, 0.04, 0.03])
    assert holm_adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_effect_sizes_match_their_definitions():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    expected_d = (1.5 - 3.5) / np.sqrt(0.5)
    assert cohens_d(a, b) == pytest.approx(expected_d)
    assert hedges_g(a, b) == pytest.approx(expected_d * (1 - 3 / (4 * 4 - 9)))
    u = stats.mannwhitneyu(np.array([3.0, 4, 5]), np.array([0.0, 1, 2]), alternative="two-sided").statistic
    assert rank_biserial(u, 3, 3) == pytest.approx(1.0)


def test_prediction_probability_directions_and_ties():
    assert pk_from_arrays([1, 2, 3], [4, 5, 6])[0] == 1.0
    assert pk_from_arrays([4, 5, 6], [1, 2, 3])[0] == 0.0
    # Three concordant pairs and one tie: (3 + 0.5) / 4.
    assert pk_from_arrays([1, 2], [2, 3])[0] == pytest.approx(0.875)
    assert pk_from_arrays([1, 1], [1, 1])[0] == pytest.approx(0.5)
    frame = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "condition": ["low", "high", "low", "high", "low", "high"],
            "score": [1.0, 2.0, 5.0, 6.0, 3.0, 4.0],
        }
    )
    paired = prediction_probability(frame, "score", "condition", "subject", "low", "high")
    assert paired.strategy == "paired"
    assert paired.pk == pytest.approx(1.0)
    assert paired.estimator.startswith("paired")
    swapped = prediction_probability(frame, "score", "condition", "subject", "high", "low")
    assert swapped.pk == pytest.approx(1.0 - paired.pk)


def test_paired_and_unpaired_concordance_answer_different_questions():
    # Every subject increases, but the ranges of the two conditions overlap.
    frame = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "condition": ["low", "high"] * 3,
            "score": [1.0, 2.0, 5.0, 6.0, 3.0, 4.0],
        }
    )
    paired = prediction_probability(frame, "score", "condition", "subject", "low", "high", strategy="paired")
    assert paired.pk == pytest.approx(1.0)
    with pytest.raises(DesignError, match="unpaired"):
        prediction_probability(frame, "score", "condition", "subject", "low", "high", strategy="unpaired")
    disjoint = pd.DataFrame(
        {
            "subject": [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)],
            "condition": ["low"] * 4 + ["high"] * 4,
            "score": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    unpaired = prediction_probability(disjoint, "score", "condition", "subject", "low", "high", strategy="unpaired")
    assert unpaired.strategy == "unpaired"
    assert 0.5 < unpaired.pk < 1.0
    assert unpaired.p_value_test.startswith("two-sided Mann-Whitney")


def test_paired_concordance_p_value_is_a_sign_test():
    # Eight small increases and one large decrease. The sign probability
    # is 8/9. Wilcoxon ranks the large decrease above the eight increases.
    rows = []
    for i, diff in enumerate([0.01] * 8 + [-100.0]):
        rows.append({"subject": f"s{i}", "condition": "low", "score": 0.0})
        rows.append({"subject": f"s{i}", "condition": "high", "score": diff})
    paired = prediction_probability(pd.DataFrame(rows), "score", "condition", "subject", "low", "high", strategy="paired")
    sign_p = float(stats.binomtest(8, 9, p=0.5, alternative="two-sided").pvalue)
    rank_p = float(stats.wilcoxon(np.array([0.01] * 8 + [-100.0]), alternative="two-sided").pvalue)
    assert paired.pk == pytest.approx(8 / 9)
    assert paired.p_value == pytest.approx(sign_p)
    assert paired.p_value < 0.05 < rank_p
    assert "sign test" in paired.p_value_test


def test_compare_groups_refuses_pseudoreplication():
    frame = _two_groups()
    # Five technical replicates of each subject-condition cell.
    repeated = pd.concat([frame] * 5, ignore_index=True)
    with pytest.raises(DesignError):
        compare_groups(repeated, ["score"], "condition", "a", "b", test="mannwhitney")
    with pytest.raises(DesignError):
        compare_groups(repeated, ["score"], "condition", "a", "b", subject="subject", unit="row", test="welch")
    result = compare_groups(
        repeated, ["score"], "condition", "a", "b", subject="subject", test="wilcoxon"
    )
    assert result.loc[0, "n_a"] == 12
    assert result.loc[0, "p_value"] < 0.01
    assert result.loc[0, "interval"].startswith("95% paired Student-t")
    # The contrast is a minus b, and b is the larger condition.
    assert result.loc[0, "effect"] < 0
    assert result.loc[0, "mean_difference_a_minus_b"] < 0
    assert np.isfinite(result.loc[0, "ci_low"])
    overlap = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "condition": ["a", "b"],
            "score": [0.0, 1.0],
        }
    )
    with pytest.raises(DesignError):
        compare_groups(overlap, ["score"], "condition", "a", "b", subject="subject", test="welch")


def test_compare_groups_warns_when_it_averages_a_ratio():
    frame = pd.DataFrame(
        {
            "subject": ["s", "s", "t", "t", "u", "u", "v", "v"],
            "condition": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "alpha_relative": [0.2, 0.4, 0.3, 0.9, 0.1, 0.2, 0.4, 0.6],
            "alpha_power": [2.0, 4.0, 3.0, 9.0, 1.0, 2.0, 4.0, 6.0],
        }
    )
    with pytest.warns(UserWarning, match="relative power"):
        compare_groups(frame, ["alpha_relative"], "condition", "a", "b", subject="subject", test="welch")
    one_row = frame.groupby(["subject", "condition"], as_index=False)[["alpha_relative", "alpha_power"]].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        compare_groups(one_row, ["alpha_power"], "condition", "a", "b", subject="subject", test="welch")


def test_compare_groups_on_an_external_table():
    rng = np.random.default_rng(5)
    rows = []
    for i, group in enumerate(("control", "task")):
        for j in range(15):
            rows.append(
                {
                    "subject": f"{group}-{j}",
                    "condition": group,
                    "custom_feature": rng.normal(i * 1.2, 0.5),
                }
            )
    table = pd.DataFrame(rows)
    result = compare_groups(
        table, ["custom_feature"], "condition", "task", "control", subject="subject", test="welch"
    )
    assert result.loc[0, "p_corrected"] < 0.05
    assert result.loc[0, "mean_difference_a_minus_b"] > 0
    summary = design_summary(table, subject="subject", condition="condition")
    assert summary["n_subjects"] == 30
    assert summary["subjects_in_multiple_conditions"] == 0


def test_likelihood_ratio_does_not_invent_a_degree_of_freedom():
    from eegmath.stats import likelihood_ratio

    frame = _two_groups(n=12, effect=1.0)
    same = likelihood_ratio(frame, "score ~ C(condition)", "score ~ C(condition)", "subject")
    assert same["df"] == 0
    assert np.isnan(same["p_value"])
    rng = np.random.default_rng(1)
    rows = []
    for i in range(20):
        for condition, shift in (("a", 0.0), ("b", 1.0)):
            rows.append(
                {
                    "subject": f"S{i}",
                    "condition": condition,
                    "age": float(i),
                    "score": rng.normal(shift + 0.01 * i, 0.2),
                }
            )
    wider = pd.DataFrame(rows)
    nested = likelihood_ratio(wider, "score ~ age + C(condition)", "score ~ age", "subject")
    assert nested["df"] == 1
    assert nested["p_value"] < 0.01
    backwards = likelihood_ratio(wider, "score ~ age", "score ~ age + C(condition)", "subject")
    assert backwards["df"] == -1
    assert np.isnan(backwards["p_value"])


def test_mixed_model_withholds_inference_when_the_subject_df_is_not_positive():
    rows = []
    for subject in range(3):
        for level, mean in enumerate((0.0, 1.0, 2.0, 3.0)):
            rows.append({"subject": f"s{subject}", "condition": f"c{level}", "y": mean + 0.01 * subject})
    fit = repeated_measures(
        pd.DataFrame(rows),
        "y",
        "condition",
        "subject",
        reference="c0",
        pairs=[("c0", "c1")],
    )
    assert fit.n_subjects == 3
    assert fit.dof == pytest.approx(-1.0)
    assert fit.coefficients["p_value"].isna().all()
    assert fit.coefficients["ci_low"].isna().all()
    assert np.isfinite(fit.coefficients["estimate"]).all()
    contrast = fit.contrasts.iloc[0]
    assert contrast["estimate"] == pytest.approx(1.0, abs=0.05)
    assert np.isnan(contrast["p_value"])
    assert np.isnan(contrast["ci_low"])


def test_contrast_matches_the_named_levels_and_rejects_a_missing_one():
    from eegmath.stats.mixed import contrast_levels, fit_mixed_model

    rng = np.random.default_rng(0)
    rows = []
    for subject in range(8):
        for arm, warm, mean in (
            ("ref", "ref", 0.0),
            ("ref", "x", 1.0),
            ("x", "ref", 2.0),
            ("x", "x", 4.0),
            ("y", "ref", 7.0),
            ("y", "x", 9.0),
        ):
            rows.append(
                {
                    "subject": f"s{subject}",
                    "arm": arm,
                    "warm": warm,
                    "y": mean + float(rng.normal(scale=0.02)),
                }
            )
    fit = fit_mixed_model(pd.DataFrame(rows), "y ~ C(Q('arm')) + C(Q('warm'))", "subject")
    contrast = contrast_levels(fit, "arm", "x", "y")
    arm_x = float(fit.fit.fe_params["C(Q('arm'))[T.x]"])
    arm_y = float(fit.fit.fe_params["C(Q('arm'))[T.y]"])
    assert contrast["estimate"] == pytest.approx(arm_y - arm_x)
    assert contrast["estimate"] == pytest.approx(5.0, abs=0.2)
    with pytest.raises(DesignError, match="not values"):
        contrast_levels(fit, "arm", "not_a_level", "y")
    plain = fit_mixed_model(
        _two_groups(n=8, effect=1.0),
        "score ~ C(condition)",
        "subject",
    )
    paired = contrast_levels(plain, "condition", "a", "b")
    assert np.isfinite(paired["estimate"])
    assert paired["estimate"] > 0


def test_repeated_measures_recovers_a_within_subject_shift():
    frame = _two_groups(n=14, effect=2.0)
    # Extra rows must be averaged before the model sees them.
    frame = pd.concat([frame, frame.assign(score=frame["score"] + 0.01)], ignore_index=True)
    fit = repeated_measures(frame, "score", "condition", "subject", reference="a", pairs=[("a", "b")])
    assert fit.converged
    assert fit.n_subjects == 14
    assert fit.omnibus["p_value"] < 0.01
    contrast = fit.contrasts.iloc[0]
    assert contrast["estimate"] == pytest.approx(2.0, abs=0.4)
    varying = frame.copy()
    varying["dose"] = np.arange(len(varying))
    with pytest.raises(DesignError):
        repeated_measures(varying, "score", "condition", "subject", covariates=["dose"])


def test_correlation_partial_correlation_and_bootstrap():
    rng = np.random.default_rng(6)
    z = rng.normal(size=80)
    x = rng.normal(size=80)
    y = x + z
    frame = pd.DataFrame({"x": x, "y": y, "z": z})
    raw = correlation(frame["x"], frame["y"])
    partial = partial_correlation(frame, "x", "y", ["z"])
    assert raw["r"] < 0.95
    assert partial["r"] == pytest.approx(1.0, abs=0.05)
    interval = bootstrap_ci(x + 3, n_boot=400, seed=1)
    assert interval["ci_low"] < interval["estimate"] < interval["ci_high"]


def test_permutation_respects_pairing_and_cluster_mass_finds_a_window():
    frame = _two_groups(n=10, effect=2.0)
    paired = permutation_test(frame, "score", "condition", "subject", "b", "a", n_perm=199, seed=0)
    assert paired["design"] == "within_subject_sign_flip"
    assert paired["p_value"] <= 0.05
    between = []
    rng = np.random.default_rng(7)
    for i in range(8):
        between.append({"subject": f"a{i}", "condition": "a", "score": rng.normal(0, 0.3)})
        between.append({"subject": f"b{i}", "condition": "b", "score": rng.normal(2, 0.3)})
    unpaired = permutation_test(pd.DataFrame(between), "score", "condition", "subject", "b", "a", n_perm=199, seed=1)
    assert unpaired["design"] == "between_subject_label_shuffle"
    assert unpaired["p_value"] <= 0.05
    differences = np.zeros((10, 30))
    differences[:, 8:18] = 2.0
    differences += np.random.default_rng(8).normal(0, 0.15, differences.shape)
    clusters = cluster_permutation_1d(differences, n_perm=199, seed=3)
    assert not clusters.empty
    hit = clusters.loc[clusters["significant"]]
    assert not hit.empty
    assert ((hit["start"] <= 12) & (hit["end"] >= 12)).any()


def test_zero_variance_posterior_still_names_its_assumption():
    posterior = mean_difference_posterior(np.array([2.0, 2.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    assert posterior["prob_mean_gt_0"] == 1.0
    assert posterior["ci_low"] == posterior["ci_high"] == posterior["mean"]
    assert np.isnan(posterior["df"])
    assert "assumption" in posterior


def test_posterior_and_bic_distinguish_a_large_shift_from_noise():
    posterior = mean_difference_posterior(np.array([5.0, 6, 7, 8]), np.array([0.0, 1, 0, 1]))
    assert posterior["prob_mean_gt_0"] > 0.95
    assert posterior["ci_low"] > 0
    frame = pd.DataFrame(
        {
            "y": np.r_[np.zeros(20), np.full(20, 3.0)],
            "condition": ["a"] * 20 + ["b"] * 20,
        }
    )
    evidence = bic_bayes_factor(frame, "y", "condition")
    assert evidence["bf10"] > 10
    means = subject_means(frame.assign(subject=np.arange(len(frame)).astype(str)), ["y"], "subject", by=["condition"])
    assert len(means) == 40


def test_ttest_interval_uses_pooled_variance():
    from eegmath.stats.effects import mean_difference_ci

    a = np.array([0.0, 0.1, -0.1, 0.2, -0.2])
    b = np.array([5.0, 10.0, 0.0, 15.0, -2.0, 8.0, 3.0])
    frame = pd.DataFrame(
        {
            "score": np.r_[a, b],
            "condition": ["a"] * a.size + ["b"] * b.size,
            "subject": [f"s{i}" for i in range(a.size + b.size)],
        }
    )
    student = compare_groups(frame, ["score"], "condition", "a", "b", subject="subject", test="ttest", unit="row")
    welch = compare_groups(frame, ["score"], "condition", "a", "b", subject="subject", test="welch", unit="row")
    _, lo, hi = mean_difference_ci(a, b, equal_var=True)
    assert student.loc[0, "ci_low"] == pytest.approx(lo)
    assert student.loc[0, "ci_high"] == pytest.approx(hi)
    assert welch.loc[0, "ci_low"] != pytest.approx(student.loc[0, "ci_low"])
    assert student.loc[0, "interval"].startswith("95% pooled-variance Student-t")
    assert welch.loc[0, "interval"].startswith("95% Welch-Satterthwaite")


def test_partial_correlation_uses_covariate_degrees_of_freedom():
    rng = np.random.default_rng(1)
    n = 40
    z = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "x": z + rng.normal(scale=0.2, size=n),
            "y": 0.5 * z + rng.normal(scale=0.3, size=n),
            "z": z,
        }
    )
    out = partial_correlation(frame, "x", "y", ["z"])
    df = n - 1 - 2
    t_stat = out["r"] * np.sqrt(df / (1.0 - out["r"] ** 2))
    assert out["df"] == df
    assert out["p_value"] == pytest.approx(float(2.0 * stats.t.sf(abs(t_stat), df)))


def test_pearson_interval_is_withheld_for_three_pairs():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = correlation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    assert np.isfinite(out["r"])
    assert np.isnan(out["ci_low"]) and np.isnan(out["ci_high"])


def test_nonfinite_pvalues_are_left_out_of_the_family():
    _rejected, adjusted = fdr_bh(np.array([0.001, np.nan, 0.04]))
    assert adjusted[0] == pytest.approx(0.002)
    assert np.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.04)


def test_opposite_sign_runs_are_separate_clusters():
    differences = np.random.default_rng(0).normal(0.0, 0.05, (12, 40))
    differences[:, 5:12] += 3.0
    differences[:, 12:20] -= 3.0
    clusters = cluster_permutation_1d(differences, n_perm=99, seed=1)
    assert ((clusters["start"] < 12) & (clusters["end"] < 12)).any()
    assert (clusters["start"] >= 12).any()
    assert not ((clusters["start"] <= 8) & (clusters["end"] >= 15)).any()


def test_paired_prediction_probability_does_not_silently_unpair():
    frame = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "group": ["low", "high", "low", "high"],
            "score": [0.0, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(DesignError, match="paired"):
        prediction_probability(frame, "score", "group", "subject", "low", "high", strategy="paired")


def test_mixed_model_ignores_an_unused_empty_column():
    from eegmath.stats import fit_mixed_model

    frame = _two_groups()
    frame["unused"] = np.nan
    fit = fit_mixed_model(frame, "score ~ C(condition)", "subject")
    assert fit.converged
    assert fit.n_obs == len(frame)
    assert fit.n_subjects == frame["subject"].nunique()


def test_missing_subject_ids_are_not_pooled_into_one_unit():
    frame = _two_groups(n=4)
    frame.loc[0, "subject"] = np.nan
    with pytest.raises(DesignError, match="no value"):
        compare_groups(frame, ["score"], "condition", "a", "b", subject="subject", test="welch")


def test_compare_groups_refuses_to_average_channels():
    frame = _two_groups(n=6)
    copy = frame.copy()
    frame["channel"] = "O1"
    copy["channel"] = "O2"
    both = pd.concat([frame, copy], ignore_index=True)
    with pytest.raises(DesignError, match="channels"):
        compare_groups(both, ["score"], "condition", "a", "b", subject="subject", test="wilcoxon")
    one = both.loc[both["channel"] == "O1"]
    result = compare_groups(one, ["score"], "condition", "a", "b", subject="subject", test="wilcoxon")
    assert result.loc[0, "n_a"] == 6


def test_compare_groups_accepts_channel_as_the_grouping_factor():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(6):
        rows.append({"subject": f"S{i}", "channel": "O1", "score": float(rng.normal(0.0, 0.3))})
        rows.append({"subject": f"S{i}", "channel": "O2", "score": float(rng.normal(1.2, 0.3))})
    result = compare_groups(
        pd.DataFrame(rows), ["score"], "channel", "O1", "O2", subject="subject", test="wilcoxon"
    )
    assert result.loc[0, "n_a"] == 6
    assert result.loc[0, "mean_difference_a_minus_b"] < 0


def test_pvalues_outside_the_unit_interval_are_rejected():
    with pytest.raises(ParameterError):
        fdr_bh([-0.01, 0.2])
    with pytest.raises(ParameterError):
        fdr_bh([0.1, 1.2])


def test_ols_recovers_a_line_and_refuses_repeated_subjects():
    from eegmath.stats import fit_ols

    x = np.linspace(0.0, 1.0, 30)
    frame = pd.DataFrame({"y": 1.5 + 2.0 * x, "x": x, "subject": [f"s{i}" for i in range(30)]})
    fit = fit_ols(frame, "y ~ x", subject="subject")
    coefficients = fit.coefficients.set_index("term")
    assert coefficients.loc["Intercept", "estimate"] == pytest.approx(1.5)
    assert coefficients.loc["x", "estimate"] == pytest.approx(2.0)
    assert fit.df_resid == 28
    with pytest.raises(DesignError, match="more than one row"):
        fit_ols(pd.concat([frame, frame], ignore_index=True), "y ~ x", subject="subject")


def test_resampling_and_posterior_reject_empty_inputs():
    with pytest.raises(ParameterError):
        permutation_test(_two_groups(n=4), "score", "condition", "subject", "b", "a", n_perm=0)
    with pytest.raises(ParameterError):
        bootstrap_ci(np.array([1.0, 2.0, 3.0]), n_boot=0)
    with pytest.raises(DesignError):
        mean_difference_posterior(np.array([1.0]), np.array([0.0, 1.0]))
