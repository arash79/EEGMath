"""Hypothesis tests that do not choose a procedure on the user's behalf.

``compare_groups`` runs the test named in ``test``. It does not switch to
another test when a diagnostic looks bad. Diagnostics are returned beside
the test. Unpaired tests refuse designs in which the same subject appears
in both groups. The default unit is the subject: technical replicates are
averaged first, and that aggregation is recorded in the result. A
``channel`` column with more than one name is refused. Averaging channels
is a spatial summary and has to be done before the test.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from eegmath.errors import DesignError, ParameterError
from eegmath.spectral.psd import is_derived_spectral_column
from eegmath.stats.design import subject_means
from eegmath.stats.effects import cohens_d, cohens_dz, hedges_g, mean_difference_ci, rank_biserial
from eegmath.stats.multiple import apply_correction


def _level_present(series: pd.Series, level) -> bool:
    if pd.isna(level):
        return bool(series.isna().any())
    return bool((series == level).any())


def _shapiro(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3 or x.size > 5000 or np.unique(x).size < 2:
        return np.nan
    return float(stats.shapiro(x).pvalue)


def compare_groups(
    frame: pd.DataFrame,
    values: Sequence[str],
    group: str,
    a: str,
    b: str,
    *,
    subject: str | None = None,
    test: str = "mannwhitney",
    unit: str = "subject",
    correction: str = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare two groups on one or more value columns.

    Parameters
    ----------
    test
        ``mannwhitney``, ``welch``, ``ttest``, or ``wilcoxon``.
        Wilcoxon is paired. The others are unpaired and raise
        :class:`~eegmath.errors.DesignError` if any subject is in both groups.
    unit
        ``subject`` averages rows within subject and group before the test.
        ``row`` treats each row as an observation and raises if a subject
        column shows repeated rows inside a group. Several channel names
        in one table raise, for either unit, unless ``channel`` itself is
        the grouping column. In that case each group is one channel and
        rows are not averaged across channels.
    correction
        Multiple-comparison method applied to this call's family of columns.
        Non-finite p-values stay missing and are not counted in that family.

    An empty table raises before any label is described as absent. A
    label in ``a`` or ``b`` that is not in ``group``, and a group with
    fewer than two finite observations, raise
    :class:`~eegmath.errors.DesignError`. The function does not return a
    row of missing p-values for that case. ``significant`` is false when
    a finite corrected p-value is not rejected. It is not a way of
    reporting that the comparison could not be run. ``alpha`` is that
    rejection threshold after correction. It does not change the
    interval. The call warns when the flag and the interval disagree,
    in either direction. The flag stays the corrected test.

    The interval is a 95% Student-t interval for the mean difference.
    It uses the pooled variance for ``ttest`` and the Welch–Satterthwaite
    interval for ``welch`` and ``mannwhitney``. The ``interval`` column
    names that choice, including when the test itself is Mann-Whitney
    or Wilcoxon. For Wilcoxon, ``normality_p_a`` is the Shapiro–Wilk
    test of the paired differences and ``normality_p_b`` is empty. The
    diagnostic does not choose the test. A zero-width interval means
    the observations do not vary. The call warns, and the effect size
    is missing.
    """
    if test not in {"mannwhitney", "welch", "ttest", "wilcoxon"}:
        raise ParameterError("test must be mannwhitney, welch, ttest, or wilcoxon.")
    if unit not in {"subject", "row"}:
        raise ParameterError("unit must be 'subject' or 'row'.")
    missing = [c for c in [group, *values] if c not in frame.columns]
    if missing:
        raise DesignError(f"Columns not in the table: {missing}.")
    if len(frame) == 0:
        raise DesignError(
            "The table has no rows. A channel filter or an earlier selection "
            "may have removed them. An empty table is not a completed test "
            "and is not evidence of no difference."
        )
    absent = [level for level in (a, b) if not _level_present(frame[group], level)]
    if absent:
        observed = frame[group].dropna().unique().tolist()
        preview = observed[:8]
        extra = "" if len(observed) <= 8 else f" ({len(observed) - 8} more not shown)"
        raise DesignError(
            f"Column {group!r} has no rows labelled {absent}. "
            f"Observed labels include {preview}{extra}. "
            "An absent label is not a completed test and is not evidence of no difference."
        )
    if unit == "subject" and subject is None:
        raise DesignError(
            "unit='subject' needs a subject column. Pass subject=, or pass "
            "unit='row' to state that each row is an independent observation."
        )
    selected = frame.loc[frame[group].isin([a, b])]
    if (
        "channel" in selected.columns
        and "channel" not in values
        and group != "channel"
        and selected["channel"].nunique(dropna=False) > 1
    ):
        n_channels = int(selected["channel"].nunique(dropna=False))
        raise DesignError(
            f"{n_channels} channels are in this table. compare_groups does not "
            "average them: the mean of channel-wise values is a spatial summary "
            "that changes ratios and nonlinear features. Pick one channel, or "
            "aggregate with eegmath.aggregate_regions (absolute band powers must "
            "be in the table) or an explicit groupby, then drop the channel column."
        )
    derived = [name for name in values if is_derived_spectral_column(str(name))]
    if unit == "subject" and derived and subject is not None and subject in selected.columns:
        replicate_counts = selected.groupby([subject, group], dropna=False).size()
        if (replicate_counts > 1).any():
            warnings.warn(
                "compare_groups averages "
                f"{derived} within subject and group. The mean of a relative power, "
                "a decibel value, or a band ratio is not that transform of the "
                "averaged absolute power. Rebuild those columns with aggregate_epochs "
                "or aggregate_regions, then pass one row per subject.",
                UserWarning,
                stacklevel=2,
            )
    work = selected.loc[:, [group, *values, *([subject] if subject else [])]].copy()
    if unit == "subject":
        work = subject_means(work, list(values), subject, by=[group])
        aggregation = "mean_within_subject_and_group"
    else:
        if subject and subject in work.columns:
            counts = work.groupby([subject, group], dropna=False).size()
            if (counts > 1).any():
                raise DesignError(
                    "unit='row' found repeated rows inside subject × group. "
                    "Average them, use a mixed model, or set unit='subject'."
                )
        aggregation = "row"
    if test == "wilcoxon":
        if subject is None:
            raise DesignError("wilcoxon needs subject= so that pairs can be formed.")
        rows = [_wilcoxon_row(work, value, group, subject, a, b) for value in values]
    else:
        _reject_overlap(work, group, subject, a, b, test)
        rows = [_unpaired_row(work, value, group, a, b, test) for value in values]
    result = pd.DataFrame(rows)
    result["interval"] = {
        "mannwhitney": "95% Welch-Satterthwaite interval for the mean difference",
        "welch": "95% Welch-Satterthwaite interval for the mean difference",
        "ttest": "95% pooled-variance Student-t interval for the mean difference",
        "wilcoxon": "95% paired Student-t interval for the mean difference",
    }[test]
    result["unit"] = unit
    result["aggregation"] = aggregation
    result["test"] = test
    result["group_a"] = a
    result["group_b"] = b
    result = apply_correction(result, method=correction, alpha=alpha)
    result["alpha"] = alpha
    result["correction"] = correction
    result["family"] = "columns passed to this compare_groups call"
    _warn_if_significance_and_interval_disagree(result)
    return result


def _warn_if_zero_width(feature: str, low: float, high: float) -> None:
    if not (np.isfinite(low) and np.isfinite(high) and low == high):
        return
    warnings.warn(
        f"The mean-difference interval for {feature!r} has zero width "
        "because those observations do not vary. Cohen's d, Hedges' g, and "
        "d_z are undefined in that case. A zero-width interval is not a "
        "precise estimate of a nonzero effect.",
        UserWarning,
        stacklevel=4,
    )


def _warn_if_significance_and_interval_disagree(result: pd.DataFrame) -> None:
    rejected_includes: list[str] = []
    retained_excludes: list[str] = []
    for row in result.itertuples(index=False):
        low = float(row.ci_low)
        high = float(row.ci_high)
        if not (np.isfinite(low) and np.isfinite(high)) or low == high:
            continue
        includes = low <= 0 <= high
        if bool(row.significant) and includes:
            rejected_includes.append(str(row.feature))
        elif not bool(row.significant) and not includes:
            retained_excludes.append(str(row.feature))
    parts: list[str] = []
    if rejected_includes:
        parts.append(
            "significant is true for "
            f"{rejected_includes} because the corrected test rejected, while the "
            "mean-difference interval includes 0"
        )
    if retained_excludes:
        parts.append(
            "significant is false for "
            f"{retained_excludes} because the corrected test did not reject, while the "
            "mean-difference interval excludes 0"
        )
    if not parts:
        return
    warnings.warn(
        ". ".join(parts)
        + ". The flag is the corrected test. The interval is a separate "
        "95% t interval for the mean difference, and alpha does not change "
        "it. Adding another column to the same call can move the flag "
        "without moving the interval. Report the mean difference and the "
        "interval with the flag.",
        UserWarning,
        stacklevel=3,
    )


def _reject_overlap(work, group, subject, a, b, test) -> None:
    if not subject or subject not in work.columns:
        return
    left = set(work.loc[work[group] == a, subject])
    right = set(work.loc[work[group] == b, subject])
    overlap = left & right
    if overlap:
        raise DesignError(
            f"{len(overlap)} subjects appear in both {a!r} and {b!r}. "
            f"test={test!r} would treat those repeated measures as independent. "
            "Use test='wilcoxon', repeated_measures(), or fit_mixed_model()."
        )


def _unpaired_row(work, value, group, a, b, test) -> dict:
    xa = work.loc[work[group] == a, value].to_numpy(dtype=float)
    xb = work.loc[work[group] == b, value].to_numpy(dtype=float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    row = {
        "feature": value,
        "n_a": int(xa.size),
        "n_b": int(xb.size),
        "statistic": np.nan,
        "p_value": np.nan,
        "effect": np.nan,
        "effect_name": "hedges_g",
        "mean_difference_a_minus_b": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "normality_p_a": _shapiro(xa),
        "normality_p_b": _shapiro(xb),
    }
    if xa.size < 2 or xb.size < 2:
        raise DesignError(
            f"{value!r} has {int(xa.size)} finite observations in {a!r} and "
            f"{int(xb.size)} in {b!r}. This test needs at least two in each group. "
            "A missing p-value is not a result."
        )
    if test == "mannwhitney":
        result = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        row["statistic"] = float(result.statistic)
        row["p_value"] = float(result.pvalue)
        row["effect"] = rank_biserial(result.statistic, xa.size, xb.size)
        row["effect_name"] = "rank_biserial"
    elif test == "welch":
        result = stats.ttest_ind(xa, xb, equal_var=False, alternative="two-sided")
        row["statistic"] = float(result.statistic)
        row["p_value"] = float(result.pvalue)
        row["effect"] = hedges_g(xa, xb)
    elif test == "ttest":
        result = stats.ttest_ind(xa, xb, equal_var=True, alternative="two-sided")
        row["statistic"] = float(result.statistic)
        row["p_value"] = float(result.pvalue)
        row["effect"] = hedges_g(xa, xb)
    mean, lo, hi = mean_difference_ci(xa, xb, paired=False, equal_var=(test == "ttest"))
    row["mean_difference_a_minus_b"] = mean
    row["ci_low"] = lo
    row["ci_high"] = hi
    _warn_if_zero_width(value, lo, hi)
    return row


def _wilcoxon_row(work, value, group, subject, a, b) -> dict:
    wide = work.pivot_table(index=subject, columns=group, values=value, aggfunc="mean")
    row = {
        "feature": value,
        "n_a": 0,
        "n_b": 0,
        "statistic": np.nan,
        "p_value": np.nan,
        "effect": np.nan,
        "effect_name": "cohens_dz",
        "mean_difference_a_minus_b": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "normality_p_a": np.nan,
        "normality_p_b": np.nan,
    }
    if a not in wide.columns or b not in wide.columns:
        raise DesignError(
            f"Column {group!r} has no finite rows for both {a!r} and {b!r} on {value!r}."
        )
    paired = wide[[a, b]].dropna()
    row["n_a"] = row["n_b"] = int(len(paired))
    if len(paired) < 2:
        raise DesignError(
            f"{value!r} has {len(paired)} finite within-subject pairs for "
            f"{a!r} versus {b!r}. Wilcoxon needs at least two pairs. "
            "A missing p-value is not a result."
        )
    diff = paired[a].to_numpy() - paired[b].to_numpy()
    # All-zero differences make SciPy's normal approximation divide by zero.
    # The exact probability is still the result when SciPy returns one.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )
        try:
            result = stats.wilcoxon(paired[a], paired[b], alternative="two-sided", zero_method="wilcox")
            row["statistic"] = float(result.statistic)
            row["p_value"] = float(result.pvalue)
        except ValueError:
            pass
    row["effect"] = cohens_dz(diff)
    mean, lo, hi = mean_difference_ci(paired[a].to_numpy(), paired[b].to_numpy(), paired=True)
    row["mean_difference_a_minus_b"] = mean
    row["ci_low"] = lo
    row["ci_high"] = hi
    row["normality_p_a"] = _shapiro(diff)
    _warn_if_zero_width(value, lo, hi)
    return row


def friedman_test(frame: pd.DataFrame, value: str, condition: str, subject: str) -> dict:
    """Friedman test on subject means. Subjects missing a condition are dropped."""
    means = subject_means(frame, [value], subject, by=[condition])
    wide = means.pivot(index=subject, columns=condition, values=value).dropna()
    if wide.shape[1] < 3 or wide.shape[0] < 2:
        raise DesignError("Friedman requires at least 3 conditions and 2 complete subjects.")
    result = stats.friedmanchisquare(*[wide[col].to_numpy() for col in wide.columns])
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_subjects": int(wide.shape[0]),
        "conditions": list(wide.columns),
        "note": "Subjects with a missing condition were excluded. This is an omnibus test, not a pairwise contrast.",
    }


def correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "pearson",
    alpha: float = 0.05,
    n_parameters: int = 0,
) -> dict:
    """Pearson or Spearman association, with a Fisher interval for Pearson's r.

    ``n_parameters`` is the number of covariates already removed, as in a
    partial correlation. The Pearson p-value then uses ``n - n_parameters - 2``
    degrees of freedom, and the Fisher interval uses
    ``n - n_parameters - 3``. The interval is omitted when that remainder
    is not positive.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if n_parameters < 0:
        raise ParameterError("n_parameters must be non-negative.")
    if x.size < 3:
        raise DesignError("Correlation needs at least 3 finite pairs.")
    if method == "pearson":
        result = stats.pearsonr(x, y)
        r = float(result.statistic)
        df = int(x.size - 2 - n_parameters)
        if n_parameters == 0:
            p_value = float(result.pvalue)
        elif df < 1 or not np.isfinite(r):
            p_value = np.nan
        elif abs(r) >= 1.0:
            p_value = 0.0
        else:
            t_stat = r * np.sqrt(df / (1.0 - r * r))
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        z_df = x.size - 3 - n_parameters
        if z_df <= 0 or not np.isfinite(r):
            lo = hi = np.nan
            note = "Fisher z interval needs n - n_parameters - 3 > 0"
        else:
            z = np.arctanh(np.clip(r, -0.999999, 0.999999))
            se = 1.0 / np.sqrt(z_df)
            crit = stats.norm.ppf(1 - alpha / 2)
            lo, hi = np.tanh(z - crit * se), np.tanh(z + crit * se)
            note = "Fisher z interval" if n_parameters == 0 else "Fisher z interval with n - n_covariates - 3"
    elif method == "spearman":
        if n_parameters:
            raise ParameterError("n_parameters applies to Pearson correlation only.")
        result = stats.spearmanr(x, y)
        r = float(result.statistic)
        p_value = float(result.pvalue)
        lo = hi = np.nan
        note = "not computed for Spearman"
    else:
        raise ParameterError("method must be 'pearson' or 'spearman'.")
    return {
        "method": method,
        "r": r,
        "p_value": p_value,
        "n": int(x.size),
        "ci_low": float(lo) if np.isfinite(lo) else np.nan,
        "ci_high": float(hi) if np.isfinite(hi) else np.nan,
        "ci_note": note,
    }


def partial_correlation(frame: pd.DataFrame, x: str, y: str, covariates: Sequence[str]) -> dict:
    """Pearson correlation of residuals after ordinary least-squares on the covariates.

    The p-value uses ``n - k - 2`` degrees of freedom and the Fisher
    interval uses ``n - k - 3``, where ``k`` is the number of covariates.
    Rows are treated as independent.
    """
    import statsmodels.api as sm

    cols = [x, y, *covariates]
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise DesignError(f"Columns not in the table: {missing}.")
    work = frame[list(cols)].dropna()
    k = len(list(covariates))
    if len(work) < k + 4:
        raise DesignError("Not enough complete rows for this partial correlation.")
    design = sm.add_constant(work[list(covariates)], has_constant="add")
    rx = sm.OLS(work[x], design).fit().resid
    ry = sm.OLS(work[y], design).fit().resid
    out = correlation(rx, ry, method="pearson", n_parameters=k)
    out["covariates"] = list(covariates)
    out["method"] = "partial_pearson"
    out["df"] = int(len(work) - k - 2)
    return out


def permutation_test(
    frame: pd.DataFrame,
    value: str,
    condition: str,
    subject: str,
    a: str,
    b: str,
    *,
    n_perm: int = 999,
    seed: int = 0,
) -> dict:
    """Permutation test of the mean difference, using subjects as the exchangeable units.

    Technical replicates are averaged first. If every analysed subject has
    both conditions, signs of the within-subject differences are flipped.
    If the groups are disjoint, condition labels are shuffled across
    subjects.     Mixed overlap is rejected.
    """
    if n_perm < 1:
        raise ParameterError("n_perm must be at least 1.")
    means = subject_means(frame, [value], subject, by=[condition])
    means = means.loc[means[condition].isin([a, b])]
    wide = means.pivot(index=subject, columns=condition, values=value)
    if a not in wide.columns or b not in wide.columns:
        raise DesignError(f"Conditions {a!r} and {b!r} are not both present.")
    both = wide[[a, b]].dropna()
    only_a = wide.loc[wide[b].isna() & wide[a].notna(), a]
    only_b = wide.loc[wide[a].isna() & wide[b].notna(), b]
    rng = np.random.default_rng(seed)
    if len(both) and (len(only_a) or len(only_b)):
        raise DesignError(
            "Some subjects are paired and some are not. Split the design or "
            "drop the incomplete subjects explicitly before permuting."
        )
    if len(both):
        diff = both[a].to_numpy() - both[b].to_numpy()
        observed = float(np.mean(diff))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size))
        null = (signs * diff).mean(axis=1)
        design = "within_subject_sign_flip"
        n_subjects = int(diff.size)
    else:
        xa = only_a.to_numpy(dtype=float)
        xb = only_b.to_numpy(dtype=float)
        if xa.size < 2 or xb.size < 2:
            raise DesignError("Each unpaired group needs at least 2 subjects.")
        observed = float(np.mean(xa) - np.mean(xb))
        labels = np.array([0] * xa.size + [1] * xb.size)
        pooled = np.concatenate([xa, xb])
        null = np.empty(n_perm)
        for i in range(n_perm):
            rng.shuffle(labels)
            null[i] = np.mean(pooled[labels == 0]) - np.mean(pooled[labels == 1])
        design = "between_subject_label_shuffle"
        n_subjects = int(xa.size + xb.size)
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1))
    return {
        "statistic": observed,
        "p_value": p,
        "n_perm": n_perm,
        "n_subjects": n_subjects,
        "design": design,
        "null_note": "The p-value includes the observed statistic in the numerator and denominator.",
    }


def bootstrap_ci(
    x: np.ndarray,
    statistic=np.mean,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap interval. The draw is i.i.d. resampling of rows."""
    if n_boot < 1:
        raise ParameterError("n_boot must be at least 1.")
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise DesignError("Bootstrap needs at least 2 finite observations.")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = statistic(x[rng.integers(0, x.size, x.size)])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(statistic(x)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(x.size),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def cluster_permutation_1d(
    differences: np.ndarray,
    *,
    alpha: float = 0.05,
    cluster_alpha: float = 0.05,
    n_perm: int = 499,
    seed: int = 0,
) -> pd.DataFrame:
    """One-sample cluster permutation test on subject-level differences.

    ``differences`` has shape ``(n_subjects, n_times)``. The statistic at
    each time is a one-sample t-test against 0. Clusters are contiguous
    samples of one sign whose absolute t exceeds the two-sided critical
    value at ``cluster_alpha``. That threshold decides which samples can
    form a cluster. ``alpha`` is only the significance level for the
    cluster p-value. Cluster mass is the sum of absolute t values.
    Exchangeability under the null is a sign flip within subject
    (Maris and Oostenveld, 2007). Epochs must already have been reduced
    to one series per subject. Positive and negative clusters are formed
    separately: a run of large positive t is not merged with an adjacent
    run of large negative t.
    """
    if n_perm < 1:
        raise ParameterError("n_perm must be at least 1.")
    if not 0 < alpha < 1 or not 0 < cluster_alpha < 1:
        raise ParameterError("alpha and cluster_alpha must be in (0, 1).")
    x = np.asarray(differences, dtype=float)
    if x.ndim != 2:
        raise ParameterError("differences must have shape (n_subjects, n_times).")
    if not np.isfinite(x).all():
        raise ParameterError("differences must be finite.")
    n_subj = int(x.shape[0])
    if n_subj < 3:
        raise DesignError("Cluster permutation needs at least 3 subjects.")
    df = n_subj - 1
    threshold = float(stats.t.ppf(1 - cluster_alpha / 2, df))

    def t_stat(sample: np.ndarray) -> np.ndarray:
        mean = np.mean(sample, axis=0)
        se = np.std(sample, axis=0, ddof=1) / np.sqrt(n_subj)
        t = np.zeros_like(mean)
        ok = se > 0
        t[ok] = mean[ok] / se[ok]
        # A nonzero mean with zero dispersion is an infinite t, not a null t.
        t[(~ok) & (mean > 0)] = np.inf
        t[(~ok) & (mean < 0)] = -np.inf
        return t

    def clusters(t_values: np.ndarray) -> list[tuple[int, int, float]]:
        found: list[tuple[int, int, float]] = []
        for direction in (1.0, -1.0):
            idx = np.flatnonzero(direction * t_values >= threshold)
            if idx.size == 0:
                continue
            splits = np.flatnonzero(np.diff(idx) > 1) + 1
            for group in np.split(idx, splits):
                start = int(group[0])
                end = int(group[-1])
                mass = float(np.sum(np.abs(t_values[start : end + 1])))
                found.append((start, end, mass))
        found.sort(key=lambda item: (item[0], item[1]))
        return found

    def max_mass(t_values: np.ndarray) -> float:
        found = clusters(t_values)
        return max((mass for _, _, mass in found), default=0.0)

    observed_t = t_stat(x)
    observed = clusters(observed_t)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n_subj))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = max_mass(t_stat(x * signs[i][:, None]))
    rows = []
    for start, end, mass in observed:
        p = float((1 + np.sum(null >= mass)) / (n_perm + 1))
        rows.append(
            {
                "start": start,
                "end": end,
                "mass": mass,
                "p_value": p,
                "significant": p <= alpha,
                "threshold_t": threshold,
                "n_subjects": n_subj,
                "n_perm": n_perm,
            }
        )
    return pd.DataFrame(rows)


def mean_difference_posterior(a: np.ndarray, b: np.ndarray | None = None, *, paired: bool = False) -> dict:
    """Student-t posterior for a mean difference under a flat mean prior.

    For paired data, pass the two aligned samples or pass differences as
    ``a`` with ``b=None``. For unpaired data the scale is the Welch
    standard error and the degrees of freedom are Satterthwaite's. That
    is an approximation to the one-sample reference-prior posterior, not
    that posterior itself. The returned probability is the posterior
    probability that
    the mean of ``a`` exceeds the mean of ``b``. This is not a Bayes factor.
    """
    a = np.asarray(a, dtype=float)
    if paired or b is None:
        d = a if b is None else a - np.asarray(b, dtype=float)
        d = d[np.isfinite(d)]
        n = d.size
        if n < 2:
            raise DesignError("The posterior needs at least 2 differences.")
        mean = float(np.mean(d))
        se = float(np.std(d, ddof=1) / np.sqrt(n))
        df = n - 1
        assumption = "Normal observations, flat prior on the mean, reference prior on the variance."
    else:
        b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            raise DesignError("Each unpaired sample needs at least 2 finite observations.")
        mean = float(np.mean(a) - np.mean(b))
        va = np.var(a, ddof=1) / a.size
        vb = np.var(b, ddof=1) / b.size
        se = float(np.sqrt(va + vb))
        denom = va**2 / (a.size - 1) + vb**2 / (b.size - 1)
        df = float((va + vb) ** 2 / denom) if denom > 0 else np.nan
        n = int(a.size + b.size)
        assumption = (
            "Approximate posterior using the Welch–Satterthwaite scale. "
            "The one-sample reference-prior result is the exact Student-t posterior."
        )
    if se == 0:
        prob = 1.0 if mean > 0 else (0.0 if mean < 0 else 0.5)
        return {
            "mean": mean,
            "ci_low": mean,
            "ci_high": mean,
            "prob_mean_gt_0": prob,
            "df": df,
            "n": n,
            "assumption": assumption,
        }
    dist = stats.t(df, loc=mean, scale=se)
    lo, hi = dist.ppf([0.025, 0.975])
    return {
        "mean": mean,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "prob_mean_gt_0": float(1.0 - dist.cdf(0.0)),
        "df": float(df),
        "n": n,
        "assumption": assumption,
    }


def bic_bayes_factor(frame: pd.DataFrame, y: str, condition: str) -> dict:
    """BIC approximation of the Bayes factor for a condition mean structure.

    BF10 approximates the evidence for ``y ~ C(condition)`` against ``y ~ 1``
    (Wagenmakers, 2007). Rows are treated as independent. Do not pass
    technical replicates from the same subject as if they were subjects.
    """
    import statsmodels.formula.api as smf

    work = frame[[y, condition]].dropna()
    if work[condition].nunique() < 2 or len(work) < 4:
        raise DesignError("The Bayes-factor comparison needs two conditions and at least 4 rows.")
    outcome = "Q(" + repr(str(y)) + ")"
    factor = "Q(" + repr(str(condition)) + ")"
    full = smf.ols(f"{outcome} ~ C({factor})", work).fit()
    reduced = smf.ols(f"{outcome} ~ 1", work).fit()
    log_bf10 = (float(reduced.bic) - float(full.bic)) / 2.0
    bf10 = float("inf") if log_bf10 > 700 else float(np.exp(log_bf10))
    return {
        "bf10": bf10,
        "bic_full": float(full.bic),
        "bic_reduced": float(reduced.bic),
        "interpretation": "BF10 > 1 favours a condition difference over a common mean. This is a BIC approximation, not a JZS Bayes factor.",
    }
