"""Concordance (prediction probability).

Smith, Dutton, and Smith (1996) defined Pk as the probability that a
randomly chosen observation from one condition is ordered as expected
relative to a randomly chosen observation from the other, with ties
counting as one half. For two independent samples this is the
Mann–Whitney/ROC area. For paired samples it is the proportion of
subjects whose within-subject difference has the expected sign. Those
two numbers answer different questions and are both returned with an
``estimator`` label so they are not pooled.

The column names ``group_low`` and ``group_high`` only fix the direction:
Pk > 0.5 means the values are larger in ``group_high``. Swapping the
labels replaces Pk with 1 − Pk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from eegmath.errors import DesignError

TIE_ATOL = 0.0
TIE_RTOL = 1e-9


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def pk_from_arrays(low: np.ndarray, high: np.ndarray, *, atol: float = TIE_ATOL, rtol: float = TIE_RTOL) -> tuple[float, int, int]:
    """Directional Pk for two unpaired samples. Returns ``(pk, n_pairs, n_ties)``."""
    low = _finite(low)
    high = _finite(high)
    n_pairs = int(low.size * high.size)
    if n_pairs == 0:
        return (np.nan, 0, 0)
    diff = high[None, :] - low[:, None]
    scale = np.maximum(np.abs(high)[None, :], np.abs(low)[:, None])
    ties = np.abs(diff) <= (atol + rtol * scale)
    n_ties = int(ties.sum())
    n_conc = int(((diff > 0) & ~ties).sum())
    return (float((n_conc + 0.5 * n_ties) / n_pairs), n_pairs, n_ties)


def pk_from_differences(diffs: np.ndarray, scale: np.ndarray | None = None, *, atol: float = TIE_ATOL, rtol: float = TIE_RTOL) -> tuple[float, int]:
    diffs = np.asarray(diffs, dtype=float)
    keep = np.isfinite(diffs)
    diffs = diffs[keep]
    if diffs.size == 0:
        return (np.nan, 0)
    if scale is None:
        scale_arr = np.maximum(np.abs(diffs), 1.0)
    else:
        scale_arr = np.asarray(scale, dtype=float)[keep]
    ties = np.abs(diffs) <= (atol + rtol * scale_arr)
    n_pos = int(((diffs > 0) & ~ties).sum())
    n_ties = int(ties.sum())
    return (float((n_pos + 0.5 * n_ties) / diffs.size), n_ties)


def _jackknife_se(estimate: float, leave_one_out: list[float]) -> float:
    values = np.asarray(leave_one_out, dtype=float)
    # Dropping a non-finite replicate would change the sample size inside
    # the pseudo-value formula, so the standard error is withheld instead.
    if values.size < 3 or not np.isfinite(estimate) or not np.isfinite(values).all():
        return np.nan
    n = int(values.size)
    pseudo = n * estimate - (n - 1) * values
    return float(np.std(pseudo, ddof=1) / np.sqrt(n))


@dataclass
class ConcordanceResult:
    feature: str
    group_low: str
    group_high: str
    pk: float
    n_pairs: int
    strategy: str
    estimator: str
    pk_abs: float = np.nan
    direction: int = 0
    se_jackknife: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    p_value: float = np.nan
    n_ties: int = 0
    n_low: int = 0
    n_high: int = 0
    p_value_test: str = ""

    def __post_init__(self) -> None:
        if np.isfinite(self.pk):
            self.pk_abs = float(max(self.pk, 1.0 - self.pk))
            self.direction = 1 if self.pk > 0.5 else (-1 if self.pk < 0.5 else 0)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "group_low": self.group_low,
            "group_high": self.group_high,
            "pk": self.pk,
            "pk_abs": self.pk_abs,
            "direction": self.direction,
            "n_pairs": self.n_pairs,
            "n_ties": self.n_ties,
            "n_low": self.n_low,
            "n_high": self.n_high,
            "se_jackknife": self.se_jackknife,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "p_value_test": self.p_value_test,
            "strategy": self.strategy,
            "estimator": self.estimator,
        }


def _prepare(frame: pd.DataFrame, feature: str, group: str, subject: str, low: str, high: str) -> pd.DataFrame:
    need = [feature, group, subject]
    missing = [col for col in need if col not in frame.columns]
    if missing:
        raise DesignError(f"Concordance table is missing columns {missing}.")
    sub = frame.loc[frame[group].isin([low, high]), need].dropna()
    if sub.empty:
        return sub
    return sub.groupby([subject, group], as_index=False)[feature].mean()


def prediction_probability(
    frame: pd.DataFrame,
    feature: str,
    group: str,
    subject: str,
    group_low: str,
    group_high: str,
    *,
    strategy: str | None = None,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> ConcordanceResult:
    """Pk for one feature and two groups.

    Rows are averaged within subject and group before the statistic is
    computed. ``strategy=None`` selects the paired sign concordance when
    at least three subjects have both groups. Subjects who have only one
    of the two groups are not included in that paired estimate.
    Otherwise it selects the unpaired Mann–Whitney concordance on subjects
    who contribute only one of the two groups. ``strategy="paired"`` with
    fewer than three paired subjects raises :class:`~eegmath.errors.DesignError`
    instead of switching estimators.
    """
    agg = _prepare(frame, feature, group, subject, group_low, group_high)
    empty = ConcordanceResult(
        feature, group_low, group_high, np.nan, 0, "paired",
        "paired within-subject sign concordance (not the ROC area)",
    )
    if agg.empty:
        return empty
    counts = agg.groupby(subject)[group].nunique()
    paired_ids = sorted(counts[counts > 1].index.tolist())
    if strategy is None:
        strategy = "paired" if len(paired_ids) >= 3 else "unpaired"
    if strategy not in {"paired", "unpaired"}:
        raise DesignError("strategy must be 'paired', 'unpaired', or None.")
    if strategy == "paired":
        if len(paired_ids) < 3:
            raise DesignError(
                "strategy='paired' needs at least 3 subjects who have both groups. "
                f"Found {len(paired_ids)}."
            )
        return _paired(agg, feature, group, subject, group_low, group_high, paired_ids, n_bootstrap, seed)
    unpaired = agg.loc[~agg[subject].isin(paired_ids)] if strategy == "unpaired" else agg
    # When the caller forces unpaired, still use every subject once; if a
    # subject has both groups those two means are not independent, so drop them.
    if strategy == "unpaired":
        both = set(paired_ids)
        unpaired = agg.loc[~agg[subject].isin(both)]
        if unpaired.empty or unpaired[group].nunique() < 2:
            raise DesignError(
                "strategy='unpaired' needs subjects who contribute only one of the two groups. "
                f"{len(paired_ids)} subjects have both groups, and those two means are not an unpaired sample."
            )
    return _unpaired(unpaired, feature, group, subject, group_low, group_high, n_bootstrap, seed)


def _unpaired(agg, feature, group, subject, low, high, n_bootstrap, seed) -> ConcordanceResult:
    low_rows = agg.loc[agg[group] == low]
    high_rows = agg.loc[agg[group] == high]
    low_vals = low_rows[feature].to_numpy(dtype=float)
    high_vals = high_rows[feature].to_numpy(dtype=float)
    pk, n_pairs, n_ties = pk_from_arrays(low_vals, high_vals)
    label = "unpaired Mann-Whitney concordance (ROC area on subject means)"
    result = ConcordanceResult(
        feature, low, high, pk, n_pairs, "unpaired", label,
        n_ties=n_ties, n_low=int(low_vals.size), n_high=int(high_vals.size),
    )
    result.p_value_test = "two-sided Mann-Whitney test of the subject means"
    if low_vals.size >= 1 and high_vals.size >= 1 and low_vals.size + high_vals.size >= 3:
        try:
            result.p_value = float(stats.mannwhitneyu(high_vals, low_vals, alternative="two-sided").pvalue)
        except ValueError:
            result.p_value = np.nan
    loo = []
    ids = list(low_rows[subject]) + list(high_rows[subject])
    if len(set(ids)) >= 3:
        for sid in sorted(set(ids)):
            pk_i, _, _ = pk_from_arrays(
                low_rows.loc[low_rows[subject] != sid, feature],
                high_rows.loc[high_rows[subject] != sid, feature],
            )
            loo.append(pk_i)
        result.se_jackknife = _jackknife_se(pk, loo)
    if n_bootstrap > 0 and low_vals.size and high_vals.size:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_bootstrap):
            a = low_vals[rng.integers(0, low_vals.size, low_vals.size)]
            b = high_vals[rng.integers(0, high_vals.size, high_vals.size)]
            draw, _, _ = pk_from_arrays(a, b)
            draws.append(draw)
        result.ci_low, result.ci_high = np.quantile(draws, [0.025, 0.975])
    return result


def _paired(agg, feature, group, subject, low, high, paired_ids, n_bootstrap, seed) -> ConcordanceResult:
    if len(paired_ids) < 3:
        raise DesignError("Paired concordance needs at least 3 subjects who have both groups.")
    wide = agg.pivot(index=subject, columns=group, values=feature).loc[paired_ids]
    lows = wide[low].to_numpy(dtype=float)
    highs = wide[high].to_numpy(dtype=float)
    diffs = highs - lows
    scale = np.maximum(np.abs(lows), np.abs(highs))
    pk, n_ties = pk_from_differences(diffs, scale)
    label = "paired within-subject sign concordance (not the ROC area)"
    result = ConcordanceResult(
        feature, low, high, pk, int(np.isfinite(diffs).sum()), "paired", label,
        n_ties=n_ties, n_low=int(np.isfinite(lows).sum()), n_high=int(np.isfinite(highs).sum()),
    )
    # Pk is a sign probability. Wilcoxon uses the ranks of the
    # differences, so one large step can hide a consistent sign.
    tol = TIE_ATOL + TIE_RTOL * scale
    finite = np.isfinite(diffs)
    tied = finite & (np.abs(diffs) <= tol)
    n_pos = int(np.count_nonzero(finite & ~tied & (diffs > 0)))
    n_nonzero = n_pos + int(np.count_nonzero(finite & ~tied & (diffs < 0)))
    result.p_value_test = "two-sided sign test of non-tied within-subject differences"
    if n_nonzero >= 1:
        result.p_value = float(stats.binomtest(n_pos, n_nonzero, p=0.5, alternative="two-sided").pvalue)
    loo = []
    for i in range(diffs.size):
        pk_i, _ = pk_from_differences(np.delete(diffs, i), np.delete(scale, i))
        loo.append(pk_i)
    result.se_jackknife = _jackknife_se(pk, loo)
    if n_bootstrap > 0 and diffs.size:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, diffs.size, diffs.size)
            draw, _ = pk_from_differences(diffs[idx], scale[idx])
            draws.append(draw)
        result.ci_low, result.ci_high = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
    return result


def batch_concordance(
    frame: pd.DataFrame,
    features: list[str],
    pairs: list[tuple[str, str]],
    *,
    group: str = "condition",
    subject: str = "subject_id",
    n_bootstrap: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        for low, high in pairs:
            rows.append(
                prediction_probability(
                    frame, feature, group, subject, low, high,
                    n_bootstrap=n_bootstrap, seed=seed,
                ).to_dict()
            )
    return pd.DataFrame(rows)
