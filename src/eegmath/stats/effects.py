"""Effect sizes.

Cohen's d uses the pooled standard deviation. Hedges' g multiplies d by
the small-sample factor ``1 - 3 / (4*(n1+n2) - 9)``. Paired d_z divides
the mean difference by the standard deviation of the differences.
Rank-biserial correlation is ``2U/(n1 n2) - 1`` with U from
``mannwhitneyu(a, b)``, so a positive value means ``a`` tends to be larger.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    d = cohens_d(a, b)
    n = np.isfinite(a).sum() + np.isfinite(b).sum()
    if n <= 2 or not np.isfinite(d):
        return d
    return float(d * (1.0 - 3.0 / (4.0 * n - 9.0)))


def cohens_dz(differences: np.ndarray) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2 or np.std(d, ddof=1) == 0:
        return np.nan
    return float(np.mean(d) / np.std(d, ddof=1))


def rank_biserial(u: float, n_a: int, n_b: int) -> float:
    if n_a < 1 or n_b < 1:
        return np.nan
    return float((2.0 * u) / (n_a * n_b) - 1.0)


def mean_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    paired: bool = False,
    equal_var: bool = False,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean of ``a - b`` and a Student-t confidence interval.

    Unpaired intervals use the Welch–Satterthwaite standard error unless
    ``equal_var=True``, which uses the pooled variance and
    ``n1 + n2 - 2`` degrees of freedom.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if paired:
        if a.shape != b.shape:
            raise ValueError("Paired samples must have the same shape.")
        d = a - b
        d = d[np.isfinite(d)]
        n = d.size
        if n < 2:
            return (np.nan, np.nan, np.nan)
        mean = float(np.mean(d))
        se = float(np.std(d, ddof=1) / np.sqrt(n))
        df = n - 1
    else:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            return (np.nan, np.nan, np.nan)
        mean = float(np.mean(a) - np.mean(b))
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        if equal_var:
            pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
            se = float(np.sqrt(pooled * (1.0 / a.size + 1.0 / b.size)))
            df = float(a.size + b.size - 2)
        else:
            se = float(np.sqrt(var_a / a.size + var_b / b.size))
            va = var_a / a.size
            vb = var_b / b.size
            denom = va**2 / (a.size - 1) + vb**2 / (b.size - 1)
            df = float((va + vb) ** 2 / denom) if denom > 0 else np.nan
    if se == 0:
        return (mean, mean, mean)
    if not np.isfinite(df) or df <= 0:
        return (mean, np.nan, np.nan)
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return (mean, mean - crit * se, mean + crit * se)
