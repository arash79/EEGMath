"""Multiple-comparison corrections.

The family is whatever set of p-values the caller passes. These functions
do not look at other tests that might have been run elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegmath.errors import ParameterError


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg adjusted p-values (Benjamini and Hochberg, 1995)."""
    return _fdr(p_values, alpha, method="bh")


def fdr_by(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Yekutieli adjustment for arbitrarily dependent tests."""
    return _fdr(p_values, alpha, method="by")


def _as_pvalues(p_values: np.ndarray) -> np.ndarray:
    p = np.atleast_1d(np.asarray(p_values, dtype=float))
    if np.any(np.isfinite(p) & ((p < 0.0) | (p > 1.0))):
        raise ParameterError("Finite p-values must lie in [0, 1].")
    return p


def _fdr(p_values: np.ndarray, alpha: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    p = _as_pvalues(p_values)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    # Non-finite p-values are not tests. They stay missing and are not
    # counted in the family, rather than being rewritten as 1.
    valid = np.isfinite(p)
    adjusted = np.full(n, np.nan)
    rejected = np.zeros(n, dtype=bool)
    if not valid.any():
        return rejected, adjusted
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, pv.size + 1, dtype=float)
    harmonic = float(np.sum(1.0 / ranks)) if method == "by" else 1.0
    raw = ranked * pv.size * harmonic / ranks
    raw = np.minimum(1.0, raw)
    for i in range(pv.size - 2, -1, -1):
        raw[i] = min(raw[i], raw[i + 1])
    placed = np.empty(pv.size)
    placed[order] = raw
    adjusted[valid] = placed
    rejected[valid] = placed <= alpha
    return rejected, adjusted


def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    p = _as_pvalues(p_values)
    finite = np.isfinite(p)
    n = int(finite.sum())
    adjusted = np.full(p.shape, np.nan, dtype=float)
    adjusted[finite] = np.minimum(1.0, p[finite] * max(n, 1))
    rejected = np.zeros(p.shape, dtype=bool)
    rejected[finite] = adjusted[finite] <= alpha
    return rejected, adjusted


def holm(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjusted p-values (Holm, 1979)."""
    p = _as_pvalues(p_values)
    n = p.size
    adjusted = np.full(n, np.nan)
    rejected = np.zeros(n, dtype=bool)
    valid = np.isfinite(p)
    if not valid.any():
        return rejected, adjusted
    pv = p[valid]
    order = np.argsort(pv)
    m = pv.size
    ranked = pv[order]
    running = 0.0
    adj_ranked = np.empty(m)
    for i, value in enumerate(ranked):
        running = max(running, (m - i) * value)
        adj_ranked[i] = min(1.0, running)
    placed = np.empty(m)
    placed[order] = adj_ranked
    adjusted[valid] = placed
    rejected[valid] = placed <= alpha
    return rejected, adjusted


def adjust_pvalues(p_values: np.ndarray, method: str = "fdr_bh", alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    methods = {
        "fdr_bh": fdr_bh,
        "fdr_by": fdr_by,
        "bonferroni": bonferroni,
        "holm": holm,
    }
    if method not in methods:
        raise ParameterError(f"Unknown correction '{method}'. Choose from {sorted(methods)}.")
    return methods[method](p_values, alpha=alpha)


def apply_correction(
    frame: pd.DataFrame,
    p_col: str = "p_value",
    *,
    method: str = "fdr_bh",
    alpha: float = 0.05,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add ``p_corrected`` and ``significant``. Groups are separate families."""
    out = frame.copy()
    out["p_corrected"] = np.nan
    out["significant"] = False

    def _one(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        rejected, corrected = adjust_pvalues(sub[p_col].to_numpy(), method=method, alpha=alpha)
        sub["p_corrected"] = corrected
        sub["significant"] = rejected & np.isfinite(corrected)
        return sub

    if group_col and group_col in out.columns:
        parts = [_one(sub) for _, sub in out.groupby(group_col, dropna=False, sort=False)]
        return pd.concat(parts, axis=0).sort_index()
    return _one(out)
