"""Ordinary least squares for tables of independent rows.

This is the between-unit linear model. Repeated rows from the same
subject are not a nuisance to be averaged silently and they are not
independent observations. Pass them to :func:`eegmath.stats.repeated_measures`
or average them before calling :func:`fit_ols`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from eegmath.errors import DesignError


@dataclass
class OLSResult:
    formula: str
    n_obs: int
    df_resid: float
    r_squared: float
    coefficients: pd.DataFrame
    note: str = (
        "Rows are treated as independent. Intervals use the residual "
        "degrees of freedom of the ordinary least-squares fit."
    )

    def to_frame(self) -> pd.DataFrame:
        return self.coefficients.copy()


def fit_ols(frame: pd.DataFrame, formula: str, *, subject: str | None = None) -> OLSResult:
    """Fit ``formula`` by ordinary least squares.

    ``subject`` is optional. When it is given, a missing identifier or a
    repeated row for any subject raises :class:`~eegmath.errors.DesignError`
    instead of being entered as another independent observation.
    """
    import statsmodels.formula.api as smf

    if subject is not None:
        if subject not in frame.columns:
            raise DesignError(f"Subject column {subject!r} is not in the table.")
        n_missing = int(frame[subject].isna().sum())
        if n_missing:
            raise DesignError(
                f"{n_missing} rows have no value in {subject!r}. "
                "Assign an identifier or drop those rows before fitting."
            )
        counts = frame.groupby(subject, dropna=False).size()
        if (counts > 1).any():
            raise DesignError(
                "fit_ols found more than one row for a subject. "
                "Average to one row per subject, or use repeated_measures() "
                "if those rows are repeated measures. This fit treats every row as independent."
            )
    try:
        fit = smf.ols(formula, data=frame).fit()
    except Exception as exc:
        raise DesignError(f"OLS model failed to fit: {exc}") from exc
    confidence = fit.conf_int()
    rows = []
    for name in fit.params.index:
        rows.append(
            {
                "term": str(name),
                "estimate": float(fit.params[name]),
                "se": float(fit.bse[name]),
                "statistic": float(fit.tvalues[name]),
                "df": float(fit.df_resid),
                "p_value": float(fit.pvalues[name]),
                "ci_low": float(confidence.loc[name, 0]),
                "ci_high": float(confidence.loc[name, 1]),
            }
        )
    return OLSResult(
        formula=formula,
        n_obs=int(fit.nobs),
        df_resid=float(fit.df_resid),
        r_squared=float(fit.rsquared),
        coefficients=pd.DataFrame(rows),
    )
