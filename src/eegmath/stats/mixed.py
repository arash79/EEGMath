"""Linear mixed models for repeated measures.

``repeated_measures`` averages technical replicates to one row per
subject and condition before fitting, unless ``aggregate=False``.
The model is a random-intercept model

``value ~ C(condition) + covariates + (1 | subject)``.

Fixed-effect p-values use a Student-t reference with degrees of freedom
``n_subjects - n_fixed_effects`` for every coefficient, on the unadjusted
REML standard error. That subtracts the whole fixed design from the
subject count. The interval is wider than a normal interval with the
same standard error. It is not a Kenward–Roger correction, not a
Satterthwaite approximation, and not the between-within split, which
would give a within-subject contrast a larger denominator. When that
difference is not positive, or the REML fit did not converge, the
coefficients remain and the t interval and p-value are withheld. They
are not recomputed at one degree of freedom. The omnibus likelihood-ratio test refits both models
by maximum likelihood; the reported coefficients stay on the REML fit.
A random intercept does not model temporal autocorrelation among
windows, which is why the helper aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from eegmath.errors import DependencyError, DesignError


def _q(name: str) -> str:
    """Quote a column name for a patsy formula."""
    return "Q(" + repr(str(name)) + ")"


def _statsmodels():
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover
        raise DependencyError("statsmodels is required for mixed models.") from exc
    return smf


@dataclass
class MixedModelResult:
    formula: str
    group: str
    converged: bool
    n_obs: int
    n_subjects: int
    dof: float
    coefficients: pd.DataFrame
    omnibus: dict[str, Any] = field(default_factory=dict)
    contrasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    aic: float = np.nan
    bic: float = np.nan
    fit: Any = None
    note: str = (
        "Fixed-effect intervals use n_subjects - n_fixed_effects degrees of "
        "freedom for every coefficient, on the unadjusted REML standard "
        "error. That reference is wider than a normal interval with the "
        "same standard error. It is not Kenward-Roger, not Satterthwaite, "
        "and not a between-within split. It is not a substitute for a "
        "pre-registered contrast. A non-positive value withholds the "
        "interval and the p-value."
    )

    def to_frame(self) -> pd.DataFrame:
        return self.coefficients.copy()


def _complete_cases(frame: pd.DataFrame, formula: str, group: str) -> pd.DataFrame:
    """Rows with complete formula columns and a grouping label.

    Columns that are not in the formula are not a reason to drop a row.
    """
    import patsy

    if group not in frame.columns:
        raise DesignError(f"Grouping column {group!r} is not in the table.")
    indexed = frame.reset_index(drop=True)
    try:
        response, _design = patsy.dmatrices(formula, indexed, return_type="dataframe", NA_action="drop")
    except Exception as exc:
        raise DesignError(f"Could not build the model matrix: {exc}") from exc
    work = indexed.loc[response.index].dropna(subset=[group]).reset_index(drop=True)
    return work


def fit_mixed_model(
    frame: pd.DataFrame,
    formula: str,
    group: str,
    *,
    reml: bool = True,
) -> MixedModelResult:
    """Fit ``formula`` with a random intercept for ``group``."""
    smf = _statsmodels()
    work = _complete_cases(frame, formula, group)
    if work[group].nunique() < 3 or len(work) < 6:
        raise DesignError("A mixed model here needs at least 3 subjects and 6 rows.")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = smf.mixedlm(formula, work, groups=work[group], re_formula="~1").fit(
                reml=reml, method="lbfgs"
            )
        except Exception as exc:
            raise DesignError(f"Mixed model failed to fit: {exc}") from exc
    n_subjects = int(work[group].nunique())
    n_fe = int(len(fit.fe_params))
    # Do not replace a non-positive difference with 1. That would still
    # be a t test, and it is not the degrees of freedom stated above.
    dof = float(n_subjects - n_fe)
    converged = bool(fit.converged)
    infer = dof >= 1 and converged
    rows = []
    crit = float(stats.t.ppf(0.975, dof)) if infer else np.nan
    for name in fit.fe_params.index:
        estimate = float(fit.fe_params[name])
        se = float(fit.bse_fe[name])
        if not infer or not np.isfinite(se) or se <= 0 or not np.isfinite(crit):
            t_stat = p = lo = hi = np.nan
        else:
            t_stat = estimate / se
            p = float(2 * stats.t.sf(abs(t_stat), dof))
            lo, hi = estimate - crit * se, estimate + crit * se
        rows.append(
            {
                "term": name,
                "estimate": estimate,
                "se": se,
                "statistic": t_stat,
                "df": dof,
                "p_value": p,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    method = "REML" if reml else "maximum likelihood"
    note = (
        "Fixed-effect intervals use n_subjects - n_fixed_effects degrees of "
        f"freedom for every coefficient, on the unadjusted {method} standard "
        "error. That reference is wider than a normal interval with the "
        "same standard error. It is not Kenward-Roger, not Satterthwaite, "
        "and not a between-within split. It is not a substitute for a "
        "pre-registered contrast. A non-positive value withholds the "
        "interval and the p-value."
    )
    if dof < 1:
        note += (
            f" Inference was withheld because n_subjects - n_fixed_effects "
            f"is {dof:g}. One degree of freedom was not substituted."
        )
    if not converged:
        note += f" Inference was withheld because the {method} fit did not converge."
    return MixedModelResult(
        formula=formula,
        group=group,
        converged=converged,
        n_obs=int(len(work)),
        n_subjects=n_subjects,
        dof=dof,
        coefficients=pd.DataFrame(rows),
        aic=float(getattr(fit, "aic", np.nan)),
        bic=float(getattr(fit, "bic", np.nan)),
        fit=fit,
        note=note,
    )


def likelihood_ratio(frame: pd.DataFrame, full: str, reduced: str, group: str) -> dict[str, Any]:
    """Maximum-likelihood likelihood-ratio test of two nested random-intercept models."""
    smf = _statsmodels()
    import warnings
    from scipy.stats import chi2

    work = _complete_cases(frame, full, group)
    # The reduced formula can only drop terms, so the same complete rows
    # remain complete for it. Refit both on that row set.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit_full = smf.mixedlm(full, work, groups=work[group], re_formula="~1").fit(reml=False, method="lbfgs")
            fit_reduced = smf.mixedlm(reduced, work, groups=work[group], re_formula="~1").fit(
                reml=False, method="lbfgs"
            )
        except Exception:
            return {"statistic": np.nan, "df": np.nan, "p_value": np.nan, "converged": False}
    if not (fit_full.converged and fit_reduced.converged):
        return {"statistic": np.nan, "df": np.nan, "p_value": np.nan, "converged": False}
    if int(fit_full.nobs) != int(fit_reduced.nobs):
        return {
            "statistic": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "converged": True,
            "note": (
                "The two fits did not use the same rows, so the likelihoods "
                "are not a nested comparison. The p-value was withheld."
            ),
        }
    stat = float(2.0 * (fit_full.llf - fit_reduced.llf))
    df = int(len(fit_full.fe_params) - len(fit_reduced.fe_params))
    note = "Both models were refit with ML because a REML likelihood-ratio test of fixed effects is not valid."
    # A chi-square likelihood-ratio test needs the full model to add fixed
    # effects. Forcing a difference of zero up to 1 degree of freedom
    # reports a test that was not performed.
    if df <= 0:
        return {
            "statistic": stat,
            "df": df,
            "p_value": np.nan,
            "converged": True,
            "note": note + " The full model does not add fixed effects. The p-value was withheld.",
        }
    # A small negative value is optimizer noise. A large one means the
    # "full" model is not a better nested model, so the chi-square tail
    # probability would not be a likelihood-ratio test.
    if stat < -1e-4:
        return {
            "statistic": stat,
            "df": df,
            "p_value": np.nan,
            "converged": True,
            "note": note + " The full-model likelihood was lower. The p-value was withheld.",
        }
    return {
        "statistic": max(stat, 0.0),
        "df": df,
        "p_value": float(chi2.sf(max(stat, 0.0), df)),
        "converged": True,
        "note": note,
    }


def _treatment_term(name: str, condition: str, level: str) -> bool:
    """True when ``name`` is the main-effect treatment term for this level.

    Accepted forms are ``C(condition)[T.level]`` and ``C(Q('condition'))[T.level]``.
    A shorter factor name is not allowed to match inside a longer one, and
    an interaction that merely ends in another factor's level is not a
    main effect.
    """
    text = str(name)
    token = f"[T.{level}]"
    if not text.endswith(token):
        return False
    factor = text[: -len(token)]
    quoted = "Q(" + repr(str(condition)) + ")"
    return factor in {f"C({condition})", f"C({quoted})"}


def _level_param(names: Sequence[str], condition: str, level: str) -> str | None:
    matches = [name for name in names if _treatment_term(name, condition, level)]
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise DesignError(
            f"More than one treatment coefficient matched {condition}={level!r}: {matches}."
        )
    return None


def _observed_levels(result: MixedModelResult, condition: str) -> set[str] | None:
    """Levels actually present in the fitted frame, if that column is there."""
    model = getattr(result.fit, "model", None)
    frame = getattr(getattr(model, "data", None), "frame", None)
    if frame is None or condition not in getattr(frame, "columns", []):
        return None
    values = frame[condition].dropna()
    return {str(value) for value in values.tolist()}


def contrast_levels(result: MixedModelResult, condition: str, a: str, b: str) -> dict[str, Any]:
    """Contrast ``b - a`` on a treatment-coded condition factor.

    The coefficient names must be ``C(condition)[T.level]`` or
    ``C(Q('condition'))[T.level]``. The omitted reference has no
    coefficient. A requested level that is absent from the fitted rows
    raises :class:`~eegmath.errors.DesignError` instead of being treated
    as that reference. Sum coding and other contrasts are not read.
    """
    if result.fit is None or not result.converged:
        return {"a": a, "b": b, "estimate": np.nan, "p_value": np.nan, "note": "model did not converge"}
    observed = _observed_levels(result, condition)
    if observed is not None:
        missing = [str(level) for level in (a, b) if str(level) not in observed]
        if missing:
            raise DesignError(
                f"Levels {missing} are not values of {condition!r} in the fitted rows. "
                "The contrast was not computed."
            )
    names = [str(name) for name in result.fit.fe_params.index]
    param_a = _level_param(names, condition, a)
    param_b = _level_param(names, condition, b)
    if param_a is None and param_b is None:
        raise DesignError(
            f"Neither {a!r} nor {b!r} has a treatment coefficient for {condition}. "
            "contrast_levels reads C(factor)[T.level] terms. One of the two levels "
            "may be the omitted reference. Other codings are not interpreted."
        )
    # MixedLM.t_test expects a numeric restriction, not a formula string.
    restriction = np.zeros((1, len(names)))
    if param_a is not None:
        restriction[0, names.index(param_a)] -= 1.0
    if param_b is not None:
        restriction[0, names.index(param_b)] += 1.0
    tested = result.fit.t_test(restriction)
    estimate = float(np.ravel(tested.effect)[0])
    se = float(np.ravel(tested.sd)[0])
    withheld = not np.isfinite(result.dof) or result.dof < 1
    if withheld or not np.isfinite(se) or se <= 0:
        stat = p = lo = hi = np.nan
    else:
        stat = estimate / se
        crit = float(stats.t.ppf(0.975, result.dof))
        p = float(2 * stats.t.sf(abs(stat), result.dof))
        lo, hi = estimate - crit * se, estimate + crit * se
    out = {
        "a": a,
        "b": b,
        "contrast": f"{b} - {a}",
        "estimate": estimate,
        "se": se,
        "statistic": stat,
        "df": result.dof,
        "p_value": p,
        "ci_low": lo,
        "ci_high": hi,
    }
    if withheld:
        out["note"] = (
            "The point estimate is the fitted contrast. The interval was withheld "
            "because the model's degrees of freedom are not positive."
        )
    return out


def repeated_measures(
    frame: pd.DataFrame,
    value: str,
    condition: str,
    subject: str,
    *,
    reference: str | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    covariates: Sequence[str] | None = None,
    aggregate: bool = True,
) -> MixedModelResult:
    """Random-intercept model of a condition effect within subject."""
    covariates = list(covariates or [])
    columns = [value, condition, subject, *covariates]
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise DesignError(f"Columns not in the table: {missing}.")
    work = frame[columns].dropna().copy()
    if aggregate:
        keys = [subject, condition]
        for cov in covariates:
            nunique = work.groupby(keys)[cov].nunique(dropna=True)
            if (nunique > 1).any():
                raise DesignError(
                    f"Covariate {cov!r} varies inside subject × condition, so it cannot be "
                    "averaged away. Set aggregate=False only if that residual variation is "
                    "the model you intend."
                )
        named = {value: (value, "mean")}
        for cov in covariates:
            named[cov] = (cov, "first")
        work = work.groupby(keys, as_index=False).agg(**named)
    work[condition] = work[condition].astype("category")
    categories = list(work[condition].cat.categories)
    if reference is not None:
        if reference not in categories:
            raise DesignError(f"Reference level {reference!r} is not present.")
        categories = [reference] + [level for level in categories if level != reference]
        work[condition] = work[condition].cat.reorder_categories(categories)
    cov_terms = " + ".join(_q(name) for name in covariates)
    formula = f"{_q(value)} ~ C({_q(condition)})" + (f" + {cov_terms}" if cov_terms else "")
    result = fit_mixed_model(work, formula, subject, reml=True)
    reduced = "1" if not covariates else cov_terms
    result.omnibus = likelihood_ratio(work, formula, f"{_q(value)} ~ {reduced}", subject)
    if pairs:
        result.contrasts = pd.DataFrame([contrast_levels(result, condition, a, b) for a, b in pairs])
    if not result.converged:
        result.coefficients["p_value"] = np.nan
        result.note += " Coefficients are reported but inference was cleared because the REML fit did not converge."
    return result
