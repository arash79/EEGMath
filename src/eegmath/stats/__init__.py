"""Statistical analysis of tables and subject-level series."""

from eegmath.stats.concordance import batch_concordance, pk_from_arrays, prediction_probability
from eegmath.stats.design import assert_independent_units, design_summary, subject_means
from eegmath.stats.effects import cohens_d, cohens_dz, hedges_g, rank_biserial
from eegmath.stats.inference import (
    bic_bayes_factor,
    bootstrap_ci,
    cluster_permutation_1d,
    compare_groups,
    correlation,
    friedman_test,
    mean_difference_posterior,
    partial_correlation,
    permutation_test,
)
from eegmath.stats.linear import fit_ols
from eegmath.stats.mixed import contrast_levels, fit_mixed_model, likelihood_ratio, repeated_measures
from eegmath.stats.multiple import adjust_pvalues, apply_correction, bonferroni, fdr_bh, fdr_by, holm

__all__ = [
    "adjust_pvalues",
    "apply_correction",
    "assert_independent_units",
    "batch_concordance",
    "bic_bayes_factor",
    "bonferroni",
    "bootstrap_ci",
    "cluster_permutation_1d",
    "cohens_d",
    "cohens_dz",
    "compare_groups",
    "contrast_levels",
    "correlation",
    "design_summary",
    "fdr_bh",
    "fdr_by",
    "fit_mixed_model",
    "fit_ols",
    "friedman_test",
    "hedges_g",
    "holm",
    "likelihood_ratio",
    "mean_difference_posterior",
    "partial_correlation",
    "permutation_test",
    "pk_from_arrays",
    "prediction_probability",
    "rank_biserial",
    "repeated_measures",
    "subject_means",
]
