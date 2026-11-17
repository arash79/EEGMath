# Statistical methods

These functions analyse a table or a subject-by-time array. They do not
import the feature extractor. A column named in the call is the outcome,
whatever produced it.

## Experimental units

An epoch, a window, and a channel are not subjects. `compare_groups`
defaults to `unit="subject"`: it averages rows that share a subject and a
group, and the result records that aggregation. `unit="row"` raises if a
subject still has more than one row inside a group, because that pattern
is pseudoreplication (Hurlbert 1984; Aarts et al. 2014).

A missing subject identifier raises. Those rows are not collected into
one unnamed subject. Unpaired tests (`mannwhitney`, `welch`, `ttest`)
raise when any subject is in both groups. Use `wilcoxon`,
`repeated_measures`, or `fit_mixed_model` for that design. The default
value of `test`, when the argument is omitted, is `mannwhitney`. The
result column `test` records which procedure ran.

A label passed as `a` or `b` that is not in the grouping column raises
`DesignError`. So does a group, or a within-subject pairing, with fewer
than two finite observations. The function does not return a finished
row with a missing p-value and `significant` False for those cases.
An empty table raises before any label is described as absent.
`significant` is the Benjamini–Hochberg decision for a finite corrected
p-value. It is not a code for "the comparison could not be run."
`alpha` is the rejection threshold for that decision. The
mean-difference interval stays a 95% t interval and does not use
`alpha`. The call warns when the flag and the interval disagree, in
either direction. A second column in the same call can turn the flag
off without moving the interval, because the correction is the family
of that call. The flag stays the corrected test. A zero-width interval
means the observations did not vary. The effect size is then missing. If a `channel` column contains more
than one name, `compare_groups` raises, unless that column is the
grouping factor. Comparing `channel="O1"` with `channel="O2"` does not
average the two electrodes. Any other multi-channel table is refused.
Averaging channels mixes a spatial summary into the experimental unit,
and it is the wrong operation for a ratio. Pick one channel, or
aggregate first and drop the channel column. The same warning applies
to rows: if `unit="subject"` averages more than one row of a relative
power, a decibel value, or a band ratio, `compare_groups` warns.
`aggregate_epochs` rebuilds those columns from absolute power. The mean
of the epoch-wise ratios is a different number. `aggregate_regions`
rebuilds spectral ratios from averaged absolute power, and its
`n_channels` column counts the electrodes that were actually present
in each recording. A channel name in the region list that is not in
the table is skipped, and the call warns.
`permutation_test` sign-flips within subject when every subject has both
conditions, and shuffles labels across subjects when the groups are
disjoint. Mixed overlap is rejected rather than given a silent rule.

`repeated_measures` averages technical replicates to one row per subject
and condition before fitting a random-intercept model. A covariate that
varies inside that cell is an error under the default `aggregate=True`.
`aggregate=False` fits every row and leaves residual temporal correlation
unmodelled; use it only when that is the model you intend.

## Tests and what they return

`compare_groups` runs the named test. A Shapiro-Wilk p-value is reported
when it can be computed. It does not switch the test.

| Test | Design | Effect |
| --- | --- | --- |
| `welch` | Unpaired, unequal variance | Hedges' g |
| `ttest` | Unpaired, equal variance | Hedges' g |
| `mannwhitney` | Unpaired ranks | Rank-biserial correlation |
| `wilcoxon` | Paired within subject | Cohen's d_z |

Each row also has the mean difference, a corrected p-value, and an
`interval` column that names the interval. The interval is a 95% Student-t
interval for that mean difference: pooled variance for `ttest`, the
Welch–Satterthwaite interval for `welch` and `mannwhitney`, and a
paired Student-t interval for `wilcoxon`. It is not an interval for the
rank-biserial correlation, and a Mann-Whitney row still carries the
Welch interval for the mean difference. For Wilcoxon, `normality_p_a`
is the Shapiro–Wilk test of the paired differences and `normality_p_b`
is empty.

The default correction is Benjamini-Hochberg applied to the columns of
that call. Non-finite p-values stay missing and are not counted in the
family. A finite p-value outside `[0, 1]` raises. That family is not every test you might have run in the
session. `fdr_by`, `holm`, and `bonferroni` are available.
`apply_correction(..., group_col=...)` makes each group its own family.

`friedman_test` is an omnibus test on subject means. Subjects missing a
condition are dropped, and the result says so.

`correlation` returns Pearson's r with a Fisher z interval, or Spearman
without that interval. The Fisher interval needs at least four pairs;
with three pairs the coefficient and its p-value are still returned and
the interval is missing. `partial_correlation` correlates residuals
after ordinary least squares on the covariates. Its t test uses
`n - k - 2` degrees of freedom and its Fisher interval uses
`n - k - 3`, where `k` is the number of covariates. Rows are treated
as independent.

`fit_ols` fits a formula by ordinary least squares. Rows are independent.
When `subject` is given, a missing identifier or a repeated row raises
instead of entering the model as another observation. Repeated measures
belong in `repeated_measures`.

## Mixed models

`repeated_measures` fits

```text
value ~ C(condition) + covariates + (1 | subject)
```

Coefficients come from REML. The omnibus likelihood-ratio test refits
both models by maximum likelihood, because a REML comparison of fixed
effects is not valid. Fixed-effect intervals use a Student-t reference
with degrees of freedom `n_subjects - n_fixed_effects` for every
coefficient, on the unadjusted REML standard error. The interval is
wider than a normal interval with the same standard error. It is not a
between-within split and not a Kenward-Roger or Satterthwaite
correction, and the result says so.
`likelihood_ratio` counts degrees of freedom as the difference in the
number of fixed effects. A difference of zero or less withholds the
p-value instead of assigning one degree of freedom. If the two fits do
not use the same rows, the p-value is withheld as well. When
`n_subjects - n_fixed_effects` is not positive, coefficient estimates
are still returned and the t interval and p-value are withheld. They
are not filled in at one degree of freedom. Contrasts are `b - a` on
treatment coding. `contrast_levels` matches `C(factor)[T.level]` or
`C(Q('factor'))[T.level]` only. A level that is not in the fitted rows
raises, rather than being treated as the omitted reference.

## Concordance

`prediction_probability` implements the prediction probability of Smith,
Dutton, and Smith (1996). For two independent samples it is the
Mann-Whitney / ROC area of the subject means. For paired samples it is
the proportion of subjects whose within-subject difference has the
expected sign, with ties counting one half. The paired p-value is a
two-sided sign test of the non-tied differences, the same ties the
probability excludes. It is not a Wilcoxon signed-rank test: one large
difference can hide a consistent sign in the rank test while the sign
test still rejects. The unpaired p-value is the Mann-Whitney test of
the subject means. `p_value_test` names which test produced the number.
The `estimator` field names which probability was used. They answer
different questions: every subject can
move in the same direction while the two conditions still overlap, and
then the paired value is 1 while the unpaired area is not.

`group_low` and `group_high` only set the direction. Swapping them
replaces Pk with `1 - Pk`. Near-equality uses a relative tolerance of
`1e-9`. A jackknife standard error and an optional bootstrap interval
are subject-level. The jackknife standard error is withheld when a
leave-one-out replicate is not finite, because dropping that replicate
would change the sample size inside the pseudo-value formula.

With `strategy=None`, three or more subjects who have both groups
select the paired estimator. Subjects who have only one group are not
folded into that paired number. Fewer than three paired subjects
selects the unpaired estimator on the subjects who have only one group.
Requesting `strategy="paired"` with fewer than three paired subjects
raises an error rather than switching estimators. Requesting
`strategy="unpaired"` when every subject has both groups also raises.
Those two means are not an unpaired sample.

## Resampling and intervals

`bootstrap_ci` is an i.i.d. percentile interval of the rows you pass.
Do not pass every epoch if the subject is the unit; average first.

`cluster_permutation_1d` takes an array shaped `(n_subjects, n_times)`
of within-subject differences. The sample statistic is a one-sample t
against zero. Clusters are contiguous samples of one sign whose
absolute t exceeds the two-sided critical value at `cluster_alpha`
(default 0.05). That threshold only decides which samples can join a
cluster. `alpha` is the significance level of the cluster p-value and
does not move the threshold. A positive run is not merged with an
adjacent negative run. Cluster mass is the sum of absolute t inside
the run. Exchange under the null is a sign flip within subject (Maris
and Oostenveld 2007). A nonzero mean with zero dispersion is treated as
an infinite t. Epochs must already have been reduced to one series per
subject, and the array must be finite.

`mean_difference_posterior` is a Student-t posterior for a mean
difference under a flat prior on the mean when the data are one sample
or paired. For two unpaired samples the scale is the Welch standard
error and the degrees of freedom are Satterthwaite's; that is an
approximation to the one-sample posterior. `prob_mean_gt_0` is a
posterior probability, not a Bayes factor. `bic_bayes_factor` is the
Wagenmakers (2007) BIC approximation of the evidence for a condition
mean structure against a common mean. Its rows are treated as
independent, so technical replicates must already have been removed.
A very large value is returned as infinity rather than an overflow.

## Reading a result

A small p-value is a statement about the test and the design that were
run. It is not, by itself, a measure of scientific or clinical
importance. The effect, interval, and sample size are part of the same
result. Exploratory correction across a catalog of features is not a
pre-registered confirmatory family; the `family` column records the set
that was actually corrected.
