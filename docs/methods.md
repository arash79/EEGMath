# Method coverage

Status values:

| Status | Meaning |
| --- | --- |
| `synthetic` | Checked against an analytical case or a reference implementation |
| `public-data` | Also run on Cambridge and/or NEMAR nm000109 in the validation notebooks |
| `excluded` | Considered and left out, with a reason |
| `future` | In scope later; not implemented |

Public-data status is evidence that the same API accepts those recordings.
It is not a claim that every dataset will behave the same way, and it is
not a claim that a group difference must be statistically significant.

## Implemented

| Method | API | Status | Anchor |
| --- | --- | --- | --- |
| Welch PSD, band power, relative power, peak frequency | `eegmath.spectral.welch_psd`, `extract_features(..., "spectral")` | synthetic; both validation notebooks | Welch 1967. Band power interpolates the density at the interval edges, so a contiguous partition sums to its union |
| Spectral edge, mean and median frequency | spectral feature set | synthetic; Cambridge notebook checks median frequency equals SEF50 | cumulative trapezoidal integral; median frequency is SEF50 |
| Spectral entropy | `spectral_entropy` column | synthetic; Cambridge example recording | Shannon entropy of the normalised Welch masses, divided by `log2(n_bins)` so the column is in [0, 1]. Inouye et al. 1991 reported the unnormalised index in bits |
| Log-log aperiodic line, one residual trim | `aperiodic_fit` | synthetic | ordinary least-squares line in log-log coordinates. The optional screen drops the upper residual tail and refits once. It is not a peak model. Donoghue et al. 2020 is the related model that is not implemented |
| Magnitude-squared coherence | `connectivity_matrices` | synthetic; Cambridge notebook, six channels, epochs pooled | Bendat and Piersol; volume conduction inflates it |
| Phase-lag index | `pli` | synthetic | Stam, Nolte, and Daffertshofer 2007 |
| Debiased squared wPLI | `wpli` | synthetic; computed beside coherence in the Cambridge notebook | Vinck et al. 2011, NeuroImage 55:1548–1565, DOI 10.1016/j.neuroimage.2011.01.055. The sample estimate lies in `[-1, 1]` and can be negative. A negative value is not a lead or a lag and is not an Onnela weight. Fewer than two segments is missing. A vanishing imaginary denominator with two or more segments is 0 |
| Phase-locking value | `plv` | synthetic | Lachaux et al. 1999. Small segment counts leave a positive bias, and the same segment minimum used for coherence withholds the matrix |
| Pairwise phase consistency | `ppc` | synthetic | Vinck, van Wingerden, Womelsdorf, Fries, and Pennartz 2010, NeuroImage 51:112–122, DOI 10.1016/j.neuroimage.2010.01.073. For `n` Fourier segments and phase-locking value `P`, the estimate is `(n P² − 1) / (n − 1)`. It estimates squared phase-locking value and can be negative. It is not an Onnela weight |
| Imaginary coherency | `imcoh` | synthetic | Nolte et al. 2004. The band matrix is a signed mean. Node features average the absolute value across frequency |
| Sensor-space correlation | `sensor_correlation` | synthetic | Pearson or Spearman in time. Epochs are combined by a Fisher-z mean. A constant channel is missing. This is not a spectral connectivity measure |
| Onnela clustering, node strength, region mean | `clustering_coefficient`, `region_connectivity` | synthetic | Onnela et al. 2005. Weights must already be in [0, 1]; they are not divided by the largest edge. A missing pair is omitted from the node mean and from node strength |
| Permutation entropy | `permutation_entropy` | synthetic, AntroPy cross-check; one Cambridge epoch | Bandt and Pompe 2002 |
| Approximate and sample entropy | `approximate_entropy`, `sample_entropy` | synthetic, AntroPy cross-check | Pincus 1991; Richman and Moorman 2000 |
| Multiscale sample entropy, radius fixed at scale 1 | `multiscale_sample_entropy`, `FeatureConfig.mse_r_fraction` | synthetic | Costa, Goldberger, and Peng 2002. Their illustration used `r = 0.15` times the original standard deviation. The default fraction is 0.2, matching sample entropy. `mse_mean` keeps non-finite scales |
| SVD entropy | `svd_entropy` | synthetic, AntroPy cross-check | Shannon entropy of `p_i = σ_i / Σσ` (Bao, Liu, and Zhang 2011, eq. 9, attributing that definition to Roberts, Penny, and Rezek 1999). Optional division by `log2(order)` is a separate `[0, 1]` scaling. Not the entropy of squared singular values |
| Lempel-Ziv complexity, Zhang normalisation | `lempel_ziv_complexity` | synthetic | Lempel and Ziv 1976; Kaspar and Schuster 1987; Zhang et al. 2009 |
| Envelope LZC | `envelope_lzc` | synthetic definition | Hilbert amplitude binarised at its mean (Schartner et al. 2015), then single-channel Zhang normalisation. Not their multichannel shuffle-normalised measure. The 1–40 Hz band is part of this feature |
| Higuchi, Petrosian, Katz dimensions | `algorithms.fractal` | synthetic, AntroPy cross-check | Higuchi 1988; Petrosian 1995; Katz 1988 |
| Detrended fluctuation | `detrended_fluctuation` | synthetic | Peng et al. 1994 |
| Hjorth activity, mobility, complexity | `hjorth_parameters` | synthetic; Cambridge occipital epoch | Hjorth 1970. The continuous-time mobility of a sine is `2 π f` radians per second. The one-sample difference is `2 * sfreq * sin(π f / sfreq)`. Complexity is 1 in continuous time and slightly above 1 on a short epoch |
| Burst-suppression ratio | `burst_suppression_ratio` | synthetic | Fraction of time `abs(x)` stays below threshold for longer than `min_duration`. Defaults are 5 µV and 0.5 s (Rampil 1998; Muhlhofer et al. 2017). Not the differentiated ±15 µV detector of Vijn and Sneyd 1998 |
| FIR band-pass, notch, average and electrode reference, resample | `eegmath.preprocess` | synthetic, including a recording constructed by MNE | MNE-Python, Gramfort et al. 2013. Band-pass and notch default to EEG-typed channels. A notch line above a recorded low-pass is omitted; lines below it are applied. Epoched data are notched per epoch. The default notch transition is 1 Hz and the Hamming FIR is about 6.6 s. A kernel longer than the samples being filtered raises `ParameterError`. Filter limits are merged on the recording and are not written back onto MNE `Info` |
| Zero-phase Morlet power | `morlet_power` | synthetic; one short segment in each notebook | Tallon-Baudry and Bertrand 1999, via MNE. The array is the squared wavelet transform after scaling to norm √2. It is not Welch band power |
| Evoked average and peak latency | `average_epochs`, `peak_measure` | synthetic | requires epochs |
| Intertrial phase clustering | `intertrial_phase_clustering` | synthetic | magnitude of the mean unit-length Morlet phase across epochs. Requires at least two epochs. It does not read an event table. Windowing supplies the window onset as a clock: a tone that completes an integer number of cycles in every window gives clustering 1. These resting recordings were not analysed that way, because the result would describe the window length. The estimate is biased upward when few epochs are available. `intertrial_phase_consistency` removes that bias and does not replace this function |
| Intertrial phase consistency | `intertrial_phase_consistency` | synthetic | The same Vinck et al. 2010 formula as pairwise phase consistency, applied to wavelet phase across epochs that share a clock: `(n P² − 1) / (n − 1)`. A window onset is a clock. It can be negative. It is not a feature-catalog column. The function does not read an event table |
| Modulation index | `modulation_index` | synthetic | Tort, Komorowski, Eichenbaum, and Kopell 2010, J Neurophysiol 104:1195–1210, DOI 10.1152/jn.00106.2010. A phase bin with no samples has no mean amplitude, and the index is then missing. An illegal band raises `ParameterError` |
| Augmented Dickey-Fuller | `adf_stationarity` | synthetic; one channel in each notebook | Dickey and Fuller 1979; rejecting a unit root is not a general stationarity proof. A non-finite sample, a constant series, or a rank-deficient regression makes the result missing. Interior gaps are not dropped and concatenated |
| KPSS stationarity test | `kpss_stationarity` | synthetic | Kwiatkowski, Phillips, Schmidt, and Shin 1992, Journal of Econometrics 54:159–178, DOI 10.1016/0304-4076(92)90104-Y. The null is level stationarity (`regression="c"`) or trend stationarity (`"ct"`). statsmodels interpolates the p-value from their table and returns 0.01 or 0.1 at the ends. Those values are bounds. The statistic itself is the statsmodels implementation of that paper |
| Welch and Student t, Mann-Whitney, Wilcoxon, Friedman | `compare_groups`, `friedman_test` | synthetic; Wilcoxon on subject means in both notebooks. Welch, Mann-Whitney, and Friedman remain synthetic | the named test is the one that runs |
| Hedges g, Cohen d and d_z, rank-biserial | `eegmath.stats.effects` | synthetic | Hedges 1981; Cohen 1988 |
| BH, BY, Holm, Bonferroni | `adjust_pvalues` | synthetic | Benjamini and Hochberg 1995; Benjamini and Yekutieli 2001; Holm 1979 |
| Sample autocorrelation and Yule-Walker partial autocorrelation | `autocorrelation`, `partial_autocorrelation` | synthetic | Biased autocorrelation divides every lag by `n`. Partial autocorrelation is Yule-Walker with that same autocovariance (`ywmle`). An AR(1) population curve is `phi` to the lag. No Bartlett interval and no white-noise test. statsmodels cites Parzen 1963 and Brockwell and Davis; those books were not opened |
| Random-intercept mixed model, ML likelihood-ratio test | `repeated_measures`, `likelihood_ratio` | synthetic; Cambridge notebook, 20 subjects and four conditions | Laird and Ware 1982. Interval degrees of freedom are `n_subjects - n_fixed_effects` for every coefficient when that value is at least 1. A smaller value withholds the t interval and p-value; 1 is not substituted. That degrees-of-freedom choice uses the unadjusted REML standard error. It is wider than a normal interval with the same standard error. It is not a between-within split and not Kenward-Roger or Satterthwaite. The likelihood-ratio degrees of freedom are the difference in the number of fixed effects; a non-positive difference, or two fits that do not use the same rows, withholds the p-value. `contrast_levels` reads treatment terms `C(factor)[T.level]` exactly and raises when a named level is absent |
| Prediction probability | `prediction_probability` | synthetic | Smith, Dutton, and Smith 1996. Unpaired Pk is the Mann-Whitney probability of the subject means, and its p-value is that test. Paired Pk is the within-subject sign concordance; its p-value is a two-sided sign test, not Wilcoxon |
| Subject-level permutation and same-sign cluster permutation | `permutation_test`, `cluster_permutation_1d` | synthetic | Maris and Oostenveld 2007 for the cluster test |
| Percentile bootstrap, Fisher interval, partial correlation | `bootstrap_ci`, `correlation`, `partial_correlation` | synthetic; Spearman correlation in both notebooks. Bootstrap, Fisher intervals, and partial correlation remain synthetic | Efron; Fisher |
| Ordinary least squares | `fit_ols` | synthetic | rows are independent. A subject column rejects missing identifiers and repeated rows |
| t posterior probability and BIC Bayes factor | `mean_difference_posterior`, `bic_bayes_factor` | synthetic | Wagenmakers 2007 for the BIC approximation |

`tests/test_public_data.py` reads one Cambridge EEGLAB file and one NEMAR
subject. The notebooks under `notebooks/` then run the cohort analyses.
`docs/validation/RESULTS.md` records what those executions produced.
Rows that say "synthetic" without a notebook were not given a real-data
confirmation in the notebooks. Cluster permutation, evoked-peak
measures, intertrial phase clustering, sensor-space correlation,
ordinary least squares, the autocorrelation functions, and intertrial
phase consistency are in that group. These recordings are not
stimulus-locked, and the notebooks do not invent events for them.
A modulation-index check on the Cambridge occipital epochs, with
amplitude-shift surrogates, was run outside this package. That call
is not in the notebooks.

## Excluded

| Method | Reason |
| --- | --- |
| Automatic ICA or artifact subspace reconstruction | Needs a checked decomposition and a policy for which components to drop. Blind removal is not offered. |
| Source reconstruction and source connectivity | Needs a montage, a forward model, and usually a structural image. Sensor-space measures stay sensor-space. |
| FOOOF / specparam peak-plus-aperiodic model | The implemented fit is one log-log line with a single residual screen. It is not that model. |
| Multitaper PSD as the default | Welch is the default because its resolution is explicit in seconds. Multitaper remains available through MNE or SciPy outside this default. |
| Granger causality, PDC, directed PLI | Directional connectivity adds model-order and stationarity choices that are not validated here. |
| Microstates | Requires a clustering model and a template policy of its own. |
| Kenward-Roger or Satterthwaite degrees of freedom | Not implemented. The mixed-model interval uses n_subjects - n_fixed on the unadjusted REML standard error. It is not a small-sample correction of that standard error. |
| Silent test selection from a normality screen | Diagnostics are reported beside the test the caller chose. |

## Future

Pairwise phase consistency is implemented for Fourier segments, and
intertrial phase consistency applies the same correction to wavelet
phase across epochs. Phase-locking value and intertrial phase clustering
keep their original definitions, including the small-sample positive
bias. The debiased functions do not replace them.

Cluster permutation has a synthetic check. The public recordings do not
include a stimulus-locked design that would make a cluster test of an
evoked waveform a valid confirmatory analysis, so that real-data use
remains open. The modulation index has the synthetic check. A check on
the Cambridge occipital epochs was run outside this package. A richer graph catalogue, multitaper
spectra as a first-class estimator, and a mixed-model Bayes factor are
the same kind of open work: useful, and not yet validated here.

## Connectivity, stated plainly

Coherence, phase-locking value, and pairwise phase consistency rise when
two sensors share a source or a reference. On exactly two EEG channels,
average reference makes one series the negative of the other, and those
three measures are then 1. `rereference` warns in that case. The lag measures on that pair are 0. That
identity is the reference. Pairwise phase consistency
removes the small-sample bias of the phase-locking value. It does not
remove zero-lag mixing. Phase-lag index, weighted phase-lag index, and imaginary
coherency are less sensitive to exact zero-lag mixing and are still
sensor-space statistics (Bastos and Schoffelen 2016). Results are
withheld when fewer than `min_segments` segments are available, because
a short average has a positive bias. Segments are not taken across epoch
boundaries.

## Sources for definitions that are easy to confuse

- Inouye T, Shinosaki K, Sakamoto H, Toi S, Ukai S, Iyama A, Katsuda Y,
  Hirano M. Quantification of EEG irregularity by use of the entropy of
  the power spectrum. Electroencephalography and clinical Neurophysiology.
  1991;79:204–210. The irregularity index is the unnormalised entropy.
- Schartner M, Seth A, Noirhomme Q, Boly M, Bruno M-A, Laureys S, Barrett A.
  PLoS ONE. 2015;10(8):e0133532. DOI 10.1371/journal.pone.0133532.
  The Hilbert-amplitude threshold is the channel mean. Their complexity
  concatenates channels and divides by a time-shuffled copy.
- Muhlhofer et al. Br J Anaesth. 2017;118:755–761. DOI 10.1093/bja/aex054.
  Restates the monitor rule: intervals longer than 0.5 s during which the
  voltage does not exceed approximately ±5 µV, scored on the preceding
  63 s. This library returns the fraction of the supplied array.
- Vijn PC, Sneyd JR. Br J Anaesth. 1998;81:415–421. DOI 10.1093/bja/81.3.415.
  Thresholds the time derivative inside ±15 µV for at least 100 ms.
  That detector is not implemented.
- Nolte G, Bai O, Wheaton L, Mari Z, Vorbach S, Hallett M. Identifying
  true brain interaction from EEG data using the imaginary part of
  coherency. Clinical Neurophysiology. 2004;115:2292–2307.
  DOI 10.1016/j.clinph.2004.04.029. The lead/lag reading of a positive
  entry is the sign of `X_i conj(X_j)` in this implementation, checked
  on a delayed sinusoid.
