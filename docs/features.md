# Features

`list_features()` returns the catalog, including a `quantity` for each
column. The `units` field on a feature row is the recording unit for
the whole row. It is not that quantity. `extract_features(eeg, features)`
computes a subset. `features` may be `"all"`, a family name (`time`,
`spectral`, `entropy`, `complexity`, `dynamics`, `connectivity`,
`variability`), a column name, or a sequence of those.
`pe`, `apen`, `sampen`, and `bsr` are aliases.

Segment features are computed inside each epoch. Recording features use
the whole recording and are marked `scope="recording"`. Connectivity
pools FFT segments across epochs without concatenating across epoch
boundaries. `rms_cv` is the coefficient of variation of RMS across
epochs, or across non-overlapping windows when the recording is continuous.

Spectral ratios are ratios of absolute power. Relative power is a share
of the full configured partition. Requesting one relative column still
uses every band in that partition; the unused band powers are not kept
in the returned table. `aggregate_epochs` and `aggregate_regions` average
absolute power and the other segment-level columns. The average of a
peak frequency or a spectral edge is the mean of those epoch values, not
the edge of the mean spectrum. They rebuild relative power, decibels,
and ratios from the averaged absolute power when the partition is
present, and they leave those columns missing when it is not. Pass
`bands` when the partition is not the default set. A direct call to
`recompute_power_ratios` leaves a stored relative column alone when the
partition is incomplete. `COARSE_10_20` is only a convenience partition
of common 10-20 and 10-10 names. T3 and T4 are the older names of the
T7 and T8 sites, and T5 and T6 are the older names of the P7 and P8
sites (ACNS Guideline 2). They are not extra electrodes.

Windowed recordings keep each window's start, in seconds on the
pre-window clock, in an `onset` column. Epochs loaded from MNE keep
`event_code` on epoch rows. When an epoched recording is windowed and
its event codes or event times have one entry per epoch, every window
inherits the code and time of its parent epoch. Neither column is a
feature, and neither is averaged when epochs are reduced.

## Time domain

RMS is the root mean square. Line length is the mean absolute successive
difference per second. Zero-crossing rate is sign changes per second of
the span from the first sample to the last, `(n - 1) / sfreq`. An exact
zero is neither sign. A touch that returns to the same side is not a
crossing. Petrosian dimension counts derivative sign changes the same way.
Peak-to-peak is the range. Skewness and kurtosis are the biased Fisher
moments (`kurtosis` is 0 for a normal sample).

Hjorth activity is the population variance. Mobility is
`sqrt(var(dx/dt) / var(x))` with the first difference scaled by the
sampling frequency, in radians per second. For a pure sinusoid of
frequency `f` the continuous-time value is `2 π f`. The one-sample
difference equals `2 * sfreq * sin(π f / sfreq)`, which approaches
`2 π f` when the sampling rate is high relative to `f`. Complexity is
the mobility of the derivative divided by the mobility of the signal.
It is 1 for a pure sinusoid in continuous time (Hjorth 1970). On a
finite epoch the second difference is shorter than the series, so the
value sits slightly above 1.

## Spectrum

The spectrum is Welch's method: Hann windows, 50% overlap, density
scaling, and a default segment length of 2 seconds so the frequency
resolution does not change silently with the sampling rate. The default
detrend removes the segment mean. Linear detrend is available and
attenuates the slowest oscillations inside a short segment.

Band power is the trapezoidal integral of the density over the requested
interval. Endpoints that fall between frequency bins are linearly
interpolated, so a band narrower than the bin spacing still has a power,
and adjacent bands that meet at one frequency do not count that point
twice. A contiguous partition therefore sums to the integral of its
union. When that union is `[freq_min, freq_max]`, the band powers also
sum to `total_power`. `freq_min` and `freq_max` do not clip the named
bands. A gamma integral of 30–45 Hz is still computed when `freq_max`
is 20, and then the band powers do not equal `total_power`. The call
warns when a named band extends outside that interval and still enters
the relative-power denominator. Default
bands are delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, and gamma
30–45 Hz. That is a common clinical partition, not a claim that any one
guideline uses these edges. The 30 Hz cut is this package's split
between beta and gamma. Gamma stops at 45 Hz so that a 50 Hz mains line,
and the transition of a 45 Hz low-pass, stay outside the band. A tone
that sits on a shared edge, such as 13 Hz, is split by the tapered
integral, so its power can fall about half in each adjacent band. The
peak-frequency bin on that edge is assigned to the higher band.

Relative power divides by the sum
of the band integrals that overlap the estimated spectrum, so a complete
partition sums to 1. A band that misses the spectrum entirely has
missing power and is left out of that denominator. A band that crosses
Nyquist is integrated only up to the last estimated frequency. Peak
frequency is a property of the bins assigned to the band. Those masks
are half-open at the upper edge, except that a band which reaches the
highest cutoff in the set includes that cutoff. The order of the
mapping does not decide which edge is closed, so a bin on a shared
boundary belongs to the higher band. A band with fewer than two
such bins has a power when the interval overlaps the spectrum, and a
missing peak frequency.
`{band}_power_db` is `10 log10` of absolute power. Non-positive power is
missing. `{band}_peak_freq` is the bin of maximum density inside the
band. It is missing when that maximum is not positive, and when every
bin in the band equals the maximum. A tiny positive maximum, including
numerical dust in an otherwise empty band, is still reported. The peak
is not by itself evidence of a rhythm. A tie among some of the bins uses
the first maximum. A positive band power divided by zero is infinite.
Zero divided by zero is missing. Nothing is added to a denominator.

`total_power`, `mean_freq`, `median_freq`, and `sef{p}` use that same
trapezoidal measure on `[freq_min, freq_max]`, limited to the frequencies
that were estimated. The spectral-edge cumulative integral starts at
zero on the lower edge of that interval, so a percentile inside the
first step is interpolated instead of being moved to the next bin. `median_freq` is always SEF50, including
when 50 is absent from `sef_percentiles`. When `sef50` is also computed,
the two columns are the same array. Spectral entropy is the Shannon
entropy of the Welch bin masses in that range, divided by
`log2(n_bins)`, so the column lies in `[0, 1]`. Inouye et al. (1991)
defined the unnormalised entropy and reported it in bits. The divisor
is not part of that index; it keeps the value from growing when the
number of Welch bins changes. A range with no spectral mass has
missing entropy. A partition with no power has missing relative power.

Ratios are `delta/alpha`, `theta/alpha`, `delta/theta`, `beta/alpha`,
and `beta/delta`. `slow_fast_ratio` is `(delta + theta) / (alpha + beta)`.
`alpha_dominance` is alpha divided by the sum of those four bands. A
ratio is omitted when either band was not requested.

`aperiodic_exponent` and `aperiodic_offset` are the slope and intercept
of `log10(psd) = offset - exponent * log10(freq)` on 2–40 Hz, refit once
after dropping bins whose residual exceeds the mean residual by one
residual standard deviation. That trims the upper tail. It is not a
peak fit, and it is not the FOOOF peak model.

## Entropy and complexity

Permutation entropy is the Shannon entropy of ordinal patterns (Bandt and
Pompe 2002), divided by `log2(order!)`. Ties use a stable argsort.

Sample entropy and approximate entropy use Chebyshev distance and a
radius of `0.2` times the population standard deviation (Richman and
Moorman 2000; Pincus 1991). Sample entropy drops self-matches and uses
the same number of templates at embedding `m` and `m+1`. Approximate
entropy keeps self-matches, so the two template counts differ. On a
short periodic series that difference can make the estimate slightly
negative.

Multiscale entropy coarse-grains by non-overlapping means (Costa,
Goldberger, and Peng 2002). The default radius is fixed from the scale-1
standard deviation, which is their original-series convention. Their
illustration used embedding 2 and a radius of 0.15 times that standard
deviation (footnote 12). The default fraction here is 0.2, matching
sample entropy. Pass `FeatureConfig(mse_r_fraction=0.15)` to match that
illustration. `mse_r_mode="scale"` recomputes the radius on each coarse
series; that is a common variant and is not the original fixed radius.
`FeatureConfig.mse_scales` chooses the scales. The default columns are
`mse_scale1`, `mse_scale5`, and `mse_scale10`; a different tuple replaces
those column names. `mse_mean` averages the configured scales. A scale
that is missing or infinite stays in that average. A length-`m+1`
sample entropy of `+inf` is the logarithm of a zero match ratio, not a
missing value.

SVD entropy is the Shannon entropy of `p_i = σ_i / Σσ` for the singular
values of a delay embedding. Bao, Liu, and Zhang (2011, eq. 9) define
that entropy and attribute it to Roberts, Penny, and Rezek (1999).
When normalised, the result is divided by `log2(order)`. That divisor
is a separate map onto `[0, 1]`. There are at most `order` singular
values, so it is not `log2(order!)`. Squared singular values are not
used.

Lempel-Ziv complexity is the Kaspar and Schuster parsing of a discrete
symbol sequence. The normalised value divides by `n / log_b(n)` (Zhang
et al. 2009). A non-integer float is rejected. Integer-valued floats,
integers, booleans, and strings are symbols. The `lzc` feature
binarises each channel at its median first. Envelope LZC band-passes at
1–40 Hz, takes the Hilbert amplitude, and binarises that amplitude where
it exceeds its mean, the threshold used by Schartner et al. (2015). The
returned number is the single-channel Zhang normalisation. It is not
their multichannel complexity, which concatenates channels and divides
by the complexity of a shuffled copy. Schartner reported broadband
signals and, separately, band-limited ones; the 1–40 Hz interval is
part of this feature.

Higuchi, Petrosian, and Katz dimensions follow those papers. A straight
line has dimension 1. Katz's amplitude-only form is not a planar
dimension and can exceed 2 when the trace oscillates many times. It is
infinite when the farthest amplitude equals the mean step. Higuchi
omits a scale whose curve length is zero and fits the remaining scales.
Detrended fluctuation analysis follows Peng et al.
(1994): window lengths grow geometrically from 4 samples to 10% of the
series. White noise is near 0.5 and integrated white noise is near 1.5.

## Dynamics and variability

Burst-suppression ratio is the fraction of time the absolute amplitude
stays strictly below a threshold for longer than `min_duration`. The
defaults are 5 µV and 0.5 s: `5e-6` when `units="V"` and `5` when
`units="uV"`. `units="z"` has no voltage, so the ratio is missing. A run of exactly 0.5 s does not count. Samples that only
cross the threshold, as a larger oscillation does, do not count either.
An array no longer than 0.5 s cannot contain a qualifying run, so the
ratio is missing. This is the human-scalp duration-and-voltage
convention associated with Rampil (1998) and restated by Muhlhofer et
al. (2017). It is not the differentiated ±15 µV detector of Vijn and
Sneyd (1998), and the ratio is taken over the supplied array rather
than a fixed monitor window. After `zscore`, or a provenance note that starts with `zscore`,
`extract_features` returns a missing burst-suppression ratio and warns
once. `rms_cv` is the coefficient of variation of window or
epoch RMS, with a sample standard deviation.

## Connectivity

Node features are the mean coupling of each channel with the others.
Each band matrix is the mean of the spectral metric across the bins in
that band in which both channels exceed `1e-12` of their own peak
autospectrum. Bins below that floor are numerical nulls. The band
column is that mean, so a tone that occupies one bin can leave the
column well below the value in that bin. Empty bins that still clear
the floor stay in the average. The imaginary
coherency matrix is the mean of the signed values, so opposite lags
inside a band cancel. A positive entry in row `i` and column `j` means
channel `i` leads channel `j`. Pairwise phase consistency
(Vinck, van Wingerden, Womelsdorf, Fries, and Pennartz 2010) is
`(n P² − 1) / (n − 1)`, where `P` is the phase-locking value and `n`
is the number of Fourier segments. It estimates the squared
phase-locking value without the small-sample positive bias of `P`.
Values lie in `[-1, 1]` and can be negative. The band average keeps
that sign. A negative value is not an Onnela graph weight.
Debiased squared weighted phase-lag index (Vinck et al. 2011) is the
same kind of pairwise estimator. It also lies in `[-1, 1]` and can be
negative. Flooring it at zero would put the small-sample positive bias
back. Fewer than two segments leaves it missing. A denominator that
vanishes because every imaginary part is zero is reported as zero.
Node-feature columns average the absolute imaginary coherency across
frequency first. `connectivity_node_features` still returns the signed
matrices under `meta["matrices"]`. `global_*` is the mean over the
broadband, not over the canonical bands. Matrices with fewer than 20
segments are returned as NaN, and the call warns. At the default 2 s
segments and 50% overlap, 20 s of continuous data yields 19 segments
and 21 s yields 20. Node columns for imaginary coherency are means of
absolute values. The signed matrix, which still says which channel
leads, is in the metadata under `matrices`, not in the feature column
`alpha_imcoh`. `FeatureConfig.detrend` is applied to
each connectivity segment. See [methods](methods.md) for the
volume-conduction limits.

Each feature row also carries `duration_sec`, `highpass_hz`, and
`lowpass_hz` from the recording. A missing filter limit stays missing.
These columns are not features. `aggregate_regions` and
`aggregate_epochs` keep them with the recording and do not average them.

## Catalog

`quantity` is what the number is. It is not the `units` column of a feature row.

<!-- catalog:start -->
| Feature | Family | Scope | Quantity |
| --- | --- | --- | --- |
| `rms` | time | segment | recording unit |
| `line_length` | time | segment | recording unit per second |
| `zero_crossing_rate` | time | segment | per second |
| `peak_to_peak` | time | segment | recording unit |
| `skewness` | time | segment | dimensionless moment |
| `kurtosis` | time | segment | dimensionless moment |
| `hjorth_activity` | time | segment | recording unit squared |
| `hjorth_mobility` | time | segment | radians per second |
| `hjorth_complexity` | time | segment | dimensionless |
| `delta_power` | spectral | segment | recording unit squared |
| `delta_relative` | spectral | segment | proportion |
| `delta_power_db` | spectral | segment | decibels |
| `delta_peak_freq` | spectral | segment | hertz |
| `theta_power` | spectral | segment | recording unit squared |
| `theta_relative` | spectral | segment | proportion |
| `theta_power_db` | spectral | segment | decibels |
| `theta_peak_freq` | spectral | segment | hertz |
| `alpha_power` | spectral | segment | recording unit squared |
| `alpha_relative` | spectral | segment | proportion |
| `alpha_power_db` | spectral | segment | decibels |
| `alpha_peak_freq` | spectral | segment | hertz |
| `beta_power` | spectral | segment | recording unit squared |
| `beta_relative` | spectral | segment | proportion |
| `beta_power_db` | spectral | segment | decibels |
| `beta_peak_freq` | spectral | segment | hertz |
| `gamma_power` | spectral | segment | recording unit squared |
| `gamma_relative` | spectral | segment | proportion |
| `gamma_power_db` | spectral | segment | decibels |
| `gamma_peak_freq` | spectral | segment | hertz |
| `delta_alpha_ratio` | spectral | segment | power ratio |
| `theta_alpha_ratio` | spectral | segment | power ratio |
| `delta_theta_ratio` | spectral | segment | power ratio |
| `beta_alpha_ratio` | spectral | segment | power ratio |
| `beta_delta_ratio` | spectral | segment | power ratio |
| `slow_fast_ratio` | spectral | segment | power ratio |
| `alpha_dominance` | spectral | segment | power ratio |
| `total_power` | spectral | segment | recording unit squared |
| `mean_freq` | spectral | segment | hertz |
| `median_freq` | spectral | segment | hertz |
| `spectral_entropy` | spectral | segment | normalized entropy |
| `sef50` | spectral | segment | hertz |
| `sef75` | spectral | segment | hertz |
| `sef90` | spectral | segment | hertz |
| `sef95` | spectral | segment | hertz |
| `aperiodic_exponent` | spectral | segment | log-log slope |
| `aperiodic_offset` | spectral | segment | log10 power density |
| `lzc` | entropy | segment | dimensionless |
| `approximate_entropy` | entropy | segment | dimensionless |
| `sample_entropy` | entropy | segment | dimensionless |
| `permutation_entropy` | entropy | segment | dimensionless |
| `svd_entropy` | entropy | segment | dimensionless |
| `mse_mean` | entropy | segment | dimensionless |
| `mse_scale1` | entropy | segment | dimensionless |
| `mse_scale5` | entropy | segment | dimensionless |
| `mse_scale10` | entropy | segment | dimensionless |
| `higuchi_fd` | complexity | segment | dimensionless |
| `petrosian_fd` | complexity | segment | dimensionless |
| `katz_fd` | complexity | segment | dimensionless |
| `dfa` | complexity | segment | dimensionless |
| `burst_suppression_ratio` | dynamics | segment | proportion |
| `envelope_lzc` | dynamics | segment | dimensionless |
| `delta_coherence` | connectivity | recording | dimensionless |
| `delta_pli` | connectivity | recording | dimensionless |
| `delta_wpli` | connectivity | recording | dimensionless |
| `delta_plv` | connectivity | recording | dimensionless |
| `delta_ppc` | connectivity | recording | dimensionless |
| `delta_imcoh` | connectivity | recording | dimensionless |
| `theta_coherence` | connectivity | recording | dimensionless |
| `theta_pli` | connectivity | recording | dimensionless |
| `theta_wpli` | connectivity | recording | dimensionless |
| `theta_plv` | connectivity | recording | dimensionless |
| `theta_ppc` | connectivity | recording | dimensionless |
| `theta_imcoh` | connectivity | recording | dimensionless |
| `alpha_coherence` | connectivity | recording | dimensionless |
| `alpha_pli` | connectivity | recording | dimensionless |
| `alpha_wpli` | connectivity | recording | dimensionless |
| `alpha_plv` | connectivity | recording | dimensionless |
| `alpha_ppc` | connectivity | recording | dimensionless |
| `alpha_imcoh` | connectivity | recording | dimensionless |
| `beta_coherence` | connectivity | recording | dimensionless |
| `beta_pli` | connectivity | recording | dimensionless |
| `beta_wpli` | connectivity | recording | dimensionless |
| `beta_plv` | connectivity | recording | dimensionless |
| `beta_ppc` | connectivity | recording | dimensionless |
| `beta_imcoh` | connectivity | recording | dimensionless |
| `gamma_coherence` | connectivity | recording | dimensionless |
| `gamma_pli` | connectivity | recording | dimensionless |
| `gamma_wpli` | connectivity | recording | dimensionless |
| `gamma_plv` | connectivity | recording | dimensionless |
| `gamma_ppc` | connectivity | recording | dimensionless |
| `gamma_imcoh` | connectivity | recording | dimensionless |
| `global_coherence` | connectivity | recording | dimensionless |
| `global_pli` | connectivity | recording | dimensionless |
| `global_wpli` | connectivity | recording | dimensionless |
| `global_plv` | connectivity | recording | dimensionless |
| `global_ppc` | connectivity | recording | dimensionless |
| `global_imcoh` | connectivity | recording | dimensionless |
| `rms_cv` | variability | recording | coefficient of variation |
<!-- catalog:end -->
