# Validation results

Signal analyses that are not catalog columns were checked on constructed
series outside this package (seed 20261026). The numbers below are that
record. They are not recomputed by the notebooks here. A 10 Hz burst is localised by Morlet
power. Eight phase-locked epochs give clustering and consistency 1.
Eight random phases give clustering 0.419 and consistency 0.058. Eight
epochs split between opposite phases give consistency −1/7. A continuous
10 Hz tone cut into 1 s windows gives clustering 1, because each window
holds an integer number of cycles. The same tone cut at 1.05 s gives
±1/15. A coupled series has modulation index 0.077 against
2.3e-6 for the carrier. An autoregressive coefficient 0.7 is recovered
by the autocorrelation and the partial autocorrelation. Dickey–Fuller
and KPSS point in opposite directions on white noise and on its
cumulative sum, and the KPSS probabilities are table bounds. These
checks are not the public-archive analyses below.

The methodological manuscript does not use the notebook means below as its
public-data estimates. A region summary kept outside this package rebuilds
relative power after averaging absolute power within caller-defined
regions. The notebooks instead average
channel-wise relative powers. On the Cambridge occipital rows the two
summaries differ by a median of 0.0038 and a maximum of 0.077. On NEMAR
the notebook mean of channel-wise alpha relatives is 0.368 at rest and
0.288 during arithmetic; the rebuilt region values are 0.386 and 0.298.
Both summaries were reproducible in that external check. They are not the same operation.
The same external check reports rebuilt relative alpha of
0.554, 0.354, 0.381, and 0.191 for the Cambridge responsive and drowsy
groups at baseline and moderate sedation. Those are the group means of
the rebuilt relative alpha, not a second estimator.

These tables are the validation record for eegmath 1.0.0. The cohort
notebooks were executed on 26 October 2026 from the installed wheel.
The interpreter was Python 3.10.18, with NumPy 2.2.6, SciPy 1.15.3, and
MNE 1.12.1. `PYTHONPATH` was unset, and `import eegmath` resolved to
that environment's `site-packages`. The three notebooks completed
without a cell error. The cohort tables match the numbers below. The
Cambridge connectivity cell reports 741 segments and a finite alpha
coherence matrix. It does not print the weighted phase-lag index range.
The notebooks do not call the modulation index or intertrial phase
consistency. Summary tables and figures are in
`docs/validation/summaries/`. Running the notebooks again writes to
`validation_output/` at the repository root when the working directory
is inside that repository, or to the directory in
`EEGMATH_VALIDATION_OUT`, and does not replace those recorded files.

These numbers are the output of the analyses below. They were not used
as thresholds that the code had to hit. A later run of this release on
the same files should match them up to numerical noise. A different
scientific question will produce different numbers, including
non-significant ones, and that is not a software failure.

## What was run

| Notebook | Recordings | What it checks |
| --- | --- | --- |
| `validation_cambridge.ipynb` | All 80 EEGLAB files, 20 subjects, four conditions | EEGLAB epochs, no extra filter or reference, spectral features, one-epoch entropy and Hjorth, connectivity on six channels, an external feature table, a random-intercept model, Wilcoxon on subject means |
| `validation_nemar.ipynb` | All 72 EDF files, 36 subjects, both tasks | BIDS labels, ECG channel type, header filters left in place, spectral features, a file list, Morlet and ADF on a 10 s crop, Wilcoxon on subject means, Spearman with age |
| `validation_cross_dataset.ipynb` | One file from each dataset | Shared loader, incompatible sampling rates rejected, shared channel names, no pooled test |

## Cambridge

The example file is epoched: 39 epochs in that file, with counts from 25 to 43 across the archive, 91 channels, 250 Hz, 10 s, in volts.
Provenance contains the load step and does not contain a band-pass
or a re-reference. The header high-pass is 0 Hz and the low-pass is 125 Hz.
The publication describes 0.5–45 Hz preprocessing; that description is not
what the header stores, so the notebook does not filter again. Median
channel standard deviation is about 7 µV. The reader note records MNE's
warning that Cz was labelled with an unknown type and that some epochs
carry more than one event.

Occipital relative alpha on the example file is 0.508 (O1), 0.686 (O2),
and 0.631 (Oz). Alpha peak frequencies are 10.6, 10.4, and 10.8 Hz.
Median frequency equals SEF50 on each channel (8.5, 10.2, and 10.1 Hz).
The PSD figure is the mean of the epoch spectra. It peaks near 10 Hz,
shows a harmonic near 20 Hz, and declines above that. A slow component
remains below 4 Hz, which agrees with the file header's high-pass of 0 Hz.

On one Oz epoch, permutation entropy is 0.643 and Hjorth mobility is
64.8 rad/s, which is about 10.3 Hz. That sits next to the alpha peak of
the same recording. RMS of that epoch is 10.3 µV and line length is
536 µV/s. The feature table stores volts; the notebook converts those
two columns for display.

Connectivity on F3, F4, P3, P4, O1, and O2 used 1 s segments pooled across
epochs: 741 segments, and the alpha coherence matrix is finite.

Subject-level occipital means. Each value is the mean of the
channel-wise relative powers at O1, O2, and Oz:

| Condition | Mean alpha relative | SD | n |
| --- | --- | --- | --- |
| baseline | 0.492 | 0.166 | 20 |
| mild | 0.469 | 0.178 | 20 |
| moderate | 0.299 | 0.144 | 20 |
| recovery | 0.551 | 0.105 | 20 |

Every subject has all four conditions, one row each. The random-intercept
model converged. The moderate-minus-baseline contrast is −0.194
(SE 0.029, df 16, interval −0.256 to −0.132). The degrees of freedom are
20 subjects minus 4 fixed effects. Wilcoxon on the same subject means
gives the same mean difference, −0.194, with a paired effect of −1.45.
The paired mean-difference interval is −0.256 to −0.131. The two
procedures agree on this contrast. The omnibus likelihood-ratio test and
the Wilcoxon p-values are small. That agreement is a check that the
design was passed through correctly. It is the result of this occipital
summary, not a property the package guarantees on other data.

Theta/alpha on the same contrast has Wilcoxon mean difference +0.154
(Student-t interval −0.056 to +0.363, paired effect +0.34). The rank
test p-value is 0.001 and the mean-difference interval includes zero.
Those are different summaries of the same pairs: the interval is not a
Wilcoxon interval. Spearman correlation of moderate-condition alpha with
reaction time uses 18 subjects: two moderate rows have no finite reaction
time in `datainfo.mat`. The coefficient is −0.03.

## NEMAR nm000109

Subject 00 has both tasks. Rest is 182 s and mental arithmetic is 62 s,
both 21 channels at 500 Hz in volts. The ECG channel has type `ecg`.
This subject's header is high-pass 0.5 Hz and low-pass 45 Hz. That pair
is one of two classes in the archive: 47 files are labelled 0–250 Hz and
25 are labelled 0.5–45 Hz. The notebook does not add another band-pass.
Age for this participant is 21. The example PSD, after dropping
ECG, peaks near 10 Hz, with the largest occipital peak at O2. RMS on
those five channels is 12–17 µV.

Five-channel subject means:

| Condition | Mean alpha relative | SD | Mean theta relative | SD | n |
| --- | --- | --- | --- | --- | --- |
| rest | 0.368 | 0.138 | 0.142 | 0.038 | 36 |
| mental arithmetic | 0.288 | 0.137 | 0.147 | 0.040 | 36 |

Wilcoxon of task minus rest, one row per subject: alpha −0.080
(interval −0.120 to −0.040, paired effect −0.67); theta +0.005
(interval −0.009 to +0.019, paired effect +0.12). Benjamini-Hochberg
is applied to those two columns only. The alpha interval excludes zero.
The theta interval includes it, and the theta rank-test p-value is 0.72.
That alpha test pools all 36 subjects, including the nine whose two
tasks do not share a header class. The same five-channel mean on the
eight subjects labelled 0.5–45 Hz on both tasks is −0.081 (interval
−0.168 to +0.007, p = 0.078). The nineteen subjects labelled 0–250 Hz
on both tasks are −0.096 (interval −0.147 to −0.045, p = 0.0012). The
pooled sentence is not the matched-header sentence.

The later manuscript partition is a different five-channel summary and
is not this notebook table. In that partition, absolute delta has mean
+6.7 µV² only because subject 1 changes by +437 µV². The median is
−0.7 µV² (17 up, 19 down, Wilcoxon p = 0.23). Without subject 1 the mean
is −5.7 µV² (16 up, 19 down, p = 0.12), while relative delta remains
positive at +0.063. Absolute theta at Fz has pooled mean +1.1 µV² and
median −1.1 µV². On the 19 matched 0–250 Hz subjects that Fz mean is
+2.7 µV² (Wilcoxon p = 0.44). On the 8 matched 0.5–45 Hz subjects every
change is negative and the mean is −3.5 µV² (p = 0.0078). The pooled
five-channel absolute-theta rank test is concentrated in the narrower
class. Relative theta on the five-channel mean stays near zero in both
matched classes.
Spearman correlation of the within-subject alpha change with age is
approximately 0 (n = 36).

A two-file list for subject 00 returns both task labels through
`feature_table`. Morlet power on a 10 s Cz crop has the expected time
length. The ADF test on that crop rejects a unit root; the notebook
prints the caveat that this is not a general stationarity result.

## Cross-dataset

The two example recordings are 250 Hz and 500 Hz, both in volts, with
19 channel names in common. `require(same_sfreq=True, same_channels=True)`
raises `DataError` and names both mismatches. `align_channels` keeps the
intersection. The same feature names are extracted from each recording.
No test is fitted to the concatenated table.

## Outstanding

These methods are implemented and covered by synthetic tests. The public
recordings do not supply the design they need, so they were not marked
as confirmed on real data:

- One-dimensional cluster permutation of a within-subject time course.
  The Cambridge epochs are resting segments, and NEMAR has no events file.
- Modulation index. Same reason: there is no stimulus-locked amplitude
  time course to interpret.
- Evoked average and peak latency. No event-related design was present.
- Phase-locking value, pairwise phase consistency, and imaginary
  coherency on these files. Coherence and weighted phase-lag index were
  the connectivity pair computed in the Cambridge notebook. Pairwise
  phase consistency was checked on synthetic tones, including a negative
  value of −1. Imaginary-coherency node features average the absolute
  value across frequency; that path was checked on synthetic
  opposite-lag tones.
- The KPSS stationarity test. The notebooks call the augmented
  Dickey-Fuller test only. The KPSS size check is a synthetic count
  kept outside this package.
- Sensor-space correlation, ordinary least squares, intertrial phase
  clustering, and intertrial phase consistency. The resting recordings
  are not event-aligned, so neither phase measure was applied to them.
  The regression and correlation checks use synthetic tables. Two
  identical synthetic epochs give consistency near 1. Two epochs in
  antiphase give consistency near −1 while the clustering magnitude
  stays near 0.
- Bootstrap intervals, partial correlation, permutation tests other than
  the Wilcoxon already reported, and the BIC approximation. They remain
  on synthetic tables.
- Sample autocorrelation and Yule-Walker partial autocorrelation. The
  notebooks do not call them. The AR(1) check is synthetic.

Excluded methods, including source reconstruction, automatic ICA, and a
FOOOF model, are listed in `docs/methods.md` and were not run.
