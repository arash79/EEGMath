# Getting started

eegmath works in an ordinary Python session. A recording is an `EEG`. A
group of recordings is an `EEGCollection`. Tables of features are pandas
DataFrames.

The shortest complete analysis is `examples/paired_alpha.py`. It builds a
synthetic rest-versus-task recording, extracts relative alpha and RMS, and
runs a paired Wilcoxon test. No download is required. Read that script
before the validation notebooks. Those notebooks need the external archives
described in [DATASETS.md](validation/DATASETS.md).

A continuous recording has `n_epochs` equal to 1, so a loop can visit it
once. `describe()` reports 0 in that field and `epoched=False`. The two
numbers are not a contradiction. `reduce_epochs=True` averages epochs that
already exist. It does not cut a continuous file into windows. Pass
`window_sec` for that. Welch spectra use 50% overlap. `window` and
`feature_table` default to no overlap, so a 2 s windowed estimate matches
continuous Welch only when `window_overlap=0.5` as well.

## Load

```python
from eegmath import load

one = load("recording.edf")                 # one EEG
several = load(["a.edf", "b.set"])          # EEGCollection
folder = load("path/to/bids_root")          # BIDS if dataset_description.json exists
array = load(samples, sfreq=250, ch_names=["Cz", "Pz"], units="uV")
```

`samples` has shape `(n_channels, n_times)` or `(n_epochs, n_channels, n_times)`.
MNE readers return volts. Pass `units="uV"` for an array that is already in
microvolts. Omitting `units` on an array labels it volts, and absolute
power is then wrong by `1e12` if the numbers were microvolts. A file or a
directory rejects `units`, because those readers are already in volts.

Relative power is a proportion either way. RMS and other amplitudes keep
the recording's unit: a file in volts stores RMS in volts, and a change of a
few microvolts prints near `1e-6`. Multiply an amplitude by `1e6` to print
microvolts. Absolute power is the square of that unit, so volts squared
becomes microvolts squared after multiplying by `1e12`. The `units`
column is that recording unit on every feature row, including a
relative-power row. One row holds many features, so `units` cannot be
the unit of each of them. `list_features()` names the quantity:
`alpha_relative` is a proportion, `rms` is the recording unit, and
`alpha_power` is that unit squared. Do not multiply `alpha_relative` by
`1e6` or `1e12`. The synthetic example is already in microvolts.

Non-EEG channels, including ECG, stay in the feature table until you drop
them. `bandpass` and average reference skip channels that are not typed as
EEG. Feature extraction does not. `detect_bad_channels` flags a non-finite
channel and a robust outlier of log-variance. The default threshold is a
robust z of 5, or a tenfold variance ratio when that scale is zero. The
screen does not drop or interpolate the channel, and the names are not a
column of the feature table. `qc_summary` reports them.

A BIDS directory accepts `subjects`, `tasks`, and `sessions`.
`feature_table` forwards those three arguments and raises if they are
passed to a file, an array, or a plain directory. Pass a list, such as
`subjects=["01"]` or `tasks=["rest"]`. A bare string is rejected, and the
example in the error matches the argument. A selection that matches
nothing names the labels that do exist. `task-rest` is not the task
label; `rest` is.

File header high-pass and low-pass are copied onto every feature row as
`highpass_hz` and `lowpass_hz`, with `duration_sec`. They describe the
file. Extraction does not filter. A missing header limit is missing. A
header that says 0 Hz stays 0, so a test can separate those recordings.
The same values remain on `recording.metadata`. A plain directory of
files does not stamp one identity onto every recording. A sibling
`*_eeg.json` stores a usable `EEGReference` as a label, a numeric
`PowerLineFrequency` when the file header has none, and
`EEGPlacementScheme`. `n/a` stays missing. The label does not
re-reference. A `status` of `bad` in `*_channels.tsv` is recorded in
`metadata["bad_channels"]` and the channel is kept. Nothing is filtered
during loading.

A BIDS filename such as `sub-01_task-rest_eeg.edf` fills `subject_id` and
`condition`. Other filenames keep the stem as `recording_id` and leave
subject and condition empty until you set them:

```python
one = one.spawn("identity", subject_id="01", condition="rest")
```

For a large directory, iterate instead of holding every recording:

```python
from eegmath import extract_features
from eegmath.io import iter_recordings

for recording in iter_recordings("path/to/files"):
    table = extract_features(recording, "spectral", reduce_epochs=True)
```

## Preprocess only when you mean to

```python
from eegmath.preprocess import bandpass, rereference, window

cleaned = rereference(bandpass(one, 0.5, 45.0), "average")
epochs = window(cleaned, window_sec=2.0, overlap=0.5)
```

Average reference on exactly two EEG channels makes one channel the
negative of the other. The function warns. Coherence, phase-locking value, and pairwise
phase consistency on that pair are then 1, and the lag measures are 0.
That identity is the reference. It is not coupling between the two sites.

`qc_summary(recording)` reports duration, sampling rate, amplitude scale,
and a robust channel screen. `detect_bad_channels` does not interpolate.
A channel with a missing sample is reported, and it is left out of the
scale so that one broken channel does not hide an intact outlier.
If a file is already filtered, leave it alone: a second filter is a
different recording, and the provenance list will say so.

`zscore` standardises each channel inside each epoch and sets
`units="z"`. That recording cannot be turned back into volts or
re-referenced. A later filter, resample, detrend, or baseline correction
warns, because it is not the z-score of the same operation on the
voltages. `window` on epochs keeps each parent epoch's event code and
event time on the windows cut from that epoch.

## Features

```python
from eegmath import extract_features, list_features

list_features()                              # names, families, scope
extract_features(epochs, "spectral")         # one family
extract_features(epochs, ["rms", "pe"])      # names; pe is permutation entropy
extract_features(epochs, "spectral", reduce_epochs=True)
```

Epoch rows are repeated observations of one recording. `reduce_epochs=True`
averages them and recomputes spectral ratios from the averaged band powers.
The mean of the epoch-wise ratios is a different number. Recording-level
measures such as connectivity and `rms_cv` are not copied onto every epoch;
the `scope` column says which rows they belong to.

Default bands are delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, and
gamma 30–45 Hz. Relative power is a share of the bands that overlap the
estimated spectrum. `freq_max` limits `total_power` and the spectral edge.
It does not drop a named band, so gamma can remain in the relative-power
denominator when `freq_max` is 20. The definitions, and the reason a tone
on a shared edge is split between two bands, are in [features](features.md).

Connectivity columns are missing, not zero, when fewer than 20 Fourier
segments are available. At the default 2 s segments and 50% overlap, 20 s
of continuous data yields 19 segments and 21 s yields 20. The call warns.

## Statistics

Pass the column names of your design. The function runs the test you name.

```python
from eegmath.stats import compare_groups, repeated_measures

compare_groups(
    table,
    ["alpha_relative"],
    group="condition",
    a="task",
    b="rest",
    subject="subject_id",
    test="wilcoxon",
)
repeated_measures(table, "alpha_relative", "condition", "subject_id", reference="rest")
```

`unit="subject"` is the default and averages technical replicates.
`unit="row"` is available when you are stating that each row is an
independent observation. If you omit `test`, the comparison is
Mann–Whitney, which is unpaired. A paired design needs `test="wilcoxon"`
or `repeated_measures`. An unpaired test raises if any subject appears
in both groups. A missing subject identifier raises. A group label that
is not in the table raises, as does a group with fewer than two finite
observations. Neither case returns `significant` False. A `channel`
column with more than one name also raises, unless `channel` is the
column you are comparing. An empty table raises as well, including
after a channel filter that matched nothing. Select the channel, or call
`aggregate_regions` from `eegmath`, before the test when the table is
still channel-wise. That function rebuilds ratios from absolute
`{band}_power` columns. If those columns are absent it warns and leaves
the ratios missing. A channel name in the region list that is not in
the table is skipped, and the call warns. `n_channels` counts only the
channels that were present. Finite p-values outside `[0, 1]` raise.

`significant` is the decision for the corrected p-value. The interval
beside it is a 95% t interval for the mean difference, including when the
test is Wilcoxon or Mann–Whitney. `alpha` changes the decision and does
not change the interval. The two can disagree in either direction:
a rejected test whose interval includes 0, or a test that is not
rejected after correction while its interval excludes 0. Adding a
second column to the same call can do the second of those, because
the correction is the family of that call. The call warns, and the
flag stays the corrected test. Read the mean difference, `n_a`,
`effect_name`, and the `interval` column before treating the flag as
the result. A zero-width interval means the observations did not vary.
The effect size is then missing, and the call says so.

## Signal measures without a feature table

```python
from eegmath.signal import (
    average_epochs,
    intertrial_phase_clustering,
    intertrial_phase_consistency,
    morlet_power,
    adf_stationarity,
    kpss_stationarity,
)
import numpy as np

# A 4 Hz, 7-cycle wavelet does not fit in the 2 s epochs above.
power, times = morlet_power(epochs, np.array([10.0, 20.0]), n_cycles=3.0)
evoked, evoked_times = average_epochs(epochs)
itpc, itpc_times = intertrial_phase_clustering(epochs, np.array([10.0]))
ppc, ppc_times = intertrial_phase_consistency(epochs, np.array([10.0]))
adf_stationarity(epochs.data[0, 0])
kpss_stationarity(epochs.data[0, 0])
```

`morlet_power` returns MNE's squared Morlet transform after the wavelet
is scaled to unit norm times √2 (Tallon-Baudry). That array is not Welch
band power and it is not in µV². A pure 10 µV sine has Welch variance 50.
The midpoint of a 3-cycle transform of that sine, 2 s at 250 Hz, is above
500. Compare values only inside one call's normalisation, and do not
multiply the array by `1e12`. A 4 Hz wavelet with seven cycles is longer than these 2 s epochs, and
the call raises `ParameterError`.

`adf_stationarity` tests a unit-root null. `kpss_stationarity` tests a
stationarity null. A KPSS p-value of 0.01 or 0.1 is the edge of the
published table, not a precise tail probability.

`average_epochs`, `intertrial_phase_clustering`, and
`intertrial_phase_consistency` require at least two epochs. They do not
read an event table. The shared clock is whatever alignment the epochs
already have. Cutting a continuous recording into windows supplies the
window onset as that clock. A periodic component that completes an
integer number of cycles in every window then has the same phase at
every onset, and both estimates equal 1. That is window-locked phase,
not an evoked response. Unrelated phases still produce a positive
clustering; the consistency removes that bias. None of these functions
concatenates across epoch boundaries. Intertrial phase consistency is
`(n P² − 1) / (n − 1)`, where `P` is the clustering value and `n` is
the number of epochs. It can be negative. When `P` is 0 the value is
`-1 / (n − 1)`. The clustering function stays the biased magnitude.
