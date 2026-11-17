# eegmath

eegmath is a mathematical and statistical toolkit for EEG analysis. It loads
recordings, applies only the preprocessing you request, computes interpretable
descriptions of the signal, and runs statistical analyses on those
descriptions or on tables you built elsewhere.

The same functions cover a single file, a list of files, a directory, an
MNE object, or a BIDS dataset. Feature extraction does not filter the
recording. Statistical routines do not treat epochs or windows as subjects.

## Install

Python 3.10 or newer. The tests run on 3.10, 3.11, and 3.12. From a checkout of this repository, in a virtual environment:

```bash
pip install .
```

Optional extras:

```bash
pip install ".[parallel]"    # process pools, via joblib
pip install ".[dev]"         # pytest
pip install ".[notebooks]"   # Jupyter, for the validation notebooks
```

Installing the package does not download recordings. The distribution
name is `eegmath`. The current version is 1.0.0.

```python
import eegmath
eegmath.__version__
```

## A first analysis

`examples/paired_alpha.py` builds eight synthetic subjects, extracts
relative alpha and RMS at Oz, and runs a paired Wilcoxon test of task
versus rest. No recording file is required.

```bash
python examples/paired_alpha.py
```

The same comparison on recordings you already have:

```python
from eegmath import aggregate_regions, feature_table
from eegmath.stats import compare_groups

design = feature_table(
    "path/to/recordings",
    ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power", "alpha_relative", "rms"],
    reduce_epochs=True,
)
oz = design.loc[design["channel"] == "Oz"]
result = compare_groups(
    oz,
    ["alpha_relative", "rms"],
    group="condition",
    a="task",
    b="rest",
    subject="subject_id",
    test="wilcoxon",
)
occipital = aggregate_regions(design, {"occipital": ["O1", "O2", "Oz"]})
```

A few choices are easy to miss, and each one changes the sentence you can write afterwards.

- `"Oz"`, `"task"`, and `"rest"` have to be labels in the table. A channel or condition name that is not there raises `DesignError`. It does not come back as `significant` False.
- If you omit `test`, the comparison is Mann–Whitney, which is unpaired. A paired design needs `test="wilcoxon"` or `repeated_measures`.
- `reduce_epochs=True` averages epochs that already exist. It does not cut a continuous file into windows. Pass `window_sec` to do that. Welch spectra use 50% overlap, while `window` defaults to none, so set `window_overlap=0.5` when you want those estimators to match.
- File readers return volts. Relative alpha is a proportion, so it is not scaled. RMS and other amplitudes are in volts: multiply by `1e6` to print microvolts. Absolute power is in volts squared: multiply by `1e12` to print microvolts squared. The `units` column is the recording unit, copied onto every feature row, including relative alpha. One wide row cannot give each column its own unit. `list_features()` has a `quantity` column: `alpha_relative` is a proportion, `rms` is the recording unit, and `alpha_power` is that unit squared. Do not multiply `alpha_relative` by either factor. On a file, a power value is in volts squared while `units` still says `V`. The synthetic example is already in microvolts, so do not scale it again.
- `aggregate_regions` rebuilds ratios from absolute `{band}_power` columns. Relative alpha alone is not enough. The function warns and leaves the ratio missing. A channel name in the region that is not in the table is skipped, and the call warns. `n_channels` is the count that remains. Oz is not in every montage.
- Channels that are not EEG, including ECG, stay in the feature table until you drop them.

`compare_groups` also raises when a `channel` column still contains more than one name, and when the table passed to it has no rows. `significant` is the decision for the corrected p-value. The interval next to it is a 95% t interval for the mean difference, including when the test itself is Wilcoxon. `alpha` changes that decision and does not change the interval. The call warns when the flag and the interval disagree, in either direction. Adding a second column can move the flag without moving the interval. The flag stays the corrected test. In the synthetic example the printed effect is Cohen's d_z. Those sinusoids barely vary across the eight subjects, so d_z is large. The mean difference is the number to report. Each feature row also carries `duration_sec`, `highpass_hz`, and `lowpass_hz`. Those are facts about the recording. Extraction does not filter. A missing header limit is missing, not 0 Hz. A header that says 0 Hz stays 0.

Enter and leave at any stage. `load` only reads. `eegmath.preprocess` runs only when called. `extract_features` accepts an `EEG` you already cleaned. `compare_groups`, `repeated_measures`, and `prediction_probability` accept any pandas table with the columns you name.

## Layout

| Import | Role |
| --- | --- |
| `eegmath.load` | Arrays, MNE objects, EDF, BDF, BrainVision, EEGLAB, FIF, BIDS |
| `eegmath.preprocess` | Filter, reference, resample, window, detrend, quality summaries |
| `eegmath.extract_features` | Named features and the feature catalog |
| `eegmath.aggregate_regions` | Region means, with spectral ratios rebuilt from absolute power |
| `eegmath.spectral` | Welch spectra and spectral summaries |
| `eegmath.connectivity` | Sensor-space spectral connectivity |
| `eegmath.signal` | Time-frequency power, evoked measures, phase clustering, stationarity, autocorrelation, modulation index |
| `eegmath.stats` | Comparisons, linear and mixed models, concordance, resampling, error rates |
| `eegmath.viz` | Matplotlib figures that the caller chooses to show or save |
| `eegmath.feature_table` | Optional loader that streams a directory and extracts features |

Further reading:

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Features](docs/features.md)
- [Statistical methods](docs/statistics.md)
- [Method coverage](docs/methods.md)
- [Validation datasets](docs/validation/DATASETS.md)
- [Validation results](docs/validation/RESULTS.md)

## Validation data

Two public datasets are external checks. They are not in this repository
and not in the wheel. The Cambridge archive alone is 3,694,326,663 bytes.
Both archives together are several gigabytes.

```bash
python scripts/fetch_validation_data.py
```

Set `EEGMATH_DATA` to the directory that contains the two archive
folders named in the dataset note if you keep the files somewhere other
than `external_data/`. The fetch script reads that variable and
otherwise writes to `external_data/` at the repository root. Licences, citations, and checksums are in
[docs/validation/DATASETS.md](docs/validation/DATASETS.md). The notebooks
are
[notebooks/validation_cambridge.ipynb](notebooks/validation_cambridge.ipynb),
[notebooks/validation_nemar.ipynb](notebooks/validation_nemar.ipynb),
and
[notebooks/validation_cross_dataset.ipynb](notebooks/validation_cross_dataset.ipynb).
Re-running them writes to `validation_output/` at the repository root
when the working directory is inside the repository, or to
`EEGMATH_VALIDATION_OUT`. Install the notebook extra first. What the
recorded run produced is in
[docs/validation/RESULTS.md](docs/validation/RESULTS.md).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test run imports the installed package, not a checkout that happens
to sit on `PYTHONPATH`.

## Citation

If you use this software, cite it as eegmath. Author names are in
[CITATION.cff](CITATION.cff) and in the package metadata. That list is
the author list of the methodological manuscript that describes the
library. Contributor roles have not been divided between the paper and
the code, and the file does not include email addresses. This checkout
does not name a public repository or an issue tracker.

## Licence

MIT. See [LICENSE](LICENSE).
