# Validation datasets

These recordings are not part of the installed package. `scripts/fetch_validation_data.py`
downloads them into `external_data/`, which is gitignored and excluded from
the source distribution and the wheel. Set `EEGMATH_DATA` to write the
archives somewhere else. The script skips an archive or file
that already has the expected size, and it checks the Cambridge MD5 before
extracting. The Cambridge archive is 3,694,326,663 bytes. Both archives
together are several gigabytes. The notebooks use `EEGMATH_DATA` when it
is set, and otherwise search parent directories for `external_data/`.
They write new figures and tables to `validation_output/` at the
repository root when the working directory is inside that repository, or
to `EEGMATH_VALIDATION_OUT` if that variable is set. They do not
overwrite `docs/validation/summaries/`.

## Cambridge propofol sedation EEG

- Repository: https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df
- Data DOI: 10.17863/CAM.68959
- Paper: Chennu, O'Connor, Adapa, Menon, and Bekinschtein (2016), *Brain connectivity during propofol sedation*, PLoS Computational Biology. DOI 10.1371/journal.pcbi.1004669
- Licence: CC BY 2.0 UK
- Archive: `sedation-restingstate.zip`, 3694326663 bytes, MD5 `c38293499cdfe090fe2c1aafa687785c`
- Content URL used by the script: `https://api.repository.cam.ac.uk/server/api/core/bitstreams/e94a6722-da5b-4e53-8673-5e8ec106e0f7/content`

The archive contains 80 EEGLAB `.set`/`.fdt` pairs from 20 participants and
`datainfo.mat`. The files are MATLAB v7.3, so reading them needs
`pymatreader` and `h5py`, which are package dependencies. In the copy
inspected here each file has 91 channels, 250 Hz, and 10-second epochs. Epoch counts are not fixed at 39. Across the 80 files they run from 25 to 43, and the condition means are 37.8, 39.0, 37.6, and 39.8. The example header stores a high-pass of 0 Hz and a low-pass of 125 Hz.
MNE reports that low-pass as the Nyquist frequency. The source paper and
the repository description say the released series were filtered at
0.5–45 Hz and average-referenced. The validation scripts do not apply a
further filter. The spatial mean of a file is much smaller than a typical channel
standard deviation, which is what an average reference looks like. The
validation notebook does not band-pass or re-reference these files.

`datainfo.mat` has one row per file and five columns. The repository
description identifies them as the recording name, a level code 1–4,
a plasma concentration, a reaction time, and the number correct out of 40.
The notebook keeps that table beside the library and does not teach the
library those column names. Level codes are mapped there to the published
condition order baseline, mild, moderate, and recovery.

The file header labels Cz with a channel type MNE does not recognise and
therefore exposes as EEG. Reader warnings are stored on
`metadata["reader_note"]`.

## NEMAR nm000109

- Dataset: https://nemar.org/dataset/nm000109
- Version: v1.0.0
- Manifest: https://data.nemar.org/nm000109/v1.0.0/manifest.json
- DOI: 10.82901/nemar.nm000109
- Citation: Zyma et al. (2019), Electroencephalograms during mental arithmetic task performance, *Data* 4(1):14, as redistributed by NEMAR
- Licence: ODC-By 1.0
- Manifest: 258 files, 182837027 bytes. The manifest checksum algorithm is git. The fetch script verifies every published file size.

The BIDS tree has 72 EDF recordings, 36 participants, and tasks `rest` and
`mentalArithmetic`. Files inspected here are 21 channels at 500 Hz. Rest
durations run from 80 s to 188 s, and every arithmetic file is 62 s. The
EDF prefilter field is not one setting. MNE reports high-pass 0 Hz and
low-pass 250 Hz on 47 files, and high-pass 0.5 Hz and low-pass 45 Hz on
25 files. Nine subjects do not carry the same pair on the two tasks.
Subjects 08, 10, 13, 17, and 22 are labelled 0–250 Hz at rest and
0.5–45 Hz during arithmetic. Subjects 11, 16, 31, and 32 are labelled
the other way. Nineteen subjects are labelled 0–250 Hz on both tasks,
and eight are labelled 0.5–45 Hz on both. `EEGReference` is `n/a` in all 72
sidecars. The data paper describes a linked-ears reference and hardware
filters at 0.5–45 Hz with a 50 Hz notch. There is no `events.tsv`; the
condition is the BIDS task entity. `channels.tsv` marks ECG as ECG and records the channel unit as µV.
`load` still returns volts, because the EDF reader has already converted
the file. Relative power is a proportion of those volts squared. RMS is
in volts; multiply by `1e6` for microvolts. Do not multiply a relative
power by that factor. The
library keeps the ECG type and does not include it in an EEG average reference.
Each feature row copies the header limits to `highpass_hz` and
`lowpass_hz`, so the two header classes, including the nine subjects
whose tasks disagree, are visible in the table. `qc_summary` can name a
high-variance channel. That screen does not remove the channel, and a
sidecar status of `good` is left as it stands.
`participants.tsv` is UTF-8 with a byte-order mark; the reader accepts it.

The notebook does not apply another band-pass. The stored samples are used
as they are, including both header classes.

## What the two datasets are for

They differ in format, sampling rate, channel set, epoch structure, and
experimental design. The same `load` and `extract_features` calls read both.
A joint statistical model of the two experiments is not a supported
conclusion: `EEGCollection.require` raises when sampling rates or channel
sets differ, and the cross-dataset notebook stops at that check.
