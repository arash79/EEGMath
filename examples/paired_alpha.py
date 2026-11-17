"""Paired relative-alpha comparison on a synthetic recording.

No external data. Eight subjects, two conditions, two channels. Rest has a
larger 10 Hz sinusoid at Oz than task. The script prints a Wilcoxon table
for that channel.

File readers are different from this example. EDF, BDF, BrainVision, EEGLAB,
and FIF come back in volts, so RMS and absolute power are then in volts and
volts squared. Relative power does not depend on that scale. Multiply an
amplitude in volts by 1e6 for microvolts, and absolute power in volts
squared by 1e12 for microvolts squared. This example is already in
microvolts. Cohen's d_z on these sinusoids is large because the eight
paired differences barely vary. The mean difference is the number to
report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegmath import extract_features, load
from eegmath.stats import compare_groups


def synthetic_recording(subject: str, condition: str, rng: np.random.Generator):
    sfreq = 250.0
    times = np.arange(0.0, 20.0, 1.0 / sfreq)
    alpha_uv = 30.0 if condition == "rest" else 8.0
    occipital = alpha_uv * np.sin(2 * np.pi * 10.0 * times)
    frontal = 5.0 * np.sin(2 * np.pi * 6.0 * times)
    noise = rng.normal(0.0, 2.0, size=(2, times.size))
    samples = np.vstack([occipital, frontal]) + noise
    return load(
        samples,
        sfreq=sfreq,
        ch_names=["Oz", "Fz"],
        units="uV",
        subject_id=subject,
        condition=condition,
    )


def feature_table(rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(7) if rng is None else rng
    frames = []
    for index in range(8):
        subject = f"s{index:02d}"
        for condition in ("rest", "task"):
            recording = synthetic_recording(subject, condition, rng)
            frames.append(extract_features(recording, ["alpha_relative", "rms"]))
    return pd.concat(frames, ignore_index=True)


def main() -> pd.DataFrame:
    table = feature_table()
    oz = table.loc[table["channel"] == "Oz"].copy()
    result = compare_groups(
        oz,
        ["alpha_relative", "rms"],
        group="condition",
        a="task",
        b="rest",
        subject="subject_id",
        test="wilcoxon",
    )
    columns = [
        "feature",
        "mean_difference_a_minus_b",
        "ci_low",
        "ci_high",
        "p_value",
        "significant",
        "n_a",
        "n_b",
        "effect",
        "effect_name",
        "test",
        "unit",
    ]
    print(result.loc[:, columns].to_string(index=False))
    print("units:", ", ".join(sorted(oz["units"].astype(str).unique())), "(recording unit on every row, including the proportion)")
    print("Relative alpha is a proportion. RMS is in the recording unit. Do not rescale alpha_relative.")
    print("The effect column is Cohen's d_z. Report the mean difference, which is printed first.")
    print(
        "Each row is one 20 s recording. The spectrum inside that row is a Welch "
        "average of 19 Hann segments of 2 s at 50% overlap. Connectivity on this "
        "same length is missing: 20 segments need 21 s."
    )
    print(
        "Cohen's d_z divides the mean paired difference by the scatter of those "
        "eight differences. The sinusoids barely vary, so d_z is large. "
        "These samples are already in microvolts."
    )
    return result


if __name__ == "__main__":
    main()
