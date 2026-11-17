"""Channel groupings supplied by the caller.

The library does not assign published region labels on its own. Pass the
channel lists that match the montage and the paper you are citing.
``COARSE_10_20`` is only a convenience partition of common 10-20 and
10-10 names into five anatomical groups. It is not a clinical region of
interest from a particular study. T3/T4 and T5/T6 are included because
the modified combinatorial nomenclature renamed those 10-20 sites T7/T8
and P7/P8, and continued use of the older labels is explicitly
acceptable (ACNS Guideline 2; the same replacement is the 1991
American Electroencephalographic Society nomenclature, J Clin
Neurophysiol 8:200-202). A file that contains both names for one site
would count that site twice.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from eegmath.spectral.psd import (
    is_derived_spectral_column,
    recompute_power_ratios,
    warn_if_ratios_left_missing,
)

COARSE_10_20: dict[str, tuple[str, ...]] = {
    "frontal": (
        "Fp1", "Fp2", "Fpz", "AF3", "AF4", "AF7", "AF8",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Fz",
    ),
    "central": (
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
        "C1", "C2", "C3", "C4", "C5", "C6", "Cz",
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
    ),
    "temporal": ("T7", "T8", "T3", "T4", "FT7", "FT8", "FT9", "FT10", "TP7", "TP8", "TP9", "TP10"),
    "parietal": ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "T5", "T6", "Pz", "PO3", "PO4", "POz"),
    "occipital": ("O1", "O2", "Oz", "PO7", "PO8"),
}


def aggregate_regions(
    features: pd.DataFrame,
    regions: Mapping[str, Sequence[str]],
    *,
    how: str = "mean",
    recompute_ratios: bool = True,
    bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Average feature rows across channels that belong to each region.

    Rows are grouped by recording identity and epoch when those columns
    exist. Channels named in a region but absent from ``features`` are
    skipped, and the call warns once with those names. A region with no
    matching channels is omitted. ``n_channels``
    is the number of distinct channels that entered each recording's
    average, so a recording that lacks one electrode is not described as
    if it had the full region. Relative power, decibels, and band ratios
    are rebuilt from the aggregated absolute powers when ``bands`` is
    complete in the table. They are not the average of the channel-wise
    ratios. ``how="median"`` aggregates the powers first and then rebuilds
    the ratios. Duration and header filter limits stay with the recording
    identity. They are not averaged as features.
    """
    if "channel" not in features.columns:
        raise ValueError("features must contain a channel column.")
    if how not in {"mean", "median"}:
        raise ValueError("how must be 'mean' or 'median'.")
    id_cols = [
        c
        for c in (
            "recording_id",
            "subject_id",
            "session_id",
            "condition",
            "epoch",
            "scope",
            "sfreq",
            "units",
            "reference",
            "duration_sec",
            "highpass_hz",
            "lowpass_hz",
            "aggregation",
            "onset",
            "event_code",
        )
        if c in features.columns
    ]
    value_cols = [c for c in features.columns if c not in id_cols + ["channel"]]
    derived = [
        column for column in value_cols if recompute_ratios and is_derived_spectral_column(column)
    ]
    plain = [column for column in value_cols if column not in derived]
    rows: list[pd.DataFrame] = []
    present = set(features["channel"])
    missing_names: list[str] = []
    seen_missing: set[str] = set()
    for channels in regions.values():
        for name in channels:
            if name not in present and name not in seen_missing:
                seen_missing.add(name)
                missing_names.append(str(name))
    if missing_names:
        shown = ", ".join(missing_names[:8])
        extra = "" if len(missing_names) <= 8 else f", and {len(missing_names) - 8} more"
        warnings.warn(
            "aggregate_regions skipped channel names that are not in the table: "
            f"{shown}{extra}. n_channels counts only channels that were present. "
            "A skipped name does not enter the region mean.",
            UserWarning,
            stacklevel=2,
        )
    for region, channels in regions.items():
        use = [ch for ch in channels if ch in present]
        if not use:
            continue
        part = features.loc[features["channel"].isin(use)]
        if id_cols:
            grouped = part.groupby(id_cols, dropna=False)
            if plain:
                summary = grouped[plain].agg(how).reset_index()
            else:
                summary = grouped.size().reset_index(name="_rows").drop(columns="_rows")
            summary["n_channels"] = grouped["channel"].nunique().to_numpy()
        else:
            if plain:
                summary = pd.DataFrame([getattr(part[plain], how)()]).reset_index(drop=True)
            else:
                summary = pd.DataFrame(index=[0])
            summary["n_channels"] = int(part["channel"].nunique(dropna=False))
        for column in derived:
            summary[column] = np.nan
        summary.insert(0, "region", region)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=["region", *value_cols])
    out = pd.concat(rows, ignore_index=True)
    if recompute_ratios:
        out = recompute_power_ratios(out, bands=bands)
        warn_if_ratios_left_missing(out, derived)
    return out
