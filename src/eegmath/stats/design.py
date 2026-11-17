"""Design checks.

Technical replicates are rows that share a subject (and, when relevant, a
condition). They are not additional subjects. Functions that compare
groups average those rows when the unit of analysis is the subject, and
they refuse an unpaired test when the same subject appears in both groups.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from eegmath.errors import DesignError


def design_summary(
    frame: pd.DataFrame,
    *,
    subject: str | None = None,
    condition: str | None = None,
    session: str | None = None,
) -> dict:
    """Counts that describe the experimental units actually present."""
    summary: dict = {"n_rows": int(len(frame))}
    if subject and subject in frame.columns:
        summary["n_subjects"] = int(frame[subject].nunique(dropna=True))
        summary["rows_per_subject"] = frame.groupby(subject, dropna=False).size().describe().to_dict()
    if condition and condition in frame.columns:
        summary["conditions"] = frame[condition].value_counts(dropna=False).to_dict()
    if subject and condition and subject in frame.columns and condition in frame.columns:
        cell = frame.groupby([subject, condition], dropna=False).size()
        summary["max_rows_per_subject_condition"] = int(cell.max()) if len(cell) else 0
        summary["subjects_in_multiple_conditions"] = int(
            (frame.groupby(subject)[condition].nunique() > 1).sum()
        )
        presence = frame.pivot_table(index=subject, columns=condition, aggfunc="size", fill_value=0)
        summary["complete_subjects"] = int((presence > 0).all(axis=1).sum()) if presence.size else 0
    if session and session in frame.columns and subject in (frame.columns if subject else []):
        summary["sessions_per_subject"] = frame.groupby(subject)[session].nunique().describe().to_dict()
    return summary


def subject_means(
    frame: pd.DataFrame,
    values: Sequence[str],
    subject: str,
    by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Average ``values`` within subject and any additional keys in ``by``."""
    keys = [subject, *(by or [])]
    missing = [key for key in keys + list(values) if key not in frame.columns]
    if missing:
        raise DesignError(f"Columns not in the table: {missing}.")
    n_missing = int(frame[subject].isna().sum())
    if n_missing:
        raise DesignError(
            f"{n_missing} rows have no value in {subject!r}. "
            "Those rows would be averaged together as one subject. "
            "Drop them or assign an identifier to each experimental unit."
        )
    return frame.groupby(keys, dropna=False, as_index=False)[list(values)].mean()


def assert_independent_units(frame: pd.DataFrame, subject: str, condition: str) -> None:
    """Raise if any subject has more than one row inside a condition."""
    counts = frame.groupby([subject, condition], dropna=False).size()
    if (counts > 1).any():
        raise DesignError(
            "Some subjects contribute more than one row to a condition. "
            "Average technical replicates with subject_means(), model them with "
            "fit_mixed_model(), or pass unit='subject' to compare_groups(). "
            "Do not treat those rows as independent subjects."
        )
