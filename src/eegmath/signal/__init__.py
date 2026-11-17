"""Analyses of the signal itself."""

from eegmath.signal.analysis import (
    adf_stationarity,
    autocorrelation,
    autocorrelation_table,
    average_epochs,
    band_power_timecourse,
    modulation_index,
    modulation_index_channels,
    intertrial_phase_clustering,
    intertrial_phase_consistency,
    kpss_stationarity,
    morlet_power,
    partial_autocorrelation,
    peak_measure,
    stationarity_table,
)

__all__ = [
    "adf_stationarity",
    "autocorrelation",
    "autocorrelation_table",
    "average_epochs",
    "band_power_timecourse",
    "intertrial_phase_clustering",
    "intertrial_phase_consistency",
    "kpss_stationarity",
    "modulation_index",
    "modulation_index_channels",
    "morlet_power",
    "partial_autocorrelation",
    "peak_measure",
    "stationarity_table",
]
