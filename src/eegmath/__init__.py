"""Mathematical and statistical toolkit for EEG analysis.

eegmath loads recordings, applies only the preprocessing you request,
computes interpretable signal features, and runs statistical analyses on
those features or on tables you constructed elsewhere. It is not a
machine-learning framework and it does not assume a particular experiment.
"""

from eegmath.features import (
    FeatureConfig,
    aggregate_epochs,
    aggregate_regions,
    extract_features,
    list_features,
)
from eegmath.io import load
from eegmath.recording import EEG, EEGCollection
from eegmath.workflow import feature_table

__version__ = "1.0.0"

__all__ = [
    "EEG",
    "EEGCollection",
    "FeatureConfig",
    "aggregate_epochs",
    "aggregate_regions",
    "extract_features",
    "feature_table",
    "list_features",
    "load",
    "__version__",
]
