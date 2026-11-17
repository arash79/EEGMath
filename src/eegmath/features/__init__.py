"""Feature catalog and extraction."""

from eegmath.features.extract import (
    ALIASES,
    FeatureConfig,
    aggregate_epochs,
    all_feature_names,
    extract_features,
    list_features,
    resolve_features,
)
from eegmath.features.regions import COARSE_10_20, aggregate_regions

__all__ = [
    "ALIASES",
    "COARSE_10_20",
    "FeatureConfig",
    "aggregate_epochs",
    "aggregate_regions",
    "all_feature_names",
    "extract_features",
    "list_features",
    "resolve_features",
]
