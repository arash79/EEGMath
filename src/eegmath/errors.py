"""Public exceptions for eegmath."""

from __future__ import annotations


class EEGMathError(Exception):
    """Base class for errors raised by eegmath."""


class DataError(EEGMathError, ValueError):
    """The input is not a usable EEG recording or table."""


class ParameterError(EEGMathError, ValueError):
    """A parameter is outside the domain of a method."""


class DependencyError(EEGMathError, ImportError):
    """An optional scientific dependency is not installed."""


class DesignError(EEGMathError, ValueError):
    """The table does not match the statistical design being requested."""
