"""Small host-side validation helpers used during static preparation."""

from __future__ import annotations
import numpy as np
import math
from itertools import zip_longest


def positive_integer(value, name: str) -> int:
    """Validate that a configuration value is a strictly positive integer."""
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value}.")
    return value


def nonnegative_float(value, name: str) -> float:
    """Validate that a configuration value is a finite nonnegative float."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative; got {value}.")
    return value


def positive_float(value, name: str) -> float:
    """Validate that a configuration value is a finite positive float."""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value}.")
    return value


def open_unit_interval(value, name: str) -> float:
    """Validate that a configuration value lies strictly between zero and one."""
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1; got {value}.")
    return value


_STRICT_ZIP_SENTINEL = object()


def strict_zip(*iterables):
    """Zip iterables while rejecting unequal lengths on Python 3.9+.

    This provides strict length checking for ``zip`` on Python 3.9 and later. The length check is performed lazily, matching ``zip``.
    """

    for values in zip_longest(*iterables, fillvalue=_STRICT_ZIP_SENTINEL):
        if any(value is _STRICT_ZIP_SENTINEL for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values


def validate_metric_diagonal(value):
    """Validate a positive diagonal metric against the projection variable dimension."""
    if value is None:
        return None
    weights = np.asarray(value, dtype=np.float64)
    if (
        weights.ndim != 1
        or weights.size == 0
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "metric_diagonal must be a nonempty vector of positive finite weights"
        )
    return tuple(float(v) for v in weights)
