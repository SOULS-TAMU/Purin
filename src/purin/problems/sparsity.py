"""Static sparse structures prepared once by the problem frontend."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def transpose_gather_padding_fraction(
    equality_support: np.ndarray,
    equality_mask: np.ndarray,
    inequality_support: np.ndarray,
    inequality_mask: np.ndarray,
    n_variables: int,
) -> float:
    """Estimate padding in a column-gather transpose representation."""
    supports: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for support, mask in (
        (equality_support, equality_mask),
        (inequality_support, inequality_mask),
    ):
        support = np.asarray(support, dtype=np.int32)
        mask = np.asarray(mask, dtype=bool)
        if support.ndim == 2 and support.shape[0]:
            supports.append(support)
            masks.append(mask)
    if not supports or n_variables <= 0:
        return 0.0
    support = np.concatenate(supports, axis=0)
    mask = np.concatenate(masks, axis=0)
    degrees = np.bincount(support[mask], minlength=n_variables)
    width = int(np.max(degrees, initial=0))
    capacity = n_variables * width
    return 0.0 if capacity == 0 else float((capacity - int(np.sum(mask))) / capacity)


def transpose_bucket_padding_fraction(
    equality_support: np.ndarray,
    equality_mask: np.ndarray,
    inequality_support: np.ndarray,
    inequality_mask: np.ndarray,
    n_variables: int,
) -> float:
    """Estimate padding for power-of-two degree-bucketed transpose gathers."""
    supports: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for support, mask in (
        (equality_support, equality_mask),
        (inequality_support, inequality_mask),
    ):
        support = np.asarray(support, dtype=np.int32)
        mask = np.asarray(mask, dtype=bool)
        if support.ndim == 2 and support.shape[0]:
            supports.append(support)
            masks.append(mask)
    if not supports or n_variables <= 0:
        return 0.0
    support = np.concatenate(supports, axis=0)
    mask = np.concatenate(masks, axis=0)
    degrees = np.bincount(support[mask], minlength=n_variables)
    capacity = 0
    valid = 0
    for degree_value in degrees:
        degree = int(degree_value)
        if degree:
            capacity += 1 << (degree - 1).bit_length()
            valid += degree
    return 0.0 if capacity == 0 else float((capacity - valid) / capacity)


@dataclass(frozen=True)
class ProjectionStructure:
    """Static sparse structure shared by every problem in a projection batch.

    The padded supports and masks are computed once in Python before JIT
    compilation.  Numerical iterations update only local derivative values.
    """

    target_parameter_indices: tuple[int, ...]
    equality_variable_indices: tuple[tuple[int, ...], ...]
    inequality_variable_indices: tuple[tuple[int, ...], ...]
    equality_jacobian_rows: tuple[int, ...]
    equality_jacobian_cols: tuple[int, ...]
    inequality_jacobian_rows: tuple[int, ...]
    inequality_jacobian_cols: tuple[int, ...]
    max_row_nonzeros: int
    equality_support: tuple[tuple[int, ...], ...]
    equality_support_mask: tuple[tuple[bool, ...], ...]
    inequality_support: tuple[tuple[int, ...], ...]
    inequality_support_mask: tuple[tuple[bool, ...], ...]
    constraints_affine_in_variables: bool
    estimated_reduced_kkt_density: float
    estimated_condensed_kkt_density: float

    @property
    def jacobian_nonzeros(self) -> int:
        """Return the number of stored constraint-Jacobian entries."""
        return sum(map(len, self.equality_variable_indices)) + sum(
            map(len, self.inequality_variable_indices)
        )
