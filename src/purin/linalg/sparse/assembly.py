"""Fixed-shape sparse numerical assembly helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp


def masked_diagonal(values, mask, *, dummy_value: float = -1.0):
    """Return physical values on retained rows and a stable dummy diagonal."""
    values = jnp.asarray(values)
    mask = jnp.asarray(mask, dtype=bool)
    return jnp.where(mask, values, jnp.asarray(dummy_value, values.dtype))


try:
    from jax.experimental import sparse as jsparse
except Exception:  # pragma: no cover
    jsparse = None


def pattern_indices(
    pattern, *, row_count: int, column_count: int, transpose: bool = False
):
    """Return static COO indices for a row-support sparsity pattern.

    ``pattern[row]`` lists the structural column indices for that row.  When
    ``transpose`` is true, the returned indices represent the transposed
    matrix without changing the numerical-value order.
    """

    if pattern is None:
        pattern = tuple(tuple(range(column_count)) for _ in range(row_count))
    indices: list[tuple[int, int]] = []
    for row, support in enumerate(pattern):
        for column in support:
            indices.append(
                (int(column), int(row)) if transpose else (int(row), int(column))
            )
    if indices:
        return jnp.asarray(indices, dtype=jnp.int32)
    return jnp.empty((0, 2), dtype=jnp.int32)


@dataclass(frozen=True)
class FixedSparseMatrixFactory:
    """Build fixed-pattern JAX sparse matrices from ordered nonzero values."""

    indices: Any
    shape: tuple[int, int]

    def build_from_values(self, values):
        """Assemble a sparse matrix from values in the factory's fixed pattern order."""
        values = jnp.asarray(values)
        if jsparse is None:
            dense = jnp.zeros(self.shape, dtype=values.dtype)
            if self.indices.shape[0] == 0:
                return dense
            return dense.at[self.indices[:, 0], self.indices[:, 1]].set(values)
        return jsparse.BCOO((values, self.indices), shape=self.shape)

    def gather_values(self, row_major_matrix):
        """Gather values in the factory's COO order from a dense matrix."""
        if self.indices.shape[0] == 0:
            return jnp.empty((0,), dtype=row_major_matrix.dtype)
        return row_major_matrix[self.indices[:, 0], self.indices[:, 1]]


def transposed_jacobian_factories(canonical):
    """Create fixed factories for ``A=J_h.T`` and ``C=J_g.T``."""

    return (
        FixedSparseMatrixFactory(
            indices=pattern_indices(
                canonical.equality_sparsity,
                row_count=canonical.equality_size,
                column_count=canonical.variable_size,
                transpose=True,
            ),
            shape=(canonical.variable_size, canonical.equality_size),
        ),
        FixedSparseMatrixFactory(
            indices=pattern_indices(
                canonical.inequality_sparsity,
                row_count=canonical.inequality_size,
                column_count=canonical.variable_size,
                transpose=True,
            ),
            shape=(canonical.variable_size, canonical.inequality_size),
        ),
    )
