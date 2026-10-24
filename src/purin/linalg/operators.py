"""Small JAX linear-operator interface; sparse storage remains explicit."""

from typing import NamedTuple
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from purin.linalg.sparse.operators import (
    _batched_local_jacobian_matvec,
    _batched_local_jacobian_transpose_matvec,
)


class DenseOperator(NamedTuple):
    """Matrix-vector and transpose products backed by a dense array."""

    matrix: object

    def mv(self, vector):
        """Apply this linear operator to a vector."""
        return jnp.einsum("...mn,...n->...m", self.matrix, vector)

    def tmv(self, vector):
        """Apply the transpose of this linear operator to a vector."""
        return jnp.einsum("...mn,...m->...n", self.matrix, vector)


class DiagonalOperator(NamedTuple):
    """Matrix-vector products represented by a diagonal vector."""

    diagonal: object

    def mv(self, vector):
        """Apply this linear operator to a vector."""
        return self.diagonal * vector

    def tmv(self, vector):
        """Apply the transpose of this linear operator to a vector."""
        return self.diagonal * vector


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["values", "support", "mask"],
    meta_fields=["variable_size"],
)
@dataclass(frozen=True)
class SparseOperator:
    """Matrix-vector and transpose products backed by a sparse matrix."""

    values: object
    support: object
    mask: object
    variable_size: int

    def mv(self, vector):
        """Apply this linear operator to a vector."""
        return _batched_local_jacobian_matvec(
            self.values, self.support, self.mask, vector
        )

    def tmv(self, vector):
        """Apply the transpose of this linear operator to a vector."""
        return _batched_local_jacobian_transpose_matvec(
            self.values, self.support, self.mask, vector, self.variable_size
        )
