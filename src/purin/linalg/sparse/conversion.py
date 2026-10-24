"""Sparse conversion helpers executed only during static preparation."""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
import numpy as np

try:
    import scipy.sparse as scipy_sparse
except Exception:  # pragma: no cover
    scipy_sparse = None

try:
    from jax.experimental import sparse as jsparse
except Exception:  # pragma: no cover
    jsparse = None


def host_dense_array(matrix: Any, *, dtype=np.float64) -> np.ndarray:
    """Convert a host matrix to a dense NumPy array during preparation only."""

    if scipy_sparse is not None and scipy_sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=dtype)
    return np.asarray(matrix, dtype=dtype)


def as_bcoo(matrix: Any):
    """Convert dense or SciPy-sparse inputs to a JAX matrix representation."""

    if jsparse is None:
        return jnp.asarray(host_dense_array(matrix))
    if isinstance(matrix, jsparse.BCOO):
        return matrix
    if scipy_sparse is not None and scipy_sparse.issparse(matrix):
        return jsparse.BCOO.from_scipy_sparse(matrix.tocoo())
    return jnp.asarray(matrix)


__all__ = ["as_bcoo", "host_dense_array"]
