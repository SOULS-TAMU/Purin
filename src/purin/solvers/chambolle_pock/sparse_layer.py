"""Packed inequality operators for the shared CP forward and implicit adjoint.

Equality matrices stay dense. Static row supports and column adjacency reuse
PURIN's local sparse gather/reduce kernels; the CP algorithm is unchanged.
"""

from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from purin.linalg.sparse.operators import (
    _batched_local_jacobian_matvec,
    _batched_local_jacobian_transpose_gather_matvec,
    _batched_local_jacobian_transpose_bucketed_matvec,
    _build_column_gather_buckets,
    _build_column_gather_structure,
)
from .dense_implicit import _STATIC, _forward, _implicit


def _validated_support(rows, n, row_count):
    """Validate and pad the structural row layout outside JAX traces."""
    if len(rows) != row_count:
        raise ValueError("inequality_support must have one row per inequality.")
    for row in rows:
        if any(
            not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= n
            for i in row
        ):
            raise ValueError("Sparse support indices must be integers in [0, n).")
        if len(set(row)) != len(row):
            raise ValueError("Sparse support rows cannot contain duplicate indices.")
    width = max(1, max(map(len, rows), default=0))
    support = np.zeros((row_count, width), dtype=np.int32)
    mask = np.zeros_like(support, dtype=bool)
    for row_index, row in enumerate(rows):
        support[row_index, : len(row)] = row
        mask[row_index, : len(row)] = True
    return support, mask


@lru_cache(maxsize=128)
def packed_implicit_solver(rows, n, row_count, accelerated=False):
    """Prepare a fixed sparsity pattern shared by primal and adjoint products.

    Args:
        rows: Tuple of structural column-index tuples, one per inequality.
            Include every entry that can be nonzero throughout the domain.
        n: Number of primal variables.
        row_count: Number of inequality rows.
        accelerated: Use accelerated forward CP with the historical fixed
            implicit-adjoint policy.

    Returns:
        A dense-to-packed gather and a compiled implicit solver. Quadratic
        diagonals remain length-n vectors. Omitted matrix entries are structural
        constants, so derivatives exist only for the supplied sparse entries.

    Notes:
        Only NumPy layout metadata is cached. JAX constants are materialized
        inside each trace, preventing traced arrays from escaping through cache.
    """
    support, mask = _validated_support(rows, n, row_count)
    column_rows, column_slots, column_mask = _build_column_gather_structure(
        support, mask, n
    )

    # Avoid global padding when a few variables participate in most rows.
    use_buckets = column_rows.size > 4 * max(1, int(np.sum(mask)))
    buckets = _build_column_gather_buckets(support, mask, n) if use_buckets else ()

    def pack(matrix):
        values = jnp.take_along_axis(matrix, jnp.asarray(support)[None], axis=2)
        return jnp.where(jnp.asarray(mask)[None], values, 0.0)

    def operator_factory(a, values):
        def mv(y):
            equalities = jnp.einsum("bmn,bn->bm", a, y)
            inequalities = _batched_local_jacobian_matvec(
                values, jnp.asarray(support), jnp.asarray(mask), y
            )
            return jnp.concatenate((equalities, inequalities), axis=1)

        def tmv(dual):
            equalities = jnp.einsum("bmn,bm->bn", a, dual[:, : a.shape[1]])
            if use_buckets:
                inequalities = _batched_local_jacobian_transpose_bucketed_matvec(
                    values,
                    tuple(tuple(jnp.asarray(x) for x in bucket) for bucket in buckets),
                    dual[:, a.shape[1] :],
                    n,
                )
            else:
                inequalities = _batched_local_jacobian_transpose_gather_matvec(
                    values,
                    jnp.asarray(column_rows),
                    jnp.asarray(column_slots),
                    jnp.asarray(column_mask),
                    dual[:, a.shape[1] :],
                )
            return equalities + inequalities

        return mv, tmv

    solve, _, _ = _implicit(_forward(accelerated, operator_factory), operator_factory)
    return pack, jax.jit(solve, static_argnames=_STATIC)
