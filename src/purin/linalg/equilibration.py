"""Reusable batched Ruiz equilibration for diagonal quadratic programs."""

import jax
import jax.numpy as jnp


def max_abs_rows(matrix):
    """Compute maximum absolute matrix entries along each row."""
    if matrix.shape[1] == 0:
        return jnp.zeros((matrix.shape[0], 0), dtype=matrix.dtype)
    return jnp.max(jnp.abs(matrix), axis=2)


def max_abs_columns(matrix):
    """Compute maximum absolute matrix entries along each column."""
    if matrix.shape[1] == 0:
        return jnp.zeros((matrix.shape[0], matrix.shape[2]), dtype=matrix.dtype)
    return jnp.max(jnp.abs(matrix), axis=1)


def ruiz_equilibrate(
    Q_diag, c, A, b, C, d, l, u, *, iterations: int = 4, eps: float = 1e-6
):
    """Equilibrate a batched diagonal QP with iterative row and column scaling."""
    arrays = [jnp.asarray(x) for x in (Q_diag, c, A, b, C, d, l, u)]
    dtype = jnp.result_type(*[x.dtype for x in arrays])
    Qs, cs, As, bs, Cs, ds, ls, us = (x.astype(dtype) for x in arrays)
    columns = jnp.ones_like(Qs)
    equalities, inequalities = jnp.ones_like(bs), jnp.ones_like(ds)
    epsilon = jnp.asarray(eps, dtype)

    def step(_, state):
        q, c, A, b, C, d, lower, upper, columns, equalities, inequalities = state
        metric = jnp.maximum(max_abs_columns(A), max_abs_columns(C))
        metric = jnp.maximum(metric, jnp.abs(q))
        update = 1.0 / jnp.sqrt(jnp.maximum(metric, epsilon))
        A, C = A * update[:, None, :], C * update[:, None, :]
        q, c = q * (update**2), c * update
        lower, upper, columns = lower / update, upper / update, columns * update
        if A.shape[1]:
            update = 1.0 / jnp.sqrt(jnp.maximum(max_abs_rows(A), epsilon))
            A, b, equalities = A * update[:, :, None], b * update, equalities * update
        if C.shape[1]:
            update = 1.0 / jnp.sqrt(jnp.maximum(max_abs_rows(C), epsilon))
            C, d, inequalities = (
                C * update[:, :, None],
                d * update,
                inequalities * update,
            )
        return q, c, A, b, C, d, lower, upper, columns, equalities, inequalities

    return jax.lax.fori_loop(
        0,
        iterations,
        step,
        (Qs, cs, As, bs, Cs, ds, ls, us, columns, equalities, inequalities),
    )
