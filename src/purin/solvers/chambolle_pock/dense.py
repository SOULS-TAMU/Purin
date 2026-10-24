"""Dense matrix adapter to the common CP iteration; no separate solver loop."""

import jax.numpy as jnp
from .kernels import sparse_cp


def dense_operators(A, C):
    """Block operators retain equality/inequality summation order."""

    def mv(y):
        return jnp.concatenate(
            (jnp.einsum("bmn,bn->bm", A, y), jnp.einsum("bmn,bn->bm", C, y)), axis=1
        )

    def tmv(dual):
        result = jnp.zeros((A.shape[0], A.shape[2]), dtype=dual.dtype)
        if A.shape[1]:
            result = result + jnp.einsum("bmn,bm->bn", A, dual[:, : A.shape[1]])
        if C.shape[1]:
            result = result + jnp.einsum("bmn,bm->bn", C, dual[:, A.shape[1] :])
        return result

    return mv, tmv


def solve_dense(
    q,
    c,
    A,
    b,
    C,
    d,
    lower,
    upper,
    *,
    accelerated=False,
    safety=0.95,
    norm_iterations=20,
    norm_seed=42,
    max_iterations=5000,
    tolerance=1e-9,
    y0=None,
    lambda0=None,
    mu0=None,
    equilibrate=None,
):
    """Adapt dense equality/inequality matrices to the shared sparse CP iteration."""
    arrays = [jnp.asarray(x) for x in (q, c, A, b, C, d, lower, upper)]
    dtype = jnp.result_type(*[x.dtype for x in arrays])
    q, c, A, b, C, d, lower, upper = (x.astype(dtype) for x in arrays)
    batch, n = q.shape
    me, mi = A.shape[1], C.shape[1]
    if equilibrate is not None:
        q, c, A, b, C, d, lower, upper, columns, eq_scale, iq_scale = equilibrate(
            q, c, A, b, C, d, lower, upper
        )
    else:
        columns = jnp.ones_like(q)
        eq_scale, iq_scale = jnp.ones_like(b), jnp.ones_like(d)
    primal = (
        jnp.zeros_like(q)
        if y0 is None
        else jnp.asarray(y0, dtype).reshape((batch, n)) / columns
    )
    eq = (
        jnp.zeros_like(b)
        if lambda0 is None
        else jnp.asarray(lambda0, dtype).reshape((batch, me))
        / jnp.where(eq_scale == 0.0, 1.0, eq_scale)
    )
    iq = (
        jnp.zeros_like(d)
        if mu0 is None
        else jnp.asarray(mu0, dtype).reshape((batch, mi))
        / jnp.where(iq_scale == 0.0, 1.0, iq_scale)
    )
    result = sparse_cp(
        q,
        c,
        None,
        jnp.concatenate((b, d), axis=1),
        None,
        None,
        me,
        lower,
        upper,
        accelerated=accelerated,
        safety=safety,
        max_iterations=max_iterations,
        tolerance=tolerance,
        norm_iterations=norm_iterations,
        norm_seed=norm_seed,
        y0=primal,
        dual0=jnp.concatenate((eq, iq), axis=1),
        operator_pair=dense_operators(A, C),
        factored_proximal=True,
    )
    return (
        columns * result.y,
        eq_scale * result.dual[:, :me],
        iq_scale * result.dual[:, me:],
    )
