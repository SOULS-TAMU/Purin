"""Historical dense CP signatures forwarding to the single shared CP loop."""

from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from purin.linalg.equilibration import ruiz_equilibrate
from purin.solvers.chambolle_pock.dense import solve_dense, dense_operators
from purin.solvers.chambolle_pock.kernels import norm_estimate, diagnostics


def _init_x(key, batch_size, n, dtype):
    keys = jax.random.split(key, batch_size)
    x = jax.vmap(lambda k: jax.random.normal(k, (n,), dtype=dtype))(keys)
    return x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def canonicalize_constraints(Q_diag, A, b, C, d):
    """Broadcast dense constraint arrays to the canonical batched QP shapes."""
    q = jnp.asarray(Q_diag)
    batch, n = q.shape
    A, b = (
        (jnp.zeros((batch, 0, n), q.dtype), jnp.zeros((batch, 0), q.dtype))
        if A is None
        else (jnp.asarray(A, q.dtype), jnp.asarray(b, q.dtype))
    )
    C, d = (
        (jnp.zeros((batch, 0, n), q.dtype), jnp.zeros((batch, 0), q.dtype))
        if C is None
        else (jnp.asarray(C, q.dtype), jnp.asarray(d, q.dtype))
    )
    return A, b, C, d


def estimate_AC_norm(A, C, key, *, iters: int = 50):
    """Estimate the norm of the stacked equality and inequality operator."""
    A, C = jnp.asarray(A), jnp.asarray(C)
    dtype = jnp.result_type(A.dtype, C.dtype)
    return norm_estimate(
        *dense_operators(A.astype(dtype), C.astype(dtype)),
        A.shape[0],
        A.shape[2],
        dtype,
        iterations=iters,
        key=key,
    )


def _primal_residual(y, A, b, C, d, l, u):
    mv, tmv = dense_operators(A, C)
    return diagnostics(
        y,
        jnp.zeros_like(jnp.concatenate((b, d), axis=1)),
        jnp.ones_like(y),
        jnp.zeros_like(y),
        jnp.concatenate((b, d), axis=1),
        l,
        u,
        mv,
        tmv,
        A.shape[1],
    )[0]


def _primal_dual_gap(Q_diag, c, A, b, C, d, l, u, y, lam, mu):
    return diagnostics(
        y,
        jnp.concatenate((lam, mu), axis=1),
        Q_diag,
        c,
        jnp.concatenate((b, d), axis=1),
        l,
        u,
        *dense_operators(A, C),
        A.shape[1],
    )[1]


def _prepare_scaled_problem(Q_diag, c, A, b, C, d, l, u, *, use_ruiz, ruiz_iters):
    if use_ruiz:
        return ruiz_equilibrate(Q_diag, c, A, b, C, d, l, u, iterations=ruiz_iters)
    return (
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        jnp.ones_like(Q_diag),
        jnp.ones_like(b),
        jnp.ones_like(d),
    )


def CP_fixed(
    Q_diag,
    c,
    A,
    b,
    C,
    d,
    l,
    u,
    *,
    safety: float = 0.95,
    knorm_iters: int = 20,
    knorm_seed: int = 42,
    max_iter: int = 5000,
    tol: float = 1e-9,
    y0=None,
    lam0=None,
    mu0=None,
    use_ruiz: bool = False,
    ruiz_iters: int = 4,
):
    """Adapt the retained fixed-step CP signature to the shared dense solver."""
    return solve_dense(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        safety=safety,
        norm_iterations=knorm_iters,
        norm_seed=knorm_seed,
        max_iterations=max_iter,
        tolerance=tol,
        y0=y0,
        lambda0=lam0,
        mu0=mu0,
        equilibrate=partial(ruiz_equilibrate, iterations=ruiz_iters)
        if use_ruiz
        else None,
    )


def CP_accelerated(
    Q_diag,
    c,
    A,
    b,
    C,
    d,
    l,
    u,
    *,
    safety: float = 0.95,
    knorm_iters: int = 20,
    knorm_seed: int = 42,
    max_iter: int = 5000,
    tol: float = 1e-9,
    y0=None,
    lam0=None,
    mu0=None,
    use_ruiz: bool = False,
    ruiz_iters: int = 4,
):
    """Adapt the retained accelerated CP signature to the shared dense solver."""
    return solve_dense(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        accelerated=True,
        safety=safety,
        norm_iterations=knorm_iters,
        norm_seed=knorm_seed,
        max_iterations=max_iter,
        tolerance=tol,
        y0=y0,
        lambda0=lam0,
        mu0=mu0,
        equilibrate=partial(ruiz_equilibrate, iterations=ruiz_iters)
        if use_ruiz
        else None,
    )


_static = (
    "safety",
    "knorm_iters",
    "knorm_seed",
    "max_iter",
    "tol",
    "use_ruiz",
    "ruiz_iters",
)
CP_fixed_jit = jax.jit(CP_fixed, static_argnames=_static)
CP_accelerated_jit = jax.jit(CP_accelerated, static_argnames=_static)
