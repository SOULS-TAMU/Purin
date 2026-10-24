"""Reusable batched dense-QP boundary, scaling, and implicit CP policy."""

import jax.numpy as jnp
from purin.linalg.equilibration import ruiz_equilibrate
from .dense_implicit import CP_fixed_implicit_jit, CP_accelerated_implicit_jit


def solve_dense_implicit(
    q,
    c,
    a,
    b,
    matrix,
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
    adjoint_iterations=50,
    equilibrate=False,
    equilibration_iterations=10,
    y0=None,
    lambda0=None,
    mu0=None,
):
    """Solve batched diagonal QPs with PURIN's implicit-differentiable CP layer.

    Minimize ``0.5 * sum(q * y**2) + sum(c * y)`` subject to
    ``a @ y = b``, ``matrix @ y <= d`` and ``lower <= y <= upper``.

    Args:
        q: Objective diagonal of shape ``(batch, n)``.
        c: Linear objective coefficients of shape ``(batch, n)``.
        a: Equality matrices of shape ``(batch, me, n)`` or ``(me, n)``.
        b: Equality right-hand sides broadcastable to ``(batch, me)``.
        matrix: Inequality matrices of shape ``(batch, mi, n)`` or ``(mi, n)``.
        d: Inequality right-hand sides broadcastable to ``(batch, mi)``.
        lower: Lower bounds broadcastable to ``(batch, n)``.
        upper: Upper bounds broadcastable to ``(batch, n)``.
        accelerated: Select accelerated instead of fixed-step forward CP.
        safety: Step-size safety factor.
        norm_iterations: Power-iteration count for operator norm estimation.
        norm_seed: Deterministic norm-estimation seed.
        max_iterations: Forward iteration limit.
        tolerance: Forward stopping tolerance.
        adjoint_iterations: Iteration limit for the implicit adjoint solve.
        equilibrate: Apply Ruiz scaling before solving when True.
        equilibration_iterations: Number of Ruiz equilibration passes.
        y0: Optional primal warm start of shape ``(batch, n)``.
        lambda0: Optional equality-multiplier warm start of shape ``(batch, me)``.
        mu0: Optional inequality-multiplier warm start of shape ``(batch, mi)``.

    Returns:
        Tuple ``(y, equality_multipliers, inequality_multipliers)`` in the
        original coordinates. Their shapes are ``(batch, n)``, ``(batch, me)``
        and ``(batch, mi)``. Zero-width constraint blocks are supported.

    Notes:
        Both forward modes use the shared CP iteration. Reverse mode uses
        implicit fixed-point differentiation rather than an unrolled iteration
        trace. Warm starts and multipliers are rescaled consistently with Ruiz.
    """
    arrays = tuple(map(jnp.asarray, (q, c, a, b, matrix, d, lower, upper)))
    dtype = jnp.result_type(*(value.dtype for value in arrays))
    q, c, a, b, matrix, d, lower, upper = (value.astype(dtype) for value in arrays)
    batch, n = q.shape
    if a.ndim == 2:
        a = jnp.broadcast_to(a[None], (batch, *a.shape))
    if matrix.ndim == 2:
        matrix = jnp.broadcast_to(matrix[None], (batch, *matrix.shape))
    b, d, lower, upper = (
        jnp.broadcast_to(value, (batch, value.shape[-1]))
        for value in (b, d, lower, upper)
    )
    y0 = jnp.zeros((batch, n), dtype) if y0 is None else jnp.asarray(y0, dtype)
    lambda0 = (
        jnp.zeros((batch, a.shape[1]), dtype)
        if lambda0 is None
        else jnp.asarray(lambda0, dtype)
    )
    mu0 = (
        jnp.zeros((batch, matrix.shape[1]), dtype)
        if mu0 is None
        else jnp.asarray(mu0, dtype)
    )
    if equilibrate:
        q, c, a, b, matrix, d, lower, upper, columns, eq_scale, iq_scale = (
            ruiz_equilibrate(
                q, c, a, b, matrix, d, lower, upper, iterations=equilibration_iterations
            )
        )
        y0 = y0 / columns
        lambda0 = lambda0 / jnp.where(eq_scale == 0.0, 1.0, eq_scale)
        mu0 = mu0 / jnp.where(iq_scale == 0.0, 1.0, iq_scale)
    solve = CP_accelerated_implicit_jit if accelerated else CP_fixed_implicit_jit
    y, equality, inequality = solve(
        q,
        c,
        a,
        b,
        matrix,
        d,
        lower,
        upper,
        safety=safety,
        knorm_iters=norm_iterations,
        knorm_seed=norm_seed,
        max_iter=max_iterations,
        tol=tolerance,
        adjoint_iters=adjoint_iterations,
        y0=y0,
        lam0=lambda0,
        mu0=mu0,
    )
    if equilibrate:
        return columns * y, eq_scale * equality, iq_scale * inequality
    return y, equality, inequality
