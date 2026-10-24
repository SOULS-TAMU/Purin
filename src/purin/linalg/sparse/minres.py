"""Pure-JAX MINRES for symmetric, possibly indefinite linear operators.

The recurrence follows the Paige--Saunders MINRES iteration.  It accepts a
matrix-free operator and an optional SPD inverse-preconditioner callable.  The
implementation uses fixed-shape arrays and ``jax.lax.while_loop``.
"""

from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
from purin.linalg.sparse.types import LinearSolveResult


def _identity(value):
    return value


def minres(
    operator: Callable,
    rhs,
    *,
    initial=None,
    preconditioner: Callable | None = None,
    tolerance: float = 1e-9,
    absolute_tolerance: float = 0.0,
    max_iterations: int = 200,
) -> LinearSolveResult:
    """Solve ``A x = rhs`` for a symmetric operator using MINRES.

    ``preconditioner`` must apply an SPD approximation of ``A^{-1}``.
    Convergence is checked with the true residual norm.
    """
    rhs = jnp.asarray(rhs)
    x0 = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial, rhs.dtype)
    apply_preconditioner = _identity if preconditioner is None else preconditioner
    norm_rhs = jnp.linalg.norm(rhs)
    threshold = (
        jnp.asarray(absolute_tolerance, rhs.dtype)
        + jnp.asarray(tolerance, rhs.dtype) * norm_rhs
    )
    threshold = jnp.maximum(threshold, jnp.asarray(absolute_tolerance, rhs.dtype))

    initial_residual = rhs - operator(x0)
    initial_y = apply_preconditioner(initial_residual)
    beta_sq = jnp.vdot(initial_residual, initial_y).real
    eps = jnp.finfo(rhs.dtype).eps
    beta1 = jnp.sqrt(jnp.maximum(beta_sq, 0.0))
    initial_norm = jnp.linalg.norm(initial_residual)
    initially_converged = initial_norm <= threshold
    initial_failure = (
        (beta_sq < -jnp.sqrt(eps))
        | (~jnp.all(jnp.isfinite(initial_residual)))
        | (~jnp.all(jnp.isfinite(initial_y)))
    )

    # State mirrors the stable MINRES short recurrence.
    state = (
        jnp.asarray(0, jnp.int32),  # iteration
        x0,
        initial_residual,  # r1
        initial_residual,  # r2
        initial_y,
        jnp.asarray(0.0, rhs.dtype),  # old beta
        beta1,
        jnp.asarray(0.0, rhs.dtype),  # dbar
        jnp.asarray(0.0, rhs.dtype),  # epsln
        beta1,  # phibar
        jnp.asarray(-1.0, rhs.dtype),  # cs
        jnp.asarray(0.0, rhs.dtype),  # sn
        jnp.zeros_like(rhs),  # w
        jnp.zeros_like(rhs),  # w2
        initial_norm,
        initially_converged,
        initial_failure,
    )

    def condition(current):
        iteration, *_middle, converged, failure = current
        return (iteration < int(max_iterations)) & (~converged) & (~failure)

    def body(current):
        (
            iteration,
            x,
            r1,
            r2,
            y,
            old_beta,
            beta,
            dbar,
            epsln,
            phibar,
            cs,
            sn,
            w,
            w2,
            _residual_norm,
            _converged,
            failure,
        ) = current
        safe_beta = jnp.where(jnp.abs(beta) > eps, beta, jnp.asarray(1.0, rhs.dtype))
        v = y / safe_beta
        lanczos = operator(v)
        lanczos = jax.lax.cond(
            iteration > 0,
            lambda value: (
                value - (beta / jnp.where(jnp.abs(old_beta) > eps, old_beta, 1.0)) * r1
            ),
            lambda value: value,
            lanczos,
        )
        alpha = jnp.vdot(v, lanczos).real
        lanczos = lanczos - (alpha / safe_beta) * r2
        next_r1 = r2
        next_r2 = lanczos
        next_y = apply_preconditioner(next_r2)
        next_beta_sq = jnp.vdot(next_r2, next_y).real
        next_beta = jnp.sqrt(jnp.maximum(next_beta_sq, 0.0))

        old_eps = epsln
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        next_epsln = sn * next_beta
        next_dbar = -cs * next_beta
        gamma = jnp.sqrt(gbar * gbar + next_beta * next_beta)
        gamma = jnp.maximum(gamma, eps)
        next_cs = gbar / gamma
        next_sn = next_beta / gamma
        phi = next_cs * phibar
        next_phibar = next_sn * phibar
        next_w = (v - old_eps * w2 - delta * w) / gamma
        next_x = x + phi * next_w

        true_residual = rhs - operator(next_x)
        residual_norm = jnp.linalg.norm(true_residual)
        finite = (
            jnp.all(jnp.isfinite(next_x))
            & jnp.all(jnp.isfinite(true_residual))
            & jnp.isfinite(next_beta)
            & jnp.isfinite(gamma)
        )
        breakdown = next_beta_sq < -jnp.sqrt(eps)
        next_failure = failure | (~finite) | breakdown
        converged = residual_norm <= threshold
        return (
            iteration + 1,
            next_x,
            next_r1,
            next_r2,
            next_y,
            beta,
            next_beta,
            next_dbar,
            next_epsln,
            next_phibar,
            next_cs,
            next_sn,
            next_w,
            w,
            residual_norm,
            converged,
            next_failure,
        )

    final = jax.lax.while_loop(condition, body, state)
    iteration, solution = final[0], final[1]
    residual_norm, converged, failure = final[-3], final[-2], final[-1]
    return LinearSolveResult(
        solution=solution,
        residual_norm=residual_norm,
        iterations=iteration,
        converged=converged & (~failure),
        numerical_failure=failure,
    )
