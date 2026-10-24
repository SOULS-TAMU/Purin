"""Specialized batched linear solvers for reduced Fischer--Burmeister systems.

The forward pivot-reduced Newton matrix is symmetric.  This module provides a
true batched MINRES implementation for that indefinite system, a batched PCG
implementation for positive-definite primal Schur complements, and a guarded
GMRES fallback for numerical breakdowns.
"""

from __future__ import annotations
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from purin.solvers.primal_dual_interior_point.operators import _batched_restarted_gmres


class BatchedLinearSolveResult(NamedTuple):
    """Batched linear solutions together with convergence and residual diagnostics."""

    solution: jax.Array
    status: jax.Array
    residual_inf_norm: jax.Array
    iterations: jax.Array


def _row_inner(left, right):
    return jnp.sum(left * right, axis=1)


def _row_inf_norm(values):
    return jnp.max(jnp.abs(values), axis=1, initial=0.0)


def _identity(value):
    return value


def batched_minres(
    operator: Callable,
    rhs,
    *,
    initial_solution=None,
    preconditioner: Callable | None = None,
    tolerance=1e-9,
    absolute_tolerance: float = 0.0,
    max_iterations: int = 100,
) -> BatchedLinearSolveResult:
    """Solve independent symmetric systems with one batched MINRES loop.

    ``operator`` and ``preconditioner`` accept and return arrays of shape
    ``(batch, width)``.  The preconditioner must be SPD.  MINRES uses its short
    recurrence and performs one final true-residual check; rows that fail that
    check can be passed to :func:`solve_symmetric_system` for GMRES fallback.
    """
    rhs = jnp.asarray(rhs)
    dtype = rhs.dtype
    batch_size = rhs.shape[0]
    x0 = (
        jnp.zeros_like(rhs)
        if initial_solution is None
        else jnp.asarray(initial_solution, dtype=dtype)
    )
    apply_preconditioner = _identity if preconditioner is None else preconditioner
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype)

    rhs_inf = _row_inf_norm(rhs)
    tol = jnp.asarray(tolerance, dtype)
    if tol.ndim == 0:
        tol = jnp.full((batch_size,), tol, dtype=dtype)
    threshold_inf = jnp.maximum(jnp.asarray(absolute_tolerance, dtype), tol * rhs_inf)

    residual0 = rhs - operator(x0)
    residual0_inf = _row_inf_norm(residual0)
    y0 = apply_preconditioner(residual0)
    beta_sq0 = _row_inner(residual0, y0)
    beta0 = jnp.sqrt(jnp.maximum(beta_sq0, 0.0))
    finite0 = (
        jnp.all(jnp.isfinite(residual0), axis=1)
        & jnp.all(jnp.isfinite(y0), axis=1)
        & jnp.isfinite(beta0)
    )
    active0 = finite0 & (residual0_inf > threshold_inf)

    state = (
        jnp.asarray(0, jnp.int32),
        x0,
        residual0,  # r1
        residual0,  # r2
        y0,
        jnp.zeros((batch_size,), dtype=dtype),  # old beta
        beta0,
        jnp.zeros((batch_size,), dtype=dtype),  # dbar
        jnp.zeros((batch_size,), dtype=dtype),  # epsln
        beta0,  # phibar
        -jnp.ones((batch_size,), dtype=dtype),  # cs
        jnp.zeros((batch_size,), dtype=dtype),  # sn
        jnp.zeros_like(rhs),  # w
        jnp.zeros_like(rhs),  # w2
        active0,
        ~finite0,
        jnp.zeros((batch_size,), dtype=jnp.int32),
    )

    def condition(current):
        iteration = current[0]
        active = current[-3]
        return (iteration < int(max_iterations)) & jnp.any(active)

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
            active,
            failure,
            row_iterations,
        ) = current

        safe_beta = jnp.where(jnp.abs(beta) > eps, beta, 1.0)
        v = y / safe_beta[:, None]
        lanczos = operator(v)
        safe_old_beta = jnp.where(jnp.abs(old_beta) > eps, old_beta, 1.0)
        lanczos = jnp.where(
            (iteration > 0), lanczos - (beta / safe_old_beta)[:, None] * r1, lanczos
        )
        alpha = _row_inner(v, lanczos)
        lanczos = lanczos - (alpha / safe_beta)[:, None] * r2

        next_r1 = r2
        next_r2 = lanczos
        next_y = apply_preconditioner(next_r2)
        next_beta_sq = _row_inner(next_r2, next_y)
        next_beta = jnp.sqrt(jnp.maximum(next_beta_sq, 0.0))

        old_eps = epsln
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        next_epsln = sn * next_beta
        next_dbar = -cs * next_beta
        gamma = jnp.sqrt(gbar * gbar + next_beta * next_beta)
        safe_gamma = jnp.maximum(gamma, eps)
        next_cs = gbar / safe_gamma
        next_sn = next_beta / safe_gamma
        phi = next_cs * phibar
        next_phibar = next_sn * phibar
        next_w = (v - old_eps[:, None] * w2 - delta[:, None] * w) / safe_gamma[:, None]
        candidate_x = x + phi[:, None] * next_w

        finite = (
            jnp.all(jnp.isfinite(candidate_x), axis=1)
            & jnp.all(jnp.isfinite(next_y), axis=1)
            & jnp.isfinite(next_beta)
            & jnp.isfinite(safe_gamma)
            & (next_beta_sq >= -jnp.sqrt(eps))
        )
        estimated_residual = jnp.abs(next_phibar)
        next_active = active & finite & (estimated_residual > threshold_inf)
        next_failure = failure | (active & ~finite)

        def choose(new, old):
            mask = active
            while mask.ndim < new.ndim:
                mask = mask[..., None]
            return jnp.where(mask, new, old)

        return (
            iteration + 1,
            choose(candidate_x, x),
            choose(next_r1, r1),
            choose(next_r2, r2),
            choose(next_y, y),
            jnp.where(active, beta, old_beta),
            jnp.where(active, next_beta, beta),
            jnp.where(active, next_dbar, dbar),
            jnp.where(active, next_epsln, epsln),
            jnp.where(active, next_phibar, phibar),
            jnp.where(active, next_cs, cs),
            jnp.where(active, next_sn, sn),
            choose(next_w, w),
            choose(w, w2),
            next_active,
            next_failure,
            row_iterations + active.astype(jnp.int32),
        )

    final = jax.lax.while_loop(condition, body, state)
    solution = final[1]
    failure = final[-2]
    iterations = final[-1]
    true_residual = rhs - operator(solution)
    residual_inf = _row_inf_norm(true_residual)
    finite = (
        jnp.all(jnp.isfinite(solution), axis=1)
        & jnp.all(jnp.isfinite(true_residual), axis=1)
        & jnp.isfinite(residual_inf)
    )
    status = jnp.where(
        ~(finite & ~failure), 2, jnp.where(residual_inf <= threshold_inf, 0, 1)
    ).astype(jnp.int32)
    return BatchedLinearSolveResult(solution, status, residual_inf, iterations)


def batched_pcg(
    operator: Callable,
    rhs,
    *,
    initial_solution=None,
    preconditioner: Callable | None = None,
    tolerance=1e-9,
    absolute_tolerance: float = 0.0,
    max_iterations: int = 100,
) -> BatchedLinearSolveResult:
    """Solve independent SPD systems with a shared batched PCG loop.

    Nonpositive curvature is reported as status ``2`` so callers can fall back
    to the symmetric-indefinite reduced KKT solve.
    """
    rhs = jnp.asarray(rhs)
    dtype = rhs.dtype
    batch_size = rhs.shape[0]
    x0 = (
        jnp.zeros_like(rhs)
        if initial_solution is None
        else jnp.asarray(initial_solution, dtype=dtype)
    )
    apply_preconditioner = _identity if preconditioner is None else preconditioner
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype)
    tol = jnp.asarray(tolerance, dtype)
    if tol.ndim == 0:
        tol = jnp.full((batch_size,), tol, dtype=dtype)
    rhs_inf = _row_inf_norm(rhs)
    threshold = jnp.maximum(jnp.asarray(absolute_tolerance, dtype), tol * rhs_inf)

    residual0 = rhs - operator(x0)
    residual_inf0 = _row_inf_norm(residual0)
    z0 = apply_preconditioner(residual0)
    rz0 = _row_inner(residual0, z0)
    finite0 = (
        jnp.all(jnp.isfinite(residual0), axis=1)
        & jnp.all(jnp.isfinite(z0), axis=1)
        & jnp.isfinite(rz0)
    )
    active0 = finite0 & (residual_inf0 > threshold)

    state = (
        jnp.asarray(0, jnp.int32),
        x0,
        residual0,
        z0,
        z0,
        rz0,
        active0,
        ~finite0,
        jnp.zeros((batch_size,), dtype=jnp.int32),
    )

    def condition(current):
        return (current[0] < int(max_iterations)) & jnp.any(current[-3])

    def body(current):
        iteration, x, residual, z, direction, rz, active, failure, row_iters = current
        image = operator(direction)
        curvature = _row_inner(direction, image)
        positive = curvature > eps
        safe_curvature = jnp.where(positive, curvature, 1.0)
        alpha = rz / safe_curvature
        candidate_x = x + alpha[:, None] * direction
        candidate_residual = residual - alpha[:, None] * image
        residual_inf = _row_inf_norm(candidate_residual)
        candidate_z = apply_preconditioner(candidate_residual)
        next_rz = _row_inner(candidate_residual, candidate_z)
        safe_rz = jnp.where(jnp.abs(rz) > eps, rz, 1.0)
        beta = next_rz / safe_rz
        candidate_direction = candidate_z + beta[:, None] * direction
        finite = (
            jnp.all(jnp.isfinite(candidate_x), axis=1)
            & jnp.all(jnp.isfinite(candidate_residual), axis=1)
            & jnp.all(jnp.isfinite(candidate_z), axis=1)
            & jnp.isfinite(next_rz)
            & positive
        )
        next_active = active & finite & (residual_inf > threshold)
        next_failure = failure | (active & ~finite)

        def choose(new, old):
            mask = active
            while mask.ndim < new.ndim:
                mask = mask[..., None]
            return jnp.where(mask, new, old)

        return (
            iteration + 1,
            choose(candidate_x, x),
            choose(candidate_residual, residual),
            choose(candidate_z, z),
            choose(candidate_direction, direction),
            jnp.where(active, next_rz, rz),
            next_active,
            next_failure,
            row_iters + active.astype(jnp.int32),
        )

    final = jax.lax.while_loop(condition, body, state)
    solution = final[1]
    failure = final[-2]
    iterations = final[-1]
    residual = rhs - operator(solution)
    residual_inf = _row_inf_norm(residual)
    finite = (
        jnp.all(jnp.isfinite(solution), axis=1)
        & jnp.all(jnp.isfinite(residual), axis=1)
        & jnp.isfinite(residual_inf)
    )
    status = jnp.where(
        ~(finite & ~failure), 2, jnp.where(residual_inf <= threshold, 0, 1)
    ).astype(jnp.int32)
    return BatchedLinearSolveResult(solution, status, residual_inf, iterations)


def solve_symmetric_system(
    operator: Callable,
    rhs,
    preconditioner: Callable,
    *,
    mode: str,
    tolerance,
    absolute_tolerance: float,
    restart: int,
    max_iterations: int,
    initial_solution=None,
    gmres_fallback: bool = True,
) -> BatchedLinearSolveResult:
    """Solve a symmetric reduced KKT system using MINRES or GMRES.

    ``mode='auto'`` selects MINRES.  Failed MINRES rows are resumed from the
    MINRES iterate with the existing restarted GMRES implementation.
    """
    selected = str(mode).strip().lower().replace("_", "-")
    if selected in {"schur", "schur-cg", "cg"}:
        raise ValueError("Schur-CG must be invoked through batched_pcg().")
    if selected == "dense":
        # Preserve compatibility with configurations that previously selected
        # a dense mode even though reduced FB always used GMRES internally.
        selected = "gmres"
    if selected not in {"auto", "minres", "gmres"}:
        raise ValueError(
            "Reduced Fischer--Burmeister linear solver mode must be "
            "'auto', 'minres', 'schur-cg', or 'gmres'."
        )

    if selected == "gmres":
        solution, status, residual, iterations = _batched_restarted_gmres(
            operator,
            rhs,
            preconditioner,
            tolerance=tolerance,
            absolute_tolerance=absolute_tolerance,
            restart=min(int(restart), int(max_iterations)),
            max_iterations=int(max_iterations),
            initial_solution=initial_solution,
            return_iterations=True,
        )
        return BatchedLinearSolveResult(solution, status, residual, iterations)

    result = batched_minres(
        operator,
        rhs,
        initial_solution=initial_solution,
        preconditioner=preconditioner,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        max_iterations=max_iterations,
    )
    if not gmres_fallback:
        return result

    use_gmres = result.status != 0

    def run_gmres(_):
        return _batched_restarted_gmres(
            operator,
            rhs,
            preconditioner,
            tolerance=tolerance,
            absolute_tolerance=absolute_tolerance,
            restart=min(int(restart), int(max_iterations)),
            max_iterations=int(max_iterations),
            initial_solution=result.solution,
            return_iterations=True,
        )

    def skip_gmres(_):
        return (
            result.solution,
            result.status,
            result.residual_inf_norm,
            jnp.zeros_like(result.iterations),
        )

    solution, gmres_status, gmres_residual, gmres_iterations = jax.lax.cond(
        jnp.any(use_gmres), run_gmres, skip_gmres, operand=None
    )
    merged_solution = jnp.where(use_gmres[:, None], solution, result.solution)
    merged_status = jnp.where(use_gmres, gmres_status, result.status)
    merged_residual = jnp.where(use_gmres, gmres_residual, result.residual_inf_norm)
    merged_iterations = result.iterations + jnp.where(use_gmres, gmres_iterations, 0)
    return BatchedLinearSolveResult(
        merged_solution, merged_status, merged_residual, merged_iterations
    )


__all__ = [
    "BatchedLinearSolveResult",
    "batched_minres",
    "batched_pcg",
    "solve_symmetric_system",
]
