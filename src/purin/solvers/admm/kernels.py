"""Sparse JAX OSQP-style indirect ADMM for positive diagonal Q.
Narrow systems reuse a sparse band factor; wider systems use indirect PCG.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from purin.solvers.fischer_burmeister.solver import batched_pcg
from purin.linalg.sparse.operators import _batched_local_jacobian_matvec
from purin.linalg.sparse.operators import _batched_local_jacobian_transpose_matvec


class ADMMOutput(NamedTuple):
    """Batched ADMM primal/dual iterates and convergence diagnostics."""

    x: object
    z: object
    dual: object
    rho: object
    iterations: object
    inner_iterations: object
    primal_residual: object
    dual_residual: object
    linear_residual: object
    converged: object
    finite: object


def operators(values, support, mask, n):
    """Construct forward and transpose products for the ADMM constraint operator."""
    apply = lambda x: _batched_local_jacobian_matvec(values, support, mask, x)
    transpose = lambda y: _batched_local_jacobian_transpose_matvec(
        values, support, mask, y, n
    )
    return apply, transpose


def inf_norm(values):
    """Compute the maximum absolute entry along each instance's final axis."""
    return jnp.max(jnp.abs(values), axis=1, initial=0.0)


def sparse_admm(
    diagonal,
    c,
    values,
    lower,
    upper,
    support,
    mask,
    *,
    rho=1.0,
    sigma=1e-6,
    alpha=1.6,
    tolerance=1e-9,
    max_iterations=10000,
    cg_iterations=32,
    cg_tolerance=1e-10,
    refinement_steps=1,
    adaptive_rho=False,
    rho_update_interval=50,
    rho_update_threshold=5.0,
    x0=None,
    dual0=None,
    factor_plan=None,
    check_interval=1,
):
    """Solve batched diagonal QPs with sparse ADMM updates."""
    batch, n = diagonal.shape
    mv, tmv = operators(values, support, mask, n)
    _, squared_transpose = operators(values**2, support, mask, n)
    penalty = jnp.broadcast_to(jnp.asarray(rho, dtype=diagonal.dtype), lower.shape)
    x = jnp.zeros_like(diagonal) if x0 is None else x0
    dual = jnp.zeros_like(lower) if dual0 is None else dual0
    z = jnp.zeros_like(lower) if x0 is None else jnp.clip(mv(x), lower, upper)

    def preconditioner(p):
        return diagonal + sigma + squared_transpose(p)

    diag = preconditioner(penalty)
    factor = (
        factor_plan.factor(diagonal, values, penalty, sigma)
        if factor_plan is not None
        else jnp.zeros((batch, 0), diagonal.dtype)
    )
    done = jnp.zeros(batch, bool)
    failed = jnp.zeros(batch, bool)
    counts = jnp.zeros(batch, jnp.int32)
    inner_counts = jnp.zeros_like(counts)

    def condition(state):
        iteration = state[0]
        done, failed = state[-2:]
        return (iteration < max_iterations) & jnp.any(~(done | failed))

    def step(state):
        (
            iteration,
            x,
            z,
            dual,
            penalty,
            diag,
            factor,
            counts,
            inner_counts,
            last_linear,
            done,
            failed,
        ) = state
        active = ~(done | failed)
        op = lambda v: (diagonal + sigma) * v + tmv(penalty * mv(v))
        rhs = sigma * x - c + tmv(penalty * z - dual)
        lin_count = jnp.zeros_like(counts)
        if factor_plan is not None:
            xt = factor_plan.solve(factor, rhs)
        else:
            xt = x
            lin_count = jnp.zeros_like(counts)
            # Normalize warm-start corrections to avoid an absolute curvature floor
            # in the shared PCG kernel from limiting small final corrections.
            for _ in range(refinement_steps):
                correction = rhs - op(xt)
                magnitude = jnp.maximum(inf_norm(correction), jnp.finfo(x.dtype).tiny)
                normalized = jnp.where(
                    active[:, None], correction / magnitude[:, None], 0.0
                )
                solved = batched_pcg(
                    op,
                    normalized,
                    preconditioner=lambda v: v / diag,
                    tolerance=cg_tolerance,
                    absolute_tolerance=cg_tolerance,
                    max_iterations=cg_iterations,
                )
                xt = xt + magnitude[:, None] * solved.solution
                lin_count = lin_count + solved.iterations
        ax = mv(xt)
        nx = alpha * xt + (1 - alpha) * x
        relaxed = alpha * ax + (1 - alpha) * z
        nz = jnp.clip(relaxed + dual / penalty, lower, upper)
        nd = dual + penalty * (relaxed - nz)
        checking = ((iteration + 1) % check_interval == 0) | (
            iteration + 1 == max_iterations
        )
        if adaptive_rho:
            checking = checking | ((iteration + 1) % rho_update_interval == 0)

        def diagnostics(_):
            # Reuse A*x_tilde rather than applying A again in the true inner
            # residual. All acceptance tests use fresh, explicit products.
            linear = inf_norm(rhs - ((diagonal + sigma) * xt + tmv(penalty * ax)))
            primal = inf_norm(mv(nx) - nz)
            dual_res = inf_norm(diagonal * nx + c + tmv(nd))
            finite = (
                jnp.all(jnp.isfinite(nx), axis=1)
                & jnp.all(jnp.isfinite(nd), axis=1)
                & jnp.isfinite(linear)
                & jnp.isfinite(primal)
                & jnp.isfinite(dual_res)
            )
            return linear, primal, dual_res, finite

        if check_interval == 1:
            linear, primal, dual_res, finite = diagnostics(None)
        else:
            linear, primal, dual_res, finite = jax.lax.cond(
                checking,
                diagnostics,
                lambda _: (
                    last_linear,
                    jnp.full_like(last_linear, jnp.inf),
                    jnp.full_like(last_linear, jnp.inf),
                    jnp.ones_like(done),
                ),
                None,
            )
        accepted = (
            checking
            & finite
            & (primal <= tolerance)
            & (dual_res <= tolerance)
            & (linear <= tolerance)
        )
        x = jnp.where(active[:, None], nx, x)
        z = jnp.where(active[:, None], nz, z)
        dual = jnp.where(active[:, None], nd, dual)
        counts = counts + active.astype(jnp.int32)
        inner_counts = inner_counts + jnp.where(active, lin_count, 0)
        last_linear = jnp.where(active & checking, linear, last_linear)
        done = done | (active & accepted)
        failed = failed | (active & checking & ~finite)
        if adaptive_rho:
            # OSQP-style normalized residual balancing, at deterministic intervals.
            pscale = jnp.maximum(jnp.maximum(inf_norm(mv(x)), inf_norm(z)), 1e-12)
            dscale = jnp.maximum(
                jnp.maximum(inf_norm(diagonal * x), inf_norm(tmv(dual))),
                jnp.maximum(inf_norm(c), 1e-12),
            )
            rho_factor = jnp.sqrt(
                (primal / pscale + 1e-16) / (dual_res / dscale + 1e-16)
            )
            change = (
                ((iteration + 1) % rho_update_interval == 0)
                & active
                & ~done
                & finite
                & (
                    (rho_factor >= rho_update_threshold)
                    | (rho_factor <= 1 / rho_update_threshold)
                )
            )
            candidate = jnp.clip(penalty * rho_factor[:, None], 1e-6, 1e6)
            penalty = jnp.where(change[:, None], candidate, penalty)
            diag = jax.lax.cond(
                jnp.any(change), lambda _: preconditioner(penalty), lambda _: diag, None
            )
            if factor_plan is not None:
                factor = jax.lax.cond(
                    jnp.any(change),
                    lambda _: factor_plan.factor(diagonal, values, penalty, sigma),
                    lambda _: factor,
                    None,
                )
        return (
            iteration + 1,
            x,
            z,
            dual,
            penalty,
            diag,
            factor,
            counts,
            inner_counts,
            last_linear,
            done,
            failed,
        )

    state = jax.lax.while_loop(
        condition,
        step,
        (
            jnp.array(0, jnp.int32),
            x,
            z,
            dual,
            penalty,
            diag,
            factor,
            counts,
            inner_counts,
            jnp.full(batch, jnp.inf, diagonal.dtype),
            done,
            failed,
        ),
    )
    _, x, z, dual, penalty, _, _, counts, inner_counts, linear, _, _ = state
    primal = inf_norm(mv(x) - z)
    dual_res = inf_norm(diagonal * x + c + tmv(dual))
    finite = (
        jnp.all(jnp.isfinite(x), axis=1)
        & jnp.all(jnp.isfinite(dual), axis=1)
        & jnp.isfinite(primal)
        & jnp.isfinite(dual_res)
        & jnp.isfinite(linear)
    )
    converged = (
        finite & (primal <= tolerance) & (dual_res <= tolerance) & (linear <= tolerance)
    )
    return ADMMOutput(
        x,
        z,
        dual,
        penalty,
        counts,
        inner_counts,
        primal,
        dual_res,
        linear,
        converged,
        finite,
    )
