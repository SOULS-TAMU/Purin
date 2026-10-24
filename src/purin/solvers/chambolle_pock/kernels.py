"""Sparse Chambolle--Pock iteration kernels using KKTHN matrix kernels.

The quadratic objective is 0.5*y.T*diag(q)*y + c.T*y. Constraints are
K[:me]@y=rhs[:me], K[me:]@y<=rhs[me:], and lower<=y<=upper.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from purin.linalg.sparse.operators import _batched_local_jacobian_matvec
from purin.linalg.sparse.operators import _batched_local_jacobian_transpose_matvec


class CPOutput(NamedTuple):
    """Batched primal/dual iterates, iteration counts, feasibility, gap and convergence."""

    y: object
    dual: object
    iterations: object
    residual: object
    gap: object
    converged: object


def layout(rows, n):
    """Pad sparse row supports and return support indices, masks and value positions."""
    width = max(1, max(map(len, rows), default=0))
    support = np.zeros((len(rows), width), dtype=np.int32)
    mask = np.zeros_like(support, dtype=bool)
    positions = []
    for i, row in enumerate(rows):
        support[i, : len(row)] = row
        mask[i, : len(row)] = True
        positions.extend(i * width + j for j in range(len(row)))
    return (
        jnp.asarray(support),
        jnp.asarray(mask),
        jnp.asarray(positions, dtype=jnp.int32),
    )


def operators(values, support, mask, n):
    """Build batched forward and transpose products from fixed sparse row supports."""
    mv = lambda x: _batched_local_jacobian_matvec(values, support, mask, x)
    tmv = lambda x: _batched_local_jacobian_transpose_matvec(
        values, support, mask, x, n
    )
    return mv, tmv


def norm_estimate(mv, tmv, batch, n, dtype, iterations=20, seed=42, key=None):
    """Estimate each instance's operator norm by power iteration on K.T @ K."""
    keys = jax.random.split(jax.random.PRNGKey(seed) if key is None else key, batch)
    x = jax.vmap(lambda key: jax.random.normal(key, (n,), dtype=dtype))(keys)
    x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def step(_, carry):
        x, eigenvalue = carry
        z = tmv(mv(x))
        eigenvalue = jnp.sum(x * z, axis=1)
        return z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-12), eigenvalue

    _, eigenvalue = jax.lax.fori_loop(0, iterations, step, (x, jnp.zeros(batch, dtype)))
    return jnp.sqrt(jnp.maximum(eigenvalue, 0)) + 1e-12


def proximal_coefficients(q, c, tau):
    """Fixed diagonal proximal map: a*y - d*(K.T@dual) + offset."""
    step = tau[:, None] if tau.ndim == 1 else tau
    inverse = 1 / (1 + step * q)
    dual_weight = step * inverse
    return inverse, dual_weight, -dual_weight * c


def diagonal_step_sizes(q, values, support, mask, safety):
    """Positive metrics with ||sqrt(S) K sqrt(T)|| <= safety < 1.

    Weighted Cauchy--Schwarz proves the bound for T_j=safety/colsum_j
    and S_i=safety/rowsum_i. Increasing zero/tiny denominators preserves it.
    This requires sparse absolute row/column sums, not a spectral estimate.
    """
    mv, tmv = operators(jnp.abs(values), support, mask, q.shape[1])
    row = mv(jnp.ones_like(q))
    column = tmv(jnp.ones_like(row))
    row = jnp.where(row > 0, jnp.maximum(row, 1e-12), 1.0)
    column = jnp.where(column > 0, jnp.maximum(column, 1e-12), 1.0)
    return jax.lax.stop_gradient(safety / column), jax.lax.stop_gradient(safety / row)


def cp_step(
    state,
    q,
    c,
    rhs,
    lower,
    upper,
    mv,
    tmv,
    me,
    tau,
    sigma,
    theta,
    proximal=None,
    factored_proximal=False,
):
    """Apply one CP primal/dual update and extrapolate the primal iterate.

    Args:
        state: Tuple ``(y, dual, ybar)`` of current batched iterates.
        q: Diagonal quadratic coefficients of shape ``(batch, n)``.
        c: Linear objective coefficients of shape ``(batch, n)``.
        rhs: Equality then inequality right-hand sides, shape ``(batch, m)``.
        lower: Broadcastable lower primal bounds.
        upper: Broadcastable upper primal bounds.
        mv: Callable evaluating the batched constraint product ``K @ y``.
        tmv: Callable evaluating the batched transpose product ``K.T @ dual``.
        me: Number of leading equality rows; remaining dual entries are
            projected onto the nonnegative orthant.
        tau: Per-instance or diagonal primal step sizes.
        sigma: Per-instance or diagonal dual step sizes.
        theta: Per-instance extrapolation factors of shape ``(batch,)``.
        proximal: Optional precomputed diagonal proximal coefficients.
        factored_proximal: Preserve the factored proximal arithmetic used by
            the dense adapter when True.

    Returns:
        Tuple ``(next_y, next_dual, next_ybar)`` with the same block shapes
        as the incoming state.
    """
    y, dual, ybar = state
    dual_step = sigma[:, None] if sigma.ndim == 1 else sigma
    next_dual = dual + dual_step * (mv(ybar) - rhs)
    next_dual = jnp.concatenate(
        (next_dual[:, :me], jnp.maximum(next_dual[:, me:], 0)), axis=1
    )
    inverse, dual_weight, offset = (
        proximal_coefficients(q, c, tau) if proximal is None else proximal
    )
    if factored_proximal:
        step = tau[:, None] if tau.ndim == 1 else tau
        next_y = inverse * ((y - step * tmv(next_dual)) - step * c)
    else:
        next_y = inverse * y - dual_weight * tmv(next_dual) + offset
    next_y = (
        jnp.minimum(jnp.maximum(next_y, lower), upper)
        if factored_proximal
        else jnp.clip(next_y, lower, upper)
    )
    return next_y, next_dual, next_y + theta[:, None] * (next_y - y)


def diagnostics(y, dual, q, c, rhs, lower, upper, mv, tmv, me):
    """Return per-instance maximum feasibility violation and normalized primal-dual gap."""
    residuals = mv(y) - rhs
    eq = jnp.max(jnp.abs(residuals[:, :me]), axis=1, initial=0.0)
    ineq = jnp.max(jnp.maximum(residuals[:, me:], 0), axis=1, initial=0.0)
    bound = jnp.maximum(
        jnp.max(jnp.maximum(lower - y, 0), axis=1, initial=0.0),
        jnp.max(jnp.maximum(y - upper, 0), axis=1, initial=0.0),
    )
    primal = jnp.sum(0.5 * q * y * y + c * y, axis=1)
    t = c + tmv(dual)
    positive = q > 1e-12
    free = -t / jnp.where(positive, q, 1.0)
    minimizer = jnp.where(
        positive, jnp.clip(free, lower, upper), jnp.where(t >= 0.0, lower, upper)
    )
    dual_objective = jnp.sum(0.5 * q * minimizer**2 + t * minimizer, axis=1) - jnp.sum(
        rhs * dual, axis=1
    )
    gap = jnp.maximum(primal - dual_objective, 0) / jnp.maximum(
        1.0, jnp.maximum(abs(primal), abs(dual_objective))
    )
    return jnp.maximum(jnp.maximum(eq, ineq), bound), gap


def sparse_cp(
    q,
    c,
    values,
    rhs,
    support,
    mask,
    me,
    lower,
    upper,
    *,
    accelerated=False,
    safety=0.95,
    max_iterations=5000,
    tolerance=1e-9,
    norm_iterations=20,
    norm_seed=42,
    y0=None,
    dual0=None,
    freeze_converged=False,
    check_stationarity=False,
    norm_bound=False,
    diagonal_preconditioning=False,
    operator_pair=None,
    factored_proximal=False,
):
    """Solve batched diagonal QPs with the shared fixed or accelerated CP iteration."""
    batch, n = q.shape
    mv, tmv = (
        operators(values, support, mask, n) if operator_pair is None else operator_pair
    )
    if diagonal_preconditioning and not accelerated:
        tau, sigma = diagonal_step_sizes(q, values, support, mask, safety)
    else:
        norm = norm_estimate(mv, tmv, batch, n, q.dtype, norm_iterations, norm_seed)
        if norm_bound:
            amv, atmv = operators(jnp.abs(values), support, mask, n)
            row_sum = amv(jnp.ones_like(q))
            column_sum = atmv(jnp.ones_like(rhs))
            norm = jnp.sqrt(
                jnp.max(row_sum, axis=1, initial=0.0)
                * jnp.max(column_sum, axis=1, initial=0.0)
            )
        L = jnp.maximum(norm, 1e-12)
        sigma = safety / L
        tau = jnp.where(
            accelerated & (jnp.min(q, axis=1) > 0),
            safety / L,
            safety / (jnp.max(q, axis=1) + L),
        )
    gamma = jnp.min(q, axis=1)
    y = jnp.zeros_like(q) if y0 is None else y0
    dual = jnp.zeros_like(rhs) if dual0 is None else dual0
    theta = jnp.ones(batch, q.dtype)
    done = jnp.zeros(batch, bool)
    counts = jnp.zeros(batch, jnp.int32)
    # Q, c and tau are invariant across fixed-step iterations. For accelerated
    # CP tau changes, so its coefficients are recomputed for each new step.
    proximal = None if accelerated else proximal_coefficients(q, c, tau)

    def condition(state):
        iteration, _, _, _, _, _, _, done, _ = state
        return (iteration < max_iterations) & ~jnp.all(done)

    def body(state):
        iteration, y, dual, ybar, tau, sigma, theta, done, counts = state
        ny, nd, nb = cp_step(
            (y, dual, ybar),
            q,
            c,
            rhs,
            lower,
            upper,
            mv,
            tmv,
            me,
            tau,
            sigma,
            theta,
            proximal=proximal,
            factored_proximal=factored_proximal,
        )
        residual, gap = diagnostics(ny, nd, q, c, rhs, lower, upper, mv, tmv, me)
        finite = (
            jnp.all(jnp.isfinite(ny), axis=1)
            & jnp.all(jnp.isfinite(nd), axis=1)
            & jnp.isfinite(gap)
        )
        # Freeze accepted samples; convergence is assessed again on returned values.
        active = ~done if freeze_converged else jnp.ones_like(done)
        y = jnp.where(active[:, None], ny, y)
        dual = jnp.where(active[:, None], nd, dual)
        ybar = jnp.where(active[:, None], nb, ybar)
        counts = counts + active.astype(jnp.int32)
        stationarity = jnp.max(
            jnp.abs(ny - jnp.clip(ny - (q * ny + c + tmv(nd)), lower, upper)), axis=1
        )
        accepted = finite & (residual <= tolerance) & (gap <= tolerance)
        if check_stationarity:
            accepted = accepted & (stationarity <= tolerance)
        done = done | accepted
        if accelerated:
            nt = 1 / jnp.sqrt(1 + gamma * tau)
            tau = nt * tau
            sigma = sigma / nt
            theta = nt
        return iteration + 1, y, dual, ybar, tau, sigma, theta, done, counts

    _, y, dual, _, _, _, _, _, counts = jax.lax.while_loop(
        condition,
        body,
        (jnp.array(0, jnp.int32), y, dual, y, tau, sigma, theta, done, counts),
    )
    residual, gap = diagnostics(y, dual, q, c, rhs, lower, upper, mv, tmv, me)
    finite = (
        jnp.all(jnp.isfinite(y), axis=1)
        & jnp.all(jnp.isfinite(dual), axis=1)
        & jnp.isfinite(gap)
        & jnp.isfinite(residual)
    )
    converged = finite & (residual <= tolerance) & (gap <= tolerance)
    if check_stationarity:
        stationarity = jnp.max(
            jnp.abs(y - jnp.clip(y - (q * y + c + tmv(dual)), lower, upper)), axis=1
        )
        converged = converged & (stationarity <= tolerance)
        residual = jnp.maximum(residual, stationarity)
    return CPOutput(y, dual, counts, residual, gap, converged)
