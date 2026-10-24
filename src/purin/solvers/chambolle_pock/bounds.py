"""CP-only bound presolve with original-row diagnostics and multiplier recovery."""

import jax
import jax.numpy as jnp
from purin.solvers.primal_dual_interior_point.operators import _batched_restarted_gmres
from purin.solvers.types import ProjectionResult
from purin.solvers.types import SolveStatus
from purin.solvers.chambolle_pock.kernels import cp_step
from purin.solvers.chambolle_pock.kernels import diagonal_step_sizes
from purin.solvers.chambolle_pock.kernels import norm_estimate
from purin.solvers.chambolle_pock.kernels import operators
from purin.solvers.chambolle_pock.kernels import sparse_cp


class BoundProjection:
    """Keep public rows intact; remove singleton inequalities only internally."""

    def __init__(self, owner, bound_rows):
        self.owner = owner
        self.bound_rows = jnp.asarray(bound_rows, dtype=jnp.int32)
        bound_set = set(bound_rows)
        self.rows = jnp.asarray(
            [
                r
                for r in range(owner.equality_size + owner.inequality_size)
                if r not in bound_set
            ],
            dtype=jnp.int32,
        )
        self.variables = owner._support[self.bound_rows, 0]
        self.support = owner._support[self.rows]
        self.mask = owner._mask[self.rows]
        self.solve_jit = jax.jit(self._make_solve())

    def _make_solve(self):
        """Build bound clipping with the configured projection-gradient policy."""
        owner = self.owner
        cfg = owner.config
        me, n = owner.equality_size, owner.output_size
        m = len(self.rows)

        def forward(values, rhs, target, initial, tolerance, lower, upper, feasible):
            out = sparse_cp(
                jnp.ones_like(target),
                -target,
                values,
                rhs,
                self.support,
                self.mask,
                me,
                lower,
                upper,
                accelerated=cfg.accelerated,
                safety=cfg.safety,
                max_iterations=cfg.max_iterations,
                tolerance=tolerance,
                norm_iterations=cfg.norm_iterations,
                norm_seed=cfg.norm_seed,
                y0=target,
                dual0=initial,
                freeze_converged=True,
                check_stationarity=True,
                norm_bound=cfg.norm_bound,
                diagonal_preconditioning=cfg.diagonal_preconditioning,
            )
            return out._replace(converged=out.converged & feasible)

        @jax.custom_vjp
        def solve(values, rhs, target, initial, tolerance, lower, upper, feasible):
            return forward(
                values, rhs, target, initial, tolerance, lower, upper, feasible
            )

        def fwd(values, rhs, target, initial, tolerance, lower, upper, feasible):
            out = forward(
                values, rhs, target, initial, tolerance, lower, upper, feasible
            )
            return out, (out, values, rhs, target, lower, upper)

        def bwd(saved, cot):
            out, values, rhs, target, lower, upper = saved
            mv, tmv = operators(values, self.support, self.mask, n)
            if cfg.diagonal_preconditioning and not cfg.accelerated:
                tau, sigma = diagonal_step_sizes(
                    jnp.ones_like(target), values, self.support, self.mask, cfg.safety
                )
            else:
                L = jnp.maximum(
                    norm_estimate(
                        mv,
                        tmv,
                        owner.batch_size,
                        n,
                        target.dtype,
                        cfg.norm_iterations,
                        cfg.norm_seed,
                    ),
                    1e-12,
                )
                tau = jax.lax.stop_gradient(cfg.safety / (1 + L))
                sigma = jax.lax.stop_gradient(cfg.safety / L)

            def unpack(z):
                return z[:, :n], z[:, n : n + m], z[:, n + m :]

            def pack(z):
                return jnp.concatenate(z, axis=1)

            def step(z, data, b, t, lo, hi):
                apply, transpose = operators(data, self.support, self.mask, n)
                return pack(
                    cp_step(
                        unpack(z),
                        jnp.ones_like(t),
                        -t,
                        b,
                        lo,
                        hi,
                        apply,
                        transpose,
                        me,
                        tau,
                        sigma,
                        jnp.ones(owner.batch_size, target.dtype),
                    )
                )

            state = pack((out.y, out.dual, out.y))
            _, pullback = jax.vjp(step, state, values, rhs, target, lower, upper)
            operator = lambda v: v - pullback(v)[0]
            g = pack(
                (
                    jnp.where(out.converged[:, None], cot.y, 0),
                    jnp.zeros_like(out.dual),
                    jnp.zeros_like(out.y),
                )
            )
            linear = cfg.linear_solver
            adjoint, code, residual = _batched_restarted_gmres(
                operator,
                g,
                lambda v: v,
                tolerance=linear.tolerance,
                absolute_tolerance=linear.atol,
                restart=linear.restart,
                max_iterations=linear.max_iterations,
            )
            valid = (
                out.converged
                & jnp.all(jnp.isfinite(adjoint), axis=1)
                & jnp.isfinite(residual)
                & ((code == 0) | (residual <= cfg.implicit_gradient.adjoint_tolerance))
            )
            _, gv, gb, gt, gl, gu = pullback(jnp.where(valid[:, None], adjoint, 0))
            clean = lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return (
                clean(gv),
                clean(gb),
                clean(gt),
                jnp.zeros_like(out.dual),
                jnp.zeros((owner.batch_size,), target.dtype),
                clean(gl),
                clean(gu),
                None,
            )

        solve.defvjp(fwd, bwd)
        return solve

    def solve(self, target, parameters, multipliers_eq, multipliers_ineq):
        """Clip primal values to bounds and return the prepared projection result."""
        owner = self.owner
        target = jax.device_put(jnp.asarray(target, dtype=owner.dtype), owner._device)
        if target.ndim == 1:
            target = target[None, :]
        if target.shape != (owner.batch_size, owner.output_size):
            raise ValueError(
                f"y_hat must have shape {(owner.batch_size, owner.output_size)}; got {target.shape}"
            )
        if parameters is None and owner.parameter_size:
            raise ValueError("parameters are required for this problem")
        theta = (
            jnp.zeros((owner.batch_size, 0), owner.dtype)
            if parameters is None
            else jnp.asarray(parameters, dtype=owner.dtype)
        )
        if theta.shape != (owner.batch_size, owner.parameter_size):
            raise ValueError("parameters shape does not match the problem")
        if not owner.config.implicit_gradient.differentiate_problem_parameters:
            theta = jax.lax.stop_gradient(theta)
        theta = jax.device_put(theta, owner._device)
        values, rhs = jax.vmap(owner._numeric_one)(theta)
        coefficient = values[:, self.bound_rows, 0]
        threshold = rhs[:, self.bound_rows] / jnp.where(
            coefficient != 0, coefficient, 1.0
        )
        lower = (
            jnp.full_like(target, -jnp.inf)
            .at[:, self.variables]
            .max(jnp.where(coefficient < 0, threshold, -jnp.inf))
        )
        upper = (
            jnp.full_like(target, jnp.inf)
            .at[:, self.variables]
            .min(jnp.where(coefficient > 0, threshold, jnp.inf))
        )
        feasible = jnp.all(lower <= upper, axis=1) & jnp.all(
            (coefficient != 0) | (rhs[:, self.bound_rows] >= 0), axis=1
        )
        kept = values[:, self.rows]
        scale = jnp.ones_like(rhs[:, self.rows])
        if owner.config.row_scaling:
            norms = jnp.sqrt(jnp.sum(kept**2, axis=2))
            scale = jnp.where(norms > 0, 1 / jnp.maximum(norms, 1e-30), 1.0)
        eq = (
            jnp.zeros((owner.batch_size, owner.equality_size), owner.dtype)
            if multipliers_eq is None
            else jnp.asarray(multipliers_eq, dtype=owner.dtype)
        )
        iq = (
            jnp.zeros((owner.batch_size, owner.inequality_size), owner.dtype)
            if multipliers_ineq is None
            else jnp.asarray(multipliers_ineq, dtype=owner.dtype)
        )
        if eq.shape != (owner.batch_size, owner.equality_size) or iq.shape != (
            owner.batch_size,
            owner.inequality_size,
        ):
            raise ValueError("warm-start multiplier shapes do not match constraints")
        initial = jax.lax.stop_gradient(
            jnp.concatenate((eq, iq), axis=1)[:, self.rows] / scale
        )
        tolerance = jax.lax.stop_gradient(
            owner.config.tolerance
            * jnp.minimum(1.0, jnp.min(scale, axis=1, initial=1.0))
        )
        out = self.solve_jit(
            kept * scale[:, :, None],
            rhs[:, self.rows] * scale,
            target,
            initial,
            tolerance,
            lower,
            upper,
            feasible,
        )
        dual = jnp.zeros_like(rhs).at[:, self.rows].set(out.dual * scale)
        mv, tmv = operators(values, owner._support, owner._mask, owner.output_size)
        base_stationarity = out.y - target + tmv(dual)
        # Split a bound's normal among tied original rows. This preserves the
        # original multiplier ordering even for redundant/scaled bounds.
        is_lower = (
            (coefficient < 0)
            & (threshold == lower[:, self.variables])
            & (out.y[:, self.variables] == lower[:, self.variables])
        )
        is_upper = (
            (coefficient > 0)
            & (threshold == upper[:, self.variables])
            & (out.y[:, self.variables] == upper[:, self.variables])
        )
        lower_count = (
            jnp.zeros_like(target)
            .at[:, self.variables]
            .add(is_lower.astype(target.dtype))
        )
        upper_count = (
            jnp.zeros_like(target)
            .at[:, self.variables]
            .add(is_upper.astype(target.dtype))
        )
        normal = base_stationarity[:, self.variables]
        bound_dual = jnp.where(
            is_lower,
            jnp.maximum(normal, 0) / jnp.maximum(lower_count[:, self.variables], 1),
            0,
        )
        bound_dual += jnp.where(
            is_upper,
            jnp.maximum(-normal, 0) / jnp.maximum(upper_count[:, self.variables], 1),
            0,
        )
        bound_dual /= jnp.where(coefficient != 0, abs(coefficient), 1.0)
        dual = dual.at[:, self.bound_rows].set(bound_dual)
        r = mv(out.y) - rhs
        eq_res = jnp.max(abs(r[:, : owner.equality_size]), axis=1, initial=0.0)
        iq_res = jnp.max(
            jnp.maximum(r[:, owner.equality_size :], 0), axis=1, initial=0.0
        )
        stationarity = jnp.max(abs(out.y - target + tmv(dual)), axis=1)
        residual = jnp.maximum(jnp.maximum(eq_res, iq_res), stationarity)
        finite = (
            jnp.all(jnp.isfinite(out.y), axis=1)
            & jnp.all(jnp.isfinite(dual), axis=1)
            & jnp.isfinite(residual)
            & jnp.isfinite(out.gap)
        )
        converged = finite & out.converged & (residual <= owner.config.tolerance)
        status = jnp.where(
            converged,
            int(SolveStatus.CONVERGED),
            jnp.where(
                finite, int(SolveStatus.MAX_ITERATIONS), int(SolveStatus.NONFINITE)
            ),
        ).astype(jnp.int32)
        stop = jax.lax.stop_gradient
        return ProjectionResult(
            out.y,
            stop(dual[:, : owner.equality_size]),
            stop(dual[:, owner.equality_size :]),
            stop(jnp.maximum(-r[:, owner.equality_size :], 0)),
            status,
            out.iterations,
            stop(residual),
            finite,
            converged,
        )


def prepare_bounds(owner):
    """Use verified row sparsity; parameter-dependent signs stay runtime data."""
    rows = owner.canonical.inequality_sparsity
    if rows is None:
        rows = [tuple(range(owner.output_size))] * owner.inequality_size
    selected = [owner.equality_size + i for i, row in enumerate(rows) if len(row) == 1]
    return BoundProjection(owner, selected) if selected else None
