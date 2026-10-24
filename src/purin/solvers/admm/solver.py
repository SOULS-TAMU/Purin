"""OSQP-style sparse ADMM with the existing projection API and KKT VJP."""

from dataclasses import dataclass, field
from typing import NamedTuple
import jax
import jax.numpy as jnp
from purin.problems.canonical import CanonicalQP
from purin.solvers.chambolle_pock.solver import PreparedChambollePockProjection
from purin.solvers.primal_dual_interior_point.operators import _batched_restarted_gmres
from purin.solvers.admm.rows import prepare_pairs
from purin.solvers.admm.factor import prepare_banded
from purin.solvers.admm.config import ADMMConfig
from purin.solvers.admm.kernels import operators
from purin.solvers.admm.kernels import sparse_admm


class ADMMProjectionOutput(NamedTuple):
    """Primal projection values and ADMM-specific solver diagnostics."""

    y: object
    dual: object
    iterations: object
    residual: object
    gap: object
    converged: object


@dataclass(frozen=True, eq=False)
class PreparedADMMProjection(PreparedChambollePockProjection):
    """Reuse affine preparation/API plumbing; forward/adjoint computations are ADMM.

    The base class supplies only constraint compilation, shape checks, row
    normalization, result packaging and the public entry points. Its CP
    iteration/adjoint code is overridden and is never executed here.
    """

    config: ADMMConfig

    def __post_init__(self):
        metadata = self.canonical.metadata or {}
        if not isinstance(self.canonical, CanonicalQP) and not (
            metadata.get("equalities_affine_in_variables") is True
            and metadata.get("inequalities_affine_in_variables") is True
        ):
            raise ValueError(
                "admm requires verified affine equalities and inequalities; nonlinear or unverified constraints are unsupported"
            )
        super().__post_init__()

    def _make_differentiable_solve(self):
        """Build the ADMM forward solve and its configured implicit pullback."""
        cfg = self.config
        me = self.equality_size
        n = self.output_size
        support = self._support
        mask = self._mask
        factor_plan = prepare_banded(support, mask, n)
        object.__setattr__(self, "_factor_plan", factor_plan)
        paired = prepare_pairs(self)
        object.__setattr__(self, "_paired_rows", paired)

        def run(
            values,
            lower,
            upper,
            target,
            initial_dual,
            solve_tolerance,
            local_support,
            local_mask,
            plan,
        ):
            penalty = (
                jnp.full_like(upper, cfg.rho)
                .at[:, :me]
                .multiply(cfg.equality_rho_multiplier)
            )
            return sparse_admm(
                jnp.ones_like(target),
                -target,
                values,
                lower,
                upper,
                local_support,
                local_mask,
                rho=penalty,
                sigma=cfg.sigma,
                alpha=cfg.alpha,
                tolerance=solve_tolerance,
                max_iterations=cfg.max_iterations,
                cg_iterations=cfg.cg_max_iterations,
                cg_tolerance=cfg.cg_tolerance,
                refinement_steps=cfg.cg_refinement_steps,
                adaptive_rho=cfg.adaptive_rho,
                rho_update_interval=cfg.rho_update_interval,
                rho_update_threshold=cfg.rho_update_threshold,
                x0=target,
                dual0=initial_dual,
                factor_plan=plan,
                check_interval=cfg.check_interval,
            )

        def forward(values, rhs, target, initial_dual, solve_tolerance):
            lower = jnp.concatenate(
                (rhs[:, :me], jnp.full_like(rhs[:, me:], -jnp.inf)), axis=1
            )

            def full(_):
                out = run(
                    values,
                    lower,
                    rhs,
                    target,
                    initial_dual,
                    solve_tolerance,
                    support,
                    mask,
                    factor_plan,
                )
                return (
                    out.x,
                    out.dual,
                    out.iterations,
                    out.primal_residual,
                    out.dual_residual,
                    out.linear_residual,
                    out.converged,
                )

            if paired is None:
                result = full(None)
            else:

                def compact(_):
                    lo, hi, initial = paired.compact(rhs, initial_dual, me)
                    out = run(
                        values[:, paired.keep],
                        lo,
                        hi,
                        target,
                        initial,
                        solve_tolerance,
                        paired.support,
                        paired.mask,
                        paired.factor_plan,
                    )
                    return (
                        out.x,
                        paired.restore(out.dual, rhs),
                        out.iterations,
                        out.primal_residual,
                        out.dual_residual,
                        out.linear_residual,
                        out.converged,
                    )

                result = jax.lax.cond(paired.verified(values, rhs), compact, full, None)
            (
                x,
                dual,
                iterations,
                primal_residual,
                dual_residual,
                linear_residual,
                converged,
            ) = result
            mv, tmv = operators(values, support, mask, n)
            primal = jnp.sum(0.5 * x**2 - target * x, axis=1)
            t = -target + tmv(dual)
            dual_objective = -0.5 * jnp.sum(t * t, axis=1) - jnp.sum(rhs * dual, axis=1)
            gap = jnp.maximum(primal - dual_objective, 0) / jnp.maximum(
                1.0, jnp.maximum(abs(primal), abs(dual_objective))
            )
            residual = jnp.maximum(
                primal_residual, jnp.maximum(dual_residual, linear_residual)
            )
            return ADMMProjectionOutput(x, dual, iterations, residual, gap, converged)

        @jax.custom_vjp
        def solve(values, rhs, target, initial_dual, solve_tolerance):
            return forward(values, rhs, target, initial_dual, solve_tolerance)

        def fwd(values, rhs, target, initial_dual, solve_tolerance):
            out = forward(values, rhs, target, initial_dual, solve_tolerance)
            return out, (out, values, rhs)

        def bwd(saved, cot):
            out, values, rhs = saved
            mv, tmv = operators(values, support, mask, n)
            # Equality rows are always active. An inequality with a positive
            # multiplier and a tight slack defines a locally active constraint.
            slack = rhs - mv(out.y)
            active = jnp.concatenate(
                (
                    jnp.ones_like(rhs[:, :me], dtype=bool),
                    (out.dual[:, me:] > cfg.active_tolerance)
                    & (
                        slack[:, me:]
                        <= jnp.maximum(cfg.active_tolerance, cfg.tolerance)
                    ),
                ),
                axis=1,
            )
            active = jax.lax.stop_gradient(active.astype(out.y.dtype))

            def operator(v):
                primal = v[:, :n]
                dual = v[:, n:]
                return jnp.concatenate(
                    (
                        primal + tmv(active * dual),
                        active * mv(primal) + (1 - active) * dual,
                    ),
                    axis=1,
                )

            rhs_adjoint = jnp.concatenate(
                (jnp.where(out.converged[:, None], cot.y, 0), jnp.zeros_like(rhs)),
                axis=1,
            )
            linear = cfg.linear_solver
            adjoint, status, residual = _batched_restarted_gmres(
                operator,
                rhs_adjoint,
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
            )
            valid = valid & (
                (status == 0) | (residual <= cfg.implicit_gradient.adjoint_tolerance)
            )
            adjoint = jnp.where(valid[:, None], adjoint, 0.0)
            primal = adjoint[:, :n]
            dual = active * adjoint[:, n:]
            # Differentiate final stationarity and active affine constraints.
            gradient_values = -(
                out.dual[:, :, None] * primal[:, support]
                + dual[:, :, None] * out.y[:, support]
            )
            gradient_values = jnp.where(mask[None, :, :], gradient_values, 0.0)
            clean = lambda value: jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return (
                clean(gradient_values),
                clean(dual),
                clean(primal),
                jnp.zeros_like(out.dual),
                jnp.zeros((self.batch_size,), dtype=out.y.dtype),
            )

        solve.defvjp(fwd, bwd)
        return solve

    def solve(self, y_hat, parameters=None, initial_state=None):
        """Solve the prepared batched ADMM projection problem."""
        if initial_state is not None:
            raise ValueError(
                "Use project_training multiplier arguments for ADMM warm starts"
            )
        return self._solve(y_hat, parameters)


@dataclass(frozen=True)
class ADMMProjection:
    """Definition of an ADMM distance projection for affine constraints."""

    problem: object
    config: ADMMConfig = field(default_factory=ADMMConfig)

    def prepare(self, *, batch_size, dtype=jnp.float64, backend="cpu"):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        if self.config.metric_diagonal is not None:
            from purin.solvers.metric import prepare_diagonal_metric

            return prepare_diagonal_metric(
                self.problem,
                self.config,
                type(self),
                batch_size=batch_size,
                dtype=dtype,
                backend=backend,
            )
        canonical = (
            self.problem.canonicalize()
            if hasattr(self.problem, "canonicalize")
            else self.problem
        )
        return PreparedADMMProjection(
            canonical, self.config, int(batch_size), dtype, str(backend)
        )
