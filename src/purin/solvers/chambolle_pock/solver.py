"""Sparse affine Euclidean projection with implicit fixed-point gradients."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from purin.problems.canonical import CanonicalQP
from purin.execution.runtime import resolve_device
from purin.solvers.primal_dual_interior_point.operators import _batched_restarted_gmres
from purin.solvers.types import ProjectionInfo
from purin.solvers.types import ProjectionResult
from purin.solvers.types import ProjectionTrainingOutput
from purin.solvers.types import SolveStatus
from purin.solvers.chambolle_pock.config import ChambollePockConfig
from purin.solvers.chambolle_pock.kernels import cp_step
from purin.solvers.chambolle_pock.kernels import diagonal_step_sizes
from purin.solvers.chambolle_pock.kernels import layout
from purin.solvers.chambolle_pock.kernels import norm_estimate
from purin.solvers.chambolle_pock.kernels import operators
from purin.solvers.chambolle_pock.kernels import sparse_cp


@dataclass(frozen=True, eq=False)
class PreparedChambollePockProjection:
    """Prepared affine distance projection using shared Chambolle--Pock kernels."""

    canonical: object
    config: ChambollePockConfig
    batch_size: int
    dtype: object
    backend: str

    def __post_init__(self):
        metadata = self.canonical.metadata or {}
        if not isinstance(self.canonical, CanonicalQP) and not (
            metadata.get("equalities_affine_in_variables") is True
            and metadata.get("inequalities_affine_in_variables") is True
        ):
            raise ValueError(
                "chambolle-pock requires verified affine equalities and inequalities; nonlinear or unverified constraints are unsupported"
            )
        if self.batch_size < 1 or self.output_size < 1:
            raise ValueError("batch_size and variable_size must be positive")
        object.__setattr__(self, "_device", resolve_device(self.backend))
        eq = self.canonical.equality_sparsity
        iq = self.canonical.inequality_sparsity
        if eq is None:
            eq = [tuple(range(self.output_size)) for _ in range(self.equality_size)]
        if iq is None:
            iq = [tuple(range(self.output_size)) for _ in range(self.inequality_size)]
        if len(eq) != self.equality_size or len(iq) != self.inequality_size:
            raise ValueError("constraint sparsity row counts do not match the problem")
        support, mask, positions = layout(tuple(eq) + tuple(iq), self.output_size)
        object.__setattr__(self, "_support", support)
        object.__setattr__(self, "_mask", mask)
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_eq_nnz", sum(map(len, eq)))
        # ADMM inherits the original preparation/plumbing but keeps its own
        # constraint representation and numerical kernels unchanged.
        if isinstance(self.config, ChambollePockConfig):
            from purin.solvers.chambolle_pock.bounds import prepare_bounds

            object.__setattr__(self, "_bound_projection", prepare_bounds(self))
        object.__setattr__(
            self, "_solve_jit", jax.jit(self._make_differentiable_solve())
        )

    @property
    def parameter_size(self):
        """Return the width of the projection problem context."""
        return int(self.canonical.parameter_size)

    @property
    def output_size(self):
        """Return the width of the projected primal vector."""
        return int(self.canonical.variable_size)

    @property
    def equality_size(self):
        """Return the number of equality residual rows."""
        return int(self.canonical.equality_size)

    @property
    def inequality_size(self):
        """Return the number of inequality residual rows."""
        return int(self.canonical.inequality_size)

    def _numeric_one(self, theta):
        """Evaluate one instance's affine constraint coefficients and bounds."""
        zero = jnp.zeros(self.output_size, dtype=self.dtype)
        me = self.equality_size
        mi = self.inequality_size
        blocks = []
        for count, offset, nnz, function, dense_function in (
            (
                me,
                0,
                self._eq_nnz,
                self.canonical.equality_jacobian_values,
                self.canonical.equality_jacobian,
            ),
            (
                mi,
                me,
                len(self._positions) - self._eq_nnz,
                self.canonical.inequality_jacobian_values,
                self.canonical.inequality_jacobian,
            ),
        ):
            support = self._support[offset : offset + count]
            mask = self._mask[offset : offset + count]
            if not count:
                block = jnp.zeros(support.shape, dtype=self.dtype)
            elif function is not None:
                positions = (
                    self._positions[:nnz]
                    if offset == 0
                    else self._positions[self._eq_nnz :] - me * self._support.shape[1]
                )
                # The inequality-only case has offset zero but no equality nonzeros.
                if offset == 0 and count == mi and me == 0:
                    positions = self._positions
                block = (
                    jnp.zeros(count * self._support.shape[1], dtype=self.dtype)
                    .at[positions]
                    .set(function(theta, zero))
                )
                block = block.reshape(support.shape)
            else:
                dense = dense_function(theta, zero).reshape((count, self.output_size))
                block = jnp.where(mask, dense[jnp.arange(count)[:, None], support], 0)
            blocks.append(block)
        rhs = -jnp.concatenate(
            (
                self.canonical.equalities(theta, zero).reshape((me,)),
                self.canonical.inequalities(theta, zero).reshape((mi,)),
            )
        )
        return jnp.concatenate(blocks, axis=0), rhs

    def _make_differentiable_solve(self):
        """Build CP projection with a custom implicit derivative through the final state."""
        me = self.equality_size
        n = self.output_size
        m = me + self.inequality_size
        cfg = self.config
        support = self._support
        mask = self._mask

        def forward(values, rhs, target, initial_dual, solve_tolerance):
            q = jnp.ones_like(target)
            lower = jnp.full_like(target, -jnp.inf)
            upper = -lower
            out = sparse_cp(
                q,
                -target,
                values,
                rhs,
                support,
                mask,
                me,
                lower,
                upper,
                accelerated=cfg.accelerated,
                safety=cfg.safety,
                max_iterations=cfg.max_iterations,
                tolerance=solve_tolerance,
                norm_iterations=cfg.norm_iterations,
                norm_seed=cfg.norm_seed,
                y0=target,
                dual0=initial_dual,
                freeze_converged=True,
                check_stationarity=True,
                norm_bound=cfg.norm_bound,
                diagonal_preconditioning=cfg.diagonal_preconditioning,
            )
            return out

        @jax.custom_vjp
        def solve(values, rhs, target, initial_dual, solve_tolerance):
            return forward(values, rhs, target, initial_dual, solve_tolerance)

        def fwd(values, rhs, target, initial_dual, solve_tolerance):
            out = forward(values, rhs, target, initial_dual, solve_tolerance)
            return out, (out, values, rhs, target)

        def bwd(saved, cot):
            out, values, rhs, target = saved
            mv, tmv = operators(values, support, mask, n)
            if cfg.diagonal_preconditioning and not cfg.accelerated:
                tau, sigma = diagonal_step_sizes(
                    jnp.ones_like(target), values, support, mask, cfg.safety
                )
            else:
                norm = norm_estimate(
                    mv,
                    tmv,
                    self.batch_size,
                    n,
                    target.dtype,
                    cfg.norm_iterations,
                    cfg.norm_seed,
                )
                L = jnp.maximum(norm, 1e-12)
                tau = jax.lax.stop_gradient(cfg.safety / (1 + L))
                sigma = jax.lax.stop_gradient(cfg.safety / L)
            lower = jnp.full_like(target, -jnp.inf)
            upper = -lower

            def pack(z):
                return jnp.concatenate(z, axis=1)

            def unpack(z):
                return z[:, :n], z[:, n : n + m], z[:, n + m :]

            state = pack((out.y, out.dual, out.y))

            def step(z, data, b, t):
                apply, transpose = operators(data, support, mask, n)
                return pack(
                    cp_step(
                        unpack(z),
                        jnp.ones_like(t),
                        -t,
                        b,
                        lower,
                        upper,
                        apply,
                        transpose,
                        me,
                        tau,
                        sigma,
                        jnp.ones(self.batch_size, target.dtype),
                    )
                )

            _, pullback = jax.vjp(step, state, values, rhs, target)
            operator = lambda v: v - pullback(v)[0]
            eligible = out.converged
            g = pack(
                (
                    jnp.where(eligible[:, None], cot.y, 0),
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
                eligible
                & jnp.all(jnp.isfinite(adjoint), axis=1)
                & jnp.isfinite(residual)
            )
            valid = valid & (
                (code == 0) | (residual <= cfg.implicit_gradient.adjoint_tolerance)
            )
            adjoint = jnp.where(valid[:, None], adjoint, 0)
            _, gv, gb, gt = pullback(adjoint)
            return (
                jnp.nan_to_num(gv, nan=0.0, posinf=0.0, neginf=0.0),
                jnp.nan_to_num(gb, nan=0.0, posinf=0.0, neginf=0.0),
                jnp.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0),
                jnp.zeros_like(out.dual),
                jnp.zeros((self.batch_size,), dtype=target.dtype),
            )

        solve.defvjp(fwd, bwd)
        return solve

    def _solve(
        self, y_hat, parameters=None, multipliers_eq=None, multipliers_ineq=None
    ):
        """Evaluate affine coefficients and run the shared batched CP iteration."""
        bounded = getattr(self, "_bound_projection", None)
        if bounded is not None:
            return bounded.solve(y_hat, parameters, multipliers_eq, multipliers_ineq)
        target = jax.device_put(jnp.asarray(y_hat, dtype=self.dtype), self._device)
        if target.ndim == 1:
            target = target[None, :]
        if target.shape != (self.batch_size, self.output_size):
            raise ValueError(
                f"y_hat must have shape {(self.batch_size, self.output_size)}; got {target.shape}"
            )
        if parameters is None and self.parameter_size:
            raise ValueError("parameters are required for this problem")
        theta = (
            jnp.zeros((self.batch_size, 0), dtype=self.dtype)
            if self.parameter_size == 0
            else jnp.asarray(parameters, dtype=self.dtype)
        )
        if theta.shape != (self.batch_size, self.parameter_size):
            raise ValueError(
                f"parameters must have shape {(self.batch_size, self.parameter_size)}; got {theta.shape}"
            )
        if not self.config.implicit_gradient.differentiate_problem_parameters:
            theta = jax.lax.stop_gradient(theta)
        theta = jax.device_put(theta, self._device)
        values, rhs = jax.vmap(self._numeric_one)(theta)
        scale = jnp.ones_like(rhs)
        if self.config.row_scaling:
            norms = jnp.sqrt(jnp.sum(values**2, axis=2))
            scale = jnp.where(norms > 0, 1 / jnp.maximum(norms, 1e-30), 1.0)
        data = values * scale[:, :, None]
        scaled_rhs = rhs * scale
        eq = (
            jnp.zeros((self.batch_size, self.equality_size), dtype=self.dtype)
            if multipliers_eq is None
            else jnp.asarray(multipliers_eq, dtype=self.dtype)
        )
        iq = (
            jnp.zeros((self.batch_size, self.inequality_size), dtype=self.dtype)
            if multipliers_ineq is None
            else jnp.asarray(multipliers_ineq, dtype=self.dtype)
        )
        if eq.shape != (self.batch_size, self.equality_size) or iq.shape != (
            self.batch_size,
            self.inequality_size,
        ):
            raise ValueError("warm-start multiplier shapes do not match constraints")
        initial = jax.lax.stop_gradient(jnp.concatenate((eq, iq), axis=1) / scale)
        solve_tolerance = jax.lax.stop_gradient(
            self.config.tolerance
            * jnp.minimum(1.0, jnp.min(scale, axis=1, initial=1.0))
        )
        out = self._solve_jit(data, scaled_rhs, target, initial, solve_tolerance)
        dual = out.dual * scale
        mv, tmv = operators(values, self._support, self._mask, self.output_size)
        r = mv(out.y) - rhs
        eq_res = jnp.max(abs(r[:, : self.equality_size]), axis=1, initial=0.0)
        ineq_res = jnp.max(
            jnp.maximum(r[:, self.equality_size :], 0), axis=1, initial=0.0
        )
        stationarity = jnp.max(abs(out.y - target + tmv(dual)), axis=1)
        residual = jnp.maximum(jnp.maximum(eq_res, ineq_res), stationarity)
        finite = (
            jnp.all(jnp.isfinite(out.y), axis=1)
            & jnp.all(jnp.isfinite(dual), axis=1)
            & jnp.isfinite(residual)
            & jnp.isfinite(out.gap)
        )
        converged = finite & out.converged & (residual <= self.config.tolerance)
        status = jnp.where(
            converged,
            int(SolveStatus.CONVERGED),
            jnp.where(
                finite, int(SolveStatus.MAX_ITERATIONS), int(SolveStatus.NONFINITE)
            ),
        ).astype(jnp.int32)
        return ProjectionResult(
            out.y,
            jax.lax.stop_gradient(dual[:, : self.equality_size]),
            jax.lax.stop_gradient(dual[:, self.equality_size :]),
            jax.lax.stop_gradient(jnp.maximum(-r[:, self.equality_size :], 0)),
            status,
            out.iterations,
            jax.lax.stop_gradient(residual),
            finite,
            converged,
        )

    def solve(self, y_hat, parameters=None, initial_state=None):
        """Solve the prepared projection with the supplied targets and problem context."""
        if initial_state is not None:
            raise ValueError(
                "Use project_training multiplier arguments for CP warm starts"
            )
        return self._solve(y_hat, parameters)

    def __call__(self, y_hat, parameters=None):
        result = self.solve(y_hat, parameters)
        return result.y[0] if jnp.asarray(y_hat).ndim == 1 else result.y

    def with_info(self, y_hat, parameters=None):
        """Evaluate the projection and return its primal values with solver diagnostics."""
        r = self.solve(y_hat, parameters)
        return (r.y[0] if jnp.asarray(y_hat).ndim == 1 else r.y), ProjectionInfo(
            r.status, r.iterations, r.residual_inf_norm, r.finite, r.gradient_eligible
        )

    def project_training(
        self, y_hat, parameters=None, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        r = self._solve(y_hat, parameters, multipliers_eq, multipliers_ineq)
        return ProjectionTrainingOutput(
            r.y,
            r.multipliers_eq,
            r.multipliers_ineq,
            r.slacks,
            r.status,
            r.iterations,
            r.residual_inf_norm,
            r.finite,
            r.gradient_eligible,
        )


@dataclass(frozen=True)
class ChambollePockProjection:
    """Chambolle--Pock distance-projection definition for verified affine constraints."""

    problem: object
    config: ChambollePockConfig = ChambollePockConfig()

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
        return PreparedChambollePockProjection(
            canonical, self.config, int(batch_size), dtype, str(backend)
        )
