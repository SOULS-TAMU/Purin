"""Batch-native sparse complete Fischer--Burmeister Newton backend.

The forward semismooth Newton method solves the complete four-block system for
``(d_y, d_lambda, d_mu, d_s)``.  As in the optimized reduced backend, all real
batch rows are advanced by one batch-level nonlinear loop and one shared
batched GMRES invocation.  Constraint Jacobian and Hessian blocks are evaluated
once per Newton iteration and reused by the linear operator, preconditioner,
merit derivative, and final implicit adjoint.

The full backend deliberately retains the slack variables, so its forward and
adjoint systems have width ``n_y + n_eq + 2 * n_ineq``.  This is the principal
structural difference from the pivot-reduced backend.  The full-system
regularization also follows the original complete-Newton convention: positive
regularization on the primal block and negative regularization on the equality
and inequality-multiplier columns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
import jax
import jax.numpy as jnp
from purin.execution.runtime import resolve_device
from purin.linalg.sparse.operators import _batched_local_constraint_hessian_diagonal
from purin.linalg.sparse.operators import _batched_local_constraint_hessian_matvec
from purin.linalg.sparse.operators import (
    _batched_local_jacobian_matvec as _batched_local_jacobian_matvec,
)
from purin.linalg.sparse.operators import _batched_schur_diagonal
from purin.linalg.sparse.operators import (
    apply_batched_sparse_transpose as apply_batched_sparse_transpose,
)
from purin.linalg.sparse.operators import local_jacobian_transpose_matvec
from purin.linalg.sparse.operators import prepare_sparse_transpose_plan
from purin.linalg.sparse.operators import select_sparse_transpose_mode
from purin.solvers.config import FischerBurmeisterConfig
from purin.solvers.primal_dual_interior_point.operators import _batched_restarted_gmres
from purin.solvers.types import ProjectionInfo as ProjectionInfo
from purin.solvers.types import ProjectionResult as ProjectionResult
from purin.solvers.types import ProjectionTrainingOutput as ProjectionTrainingOutput
from purin.solvers.types import SolveStatus
from purin.solvers.fischer_burmeister.common import _FBProjectionSupport

if TYPE_CHECKING:
    from jax.typing import DTypeLike
    from purin.problems.canonical import CanonicalProblem

from purin.solvers.fischer_burmeister.common import (
    FischerBurmeisterState as FischerBurmeisterState,
)
from purin.solvers.fischer_burmeister.common import fischer_burmeister
from purin.solvers.fischer_burmeister.common import fischer_burmeister_partials
from purin.solvers.fischer_burmeister.common import initial_state as initial_state
from purin.solvers.fischer_burmeister.common import pack_state as pack_state


class FullFBNumericData(NamedTuple):
    """Numerical full-FB data evaluated once per batch Newton iteration."""

    residual: jax.Array
    residual_dual: jax.Array
    residual_equality: jax.Array
    residual_inequality: jax.Array
    residual_complementarity: jax.Array
    equality_jacobian: jax.Array
    inequality_jacobian: jax.Array
    equality_hessian: jax.Array
    inequality_hessian: jax.Array
    derivative_mu: jax.Array
    derivative_slack: jax.Array


class FullFBLinearization(NamedTuple):
    """Final local derivative blocks reused by the full implicit adjoint."""

    equality_jacobian: jax.Array
    inequality_jacobian: jax.Array
    equality_hessian: jax.Array
    inequality_hessian: jax.Array
    derivative_mu: jax.Array
    derivative_slack: jax.Array
    lambda_eq: jax.Array
    mu_ineq: jax.Array


@dataclass(frozen=True, eq=False)
class PreparedFullFischerBurmeisterProjection(_FBProjectionSupport):
    """Prepared full-system Fischer--Burmeister projection and derivative resources."""

    if TYPE_CHECKING:
        canonical: CanonicalProblem
    else:
        canonical: object
    config: FischerBurmeisterConfig
    batch_size: int
    if TYPE_CHECKING:
        dtype: DTypeLike
    else:
        dtype: object
    backend: str

    def __post_init__(self):
        """Prepare full-system residual, derivative and batch solve resources."""
        n = int(self.canonical.variable_size)
        me = int(self.canonical.equality_size)
        mi = int(self.canonical.inequality_size)

        all_rows = tuple(self.canonical.equality_sparsity or ()) + tuple(
            self.canonical.inequality_sparsity or ()
        )
        width = max((len(row) for row in all_rows), default=1)

        def make_layout(rows):
            rows = tuple(rows or ())
            support = []
            mask = []
            flat_positions = []
            for row_index, row in enumerate(rows):
                row = tuple(int(index) for index in row)
                support.append(row + (0,) * (width - len(row)))
                mask.append((True,) * len(row) + (False,) * (width - len(row)))
                flat_positions.extend(
                    row_index * width + slot for slot in range(len(row))
                )
            return (
                jnp.asarray(support, dtype=jnp.int32).reshape((len(rows), width)),
                jnp.asarray(mask, dtype=bool).reshape((len(rows), width)),
                jnp.asarray(flat_positions, dtype=jnp.int32),
            )

        eq_support, eq_mask, eq_positions = make_layout(
            self.canonical.equality_sparsity
        )
        ineq_support, ineq_mask, ineq_positions = make_layout(
            self.canonical.inequality_sparsity
        )
        combined_support = jnp.concatenate((eq_support, ineq_support), axis=0)
        combined_mask = jnp.concatenate((eq_mask, ineq_mask), axis=0)

        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_me", me)
        object.__setattr__(self, "_mi", mi)
        object.__setattr__(self, "_state_width", n + me + 2 * mi)
        object.__setattr__(self, "_row_width", width)
        object.__setattr__(self, "_equality_support", eq_support)
        object.__setattr__(self, "_equality_mask", eq_mask)
        object.__setattr__(self, "_equality_positions", eq_positions)
        object.__setattr__(self, "_inequality_support", ineq_support)
        object.__setattr__(self, "_inequality_mask", ineq_mask)
        object.__setattr__(self, "_inequality_positions", ineq_positions)
        object.__setattr__(self, "_combined_support", combined_support)
        object.__setattr__(self, "_combined_mask", combined_mask)

        selected_device = resolve_device(self.backend)
        if selected_device is None:
            selected_device = jax.devices()[0]
        transpose_mode = select_sparse_transpose_mode(
            eq_support,
            eq_mask,
            ineq_support,
            ineq_mask,
            n,
            backend=selected_device.platform,
            device_kind=str(getattr(selected_device, "device_kind", "")),
            mode=self.config.transpose_mode,
        )
        transpose_plan = prepare_sparse_transpose_plan(
            combined_support, combined_mask, n, mode=transpose_mode
        )
        object.__setattr__(self, "_transpose_mode", transpose_mode)
        object.__setattr__(self, "_combined_transpose_plan", transpose_plan)

        # Transform only numerical row kernels.  The nonlinear solver itself is
        # batch-native and is never wrapped in an outer vmap.
        object.__setattr__(
            self,
            "_jacobians_batch",
            jax.jit(
                jax.vmap(self._local_constraint_jacobians, in_axes=(0, 0)), inline=False
            ),
        )
        object.__setattr__(
            self,
            "_derivatives_batch",
            jax.jit(
                jax.vmap(self._local_constraint_derivatives, in_axes=(0, 0)),
                inline=False,
            ),
        )
        object.__setattr__(
            self,
            "_equalities_batch",
            jax.jit(jax.vmap(self.canonical.equalities, in_axes=(0, 0)), inline=False),
        )
        object.__setattr__(
            self,
            "_inequalities_batch",
            jax.jit(
                jax.vmap(self.canonical.inequalities, in_axes=(0, 0)), inline=False
            ),
        )
        object.__setattr__(
            self, "_residual_kernel", jax.jit(self._residual_batch_impl, inline=False)
        )
        object.__setattr__(
            self, "_numeric_kernel", jax.jit(self._evaluate_numeric_impl, inline=False)
        )
        object.__setattr__(
            self,
            "_linearization_kernel",
            jax.jit(self._evaluate_linearization_impl, inline=False),
        )

        solve_batch = jax.jit(self._solve_all_core)
        object.__setattr__(self, "_solve_batch", solve_batch)
        object.__setattr__(
            self,
            "_adjoint_kernel",
            jax.jit(self._solve_full_adjoint_batch, inline=False),
        )
        object.__setattr__(
            self,
            "_parameter_gradient_kernel",
            jax.jit(self._parameter_gradient_batch, inline=False),
        )
        object.__setattr__(self, "_implicit_project", self._make_implicit_projector())
        object.__setattr__(
            self, "_implicit_project_state", self._make_implicit_state_projector()
        )

    def _split_full_batch(self, vector):
        state = self._split_state_batch(vector)
        return state.y, state.lambda_eq, state.mu_ineq, state.slacks

    def _evaluate_numeric_impl(self, vector, y_hat, theta):
        """Evaluate residual and all local derivatives once for a Newton step."""
        state = self._split_state_batch(vector)
        (equality_values, equality_hessian, inequality_values, inequality_hessian) = (
            self._derivatives_batch(state.y, theta)
        )
        equality = self._equalities_batch(theta, state.y).reshape(
            (vector.shape[0], self._me)
        )
        inequality = self._inequalities_batch(theta, state.y).reshape(
            (vector.shape[0], self._mi)
        )
        combined_values = jnp.concatenate((equality_values, inequality_values), axis=1)
        combined_duals = jnp.concatenate((state.lambda_eq, state.mu_ineq), axis=1)
        residual_dual = (
            state.y - y_hat + self._combined_jtv(combined_values, combined_duals)
        )
        residual_inequality = inequality + state.slacks
        residual_complementarity = fischer_burmeister(
            state.mu_ineq, state.slacks, smoothing=self.config.smoothing
        )
        derivative_mu, derivative_slack = fischer_burmeister_partials(
            state.mu_ineq, state.slacks, smoothing=self.config.smoothing
        )
        residual = jnp.concatenate(
            (residual_dual, equality, residual_inequality, residual_complementarity),
            axis=1,
        )
        return FullFBNumericData(
            residual=residual,
            residual_dual=residual_dual,
            residual_equality=equality,
            residual_inequality=residual_inequality,
            residual_complementarity=residual_complementarity,
            equality_jacobian=equality_values,
            inequality_jacobian=inequality_values,
            equality_hessian=equality_hessian,
            inequality_hessian=inequality_hessian,
            derivative_mu=derivative_mu,
            derivative_slack=derivative_slack,
        )

    @staticmethod
    def _safe_signed(values, tolerance):
        sign = jnp.where(values >= 0.0, 1.0, -1.0)
        return jnp.where(jnp.abs(values) > tolerance, values, sign * tolerance)

    def _forward_preconditioner_data(self, numeric, state, regularization):
        """PDIP-style primal/Schur data with exact local FB 2x2 blocks."""
        dtype = state.y.dtype
        pivot_tolerance = jnp.asarray(self.config.pivot_tolerance, dtype)
        hessian_diagonal = self._weighted_hessian_diagonal(numeric, state)
        primal_scale = jnp.maximum(
            jnp.abs(hessian_diagonal) + regularization[:, None], pivot_tolerance
        )
        inverse_primal = 1.0 / primal_scale
        equality_scale = _batched_schur_diagonal(
            numeric.equality_jacobian,
            self._equality_support,
            self._equality_mask,
            inverse_primal,
            regularization[:, None],
        )
        inequality_schur = _batched_schur_diagonal(
            numeric.inequality_jacobian,
            self._inequality_support,
            self._inequality_mask,
            inverse_primal,
            jnp.zeros((state.y.shape[0], self._mi), dtype=dtype),
        )
        equality_scale = jnp.maximum(jnp.abs(equality_scale), pivot_tolerance)

        # After eliminating the primal approximation, the local (mu, s) block
        # is [[-(C D^-1 C^T + delta), 1], [a, b]].
        q = -(inequality_schur + regularization[:, None])
        determinant = q * numeric.derivative_slack - numeric.derivative_mu
        safe_determinant = self._safe_signed(determinant, pivot_tolerance)
        return primal_scale, equality_scale, q, safe_determinant

    def _linear_direction_batch(
        self, vector, numeric, active, regularization, initial_direction, residual_norm
    ):
        """Solve one complete four-block Newton system for the real batch."""
        state = self._split_state_batch(vector)
        dtype = vector.dtype
        combined_values = jnp.concatenate(
            (numeric.equality_jacobian, numeric.inequality_jacobian), axis=1
        )

        def operator(direction):
            d_y, d_lambda, d_mu, d_slack = self._split_full_batch(direction)
            combined_forward = self._combined_jv(combined_values, d_y)
            equality = combined_forward[:, : self._me]
            inequality = combined_forward[:, self._me :]
            combined_dual = jnp.concatenate((d_lambda, d_mu), axis=1)
            primal = self._weighted_hessian_matvec(numeric, state, d_y)
            primal = primal + self._combined_jtv(combined_values, combined_dual)

            # Preserve the original full-Newton regularization convention.
            primal = primal + regularization[:, None] * d_y
            equality = equality - regularization[:, None] * d_lambda
            inequality = inequality + d_slack - regularization[:, None] * d_mu
            complementarity = (
                numeric.derivative_mu * d_mu + numeric.derivative_slack * d_slack
            )
            return jnp.concatenate(
                (primal, equality, inequality, complementarity), axis=1
            )

        rhs = jnp.where(active[:, None], -numeric.residual, 0.0)
        initial_direction = jnp.where(active[:, None], initial_direction, 0.0)

        (primal_scale, equality_scale, q, safe_determinant) = (
            self._forward_preconditioner_data(numeric, state, regularization)
        )

        if self.config.linear_solver.use_preconditioner:

            def preconditioner(value):
                r_y, r_h, r_g, r_phi = self._split_full_batch(value)
                d_y = r_y / primal_scale
                d_lambda = r_h / equality_scale
                d_mu = (numeric.derivative_slack * r_g - r_phi) / safe_determinant
                d_slack = (-numeric.derivative_mu * r_g + q * r_phi) / safe_determinant
                return jnp.concatenate((d_y, d_lambda, d_mu, d_slack), axis=1)

        else:

            def preconditioner(value):
                return value

        # Match the reduced backend's inexact-Newton forcing rule.
        forcing = jnp.minimum(
            jnp.asarray(0.1, dtype),
            jnp.sqrt(jnp.maximum(residual_norm, jnp.asarray(0.0, dtype))),
        )
        linear_tolerance = jnp.maximum(
            jnp.asarray(self.config.linear_solver.tolerance, dtype), forcing
        )
        direction, linear_status, linear_residual, linear_iterations = (
            _batched_restarted_gmres(
                operator,
                rhs,
                preconditioner,
                tolerance=linear_tolerance,
                absolute_tolerance=self.config.linear_solver.atol,
                restart=min(
                    self.config.linear_solver.restart,
                    self.config.linear_solver.max_iterations,
                ),
                max_iterations=self.config.linear_solver.max_iterations,
                initial_solution=initial_direction,
                return_iterations=True,
            )
        )

        # Compute the merit derivative from the cached unregularized
        # generalized Jacobian instead of differentiating the residual again.
        d_y, d_lambda, d_mu, d_slack = self._split_full_batch(direction)
        combined_forward = self._combined_jv(combined_values, d_y)
        combined_dual = jnp.concatenate((d_lambda, d_mu), axis=1)
        j_dual = self._weighted_hessian_matvec(numeric, state, d_y)
        j_dual = j_dual + self._combined_jtv(combined_values, combined_dual)
        j_equality = combined_forward[:, : self._me]
        j_inequality = combined_forward[:, self._me :] + d_slack
        j_complementarity = (
            numeric.derivative_mu * d_mu + numeric.derivative_slack * d_slack
        )
        jacobian_direction = jnp.concatenate(
            (j_dual, j_equality, j_inequality, j_complementarity), axis=1
        )
        directional_derivative = jnp.sum(numeric.residual * jacobian_direction, axis=1)

        rhs_scale = jnp.max(jnp.abs(rhs), axis=1, initial=0.0)
        accepted_linear_residual = jnp.maximum(
            jnp.asarray(self.config.linear_solver.atol, dtype),
            linear_tolerance * rhs_scale,
        )
        finite = (
            jnp.all(jnp.isfinite(direction), axis=1)
            & jnp.isfinite(directional_derivative)
            & jnp.all(jnp.isfinite(numeric.derivative_mu), axis=1)
            & jnp.all(jnp.isfinite(numeric.derivative_slack), axis=1)
        )
        accurate = linear_residual <= accepted_linear_residual
        valid = (
            active
            & finite
            & ((linear_status == 0) | accurate)
            & (directional_derivative < 0.0)
        )
        return (
            direction,
            directional_derivative,
            valid,
            linear_status,
            linear_residual,
            linear_iterations,
        )

    def _evaluate_linearization_impl(self, vector, theta):
        """Evaluate the final local blocks needed by the explicit adjoint."""
        state = self._split_state_batch(vector)
        (equality_values, equality_hessian, inequality_values, inequality_hessian) = (
            self._derivatives_batch(state.y, theta)
        )
        derivative_mu, derivative_slack = fischer_burmeister_partials(
            state.mu_ineq, state.slacks, smoothing=self.config.smoothing
        )
        return FullFBLinearization(
            equality_jacobian=equality_values,
            inequality_jacobian=inequality_values,
            equality_hessian=equality_hessian,
            inequality_hessian=inequality_hessian,
            derivative_mu=derivative_mu,
            derivative_slack=derivative_slack,
            lambda_eq=state.lambda_eq,
            mu_ineq=state.mu_ineq,
        )

    def _solve_all_core(self, y_hat, theta, initial_vector):
        """Advance every sample with one batch-level semismooth Newton loop."""
        config = self.config
        batch_size = initial_vector.shape[0]
        residual0 = self._residual_kernel(initial_vector, y_hat, theta)
        norm0 = self._row_inf_norm(residual0)
        finite0 = jnp.all(jnp.isfinite(initial_vector), axis=1) & jnp.all(
            jnp.isfinite(residual0), axis=1
        )
        converged0 = finite0 & (norm0 <= config.tolerance)
        status0 = jnp.where(
            ~finite0,
            int(SolveStatus.NONFINITE),
            jnp.where(
                converged0, int(SolveStatus.CONVERGED), int(SolveStatus.MAX_ITERATIONS)
            ),
        ).astype(jnp.int32)
        regularization0 = jnp.full(
            (batch_size,), config.initial_regularization, dtype=initial_vector.dtype
        )
        direction0 = jnp.zeros_like(initial_vector)
        row_iterations0 = jnp.zeros((batch_size,), dtype=jnp.int32)
        global_iteration0 = jnp.asarray(0, dtype=jnp.int32)

        def condition(carry):
            (
                global_iteration,
                _vector,
                _direction,
                _regularization,
                status,
                _row_iterations,
                _norm,
            ) = carry
            return (global_iteration < config.max_iterations) & jnp.any(
                status == int(SolveStatus.MAX_ITERATIONS)
            )

        def body(carry):
            (
                global_iteration,
                vector,
                direction_guess,
                regularization,
                status,
                row_iterations,
                _previous_norm,
            ) = carry
            numeric = self._numeric_kernel(vector, y_hat, theta)
            residual = numeric.residual
            residual_norm = self._row_inf_norm(residual)
            finite = jnp.all(jnp.isfinite(vector), axis=1) & jnp.all(
                jnp.isfinite(residual), axis=1
            )
            active_before = status == int(SolveStatus.MAX_ITERATIONS)
            newly_nonfinite = active_before & ~finite
            newly_converged = (
                active_before & finite & (residual_norm <= config.tolerance)
            )
            status = jnp.where(newly_nonfinite, int(SolveStatus.NONFINITE), status)
            status = jnp.where(
                newly_converged, int(SolveStatus.CONVERGED), status
            ).astype(jnp.int32)
            active = status == int(SolveStatus.MAX_ITERATIONS)
            row_iterations = row_iterations + active.astype(jnp.int32)

            (
                direction,
                directional_derivative,
                valid,
                _linear_status,
                _linear_residual,
                _linear_iterations,
            ) = self._linear_direction_batch(
                vector, numeric, active, regularization, direction_guess, residual_norm
            )

            current_merit = self._row_merit(residual)
            search_mask = active & valid
            selected, selected_residual, accepted_alpha, accepted = (
                self._batch_backtracking(
                    vector,
                    direction,
                    y_hat,
                    theta,
                    residual,
                    current_merit,
                    directional_derivative,
                    search_mask,
                )
            )
            next_vector = jnp.where((active & accepted)[:, None], selected, vector)
            next_norm = jnp.where(
                active & accepted, self._row_inf_norm(selected_residual), residual_norm
            )

            # There is exactly one linear solve per nonlinear iteration.  The
            # accepted regularization is carried between iterations rather than
            # retried inside a nested regularization loop.
            growth = jnp.asarray(config.regularization_growth, vector.dtype)
            minimum_regularization = jnp.asarray(
                config.initial_regularization, vector.dtype
            )
            maximum_regularization = jnp.asarray(
                config.maximum_regularization, vector.dtype
            )
            decreased = jnp.maximum(minimum_regularization, regularization / growth)
            increased = jnp.minimum(maximum_regularization, regularization * growth)
            full_step = accepted & (
                accepted_alpha >= jnp.asarray(config.initial_step_size, vector.dtype)
            )
            next_regularization = jnp.where(
                active & full_step,
                decreased,
                jnp.where(
                    active & accepted,
                    regularization,
                    jnp.where(active, increased, regularization),
                ),
            )

            at_maximum = regularization >= maximum_regularization
            exhausted = active & ~accepted & at_maximum
            failure_status = jnp.where(
                valid,
                int(SolveStatus.LINE_SEARCH_FAILURE),
                int(SolveStatus.LINEAR_SOLVE_FAILURE),
            )
            status = jnp.where(exhausted, failure_status, status).astype(jnp.int32)
            finite_direction = jnp.all(jnp.isfinite(direction), axis=1)
            next_direction = jnp.where(
                (active & finite_direction)[:, None], direction, direction_guess
            )
            return (
                global_iteration + 1,
                next_vector,
                next_direction,
                next_regularization,
                status,
                row_iterations,
                next_norm,
            )

        (
            _global_iteration,
            vector,
            _direction,
            _regularization,
            status,
            row_iterations,
            _norm,
        ) = jax.lax.while_loop(
            condition,
            body,
            (
                global_iteration0,
                initial_vector,
                direction0,
                regularization0,
                status0,
                row_iterations0,
                norm0,
            ),
        )
        final_residual = self._residual_kernel(vector, y_hat, theta)
        final_norm = self._row_inf_norm(final_residual)
        finite = jnp.all(jnp.isfinite(vector), axis=1) & jnp.all(
            jnp.isfinite(final_residual), axis=1
        )
        converged = finite & (final_norm <= config.tolerance)
        status = jnp.where(converged, int(SolveStatus.CONVERGED), status)
        status = jnp.where(~finite, int(SolveStatus.NONFINITE), status).astype(
            jnp.int32
        )
        state = self._split_state_batch(vector)
        if config.implicit_gradient.policy == "strict":
            eligible = converged
        elif config.implicit_gradient.policy == "residual_checked":
            eligible = finite & (
                converged | (final_norm <= config.implicit_gradient.inexact_tolerance)
            )
        else:
            eligible = finite
        return (
            state.y,
            state.lambda_eq,
            state.mu_ineq,
            state.slacks,
            status,
            row_iterations,
            final_norm,
            finite,
            eligible,
        )

    def _full_root_residual(self, vector, y_hat, theta):
        """Single-row full FB residual used only for optional parameter VJPs."""
        n, me, mi = self._n, self._me, self._mi
        y = vector[:n]
        lambda_eq = vector[n : n + me]
        mu_ineq = vector[n + me : n + me + mi]
        slacks = vector[n + me + mi : n + me + 2 * mi]
        equality_values, inequality_values = self._local_constraint_jacobians(y, theta)
        equality = self.canonical.equalities(theta, y).reshape((me,))
        inequality = self.canonical.inequalities(theta, y).reshape((mi,))
        stationarity = y - y_hat
        if me:
            stationarity = stationarity + local_jacobian_transpose_matvec(
                equality_values,
                self._equality_support,
                self._equality_mask,
                lambda_eq,
                n,
            )
        if mi:
            stationarity = stationarity + local_jacobian_transpose_matvec(
                inequality_values,
                self._inequality_support,
                self._inequality_mask,
                mu_ineq,
                n,
            )
        return jnp.concatenate(
            (
                stationarity,
                equality,
                inequality + slacks,
                fischer_burmeister(mu_ineq, slacks, smoothing=self.config.smoothing),
            )
        )

    def _adjoint_preconditioner_data(self, linearization):
        """Build full-transpose Schur data from the cached final root blocks."""
        dtype = linearization.mu_ineq.dtype
        pivot_tolerance = jnp.asarray(self.config.pivot_tolerance, dtype)
        hessian_diagonal = 1.0 + _batched_local_constraint_hessian_diagonal(
            linearization.equality_hessian,
            linearization.inequality_hessian,
            self._equality_support,
            self._equality_mask,
            self._inequality_support,
            self._inequality_mask,
            linearization.lambda_eq,
            linearization.mu_ineq,
            self._n,
        )
        primal_scale = jnp.maximum(jnp.abs(hessian_diagonal), pivot_tolerance)
        inverse_primal = 1.0 / primal_scale
        equality_scale = _batched_schur_diagonal(
            linearization.equality_jacobian,
            self._equality_support,
            self._equality_mask,
            inverse_primal,
            jnp.zeros((linearization.lambda_eq.shape[0], self._me), dtype=dtype),
        )
        inequality_schur = _batched_schur_diagonal(
            linearization.inequality_jacobian,
            self._inequality_support,
            self._inequality_mask,
            inverse_primal,
            jnp.zeros((linearization.mu_ineq.shape[0], self._mi), dtype=dtype),
        )
        equality_scale = jnp.maximum(jnp.abs(equality_scale), pivot_tolerance)
        q = -inequality_schur
        determinant = q * linearization.derivative_slack - linearization.derivative_mu
        safe_determinant = self._safe_signed(determinant, pivot_tolerance)
        return primal_scale, equality_scale, q, safe_determinant

    def _solve_full_adjoint_batch(self, linearization, cotangent_y, eligible):
        """Solve the full cached transpose system with one GMRES cycle."""
        combined_values = jnp.concatenate(
            (linearization.equality_jacobian, linearization.inequality_jacobian), axis=1
        )

        def operator(adjoint_vector):
            value_dual, value_h, value_g, value_phi = self._split_full_batch(
                adjoint_vector
            )
            hessian_product = value_dual + _batched_local_constraint_hessian_matvec(
                linearization.equality_hessian,
                linearization.inequality_hessian,
                self._equality_support,
                self._equality_mask,
                self._inequality_support,
                self._inequality_mask,
                linearization.lambda_eq,
                linearization.mu_ineq,
                value_dual,
            )
            primal = hessian_product + self._combined_jtv(
                combined_values, jnp.concatenate((value_h, value_g), axis=1)
            )
            combined_forward = self._combined_jv(combined_values, value_dual)
            equality = combined_forward[:, : self._me]
            multiplier = (
                combined_forward[:, self._me :]
                + linearization.derivative_mu * value_phi
            )
            slack = value_g + linearization.derivative_slack * value_phi
            return jnp.concatenate((primal, equality, multiplier, slack), axis=1)

        rhs = jnp.concatenate(
            (
                cotangent_y,
                jnp.zeros(
                    (cotangent_y.shape[0], self._me + 2 * self._mi),
                    dtype=cotangent_y.dtype,
                ),
            ),
            axis=1,
        )
        rhs = jnp.where(eligible[:, None], rhs, 0.0)
        (primal_scale, equality_scale, q, safe_determinant) = (
            self._adjoint_preconditioner_data(linearization)
        )

        if self.config.linear_solver.use_preconditioner:

            def preconditioner(value):
                r_y, r_lambda, r_mu, r_slack = self._split_full_batch(value)
                value_dual = r_y / primal_scale
                value_h = r_lambda / equality_scale
                value_g = (
                    linearization.derivative_slack * r_mu
                    - linearization.derivative_mu * r_slack
                ) / safe_determinant
                value_phi = (-r_mu + q * r_slack) / safe_determinant
                return jnp.concatenate(
                    (value_dual, value_h, value_g, value_phi), axis=1
                )

        else:

            def preconditioner(value):
                return value

        # Ignore configured restart here.  max_iterations is the total Krylov
        # budget and is used as one single GMRES cycle, capped by system width.
        max_iterations = min(
            self.config.linear_solver.max_iterations, self._state_width
        )
        adjoint, status, residual, iterations = _batched_restarted_gmres(
            operator,
            rhs,
            preconditioner,
            tolerance=self.config.implicit_gradient.adjoint_tolerance,
            absolute_tolerance=self.config.linear_solver.atol,
            restart=max_iterations,
            max_iterations=max_iterations,
            initial_solution=jnp.zeros_like(rhs),
            return_iterations=True,
        )
        rhs_scale = jnp.max(jnp.abs(rhs), axis=1, initial=0.0)
        threshold = jnp.maximum(
            self.config.linear_solver.atol,
            self.config.implicit_gradient.adjoint_tolerance * rhs_scale,
        )
        finite = jnp.all(jnp.isfinite(adjoint), axis=1) & jnp.isfinite(residual)
        if self.config.implicit_gradient.policy == "always":
            adjoint_valid = finite
        else:
            adjoint_valid = finite & ((status == 0) | (residual <= threshold))
        valid = eligible & adjoint_valid
        return adjoint, valid, status, residual, iterations

    def _parameter_gradient_batch(self, solution_vector, y_hat, theta, adjoint, valid):
        if not (
            self.parameter_size > 0
            and self.config.implicit_gradient.differentiate_problem_parameters
        ):
            return jnp.zeros_like(theta)

        def one_gradient(vector, target, parameters, row_adjoint):
            _, pullback = jax.vjp(
                lambda current_theta: self._full_root_residual(
                    vector, target, current_theta
                ),
                parameters,
            )
            return -pullback(row_adjoint)[0]

        gradient = jax.vmap(one_gradient)(solution_vector, y_hat, theta, adjoint)
        return jnp.where(valid[:, None], gradient, 0.0)


@dataclass(frozen=True)
class FullFischerBurmeisterProjection:
    """Full-system Fischer--Burmeister distance-projection definition."""

    problem: object
    config: FischerBurmeisterConfig = FischerBurmeisterConfig()

    def prepare(self, *, batch_size: int, dtype=jnp.float64, backend: str = "cpu"):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        canonical = (
            self.problem.canonicalize()
            if hasattr(self.problem, "canonicalize")
            else self.problem
        )
        return PreparedFullFischerBurmeisterProjection(
            canonical=canonical,
            config=self.config,
            batch_size=int(batch_size),
            dtype=dtype,
            backend=str(backend),
        )
