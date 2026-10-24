"""Shared Fischer--Burmeister residuals, derivatives, and sparse preparation."""

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import jax
import jax.numpy as jnp
from purin.autodiff.complementarity import value as complementarity_value, partials

if TYPE_CHECKING:
    from collections.abc import Callable
    from jax.typing import DTypeLike
    from purin.problems.canonical import CanonicalProblem
    from purin.solvers.config import FischerBurmeisterConfig
    from purin.linalg.sparse.operators import SparseTransposePlan

ORIGIN_DERIVATIVE = 1.0 / jnp.sqrt(2.0) - 1.0


class FischerBurmeisterState(NamedTuple):
    """Primal values and multipliers used by the Fischer--Burmeister residual system."""

    y: jax.Array
    lambda_eq: jax.Array
    mu_ineq: jax.Array
    slacks: jax.Array


def fischer_burmeister(mu, slacks, *, smoothing: float = 0.0):
    """Evaluate the optionally smoothed Fischer--Burmeister mapping."""
    mu = jnp.asarray(mu)
    slacks = jnp.asarray(slacks)
    smoothing_value = jnp.asarray(smoothing, dtype=jnp.result_type(mu, slacks))
    return complementarity_value(mu, slacks, smoothing_value)


def fischer_burmeister_partials(mu, slacks, *, smoothing: float = 0.0):
    """Return the requested symmetric generalized derivative at the origin."""
    mu = jnp.asarray(mu)
    slacks = jnp.asarray(slacks)
    smoothing_value = jnp.asarray(smoothing, dtype=jnp.result_type(mu, slacks))
    return partials(mu, slacks, smoothing_value, origin=ORIGIN_DERIVATIVE)


def pack_state(state: FischerBurmeisterState):
    """Pack primal values and multiplier blocks into a Fischer--Burmeister state."""
    return jnp.concatenate((state.y, state.lambda_eq, state.mu_ineq, state.slacks))


def initial_state(canonical, y_hat, theta, *, slack_floor: float, multiplier: float):
    """Construct the initial primal and multiplier state for projection."""
    g = canonical.inequalities(theta, y_hat).reshape((canonical.inequality_size,))
    return FischerBurmeisterState(
        y=jnp.array(y_hat),
        lambda_eq=jnp.zeros((canonical.equality_size,), dtype=y_hat.dtype),
        mu_ineq=jnp.full((canonical.inequality_size,), multiplier, dtype=y_hat.dtype),
        slacks=jnp.maximum(-g, jnp.asarray(slack_floor, y_hat.dtype)),
    )


# Final-KKT implicit differentiation.


from purin.solvers.types import ProjectionInfo
from purin.solvers.types import ProjectionResult
from purin.solvers.types import ProjectionTrainingOutput
from purin.linalg.sparse.operators import _batched_local_constraint_hessian_diagonal
from purin.linalg.sparse.operators import _batched_local_constraint_hessian_matvec
from purin.linalg.sparse.operators import _batched_local_jacobian_matvec
from purin.linalg.sparse.operators import apply_batched_sparse_transpose


class _FBProjectionSupport:
    """Identical full/reduced projection support; numerical operators stay backend-specific."""

    # Construction in each backend supplies these resources once. Type-only
    # declarations preserve the existing dataclass fields and runtime hints.
    if TYPE_CHECKING:
        canonical: CanonicalProblem
        config: FischerBurmeisterConfig
        batch_size: int
        dtype: DTypeLike
        _n: int
        _me: int
        _mi: int
        _row_width: int
        _state_width: int
        _equality_support: jax.Array
        _equality_mask: jax.Array
        _equality_positions: jax.Array
        _inequality_support: jax.Array
        _inequality_mask: jax.Array
        _inequality_positions: jax.Array
        _combined_support: jax.Array
        _combined_mask: jax.Array
        _combined_transpose_plan: SparseTransposePlan
        _jacobians_batch: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
        _equalities_batch: Callable[[jax.Array, jax.Array], jax.Array]
        _inequalities_batch: Callable[[jax.Array, jax.Array], jax.Array]
        _residual_kernel: Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
        _solve_batch: Callable[
            [jax.Array, jax.Array, jax.Array],
            tuple[
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
        ]
        _linearization_kernel: Callable[[jax.Array, jax.Array], object]
        _adjoint_kernel: Callable[
            [object, jax.Array, jax.Array],
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ]
        _parameter_gradient_kernel: Callable[
            [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
        ]
        _implicit_project: Callable[[jax.Array, jax.Array], jax.Array]
        _implicit_project_state: Callable[
            [jax.Array, jax.Array, jax.Array], ProjectionTrainingOutput
        ]

    @property
    def parameter_size(self):
        """Return the width of the projection problem context."""
        return int(self.canonical.parameter_size)

    @property
    def output_size(self):
        """Return the width of the projected primal vector."""
        return int(self.canonical.variable_size)

    def _split_state_batch(self, vector):
        n, me, mi = self._n, self._me, self._mi
        return FischerBurmeisterState(
            y=vector[:, :n],
            lambda_eq=vector[:, n : n + me],
            mu_ineq=vector[:, n + me : n + me + mi],
            slacks=vector[:, n + me + mi : n + me + 2 * mi],
        )

    def _pack_state_batch(self, state):
        return jnp.concatenate(
            (state.y, state.lambda_eq, state.mu_ineq, state.slacks), axis=1
        )

    def _local_constraint_jacobians(self, y, theta):
        """Return fixed-width row-local Jacobian values without Hessians."""

        def local_values(count, function, dense_function, support, mask, positions):
            if count == 0:
                return jnp.empty((0, self._row_width), dtype=y.dtype)
            if function is None:
                dense = dense_function(theta, y).reshape((count, self._n))
                rows = jnp.arange(count, dtype=jnp.int32)[:, None]
                return jnp.where(mask, dense[rows, support], 0.0)
            flat = function(theta, y)
            padded = jnp.zeros((count * self._row_width,), dtype=y.dtype)
            padded = padded.at[positions].set(flat)
            return padded.reshape((count, self._row_width))

        equality_values = local_values(
            self._me,
            getattr(self.canonical, "equality_jacobian_values", None),
            self.canonical.equality_jacobian,
            self._equality_support,
            self._equality_mask,
            self._equality_positions,
        )
        inequality_values = local_values(
            self._mi,
            getattr(self.canonical, "inequality_jacobian_values", None),
            self.canonical.inequality_jacobian,
            self._inequality_support,
            self._inequality_mask,
            self._inequality_positions,
        )
        return equality_values, inequality_values

    def _local_constraint_derivatives(self, y, theta):
        """Return local Jacobian/Hessian blocks through the PDIP derivative path."""

        def fallback(function, count, support, mask):
            if count == 0:
                return (
                    jnp.empty((0, self._row_width), dtype=y.dtype),
                    jnp.empty((0, self._row_width, self._row_width), dtype=y.dtype),
                )
            jacobian = jax.jacfwd(function, argnums=1)(theta, y).reshape(
                (count, self._n)
            )
            hessian = jax.jacfwd(jax.jacfwd(function, argnums=1), argnums=1)(
                theta, y
            ).reshape((count, self._n, self._n))
            rows = jnp.arange(count, dtype=jnp.int32)[:, None]
            local_jacobian = jnp.where(mask, jacobian[rows, support], 0.0)
            local_hessian = hessian[
                rows[:, :, None], support[:, :, None], support[:, None, :]
            ]
            local_hessian = jnp.where(
                mask[:, :, None] & mask[:, None, :], local_hessian, 0.0
            )
            return local_jacobian, local_hessian

        equality_function = getattr(self.canonical, "equality_local_derivatives", None)
        inequality_function = getattr(
            self.canonical, "inequality_local_derivatives", None
        )
        if equality_function is None:
            equality_values, equality_hessian = fallback(
                self.canonical.equalities,
                self._me,
                self._equality_support,
                self._equality_mask,
            )
        else:
            equality_values, equality_hessian = equality_function(theta, y)
        if inequality_function is None:
            inequality_values, inequality_hessian = fallback(
                self.canonical.inequalities,
                self._mi,
                self._inequality_support,
                self._inequality_mask,
            )
        else:
            inequality_values, inequality_hessian = inequality_function(theta, y)
        return (
            equality_values,
            equality_hessian,
            inequality_values,
            inequality_hessian,
        )

    def _combined_jv(self, values, vector):
        return _batched_local_jacobian_matvec(
            values, self._combined_support, self._combined_mask, vector
        )

    def _combined_jtv(self, values, vector):
        return apply_batched_sparse_transpose(
            self._combined_transpose_plan, values, vector
        )

    def _residual_batch_impl(self, vector, y_hat, theta):
        """Evaluate the complete FB residual for the real batch."""
        state = self._split_state_batch(vector)
        equality_values, inequality_values = self._jacobians_batch(state.y, theta)
        equality = self._equalities_batch(theta, state.y).reshape(
            (vector.shape[0], self._me)
        )
        inequality = self._inequalities_batch(theta, state.y).reshape(
            (vector.shape[0], self._mi)
        )
        combined_values = jnp.concatenate((equality_values, inequality_values), axis=1)
        combined_duals = jnp.concatenate((state.lambda_eq, state.mu_ineq), axis=1)
        stationarity = (
            state.y - y_hat + self._combined_jtv(combined_values, combined_duals)
        )
        feasibility = inequality + state.slacks
        complementarity = fischer_burmeister(
            state.mu_ineq, state.slacks, smoothing=self.config.smoothing
        )
        return jnp.concatenate(
            (stationarity, equality, feasibility, complementarity), axis=1
        )

    @staticmethod
    def _row_inf_norm(values):
        return jnp.max(jnp.abs(values), axis=1, initial=0.0)

    @staticmethod
    def _row_merit(values):
        return 0.5 * jnp.sum(values * values, axis=1)

    def _weighted_hessian_matvec(self, numeric, state, vector):
        return vector + _batched_local_constraint_hessian_matvec(
            numeric.equality_hessian,
            numeric.inequality_hessian,
            self._equality_support,
            self._equality_mask,
            self._inequality_support,
            self._inequality_mask,
            state.lambda_eq,
            state.mu_ineq,
            vector,
        )

    def _weighted_hessian_diagonal(self, numeric, state):
        return 1.0 + _batched_local_constraint_hessian_diagonal(
            numeric.equality_hessian,
            numeric.inequality_hessian,
            self._equality_support,
            self._equality_mask,
            self._inequality_support,
            self._inequality_mask,
            state.lambda_eq,
            state.mu_ineq,
            self._n,
        )

    def _batch_backtracking(
        self,
        vector,
        direction,
        y_hat,
        theta,
        current_residual,
        current_merit,
        directional_derivative,
        search_mask,
    ):
        """One scalar-control-flow Armijo loop for the complete batch."""
        dtype = vector.dtype
        initial_alpha = jnp.full(
            (vector.shape[0],), self.config.initial_step_size, dtype=dtype
        )
        minimum_alpha = jnp.asarray(self.config.minimum_step_size, dtype)
        reduction = jnp.asarray(self.config.backtracking_factor, dtype)
        armijo = jnp.asarray(self.config.armijo_constant, dtype)
        carry = (
            jnp.asarray(0, jnp.int32),
            initial_alpha,
            vector,
            current_residual,
            jnp.zeros((vector.shape[0],), dtype=bool),
        )

        def condition(values):
            step, alpha, _trial, _trial_residual, accepted = values
            searching = search_mask & ~accepted & (alpha >= minimum_alpha)
            return (step < int(self.config.maximum_backtracking_steps)) & jnp.any(
                searching
            )

        def body(values):
            step, alpha, selected, selected_residual, accepted = values
            searching = search_mask & ~accepted & (alpha >= minimum_alpha)
            candidate = vector + alpha[:, None] * direction
            candidate = jnp.where(searching[:, None], candidate, vector)
            candidate_residual = self._residual_kernel(candidate, y_hat, theta)
            candidate_merit = self._row_merit(candidate_residual)
            finite = jnp.all(jnp.isfinite(candidate), axis=1) & jnp.all(
                jnp.isfinite(candidate_residual), axis=1
            )
            accepted_now = (
                searching
                & finite
                & (
                    candidate_merit
                    <= current_merit + armijo * alpha * directional_derivative
                )
            )
            selected = jnp.where(accepted_now[:, None], candidate, selected)
            selected_residual = jnp.where(
                accepted_now[:, None], candidate_residual, selected_residual
            )
            accepted = accepted | accepted_now
            alpha = jnp.where(accepted_now, alpha, alpha * reduction)
            return step + 1, alpha, selected, selected_residual, accepted

        _step, alpha, selected, selected_residual, accepted = jax.lax.while_loop(
            condition, body, carry
        )
        accepted_alpha = jnp.where(accepted, alpha, 0.0)
        return selected, selected_residual, accepted_alpha, accepted

    def _initial_vectors_batch(self, y_hat, theta):
        return jax.vmap(
            lambda target, parameters: pack_state(
                initial_state(
                    self.canonical,
                    target,
                    parameters,
                    slack_floor=self.config.initial_slack_floor,
                    multiplier=self.config.initial_multiplier,
                )
            )
        )(y_hat, theta)

    def _make_implicit_projector(self):
        @jax.custom_vjp
        def project(y_hat, theta):
            values = self._solve_batch(
                y_hat, theta, self._initial_vectors_batch(y_hat, theta)
            )
            return values[0]

        def forward(y_hat, theta):
            values = self._solve_batch(
                y_hat, theta, self._initial_vectors_batch(y_hat, theta)
            )
            solution_vector = jnp.concatenate(values[:4], axis=1)
            return values[0], (solution_vector, y_hat, theta, values[8])

        def backward(saved, cotangent_y):
            solution_vector, y_hat, theta, eligible = saved
            linearization = self._linearization_kernel(solution_vector, theta)
            adjoint, valid, _status, _residual, _iterations = self._adjoint_kernel(
                linearization, cotangent_y, eligible
            )
            grad_y_hat = jnp.where(valid[:, None], adjoint[:, : self._n], 0.0)
            grad_theta = self._parameter_gradient_kernel(
                solution_vector, y_hat, theta, adjoint, valid
            )
            return grad_y_hat, grad_theta

        project.defvjp(forward, backward)
        return project

    def _make_implicit_state_projector(self):
        """Attach the full-state implicit pullback to the forward projection solve."""

        @jax.custom_vjp
        def project_state(y_hat, theta, initial_vectors):
            values = self._solve_batch(y_hat, theta, initial_vectors)
            return ProjectionTrainingOutput(
                y_tilde=values[0],
                multipliers_eq=jax.lax.stop_gradient(values[1]),
                multipliers_ineq=jax.lax.stop_gradient(values[2]),
                slacks=jax.lax.stop_gradient(values[3]),
                forward_status_codes=jax.lax.stop_gradient(values[4]),
                forward_iterations=jax.lax.stop_gradient(values[5]),
                forward_residual_inf_norm=jax.lax.stop_gradient(values[6]),
                forward_state_finite=jax.lax.stop_gradient(values[7]),
                forward_gradient_eligible=jax.lax.stop_gradient(values[8]),
            )

        def forward(y_hat, theta, initial_vectors):
            values = self._solve_batch(y_hat, theta, initial_vectors)
            output = ProjectionTrainingOutput(
                y_tilde=values[0],
                multipliers_eq=jax.lax.stop_gradient(values[1]),
                multipliers_ineq=jax.lax.stop_gradient(values[2]),
                slacks=jax.lax.stop_gradient(values[3]),
                forward_status_codes=jax.lax.stop_gradient(values[4]),
                forward_iterations=jax.lax.stop_gradient(values[5]),
                forward_residual_inf_norm=jax.lax.stop_gradient(values[6]),
                forward_state_finite=jax.lax.stop_gradient(values[7]),
                forward_gradient_eligible=jax.lax.stop_gradient(values[8]),
            )
            solution_vector = jnp.concatenate(values[:4], axis=1)
            return output, (solution_vector, y_hat, theta, values[8], initial_vectors)

        def backward(saved, output_cotangent):
            (solution_vector, y_hat, theta, eligible, initial_vectors) = saved
            linearization = self._linearization_kernel(solution_vector, theta)
            adjoint, valid, _status, _residual, _iterations = self._adjoint_kernel(
                linearization, output_cotangent.y_tilde, eligible
            )
            grad_y_hat = jnp.where(valid[:, None], adjoint[:, : self._n], 0.0)
            grad_theta = self._parameter_gradient_kernel(
                solution_vector, y_hat, theta, adjoint, valid
            )
            return grad_y_hat, grad_theta, jnp.zeros_like(initial_vectors)

        project_state.defvjp(forward, backward)
        return project_state

    def _normalize_inputs(self, y_hat, parameters, initial_state_value):
        y_hat = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = y_hat.ndim == 1
        if squeeze:
            y_hat = y_hat[None, :]
        if y_hat.shape != (self.batch_size, self.output_size):
            raise ValueError(
                f"y_hat must have shape {(self.batch_size, self.output_size)}; got {y_hat.shape}."
            )
        if self.parameter_size == 0:
            theta = jnp.zeros((self.batch_size, 0), dtype=self.dtype)
        else:
            if parameters is None:
                raise ValueError("parameters are required for this problem")
            theta = jnp.asarray(parameters, dtype=self.dtype)
            if theta.shape != (self.batch_size, self.parameter_size):
                raise ValueError(
                    f"parameters must have shape "
                    f"{(self.batch_size, self.parameter_size)}; got {theta.shape}."
                )
        if initial_state_value is None:
            initial_vectors = self._initial_vectors_batch(y_hat, theta)
        elif isinstance(initial_state_value, FischerBurmeisterState):
            initial_vectors = self._pack_state_batch(initial_state_value)
        else:
            initial_vectors = jnp.asarray(initial_state_value, dtype=self.dtype)
        if initial_vectors.shape != (self.batch_size, self._state_width):
            raise ValueError(
                f"initial_state must have shape "
                f"{(self.batch_size, self._state_width)}; got {initial_vectors.shape}."
            )
        return y_hat, theta, initial_vectors, squeeze

    def solve(self, y_hat, parameters=None, initial_state=None):
        """Solve the prepared projection with the supplied targets and problem context."""
        y_hat, theta, initial_vectors, _squeeze = self._normalize_inputs(
            y_hat, parameters, initial_state
        )
        return ProjectionResult(*self._solve_batch(y_hat, theta, initial_vectors))

    def project_training(
        self, y_hat, parameters=None, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        values, theta, default_vectors, _squeeze = self._normalize_inputs(
            y_hat, parameters, None
        )
        if multipliers_eq is None and multipliers_ineq is None:
            initial_vectors = default_vectors
        else:
            default_state = self._split_state_batch(default_vectors)
            lambda_eq = (
                default_state.lambda_eq
                if multipliers_eq is None
                else jnp.asarray(multipliers_eq, dtype=self.dtype)
            )
            mu_ineq = (
                default_state.mu_ineq
                if multipliers_ineq is None
                else jnp.asarray(multipliers_ineq, dtype=self.dtype)
            )
            initial_vectors = self._pack_state_batch(
                FischerBurmeisterState(values, lambda_eq, mu_ineq, default_state.slacks)
            )
        return self._implicit_project_state(values, theta, initial_vectors)

    def __call__(self, y_hat, parameters=None):
        values, theta, _initial_vectors, squeeze = self._normalize_inputs(
            y_hat, parameters, None
        )
        projected = self._implicit_project(values, theta)
        return projected[0] if squeeze else projected

    def with_info(self, y_hat, parameters=None):
        """Evaluate the projection and return its primal values with solver diagnostics."""
        result = self.solve(y_hat, parameters)
        projected = result.y[0] if jnp.asarray(y_hat).ndim == 1 else result.y
        return projected, ProjectionInfo(
            result.status,
            result.iterations,
            result.residual_inf_norm,
            result.finite,
            result.gradient_eligible,
        )
