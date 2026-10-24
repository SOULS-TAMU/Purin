"""Prepared and public sparse primal-dual interior-point projection solvers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from purin.problems.problem import Problem as CoreProblem
from purin.linalg.sparse.operators import _batched_local_jacobian_matvec
from purin.linalg.sparse.operators import local_jacobian_matvec
from purin.linalg.sparse.operators import prepare_sparse_transpose_plan
from purin._utils.validation import strict_zip
from purin.solvers.config import ProjectionConfig
from purin.solvers.implicit import make_implicit_projection_layer
from purin.solvers.types import ProjectionInfo
from purin.solvers.types import ProjectionResult
from purin.solvers.types import SolveStatus
from purin.solvers.primal_dual_interior_point.assembly import (
    _assemble_numeric_data_from_constraint_values,
)
from purin.solvers.primal_dual_interior_point.assembly import _evaluate_numeric_data
from purin.solvers.primal_dual_interior_point.assembly import _residual_inf_norm
from purin.solvers.primal_dual_interior_point.config import FixedMuOptions
from purin.solvers.primal_dual_interior_point.newton import apply_primal_dual_step
from purin.solvers.primal_dual_interior_point.newton import compute_bound_slacks
from purin.solvers.primal_dual_interior_point.newton import (
    compute_fraction_to_boundary_steps,
)
from purin.solvers.primal_dual_interior_point.newton import (
    solve_primal_dual_newton_system,
)
from purin.solvers.primal_dual_interior_point.operators import (
    _build_sparse_projection_linearization,
)
from purin.solvers.primal_dual_interior_point.operators import (
    _solve_condensed_direction_batch,
)
from purin.solvers.primal_dual_interior_point.operators import (
    _solve_condensed_yhat_adjoint_batch,
)
from purin.solvers.primal_dual_interior_point.operators import (
    build_affine_constant_kkt_pallas_matvec,
)
from purin.solvers.primal_dual_interior_point.operators import pallas_available
from purin.solvers.primal_dual_interior_point.residuals import compute_kkt_residual
from purin.solvers.primal_dual_interior_point.residuals import residual_infinity_norm
from purin.solvers.primal_dual_interior_point.types import (
    BATCH_STATUS_ACTIVE as BATCH_STATUS_ACTIVE,
)
from purin.solvers.primal_dual_interior_point.types import (
    BATCH_STATUS_CONVERGED as BATCH_STATUS_CONVERGED,
)
from purin.solvers.primal_dual_interior_point.types import (
    BATCH_STATUS_MAXIMUM_ITERATIONS as BATCH_STATUS_MAXIMUM_ITERATIONS,
)
from purin.solvers.primal_dual_interior_point.types import (
    BATCH_STATUS_NAMES as BATCH_STATUS_NAMES,
)
from purin.solvers.primal_dual_interior_point.types import (
    BATCH_STATUS_NUMERICAL_FAILURE as BATCH_STATUS_NUMERICAL_FAILURE,
)
from purin.solvers.primal_dual_interior_point.types import (
    BATCH_STATUS_STALLED as BATCH_STATUS_STALLED,
)
from purin.solvers.primal_dual_interior_point.types import BoundIndices
from purin.solvers.primal_dual_interior_point.types import BoundValues
from purin.solvers.primal_dual_interior_point.types import NewtonRegularization
from purin.solvers.primal_dual_interior_point.types import NLPQuantities
from purin.solvers.primal_dual_interior_point.types import PrimalDualState
from purin.solvers.primal_dual_interior_point.types import SparseNumericData
from purin.solvers.primal_dual_interior_point.types import SparseProjectionAdjointResult
from purin.solvers.primal_dual_interior_point.types import SparseProjectionLinearization
from purin.solvers.primal_dual_interior_point.types import (
    SparseProjectionLossGradientResult,
)
from purin.solvers.primal_dual_interior_point.types import SparseProjectionSolveResult
from purin.solvers.primal_dual_interior_point.types import (
    PrimalDualState as JaxPrimalDualState,
)


class BatchedFixedMuResult(NamedTuple):
    """JAX result for a batch of fixed-barrier solves."""

    state: PrimalDualState
    status_codes: jax.Array
    iterations: jax.Array
    final_residual_inf_norm: jax.Array
    residual_history: jax.Array

    @property
    def converged(self):
        """Return per-instance convergence flags for the fixed-barrier solve."""
        return self.status_codes == BATCH_STATUS_CONVERGED


BatchNLPEvaluator = Callable[[PrimalDualState, jax.Array], NLPQuantities]


def _is_jax_array(value) -> bool:
    return isinstance(value, jax.Array)


def _tree_batch_where(mask, new_tree, old_tree):
    """Select complete batch rows from two identically shaped pytrees."""

    def select(new, old):
        expanded = mask.reshape((mask.shape[0],) + (1,) * (new.ndim - 1))
        return jnp.where(expanded, new, old)

    return jax.tree_util.tree_map(select, new_tree, old_tree)


def _state_rows_are_finite(state: PrimalDualState):
    finite = jnp.ones((state.x.shape[0],), dtype=bool)
    for block in state:
        axes = tuple(range(1, block.ndim))
        block_finite = (
            jnp.all(jnp.isfinite(block), axis=axes) if axes else jnp.isfinite(block)
        )
        finite = finite & block_finite
    return finite


def run_batched_fixed_mu_iterations(
    initial_state: PrimalDualState,
    parameter_values,
    evaluate_nlp: BatchNLPEvaluator,
    indices: BoundIndices,
    bounds: BoundValues,
    mu,
    *,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-9,
    tau: float = 0.995,
    step_scale: float = 1.0,
    minimum_step: float = 1e-14,
    regularization: NewtonRegularization | None = None,
    check_newton_residual: bool = False,
    jit: bool = True,
) -> BatchedFixedMuResult:
    """Solve a same-structure batch without looping over problems in Python.

    Parameters have shape ``(batch, n_parameter)`` and every state block has a
    leading batch dimension.  The direct fraction-to-boundary update is used;
    discrete filter-line-search state is intentionally not part of this pure
    batched kernel.
    """

    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")
    residual_tolerance = float(residual_tolerance)
    tau = float(tau)
    step_scale = float(step_scale)
    minimum_step = float(minimum_step)
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must satisfy 0 < tau < 1.")
    if not 0.0 < step_scale <= 1.0:
        raise ValueError("step_scale must satisfy 0 < step_scale <= 1.")
    if minimum_step < 0.0:
        raise ValueError("minimum_step must be nonnegative.")

    if _is_jax_array(parameter_values):
        parameters = jnp.asarray(parameter_values, dtype=initial_state.x.dtype)
    else:
        parameter_host = np.asarray(parameter_values, dtype=np.float64)
        if not np.all(np.isfinite(parameter_host)):
            raise ValueError("parameter_values must be finite.")
        parameters = jnp.asarray(parameter_host, dtype=initial_state.x.dtype)
    if parameters.ndim != 2:
        raise ValueError(
            f"parameter_values must have shape (batch, n_parameter); received {parameters.shape}."
        )
    batch_size = parameters.shape[0]
    for block_name, block in strict_zip(
        ("x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U"), initial_state
    ):
        if block.ndim != 2 or block.shape[0] != batch_size:
            raise ValueError(
                f"Initial-state block {block_name} must have shape (batch, width); "
                f"received {block.shape}."
            )

    if _is_jax_array(mu):
        mu_values = jnp.asarray(mu, dtype=initial_state.x.dtype)
    else:
        mu_host = np.asarray(mu, dtype=np.float64)
        if not np.all(np.isfinite(mu_host)) or np.any(mu_host < 0.0):
            raise ValueError("mu must contain finite nonnegative values.")
        mu_values = jnp.asarray(mu_host, dtype=initial_state.x.dtype)
    if mu_values.ndim == 0:
        mu_values = jnp.broadcast_to(mu_values, (batch_size,))
    elif mu_values.shape != (batch_size,):
        raise ValueError(
            f"mu must be scalar or have shape ({batch_size},); got {mu_values.shape}."
        )

    reg = regularization or NewtonRegularization()

    evaluate_batch = jax.vmap(evaluate_nlp, in_axes=(0, 0))
    solve_batch = jax.vmap(
        lambda state, nlp, mu_i: solve_primal_dual_newton_system(
            state,
            nlp,
            indices,
            bounds,
            mu_i,
            regularization=reg,
            check_full_residual=check_newton_residual,
        ),
        in_axes=(0, 0, 0),
    )
    step_limit_batch = jax.vmap(
        lambda state, direction: compute_fraction_to_boundary_steps(
            state, direction, indices, bounds, tau
        ),
        in_axes=(0, 0),
    )
    update_batch = jax.vmap(
        lambda state, direction, alpha_primal, alpha_dual: apply_primal_dual_step(
            state, direction, alpha_primal=alpha_primal, alpha_dual=alpha_dual
        ),
        in_axes=(0, 0, 0, 0),
    )
    slack_batch = jax.vmap(
        lambda state: compute_bound_slacks(state, indices, bounds), in_axes=0
    )
    residual_batch = jax.vmap(
        lambda state, nlp, slacks, mu_i: compute_kkt_residual(
            state, nlp, indices, slacks, mu_i
        ),
        in_axes=(0, 0, 0, 0),
    )
    residual_norm_batch = jax.vmap(residual_infinity_norm, in_axes=0)

    def solve_all(state0, params, mus):
        status0 = jnp.full((batch_size,), BATCH_STATUS_ACTIVE, dtype=jnp.int32)
        iteration0 = jnp.full((batch_size,), max_iterations, dtype=jnp.int32)

        def iteration_step(carry, iteration):
            state, status, stopping_iterations = carry
            nlp = evaluate_batch(state, params)
            slacks = slack_batch(state)
            residual = residual_batch(state, nlp, slacks, mus)
            residual_norm = residual_norm_batch(residual)

            active = status == BATCH_STATUS_ACTIVE
            residual_finite = jnp.isfinite(residual_norm) & _state_rows_are_finite(
                state
            )
            residual_failure = active & ~residual_finite
            status = jnp.where(residual_failure, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(
                residual_failure, iteration, stopping_iterations
            )

            active = status == BATCH_STATUS_ACTIVE
            converged = active & (residual_norm <= residual_tolerance)
            status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
            stopping_iterations = jnp.where(converged, iteration, stopping_iterations)

            newton = solve_batch(state, nlp, mus)
            active = status == BATCH_STATUS_ACTIVE
            direction_finite = _state_rows_are_finite(newton.direction)
            numerical_failure = active & ~direction_finite
            status = jnp.where(
                numerical_failure, BATCH_STATUS_NUMERICAL_FAILURE, status
            )
            stopping_iterations = jnp.where(
                numerical_failure, iteration, stopping_iterations
            )

            active = status == BATCH_STATUS_ACTIVE
            limits = step_limit_batch(state, newton.direction)
            alpha_primal = step_scale * limits.alpha_primal
            alpha_dual = step_scale * limits.alpha_dual
            too_small = jnp.minimum(alpha_primal, alpha_dual) <= minimum_step
            stalled = active & too_small
            status = jnp.where(stalled, BATCH_STATUS_STALLED, status)
            stopping_iterations = jnp.where(stalled, iteration, stopping_iterations)

            should_update = status == BATCH_STATUS_ACTIVE
            candidate = update_batch(state, newton.direction, alpha_primal, alpha_dual)
            candidate_finite = _state_rows_are_finite(candidate)
            update_failure = should_update & ~candidate_finite
            status = jnp.where(update_failure, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(
                update_failure, iteration, stopping_iterations
            )
            should_update = status == BATCH_STATUS_ACTIVE
            state = _tree_batch_where(should_update, candidate, state)
            return (state, status, stopping_iterations), residual_norm

        iteration_indices = jnp.arange(max_iterations, dtype=jnp.int32)
        (state, status, stopping_iterations), history = jax.lax.scan(
            iteration_step, (state0, status0, iteration0), iteration_indices
        )

        final_nlp = evaluate_batch(state, params)
        final_slacks = slack_batch(state)
        final_residual = residual_batch(state, final_nlp, final_slacks, mus)
        final_norm = residual_norm_batch(final_residual)

        active = status == BATCH_STATUS_ACTIVE
        final_finite = jnp.isfinite(final_norm) & _state_rows_are_finite(state)
        status = jnp.where(
            active & ~final_finite, BATCH_STATUS_NUMERICAL_FAILURE, status
        )
        active = status == BATCH_STATUS_ACTIVE
        final_converged = active & (final_norm <= residual_tolerance)
        status = jnp.where(final_converged, BATCH_STATUS_CONVERGED, status)
        status = jnp.where(
            status == BATCH_STATUS_ACTIVE, BATCH_STATUS_MAXIMUM_ITERATIONS, status
        )
        stopping_iterations = jnp.where(
            final_converged, max_iterations, stopping_iterations
        )
        full_history = jnp.concatenate((history, final_norm[None, :]), axis=0)
        return BatchedFixedMuResult(
            state=state,
            status_codes=status,
            iterations=stopping_iterations,
            final_residual_inf_norm=final_norm,
            residual_history=full_history,
        )

    runner = jax.jit(solve_all) if jit else solve_all
    return runner(initial_state, parameters, mu_values)


VectorFunction = Callable[[jax.Array, jax.Array], jax.Array]
SparseValuesFunction = Callable[[jax.Array, jax.Array], jax.Array]
ConstraintHVPFunction = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]


def run_shared_linear_projection_batch(
    initial_state: PrimalDualState,
    parameter_values: jax.Array,
    equality_function: VectorFunction,
    equality_jacobian: jax.Array,
    target_parameter_indices: jax.Array,
    *,
    residual_tolerance: float = 1e-9,
    jit: bool = True,
) -> BatchedFixedMuResult:
    """Project a batch onto one shared affine equality system.

    The common KKT matrix is solved once with a matrix right-hand side.  This
    mode intentionally supports equality-only projection problems.
    """
    parameters = jnp.asarray(parameter_values)
    batch_size = parameters.shape[0]
    target_indices = jnp.asarray(target_parameter_indices, dtype=jnp.int32)
    matrix_a = jnp.asarray(equality_jacobian, dtype=initial_state.x.dtype)
    n_x = initial_state.x.shape[1]
    n_c = matrix_a.shape[0]

    def solve_all(params):
        zero_x = jnp.zeros((n_x,), dtype=initial_state.x.dtype)
        offsets = jax.vmap(lambda p: equality_function(zero_x, p))(params)
        targets = params[:, target_indices]
        identity = jnp.eye(n_x, dtype=initial_state.x.dtype)
        zero_cc = jnp.zeros((n_c, n_c), dtype=initial_state.x.dtype)
        kkt = jnp.block([[identity, matrix_a.T], [matrix_a, zero_cc]])
        rhs = jnp.concatenate((targets, -offsets), axis=1)
        solution = jnp.linalg.solve(kkt, rhs.T).T
        x = solution[:, :n_x]
        y_c = solution[:, n_x:]

        c = jax.vmap(equality_function)(x, params)
        stationarity = x - targets + y_c @ matrix_a
        residual_norm = jnp.maximum(
            jnp.max(jnp.abs(stationarity), axis=1, initial=0.0),
            jnp.max(jnp.abs(c), axis=1, initial=0.0),
        )
        finite = (
            jnp.all(jnp.isfinite(x), axis=1)
            & jnp.all(jnp.isfinite(y_c), axis=1)
            & jnp.isfinite(residual_norm)
        )
        converged = finite & (residual_norm <= residual_tolerance)
        status = jnp.where(
            converged, BATCH_STATUS_CONVERGED, BATCH_STATUS_NUMERICAL_FAILURE
        ).astype(jnp.int32)

        empty = jnp.empty((batch_size, 0), dtype=initial_state.x.dtype)
        state = PrimalDualState(
            x=x, s=empty, y_c=y_c, y_d=empty, z_L=empty, z_U=empty, v_L=empty, v_U=empty
        )

        initial_c = jax.vmap(equality_function)(initial_state.x, params)
        initial_stationarity = initial_state.x - targets + initial_state.y_c @ matrix_a
        initial_norm = jnp.maximum(
            jnp.max(jnp.abs(initial_stationarity), axis=1, initial=0.0),
            jnp.max(jnp.abs(initial_c), axis=1, initial=0.0),
        )
        return BatchedFixedMuResult(
            state=state,
            status_codes=status,
            iterations=jnp.where(converged, 1, 0).astype(jnp.int32),
            final_residual_inf_norm=residual_norm,
            residual_history=jnp.stack((initial_norm, residual_norm), axis=0),
        )

    runner = jax.jit(solve_all) if jit else solve_all
    return runner(parameters)


VectorFunction = Callable[[jax.Array, jax.Array], jax.Array]
LocalDerivativeFunction = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


class PreparedSparseCondensedBatchSolver:
    """Persistent fixed-shape sparse solver with reusable JIT kernels.

    The symbolic functions, sparse supports, solver options, and batch shape are
    captured once.  The first :meth:`solve` call compiles the staged JAX kernels;
    subsequent calls with the same shapes execute the cached executables.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        equality_function: VectorFunction,
        inequality_function: VectorFunction,
        equality_local_derivatives: LocalDerivativeFunction,
        inequality_local_derivatives: LocalDerivativeFunction,
        equality_affine_offsets: VectorFunction | None = None,
        inequality_affine_offsets: VectorFunction | None = None,
        equality_constant_local_jacobian: jax.Array | None = None,
        inequality_constant_local_jacobian: jax.Array | None = None,
        equality_support: jax.Array,
        equality_mask: jax.Array,
        inequality_support: jax.Array,
        inequality_mask: jax.Array,
        target_parameter_indices: jax.Array,
        indices: BoundIndices,
        bounds: BoundValues,
        max_iterations: int = 50,
        residual_tolerance: float = 1e-9,
        tau: float = 0.995,
        step_scale: float = 1.0,
        minimum_step: float = 1e-14,
        regularization: NewtonRegularization | None = None,
        gmres_tolerance: float = 1e-9,
        gmres_atol: float = 0.0,
        gmres_restart: int = 30,
        gmres_maxiter: int = 50,
        use_preconditioner: bool = True,
        backtracking: bool = False,
        max_backtracking_steps: int = 8,
        backtracking_factor: float = 0.5,
        constraints_affine: bool = False,
        use_column_gather_transpose: bool | None = None,
        use_column_bucketed_transpose: bool = False,
        use_shared_constant_jacobian: bool = False,
        gmres_unroll: int = 1,
        arnoldi_prefix_block: int = 0,
        gpu_kkt_mode: str = "xla",
        pallas_kkt_block_size: int = 64,
        pallas_interpret: bool = False,
        store_residual_history: bool = True,
        adjoint_gmres_tolerance: float | None = None,
        adjoint_gmres_atol: float | None = None,
        adjoint_gmres_restart: int | None = None,
        adjoint_gmres_maxiter: int | None = None,
        adjoint_gmres_unroll: int | None = None,
        adjoint_arnoldi_prefix_block: int | None = None,
        adjoint_solver_mode: str = "gmres",
    ) -> None:
        """Prepare fixed sparse patterns, Newton kernels and implicit solve resources."""
        _validate_sparse_iteration_options(
            batch_size,
            gmres_maxiter,
            gmres_restart,
            gmres_tolerance,
            max_iterations,
            residual_tolerance,
            step_scale,
            tau,
        )
        _validate_sparse_execution_options(
            adjoint_solver_mode,
            arnoldi_prefix_block,
            backtracking_factor,
            gmres_unroll,
            gpu_kkt_mode,
            max_backtracking_steps,
            pallas_kkt_block_size,
        )
        (
            adjoint_arnoldi_prefix_block,
            adjoint_gmres_atol,
            adjoint_gmres_maxiter,
            adjoint_gmres_restart,
            adjoint_gmres_tolerance,
            adjoint_gmres_unroll,
        ) = _resolve_sparse_adjoint_options(
            adjoint_arnoldi_prefix_block,
            adjoint_gmres_atol,
            adjoint_gmres_maxiter,
            adjoint_gmres_restart,
            adjoint_gmres_tolerance,
            adjoint_gmres_unroll,
            arnoldi_prefix_block,
            gmres_atol,
            gmres_maxiter,
            gmres_restart,
            gmres_tolerance,
            gmres_unroll,
        )

        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.residual_tolerance = float(residual_tolerance)
        self.gmres_unroll = int(gmres_unroll)
        self.arnoldi_prefix_block = int(arnoldi_prefix_block)
        self.gpu_kkt_mode = str(gpu_kkt_mode)
        self.pallas_kkt_block_size = int(pallas_kkt_block_size)
        self.store_residual_history = bool(store_residual_history)
        self.adjoint_gmres_tolerance = float(adjoint_gmres_tolerance)
        self.adjoint_gmres_atol = float(adjoint_gmres_atol)
        self.adjoint_gmres_restart = int(adjoint_gmres_restart)
        self.adjoint_gmres_maxiter = int(adjoint_gmres_maxiter)
        self.adjoint_gmres_unroll = int(adjoint_gmres_unroll)
        self.adjoint_arnoldi_prefix_block = int(adjoint_arnoldi_prefix_block)
        self.adjoint_solver_mode = str(adjoint_solver_mode)
        self.solve_calls = 0

        (
            combined_full_support,
            combined_mask_np,
            combined_support_np,
            equality_full_support,
            equality_mask_np,
            equality_support_np,
            inequality_full_support,
            inequality_mask_np,
            inequality_support_np,
            n_variables,
            reg,
        ) = _prepare_sparse_support_layout(
            equality_mask,
            equality_support,
            inequality_mask,
            inequality_support,
            regularization,
            target_parameter_indices,
        )
        # Preserve the existing low-level PDIP behavior exactly: direct GPU
        # construction defaults to global gather unless the high-level problem
        # planner has already selected a specific mode.  The shared plan only
        # centralizes sparse metadata construction; it does not alter policy.
        (
            transpose_plan,
            use_column_bucketed_transpose,
            use_column_gather_transpose,
            use_shared_constant_jacobian,
        ) = _prepare_sparse_transpose(
            combined_mask_np,
            combined_support_np,
            n_variables,
            use_column_bucketed_transpose,
            use_column_gather_transpose,
            use_shared_constant_jacobian,
        )

        equality_support = jnp.asarray(equality_support_np, dtype=jnp.int32)
        inequality_support = jnp.asarray(inequality_support_np, dtype=jnp.int32)
        equality_mask = jnp.asarray(equality_mask_np, dtype=bool)
        inequality_mask = jnp.asarray(inequality_mask_np, dtype=bool)
        combined_support = transpose_plan.support
        combined_mask = transpose_plan.mask
        column_rows = transpose_plan.column_rows
        column_slots = transpose_plan.column_slots
        column_mask = transpose_plan.column_mask
        column_buckets = transpose_plan.column_buckets
        target_parameter_indices = jnp.asarray(
            target_parameter_indices, dtype=jnp.int32
        )

        constraints_affine = bool(constraints_affine)

        (
            combined_constant_local_jacobian,
            combined_constant_local_jacobian_squared,
            equality_constant_local_jacobian,
            inequality_constant_local_jacobian,
        ) = _prepare_constant_jacobians(
            equality_constant_local_jacobian, inequality_constant_local_jacobian
        )

        pallas_kkt_matvec = _prepare_pallas_product(
            column_mask,
            column_rows,
            column_slots,
            combined_constant_local_jacobian,
            combined_mask,
            combined_support,
            constraints_affine,
            gpu_kkt_mode,
            n_variables,
            pallas_interpret,
            pallas_kkt_block_size,
            use_shared_constant_jacobian,
        )

        (
            affine_offsets_batch,
            constraint_values_batch,
            derivative_batch,
            evaluate_dynamic_batch,
            evaluate_fixed_batch,
        ) = _prepare_sparse_evaluators(
            bounds,
            equality_affine_offsets,
            equality_constant_local_jacobian,
            equality_full_support,
            equality_function,
            equality_local_derivatives,
            equality_mask,
            equality_support,
            indices,
            inequality_affine_offsets,
            inequality_constant_local_jacobian,
            inequality_full_support,
            inequality_function,
            inequality_local_derivatives,
            inequality_mask,
            inequality_support,
            target_parameter_indices,
            use_shared_constant_jacobian,
        )
        residual_norm_batch = jax.vmap(_residual_inf_norm)

        def evaluate_numeric(state, params, mus, fixed_derivatives):
            if constraints_affine:
                eq_j, ineq_j, eq_offset, ineq_offset = fixed_derivatives
                residual, slacks, sigma_x, sigma_s = evaluate_fixed_batch(
                    state, params, mus, eq_j, ineq_j, eq_offset, ineq_offset
                )
                empty_hessian = jnp.empty((0, 0, 0), dtype=state.x.dtype)
                return SparseNumericData(
                    residual=residual,
                    slacks=slacks,
                    sigma_x=sigma_x,
                    sigma_s=sigma_s,
                    equality_jacobian=eq_j,
                    inequality_jacobian=ineq_j,
                    equality_hessian=empty_hessian,
                    inequality_hessian=empty_hessian,
                )
            return evaluate_dynamic_batch(state, params, mus)

        def build_fixed_derivatives(reference_state, parameters):
            """Evaluate affine derivative data once for a complete batch."""

            if not constraints_affine:
                return None
            if (
                equality_constant_local_jacobian is None
                or inequality_constant_local_jacobian is None
            ):
                eq_j_dynamic, _eq_h, ineq_j_dynamic, _ineq_h = derivative_batch(
                    reference_state, parameters
                )
                eq_j = (
                    eq_j_dynamic
                    if equality_constant_local_jacobian is None
                    else equality_constant_local_jacobian
                )
                ineq_j = (
                    ineq_j_dynamic
                    if inequality_constant_local_jacobian is None
                    else inequality_constant_local_jacobian
                )
            else:
                if use_shared_constant_jacobian:
                    eq_j = equality_constant_local_jacobian
                    ineq_j = inequality_constant_local_jacobian
                else:
                    eq_j = jnp.broadcast_to(
                        equality_constant_local_jacobian,
                        (batch_size,) + equality_constant_local_jacobian.shape,
                    )
                    ineq_j = jnp.broadcast_to(
                        inequality_constant_local_jacobian,
                        (batch_size,) + inequality_constant_local_jacobian.shape,
                    )

            if affine_offsets_batch is not None:
                equality_offset, inequality_offset = affine_offsets_batch(
                    reference_state, parameters
                )
            else:
                equality_values, inequality_values = constraint_values_batch(
                    reference_state, parameters
                )
                equality_offset = equality_values - _batched_local_jacobian_matvec(
                    eq_j,
                    equality_support,
                    equality_mask,
                    reference_state.x,
                    full_support=equality_full_support,
                )
                inequality_offset = inequality_values - _batched_local_jacobian_matvec(
                    ineq_j,
                    inequality_support,
                    inequality_mask,
                    reference_state.x,
                    full_support=inequality_full_support,
                )
            return eq_j, ineq_j, equality_offset, inequality_offset

        def solve_batch(state, numeric):
            return _solve_condensed_direction_batch(
                state,
                numeric,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                indices,
                reg,
                tolerance=gmres_tolerance,
                absolute_tolerance=gmres_atol,
                max_iterations=gmres_maxiter,
                restart=gmres_restart,
                use_preconditioner=use_preconditioner,
                constraints_affine=constraints_affine,
                equality_full_support=equality_full_support,
                inequality_full_support=inequality_full_support,
                combined_full_support=combined_full_support,
                combined_support=combined_support,
                combined_mask=combined_mask,
                column_rows=column_rows if use_column_gather_transpose else None,
                column_slots=column_slots if use_column_gather_transpose else None,
                column_mask=column_mask if use_column_gather_transpose else None,
                column_buckets=(
                    column_buckets if use_column_bucketed_transpose else None
                ),
                constant_combined_jacobian=(
                    combined_constant_local_jacobian
                    if use_shared_constant_jacobian
                    else None
                ),
                constant_combined_jacobian_squared=(
                    combined_constant_local_jacobian_squared
                    if use_shared_constant_jacobian
                    else None
                ),
                affine_kkt_matvec=pallas_kkt_matvec,
                gmres_unroll=gmres_unroll,
                arnoldi_prefix_block=arnoldi_prefix_block,
            )

        def build_final_linearization(state, numeric):
            return _build_sparse_projection_linearization(
                state,
                numeric,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                reg,
                use_preconditioner=use_preconditioner,
                constraints_affine=constraints_affine,
                equality_full_support=equality_full_support,
                inequality_full_support=inequality_full_support,
                combined_full_support=combined_full_support,
                combined_support=combined_support,
                combined_mask=combined_mask,
                constant_combined_jacobian=(
                    combined_constant_local_jacobian
                    if use_shared_constant_jacobian
                    else None
                ),
                constant_combined_jacobian_squared=(
                    combined_constant_local_jacobian_squared
                    if use_shared_constant_jacobian
                    else None
                ),
            )

        def solve_yhat_adjoint_batch(state, linearization, grad_x, initial_solution):
            return _solve_condensed_yhat_adjoint_batch(
                state,
                linearization,
                grad_x,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                reg,
                tolerance=adjoint_gmres_tolerance,
                absolute_tolerance=adjoint_gmres_atol,
                max_iterations=adjoint_gmres_maxiter,
                restart=adjoint_gmres_restart,
                use_preconditioner=use_preconditioner,
                constraints_affine=constraints_affine,
                equality_full_support=equality_full_support,
                inequality_full_support=inequality_full_support,
                combined_full_support=combined_full_support,
                combined_support=combined_support,
                combined_mask=combined_mask,
                column_rows=(column_rows if use_column_gather_transpose else None),
                column_slots=(column_slots if use_column_gather_transpose else None),
                column_mask=(column_mask if use_column_gather_transpose else None),
                column_buckets=(
                    column_buckets if use_column_bucketed_transpose else None
                ),
                affine_kkt_matvec=pallas_kkt_matvec,
                gmres_unroll=adjoint_gmres_unroll,
                arnoldi_prefix_block=adjoint_arnoldi_prefix_block,
                solver_mode=adjoint_solver_mode,
                initial_solution=initial_solution,
            )

        step_limit_batch = jax.vmap(
            lambda state, direction: compute_fraction_to_boundary_steps(
                state, direction, indices, bounds, tau
            ),
            in_axes=(0, 0),
        )
        update_batch = jax.vmap(
            lambda state, direction, alpha_primal, alpha_dual: apply_primal_dual_step(
                state, direction, alpha_primal=alpha_primal, alpha_dual=alpha_dual
            ),
            in_axes=(0, 0, 0, 0),
        )

        def select_backtracked_candidate(
            state: PrimalDualState,
            direction: PrimalDualState,
            alpha_primal: jax.Array,
            alpha_dual: jax.Array,
            params: jax.Array,
            mus: jax.Array,
            current_norm: jax.Array,
            fixed_derivatives,
        ) -> tuple[PrimalDualState, jax.Array]:
            if not backtracking:
                candidate = update_batch(state, direction, alpha_primal, alpha_dual)
                return candidate, jnp.ones((batch_size,), dtype=bool)

            accepted0 = jnp.zeros((batch_size,), dtype=bool)
            selected0 = state
            scale0 = jnp.ones((batch_size,), dtype=state.x.dtype)

            def body(_, carry):
                scale, accepted, selected = carry
                candidate = update_batch(
                    state, direction, alpha_primal * scale, alpha_dual * scale
                )
                trial_numeric = evaluate_numeric(
                    candidate, params, mus, fixed_derivatives
                )
                trial_norm = residual_norm_batch(trial_numeric.residual)
                accept_now = (
                    ~accepted
                    & _state_rows_are_finite(candidate)
                    & jnp.isfinite(trial_norm)
                    & (trial_norm <= current_norm)
                )
                selected = _tree_batch_where(accept_now, candidate, selected)
                accepted = accepted | accept_now
                scale = jnp.where(accepted, scale, scale * backtracking_factor)
                return scale, accepted, selected

            _, accepted, selected = jax.lax.fori_loop(
                0, max_backtracking_steps, body, (scale0, accepted0, selected0)
            )
            return selected, accepted

        def precheck_status(
            state, status, stopping_iterations, residual_norm, iteration
        ):
            active = status == BATCH_STATUS_ACTIVE
            finite = jnp.isfinite(residual_norm) & _state_rows_are_finite(state)
            failed = active & ~finite
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(failed, iteration, stopping_iterations)

            active = status == BATCH_STATUS_ACTIVE
            converged = active & (residual_norm <= residual_tolerance)
            status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
            stopping_iterations = jnp.where(converged, iteration, stopping_iterations)
            return status, stopping_iterations

        def advance_after_direction(
            state,
            status,
            stopping_iterations,
            direction,
            linear_residual,
            residual_norm,
            iteration,
            params,
            mus,
            fixed_derivatives,
        ):
            forcing = jnp.minimum(0.1, jnp.sqrt(jnp.maximum(residual_norm, 0.0)))
            allowed_linear_residual = jnp.maximum(
                1e-10, forcing * jnp.maximum(residual_norm, 1e-16)
            )
            active = status == BATCH_STATUS_ACTIVE
            direction_ok = (
                _state_rows_are_finite(direction)
                & jnp.isfinite(linear_residual)
                & (linear_residual <= allowed_linear_residual)
            )
            failed = active & ~direction_ok
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(failed, iteration, stopping_iterations)

            active = status == BATCH_STATUS_ACTIVE
            limits = step_limit_batch(state, direction)
            alpha_primal = step_scale * limits.alpha_primal
            alpha_dual = step_scale * limits.alpha_dual
            stalled = active & (jnp.minimum(alpha_primal, alpha_dual) <= minimum_step)
            status = jnp.where(stalled, BATCH_STATUS_STALLED, status)
            stopping_iterations = jnp.where(stalled, iteration, stopping_iterations)

            candidate, accepted = select_backtracked_candidate(
                state,
                direction,
                alpha_primal,
                alpha_dual,
                params,
                mus,
                residual_norm,
                fixed_derivatives,
            )
            should_update = status == BATCH_STATUS_ACTIVE
            failed = should_update & (~accepted | ~_state_rows_are_finite(candidate))
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(failed, iteration, stopping_iterations)
            should_update = status == BATCH_STATUS_ACTIVE
            state = _tree_batch_where(should_update, candidate, state)
            return state, status, stopping_iterations

        def solve_all_core(initial_state, parameters, mu_values):
            fixed_derivatives = build_fixed_derivatives(initial_state, parameters)

            status0 = jnp.full((batch_size,), BATCH_STATUS_ACTIVE, dtype=jnp.int32)
            stopping0 = jnp.full((batch_size,), max_iterations, dtype=jnp.int32)
            history0 = jnp.zeros(
                (max_iterations + 1, batch_size)
                if store_residual_history
                else (0, batch_size),
                dtype=initial_state.x.dtype,
            )
            iteration0 = jnp.asarray(0, dtype=jnp.int32)

            zero_direction = jax.tree_util.tree_map(jnp.zeros_like, initial_state)
            zero_info = jnp.zeros((batch_size,), dtype=jnp.int32)
            zero_linear_residual = jnp.zeros((batch_size,), dtype=initial_state.x.dtype)

            def loop_condition(carry):
                iteration, _state, status, _stopping, _history = carry
                return (iteration < max_iterations) & jnp.any(
                    status == BATCH_STATUS_ACTIVE
                )

            def loop_body(carry):
                iteration, state, status, stopping_iterations, history = carry
                numeric = evaluate_numeric(
                    state, parameters, mu_values, fixed_derivatives
                )
                residual_norm = residual_norm_batch(numeric.residual)
                if store_residual_history:
                    history = history.at[iteration].set(residual_norm)
                status, stopping_iterations = precheck_status(
                    state, status, stopping_iterations, residual_norm, iteration
                )

                def perform_solve(_):
                    return solve_batch(state, numeric)

                def skip_solve(_):
                    return zero_direction, zero_info, zero_linear_residual

                direction, _info, linear_residual = jax.lax.cond(
                    jnp.any(status == BATCH_STATUS_ACTIVE),
                    perform_solve,
                    skip_solve,
                    operand=None,
                )
                state, status, stopping_iterations = advance_after_direction(
                    state,
                    status,
                    stopping_iterations,
                    direction,
                    linear_residual,
                    residual_norm,
                    iteration,
                    parameters,
                    mu_values,
                    fixed_derivatives,
                )
                return (iteration + 1, state, status, stopping_iterations, history)

            iteration, state, status, stopping_iterations, history = jax.lax.while_loop(
                loop_condition,
                loop_body,
                (iteration0, initial_state, status0, stopping0, history0),
            )

            final_numeric = evaluate_numeric(
                state, parameters, mu_values, fixed_derivatives
            )
            final_norm = residual_norm_batch(final_numeric.residual)
            active = status == BATCH_STATUS_ACTIVE
            finite = jnp.isfinite(final_norm) & _state_rows_are_finite(state)
            status = jnp.where(active & ~finite, BATCH_STATUS_NUMERICAL_FAILURE, status)
            active = status == BATCH_STATUS_ACTIVE
            converged = active & (final_norm <= residual_tolerance)
            status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
            status = jnp.where(
                status == BATCH_STATUS_ACTIVE, BATCH_STATUS_MAXIMUM_ITERATIONS, status
            )
            stopping_iterations = jnp.where(
                converged, max_iterations, stopping_iterations
            )

            # The previous fixed-length implementation evaluated the frozen
            # converged state on every remaining iteration.  Filling the tail
            # with the final norm preserves the public history shape and values
            # while avoiding those expensive evaluations and linear solves.
            if store_residual_history:
                tail = jnp.arange(max_iterations + 1, dtype=jnp.int32)[:, None]
                history = jnp.where(tail >= iteration, final_norm[None, :], history)
            return (
                state,
                status,
                stopping_iterations,
                final_norm,
                history,
                final_numeric,
            )

        def solve_all(initial_state, parameters, mu_values):
            state, status, iterations, final_norm, history, _numeric = solve_all_core(
                initial_state, parameters, mu_values
            )
            return BatchedFixedMuResult(
                state=state,
                status_codes=status,
                iterations=iterations,
                final_residual_inf_norm=final_norm,
                residual_history=history,
            )

        def solve_all_with_linearization(initial_state, parameters, mu_values):
            state, status, iterations, final_norm, history, final_numeric = (
                solve_all_core(initial_state, parameters, mu_values)
            )
            return SparseProjectionSolveResult(
                state=state,
                status_codes=status,
                iterations=iterations,
                final_residual_inf_norm=final_norm,
                residual_history=history,
                final_linearization=build_final_linearization(state, final_numeric),
            )

        def solve_yhat_adjoint(
            final_state, final_linearization, grad_x, initial_solution
        ):
            return solve_yhat_adjoint_batch(
                final_state, final_linearization, grad_x, initial_solution
            )

        def solve_yhat_adjoint_recompute(
            final_state, parameters, mu_values, grad_x, initial_solution
        ):
            fixed_derivatives = build_fixed_derivatives(final_state, parameters)
            final_numeric = evaluate_numeric(
                final_state, parameters, mu_values, fixed_derivatives
            )
            final_linearization = build_final_linearization(final_state, final_numeric)
            return solve_yhat_adjoint_batch(
                final_state, final_linearization, grad_x, initial_solution
            )

        def make_loss_gradient_function(q_mode):
            def solve_loss_gradient(
                initial_state,
                parameters,
                mu_values,
                q_data,
                valid_mask,
                initial_adjoint,
            ):
                forward = solve_all_with_linearization(
                    initial_state, parameters, mu_values
                )
                y_tilde = forward.state.x
                if q_mode == "dense":
                    q_y = y_tilde @ q_data.T
                else:
                    q_y = q_data[None, :] * y_tilde
                losses = jnp.sum(y_tilde * q_y, axis=1)
                grad_x = jnp.where(valid_mask[:, None], 2.0 * q_y, 0.0)
                adjoint = solve_yhat_adjoint_batch(
                    forward.state, forward.final_linearization, grad_x, initial_adjoint
                )
                return SparseProjectionLossGradientResult(
                    losses=losses,
                    y_tilde=y_tilde,
                    grad_y_hat=adjoint.x,
                    forward_status_codes=forward.status_codes,
                    forward_iterations=forward.iterations,
                    forward_residual_inf_norm=forward.final_residual_inf_norm,
                    adjoint_status_codes=adjoint.linear_status,
                    adjoint_iterations=adjoint.linear_iterations,
                    adjoint_residual_inf_norm=adjoint.linear_residual_inf_norm,
                )

            return solve_loss_gradient

        # One compiled executable owns the complete Newton loop.  The first
        # fixed-shape solve traces/compiles it; later batches perform one JAX
        # dispatch and stop as soon as no row remains active.
        self._solve_all_function = jax.jit(solve_all)
        self._solve_all_with_linearization_function = jax.jit(
            solve_all_with_linearization
        )
        self._solve_yhat_adjoint_function = jax.jit(solve_yhat_adjoint)
        self._solve_yhat_adjoint_recompute_function = jax.jit(
            solve_yhat_adjoint_recompute
        )
        self._solve_loss_gradient_diagonal_function = jax.jit(
            make_loss_gradient_function("diagonal")
        )
        self._solve_loss_gradient_dense_function = jax.jit(
            make_loss_gradient_function("dense")
        )

    def solve(
        self, *, initial_state: PrimalDualState, parameter_values: jax.Array, mu
    ) -> BatchedFixedMuResult:
        """Solve one batch with one reusable, dynamically terminating graph."""

        target_device = next(iter(initial_state.x.devices()))
        parameters = jax.device_put(
            jnp.asarray(parameter_values, dtype=initial_state.x.dtype), target_device
        )
        if parameters.ndim != 2:
            raise ValueError("parameter_values must be a rank-two array.")
        if parameters.shape[0] != self.batch_size:
            raise ValueError(
                "Prepared solver batch-size mismatch: expected "
                f"{self.batch_size}, received {parameters.shape[0]}."
            )

        if isinstance(mu, jax.Array):
            mu_values = jax.device_put(
                jnp.asarray(mu, dtype=initial_state.x.dtype), target_device
            )
        else:
            mu_host = np.asarray(mu, dtype=np.float64)
            if not np.all(np.isfinite(mu_host)) or np.any(mu_host < 0.0):
                raise ValueError("mu must contain finite nonnegative values.")
            mu_values = jax.device_put(
                jnp.asarray(mu_host, dtype=initial_state.x.dtype), target_device
            )
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        if mu_values.shape != (self.batch_size,):
            raise ValueError(f"mu must be scalar or have shape ({self.batch_size},).")

        result = self._solve_all_function(initial_state, parameters, mu_values)
        self.solve_calls += 1
        return result

    def solve_with_linearization(
        self, *, initial_state: PrimalDualState, parameter_values: jax.Array, mu
    ) -> SparseProjectionSolveResult:
        """Solve and retain the final KKT linearization for an adjoint call."""

        target_device = next(iter(initial_state.x.devices()))
        parameters = jax.device_put(
            jnp.asarray(parameter_values, dtype=initial_state.x.dtype), target_device
        )
        mu_values = jax.device_put(
            jnp.asarray(mu, dtype=initial_state.x.dtype), target_device
        )
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        result = self._solve_all_with_linearization_function(
            initial_state, parameters, mu_values
        )
        self.solve_calls += 1
        return result

    def solve_yhat_adjoint(
        self,
        *,
        final_state: PrimalDualState,
        grad_x: jax.Array,
        final_linearization: SparseProjectionLinearization | None = None,
        parameter_values: jax.Array | None = None,
        mu=None,
        initial_solution: jax.Array | None = None,
    ) -> SparseProjectionAdjointResult:
        """Solve the sparse final-KKT adjoint for a projection loss.

        ``grad_x`` is the per-sample loss cotangent with respect to the final
        projected primal output.  The returned ``x`` block is the gradient with
        respect to the distance-minimization target ``y_hat``.
        """

        target_device = next(iter(final_state.x.devices()))
        gradient = jax.device_put(
            jnp.asarray(grad_x, dtype=final_state.x.dtype), target_device
        )
        expected_gradient_shape = (self.batch_size, final_state.x.shape[1])
        if tuple(gradient.shape) != expected_gradient_shape:
            raise ValueError(
                f"grad_x must have shape {expected_gradient_shape}; "
                f"received {tuple(gradient.shape)}."
            )
        width = (
            final_state.x.shape[1] + final_state.y_c.shape[1] + final_state.y_d.shape[1]
        )
        if initial_solution is None:
            initial = jnp.zeros((self.batch_size, width), dtype=final_state.x.dtype)
        else:
            initial = jax.device_put(
                jnp.asarray(initial_solution, dtype=final_state.x.dtype), target_device
            )
            if tuple(initial.shape) != (self.batch_size, width):
                raise ValueError(
                    f"initial_solution must have shape {(self.batch_size, width)}; "
                    f"received {tuple(initial.shape)}."
                )
        if final_linearization is not None:
            return self._solve_yhat_adjoint_function(
                final_state, final_linearization, gradient, initial
            )
        if parameter_values is None:
            raise ValueError(
                "parameter_values are required when final_linearization is omitted."
            )
        parameters = jax.device_put(
            jnp.asarray(parameter_values, dtype=final_state.x.dtype), target_device
        )
        if parameters.ndim != 2 or parameters.shape[0] != self.batch_size:
            raise ValueError(
                f"parameter_values must have shape ({self.batch_size}, parameter_count)."
            )
        mu_values = jax.device_put(
            jnp.asarray(mu, dtype=final_state.x.dtype), target_device
        )
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        return self._solve_yhat_adjoint_recompute_function(
            final_state, parameters, mu_values, gradient, initial
        )

    def solve_loss_yhat_gradient(
        self,
        *,
        initial_state: PrimalDualState,
        parameter_values: jax.Array,
        mu_values: jax.Array,
        q_data: jax.Array,
        valid_mask: jax.Array,
        q_mode: str,
        initial_adjoint: jax.Array | None = None,
    ) -> SparseProjectionLossGradientResult:
        """Run forward projection, quadratic loss, and sparse adjoint together."""

        if q_mode not in {"identity", "diagonal", "dense"}:
            raise ValueError("q_mode must be identity, diagonal, or dense.")
        width = (
            initial_state.x.shape[1]
            + initial_state.y_c.shape[1]
            + initial_state.y_d.shape[1]
        )
        if initial_adjoint is None:
            initial_adjoint = jnp.zeros(
                (self.batch_size, width), dtype=initial_state.x.dtype
            )
        function = (
            self._solve_loss_gradient_dense_function
            if q_mode == "dense"
            else self._solve_loss_gradient_diagonal_function
        )
        return function(
            initial_state,
            parameter_values,
            mu_values,
            q_data,
            valid_mask,
            initial_adjoint,
        )


def prepare_batched_projection_condensed_sparse_solver(
    **kwargs,
) -> PreparedSparseCondensedBatchSolver:
    """Create a persistent fixed-shape sparse solver."""

    return PreparedSparseCondensedBatchSolver(**kwargs)


# ---------------------------------------------------------------------------
# Compact batched Krylov kernels
# ---------------------------------------------------------------------------


@dataclass
class PreparedSparseProjectionBatchSolver:
    """Persistent device-resident sparse projection executor.

    The symbolic structure, initialization graph, sparse solver executable and
    static bound data are prepared once for a fixed batch shape.  Calls to
    :meth:`solve` accept NumPy or JAX arrays and return JAX arrays on the
    selected device without forcing a host synchronization.
    """

    compiled: Any
    sparse_solver: Any
    device: Any
    interior_margin: float
    target_parameter_indices: jax.Array
    physical_parameter_indices: jax.Array
    batch_size: int
    parameter_count: int
    transpose_mode: str = "scatter"
    gmres_unroll: int = 1
    arnoldi_prefix_block: int = 0
    gpu_kkt_mode: str = "xla"
    pallas_kkt_block_size: int = 64
    store_residual_history: bool = True
    adjoint_solver_mode: str = "gmres"
    adjoint_gmres_restart: int = 30
    adjoint_gmres_maxiter: int = 50
    adjoint_gmres_unroll: int = 1
    adjoint_arnoldi_prefix_block: int = 0
    solve_default_function: Any | None = None
    solve_default_with_linearization_function: Any | None = None
    solve_default_loss_gradient_functions: dict[str, Any] | None = None

    @property
    def backend(self) -> str:
        """Return the backend associated with this prepared batch solver."""
        return "cuda" if self.device.platform == "gpu" else self.device.platform

    def put_parameters(self, parameters: Any) -> jax.Array:
        """Place parameter arrays on the prepared solver's device."""
        array = jnp.asarray(parameters, dtype=jnp.float64)
        if array.ndim != 2:
            raise ValueError("parameters must be a rank-two array.")
        expected = (self.batch_size, self.parameter_count)
        if tuple(array.shape) != expected:
            raise ValueError(
                f"Prepared solver expected parameter shape {expected}; "
                f"received {tuple(array.shape)}."
            )
        return jax.device_put(array, self.device)

    def put_mu(self, mu: float | Sequence[float] | jax.Array) -> jax.Array:
        """Place barrier values on the prepared solver's device."""
        mu_array = jax.device_put(jnp.asarray(mu, dtype=jnp.float64), self.device)
        if mu_array.ndim == 0:
            mu_array = jnp.broadcast_to(mu_array, (self.batch_size,))
        if tuple(mu_array.shape) != (self.batch_size,):
            raise ValueError(f"mu must be scalar or have shape ({self.batch_size},).")
        return mu_array

    def pack_projection_parameters_jax(
        self, problem_parameters: jax.Array, y_hat: jax.Array
    ) -> jax.Array:
        """Pack physical parameters and a projection target without host work.

        This method is intended for use inside larger ``jax.jit`` graphs.  The
        target columns are placed at ``target_parameter_indices`` and all
        remaining problem parameters are placed at
        ``physical_parameter_indices``.
        """

        problem_parameters = jnp.asarray(problem_parameters, dtype=jnp.float64)
        y_hat = jnp.asarray(y_hat, dtype=jnp.float64)
        packed = jnp.zeros((self.batch_size, self.parameter_count), dtype=y_hat.dtype)
        packed = packed.at[:, self.target_parameter_indices].set(y_hat)
        return packed.at[:, self.physical_parameter_indices].set(problem_parameters)

    def solve_with_linearization_jax(
        self, parameter_values: jax.Array, mu: float | jax.Array = 1e-4
    ):
        """Trace-safe fixed-shape projection solve for composed JAX graphs."""

        parameter_values = jnp.asarray(parameter_values, dtype=jnp.float64)
        mu_values = jnp.asarray(mu, dtype=jnp.float64)
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        if self.solve_default_with_linearization_function is not None:
            return self.solve_default_with_linearization_function(
                parameter_values, mu_values
            )
        variable_values = parameter_values[:, self.target_parameter_indices]
        margin = jnp.asarray(self.interior_margin, dtype=jnp.float64)
        initial_state = self.compiled.initialize_state_function(
            variable_values, parameter_values, margin
        )
        return self.sparse_solver._solve_all_with_linearization_function(
            initial_state, parameter_values, mu_values
        )

    def solve_with_initial_state_jax(
        self,
        initial_state: JaxPrimalDualState,
        parameter_values: jax.Array,
        mu: float | jax.Array = 1e-4,
    ):
        """Trace-safe solve from a caller-supplied primal-dual state."""
        parameter_values = jnp.asarray(parameter_values, dtype=initial_state.x.dtype)
        mu_values = jnp.asarray(mu, dtype=initial_state.x.dtype)
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        return self.sparse_solver._solve_all_with_linearization_function(
            initial_state, parameter_values, mu_values
        )

    def solve_yhat_adjoint_jax(
        self,
        final_state: JaxPrimalDualState,
        final_linearization: Any,
        grad_y_tilde: jax.Array,
        initial_adjoint: jax.Array | None = None,
    ):
        """Trace-safe final-KKT adjoint used by projected neural models."""

        grad_y_tilde = jnp.asarray(grad_y_tilde, dtype=final_state.x.dtype)
        width = (
            final_state.x.shape[1] + final_state.y_c.shape[1] + final_state.y_d.shape[1]
        )
        if initial_adjoint is None:
            initial_adjoint = jnp.zeros(
                (self.batch_size, width), dtype=final_state.x.dtype
            )
        return self.sparse_solver._solve_yhat_adjoint_function(
            final_state, final_linearization, grad_y_tilde, initial_adjoint
        )

    def parameter_vjp_from_adjoint_jax(
        self, final_state: JaxPrimalDualState, parameter_values: jax.Array, adjoint: Any
    ) -> jax.Array:
        """Return the full projection-parameter VJP without another solve.

        The final sparse adjoint supplies the cotangent blocks for stationarity,
        equalities, and inequalities.  Differentiating their contracted KKT
        residual with respect to the packed problem parameters produces
        ``-F_p.T @ lambda``.  Static variable bounds do not contribute.
        """

        def one_sample(state, parameters, lambda_x, lambda_c, lambda_d):
            state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)
            lambda_x = jax.lax.stop_gradient(lambda_x)
            lambda_c = jax.lax.stop_gradient(lambda_c)
            lambda_d = jax.lax.stop_gradient(lambda_d)

            def contracted_residual(current_parameters):
                nlp = self.compiled.evaluate_nlp_single(state, current_parameters)
                stationarity = nlp.grad_f
                if nlp.c.shape[0]:
                    stationarity = stationarity + nlp.jac_c.T @ state.y_c
                if nlp.d.shape[0]:
                    stationarity = stationarity + nlp.jac_d.T @ state.y_d
                value = jnp.vdot(lambda_x, stationarity)
                if nlp.c.shape[0]:
                    value = value + jnp.vdot(lambda_c, nlp.c)
                if nlp.d.shape[0]:
                    value = value + jnp.vdot(lambda_d, nlp.d)
                return value

            return -jax.grad(contracted_residual)(parameters)

        return jax.vmap(one_sample)(
            final_state, parameter_values, adjoint.x, adjoint.y_c, adjoint.y_d
        )

    def solve(
        self,
        parameters: Any,
        *,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        initial_values: Any | None = None,
    ) -> BatchedFixedMuResult:
        """Solve a batch of sparse projection problems with the prepared resources."""
        parameter_array = self.put_parameters(parameters)
        mu_array = self.put_mu(mu)
        if initial_values is None and self.solve_default_function is not None:
            # CUDA and repeated CPU workloads use one outer compiled entry point
            # for target initialization plus the complete Newton solve.
            result = self.solve_default_function(parameter_array, mu_array)
            self.sparse_solver.solve_calls += 1
            return result

        if initial_values is None:
            variable_array = parameter_array[:, self.target_parameter_indices]
        else:
            variable_array = jax.device_put(
                jnp.asarray(initial_values, dtype=jnp.float64), self.device
            )
            expected = (self.batch_size, len(self.compiled.variable_names))
            if tuple(variable_array.shape) != expected:
                raise ValueError(
                    f"initial_values must have shape {expected}; "
                    f"received {tuple(variable_array.shape)}."
                )
        margin = jax.device_put(
            jnp.asarray(self.interior_margin, dtype=jnp.float64), self.device
        )
        initial_state = self.compiled.initialize_state_function(
            variable_array, parameter_array, margin
        )
        return self.sparse_solver.solve(
            initial_state=initial_state, parameter_values=parameter_array, mu=mu_array
        )

    def solve_with_linearization(
        self,
        parameters: Any,
        *,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        initial_values: Any | None = None,
    ):
        """Solve and retain the final sparse KKT data for an adjoint call."""

        parameter_array = self.put_parameters(parameters)
        mu_array = self.put_mu(mu)
        if (
            initial_values is None
            and self.solve_default_with_linearization_function is not None
        ):
            result = self.solve_default_with_linearization_function(
                parameter_array, mu_array
            )
            self.sparse_solver.solve_calls += 1
            return result
        if initial_values is None:
            variable_array = parameter_array[:, self.target_parameter_indices]
        else:
            variable_array = jax.device_put(
                jnp.asarray(initial_values, dtype=jnp.float64), self.device
            )
        margin = jax.device_put(
            jnp.asarray(self.interior_margin, dtype=jnp.float64), self.device
        )
        initial_state = self.compiled.initialize_state_function(
            variable_array, parameter_array, margin
        )
        return self.sparse_solver.solve_with_linearization(
            initial_state=initial_state, parameter_values=parameter_array, mu=mu_array
        )

    def solve_yhat_adjoint(
        self,
        final_state: JaxPrimalDualState,
        parameters: Any,
        grad_y_tilde: Any,
        *,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        final_linearization: Any | None = None,
        initial_adjoint: Any | None = None,
    ):
        """Return the final-KKT adjoint for projection-input gradients.

        When ``final_linearization`` comes from :meth:`solve`, no final
        derivative or preconditioner data is reevaluated.  ``parameters`` and
        ``mu`` remain accepted for API compatibility but are unused in that
        optimized path.
        """

        gradient_array = jax.device_put(
            jnp.asarray(grad_y_tilde, dtype=jnp.float64), self.device
        )
        expected = (self.batch_size, len(self.compiled.variable_names))
        if tuple(gradient_array.shape) != expected:
            raise ValueError(
                f"grad_y_tilde must have shape {expected}; received {tuple(gradient_array.shape)}."
            )
        parameter_array = None
        mu_array = None
        if final_linearization is None:
            parameter_array = self.put_parameters(parameters)
            mu_array = self.put_mu(mu)
        width = (
            final_state.x.shape[1] + final_state.y_c.shape[1] + final_state.y_d.shape[1]
        )
        initial_solution = None
        if initial_adjoint is not None:
            if hasattr(initial_adjoint, "x"):
                initial_solution = jnp.concatenate(
                    (initial_adjoint.x, initial_adjoint.y_c, initial_adjoint.y_d),
                    axis=1,
                )
            else:
                initial_solution = jax.device_put(
                    jnp.asarray(initial_adjoint, dtype=jnp.float64), self.device
                )
            if tuple(initial_solution.shape) != (self.batch_size, width):
                raise ValueError(
                    f"initial_adjoint must be a packed array with shape {(self.batch_size, width)}."
                )
        return self.sparse_solver.solve_yhat_adjoint(
            final_state=final_state,
            final_linearization=final_linearization,
            parameter_values=parameter_array,
            grad_x=gradient_array,
            mu=mu_array,
            initial_solution=initial_solution,
        )

    def solve_loss_yhat_gradient(
        self,
        parameters: Any,
        q_data: Any,
        *,
        q_mode: str = "diagonal",
        valid_mask: Any | None = None,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        initial_values: Any | None = None,
        initial_adjoint: Any | None = None,
    ):
        """Fuse projection, quadratic loss, and implicit ``dL/dy_hat``."""

        if q_mode not in {"identity", "diagonal", "dense"}:
            raise ValueError("q_mode must be identity, diagonal, or dense.")
        parameter_array = self.put_parameters(parameters)
        mu_array = self.put_mu(mu)
        if valid_mask is None:
            valid_mask_array = jnp.ones((self.batch_size,), dtype=bool)
        else:
            valid_mask_array = jax.device_put(
                jnp.asarray(valid_mask, dtype=bool), self.device
            )
            if tuple(valid_mask_array.shape) != (self.batch_size,):
                raise ValueError(f"valid_mask must have shape ({self.batch_size},).")
        q_array = jax.device_put(jnp.asarray(q_data, dtype=jnp.float64), self.device)
        expected_q = (
            (len(self.compiled.variable_names), len(self.compiled.variable_names))
            if q_mode == "dense"
            else (len(self.compiled.variable_names),)
        )
        if tuple(q_array.shape) != expected_q:
            raise ValueError(f"q_data must have shape {expected_q} for {q_mode} mode.")

        # The condensed width is x + equality dual + inequality dual.
        structure = self.compiled.projection_structure
        if structure is None:
            raise RuntimeError(
                "Prepared sparse solver is missing projection structure."
            )
        n_eq = len(structure.equality_support)
        n_ineq = len(structure.inequality_support)
        width = len(self.compiled.variable_names) + n_eq + n_ineq
        if initial_adjoint is None:
            initial_adjoint_array = jnp.zeros(
                (self.batch_size, width), dtype=jnp.float64
            )
        elif hasattr(initial_adjoint, "x"):
            initial_adjoint_array = jnp.concatenate(
                (initial_adjoint.x, initial_adjoint.y_c, initial_adjoint.y_d), axis=1
            )
        else:
            initial_adjoint_array = jax.device_put(
                jnp.asarray(initial_adjoint, dtype=jnp.float64), self.device
            )
        if tuple(initial_adjoint_array.shape) != (self.batch_size, width):
            raise ValueError(
                f"initial_adjoint must have shape {(self.batch_size, width)}."
            )

        if (
            initial_values is None
            and self.solve_default_loss_gradient_functions is not None
        ):
            function = self.solve_default_loss_gradient_functions[
                "dense" if q_mode == "dense" else "diagonal"
            ]
            result = function(
                parameter_array,
                mu_array,
                q_array,
                valid_mask_array,
                initial_adjoint_array,
            )
            self.sparse_solver.solve_calls += 1
            return result

        if initial_values is None:
            variable_array = parameter_array[:, self.target_parameter_indices]
        else:
            variable_array = jax.device_put(
                jnp.asarray(initial_values, dtype=jnp.float64), self.device
            )
        margin = jax.device_put(
            jnp.asarray(self.interior_margin, dtype=jnp.float64), self.device
        )
        initial_state = self.compiled.initialize_state_function(
            variable_array, parameter_array, margin
        )
        return self.sparse_solver.solve_loss_yhat_gradient(
            initial_state=initial_state,
            parameter_values=parameter_array,
            mu_values=mu_array,
            q_data=q_array,
            valid_mask=valid_mask_array,
            q_mode=q_mode,
            initial_adjoint=initial_adjoint_array,
        )


@dataclass(frozen=True, eq=False)
class PreparedPrimalDualInteriorPointProjection:
    """Prepared sparse interior-point projection and implicit adjoint resources."""

    _prepared_solver: Any
    _layer: Any
    problem: Any
    config: ProjectionConfig
    batch_size: int
    dtype: Any
    backend: str

    @property
    def output_size(self):
        """Return the width of the projected primal vector."""
        return int(self._layer.output_dim)

    @property
    def parameter_size(self):
        """Return the width of the projection problem context."""
        return int(self._layer.physical_parameter_count)

    def _parameters(self, parameters):
        if self.parameter_size == 0:
            return jnp.empty((self.batch_size, 0), dtype=self.dtype)
        if parameters is None:
            raise ValueError("parameters are required for this problem")
        values = jnp.asarray(parameters, dtype=self.dtype)
        expected = (self.batch_size, self.parameter_size)
        if values.shape != expected:
            raise ValueError(
                f"parameters must have shape {expected}; got {values.shape}"
            )
        return values

    def project_training(
        self, y_hat, parameters=None, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        values = jnp.asarray(y_hat, dtype=self.dtype)
        if values.ndim == 1:
            values = values[None, :]
        physical = self._parameters(parameters)
        n_eq = int(self._prepared_solver.compiled.initial_state.y_c.shape[1])
        n_ineq = int(self._prepared_solver.compiled.initial_state.y_d.shape[1])
        lambda_eq = (
            jnp.zeros((self.batch_size, n_eq), dtype=self.dtype)
            if multipliers_eq is None
            else jnp.asarray(multipliers_eq, dtype=self.dtype)
        )
        mu_ineq = (
            jnp.ones((self.batch_size, n_ineq), dtype=self.dtype)
            if multipliers_ineq is None
            else jnp.asarray(multipliers_ineq, dtype=self.dtype)
        )
        output = self._layer.project_training(physical, values, lambda_eq, mu_ineq)
        status = jnp.where(
            output.forward_status_codes == BATCH_STATUS_CONVERGED,
            int(SolveStatus.CONVERGED),
            jnp.where(
                output.forward_status_codes == BATCH_STATUS_NUMERICAL_FAILURE,
                int(SolveStatus.NONFINITE),
                int(SolveStatus.MAX_ITERATIONS),
            ),
        ).astype(jnp.int32)
        from purin.solvers.types import ProjectionTrainingOutput

        return ProjectionTrainingOutput(
            y_tilde=output.y_tilde,
            multipliers_eq=output.multipliers_eq,
            multipliers_ineq=output.multipliers_ineq,
            slacks=output.slacks,
            forward_status_codes=status,
            forward_iterations=output.forward_iterations,
            forward_residual_inf_norm=output.forward_residual_inf_norm,
            forward_state_finite=output.forward_state_finite,
            forward_gradient_eligible=output.forward_gradient_eligible,
        )

    def __call__(self, y_hat, parameters=None):
        values = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None, :]
        projected = self._layer.project(self._parameters(parameters), values)
        return projected[0] if squeeze else projected

    def with_info(self, y_hat, parameters=None):
        """Evaluate the projection and return its primal values with solver diagnostics."""
        values = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None, :]
        output = self._layer.project_with_info(self._parameters(parameters), values)
        projected = output.y_tilde[0] if squeeze else output.y_tilde
        status = jnp.where(
            output.forward_status_codes == BATCH_STATUS_CONVERGED,
            int(SolveStatus.CONVERGED),
            jnp.where(
                output.forward_status_codes == BATCH_STATUS_NUMERICAL_FAILURE,
                int(SolveStatus.NONFINITE),
                int(SolveStatus.MAX_ITERATIONS),
            ),
        ).astype(jnp.int32)
        return projected, ProjectionInfo(
            status,
            output.forward_iterations,
            output.forward_residual_inf_norm,
            output.forward_state_finite,
            output.forward_gradient_eligible,
        )

    def solve(self, y_hat, parameters=None, initial_state=None):
        """Solve the prepared projection with the supplied targets and problem context."""
        values = jnp.asarray(y_hat, dtype=self.dtype)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape != (self.batch_size, self.output_size):
            raise ValueError(
                f"y_hat must have shape {(self.batch_size, self.output_size)}; got {values.shape}"
            )
        physical = self._parameters(parameters)
        packed = self._layer.pack_parameters(physical, values)
        mu = jnp.full((self.batch_size,), self.config.mu, dtype=self.dtype)
        if initial_state is None:
            forward = self._prepared_solver.solve_with_linearization_jax(packed, mu)
        else:
            # The preserved solver accepts initial primal values, not a generic
            # backend-neutral state.  Preserve that contract explicitly.
            forward = self._prepared_solver.solve_with_linearization(
                packed, mu=mu, initial_values=initial_state
            )
        state = forward.state
        finite = jnp.ones((self.batch_size,), dtype=bool)
        for block in (state.x, state.s, state.y_c, state.y_d, state.z_L, state.z_U):
            finite = finite & jnp.all(jnp.isfinite(block), axis=1)
        converged = forward.status_codes == BATCH_STATUS_CONVERGED
        status = jnp.where(
            converged,
            int(SolveStatus.CONVERGED),
            jnp.where(
                forward.status_codes == BATCH_STATUS_NUMERICAL_FAILURE,
                int(SolveStatus.NONFINITE),
                int(SolveStatus.MAX_ITERATIONS),
            ),
        ).astype(jnp.int32)
        policy = self.config.implicit_gradient.policy
        if policy == "strict":
            eligible = finite & converged
        elif policy == "residual_checked":
            eligible = finite & (
                converged
                | (
                    forward.final_residual_inf_norm
                    <= self.config.implicit_gradient.inexact_tolerance
                )
            )
        else:
            eligible = finite
        return ProjectionResult(
            y=state.x,
            multipliers_eq=state.y_c,
            multipliers_ineq=state.y_d,
            slacks=state.s,
            status=status,
            iterations=forward.iterations,
            residual_inf_norm=forward.final_residual_inf_norm,
            finite=finite,
            gradient_eligible=eligible,
        )


@dataclass(frozen=True)
class PrimalDualInteriorPointProjection:
    """Sparse interior-point backend with unchanged numerical iterations."""

    problem: Any
    config: ProjectionConfig = ProjectionConfig()

    def prepare(self, *, batch_size: int, dtype=jnp.float64, backend: str = "cpu"):
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
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        projection_problem = (
            self.problem.as_projection_problem()
            if hasattr(self.problem, "as_projection_problem")
            else self.problem
        )
        if not isinstance(projection_problem, CoreProblem):
            if hasattr(projection_problem, "to_symbolic_problem"):
                projection_problem = projection_problem.to_symbolic_problem()
            else:
                raise TypeError(
                    "problem must be symbolic or provide as_projection_problem()"
                )
        parameter_count = len(projection_problem.parameter_names)
        cfg = self.config
        linear = cfg.linear_solver
        implicit = cfg.implicit_gradient
        prepared = projection_problem.prepare_sparse_projection_batch_solver(
            parameters=np.zeros((batch_size, parameter_count), dtype=np.float64),
            batch_size=batch_size,
            options=FixedMuOptions(
                max_iterations=cfg.max_iterations,
                residual_tolerance=cfg.tolerance,
                line_search=None,
            ),
            interior_margin=cfg.interior_margin,
            gmres_tolerance=linear.tolerance,
            gmres_atol=linear.atol,
            gmres_restart=linear.restart,
            gmres_maxiter=linear.max_iterations,
            use_preconditioner=linear.use_preconditioner,
            backtracking=cfg.backtracking,
            max_backtracking_steps=cfg.max_backtracking_steps,
            backtracking_factor=cfg.backtracking_factor,
            backend=backend,
            transpose_mode=cfg.transpose_mode,
            store_residual_history=cfg.store_residual_history,
            adjoint_gmres_tolerance=implicit.adjoint_tolerance,
            adjoint_gmres_atol=linear.atol,
            adjoint_gmres_restart=linear.restart,
            adjoint_gmres_maxiter=linear.max_iterations,
            adjoint_solver_mode=linear.mode,
        )
        layer = make_implicit_projection_layer(
            prepared,
            mu=cfg.mu,
            differentiate_problem_parameters=implicit.differentiate_problem_parameters,
            gradient_policy=implicit.policy,
            inexact_gradient_tolerance=implicit.inexact_tolerance,
            adjoint_gradient_tolerance=implicit.adjoint_tolerance,
        )
        return PreparedPrimalDualInteriorPointProjection(
            prepared, layer, projection_problem, cfg, batch_size, dtype, str(backend)
        )


def _prepare_sparse_evaluators(
    bounds,
    equality_affine_offsets,
    equality_constant_local_jacobian,
    equality_full_support,
    equality_function,
    equality_local_derivatives,
    equality_mask,
    equality_support,
    indices,
    inequality_affine_offsets,
    inequality_constant_local_jacobian,
    inequality_full_support,
    inequality_function,
    inequality_local_derivatives,
    inequality_mask,
    inequality_support,
    target_parameter_indices,
    use_shared_constant_jacobian,
):
    """Prepare derivative and residual evaluation callables for fixed batch shapes."""
    derivative_batch = jax.vmap(
        lambda state, params: (
            *equality_local_derivatives(state.x, params),
            *inequality_local_derivatives(state.x, params),
        ),
        in_axes=(0, 0),
    )
    constraint_values_batch = jax.vmap(
        lambda state, params: (
            equality_function(state.x, params),
            inequality_function(state.x, params),
        ),
        in_axes=(0, 0),
    )
    affine_offsets_batch = None
    if equality_affine_offsets is not None and inequality_affine_offsets is not None:
        affine_offsets_batch = jax.vmap(
            lambda state, params: (
                equality_affine_offsets(state.x, params),
                inequality_affine_offsets(state.x, params),
            ),
            in_axes=(0, 0),
        )

    def evaluate_dynamic_one(state, params, mu_i):
        return _evaluate_numeric_data(
            state,
            params,
            equality_function,
            inequality_function,
            equality_local_derivatives,
            inequality_local_derivatives,
            equality_support,
            equality_mask,
            inequality_support,
            inequality_mask,
            target_parameter_indices,
            indices,
            bounds,
            mu_i,
        )

    evaluate_dynamic_batch = jax.vmap(evaluate_dynamic_one, in_axes=(0, 0, 0))

    equality_jacobian_batch_axis = (
        None
        if use_shared_constant_jacobian and equality_constant_local_jacobian is not None
        else 0
    )
    inequality_jacobian_batch_axis = (
        None
        if use_shared_constant_jacobian
        and inequality_constant_local_jacobian is not None
        else 0
    )

    def evaluate_fixed_one(state, params, mu_i, eq_j, ineq_j, eq_offset, ineq_offset):
        equalities = (
            local_jacobian_matvec(
                eq_j,
                equality_support,
                equality_mask,
                state.x,
                full_support=equality_full_support,
            )
            + eq_offset
        )
        inequalities = (
            local_jacobian_matvec(
                ineq_j,
                inequality_support,
                inequality_mask,
                state.x,
                full_support=inequality_full_support,
            )
            + ineq_offset
        )
        empty_hessian = jnp.empty((0, 0, 0), dtype=state.x.dtype)
        numeric = _assemble_numeric_data_from_constraint_values(
            state,
            params,
            equalities,
            inequalities,
            eq_j,
            ineq_j,
            empty_hessian,
            empty_hessian,
            equality_support,
            equality_mask,
            inequality_support,
            inequality_mask,
            target_parameter_indices,
            indices,
            bounds,
            mu_i,
        )
        # Keep constant Jacobians outside vmap output so they remain truly
        # unbatched in the Newton/GMRES loop.
        return (numeric.residual, numeric.slacks, numeric.sigma_x, numeric.sigma_s)

    evaluate_fixed_batch = jax.vmap(
        evaluate_fixed_one,
        in_axes=(
            0,
            0,
            0,
            equality_jacobian_batch_axis,
            inequality_jacobian_batch_axis,
            0,
            0,
        ),
    )
    return (
        affine_offsets_batch,
        constraint_values_batch,
        derivative_batch,
        evaluate_dynamic_batch,
        evaluate_fixed_batch,
    )


def _prepare_pallas_product(
    column_mask,
    column_rows,
    column_slots,
    combined_constant_local_jacobian,
    combined_mask,
    combined_support,
    constraints_affine,
    gpu_kkt_mode,
    n_variables,
    pallas_interpret,
    pallas_kkt_block_size,
    use_shared_constant_jacobian,
):
    """Prepare the optional Pallas KKT product without changing backend policy."""
    pallas_kkt_matvec = None
    if gpu_kkt_mode == "pallas":
        if not use_shared_constant_jacobian:
            raise ValueError(
                "The Pallas KKT kernel requires shared constant Jacobians."
            )
        if not constraints_affine:
            raise ValueError(
                "The Pallas KKT kernel currently supports affine constraints only."
            )
        if combined_constant_local_jacobian is None:
            raise ValueError(
                "The Pallas KKT kernel requires a parameter-independent Jacobian."
            )
        if not pallas_available():
            raise RuntimeError("Pallas is unavailable in this JAX installation.")
        pallas_kkt_matvec = build_affine_constant_kkt_pallas_matvec(
            jacobian=combined_constant_local_jacobian,
            row_support=combined_support,
            row_mask=combined_mask,
            column_rows=column_rows,
            column_slots=column_slots,
            column_mask=column_mask,
            n_variables=n_variables,
            block_size=pallas_kkt_block_size,
            interpret=bool(pallas_interpret),
        )
    return pallas_kkt_matvec


def _prepare_constant_jacobians(
    equality_constant_local_jacobian, inequality_constant_local_jacobian
):
    """Prepare the optional shared constant Jacobian and its squared values."""
    equality_constant_local_jacobian = (
        None
        if equality_constant_local_jacobian is None
        else jnp.asarray(equality_constant_local_jacobian, dtype=jnp.float64)
    )
    inequality_constant_local_jacobian = (
        None
        if inequality_constant_local_jacobian is None
        else jnp.asarray(inequality_constant_local_jacobian, dtype=jnp.float64)
    )
    combined_constant_local_jacobian = None
    combined_constant_local_jacobian_squared = None
    if (
        equality_constant_local_jacobian is not None
        and inequality_constant_local_jacobian is not None
    ):
        if equality_constant_local_jacobian.shape[0] == 0:
            combined_constant_local_jacobian = inequality_constant_local_jacobian
        elif inequality_constant_local_jacobian.shape[0] == 0:
            combined_constant_local_jacobian = equality_constant_local_jacobian
        else:
            combined_constant_local_jacobian = jnp.concatenate(
                (
                    equality_constant_local_jacobian,
                    inequality_constant_local_jacobian,
                ),
                axis=0,
            )
        combined_constant_local_jacobian_squared = (
            combined_constant_local_jacobian * combined_constant_local_jacobian
        )
    return (
        combined_constant_local_jacobian,
        combined_constant_local_jacobian_squared,
        equality_constant_local_jacobian,
        inequality_constant_local_jacobian,
    )


def _prepare_sparse_transpose(
    combined_mask_np,
    combined_support_np,
    n_variables,
    use_column_bucketed_transpose,
    use_column_gather_transpose,
    use_shared_constant_jacobian,
):
    """Select the existing sparse transpose mode and prepare its fixed plan."""
    if use_column_gather_transpose is None:
        use_column_gather_transpose = jax.default_backend() != "cpu"
    use_column_gather_transpose = bool(use_column_gather_transpose)
    use_column_bucketed_transpose = bool(use_column_bucketed_transpose)
    use_shared_constant_jacobian = bool(use_shared_constant_jacobian)
    if use_column_gather_transpose and use_column_bucketed_transpose:
        raise ValueError(
            "Global gather and bucketed gather transpose modes are mutually exclusive."
        )
    transpose_mode = (
        "bucketed"
        if use_column_bucketed_transpose
        else "gather"
        if use_column_gather_transpose
        else "scatter"
    )
    transpose_plan = prepare_sparse_transpose_plan(
        combined_support_np,
        combined_mask_np,
        n_variables,
        mode=transpose_mode,
        build_all=True,
    )
    return (
        transpose_plan,
        use_column_bucketed_transpose,
        use_column_gather_transpose,
        use_shared_constant_jacobian,
    )


def _prepare_sparse_support_layout(
    equality_mask,
    equality_support,
    inequality_mask,
    inequality_support,
    regularization,
    target_parameter_indices,
):
    """Normalize sparse supports and build the combined constraint layout."""
    reg = regularization or NewtonRegularization()
    equality_support_np = np.asarray(equality_support, dtype=np.int32)
    inequality_support_np = np.asarray(inequality_support, dtype=np.int32)
    equality_mask_np = np.asarray(equality_mask, dtype=bool)
    inequality_mask_np = np.asarray(inequality_mask, dtype=bool)
    support_width = next(
        (
            int(array.shape[1])
            for array in (equality_support_np, inequality_support_np)
            if array.ndim == 2
        ),
        0,
    )
    if equality_support_np.ndim == 1 and equality_support_np.size == 0:
        equality_support_np = equality_support_np.reshape(0, support_width)
        equality_mask_np = equality_mask_np.reshape(0, support_width)
    if inequality_support_np.ndim == 1 and inequality_support_np.size == 0:
        inequality_support_np = inequality_support_np.reshape(0, support_width)
        inequality_mask_np = inequality_mask_np.reshape(0, support_width)
    equality_full_support = bool(np.all(equality_mask_np))
    inequality_full_support = bool(np.all(inequality_mask_np))
    combined_full_support = equality_full_support and inequality_full_support
    if equality_support_np.shape[0] == 0:
        combined_support_np = inequality_support_np
        combined_mask_np = inequality_mask_np
    elif inequality_support_np.shape[0] == 0:
        combined_support_np = equality_support_np
        combined_mask_np = equality_mask_np
    else:
        combined_support_np = np.concatenate(
            (equality_support_np, inequality_support_np), axis=0
        )
        combined_mask_np = np.concatenate(
            (equality_mask_np, inequality_mask_np), axis=0
        )
    n_variables = int(target_parameter_indices.shape[0])
    return (
        combined_full_support,
        combined_mask_np,
        combined_support_np,
        equality_full_support,
        equality_mask_np,
        equality_support_np,
        inequality_full_support,
        inequality_mask_np,
        inequality_support_np,
        n_variables,
        reg,
    )


def _resolve_sparse_adjoint_options(
    adjoint_arnoldi_prefix_block,
    adjoint_gmres_atol,
    adjoint_gmres_maxiter,
    adjoint_gmres_restart,
    adjoint_gmres_tolerance,
    adjoint_gmres_unroll,
    arnoldi_prefix_block,
    gmres_atol,
    gmres_maxiter,
    gmres_restart,
    gmres_tolerance,
    gmres_unroll,
):
    """Resolve inherited adjoint defaults and validate their original limits."""
    adjoint_gmres_tolerance = (
        gmres_tolerance
        if adjoint_gmres_tolerance is None
        else float(adjoint_gmres_tolerance)
    )
    adjoint_gmres_atol = (
        gmres_atol if adjoint_gmres_atol is None else float(adjoint_gmres_atol)
    )
    adjoint_gmres_restart = (
        gmres_restart if adjoint_gmres_restart is None else int(adjoint_gmres_restart)
    )
    adjoint_gmres_maxiter = (
        gmres_maxiter if adjoint_gmres_maxiter is None else int(adjoint_gmres_maxiter)
    )
    adjoint_gmres_unroll = (
        gmres_unroll if adjoint_gmres_unroll is None else int(adjoint_gmres_unroll)
    )
    adjoint_arnoldi_prefix_block = (
        arnoldi_prefix_block
        if adjoint_arnoldi_prefix_block is None
        else int(adjoint_arnoldi_prefix_block)
    )
    if adjoint_gmres_tolerance <= 0.0:
        raise ValueError("adjoint_gmres_tolerance must be positive.")
    if adjoint_gmres_restart < 1:
        raise ValueError("adjoint_gmres_restart must be positive.")
    if adjoint_gmres_maxiter < 1:
        raise ValueError("adjoint_gmres_maxiter must be positive.")
    if adjoint_gmres_unroll < 1:
        raise ValueError("adjoint_gmres_unroll must be positive.")
    if adjoint_arnoldi_prefix_block < 0:
        raise ValueError("adjoint_arnoldi_prefix_block must be nonnegative.")
    return (
        adjoint_arnoldi_prefix_block,
        adjoint_gmres_atol,
        adjoint_gmres_maxiter,
        adjoint_gmres_restart,
        adjoint_gmres_tolerance,
        adjoint_gmres_unroll,
    )


def _validate_sparse_execution_options(
    adjoint_solver_mode,
    arnoldi_prefix_block,
    backtracking_factor,
    gmres_unroll,
    gpu_kkt_mode,
    max_backtracking_steps,
    pallas_kkt_block_size,
):
    """Validate backtracking, Krylov and device execution options."""
    if not isinstance(max_backtracking_steps, int) or max_backtracking_steps < 1:
        raise ValueError("max_backtracking_steps must be a positive integer.")
    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must satisfy 0 < factor < 1.")
    if not isinstance(gmres_unroll, int) or gmres_unroll < 1:
        raise ValueError("gmres_unroll must be a positive integer.")
    if not isinstance(arnoldi_prefix_block, int) or arnoldi_prefix_block < 0:
        raise ValueError("arnoldi_prefix_block must be a nonnegative integer.")
    if gpu_kkt_mode not in {"xla", "pallas"}:
        raise ValueError("gpu_kkt_mode must be 'xla' or 'pallas'.")
    if not isinstance(pallas_kkt_block_size, int) or pallas_kkt_block_size < 1:
        raise ValueError("pallas_kkt_block_size must be a positive integer.")
    if adjoint_solver_mode not in {"gmres", "dense"}:
        raise ValueError("adjoint_solver_mode must be 'gmres' or 'dense'.")


def _validate_sparse_iteration_options(
    batch_size,
    gmres_maxiter,
    gmres_restart,
    gmres_tolerance,
    max_iterations,
    residual_tolerance,
    step_scale,
    tau,
):
    """Validate batch dimensions, iteration limits and primal step settings."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")
    if not isinstance(gmres_maxiter, int) or gmres_maxiter < 1:
        raise ValueError("gmres_maxiter must be a positive integer.")
    if not isinstance(gmres_restart, int) or gmres_restart < 1:
        raise ValueError("gmres_restart must be a positive integer.")
    if residual_tolerance <= 0.0 or gmres_tolerance <= 0.0:
        raise ValueError("residual and GMRES tolerances must be positive.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must satisfy 0 < tau < 1.")
    if not 0.0 < step_scale <= 1.0:
        raise ValueError("step_scale must satisfy 0 < step_scale <= 1.")
