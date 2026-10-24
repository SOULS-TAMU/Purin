"""Primal-dual interior-point state, sparse numerical data, and result types."""

from __future__ import annotations
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as _np
from purin.problems.types import BatchNamedValues
from purin.problems.types import NamedValues
from purin._utils.validation import strict_zip

Array = jax.Array


class PrimalDualState(NamedTuple):
    """Primal variables, inequality slacks, residual multipliers and bound multipliers."""

    x: Array
    s: Array
    y_c: Array
    y_d: Array
    z_L: Array
    z_U: Array
    v_L: Array
    v_U: Array

    def blocks(self) -> tuple[Array, ...]:
        """Return state arrays in the canonical KKT block order."""
        return tuple(self)

    def flatten(self) -> Array:
        """Concatenate state blocks into one vector in canonical KKT order."""
        return jnp.concatenate(self.blocks())

    def axpy(
        self, alpha: Array | float, direction: "PrimalDualState"
    ) -> "PrimalDualState":
        """Return a state with each block advanced by alpha times its direction."""
        return PrimalDualState(
            *(block + alpha * delta for block, delta in strict_zip(self, direction))
        )


class BoundIndices(NamedTuple):
    """Indices of finite lower and upper bounds on primal variables and slacks."""

    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array


class BoundValues(NamedTuple):
    """Finite lower and upper bound values for primal variables and slacks."""

    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array


class BoundSlacks(NamedTuple):
    """Distances from primal variables and inequality slacks to their finite bounds."""

    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array

    def blocks(self) -> tuple[Array, ...]:
        """Return lower and upper slack arrays in their canonical order."""
        return tuple(self)


class NLPQuantities(NamedTuple):
    """Objective derivatives, constraint residuals and Jacobians at a primal point."""

    grad_f: Array
    c: Array
    d: Array
    jac_c: Array
    jac_d: Array
    hessian_lagrangian: Array
    objective_value: Array | None = None


class KKTResidual(NamedTuple):
    """Stationarity, feasibility and complementarity residual blocks in state order."""

    x: Array
    s: Array
    y_c: Array
    y_d: Array
    z_L: Array
    z_U: Array
    v_L: Array
    v_U: Array

    def blocks(self) -> tuple[Array, ...]:
        """Return residual arrays in the canonical KKT block order."""
        return tuple(self)

    def flatten(self) -> Array:
        """Concatenate all KKT residual blocks into one vector."""
        return jnp.concatenate(self.blocks())


class NewtonRegularization(NamedTuple):
    """Diagonal regularization for primal, slack and constraint Newton blocks."""

    delta_x: float = 0.0
    delta_s: float = 0.0
    delta_c: float = 0.0
    delta_d: float = 0.0


class ReducedKKTSystem(NamedTuple):
    """Condensed Newton matrix, right-hand side and bound curvature terms."""

    matrix: Array
    rhs: Array
    sigma_x: Array
    sigma_s: Array


class NewtonSolveResult(NamedTuple):
    """Recovered Newton direction and reduced/full-system residual diagnostics."""

    direction: PrimalDualState
    residual: KKTResidual
    slacks: BoundSlacks
    reduced_system: ReducedKKTSystem
    reduced_solution: Array
    reduced_residual_inf_norm: Array
    full_residual_inf_norm: Array


class FractionToBoundarySteps(NamedTuple):
    """Maximum admissible step lengths for each bound and multiplier block."""

    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array
    z_L: Array
    z_U: Array
    v_L: Array
    v_U: Array

    @property
    def alpha_primal(self) -> Array:
        """Return the most restrictive primal-variable or slack step length."""
        return jnp.minimum(
            jnp.minimum(self.x_lower, self.x_upper),
            jnp.minimum(self.s_lower, self.s_upper),
        )

    @property
    def alpha_dual(self) -> Array:
        """Return the most restrictive bound-multiplier step length."""
        return jnp.minimum(
            jnp.minimum(self.z_L, self.z_U), jnp.minimum(self.v_L, self.v_U)
        )


class SparseNumericData(NamedTuple):
    """Numerical sparse data evaluated once per Newton iteration."""

    residual: KKTResidual
    slacks: BoundSlacks
    sigma_x: jax.Array
    sigma_s: jax.Array
    equality_jacobian: jax.Array
    inequality_jacobian: jax.Array
    equality_hessian: jax.Array
    inequality_hessian: jax.Array


class SparseProjectionLinearization(NamedTuple):
    """Final condensed-KKT data reused by implicit adjoint solves."""

    combined_jacobian: jax.Array
    equality_hessian: jax.Array
    inequality_hessian: jax.Array
    inverse_slack: jax.Array
    primal_kkt_diagonal: jax.Array
    dual_kkt_diagonal: jax.Array
    preconditioner_diagonal: jax.Array


class SparseProjectionSolveResult(NamedTuple):
    """Forward result with the final sparse KKT linearization attached."""

    state: PrimalDualState
    status_codes: jax.Array
    iterations: jax.Array
    final_residual_inf_norm: jax.Array
    residual_history: jax.Array
    final_linearization: SparseProjectionLinearization

    @property
    def converged(self):
        """Return the per-instance mask of successful forward projection solves."""
        return self.status_codes == BATCH_STATUS_CONVERGED


class SparseProjectionAdjointResult(NamedTuple):
    """Condensed final-KKT adjoint result for projection-input gradients."""

    x: jax.Array
    y_c: jax.Array
    y_d: jax.Array
    linear_status: jax.Array
    linear_residual_inf_norm: jax.Array
    linear_iterations: jax.Array


class SparseProjectionLossGradientResult(NamedTuple):
    """Fused projection, quadratic loss, and ``dL/dy_hat`` result."""

    losses: jax.Array
    y_tilde: jax.Array
    grad_y_hat: jax.Array
    forward_status_codes: jax.Array
    forward_iterations: jax.Array
    forward_residual_inf_norm: jax.Array
    adjoint_status_codes: jax.Array
    adjoint_iterations: jax.Array
    adjoint_residual_inf_norm: jax.Array


# High-level result views.

BATCH_STATUS_ACTIVE = 0
BATCH_STATUS_CONVERGED = 1
BATCH_STATUS_MAXIMUM_ITERATIONS = 2
BATCH_STATUS_STALLED = 3
BATCH_STATUS_NUMERICAL_FAILURE = 4
BATCH_STATUS_NAMES = (
    "active",
    "converged",
    "maximum_iterations",
    "stalled",
    "numerical_failure",
)


@_dataclass(frozen=True)
class ProblemSolution:
    """Named single-problem solution with objective, state and solver diagnostics."""

    problem_name: str
    variables: NamedValues
    parameters: NamedValues
    objective: float
    equalities: _np.ndarray
    inequalities: _np.ndarray
    raw_result: _Any

    @property
    def state(self):
        """Return the primal-dual state attached to this solution."""
        return self.raw_result.state

    @property
    def status(self) -> str:
        """Return the solver termination status for this solution."""
        return str(self.raw_result.status)

    @property
    def converged(self) -> bool:
        """Report whether the solve terminated successfully."""
        return bool(self.raw_result.converged)

    @property
    def iterations(self) -> int:
        """Return the number of solver iterations performed."""
        return int(self.raw_result.iterations)

    @property
    def residual_inf_norm(self) -> float:
        """Return the final maximum absolute KKT residual."""
        return float(self.raw_result.final_residual_inf_norm)

    def __getitem__(self, variable_name: str) -> float:
        return self.variables[variable_name]


@_dataclass(frozen=True)
class BatchProblemSolution:
    """Named batched solution with per-instance objectives and solver diagnostics."""

    problem_name: str
    variables: BatchNamedValues
    parameters: BatchNamedValues
    objectives: _np.ndarray
    equalities: _np.ndarray
    inequalities: _np.ndarray
    raw_result: _Any
    projection_mode: str = "generic_dense"
    device_platform: str = "cpu"
    status_codes_host: _np.ndarray | None = None
    converged_host: _np.ndarray | None = None
    iterations_host: _np.ndarray | None = None
    residual_inf_norm_host: _np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        """Return the number of solved problem instances."""
        return self.variables.batch_size

    @property
    def objective(self) -> _np.ndarray:
        """Return the objective values of the solved instances."""
        return self.objectives

    @property
    def status(self) -> tuple[str, ...]:
        """Return the per-instance solver termination statuses."""
        codes = (
            self.status_codes_host
            if self.status_codes_host is not None
            else _np.asarray(self.raw_result.status_codes, dtype=_np.int32)
        )
        return tuple(BATCH_STATUS_NAMES[int(code)] for code in codes)

    @property
    def converged(self) -> _np.ndarray:
        """Return the per-instance convergence mask."""
        if self.converged_host is not None:
            return self.converged_host
        return _np.asarray(self.raw_result.converged, dtype=bool)

    @property
    def all_converged(self) -> bool:
        """Report whether every instance in the batch converged."""
        return bool(_np.all(self.converged))

    @property
    def iterations(self) -> _np.ndarray:
        """Return the solver iteration count for each instance."""
        if self.iterations_host is not None:
            return self.iterations_host
        return _np.asarray(self.raw_result.iterations, dtype=_np.int32)

    @property
    def residual_inf_norm(self) -> _np.ndarray:
        """Return each instance's final maximum absolute KKT residual."""
        if self.residual_inf_norm_host is not None:
            return self.residual_inf_norm_host
        return _np.asarray(self.raw_result.final_residual_inf_norm, dtype=_np.float64)

    @property
    def state(self):
        """Return the batched primal-dual state attached to this solution."""
        return self.raw_result.state

    def __getitem__(self, variable_name: str) -> _np.ndarray:
        return self.variables[variable_name]
