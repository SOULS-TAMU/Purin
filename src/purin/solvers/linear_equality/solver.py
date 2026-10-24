"""Sparse Euclidean projection onto fixed linear equalities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import cg

try:
    import scipy.sparse as scipy_sparse
except Exception:  # pragma: no cover
    scipy_sparse = None

from purin.linalg.sparse import as_bcoo
from purin.solvers.config import LinearEqualityConfig
from purin.solvers.types import ProjectionInfo
from purin.solvers.types import ProjectionResult
from purin.solvers.types import ProjectionTrainingOutput
from purin.solvers.types import SolveStatus


def _constraint_tolerance(dtype):
    # Absolute feasibility threshold, including the precision floor for float32.
    return max(1e-10, 32.0 * float(jnp.finfo(dtype).eps))


def _sparse_multiplier(matrix, residual, regularization):
    """Solve the Gram system without assembling it; differentiate implicitly."""

    def operator(vector):
        return matrix @ (matrix.T @ vector) + regularization * vector

    diagonal = (
        jnp.zeros((matrix.shape[0],), dtype=residual.dtype)
        .at[matrix.indices[:, 0]]
        .add(matrix.data**2)
        + regularization
    )
    diagonal = jnp.maximum(diagonal, jnp.finfo(residual.dtype).tiny)
    tolerance = 8.0 * float(jnp.finfo(residual.dtype).eps)

    def solve(rhs):
        return cg(
            operator,
            rhs,
            tol=tolerance,
            atol=tolerance,
            maxiter=max(32, 4 * matrix.shape[0]),
            M=lambda v: v / diagonal,
        )[0]

    multiplier = solve(residual)
    # Fixed refinement passes also preserve reverse-mode differentiation.
    for _ in range(2):
        multiplier = multiplier + solve(residual - operator(multiplier))
    return multiplier


def _host_matrix(value, name: str):
    if scipy_sparse is not None and scipy_sparse.issparse(value):
        matrix = value.astype(np.float64).tocsr()
    else:
        matrix = np.asarray(value, dtype=np.float64)
    if len(matrix.shape) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return matrix


@dataclass(frozen=True, eq=False)
class PreparedLinearEqualityProjection:
    """Prepared affine equality projection using fixed linear-system resources."""

    A: Any
    b0: jax.Array
    b_parameter_map: jax.Array
    cholesky_factor: (
        jax.Array | None
    )  # Retained constructor field; sparse solves need no factor.
    batch_size: int
    dtype: Any
    backend: str
    regularization: float

    @property
    def parameter_size(self):
        """Return the width of the projection problem context."""
        return int(self.b_parameter_map.shape[1])

    @property
    def output_size(self):
        """Return the width of the projected primal vector."""
        return int(self.A.shape[1])

    @property
    def equality_size(self):
        """Return the number of equality residual rows."""
        return int(self.A.shape[0])

    def _parameters(self, parameters):
        if self.parameter_size == 0:
            return jnp.zeros((self.batch_size, 0), dtype=self.dtype)
        if parameters is None:
            raise ValueError("parameters are required for parameter-dependent b")
        values = jnp.asarray(parameters, dtype=self.dtype)
        expected = (self.batch_size, self.parameter_size)
        if values.shape != expected:
            raise ValueError(
                f"parameters must have shape {expected}; got {values.shape}"
            )
        return values

    def _solve_batch(self, y_hat, theta):
        rhs = self.b0 + theta @ self.b_parameter_map.T
        if self.equality_size == 0:
            projected = y_hat
            multipliers = jnp.empty((self.batch_size, 0), dtype=y_hat.dtype)
        else:
            residual = jax.vmap(lambda y: self.A @ y)(y_hat) - rhs
            multipliers = jax.vmap(
                lambda value: _sparse_multiplier(self.A, value, self.regularization)
            )(residual)
            correction = jax.vmap(lambda value: self.A.T @ value)(multipliers)
            projected = y_hat - correction
        final_residual = (
            jax.vmap(lambda y: self.A @ y)(projected) - rhs
            if self.equality_size
            else jnp.zeros((self.batch_size, 0), dtype=y_hat.dtype)
        )
        residual_norm = jnp.max(
            jnp.abs(final_residual), axis=1, initial=jnp.asarray(0.0, y_hat.dtype)
        )
        finite = (
            jnp.all(jnp.isfinite(projected), axis=1)
            & jnp.all(jnp.isfinite(multipliers), axis=1)
            & jnp.isfinite(residual_norm)
        )
        converged = finite & (residual_norm <= _constraint_tolerance(y_hat.dtype))
        status = jnp.where(
            converged,
            int(SolveStatus.CONVERGED),
            jnp.where(
                finite,
                int(SolveStatus.LINEAR_SOLVE_FAILURE),
                int(SolveStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return projected, multipliers, status, residual_norm, finite

    def solve(self, y_hat, parameters=None, initial_state=None):
        """Solve the prepared projection with the supplied targets and problem context."""
        del initial_state
        values = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None, :]
        expected = (self.batch_size, self.output_size)
        if values.shape != expected:
            raise ValueError(f"y_hat must have shape {expected}; got {values.shape}")
        theta = self._parameters(parameters)
        projected, multipliers, status, residual, finite = self._solve_batch(
            values, theta
        )
        return ProjectionResult(
            y=projected,
            multipliers_eq=multipliers,
            multipliers_ineq=jnp.empty((self.batch_size, 0), dtype=values.dtype),
            slacks=jnp.empty((self.batch_size, 0), dtype=values.dtype),
            status=status,
            iterations=jnp.zeros((self.batch_size,), dtype=jnp.int32),
            residual_inf_norm=residual,
            finite=finite,
            gradient_eligible=finite & (status == int(SolveStatus.CONVERGED)),
        )

    def project_training(
        self, y_hat, parameters=None, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        del multipliers_eq, multipliers_ineq
        result = self.solve(y_hat, parameters)
        return ProjectionTrainingOutput(
            y_tilde=result.y,
            multipliers_eq=jax.lax.stop_gradient(result.multipliers_eq),
            multipliers_ineq=jax.lax.stop_gradient(result.multipliers_ineq),
            slacks=jax.lax.stop_gradient(result.slacks),
            forward_status_codes=result.status,
            forward_iterations=result.iterations,
            forward_residual_inf_norm=result.residual_inf_norm,
            forward_state_finite=result.finite,
            forward_gradient_eligible=result.gradient_eligible,
        )

    def __call__(self, y_hat, parameters=None):
        result = self.solve(y_hat, parameters)
        return result.y[0] if jnp.asarray(y_hat).ndim == 1 else result.y

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


@dataclass(frozen=True)
class LinearEqualityProjection:
    """Affine equality projection defined by a matrix and parameterized right-hand side."""

    A: Any
    b: Any | None = None
    b0: Any | None = None
    b_parameter_map: Any | None = None
    config: LinearEqualityConfig = LinearEqualityConfig()
    regularization: float | None = None

    def prepare(self, *, batch_size: int, dtype=jnp.float64, backend: str = "cpu"):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        if int(batch_size) < 1:
            raise ValueError("batch size must be positive")
        regularization = (
            self.config.regularization
            if self.regularization is None
            else float(self.regularization)
        )
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and nonnegative")
        A_host = _host_matrix(self.A, "A")
        equality_size, variable_size = A_host.shape
        source_b = self.b0 if self.b0 is not None else self.b
        b0 = (
            np.zeros((equality_size,), dtype=np.float64)
            if source_b is None
            else np.asarray(source_b, dtype=np.float64)
        )
        if b0.shape != (equality_size,):
            raise ValueError(f"b must have shape {(equality_size,)}; got {b0.shape}")
        if self.b_parameter_map is None:
            parameter_map = np.zeros((equality_size, 0), dtype=np.float64)
        else:
            parameter_map = np.asarray(self.b_parameter_map, dtype=np.float64)
            if parameter_map.ndim != 2 or parameter_map.shape[0] != equality_size:
                raise ValueError(
                    "b_parameter_map must have shape (number of equalities, m)"
                )
        regularization = (
            self.config.regularization
            if self.regularization is None
            else float(self.regularization)
        )
        # Sparse preparation preserves the historical full-row-rank validation.
        A_host = scipy_sparse.csr_matrix(A_host)
        if equality_size:
            gram = (A_host @ A_host.T).tocsc()
            gram = gram + regularization * scipy_sparse.eye(equality_size, format="csc")
            try:
                scipy_sparse.linalg.splu(gram)
            except RuntimeError as error:
                raise ValueError(
                    "A must have full row rank when regularization is zero"
                ) from error
        return PreparedLinearEqualityProjection(
            A=as_bcoo(A_host).astype(dtype),
            b0=jnp.asarray(b0, dtype=dtype),
            b_parameter_map=jnp.asarray(parameter_map, dtype=dtype),
            cholesky_factor=None,
            batch_size=int(batch_size),
            dtype=dtype,
            backend=str(backend),
            regularization=regularization,
        )


@dataclass(frozen=True, eq=False)
class PreparedCanonicalLinearEqualityProjection:
    """Exact projection for affine symbolic equalities.

    The equality Jacobian may depend on problem parameters, but it must not
    depend on the decision variables. Sparse Jacobian products and a matrix-free
    Gram solve are used per sample; execution remains JIT-compatible.
    """

    canonical: Any
    config: LinearEqualityConfig
    batch_size: int
    dtype: Any
    backend: str

    def __post_init__(self):
        rows = self.canonical.equality_sparsity
        if rows is None:
            rows = [range(self.output_size) for _ in range(self.equality_size)]
        indices = [
            (row, column) for row, columns in enumerate(rows) for column in columns
        ]
        object.__setattr__(
            self,
            "_jacobian_indices",
            jnp.asarray(indices, dtype=jnp.int32).reshape((-1, 2)),
        )
        object.__setattr__(
            self, "_solve_batch_jit", jax.jit(jax.vmap(self._solve_one, in_axes=(0, 0)))
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

    def _solve_one(self, y_hat, theta):
        me = self.equality_size
        if me == 0:
            return (
                y_hat,
                jnp.empty((0,), dtype=y_hat.dtype),
                jnp.asarray(0.0, y_hat.dtype),
                jnp.asarray(True),
            )
        indices = self._jacobian_indices
        values_function = self.canonical.equality_jacobian_values
        if values_function is not None:
            data = values_function(theta, y_hat)
        else:
            dense = self.canonical.equality_jacobian(theta, y_hat).reshape(
                (me, self.output_size)
            )
            data = dense[indices[:, 0], indices[:, 1]]
        jacobian = BCOO((data, indices), shape=(me, self.output_size))
        residual = self.canonical.equalities(theta, y_hat).reshape((me,))
        regularization = jnp.asarray(self.config.regularization, y_hat.dtype)
        multiplier = _sparse_multiplier(jacobian, residual, regularization)
        projected = y_hat - jacobian.T @ multiplier
        final_residual = self.canonical.equalities(theta, projected).reshape((me,))
        residual_norm = jnp.max(
            jnp.abs(final_residual), initial=jnp.asarray(0.0, y_hat.dtype)
        )
        finite = (
            jnp.all(jnp.isfinite(projected))
            & jnp.all(jnp.isfinite(multiplier))
            & jnp.isfinite(residual_norm)
        )
        return projected, multiplier, residual_norm, finite

    def _normalize(self, y_hat, parameters):
        values = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None, :]
        expected_y = (self.batch_size, self.output_size)
        if values.shape != expected_y:
            raise ValueError(f"y_hat must have shape {expected_y}; got {values.shape}")
        if self.parameter_size == 0:
            theta = jnp.zeros((self.batch_size, 0), dtype=self.dtype)
        else:
            if parameters is None:
                raise ValueError("parameters are required for this problem")
            theta = jnp.asarray(parameters, dtype=self.dtype)
            expected_theta = (self.batch_size, self.parameter_size)
            if theta.shape != expected_theta:
                raise ValueError(
                    f"parameters must have shape {expected_theta}; got {theta.shape}"
                )
        return values, theta, squeeze

    def solve(self, y_hat, parameters=None, initial_state=None):
        """Solve the prepared projection with the supplied targets and problem context."""
        del initial_state
        values, theta, _squeeze = self._normalize(y_hat, parameters)
        projected, multipliers, residual, finite = self._solve_batch_jit(values, theta)
        converged = finite & (residual <= _constraint_tolerance(values.dtype))
        status = jnp.where(
            converged,
            int(SolveStatus.CONVERGED),
            jnp.where(
                finite,
                int(SolveStatus.LINEAR_SOLVE_FAILURE),
                int(SolveStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return ProjectionResult(
            y=projected,
            multipliers_eq=multipliers,
            multipliers_ineq=jnp.empty((self.batch_size, 0), dtype=values.dtype),
            slacks=jnp.empty((self.batch_size, 0), dtype=values.dtype),
            status=status,
            iterations=jnp.zeros((self.batch_size,), dtype=jnp.int32),
            residual_inf_norm=residual,
            finite=finite,
            gradient_eligible=converged,
        )

    def project_training(
        self, y_hat, parameters=None, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        del multipliers_eq, multipliers_ineq
        result = self.solve(y_hat, parameters)
        return ProjectionTrainingOutput(
            y_tilde=result.y,
            multipliers_eq=jax.lax.stop_gradient(result.multipliers_eq),
            multipliers_ineq=jax.lax.stop_gradient(result.multipliers_ineq),
            slacks=jax.lax.stop_gradient(result.slacks),
            forward_status_codes=result.status,
            forward_iterations=result.iterations,
            forward_residual_inf_norm=result.residual_inf_norm,
            forward_state_finite=result.finite,
            forward_gradient_eligible=result.gradient_eligible,
        )

    def __call__(self, y_hat, parameters=None):
        values, theta, squeeze = self._normalize(y_hat, parameters)
        projected = self._solve_batch_jit(values, theta)[0]
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


@dataclass(frozen=True)
class CanonicalLinearEqualityProjection:
    """Definition object for affine equalities compiled from the public frontend."""

    canonical: Any
    config: LinearEqualityConfig = LinearEqualityConfig()

    def prepare(self, *, batch_size: int, dtype=jnp.float64, backend: str = "cpu"):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        return PreparedCanonicalLinearEqualityProjection(
            canonical=self.canonical,
            config=self.config,
            batch_size=int(batch_size),
            dtype=dtype,
            backend=str(backend),
        )
