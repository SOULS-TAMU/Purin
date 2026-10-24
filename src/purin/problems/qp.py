"""Structured sparse quadratic-program adapters.

``ParametricQP`` is an adapter, not a competing frontend.  It compiles to the
same :class:`~kkthn.problems.canonical.CanonicalProblem` representation used by
the public :class:`~kkthn.problems.problem.Problem` builder.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

try:
    import scipy.sparse as scipy_sparse
except Exception:  # pragma: no cover
    scipy_sparse = None

from purin.linalg.sparse import as_bcoo
from purin.problems.canonical import CanonicalQP
from purin.problems.symbolic import SymbolicProblem


def _is_sparse(value: Any) -> bool:
    return scipy_sparse is not None and scipy_sparse.issparse(value)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


def _dense_vector(value: Any, size: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros((size,), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    return array


def _matrix(value: Any, shape: tuple[int, int], name: str):
    if value is None:
        if scipy_sparse is not None:
            return scipy_sparse.csr_matrix(shape, dtype=np.float64)
        return np.zeros(shape, dtype=np.float64)
    if _shape(value) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {_shape(value)}")
    if _is_sparse(value):
        return value.astype(np.float64).tocsr()
    return np.asarray(value, dtype=np.float64)


def _jax_matrix(value: Any):
    return as_bcoo(value) if _is_sparse(value) else jnp.asarray(value)


def _iter_row(matrix: Any, row: int):
    if _is_sparse(matrix):
        csr = matrix.tocsr()
        start = csr.indptr[row]
        end = csr.indptr[row + 1]
        return zip(csr.indices[start:end].tolist(), csr.data[start:end].tolist())
    values = np.asarray(matrix[row])
    indices = np.flatnonzero(values)
    return zip(indices.tolist(), values[indices].tolist())


def _row_expression(matrix: Any, row: int, symbols) -> Any:
    return sum(
        (
            float(value) * symbols[int(column)]
            for column, value in _iter_row(matrix, row)
        ),
        0.0,
    )


def _affine_rhs(base, mapping, row: int, parameters) -> Any:
    return float(base[row]) + _row_expression(mapping, row, parameters)


@dataclass(frozen=True)
class ParametricQP:
    """Matrix-form quadratic-program adapter with affine parameters."""

    P: Any
    q0: Any
    q_parameter_map: Any = None
    A_eq: Any = None
    b_eq0: Any = None
    b_eq_parameter_map: Any = None
    A_ineq: Any = None
    b_ineq0: Any = None
    b_ineq_parameter_map: Any = None
    name: str = "parametric_qp"

    def __post_init__(self) -> None:
        if len(_shape(self.P)) != 2 or _shape(self.P)[0] != _shape(self.P)[1]:
            raise ValueError("P must be square")

        variable_size = _shape(self.P)[0]
        q0 = _dense_vector(self.q0, variable_size, "q0")
        parameter_sizes = [
            _shape(value)[1]
            for value in (
                self.q_parameter_map,
                self.b_eq_parameter_map,
                self.b_ineq_parameter_map,
            )
            if value is not None
        ]
        parameter_size = parameter_sizes[0] if parameter_sizes else 0
        if any(size != parameter_size for size in parameter_sizes):
            raise ValueError("parameter maps must share one parameter dimension")

        equality_size = 0 if self.A_eq is None else _shape(self.A_eq)[0]
        inequality_size = 0 if self.A_ineq is None else _shape(self.A_ineq)[0]
        normalized = {
            "P": _matrix(self.P, (variable_size, variable_size), "P"),
            "q0": q0,
            "q_parameter_map": _matrix(
                self.q_parameter_map, (variable_size, parameter_size), "q_parameter_map"
            ),
            "A_eq": _matrix(self.A_eq, (equality_size, variable_size), "A_eq"),
            "b_eq0": _dense_vector(self.b_eq0, equality_size, "b_eq0"),
            "b_eq_parameter_map": _matrix(
                self.b_eq_parameter_map,
                (equality_size, parameter_size),
                "b_eq_parameter_map",
            ),
            "A_ineq": _matrix(self.A_ineq, (inequality_size, variable_size), "A_ineq"),
            "b_ineq0": _dense_vector(self.b_ineq0, inequality_size, "b_ineq0"),
            "b_ineq_parameter_map": _matrix(
                self.b_ineq_parameter_map,
                (inequality_size, parameter_size),
                "b_ineq_parameter_map",
            ),
        }
        for field_name, field_value in normalized.items():
            object.__setattr__(self, field_name, field_value)
        object.__setattr__(self, "_parameter_size", parameter_size)

    @property
    def variable_size(self) -> int:
        """Return the number of primal optimization variables."""
        return int(_shape(self.P)[0])

    @property
    def parameter_size(self) -> int:
        """Return the number of context parameters."""
        return int(self._parameter_size)

    @property
    def equality_size(self) -> int:
        """Return the number of equality constraint rows."""
        return int(_shape(self.A_eq)[0])

    @property
    def inequality_size(self) -> int:
        """Return the number of inequality constraint rows."""
        return int(_shape(self.A_ineq)[0])

    def objective(self, theta, y):
        """Evaluate the parametric quadratic objective at a primal vector."""
        P = _jax_matrix(self.P)
        parameter_map = _jax_matrix(self.q_parameter_map)
        linear_term = jnp.asarray(self.q0) + parameter_map @ theta
        return 0.5 * (y @ (P @ y)) + linear_term @ y

    def equalities(self, theta, y):
        """Evaluate affine equality residuals for the parameterized QP."""
        return (
            _jax_matrix(self.A_eq) @ y
            - jnp.asarray(self.b_eq0)
            - _jax_matrix(self.b_eq_parameter_map) @ theta
        )

    def inequalities(self, theta, y):
        """Evaluate affine inequality residuals for the parameterized QP."""
        return (
            _jax_matrix(self.A_ineq) @ y
            - jnp.asarray(self.b_ineq0)
            - _jax_matrix(self.b_ineq_parameter_map) @ theta
        )

    def compile(self) -> CanonicalQP:
        """Return the backend-neutral canonical QP representation."""

        objective_gradient = jax.grad(self.objective, argnums=1)
        equality_jacobian = jax.jacfwd(self.equalities, argnums=1)
        inequality_jacobian = jax.jacfwd(self.inequalities, argnums=1)

        def lagrangian(theta, y, lambda_eq, mu_ineq):
            value = self.objective(theta, y)
            if self.equality_size:
                value = value + jnp.vdot(lambda_eq, self.equalities(theta, y))
            if self.inequality_size:
                value = value + jnp.vdot(mu_ineq, self.inequalities(theta, y))
            return value

        equality_sparsity = tuple(
            tuple(int(index) for index, _ in _iter_row(self.A_eq, row))
            for row in range(self.equality_size)
        )
        inequality_sparsity = tuple(
            tuple(int(index) for index, _ in _iter_row(self.A_ineq, row))
            for row in range(self.inequality_size)
        )
        return CanonicalQP(
            name=self.name,
            parameter_size=self.parameter_size,
            variable_size=self.variable_size,
            equality_size=self.equality_size,
            inequality_size=self.inequality_size,
            objective=self.objective,
            equalities=self.equalities,
            inequalities=self.inequalities,
            objective_gradient=objective_gradient,
            equality_jacobian=equality_jacobian,
            inequality_jacobian=inequality_jacobian,
            lagrangian_hessian=jax.hessian(lagrangian, argnums=1),
            equality_sparsity=equality_sparsity,
            inequality_sparsity=inequality_sparsity,
            P=self.P,
            q0=self.q0,
            q_parameter_map=self.q_parameter_map,
            A_eq=self.A_eq,
            b_eq0=self.b_eq0,
            b_eq_parameter_map=self.b_eq_parameter_map,
            A_ineq=self.A_ineq,
            b_ineq0=self.b_ineq0,
            b_ineq_parameter_map=self.b_ineq_parameter_map,
            metadata={"name": self.name, "kind": "parametric_qp"},
        )

    def canonicalize(self) -> CanonicalQP:
        """Compatibility alias for :meth:`compile`."""

        return self.compile()

    def to_symbolic_problem(self, *, projection: bool = False):
        """Expose the parametric QP as a symbolic problem definition."""
        model = SymbolicProblem(self.name + ("_projection" if projection else ""))
        target_names = (
            [f"__target_{index}" for index in range(self.variable_size)]
            if projection
            else []
        )
        parameter_names = [f"theta_{index}" for index in range(self.parameter_size)]
        if target_names or parameter_names:
            parameters = model.add_parameter(target_names + parameter_names)
        else:
            parameters = model.parameter
        variables = model.add_variable(
            [f"y_{index}" for index in range(self.variable_size)]
        )
        variable_vector = variables.vector()
        theta = [getattr(parameters, name) for name in parameter_names]

        if projection:
            targets = [getattr(parameters, name) for name in target_names]
            model.set_projection_objective(
                variables=variable_vector,
                target=type(variable_vector)(targets, "target"),
            )
        else:
            expression = 0.0
            for row in range(self.variable_size):
                for column, value in _iter_row(self.P, row):
                    expression = (
                        expression
                        + 0.5
                        * float(value)
                        * variable_vector[row]
                        * variable_vector[column]
                    )
                expression = expression + float(self.q0[row]) * variable_vector[row]
                for parameter, value in _iter_row(self.q_parameter_map, row):
                    expression = (
                        expression
                        + float(value) * theta[parameter] * variable_vector[row]
                    )
            model.objective = expression

        for row in range(self.equality_size):
            model.constraints.add(
                _row_expression(self.A_eq, row, variable_vector)
                == _affine_rhs(self.b_eq0, self.b_eq_parameter_map, row, theta)
            )
        for row in range(self.inequality_size):
            model.constraints.add(
                _row_expression(self.A_ineq, row, variable_vector)
                <= _affine_rhs(self.b_ineq0, self.b_ineq_parameter_map, row, theta)
            )
        return model

    def as_projection_problem(self):
        """Expose the QP through the shared projection problem contract."""
        return self.to_symbolic_problem(projection=True)

    def solve(self, *, parameters=None, **kwargs):
        """Solve the parameterized QP using the selected projection solver."""
        return self.to_symbolic_problem(projection=False).solve(
            parameters=parameters, **kwargs
        )

    @classmethod
    def from_bounds(
        cls,
        *,
        P,
        q0,
        q_parameter_map=None,
        A,
        lower0,
        upper0,
        lower_parameter_map=None,
        upper_parameter_map=None,
        name="parametric_qp",
    ):
        """Construct a parametric QP with the supplied objective and variable bounds."""
        lower0 = np.asarray(lower0, dtype=np.float64)
        upper0 = np.asarray(upper0, dtype=np.float64)
        if lower0.shape != upper0.shape:
            raise ValueError("lower0 and upper0 shapes must match")

        equality_mask = np.isfinite(lower0) & np.isfinite(upper0) & (lower0 == upper0)
        upper_mask = np.isfinite(upper0) & ~equality_mask
        lower_mask = np.isfinite(lower0) & ~equality_mask
        A_matrix = A.tocsr() if _is_sparse(A) else np.asarray(A)

        def subset(matrix, mask):
            return matrix[mask] if _is_sparse(matrix) else np.asarray(matrix)[mask]

        A_eq = subset(A_matrix, equality_mask)
        b_eq = upper0[equality_mask]
        parameter_size = 0
        for candidate in (q_parameter_map, lower_parameter_map, upper_parameter_map):
            if candidate is not None:
                parameter_size = _shape(candidate)[1]
                break
        b_eq_map = (
            np.zeros((equality_mask.sum(), parameter_size))
            if upper_parameter_map is None
            else np.asarray(upper_parameter_map)[equality_mask]
        )

        inequality_rows = []
        inequality_rhs = []
        inequality_maps = []
        if upper_mask.any():
            inequality_rows.append(subset(A_matrix, upper_mask))
            inequality_rhs.append(upper0[upper_mask])
            inequality_maps.append(
                np.zeros((upper_mask.sum(), parameter_size))
                if upper_parameter_map is None
                else np.asarray(upper_parameter_map)[upper_mask]
            )
        if lower_mask.any():
            inequality_rows.append(-subset(A_matrix, lower_mask))
            inequality_rhs.append(-lower0[lower_mask])
            inequality_maps.append(
                np.zeros((lower_mask.sum(), parameter_size))
                if lower_parameter_map is None
                else -np.asarray(lower_parameter_map)[lower_mask]
            )

        if _is_sparse(A_matrix):
            A_ineq = (
                scipy_sparse.vstack(inequality_rows, format="csr")
                if inequality_rows
                else scipy_sparse.csr_matrix((0, _shape(A_matrix)[1]))
            )
        else:
            A_ineq = (
                np.vstack(inequality_rows)
                if inequality_rows
                else np.zeros((0, _shape(A_matrix)[1]))
            )
        b_ineq = np.concatenate(inequality_rhs) if inequality_rhs else np.zeros((0,))
        b_ineq_map = (
            np.vstack(inequality_maps)
            if inequality_maps
            else np.zeros((0, parameter_size))
        )
        return cls(
            P=P,
            q0=q0,
            q_parameter_map=q_parameter_map,
            A_eq=A_eq,
            b_eq0=b_eq,
            b_eq_parameter_map=b_eq_map,
            A_ineq=A_ineq,
            b_ineq0=b_ineq,
            b_ineq_parameter_map=b_ineq_map,
            name=name,
        )


ParametricLP = ParametricQP
MatrixProblem = ParametricQP
