"""High-level symbolic problem-definition frontend.

The public modeling flow mirrors KKT-HardNet's problem-definition API::

    problem = Problem(name="example")
    p = problem.add_parameter(["p1", "p2"])
    y = problem.add_variable(["y1", "y2"])
    problem.objective = 0.5 * (y.y1**2 + y.y2**2)
    problem.constraints.add(y.y1 + y.y2 == p.p1, y.y1 >= 0.0)
    solution = problem.optimize(parameters={"p1": 1.0, "p2": 0.0})

The public ``compile()`` method returns a backend-neutral canonical problem.
Legacy convenience solve methods delegate explicitly to the primal-dual backend.
"""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from purin.execution.runtime import resolve_device as _resolve_jax_device
from purin.linalg.sparse.operators import select_sparse_transpose_mode
from purin._utils.validation import strict_zip
from purin.problems.expressions import Constraint
from purin.problems.expressions import ConstraintList
from purin.problems.expressions import Expression
from purin.problems.expressions import VectorExpression
from purin.problems.expressions import _as_expression
from purin.problems.expressions import _coerce_names
from purin.problems.expressions import _EvaluationContext
from purin.problems.expressions import _SymbolNamespace
from purin.problems.sparsity import ProjectionStructure
from purin.problems.types import BatchNamedValues
from purin.problems.types import NamedValues

ArrayLike = Any
ScalarFunction = Callable[[jnp.ndarray, jnp.ndarray], jax.Array]
VectorFunction = Callable[[jnp.ndarray, jnp.ndarray], jax.Array]

if TYPE_CHECKING:
    from purin.solvers.primal_dual_interior_point.config import FixedMuOptions
    from purin.solvers.primal_dual_interior_point.derivatives import (
        CompiledBatchProblem,
    )
    from purin.solvers.primal_dual_interior_point.solver import (
        PreparedSparseProjectionBatchSolver,
    )
    from purin.solvers.primal_dual_interior_point.types import BatchProblemSolution
    from purin.solvers.primal_dual_interior_point.types import ProblemSolution


class Problem:
    """Symbolic high-level problem builder shared by every backend."""

    def __init__(self, name: str = "problem") -> None:
        self.name = str(name)
        self.parameter_names: list[str] = []
        self.variable_names: list[str] = []
        self.parameter = _SymbolNamespace(self, "parameter")
        self.variable = _SymbolNamespace(self, "variable")
        self.parameters = self.parameter
        self.variables = self.variable
        self.constraints = ConstraintList()
        self._objective: Expression | None = None
        self._projection_target_names: tuple[str, ...] | None = None
        self._compiled_batch_structure_cache: dict[
            tuple[Any, ...], CompiledBatchProblem
        ] = {}
        self._compiled_batch_structure_cache_hits = 0
        self._compiled_batch_structure_cache_misses = 0
        self._prepared_sparse_solver_cache: dict[tuple[Any, ...], Any] = {}
        self._prepared_sparse_solver_cache_hits = 0
        self._prepared_sparse_solver_cache_misses = 0
        # Optional exact row-support patterns restored from a saved artifact.
        # They are validated against the reconstructed symbolic expressions
        # during compilation before any backend uses them.
        self._sparsity_override = None

    @property
    def objective(self) -> Expression | None:
        """Access or replace the symbolic scalar optimization objective."""
        return self._objective

    @objective.setter
    def objective(self, expression: Any) -> None:
        """Access or replace the symbolic scalar optimization objective."""
        symbolic = _as_expression(expression)
        self._objective = symbolic
        self._projection_target_names = symbolic.projection_target_names

    def squared_distance(
        self,
        variables: VectorExpression | _SymbolNamespace,
        target: VectorExpression | _SymbolNamespace,
    ) -> Expression:
        """Return ``0.5 * ||variables - target||_2^2`` with projection metadata.

        The specialized projection solvers require direct decision-variable
        symbols and direct parameter symbols in matching order.
        """
        variable_vector = (
            variables.vector() if isinstance(variables, _SymbolNamespace) else variables
        )
        target_vector = (
            target.vector() if isinstance(target, _SymbolNamespace) else target
        )
        if isinstance(variable_vector, Sequence) and not isinstance(
            variable_vector, VectorExpression
        ):
            variable_vector = VectorExpression(
                [_as_expression(item) for item in variable_vector],
                "projection_variables",
            )
        if isinstance(target_vector, Sequence) and not isinstance(
            target_vector, VectorExpression
        ):
            target_vector = VectorExpression(
                [_as_expression(item) for item in target_vector], "projection_target"
            )
        if not isinstance(variable_vector, VectorExpression) or not isinstance(
            target_vector, VectorExpression
        ):
            raise TypeError(
                "squared_distance expects symbolic vectors, namespaces, or sequences "
                "of scalar symbolic expressions."
            )
        if variable_vector.size != target_vector.size:
            raise ValueError(
                "Projection variables and target parameters must have the same size; "
                f"received {variable_vector.size} and {target_vector.size}."
            )
        if variable_vector.size != len(self.variable_names):
            raise ValueError(
                "The projection objective must include every decision variable exactly once."
            )
        variable_names = []
        target_names = []
        for index, (variable, parameter) in enumerate(
            strict_zip(variable_vector, target_vector)
        ):
            if variable.symbol_kind != "variable" or variable.symbol_name is None:
                raise ValueError(
                    f"Projection component {index} is not a direct decision-variable symbol."
                )
            if parameter.symbol_kind != "parameter" or parameter.symbol_name is None:
                raise ValueError(
                    f"Projection target component {index} is not a direct parameter symbol."
                )
            variable_names.append(variable.symbol_name)
            target_names.append(parameter.symbol_name)
        if tuple(variable_names) != tuple(self.variable_names):
            raise ValueError(
                "Projection variables must follow the declaration order "
                f"{self.variable_names}; received {variable_names}."
            )

        components = [
            variable - parameter
            for variable, parameter in strict_zip(variable_vector, target_vector)
        ]
        expression = 0.5 * sum(
            (component**2 for component in components), _as_expression(0.0)
        )
        expression.projection_target_names = tuple(target_names)
        return expression

    def set_projection_objective(
        self,
        *,
        variables: VectorExpression | _SymbolNamespace,
        target: VectorExpression | _SymbolNamespace,
    ) -> "Problem":
        """Assign the shared identity-Hessian distance objective."""
        self.objective = self.squared_distance(variables, target)
        return self

    def add_parameter(self, names: str | Iterable[str]) -> _SymbolNamespace:
        """Declare a parameter and register it in the problem namespace."""
        for name in _coerce_names(names):
            self._register_symbol(name, "parameter")
        return self.parameter

    def add_variable(self, names: str | Iterable[str]) -> _SymbolNamespace:
        """Declare an optimization variable and its bounds."""
        for name in _coerce_names(names):
            self._register_symbol(name, "variable")
        return self.variable

    def _register_symbol(self, name: str, kind: str) -> None:
        if name in self.parameter_names or name in self.variable_names:
            raise ValueError(f"Duplicate symbol name {name!r}.")
        if kind == "parameter":
            self.parameter_names.append(name)
        elif kind == "variable":
            self.variable_names.append(name)
        else:
            raise ValueError(kind)

    def _check_symbol(self, kind: str, name: str) -> None:
        names = self.parameter_names if kind == "parameter" else self.variable_names
        if name not in names:
            raise AttributeError(f"Unknown {kind} symbol {name!r}.")

    # Common nonlinear expression helpers.
    def _unary(self, expression: Any, function: Callable, name: str) -> Expression:
        symbolic = _as_expression(expression)
        return Expression(
            lambda context, symbolic=symbolic: function(symbolic.evaluate(context)),
            f"{name}({symbolic.text})",
            variable_dependencies=symbolic.variable_dependencies,
            parameter_dependencies=symbolic.parameter_dependencies,
            affine_in_variables=not symbolic.variable_dependencies,
        )

    def sin(self, expression: Any) -> Expression:
        """Construct a symbolic sine expression."""
        return self._unary(expression, jnp.sin, "sin")

    def cos(self, expression: Any) -> Expression:
        """Construct a symbolic cosine expression."""
        return self._unary(expression, jnp.cos, "cos")

    def exp(self, expression: Any) -> Expression:
        """Construct a symbolic exponential expression."""
        return self._unary(expression, jnp.exp, "exp")

    def log(self, expression: Any) -> Expression:
        """Construct a symbolic natural-logarithm expression."""
        return self._unary(expression, jnp.log, "log")

    def sqrt(self, expression: Any) -> Expression:
        """Construct a symbolic square-root expression."""
        return self._unary(expression, jnp.sqrt, "sqrt")

    def abs(self, expression: Any) -> Expression:
        """Construct a symbolic absolute-value expression."""
        return self._unary(expression, jnp.abs, "abs")

    def matrix(self, values: ArrayLike) -> np.ndarray:
        """Return a validated constant matrix for use in symbolic helpers."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                f"matrix(...) expects a two-dimensional array; got {array.shape}."
            )
        return array

    def vector(self, values: ArrayLike) -> np.ndarray:
        """Return a validated constant vector for comparisons or linear maps."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(
                f"vector(...) expects a one-dimensional array; got {array.shape}."
            )
        return array

    def tensor(self, values: ArrayLike) -> np.ndarray:
        """Return a validated rank-three constant tensor."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(
                f"tensor(...) expects a three-dimensional array; got {array.shape}."
            )
        return array

    def lin(
        self, matrix: ArrayLike, vector: VectorExpression | _SymbolNamespace
    ) -> VectorExpression:
        """Construct a linear symbolic expression from coefficients and variables."""
        symbolic_vector = (
            vector.vector() if isinstance(vector, _SymbolNamespace) else vector
        )
        if not isinstance(symbolic_vector, VectorExpression):
            raise TypeError(
                "lin(matrix, vector) expects a symbolic vector or namespace."
            )
        coefficient = np.asarray(matrix, dtype=np.float64)
        if coefficient.ndim == 1:
            coefficient = coefficient.reshape(1, -1)
        if coefficient.ndim != 2 or coefficient.shape[1] != symbolic_vector.size:
            raise ValueError(
                "Coefficient matrix shape must be (rows, vector_size); received "
                f"{coefficient.shape} for vector size {symbolic_vector.size}."
            )
        rows: list[Expression] = []
        for row in coefficient:
            rows.append(
                sum(
                    (
                        float(value) * component
                        for value, component in strict_zip(row, symbolic_vector)
                    ),
                    _as_expression(0.0),
                )
            )
        return VectorExpression(rows, "A @ vector")

    def quad(
        self, matrix: ArrayLike, vector: VectorExpression | _SymbolNamespace
    ) -> Expression:
        """Construct a quadratic symbolic expression from its coefficient resources."""
        symbolic_vector = (
            vector.vector() if isinstance(vector, _SymbolNamespace) else vector
        )
        if not isinstance(symbolic_vector, VectorExpression):
            raise TypeError(
                "quad(matrix, vector) expects a symbolic vector or namespace."
            )
        coefficient = np.asarray(matrix, dtype=np.float64)
        expected = (symbolic_vector.size, symbolic_vector.size)
        if coefficient.shape != expected:
            raise ValueError(
                f"Quadratic matrix must have shape {expected}; got {coefficient.shape}."
            )
        result = _as_expression(0.0)
        for i, left in enumerate(symbolic_vector):
            for j, right in enumerate(symbolic_vector):
                value = coefficient[i, j]
                if value != 0.0:
                    result = result + float(value) * left * right
        return result

    def _build_projection_structure(
        self, *, require_projection_target: bool = True
    ) -> ProjectionStructure | None:
        """Derive variable, residual and derivative sparsity for solver preparation."""
        if self._projection_target_names is None and require_projection_target:
            return None
        if self._projection_target_names is None:
            target_indices = ()
        else:
            if len(self._projection_target_names) != len(self.variable_names):
                raise ValueError(
                    "Projection target dimension must match the decision-variable dimension."
                )
            parameter_index = {
                name: index for index, name in enumerate(self.parameter_names)
            }
            try:
                target_indices = tuple(
                    parameter_index[name] for name in self._projection_target_names
                )
            except KeyError as error:
                raise ValueError(
                    f"Unknown projection target parameter {error.args[0]!r}."
                ) from error

        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        equalities = [
            constraint for constraint in self.constraints if constraint.kind == "eq"
        ]
        inequalities = [
            constraint for constraint in self.constraints if constraint.kind == "ineq"
        ]

        def dependency_indices(constraint: Constraint) -> tuple[int, ...]:
            return tuple(
                sorted(
                    variable_index[name]
                    for name in constraint.residual.variable_dependencies
                )
            )

        equality_dependencies = tuple(dependency_indices(item) for item in equalities)
        inequality_dependencies = tuple(
            dependency_indices(item) for item in inequalities
        )
        affine, condensed_density, density, max_row_nonzeros = (
            _projection_structure_counts(
                self,
                equalities,
                equality_dependencies,
                inequalities,
                inequality_dependencies,
            )
        )

        (
            equality_cols,
            equality_rows,
            equality_support,
            equality_support_mask,
            inequality_cols,
            inequality_rows,
            inequality_support,
            inequality_support_mask,
        ) = _projection_support_arrays(
            equality_dependencies, inequality_dependencies, max_row_nonzeros
        )
        return ProjectionStructure(
            target_parameter_indices=target_indices,
            equality_variable_indices=equality_dependencies,
            inequality_variable_indices=inequality_dependencies,
            equality_jacobian_rows=equality_rows,
            equality_jacobian_cols=equality_cols,
            inequality_jacobian_rows=inequality_rows,
            inequality_jacobian_cols=inequality_cols,
            max_row_nonzeros=max_row_nonzeros,
            equality_support=equality_support,
            equality_support_mask=equality_support_mask,
            inequality_support=inequality_support,
            inequality_support_mask=inequality_support_mask,
            constraints_affine_in_variables=affine,
            estimated_reduced_kkt_density=density,
            estimated_condensed_kkt_density=condensed_density,
        )

    def _build_sparse_projection_derivatives(
        self, structure: ProjectionStructure
    ) -> tuple[Callable, Callable, Callable]:
        """Compile nonzero Jacobian values and local constraint HVPs."""
        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        parameter_index = {
            name: index for index, name in enumerate(self.parameter_names)
        }
        equality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "eq"
        ]
        inequality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "ineq"
        ]

        def make_context(variables, parameters):
            return _EvaluationContext(
                variables=variables,
                parameters=parameters,
                variable_index=variable_index,
                parameter_index=parameter_index,
            )

        def sparse_values(constraints, dependencies, variables, parameters):
            blocks = []
            for constraint, dependency_tuple in strict_zip(constraints, dependencies):
                if not dependency_tuple:
                    continue
                indices = jnp.asarray(dependency_tuple, dtype=jnp.int32)
                local_variables = variables[indices]

                def local_residual(local_values):
                    expanded = variables.at[indices].set(local_values)
                    return constraint.residual.evaluate(
                        make_context(expanded, parameters)
                    )

                blocks.append(jax.grad(local_residual)(local_variables))
            if not blocks:
                return jnp.empty((0,), dtype=variables.dtype)
            return jnp.concatenate(blocks)

        def equality_sparse_values(variables, parameters):
            return sparse_values(
                equality_constraints,
                structure.equality_variable_indices,
                variables,
                parameters,
            )

        def inequality_sparse_values(variables, parameters):
            return sparse_values(
                inequality_constraints,
                structure.inequality_variable_indices,
                variables,
                parameters,
            )

        all_constraints = [*equality_constraints, *inequality_constraints]
        all_dependencies = [
            *structure.equality_variable_indices,
            *structure.inequality_variable_indices,
        ]

        def constraint_hvp(
            variables, parameters, equality_multipliers, inequality_multipliers, vector
        ):
            multipliers = jnp.concatenate(
                (equality_multipliers, inequality_multipliers)
            )
            result = jnp.zeros_like(variables)
            for row, (constraint, dependency_tuple) in enumerate(
                strict_zip(all_constraints, all_dependencies)
            ):
                if constraint.residual.affine_in_variables or not dependency_tuple:
                    continue
                indices = jnp.asarray(dependency_tuple, dtype=jnp.int32)
                local_variables = variables[indices]
                local_vector = vector[indices]

                def local_residual(local_values):
                    expanded = variables.at[indices].set(local_values)
                    return constraint.residual.evaluate(
                        make_context(expanded, parameters)
                    )

                local_hvp = jax.jvp(
                    jax.grad(local_residual), (local_variables,), (local_vector,)
                )[1]
                result = result.at[indices].add(multipliers[row] * local_hvp)
            return result

        return equality_sparse_values, inequality_sparse_values, constraint_hvp

    def _build_local_projection_derivatives(
        self, structure: ProjectionStructure
    ) -> tuple[
        Callable,
        Callable,
        Callable | None,
        Callable | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Compile fixed-width local Jacobian and Hessian blocks.

        Affine groups use the symbolic affine decomposition and therefore do
        not differentiate a dense Jacobian.  Nonlinear rows are differentiated
        only on their local supports, so their Hessian blocks never have shape
        ``(n, n)``.
        """

        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        parameter_index = {
            name: index for index, name in enumerate(self.parameter_names)
        }
        equality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "eq"
        ]
        inequality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "ineq"
        ]
        width = int(structure.max_row_nonzeros)

        def make_context(variables, parameters):
            return _EvaluationContext(
                variables=variables,
                parameters=parameters,
                variable_index=variable_index,
                parameter_index=parameter_index,
            )

        def build_function(constraints, dependencies, padded_support, padded_mask):
            count = len(constraints)
            all_affine = all(
                constraint.residual.affine_in_variables for constraint in constraints
            )
            support_array = jnp.asarray(padded_support, dtype=jnp.int32)
            mask_array = jnp.asarray(padded_mask, dtype=bool)

            if all_affine:
                expressions = [constraint.residual for constraint in constraints]

                metadata_available = all(
                    expression.affine_coefficients is not None
                    and expression.affine_offset is not None
                    for expression in expressions
                )

                if metadata_available:

                    def affine_local_derivatives(variables, parameters):
                        if count == 0:
                            return (
                                jnp.empty((0, width), dtype=variables.dtype),
                                jnp.empty((0, width, width), dtype=variables.dtype),
                            )
                        context = make_context(variables, parameters)
                        rows = []
                        for expression, dependency_tuple in strict_zip(
                            expressions, dependencies
                        ):
                            row = [
                                expression.affine_coefficients[
                                    self.variable_names[index]
                                ].evaluate(context)
                                for index in dependency_tuple
                            ]
                            if row:
                                local = jnp.stack(row)
                            else:
                                local = jnp.empty((0,), dtype=variables.dtype)
                            rows.append(
                                jnp.pad(local, (0, width - len(dependency_tuple)))
                            )
                        local = jnp.stack(rows)
                        hessian = jnp.zeros(
                            (count, width, width), dtype=variables.dtype
                        )
                        return local, hessian

                    def affine_offsets(variables, parameters):
                        if count == 0:
                            return jnp.empty((0,), dtype=variables.dtype)
                        context = make_context(variables, parameters)
                        return jnp.stack(
                            [
                                expression.affine_offset.evaluate(context)
                                for expression in expressions
                            ]
                        )

                    coefficients_are_constant = all(
                        not expression.affine_coefficients[
                            self.variable_names[index]
                        ].parameter_dependencies
                        for expression, dependency_tuple in strict_zip(
                            expressions, dependencies
                        )
                        for index in dependency_tuple
                    )
                    constant_local = None
                    if coefficients_are_constant:
                        zero_context = make_context(
                            jnp.zeros((len(self.variable_names),), dtype=jnp.float64),
                            jnp.zeros((len(self.parameter_names),), dtype=jnp.float64),
                        )
                        constant_local = np.zeros((count, width), dtype=np.float64)
                        for row_index, (expression, dependency_tuple) in enumerate(
                            strict_zip(expressions, dependencies)
                        ):
                            for local_index, variable_position in enumerate(
                                dependency_tuple
                            ):
                                constant_local[row_index, local_index] = float(
                                    np.asarray(
                                        expression.affine_coefficients[
                                            self.variable_names[variable_position]
                                        ].evaluate(zero_context)
                                    )
                                )

                    return affine_local_derivatives, affine_offsets, constant_local

                def vector_function(variables, parameters):
                    if not expressions:
                        return jnp.empty((0,), dtype=variables.dtype)
                    context = make_context(variables, parameters)
                    return jnp.stack(
                        [expression.evaluate(context) for expression in expressions]
                    )

                full_jacobian = jax.jacfwd(vector_function, argnums=0)

                def affine_local_derivatives(variables, parameters):
                    if count == 0:
                        return (
                            jnp.empty((0, width), dtype=variables.dtype),
                            jnp.empty((0, width, width), dtype=variables.dtype),
                        )
                    dense = full_jacobian(variables, parameters)
                    rows = jnp.arange(count, dtype=jnp.int32)[:, None]
                    local = dense[rows, support_array]
                    local = jnp.where(mask_array, local, 0.0)
                    hessian = jnp.zeros((count, width, width), dtype=variables.dtype)
                    return local, hessian

                return affine_local_derivatives, None, None

            def nonlinear_local_derivatives(variables, parameters):
                jacobian_rows = []
                hessian_rows = []
                for constraint, dependency_tuple in strict_zip(
                    constraints, dependencies
                ):
                    if not dependency_tuple:
                        jacobian_rows.append(jnp.zeros((width,), dtype=variables.dtype))
                        hessian_rows.append(
                            jnp.zeros((width, width), dtype=variables.dtype)
                        )
                        continue
                    indices = jnp.asarray(dependency_tuple, dtype=jnp.int32)
                    local_variables = variables[indices]

                    def local_residual(local_values):
                        expanded = variables.at[indices].set(local_values)
                        return constraint.residual.evaluate(
                            make_context(expanded, parameters)
                        )

                    local_gradient = jax.grad(local_residual)(local_variables)
                    if constraint.residual.affine_in_variables:
                        local_hessian = jnp.zeros(
                            (len(dependency_tuple), len(dependency_tuple)),
                            dtype=variables.dtype,
                        )
                    else:
                        local_hessian = jax.hessian(local_residual)(local_variables)
                    padding = width - len(dependency_tuple)
                    jacobian_rows.append(jnp.pad(local_gradient, (0, padding)))
                    hessian_rows.append(
                        jnp.pad(local_hessian, ((0, padding), (0, padding)))
                    )
                if not jacobian_rows:
                    return (
                        jnp.empty((0, width), dtype=variables.dtype),
                        jnp.empty((0, width, width), dtype=variables.dtype),
                    )
                return jnp.stack(jacobian_rows), jnp.stack(hessian_rows)

            return nonlinear_local_derivatives, None, None

        equality_derivatives, equality_offsets, equality_constant = build_function(
            equality_constraints,
            structure.equality_variable_indices,
            structure.equality_support,
            structure.equality_support_mask,
        )
        inequality_derivatives, inequality_offsets, inequality_constant = (
            build_function(
                inequality_constraints,
                structure.inequality_variable_indices,
                structure.inequality_support,
                structure.inequality_support_mask,
            )
        )
        return (
            equality_derivatives,
            inequality_derivatives,
            equality_offsets,
            inequality_offsets,
            equality_constant,
            inequality_constant,
        )

    def _build_functions(self) -> tuple[ScalarFunction, VectorFunction, VectorFunction]:
        """Compile symbolic objective and constraint evaluators with their derivatives."""
        if not self.variable_names:
            raise ValueError("At least one decision variable must be declared.")
        if self._objective is None:
            raise ValueError(
                "An objective must be assigned before compiling or optimizing."
            )

        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        parameter_index = {
            name: index for index, name in enumerate(self.parameter_names)
        }
        equality_expressions = [
            constraint.residual
            for constraint in self.constraints
            if constraint.kind == "eq"
        ]
        inequality_expressions = [
            constraint.residual
            for constraint in self.constraints
            if constraint.kind == "ineq"
        ]
        objective_expression = self._objective
        projection_target_indices = (
            None
            if self._projection_target_names is None
            else jnp.asarray(
                [parameter_index[name] for name in self._projection_target_names],
                dtype=jnp.int32,
            )
        )

        def context(variables, parameters):
            return _EvaluationContext(
                variables=variables,
                parameters=parameters,
                variable_index=variable_index,
                parameter_index=parameter_index,
            )

        def objective_function(variables, parameters):
            if projection_target_indices is not None:
                difference = variables - parameters[projection_target_indices]
                return 0.5 * jnp.vdot(difference, difference)
            return jnp.asarray(
                objective_expression.evaluate(context(variables, parameters))
            )

        def evaluate_vector(expressions: Sequence[Expression], variables, parameters):
            if not expressions:
                return jnp.empty((0,), dtype=variables.dtype)
            current_context = context(variables, parameters)
            return jnp.stack(
                [expression.evaluate(current_context) for expression in expressions]
            )

        def equality_function(variables, parameters):
            return evaluate_vector(equality_expressions, variables, parameters)

        def inequality_function(variables, parameters):
            return evaluate_vector(inequality_expressions, variables, parameters)

        return objective_function, equality_function, inequality_function

    def compile(self):
        """Compile the public frontend into a backend-neutral canonical problem."""

        from purin.problems.compile import compile_symbolic_problem

        return compile_symbolic_problem(self)

    def canonicalize(self):
        """Compatibility alias for :meth:`compile`."""

        return self.compile()

    def set_sparsity_override(self, equality_sparsity, inequality_sparsity):
        """Install exact saved sparse row supports for later compilation.

        The compiler verifies that these supports match the reconstructed
        symbolic dependencies. This preserves the sparse prediction structure
        while preventing stale or incompatible artifacts from being accepted.
        """
        self._sparsity_override = (
            tuple(tuple(int(v) for v in row) for row in equality_sparsity),
            tuple(tuple(int(v) for v in row) for row in inequality_sparsity),
        )
        return self

    def _bind_parameters(
        self, values: Mapping[str, float] | Sequence[float] | None
    ) -> np.ndarray:
        count = len(self.parameter_names)
        if values is None:
            if count:
                raise ValueError(
                    f"Parameter values are required for {self.parameter_names}."
                )
            return np.empty((0,), dtype=np.float64)
        if isinstance(values, Mapping):
            missing = [name for name in self.parameter_names if name not in values]
            extra = [name for name in values if name not in self.parameter_names]
            if missing or extra:
                raise ValueError(
                    f"Parameter binding mismatch; missing={missing}, extra={extra}."
                )
            return np.asarray(
                [values[name] for name in self.parameter_names], dtype=np.float64
            )
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != count:
            raise ValueError(
                f"Expected {count} parameter values; received {vector.size}."
            )
        return vector

    def _bind_initial_values(
        self, values: Mapping[str, float] | Sequence[float] | None
    ) -> np.ndarray:
        count = len(self.variable_names)
        if values is None:
            return np.zeros((count,), dtype=np.float64)
        if isinstance(values, Mapping):
            unknown = [name for name in values if name not in self.variable_names]
            if unknown:
                raise ValueError(f"Unknown initial-value names: {unknown}.")
            return np.asarray(
                [values.get(name, 0.0) for name in self.variable_names],
                dtype=np.float64,
            )
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != count:
            raise ValueError(
                f"Expected {count} initial values; received {vector.size}."
            )
        return vector

    def _bind_parameter_batch(
        self, values: Any, *, batch_size: int | None = None
    ) -> np.ndarray:
        """Normalize parameter batches to ``(batch, n_parameter)``."""
        count = len(self.parameter_names)
        requested = None if batch_size is None else int(batch_size)
        if requested is not None and requested <= 0:
            raise ValueError("batch_size must be positive.")

        if values is None:
            if count:
                raise ValueError(
                    f"Batched parameter values are required for {self.parameter_names}."
                )
            size = requested or 1
            return np.empty((size, 0), dtype=np.float64)

        if isinstance(values, Sequence) and not isinstance(
            values, (str, bytes, np.ndarray)
        ):
            rows = list(values)
            if rows and isinstance(rows[0], Mapping):
                matrix = np.stack([self._bind_parameters(row) for row in rows], axis=0)
                if requested is not None and matrix.shape[0] != requested:
                    raise ValueError(
                        f"Expected batch_size={requested}; received {matrix.shape[0]} rows."
                    )
                return matrix

        if isinstance(values, Mapping):
            missing = [name for name in self.parameter_names if name not in values]
            extra = [name for name in values if name not in self.parameter_names]
            if missing or extra:
                raise ValueError(
                    f"Parameter binding mismatch; missing={missing}, extra={extra}."
                )
            columns: list[np.ndarray] = []
            inferred_sizes: set[int] = set()
            for name in self.parameter_names:
                column = np.asarray(values[name], dtype=np.float64)
                if column.ndim == 0:
                    columns.append(column.reshape(1))
                elif column.ndim == 1:
                    columns.append(column)
                    if column.size > 1:
                        inferred_sizes.add(int(column.size))
                else:
                    raise ValueError(
                        f"Batched parameter {name!r} must be scalar or one-dimensional; "
                        f"received {column.shape}."
                    )
            if len(inferred_sizes) > 1:
                raise ValueError(
                    "All non-scalar parameter columns must have the same batch length; "
                    f"received {sorted(inferred_sizes)}."
                )
            size = requested or (next(iter(inferred_sizes)) if inferred_sizes else 1)
            broadcasted = []
            for name, column in strict_zip(self.parameter_names, columns):
                if column.size == 1:
                    broadcasted.append(
                        np.full((size,), float(column[0]), dtype=np.float64)
                    )
                elif column.size == size:
                    broadcasted.append(column.astype(np.float64, copy=False))
                else:
                    raise ValueError(
                        f"Parameter column {name!r} has length {column.size}; expected {size}."
                    )
            if not broadcasted:
                return np.empty((size, 0), dtype=np.float64)
            return np.stack(broadcasted, axis=1)

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            if count == 1:
                array = array.reshape(-1, 1)
            elif array.size == count:
                array = array.reshape(1, count)
            else:
                raise ValueError(
                    f"Expected shape (batch, {count}); received {array.shape}."
                )
        if array.ndim != 2 or array.shape[1] != count:
            raise ValueError(
                f"Expected parameter batch shape (batch, {count}); received {array.shape}."
            )
        if requested is not None:
            if array.shape[0] == 1 and requested > 1:
                array = np.broadcast_to(array, (requested, count)).copy()
            elif array.shape[0] != requested:
                raise ValueError(
                    f"Expected batch_size={requested}; received {array.shape[0]} rows."
                )
        return array

    def _bind_initial_value_batch(self, values: Any, batch_size: int) -> np.ndarray:
        """Normalize initial decision values to ``(batch, n_variable)``."""
        count = len(self.variable_names)
        if values is None:
            return np.zeros((batch_size, count), dtype=np.float64)

        if isinstance(values, Sequence) and not isinstance(
            values, (str, bytes, np.ndarray)
        ):
            rows = list(values)
            if rows and isinstance(rows[0], Mapping):
                matrix = np.stack(
                    [self._bind_initial_values(row) for row in rows], axis=0
                )
                if matrix.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} initial-value rows; received {matrix.shape[0]}."
                    )
                return matrix

        if isinstance(values, Mapping):
            unknown = [name for name in values if name not in self.variable_names]
            if unknown:
                raise ValueError(f"Unknown initial-value names: {unknown}.")
            columns = []
            for name in self.variable_names:
                column = np.asarray(values.get(name, 0.0), dtype=np.float64)
                if column.ndim == 0:
                    columns.append(
                        np.full((batch_size,), float(column), dtype=np.float64)
                    )
                elif column.ndim == 1 and column.size == batch_size:
                    columns.append(column)
                elif column.ndim == 1 and column.size == 1:
                    columns.append(
                        np.full((batch_size,), float(column[0]), dtype=np.float64)
                    )
                else:
                    raise ValueError(
                        f"Initial-value column {name!r} must be scalar or length {batch_size}; "
                        f"received {column.shape}."
                    )
            return np.stack(columns, axis=1)

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            if count == 1 and array.size == batch_size:
                array = array.reshape(batch_size, 1)
            elif array.size == count:
                array = np.broadcast_to(
                    array.reshape(1, count), (batch_size, count)
                ).copy()
            else:
                raise ValueError(
                    f"Expected one initial vector of length {count} or shape "
                    f"({batch_size}, {count}); received {array.shape}."
                )
        if array.ndim != 2 or array.shape != (batch_size, count):
            raise ValueError(
                f"Expected initial-value batch shape ({batch_size}, {count}); "
                f"received {array.shape}."
            )
        return array

    def compile_batch(
        self,
        *,
        parameters: Any,
        initial_values: Any = None,
        batch_size: int | None = None,
        interior_margin: float = 1.0,
        device: Any | None = None,
    ) -> CompiledBatchProblem:
        """Compile a same-structure parameter batch for vectorized JAX solves."""
        from purin.solvers.primal_dual_interior_point.derivatives import (
            CompiledBatchProblem,
        )
        from purin.solvers.primal_dual_interior_point.types import BoundIndices
        from purin.solvers.primal_dual_interior_point.types import BoundValues
        from purin.solvers.primal_dual_interior_point.types import (
            NLPQuantities as JaxNLPQuantities,
        )
        from purin.solvers.primal_dual_interior_point.types import (
            PrimalDualState as JaxPrimalDualState,
        )

        selected_device = _resolve_jax_device(device)
        parameter_values = self._bind_parameter_batch(parameters, batch_size=batch_size)
        batch_count = int(parameter_values.shape[0])
        variable_values = self._bind_initial_value_batch(initial_values, batch_count)
        margin = float(interior_margin)
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                f"interior_margin must be finite and positive; got {margin}."
            )

        model_signature = (
            tuple(self.parameter_names),
            tuple(self.variable_names),
            None if self._objective is None else self._objective.text,
            tuple((item.kind, item.text) for item in self.constraints),
        )
        device_key = (
            (selected_device.platform, selected_device.id)
            if selected_device is not None
            else ("default", -1)
        )
        structure_cache_key = (model_signature, batch_count, device_key)
        cached = self._compiled_batch_structure_cache.get(structure_cache_key)
        if cached is not None:
            projection_structure = cached.projection_structure
            if projection_structure is not None and initial_values is None:
                variable_values = parameter_values[
                    :,
                    np.asarray(
                        projection_structure.target_parameter_indices, dtype=np.int64
                    ),
                ].copy()
            variable_jax = jnp.asarray(variable_values, dtype=jnp.float64)
            parameter_jax = jnp.asarray(parameter_values, dtype=jnp.float64)
            margin_jax = jnp.asarray(margin, dtype=jnp.float64)
            if selected_device is not None:
                variable_jax = jax.device_put(variable_jax, selected_device)
                parameter_jax = jax.device_put(parameter_jax, selected_device)
                margin_jax = jax.device_put(margin_jax, selected_device)
            initial_state = cached.initialize_state_function(
                variable_jax, parameter_jax, margin_jax
            )
            self._compiled_batch_structure_cache_hits += 1
            return replace(
                cached, parameter_values=parameter_values, initial_state=initial_state
            )

        self._compiled_batch_structure_cache_misses += 1
        objective_function, equality_function, inequality_function = (
            self._build_functions()
        )
        projection_structure = self._build_projection_structure()
        if projection_structure is not None and initial_values is None:
            variable_values = parameter_values[
                :,
                np.asarray(
                    projection_structure.target_parameter_indices, dtype=np.int64
                ),
            ].copy()
        if projection_structure is None:
            equality_sparse_values = None
            inequality_sparse_values = None
            constraint_hvp = None
            equality_local_derivatives = None
            inequality_local_derivatives = None
            equality_affine_offsets = None
            inequality_affine_offsets = None
            equality_constant_local_jacobian = None
            inequality_constant_local_jacobian = None
        else:
            (equality_sparse_values, inequality_sparse_values, constraint_hvp) = (
                self._build_sparse_projection_derivatives(projection_structure)
            )
            (
                equality_local_derivatives,
                inequality_local_derivatives,
                equality_affine_offsets,
                inequality_affine_offsets,
                equality_constant_local_jacobian,
                inequality_constant_local_jacobian,
            ) = self._build_local_projection_derivatives(projection_structure)
        equality_jacobian = jax.jacobian(equality_function, argnums=0)
        inequality_jacobian = jax.jacobian(inequality_function, argnums=0)

        if projection_structure is None:
            objective_gradient = jax.grad(objective_function, argnums=0)

            def lagrangian(
                variables, parameters_, equality_multipliers, inequality_multipliers
            ):
                return (
                    objective_function(variables, parameters_)
                    + jnp.vdot(
                        equality_multipliers, equality_function(variables, parameters_)
                    )
                    + jnp.vdot(
                        inequality_multipliers,
                        inequality_function(variables, parameters_),
                    )
                )

            lagrangian_hessian = jax.hessian(lagrangian, argnums=0)
        else:
            target_indices = jnp.asarray(
                projection_structure.target_parameter_indices, dtype=jnp.int32
            )

            def objective_gradient(variables, parameters_):
                return variables - parameters_[target_indices]

            def constraint_lagrangian(
                variables, parameters_, equality_multipliers, inequality_multipliers
            ):
                return jnp.vdot(
                    equality_multipliers, equality_function(variables, parameters_)
                ) + jnp.vdot(
                    inequality_multipliers, inequality_function(variables, parameters_)
                )

            constraint_hessian = jax.hessian(constraint_lagrangian, argnums=0)

            def lagrangian_hessian(
                variables, parameters_, equality_multipliers, inequality_multipliers
            ):
                return jnp.eye(
                    variables.shape[0], dtype=variables.dtype
                ) + constraint_hessian(
                    variables, parameters_, equality_multipliers, inequality_multipliers
                )

        def evaluate_nlp_single(
            state: JaxPrimalDualState, parameter_vector: jax.Array
        ) -> JaxNLPQuantities:
            variables = state.x
            return JaxNLPQuantities(
                grad_f=objective_gradient(variables, parameter_vector),
                c=equality_function(variables, parameter_vector),
                d=inequality_function(variables, parameter_vector),
                jac_c=equality_jacobian(variables, parameter_vector),
                jac_d=inequality_jacobian(variables, parameter_vector),
                hessian_lagrangian=lagrangian_hessian(
                    variables, parameter_vector, state.y_c, state.y_d
                ),
                objective_value=objective_function(variables, parameter_vector),
            )

        variable_jax = jnp.asarray(variable_values, dtype=jnp.float64)
        parameter_jax = jnp.asarray(parameter_values, dtype=jnp.float64)
        margin_jax = jnp.asarray(margin, dtype=jnp.float64)
        if selected_device is not None:
            variable_jax = jax.device_put(variable_jax, selected_device)
            parameter_jax = jax.device_put(parameter_jax, selected_device)
            margin_jax = jax.device_put(margin_jax, selected_device)
        n_equalities = sum(item.kind == "eq" for item in self.constraints)
        n_inequalities = sum(item.kind == "ineq" for item in self.constraints)

        empty_float = np.empty((0,), dtype=np.float64)
        empty_index = np.empty((0,), dtype=np.int64)
        numpy_indices = BoundIndices(
            x_lower=empty_index.copy(),
            x_upper=empty_index.copy(),
            s_lower=empty_index.copy(),
            s_upper=np.arange(n_inequalities, dtype=np.int64),
        )
        numpy_bounds = BoundValues(
            x_lower=empty_float.copy(),
            x_upper=empty_float.copy(),
            s_lower=empty_float.copy(),
            s_upper=np.zeros((n_inequalities,), dtype=np.float64),
        )

        equality_batch_function = jax.jit(jax.vmap(equality_function))
        inequality_batch_function = jax.jit(jax.vmap(inequality_function))
        objective_batch_function = jax.jit(jax.vmap(objective_function))

        def initialize_state_function(variable_batch, parameter_batch, margin_value):
            inequality_initial = inequality_batch_function(
                variable_batch, parameter_batch
            )
            slack_initial = jnp.minimum(inequality_initial, -margin_value)
            return JaxPrimalDualState(
                x=variable_batch,
                s=slack_initial,
                y_c=jnp.zeros(
                    (variable_batch.shape[0], n_equalities), dtype=variable_batch.dtype
                ),
                y_d=jnp.ones(
                    (variable_batch.shape[0], n_inequalities),
                    dtype=variable_batch.dtype,
                ),
                z_L=jnp.empty((variable_batch.shape[0], 0), dtype=variable_batch.dtype),
                z_U=jnp.empty((variable_batch.shape[0], 0), dtype=variable_batch.dtype),
                v_L=jnp.empty((variable_batch.shape[0], 0), dtype=variable_batch.dtype),
                v_U=jnp.ones(
                    (variable_batch.shape[0], n_inequalities),
                    dtype=variable_batch.dtype,
                ),
            )

        initialize_state_function = jax.jit(initialize_state_function)
        initial_state = initialize_state_function(
            variable_jax, parameter_jax, margin_jax
        )

        compiled_indices = BoundIndices(
            *(jnp.asarray(block, dtype=jnp.int32) for block in numpy_indices)
        )
        compiled_bounds = BoundValues(
            *(jnp.asarray(block, dtype=jnp.float64) for block in numpy_bounds)
        )
        if selected_device is not None:
            compiled_indices = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), compiled_indices
            )
            compiled_bounds = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), compiled_bounds
            )

        compiled = CompiledBatchProblem(
            name=self.name,
            variable_names=tuple(self.variable_names),
            parameter_names=tuple(self.parameter_names),
            parameter_values=parameter_values,
            initial_state=initial_state,
            indices=compiled_indices,
            bounds=compiled_bounds,
            evaluate_nlp_single=evaluate_nlp_single,
            objective_function=objective_function,
            equality_function=equality_function,
            inequality_function=inequality_function,
            initialize_state_function=initialize_state_function,
            objective_batch_function=objective_batch_function,
            equality_batch_function=equality_batch_function,
            inequality_batch_function=inequality_batch_function,
            equality_jacobian_function=equality_jacobian,
            inequality_jacobian_function=inequality_jacobian,
            equality_sparse_jacobian_values_function=equality_sparse_values,
            inequality_sparse_jacobian_values_function=inequality_sparse_values,
            constraint_hvp_function=constraint_hvp,
            equality_local_derivatives_function=equality_local_derivatives,
            inequality_local_derivatives_function=inequality_local_derivatives,
            equality_affine_offset_function=equality_affine_offsets,
            inequality_affine_offset_function=inequality_affine_offsets,
            equality_constant_local_jacobian=equality_constant_local_jacobian,
            inequality_constant_local_jacobian=inequality_constant_local_jacobian,
            projection_structure=projection_structure,
        )
        self._compiled_batch_structure_cache[structure_cache_key] = compiled
        return compiled

    def prepare_sparse_projection_batch_solver(
        self,
        *,
        parameters: Any,
        batch_size: int | None = None,
        options: FixedMuOptions | None = None,
        interior_margin: float = 1.0,
        gmres_tolerance: float = 1e-9,
        gmres_atol: float = 0.0,
        gmres_restart: int = 30,
        gmres_maxiter: int = 50,
        use_preconditioner: bool = True,
        backtracking: bool = False,
        max_backtracking_steps: int = 8,
        backtracking_factor: float = 0.5,
        backend: Any | None = None,
        device: Any | None = None,
        transpose_mode: str = "auto",
        gmres_unroll: int | None = None,
        arnoldi_prefix_block: int | None = None,
        gpu_kkt_mode: str = "auto",
        pallas_kkt_block_size: int = 64,
        pallas_interpret: bool = False,
        store_residual_history: bool = True,
        fuse_initialization: bool | None = None,
        adjoint_gmres_tolerance: float | None = None,
        adjoint_gmres_atol: float | None = None,
        adjoint_gmres_restart: int | None = None,
        adjoint_gmres_maxiter: int | None = None,
        adjoint_gmres_unroll: int | None = None,
        adjoint_arnoldi_prefix_block: int | None = None,
        adjoint_solver_mode: str = "auto",
        adjoint_dense_threshold: int = 64,
    ) -> PreparedSparseProjectionBatchSolver:
        """Prepare one fixed-shape sparse projection solver on a device.

        This API is intended for repeated CPU/CUDA batches.  It avoids the
        per-batch high-level compilation/binding path and keeps inputs and
        outputs device-resident until the caller explicitly uses
        :func:`jax.device_get`.
        """
        from purin.solvers.primal_dual_interior_point.config import FixedMuOptions
        from purin.solvers.primal_dual_interior_point.solver import (
            prepare_batched_projection_condensed_sparse_solver,
        )
        from purin.solvers.primal_dual_interior_point.types import (
            NewtonRegularization as JaxNewtonRegularization,
        )

        if backend is not None and device is not None:
            backend_device = _resolve_jax_device(backend)
            explicit_device = _resolve_jax_device(device)
            if (backend_device.platform, backend_device.id) != (
                explicit_device.platform,
                explicit_device.id,
            ):
                raise ValueError("backend and device select different JAX devices.")
            selected_device = backend_device
        else:
            selected_device = _resolve_jax_device(
                backend if backend is not None else device
            )
        if selected_device is None:
            selected_device = jax.devices()[0]

        mode = str(transpose_mode).strip().lower()
        if mode not in {"auto", "scatter", "gather", "bucketed"}:
            raise ValueError(
                "transpose_mode must be one of 'auto', 'scatter', 'gather', or 'bucketed'."
            )
        requested_gpu_kkt_mode = str(gpu_kkt_mode).strip().lower()
        if requested_gpu_kkt_mode not in {"auto", "xla", "pallas"}:
            raise ValueError("gpu_kkt_mode must be one of 'auto', 'xla', or 'pallas'.")
        if gmres_unroll is None:
            gmres_unroll = 4 if selected_device.platform == "gpu" else 1
        if not isinstance(gmres_unroll, int) or gmres_unroll < 1:
            raise ValueError("gmres_unroll must be a positive integer or None.")
        if arnoldi_prefix_block is None:
            arnoldi_prefix_block = 8  # if selected_device.platform == "gpu" else 0
        if not isinstance(arnoldi_prefix_block, int) or arnoldi_prefix_block < 0:
            raise ValueError(
                "arnoldi_prefix_block must be a nonnegative integer or None."
            )
        if not isinstance(pallas_kkt_block_size, int) or pallas_kkt_block_size < 1:
            raise ValueError("pallas_kkt_block_size must be a positive integer.")
        if fuse_initialization is None:
            fuse_initialization = selected_device.platform == "gpu"

        compiled = self.compile_batch(
            parameters=parameters,
            batch_size=batch_size,
            interior_margin=interior_margin,
            device=selected_device,
        )
        structure = compiled.projection_structure
        if structure is None:
            raise ValueError(
                "prepare_sparse_projection_batch_solver requires an objective "
                "created with set_projection_objective(...)."
            )
        if (
            compiled.equality_local_derivatives_function is None
            or compiled.inequality_local_derivatives_function is None
        ):
            raise RuntimeError("Sparse local derivative functions were not prepared.")

        selected_transpose_mode = select_sparse_transpose_mode(
            structure.equality_support,
            structure.equality_support_mask,
            structure.inequality_support,
            structure.inequality_support_mask,
            len(self.variable_names),
            backend=selected_device.platform,
            device_kind=str(getattr(selected_device, "device_kind", "")),
            mode=mode,
        )
        use_column_gather_transpose = selected_transpose_mode == "gather"
        use_column_bucketed_transpose = selected_transpose_mode == "bucketed"

        pallas_compatible = (
            selected_device.platform == "gpu"
            and structure.constraints_affine_in_variables
            and compiled.equality_constant_local_jacobian is not None
            and compiled.inequality_constant_local_jacobian is not None
        )
        if requested_gpu_kkt_mode == "auto":
            # Keep the proven XLA path as the portable default.  The Pallas
            # kernel is available explicitly for GPU benchmarking and can be
            # promoted after device-specific parity/performance validation.
            selected_gpu_kkt_mode = "xla"
        else:
            selected_gpu_kkt_mode = requested_gpu_kkt_mode
        if selected_gpu_kkt_mode == "pallas" and not pallas_compatible:
            raise ValueError(
                "gpu_kkt_mode='pallas' requires a CUDA device and affine "
                "constraints with a parameter-independent Jacobian."
            )

        opts = options or FixedMuOptions(line_search=None)
        if opts.line_search is not None:
            raise ValueError(
                "Prepared batched sparse solves require FixedMuOptions(line_search=None)."
            )
        if opts.regularization is None:
            regularization = JaxNewtonRegularization()
        else:
            regularization = JaxNewtonRegularization(
                delta_x=opts.regularization.delta_x,
                delta_s=opts.regularization.delta_s,
                delta_c=opts.regularization.delta_c,
                delta_d=opts.regularization.delta_d,
            )

        if adjoint_solver_mode not in {"auto", "gmres", "dense"}:
            raise ValueError("adjoint_solver_mode must be auto, gmres, or dense.")
        if not isinstance(adjoint_dense_threshold, int) or adjoint_dense_threshold < 1:
            raise ValueError("adjoint_dense_threshold must be positive.")
        condensed_width = (
            len(self.variable_names)
            + len(structure.equality_support)
            + len(structure.inequality_support)
        )
        if adjoint_solver_mode == "auto":
            selected_adjoint_solver_mode = (
                "dense"
                if selected_device.platform == "cpu"
                and condensed_width <= adjoint_dense_threshold
                and compiled.batch_size <= 32
                else "gmres"
            )
        else:
            selected_adjoint_solver_mode = adjoint_solver_mode
        selected_adjoint_gmres_tolerance = (
            gmres_tolerance
            if adjoint_gmres_tolerance is None
            else float(adjoint_gmres_tolerance)
        )
        selected_adjoint_gmres_atol = (
            gmres_atol if adjoint_gmres_atol is None else float(adjoint_gmres_atol)
        )
        selected_adjoint_gmres_restart = (
            gmres_restart
            if adjoint_gmres_restart is None
            else int(adjoint_gmres_restart)
        )
        selected_adjoint_gmres_maxiter = (
            gmres_maxiter
            if adjoint_gmres_maxiter is None
            else int(adjoint_gmres_maxiter)
        )
        selected_adjoint_gmres_unroll = (
            gmres_unroll if adjoint_gmres_unroll is None else int(adjoint_gmres_unroll)
        )
        selected_adjoint_prefix = (
            arnoldi_prefix_block
            if adjoint_arnoldi_prefix_block is None
            else int(adjoint_arnoldi_prefix_block)
        )

        indices = jax.tree_util.tree_map(
            lambda value: jax.device_put(value, selected_device), compiled.indices
        )
        bounds = jax.tree_util.tree_map(
            lambda value: jax.device_put(value, selected_device), compiled.bounds
        )
        model_signature = (
            tuple(self.parameter_names),
            tuple(self.variable_names),
            None if self._objective is None else self._objective.text,
            tuple((item.kind, item.text) for item in self.constraints),
        )
        device_key = (selected_device.platform, selected_device.id)
        sparse_cache_key = (
            model_signature,
            int(compiled.batch_size),
            device_key,
            int(opts.max_iterations),
            float(opts.residual_tolerance),
            float(opts.tau),
            float(opts.step_scale),
            float(opts.minimum_step),
            tuple(float(value) for value in regularization),
            float(gmres_tolerance),
            float(gmres_atol),
            int(gmres_restart),
            int(gmres_maxiter),
            bool(use_preconditioner),
            bool(backtracking),
            int(max_backtracking_steps),
            float(backtracking_factor),
            str(selected_transpose_mode),
            bool(selected_device.platform == "gpu"),
            int(gmres_unroll),
            int(arnoldi_prefix_block),
            str(selected_gpu_kkt_mode),
            int(pallas_kkt_block_size),
            bool(pallas_interpret),
            bool(store_residual_history),
            str(selected_adjoint_solver_mode),
            float(selected_adjoint_gmres_tolerance),
            float(selected_adjoint_gmres_atol),
            int(selected_adjoint_gmres_restart),
            int(selected_adjoint_gmres_maxiter),
            int(selected_adjoint_gmres_unroll),
            int(selected_adjoint_prefix),
        )
        sparse_solver = self._prepared_sparse_solver_cache.get(sparse_cache_key)
        if sparse_solver is None:
            with jax.default_device(selected_device):
                sparse_solver = prepare_batched_projection_condensed_sparse_solver(
                    batch_size=int(compiled.batch_size),
                    equality_function=compiled.equality_function,
                    inequality_function=compiled.inequality_function,
                    equality_local_derivatives=compiled.equality_local_derivatives_function,
                    inequality_local_derivatives=compiled.inequality_local_derivatives_function,
                    equality_affine_offsets=compiled.equality_affine_offset_function,
                    inequality_affine_offsets=compiled.inequality_affine_offset_function,
                    equality_constant_local_jacobian=compiled.equality_constant_local_jacobian,
                    inequality_constant_local_jacobian=compiled.inequality_constant_local_jacobian,
                    equality_support=np.asarray(
                        structure.equality_support, dtype=np.int32
                    ),
                    equality_mask=np.asarray(
                        structure.equality_support_mask, dtype=bool
                    ),
                    inequality_support=np.asarray(
                        structure.inequality_support, dtype=np.int32
                    ),
                    inequality_mask=np.asarray(
                        structure.inequality_support_mask, dtype=bool
                    ),
                    target_parameter_indices=np.asarray(
                        structure.target_parameter_indices, dtype=np.int32
                    ),
                    indices=indices,
                    bounds=bounds,
                    max_iterations=opts.max_iterations,
                    residual_tolerance=opts.residual_tolerance,
                    tau=opts.tau,
                    step_scale=opts.step_scale,
                    minimum_step=opts.minimum_step,
                    regularization=regularization,
                    gmres_tolerance=gmres_tolerance,
                    gmres_atol=gmres_atol,
                    gmres_restart=gmres_restart,
                    gmres_maxiter=gmres_maxiter,
                    use_preconditioner=use_preconditioner,
                    backtracking=backtracking,
                    max_backtracking_steps=max_backtracking_steps,
                    backtracking_factor=backtracking_factor,
                    constraints_affine=structure.constraints_affine_in_variables,
                    use_column_gather_transpose=use_column_gather_transpose,
                    use_column_bucketed_transpose=use_column_bucketed_transpose,
                    use_shared_constant_jacobian=(selected_device.platform == "gpu"),
                    gmres_unroll=gmres_unroll,
                    arnoldi_prefix_block=arnoldi_prefix_block,
                    gpu_kkt_mode=selected_gpu_kkt_mode,
                    pallas_kkt_block_size=pallas_kkt_block_size,
                    pallas_interpret=pallas_interpret,
                    store_residual_history=store_residual_history,
                    adjoint_gmres_tolerance=selected_adjoint_gmres_tolerance,
                    adjoint_gmres_atol=selected_adjoint_gmres_atol,
                    adjoint_gmres_restart=selected_adjoint_gmres_restart,
                    adjoint_gmres_maxiter=selected_adjoint_gmres_maxiter,
                    adjoint_gmres_unroll=selected_adjoint_gmres_unroll,
                    adjoint_arnoldi_prefix_block=selected_adjoint_prefix,
                    adjoint_solver_mode=selected_adjoint_solver_mode,
                )
            self._prepared_sparse_solver_cache[sparse_cache_key] = sparse_solver
            self._prepared_sparse_solver_cache_misses += 1
        else:
            self._prepared_sparse_solver_cache_hits += 1

        target_parameter_indices = jax.device_put(
            jnp.asarray(structure.target_parameter_indices, dtype=jnp.int32),
            selected_device,
        )
        target_index_host = np.asarray(
            structure.target_parameter_indices, dtype=np.int32
        )
        physical_index_host = np.asarray(
            [
                index
                for index in range(len(self.parameter_names))
                if index not in set(target_index_host.tolist())
            ],
            dtype=np.int32,
        )
        physical_parameter_indices = jax.device_put(
            jnp.asarray(physical_index_host, dtype=jnp.int32), selected_device
        )
        solve_default_function = None
        solve_default_with_linearization_function = None
        solve_default_loss_gradient_functions = None
        if fuse_initialization:
            margin = jax.device_put(
                jnp.asarray(interior_margin, dtype=jnp.float64), selected_device
            )

            def solve_from_parameters(parameter_array, mu_values):
                variable_array = parameter_array[:, target_parameter_indices]
                initial_state = compiled.initialize_state_function(
                    variable_array, parameter_array, margin
                )
                return sparse_solver._solve_all_function(
                    initial_state, parameter_array, mu_values
                )

            def solve_from_parameters_with_linearization(parameter_array, mu_values):
                variable_array = parameter_array[:, target_parameter_indices]
                initial_state = compiled.initialize_state_function(
                    variable_array, parameter_array, margin
                )
                return sparse_solver._solve_all_with_linearization_function(
                    initial_state, parameter_array, mu_values
                )

            def make_default_loss_gradient(q_mode):
                def solve_from_parameters_loss_gradient(
                    parameter_array, mu_values, q_data, valid_mask, initial_adjoint
                ):
                    variable_array = parameter_array[:, target_parameter_indices]
                    initial_state = compiled.initialize_state_function(
                        variable_array, parameter_array, margin
                    )
                    return sparse_solver.solve_loss_yhat_gradient(
                        initial_state=initial_state,
                        parameter_values=parameter_array,
                        mu_values=mu_values,
                        q_data=q_data,
                        valid_mask=valid_mask,
                        q_mode=q_mode,
                        initial_adjoint=initial_adjoint,
                    )

                return solve_from_parameters_loss_gradient

            with jax.default_device(selected_device):
                solve_default_function = jax.jit(solve_from_parameters)
                solve_default_with_linearization_function = jax.jit(
                    solve_from_parameters_with_linearization
                )
                solve_default_loss_gradient_functions = {
                    "diagonal": jax.jit(make_default_loss_gradient("diagonal")),
                    "dense": jax.jit(make_default_loss_gradient("dense")),
                }

        from purin.solvers.primal_dual_interior_point.solver import (
            PreparedSparseProjectionBatchSolver,
        )

        return PreparedSparseProjectionBatchSolver(
            compiled=compiled,
            sparse_solver=sparse_solver,
            device=selected_device,
            interior_margin=float(interior_margin),
            target_parameter_indices=target_parameter_indices,
            physical_parameter_indices=physical_parameter_indices,
            batch_size=int(compiled.batch_size),
            parameter_count=len(self.parameter_names),
            transpose_mode=selected_transpose_mode,
            gmres_unroll=int(gmres_unroll),
            arnoldi_prefix_block=int(arnoldi_prefix_block),
            gpu_kkt_mode=selected_gpu_kkt_mode,
            pallas_kkt_block_size=int(pallas_kkt_block_size),
            store_residual_history=bool(store_residual_history),
            adjoint_solver_mode=selected_adjoint_solver_mode,
            adjoint_gmres_restart=int(selected_adjoint_gmres_restart),
            adjoint_gmres_maxiter=int(selected_adjoint_gmres_maxiter),
            adjoint_gmres_unroll=int(selected_adjoint_gmres_unroll),
            adjoint_arnoldi_prefix_block=int(selected_adjoint_prefix),
            solve_default_function=solve_default_function,
            solve_default_with_linearization_function=solve_default_with_linearization_function,
            solve_default_loss_gradient_functions=solve_default_loss_gradient_functions,
        )

    def solve_batch(
        self,
        *,
        parameters: Any,
        initial_values: Any = None,
        batch_size: int | None = None,
        mu: float | Sequence[float] = 1e-4,
        options: FixedMuOptions | None = None,
        interior_margin: float = 1.0,
        projection_mode: str = "auto",
        sparse_density_threshold: float = 0.12,
        dense_dimension_threshold: int = 192,
        gmres_tolerance: float = 1e-9,
        gmres_atol: float = 0.0,
        gmres_restart: int = 30,
        gmres_maxiter: int = 50,
        use_preconditioner: bool = True,
        backtracking: bool = False,
        max_backtracking_steps: int = 8,
        backtracking_factor: float = 0.5,
        backend: Any | None = None,
        device: Any | None = None,
        raise_on_failure: bool = False,
        evaluate_outputs: bool = True,
        jit: bool = True,
    ) -> BatchProblemSolution:
        """Solve a parameter batch with a generic or projection-specialized mode.

        ``projection_mode`` may be ``"auto"``, ``"dense_batched"``,
        ``"sparse_iterative"``, ``"shared_linear"``, or ``"generic_dense"``.
        Projection modes require an objective created by
        :meth:`set_projection_objective` or :meth:`squared_distance`.
        """
        from purin.solvers.primal_dual_interior_point.config import FixedMuOptions
        from purin.solvers.primal_dual_interior_point.solver import (
            prepare_batched_projection_condensed_sparse_solver,
        )
        from purin.solvers.primal_dual_interior_point.solver import (
            run_batched_fixed_mu_iterations,
        )
        from purin.solvers.primal_dual_interior_point.solver import (
            run_shared_linear_projection_batch,
        )
        from purin.solvers.primal_dual_interior_point.types import BATCH_STATUS_NAMES
        from purin.solvers.primal_dual_interior_point.types import BatchProblemSolution
        from purin.solvers.primal_dual_interior_point.types import (
            NewtonRegularization as JaxNewtonRegularization,
        )

        if backend is not None and device is not None:
            backend_device = _resolve_jax_device(backend)
            explicit_device = _resolve_jax_device(device)
            if (backend_device.platform, backend_device.id) != (
                explicit_device.platform,
                explicit_device.id,
            ):
                raise ValueError("backend and device select different JAX devices.")
            selected_device = backend_device
        else:
            selected_device = _resolve_jax_device(
                backend if backend is not None else device
            )
        compiled = self.compile_batch(
            parameters=parameters,
            initial_values=initial_values,
            batch_size=batch_size,
            interior_margin=interior_margin,
            device=selected_device,
        )
        opts = options or FixedMuOptions(line_search=None)
        if opts.line_search is not None:
            raise ValueError(
                "solve_batch requires FixedMuOptions(line_search=None); the batched "
                "kernels use vectorized fraction-to-the-boundary updates."
            )
        if opts.regularization is None:
            regularization = JaxNewtonRegularization()
        else:
            regularization = JaxNewtonRegularization(
                delta_x=opts.regularization.delta_x,
                delta_s=opts.regularization.delta_s,
                delta_c=opts.regularization.delta_c,
                delta_d=opts.regularization.delta_d,
            )

        aliases = {
            "dense": "dense_batched",
            "sparse": "sparse_iterative",
            "sparse_condensed": "sparse_iterative",
            "condensed_sparse": "sparse_iterative",
            "shared": "shared_linear",
            "generic": "generic_dense",
        }
        requested_mode = aliases.get(
            str(projection_mode).lower(), str(projection_mode).lower()
        )
        valid_modes = {
            "auto",
            "dense_batched",
            "sparse_iterative",
            "shared_linear",
            "generic_dense",
        }
        if requested_mode not in valid_modes:
            raise ValueError(
                f"Unknown projection_mode {projection_mode!r}; expected one of {sorted(valid_modes)}."
            )

        structure = compiled.projection_structure
        if structure is None and requested_mode not in {
            "auto",
            "generic_dense",
            "dense_batched",
        }:
            raise ValueError(
                f"projection_mode={requested_mode!r} requires an identity-Hessian "
                "distance objective created with set_projection_objective(...)."
            )

        initial_state = compiled.initial_state
        parameter_array = jnp.asarray(compiled.parameter_values, dtype=jnp.float64)
        indices = compiled.indices
        bounds = compiled.bounds
        if selected_device is not None:
            initial_state = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), initial_state
            )
            parameter_array = jax.device_put(parameter_array, selected_device)
            indices = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), indices
            )
            bounds = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), bounds
            )

        equality_jacobians = None
        shared_linear_available = False
        if (
            requested_mode in {"auto", "shared_linear"}
            and structure is not None
            and initial_state.y_d.shape[1] == 0
            and structure.constraints_affine_in_variables
        ):
            equality_jacobians = jax.vmap(compiled.equality_jacobian_function)(
                initial_state.x, parameter_array
            )
            jacobian_host = np.asarray(equality_jacobians, dtype=np.float64)
            shared_linear_available = bool(
                jacobian_host.shape[0] == 0
                or np.allclose(jacobian_host, jacobian_host[:1], rtol=1e-12, atol=1e-12)
            )

        if requested_mode == "auto":
            if structure is None:
                selected_mode = "generic_dense"
            elif shared_linear_available:
                selected_mode = "shared_linear"
            else:
                reduced_dimension = (
                    initial_state.x.shape[1]
                    + initial_state.s.shape[1]
                    + initial_state.y_c.shape[1]
                    + initial_state.y_d.shape[1]
                )
                if reduced_dimension >= int(
                    dense_dimension_threshold
                ) and structure.estimated_condensed_kkt_density <= float(
                    sparse_density_threshold
                ):
                    selected_mode = "sparse_iterative"
                else:
                    selected_mode = "dense_batched"
        else:
            selected_mode = requested_mode

        if selected_mode == "generic_dense" and structure is not None:
            selected_mode = "dense_batched"
        if selected_mode == "shared_linear" and not shared_linear_available:
            raise ValueError(
                "shared_linear requires an equality-only affine projection problem "
                "whose equality Jacobian is identical for every batch row."
            )
        if (
            selected_mode in {"dense_batched", "sparse_iterative", "shared_linear"}
            and structure is None
        ):
            if selected_mode == "dense_batched":
                selected_mode = "generic_dense"
            else:
                raise ValueError(
                    f"{selected_mode} requires set_projection_objective(...)."
                )

        if selected_mode == "shared_linear":
            assert equality_jacobians is not None
            result = run_shared_linear_projection_batch(
                initial_state=initial_state,
                parameter_values=parameter_array,
                equality_function=compiled.equality_function,
                equality_jacobian=equality_jacobians[0],
                target_parameter_indices=jnp.asarray(
                    structure.target_parameter_indices, dtype=jnp.int32
                ),
                residual_tolerance=opts.residual_tolerance,
                jit=jit,
            )
        elif selected_mode == "sparse_iterative":
            assert compiled.equality_local_derivatives_function is not None
            assert compiled.inequality_local_derivatives_function is not None
            model_signature = (
                tuple(self.parameter_names),
                tuple(self.variable_names),
                None if self._objective is None else self._objective.text,
                tuple((item.kind, item.text) for item in self.constraints),
            )
            result_device_key = next(iter(initial_state.x.devices()), None)
            device_key = (
                (result_device_key.platform, result_device_key.id)
                if result_device_key is not None
                else ("default", -1)
            )
            sparse_cache_key = (
                model_signature,
                int(parameter_array.shape[0]),
                device_key,
                int(opts.max_iterations),
                float(opts.residual_tolerance),
                float(opts.tau),
                float(opts.step_scale),
                float(opts.minimum_step),
                tuple(float(value) for value in regularization),
                float(gmres_tolerance),
                float(gmres_atol),
                int(gmres_restart),
                int(gmres_maxiter),
                bool(use_preconditioner),
                bool(backtracking),
                int(max_backtracking_steps),
                float(backtracking_factor),
            )
            sparse_solver = self._prepared_sparse_solver_cache.get(sparse_cache_key)
            if sparse_solver is None:
                sparse_solver = prepare_batched_projection_condensed_sparse_solver(
                    batch_size=int(parameter_array.shape[0]),
                    equality_function=compiled.equality_function,
                    inequality_function=compiled.inequality_function,
                    equality_local_derivatives=compiled.equality_local_derivatives_function,
                    inequality_local_derivatives=compiled.inequality_local_derivatives_function,
                    equality_affine_offsets=compiled.equality_affine_offset_function,
                    inequality_affine_offsets=compiled.inequality_affine_offset_function,
                    equality_constant_local_jacobian=compiled.equality_constant_local_jacobian,
                    inequality_constant_local_jacobian=compiled.inequality_constant_local_jacobian,
                    equality_support=jnp.asarray(
                        structure.equality_support, dtype=jnp.int32
                    ),
                    equality_mask=jnp.asarray(
                        structure.equality_support_mask, dtype=bool
                    ),
                    inequality_support=jnp.asarray(
                        structure.inequality_support, dtype=jnp.int32
                    ),
                    inequality_mask=jnp.asarray(
                        structure.inequality_support_mask, dtype=bool
                    ),
                    target_parameter_indices=jnp.asarray(
                        structure.target_parameter_indices, dtype=jnp.int32
                    ),
                    indices=indices,
                    bounds=bounds,
                    max_iterations=opts.max_iterations,
                    residual_tolerance=opts.residual_tolerance,
                    tau=opts.tau,
                    step_scale=opts.step_scale,
                    minimum_step=opts.minimum_step,
                    regularization=regularization,
                    gmres_tolerance=gmres_tolerance,
                    gmres_atol=gmres_atol,
                    gmres_restart=gmres_restart,
                    gmres_maxiter=gmres_maxiter,
                    use_preconditioner=use_preconditioner,
                    backtracking=backtracking,
                    max_backtracking_steps=max_backtracking_steps,
                    backtracking_factor=backtracking_factor,
                    constraints_affine=structure.constraints_affine_in_variables,
                    use_column_gather_transpose=device_key[0] != "cpu",
                )
                self._prepared_sparse_solver_cache[sparse_cache_key] = sparse_solver
                self._prepared_sparse_solver_cache_misses += 1
            else:
                self._prepared_sparse_solver_cache_hits += 1
            result = sparse_solver.solve(
                initial_state=initial_state, parameter_values=parameter_array, mu=mu
            )
        else:
            result = run_batched_fixed_mu_iterations(
                initial_state=initial_state,
                parameter_values=parameter_array,
                evaluate_nlp=compiled.evaluate_nlp_single,
                indices=indices,
                bounds=bounds,
                mu=mu,
                max_iterations=opts.max_iterations,
                residual_tolerance=opts.residual_tolerance,
                tau=opts.tau,
                step_scale=opts.step_scale,
                minimum_step=opts.minimum_step,
                regularization=regularization,
                check_newton_residual=opts.check_newton_residual,
                jit=jit,
            )

        result_device = next(iter(result.state.x.devices()))
        (
            status_device,
            converged_device,
            iterations_device,
            residual_device,
            final_variables_device,
        ) = jax.device_get(
            (
                result.status_codes,
                result.converged,
                result.iterations,
                result.final_residual_inf_norm,
                result.state.x,
            )
        )
        status_codes = np.asarray(status_device, dtype=np.int32)
        converged = np.asarray(converged_device, dtype=bool)
        iterations_host = np.asarray(iterations_device, dtype=np.int32)
        residual_host = np.asarray(residual_device, dtype=np.float64)
        if raise_on_failure and not np.all(converged):
            failures = [
                (
                    int(index),
                    BATCH_STATUS_NAMES[int(status_codes[index])],
                    float(result.final_residual_inf_norm[index]),
                )
                for index in np.flatnonzero(~converged)
            ]
            raise RuntimeError(
                f"Batched problem {self.name!r} had nonconverged rows in "
                f"mode {selected_mode!r}: {failures}."
            )

        final_variables = np.asarray(final_variables_device, dtype=np.float64)
        if evaluate_outputs:
            objectives = compiled.objectives(final_variables)
            equalities = compiled.equalities(final_variables)
            inequalities = compiled.inequalities(final_variables)
        else:
            objectives = np.full((final_variables.shape[0],), np.nan, dtype=np.float64)
            equalities = np.full(
                (final_variables.shape[0], result.state.y_c.shape[1]),
                np.nan,
                dtype=np.float64,
            )
            inequalities = np.full(
                (final_variables.shape[0], result.state.y_d.shape[1]),
                np.nan,
                dtype=np.float64,
            )
        return BatchProblemSolution(
            problem_name=self.name,
            variables=BatchNamedValues(self.variable_names, final_variables),
            parameters=BatchNamedValues(
                self.parameter_names, compiled.parameter_values
            ),
            objectives=objectives,
            equalities=equalities,
            inequalities=inequalities,
            raw_result=result,
            projection_mode=selected_mode,
            device_platform=result_device.platform,
            status_codes_host=status_codes,
            converged_host=converged,
            iterations_host=iterations_host,
            residual_inf_norm_host=residual_host,
        )

    def batch_solver_cache_info(self) -> dict[str, int]:
        """Return persistent sparse-solver cache statistics."""

        return {
            "compiled_structure_size": len(self._compiled_batch_structure_cache),
            "compiled_structure_hits": self._compiled_batch_structure_cache_hits,
            "compiled_structure_misses": self._compiled_batch_structure_cache_misses,
            "sparse_solver_size": len(self._prepared_sparse_solver_cache),
            "sparse_solver_hits": self._prepared_sparse_solver_cache_hits,
            "sparse_solver_misses": self._prepared_sparse_solver_cache_misses,
        }

    def optimize_batch(self, **kwargs: Any) -> BatchProblemSolution:
        """Alias for :meth:`solve_batch`."""
        return self.solve_batch(**kwargs)

    def solve(
        self,
        *,
        parameters: Mapping[str, float] | Sequence[float] | None = None,
        initial_values: Mapping[str, float] | Sequence[float] | None = None,
        mu: float = 1e-4,
        options: FixedMuOptions | None = None,
        interior_margin: float = 1.0,
        use_filter_line_search: bool = True,
        raise_on_failure: bool = False,
    ) -> ProblemSolution:
        """Solve one row through the single sparse production solver path.

        ``use_filter_line_search`` is retained as a source-compatible argument.
        The unified JAX batch kernel uses its preserved vectorized
        fraction-to-the-boundary/backtracking configuration.
        """
        from types import SimpleNamespace
        from purin.solvers.primal_dual_interior_point.config import FixedMuOptions
        from purin.solvers.primal_dual_interior_point.types import BATCH_STATUS_NAMES
        from purin.solvers.primal_dual_interior_point.types import ProblemSolution

        del use_filter_line_search
        parameter_row = self._bind_parameters(parameters)
        initial_row = self._bind_initial_values(initial_values)
        selected_options = options or FixedMuOptions(line_search=None)
        if selected_options.line_search is not None:
            selected_options = replace(selected_options, line_search=None)
        batch = self.solve_batch(
            parameters=parameter_row[None, :],
            initial_values=initial_row[None, :],
            batch_size=1,
            mu=mu,
            options=selected_options,
            interior_margin=interior_margin,
            raise_on_failure=raise_on_failure,
        )
        raw_batch = batch.raw_result
        state = jax.tree_util.tree_map(lambda value: value[0], raw_batch.state)
        code = int(np.asarray(raw_batch.status_codes)[0])
        raw_result = SimpleNamespace(
            state=state,
            status=BATCH_STATUS_NAMES[code],
            converged=bool(np.asarray(raw_batch.converged)[0]),
            iterations=int(np.asarray(raw_batch.iterations)[0]),
            final_residual_inf_norm=float(
                np.asarray(raw_batch.final_residual_inf_norm)[0]
            ),
            residual_history=np.asarray(raw_batch.residual_history)[:, 0],
        )
        return ProblemSolution(
            problem_name=self.name,
            variables=NamedValues(
                self.variable_names, batch.variables.matrix(copy=False)[0]
            ),
            parameters=NamedValues(self.parameter_names, parameter_row),
            objective=float(np.asarray(batch.objectives)[0]),
            equalities=np.asarray(batch.equalities)[0],
            inequalities=np.asarray(batch.inequalities)[0],
            raw_result=raw_result,
        )

    def optimize(self, **kwargs: Any) -> ProblemSolution:
        """Alias for :meth:`solve`, matching the reference modeling API."""
        return self.solve(**kwargs)


# Descriptive aliases for different user preferences.
PrimalDualProblem = Problem
OptimizationProblem = Problem


def _projection_support_arrays(
    equality_dependencies, inequality_dependencies, max_row_nonzeros
):
    """Build padded row supports and canonical sparse row/column positions."""

    def padded_support(
        dependencies: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[bool, ...], ...]]:
        support_rows = []
        mask_rows = []
        for dependency in dependencies:
            padding = max_row_nonzeros - len(dependency)
            support_rows.append(tuple(dependency) + (0,) * padding)
            mask_rows.append((True,) * len(dependency) + (False,) * padding)
        return tuple(support_rows), tuple(mask_rows)

    equality_support, equality_support_mask = padded_support(equality_dependencies)
    inequality_support, inequality_support_mask = padded_support(
        inequality_dependencies
    )
    equality_rows = tuple(
        row
        for row, dependencies in enumerate(equality_dependencies)
        for _ in dependencies
    )
    equality_cols = tuple(
        column for dependencies in equality_dependencies for column in dependencies
    )
    inequality_rows = tuple(
        row
        for row, dependencies in enumerate(inequality_dependencies)
        for _ in dependencies
    )
    inequality_cols = tuple(
        column for dependencies in inequality_dependencies for column in dependencies
    )
    return (
        equality_cols,
        equality_rows,
        equality_support,
        equality_support_mask,
        inequality_cols,
        inequality_rows,
        inequality_support,
        inequality_support_mask,
    )


def _projection_structure_counts(
    self, equalities, equality_dependencies, inequalities, inequality_dependencies
):
    """Compute sparse dimensions, nonzero estimates and row width."""
    affine = all(
        constraint.residual.affine_in_variables for constraint in self.constraints
    )

    n_x = len(self.variable_names)
    n_c = len(equalities)
    n_d = len(inequalities)
    n_s = n_d
    reduced_dimension = n_x + n_s + n_c + n_d
    condensed_dimension = n_x + n_c + n_d
    jacobian_nonzeros = sum(map(len, equality_dependencies)) + sum(
        map(len, inequality_dependencies)
    )
    nonlinear_hessian_nonzeros = sum(
        len(dependencies) ** 2
        for constraint, dependencies in strict_zip(
            [*equalities, *inequalities],
            [*equality_dependencies, *inequality_dependencies],
        )
        if not constraint.residual.affine_in_variables
    )
    approximate_nonzeros = (
        n_x
        + nonlinear_hessian_nonzeros
        + 2 * jacobian_nonzeros
        + n_s
        + 2 * n_d
        + n_c
        + n_d
    )
    density = (
        min(1.0, float(approximate_nonzeros) / float(reduced_dimension**2))
        if reduced_dimension
        else 0.0
    )
    condensed_nonzeros = (
        n_x + nonlinear_hessian_nonzeros + 2 * jacobian_nonzeros + n_c + n_d
    )
    condensed_density = (
        min(1.0, float(condensed_nonzeros) / float(condensed_dimension**2))
        if condensed_dimension
        else 0.0
    )

    max_row_nonzeros = max(
        [
            1,
            *(len(item) for item in equality_dependencies),
            *(len(item) for item in inequality_dependencies),
        ]
    )
    return affine, condensed_density, density, max_row_nonzeros
