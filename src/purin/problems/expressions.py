"""Symbolic expressions and normalized constraints for problem definitions.

The objects in this module are frontend-only: they construct traceable scalar
and vector expressions and normalize all inequalities to ``g(theta, y) <= 0``.
They do not import or invoke an optimization backend.
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from purin._utils.validation import strict_zip

if TYPE_CHECKING:
    from purin.problems.problem import Problem


def _coerce_names(
    names: str | Iterable[str], *, allow_empty: bool = False
) -> list[str]:
    if isinstance(names, str):
        result = [names]
    else:
        result = [str(name) for name in names]
    if not result and not allow_empty:
        raise ValueError("At least one name is required.")
    if any(not name for name in result):
        raise ValueError("Symbol names must be nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"Symbol names must be unique; received {result}.")
    return result


@dataclass(frozen=True)
class _EvaluationContext:
    variables: jax.Array
    parameters: jax.Array
    variable_index: Mapping[str, int]
    parameter_index: Mapping[str, int]

    def value(self, kind: str, name: str):
        """Look up a named symbol in the current expression evaluation context."""
        if kind == "variable":
            return self.variables[self.variable_index[name]]
        if kind == "parameter":
            return self.parameters[self.parameter_index[name]]
        raise KeyError(f"Unknown symbol kind {kind!r}.")


@dataclass(frozen=True)
class _AffineScalar:
    """Parameter-only scalar used by the affine expression decomposition."""

    function: Callable[[_EvaluationContext], Any]
    parameter_dependencies: frozenset[str] = frozenset()

    def evaluate(self, context: _EvaluationContext):
        """Evaluate an affine scalar expression in the supplied context."""
        return self.function(context)


def _constant_affine_scalar(value: float) -> _AffineScalar:
    scalar = float(value)
    return _AffineScalar(
        lambda context, scalar=scalar: jnp.asarray(
            scalar, dtype=context.variables.dtype
        )
    )


def _combine_affine_scalars(
    left: _AffineScalar, right: _AffineScalar, operator: Callable
) -> _AffineScalar:
    return _AffineScalar(
        lambda context, left=left, right=right, operator=operator: operator(
            left.evaluate(context), right.evaluate(context)
        ),
        left.parameter_dependencies | right.parameter_dependencies,
    )


class Expression:
    """Scalar symbolic expression backed by a JAX-traceable callable.

    The dependency metadata is static and is used only to select and compile
    projection-specialized sparse kernels.  Numerical derivatives are still
    evaluated by JAX.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        function: Callable[[_EvaluationContext], Any],
        text: str,
        *,
        variable_dependencies: Iterable[str] = (),
        parameter_dependencies: Iterable[str] = (),
        affine_in_variables: bool = True,
        symbol_kind: str | None = None,
        symbol_name: str | None = None,
        projection_target_names: tuple[str, ...] | None = None,
        affine_coefficients: Mapping[str, _AffineScalar] | None = None,
        affine_offset: _AffineScalar | None = None,
    ) -> None:
        self._function = function
        self.text = str(text)
        self.variable_dependencies = frozenset(variable_dependencies)
        self.parameter_dependencies = frozenset(parameter_dependencies)
        self.affine_in_variables = bool(affine_in_variables)
        self.symbol_kind = symbol_kind
        self.symbol_name = symbol_name
        self.projection_target_names = projection_target_names
        if affine_coefficients is not None:
            self.affine_coefficients = dict(affine_coefficients)
            self.affine_offset = affine_offset or _constant_affine_scalar(0.0)
        elif self.affine_in_variables and not self.variable_dependencies:
            self.affine_coefficients = {}
            self.affine_offset = _AffineScalar(
                function, frozenset(parameter_dependencies)
            )
        else:
            self.affine_coefficients = None
            self.affine_offset = None

    def evaluate(self, context: _EvaluationContext):
        """Evaluate this symbolic scalar expression in the supplied context."""
        return self._function(context)

    def _combine_metadata(
        self, right: "Expression", symbol: str
    ) -> tuple[frozenset[str], frozenset[str], bool]:
        variable_dependencies = self.variable_dependencies | right.variable_dependencies
        parameter_dependencies = (
            self.parameter_dependencies | right.parameter_dependencies
        )
        if symbol in {"+", "-"}:
            affine = self.affine_in_variables and right.affine_in_variables
        elif symbol == "*":
            affine = (
                self.affine_in_variables
                and right.affine_in_variables
                and (not self.variable_dependencies or not right.variable_dependencies)
            )
        elif symbol == "/":
            affine = self.affine_in_variables and not right.variable_dependencies
        else:
            affine = not variable_dependencies
        return variable_dependencies, parameter_dependencies, affine

    def _binary(self, other: Any, operator: Callable, symbol: str) -> "Expression":
        """Compose a binary expression while propagating affine and sparsity metadata."""
        right = _as_expression(other)
        variable_dependencies, parameter_dependencies, affine = self._combine_metadata(
            right, symbol
        )
        affine_coefficients = None
        affine_offset = None
        if (
            affine
            and self.affine_coefficients is not None
            and right.affine_coefficients is not None
        ):
            if symbol in {"+", "-"}:
                names = self.variable_dependencies | right.variable_dependencies
                affine_coefficients = {}
                zero = _constant_affine_scalar(0.0)
                for name in names:
                    affine_coefficients[name] = _combine_affine_scalars(
                        self.affine_coefficients.get(name, zero),
                        right.affine_coefficients.get(name, zero),
                        operator,
                    )
                affine_offset = _combine_affine_scalars(
                    self.affine_offset, right.affine_offset, operator
                )
            elif symbol == "*":
                if not self.variable_dependencies:
                    scalar = self.affine_offset
                    variable_expression = right
                else:
                    scalar = right.affine_offset
                    variable_expression = self
                affine_coefficients = {
                    name: _combine_affine_scalars(coefficient, scalar, operator)
                    for name, coefficient in variable_expression.affine_coefficients.items()
                }
                affine_offset = _combine_affine_scalars(
                    variable_expression.affine_offset, scalar, operator
                )
            elif symbol == "/":
                affine_coefficients = {
                    name: _combine_affine_scalars(
                        coefficient, right.affine_offset, operator
                    )
                    for name, coefficient in self.affine_coefficients.items()
                }
                affine_offset = _combine_affine_scalars(
                    self.affine_offset, right.affine_offset, operator
                )
        return Expression(
            lambda context, left=self, right=right: operator(
                left.evaluate(context), right.evaluate(context)
            ),
            f"({self.text} {symbol} {right.text})",
            variable_dependencies=variable_dependencies,
            parameter_dependencies=parameter_dependencies,
            affine_in_variables=affine,
            affine_coefficients=affine_coefficients,
            affine_offset=affine_offset,
        )

    def _reverse_binary(
        self, other: Any, operator: Callable, symbol: str
    ) -> "Expression":
        left = _as_expression(other)
        return left._binary(self, operator, symbol)

    def __add__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left + right, "+")

    def __radd__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left + right, "+")

    def __sub__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left - right, "-")

    def __rsub__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left - right, "-")

    def __mul__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left * right, "*")

    def __rmul__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left * right, "*")

    def __truediv__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left / right, "/")

    def __rtruediv__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left / right, "/")

    def __pow__(self, other: Any) -> "Expression":
        right = _as_expression(other)
        variable_dependencies = self.variable_dependencies | right.variable_dependencies
        parameter_dependencies = (
            self.parameter_dependencies | right.parameter_dependencies
        )
        return Expression(
            lambda context, left=self, right=right: (
                left.evaluate(context) ** right.evaluate(context)
            ),
            f"({self.text} ** {right.text})",
            variable_dependencies=variable_dependencies,
            parameter_dependencies=parameter_dependencies,
            affine_in_variables=not variable_dependencies,
        )

    def __rpow__(self, other: Any) -> "Expression":
        return _as_expression(other).__pow__(self)

    def __neg__(self) -> "Expression":
        affine_coefficients = None
        affine_offset = None
        if self.affine_coefficients is not None:
            minus_one = _constant_affine_scalar(-1.0)
            affine_coefficients = {
                name: _combine_affine_scalars(
                    coefficient, minus_one, lambda left, right: left * right
                )
                for name, coefficient in self.affine_coefficients.items()
            }
            affine_offset = _combine_affine_scalars(
                self.affine_offset, minus_one, lambda left, right: left * right
            )
        return Expression(
            lambda context, expression=self: -expression.evaluate(context),
            f"(-{self.text})",
            variable_dependencies=self.variable_dependencies,
            parameter_dependencies=self.parameter_dependencies,
            affine_in_variables=self.affine_in_variables,
            affine_coefficients=affine_coefficients,
            affine_offset=affine_offset,
        )

    def __abs__(self) -> "Expression":
        return Expression(
            lambda context, expression=self: jnp.abs(expression.evaluate(context)),
            f"abs({self.text})",
            variable_dependencies=self.variable_dependencies,
            parameter_dependencies=self.parameter_dependencies,
            affine_in_variables=not self.variable_dependencies,
        )

    def __le__(self, other: Any) -> "Constraint":
        right = _as_expression(other)
        return Constraint(self - right, "ineq", f"{self.text} <= {right.text}")

    def __ge__(self, other: Any) -> "Constraint":
        right = _as_expression(other)
        # Normalize every inequality to residual <= 0.
        return Constraint(right - self, "ineq", f"{self.text} >= {right.text}")

    def __eq__(self, other: Any) -> "Constraint":  # type: ignore[override]
        right = _as_expression(other)
        return Constraint(self - right, "eq", f"{self.text} == {right.text}")

    def __bool__(self) -> bool:
        raise TypeError(
            "Symbolic expressions do not have a truth value. Use ==, <=, or >= "
            "inside problem.constraints.add(...)."
        )

    def __repr__(self) -> str:
        return f"Expression({self.text})"


class VectorExpression:
    """One-dimensional symbolic expression with elementwise comparisons."""

    __array_priority__ = 1000

    def __init__(self, components: Sequence[Expression], text: str) -> None:
        self.components = tuple(components)
        self.text = str(text)

    @property
    def size(self) -> int:
        """Return the number of scalar expressions in the vector."""
        return len(self.components)

    def evaluate(self, context: _EvaluationContext):
        """Evaluate and stack this vector's scalar expressions."""
        if not self.components:
            return jnp.empty((0,), dtype=context.variables.dtype)
        return jnp.stack([component.evaluate(context) for component in self.components])

    def __getitem__(self, index: int) -> Expression:
        return self.components[index]

    def __iter__(self) -> Iterator[Expression]:
        return iter(self.components)

    def _coerce(self, other: Any) -> "VectorExpression":
        if isinstance(other, VectorExpression):
            if other.size != self.size:
                raise ValueError(
                    f"Vector sizes must match; received {self.size} and {other.size}."
                )
            return other
        if isinstance(other, _SymbolNamespace):
            return self._coerce(other.vector())
        if np.isscalar(other) or isinstance(other, Expression):
            scalar = _as_expression(other)
            return VectorExpression([scalar] * self.size, scalar.text)
        values = np.asarray(other)
        if values.ndim != 1 or values.size != self.size:
            raise ValueError(
                f"Expected a vector of length {self.size}; received shape {values.shape}."
            )
        return VectorExpression(
            [_as_expression(value) for value in values.tolist()], repr(values.tolist())
        )

    def _binary(
        self, other: Any, operator: Callable, symbol: str
    ) -> "VectorExpression":
        right = self._coerce(other)
        return VectorExpression(
            [operator(left, rhs) for left, rhs in strict_zip(self, right)],
            f"({self.text} {symbol} {right.text})",
        )

    def __add__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left + right, "+")

    def __radd__(self, other: Any) -> "VectorExpression":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left - right, "-")

    def __rsub__(self, other: Any) -> "VectorExpression":
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left * right, "*")

    def __rmul__(self, other: Any) -> "VectorExpression":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left / right, "/")

    def __neg__(self) -> "VectorExpression":
        return VectorExpression([-component for component in self], f"(-{self.text})")

    def __le__(self, other: Any) -> list["Constraint"]:
        right = self._coerce(other)
        return [left <= rhs for left, rhs in strict_zip(self, right)]

    def __ge__(self, other: Any) -> list["Constraint"]:
        right = self._coerce(other)
        return [left >= rhs for left, rhs in strict_zip(self, right)]

    def __eq__(self, other: Any) -> list["Constraint"]:  # type: ignore[override]
        right = self._coerce(other)
        return [left == rhs for left, rhs in strict_zip(self, right)]

    def sum(self) -> Expression:
        """Construct a symbolic sum of the vector's entries."""
        return sum(self.components, _as_expression(0.0))

    def __repr__(self) -> str:
        return f"VectorExpression(size={self.size}, text={self.text!r})"


def _as_expression(value: Any) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not np.isscalar(value):
        raise TypeError(
            "Expected a scalar symbolic expression or numeric constant; "
            f"received {type(value).__name__}."
        )
    scalar = float(value)
    return Expression(lambda _context, value=scalar: jnp.asarray(value), repr(scalar))


@dataclass(frozen=True)
class Constraint:
    """Normalized scalar constraint."""

    residual: Expression
    kind: str
    text: str

    def __post_init__(self) -> None:
        if self.kind not in {"eq", "ineq"}:
            raise ValueError(
                f"Constraint kind must be 'eq' or 'ineq'; got {self.kind!r}."
            )


class ConstraintList:
    """Mutable collection used by ``problem.constraints.add(...)``."""

    def __init__(self) -> None:
        self.items: list[Constraint] = []

    def add(self, *constraints: Constraint | Sequence[Constraint]) -> "ConstraintList":
        """Add one constraint or a supported collection of constraints."""
        for constraint in constraints:
            if isinstance(constraint, (list, tuple)):
                self.add(*constraint)
                continue
            if not isinstance(constraint, Constraint):
                raise TypeError(
                    "constraints.add(...) expects symbolic comparisons created with ==, <=, or >=."
                )
            self.items.append(constraint)
        return self

    def equality(self, expression: Any, *, name: str | None = None) -> "ConstraintList":
        """Construct and register an equality residual."""
        residual = _as_expression(expression)
        self.items.append(Constraint(residual, "eq", name or f"{residual.text} == 0"))
        return self

    def inequality(
        self, expression: Any, *, name: str | None = None
    ) -> "ConstraintList":
        """Construct and register an inequality residual."""
        residual = _as_expression(expression)
        self.items.append(Constraint(residual, "ineq", name or f"{residual.text} <= 0"))
        return self

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self.items)


class _SymbolNamespace:
    """Ordered symbol lookup and vector construction for one problem namespace."""

    def __init__(self, problem: "Problem", kind: str) -> None:
        self._problem = problem
        self._kind = kind

    @property
    def names(self) -> tuple[str, ...]:
        """Return symbol names in their declared order."""
        if self._kind == "parameter":
            return tuple(self._problem.parameter_names)
        return tuple(self._problem.variable_names)

    def __getattr__(self, name: str) -> Expression:
        return self[name]

    def __getitem__(self, name_or_index: str | int) -> Expression:
        if isinstance(name_or_index, int):
            name = self.names[name_or_index]
        else:
            name = str(name_or_index)
        self._problem._check_symbol(self._kind, name)
        affine_coefficients = None
        affine_offset = None
        if self._kind == "variable":
            affine_coefficients = {name: _constant_affine_scalar(1.0)}
            affine_offset = _constant_affine_scalar(0.0)
        return Expression(
            lambda context, kind=self._kind, name=name: context.value(kind, name),
            name,
            variable_dependencies=(name,) if self._kind == "variable" else (),
            parameter_dependencies=(name,) if self._kind == "parameter" else (),
            affine_in_variables=True,
            symbol_kind=self._kind,
            symbol_name=name,
            affine_coefficients=affine_coefficients,
            affine_offset=affine_offset,
        )

    def vector(self) -> VectorExpression:
        """Construct a symbolic vector from selected namespace entries."""
        token = "p" if self._kind == "parameter" else "y"
        return VectorExpression([self[name] for name in self.names], token)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[Expression]:
        return iter(self.vector())
