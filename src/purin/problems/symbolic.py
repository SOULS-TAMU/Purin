"""Named symbolic optimization problems with a small convenience façade."""

from __future__ import annotations
from typing import Any
from purin.problems.problem import Problem as _CoreProblem


class SymbolicProblem:
    """Thin construction-time façade over the unchanged core ``Problem``.

    The wrapper adds ``parameters()``, ``variables()``, and ``subject_to()``
    without modifying the numerical implementation. All other attributes and
    methods are delegated to the core object.
    """

    __slots__ = ("_core",)

    def __init__(self, name: str = "problem") -> None:
        object.__setattr__(self, "_core", _CoreProblem(name=name))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_core":
            object.__setattr__(self, name, value)
        else:
            setattr(self._core, name, value)

    def parameters(self, names):
        """Return declared parameter symbols in canonical order."""
        return self._core.add_parameter(names)

    def variables(self, names):
        """Return declared variable symbols in canonical order."""
        return self._core.add_variable(names)

    def minimize(self, expression):
        """Assign the scalar objective and return this problem."""
        self._core.objective = expression
        return self

    def subject_to(self, *constraints):
        """Add one or more equality/inequality constraints."""
        self._core.constraints.add(*constraints)
        return self

    def canonicalize(self):
        """Compile once to the backend-neutral immutable representation."""
        return self._core.canonicalize()

    def to_symbolic_problem(self) -> _CoreProblem:
        """Return this symbolic problem without further adaptation."""
        return self._core

    def as_projection_problem(self) -> _CoreProblem:
        """Expose the symbolic resources required by projection preparation."""
        source = self._core
        target_names = [f"__target_{i}" for i in range(len(source.variable_names))]
        projected = _CoreProblem(name=f"{source.name}_projection")
        projected.add_parameter(target_names + list(source.parameter_names))
        projected.add_variable(list(source.variable_names))
        from purin.problems.expressions import VectorExpression

        target = VectorExpression(
            [getattr(projected.parameter, name) for name in target_names],
            "projection target",
        )
        projected.set_projection_objective(variables=projected.variable, target=target)
        projected.constraints.add(*list(source.constraints))
        return projected

    def solve(self, **kwargs):
        """Solve the symbolic problem for supplied parameter values."""
        return self._core.solve(**kwargs)

    def optimize(self, **kwargs):
        """Run the problem's optimization entry point with the selected solver."""
        return self._core.optimize(**kwargs)

    def __repr__(self) -> str:
        return f"SymbolicProblem({self._core!r})"


Problem = SymbolicProblem
