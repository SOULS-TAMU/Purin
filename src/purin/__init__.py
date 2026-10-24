"""Differentiable constrained projections and numerical kernels in JAX."""

from importlib import import_module

__version__ = "0.1.0"
_EXPORTS = {
    "Problem": ("problems", "Problem"),
    "Projection": ("problems", "Projection"),
    "FunctionalProblem": ("problems", "FunctionalProblem"),
    "SymbolicProblem": ("problems", "SymbolicProblem"),
    "MatrixProblem": ("problems", "MatrixProblem"),
    "ParametricQP": ("problems", "ParametricQP"),
    "ParametricLP": ("problems", "ParametricLP"),
    "prepare_projection": ("solvers", "prepare_projection"),
    "registered_methods": ("solvers", "registered_methods"),
    "ProjectionMethod": ("solvers", "ProjectionMethod"),
    "ProjectionConfig": ("solvers", "ProjectionConfig"),
    "ChambollePockConfig": ("solvers", "ChambollePockConfig"),
    "ADMMConfig": ("solvers", "ADMMConfig"),
    "FischerBurmeisterConfig": ("solvers", "FischerBurmeisterConfig"),
    "LinearEqualityConfig": ("solvers", "LinearEqualityConfig"),
    "LinearSolverConfig": ("solvers", "LinearSolverConfig"),
    "ImplicitGradientConfig": ("solvers", "ImplicitGradientConfig"),
    "SolveStatus": ("solvers", "SolveStatus"),
    "Distance": ("projections", "Distance"),
    "WeightedDistance": ("projections", "WeightedDistance"),
    "LocalQuadratic": ("projections", "LocalQuadratic"),
    "Sequential": ("projections", "Sequential"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module, symbol = _EXPORTS[name]
        value = getattr(import_module("purin." + module), symbol)
    else:
        try:
            value = import_module("purin." + name)
        except ModuleNotFoundError as error:
            if error.name != "purin." + name:
                raise
            raise AttributeError(name) from error
    globals()[name] = value
    return value
