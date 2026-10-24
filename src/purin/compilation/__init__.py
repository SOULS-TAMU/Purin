"""Compile definitions once; evaluate numerical values at runtime."""

from purin.problems.canonical import CanonicalProblem, CanonicalQP as CanonicalQP
from purin.problems.compile import compile_symbolic_problem as compile_symbolic_problem


def compile_problem(problem):
    """Convert a supported problem definition to the canonical projection representation."""
    if isinstance(problem, CanonicalProblem):
        return problem
    if hasattr(problem, "canonicalize"):
        return problem.canonicalize()
    raise TypeError("Expected a canonical or compilable optimization problem")
