"""Optimization problem definitions."""

from purin.problems.canonical import CanonicalProblem
from purin.problems.canonical import CanonicalQP
from purin.problems.functional import FunctionalProblem
from purin.problems.functional import functional_problem
from purin.problems.qp import MatrixProblem
from purin.problems.qp import ParametricLP
from purin.problems.qp import ParametricQP
from purin.problems.symbolic import Problem
from purin.problems.symbolic import SymbolicProblem

__all__ = [
    "CanonicalProblem",
    "CanonicalQP",
    "SymbolicProblem",
    "Problem",
    "FunctionalProblem",
    "functional_problem",
    "ParametricQP",
    "ParametricLP",
    "MatrixProblem",
]

from purin.problems.projection import Projection

try:
    __all__.append("Projection")
except NameError:
    __all__ = ["Projection"]
