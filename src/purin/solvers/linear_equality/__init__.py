"""Linear-equality projection backend."""

from purin.solvers.linear_equality.config import LinearEqualityConfig
from purin.solvers.linear_equality.solver import CanonicalLinearEqualityProjection
from purin.solvers.linear_equality.solver import LinearEqualityProjection
from purin.solvers.linear_equality.solver import (
    PreparedCanonicalLinearEqualityProjection,
)
from purin.solvers.linear_equality.solver import PreparedLinearEqualityProjection

# One-release compatibility alias; the implementation remains singular.
ExplicitLinearProjection = LinearEqualityProjection

__all__ = [
    "LinearEqualityConfig",
    "LinearEqualityProjection",
    "CanonicalLinearEqualityProjection",
    "ExplicitLinearProjection",
    "PreparedLinearEqualityProjection",
    "PreparedCanonicalLinearEqualityProjection",
]
