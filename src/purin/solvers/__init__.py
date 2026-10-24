"""Interchangeable differentiable projection backends."""

from purin.solvers.admm import ADMMConfig
from purin.solvers.admm import ADMMProjection
from purin.solvers.api import prepare_projection
from purin.solvers.api import registered_methods
from purin.solvers.chambolle_pock import ChambollePockConfig
from purin.solvers.chambolle_pock import ChambollePockProjection
from purin.solvers.config import FischerBurmeisterConfig
from purin.solvers.config import ImplicitGradientConfig
from purin.solvers.config import LinearEqualityConfig
from purin.solvers.config import LinearSolverConfig
from purin.solvers.config import ProjectionConfig
from purin.solvers.types import ProjectionInfo
from purin.solvers.types import ProjectionMethod
from purin.solvers.types import ProjectionResult
from purin.solvers.types import ProjectionTrainingOutput
from purin.solvers.types import SolveStatus

__all__ = [
    "ADMMConfig",
    "ADMMProjection",
    "ChambollePockConfig",
    "ChambollePockProjection",
    "prepare_projection",
    "registered_methods",
    "ProjectionMethod",
    "SolveStatus",
    "ProjectionResult",
    "ProjectionTrainingOutput",
    "ProjectionInfo",
    "ProjectionConfig",
    "FischerBurmeisterConfig",
    "LinearEqualityConfig",
    "LinearSolverConfig",
    "ImplicitGradientConfig",
]
