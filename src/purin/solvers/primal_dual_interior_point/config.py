"""Primal-dual interior-point configuration."""

from __future__ import annotations
from dataclasses import dataclass
from purin.solvers.config import ImplicitGradientConfig
from purin.solvers.config import LinearSolverConfig
from purin.solvers.config import ProjectionConfig


@dataclass(frozen=True)
class FixedMuOptions:
    """Compatibility view of the fixed-barrier iteration options."""

    max_iterations: int = 50
    residual_tolerance: float = 1e-9
    tau: float = 0.995
    step_scale: float = 1.0
    minimum_step: float = 1e-14
    regularization: object | None = None
    require_positive_multipliers: bool = True
    positivity_tolerance: float = 0.0
    check_newton_residual: bool = False
    line_search: object | None = None


__all__ = [
    "ProjectionConfig",
    "LinearSolverConfig",
    "ImplicitGradientConfig",
    "FixedMuOptions",
]
