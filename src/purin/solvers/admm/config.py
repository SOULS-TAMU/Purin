"""Configuration for sparse OSQP-style ADMM Euclidean projections."""

from purin._utils.validation import validate_metric_diagonal
from dataclasses import dataclass, field
import math
from purin.solvers.config import ImplicitGradientConfig
from purin.solvers.config import LinearSolverConfig


@dataclass(frozen=True)
class ADMMConfig:
    """Penalty, relaxation, stopping and factorization options for ADMM projection."""

    tolerance: float = 1e-9
    max_iterations: int = 10000
    rho: float = 1.0
    sigma: float = 1e-6
    alpha: float = 1.6
    equality_rho_multiplier: float = 1.0
    cg_max_iterations: int = 32
    cg_tolerance: float = 1e-10
    cg_refinement_steps: int = 1
    adaptive_rho: bool = False
    rho_update_interval: int = 50
    rho_update_threshold: float = 5.0
    row_scaling: bool = True
    active_tolerance: float = 1e-8
    linear_solver: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    implicit_gradient: ImplicitGradientConfig = field(
        default_factory=ImplicitGradientConfig
    )

    check_interval: int = 25

    metric_diagonal: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "metric_diagonal", validate_metric_diagonal(self.metric_diagonal)
        )
        for name in (
            "tolerance",
            "rho",
            "sigma",
            "equality_rho_multiplier",
            "cg_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "max_iterations",
            "cg_max_iterations",
            "cg_refinement_steps",
            "rho_update_interval",
            "check_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 2:
            raise ValueError("alpha must be between zero and two")
        if (
            not math.isfinite(self.rho_update_threshold)
            or self.rho_update_threshold < 1
        ):
            raise ValueError("rho_update_threshold must be finite and at least one")
        if not math.isfinite(self.active_tolerance) or self.active_tolerance < 0:
            raise ValueError("active_tolerance must be finite and nonnegative")
