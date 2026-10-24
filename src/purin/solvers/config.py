"""Immutable host-side configurations for projection backends."""

from __future__ import annotations
import math
from purin._utils.validation import validate_metric_diagonal
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LinearSolverConfig:
    """Linear-system solver choices and tolerances for projection steps."""

    tolerance: float = 1e-9
    atol: float = 0.0
    restart: int = 30
    max_iterations: int = 100
    use_preconditioner: bool = True
    mode: str = "auto"

    def __post_init__(self):
        if self.tolerance <= 0.0 or self.atol < 0.0:
            raise ValueError("linear tolerances are invalid")
        if self.restart < 1 or self.max_iterations < 1:
            raise ValueError("linear iteration limits must be positive")


@dataclass(frozen=True)
class ImplicitGradientConfig:
    """Adjoint solve settings for implicit differentiation through a projection."""

    policy: str = "residual_checked"
    inexact_tolerance: float = 1e-6
    adjoint_tolerance: float = 1e-8
    invalid_batch_policy: str = "skip"
    differentiate_problem_parameters: bool = False

    def __post_init__(self):
        if self.policy not in {"strict", "residual_checked", "always"}:
            raise ValueError("invalid gradient policy")
        if self.invalid_batch_policy not in {"skip", "use_valid"}:
            raise ValueError("invalid batch policy")
        if self.inexact_tolerance < 0.0 or self.adjoint_tolerance < 0.0:
            raise ValueError("gradient tolerances must be nonnegative")


@dataclass(frozen=True)
class ProjectionConfig:
    """Primal-dual options with the existing identity-projection defaults."""

    tolerance: float = 1e-9
    max_iterations: int = 100
    mu: float = 1e-4
    backtracking: bool = False
    max_backtracking_steps: int = 8
    backtracking_factor: float = 0.5
    interior_margin: float = 1.0
    linear_solver: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    implicit_gradient: ImplicitGradientConfig = field(
        default_factory=ImplicitGradientConfig
    )
    store_residual_history: bool = False
    transpose_mode: str = "auto"

    metric_diagonal: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "metric_diagonal", validate_metric_diagonal(self.metric_diagonal)
        )
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        if self.mu < 0.0:
            raise ValueError("mu must be nonnegative")
        if self.max_backtracking_steps < 0:
            raise ValueError("max_backtracking_steps must be nonnegative")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must be in (0,1)")


@dataclass(frozen=True)
class LinearEqualityConfig:
    """Preparation and solve options for affine equality projection."""

    regularization: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.regularization) or self.regularization < 0.0:
            raise ValueError("regularization must be finite and nonnegative")


@dataclass(frozen=True)
class FischerBurmeisterConfig:
    """Shared full and pivot-reduced Fischer--Burmeister configuration."""

    # These defaults intentionally match the stable public frontend used by
    # KKT-HardNet before the backend refactor.  The exact-Newton backends can
    # use tighter settings explicitly, but starting every inequality with a
    # unit multiplier and almost no regularization made the first Newton
    # direction unnecessarily large and caused avoidable Armijo failures.
    tolerance: float = 1e-6
    max_iterations: int = 30
    smoothing: float = 1e-8
    initial_slack_floor: float = 1e-3
    initial_multiplier: float = 0.0
    initial_regularization: float = 1e-3
    regularization_growth: float = 10.0
    maximum_regularization: float = 1e8
    maximum_regularization_attempts: int = 8
    pivot_hysteresis: float = 0.05
    pivot_tolerance: float = 1e-12
    armijo_constant: float = 1e-4
    initial_step_size: float = 1.0
    backtracking_factor: float = 0.5
    maximum_backtracking_steps: int = 24
    minimum_step_size: float = 1e-8
    linear_solver: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    implicit_gradient: ImplicitGradientConfig = field(
        default_factory=ImplicitGradientConfig
    )
    transpose_mode: str = "auto"

    def __post_init__(self):
        positive = {
            "tolerance": self.tolerance,
            "regularization_growth": self.regularization_growth,
            "maximum_regularization": self.maximum_regularization,
            "minimum_step_size": self.minimum_step_size,
            "initial_step_size": self.initial_step_size,
        }
        if any(
            (not math.isfinite(value) or value <= 0.0) for value in positive.values()
        ):
            raise ValueError(f"positive finite values required: {positive}")
        nonnegative = {
            "smoothing": self.smoothing,
            "initial_slack_floor": self.initial_slack_floor,
            "initial_multiplier": self.initial_multiplier,
            "initial_regularization": self.initial_regularization,
            "pivot_hysteresis": self.pivot_hysteresis,
            "pivot_tolerance": self.pivot_tolerance,
        }
        if any(
            (not math.isfinite(value) or value < 0.0) for value in nonnegative.values()
        ):
            raise ValueError(f"nonnegative finite values required: {nonnegative}")
        if self.max_iterations < 1 or self.maximum_regularization_attempts < 1:
            raise ValueError("iteration limits must be positive")
        if self.maximum_backtracking_steps < 1:
            raise ValueError("maximum_backtracking_steps must be positive")
        if not 0.0 < self.armijo_constant < 1.0:
            raise ValueError("armijo_constant must be in (0,1)")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must be in (0,1)")
        if self.regularization_growth <= 1.0:
            raise ValueError("regularization_growth must exceed one")
        if str(self.transpose_mode).strip().lower() not in {
            "auto",
            "scatter",
            "gather",
            "bucketed",
        }:
            raise ValueError(
                "transpose_mode must be one of 'auto', 'scatter', 'gather', or 'bucketed'"
            )
