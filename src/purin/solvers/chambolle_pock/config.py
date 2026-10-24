"""Configuration for affine Euclidean Chambolle--Pock projection."""

from purin._utils.validation import validate_metric_diagonal
from dataclasses import dataclass, field
import math
from purin.solvers.config import ImplicitGradientConfig
from purin.solvers.config import LinearSolverConfig


@dataclass(frozen=True)
class ChambollePockConfig:
    """Configure affine distance projection with Chambolle--Pock iterations.

    Attributes:
        tolerance: Positive forward convergence tolerance.
        max_iterations: Positive forward iteration limit.
        accelerated: Use the accelerated forward update when True.
        safety: Step-size safety factor strictly between zero and one.
        norm_iterations: Positive power-iteration count for norm estimation.
        norm_seed: Deterministic seed used by the norm estimator.
        row_scaling: Enable constraint-row normalization.
        norm_bound: Use the sparse operator norm-bound policy.
        linear_solver: Linear-system options for derivative solves.
        implicit_gradient: Implicit derivative policy and adjoint configuration.
        diagonal_preconditioning: Use diagonal step metrics for fixed-step CP.
        metric_diagonal: Optional positive variable-distance metric.

    Notes:
        This configuration controls PURIN's affine projection interface.
        NLPOPTNet's nonlinear frontend maps its local QP options to the dense
        implicit layer, which uses the same CP numerical kernels.
    """

    tolerance: float = 1e-9
    max_iterations: int = 10000
    accelerated: bool = False
    safety: float = 0.95
    norm_iterations: int = 20
    norm_seed: int = 42
    row_scaling: bool = True
    norm_bound: bool = True
    linear_solver: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    implicit_gradient: ImplicitGradientConfig = field(
        default_factory=ImplicitGradientConfig
    )
    diagonal_preconditioning: bool = False

    metric_diagonal: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "metric_diagonal", validate_metric_diagonal(self.metric_diagonal)
        )
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not 0 < self.safety < 1:
            raise ValueError("safety must be between zero and one")
        if not isinstance(self.norm_iterations, int) or self.norm_iterations < 1:
            raise ValueError("norm_iterations must be a positive integer")
