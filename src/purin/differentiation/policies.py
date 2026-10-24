"""Policies controlling projection-gradient eligibility and derivative behavior."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GradientPolicy:
    """Policy selecting how projection results participate in differentiation."""

    mode: str = "residual_checked"
    differentiate_coefficients: bool = True
    differentiate_duals: bool = False
    adjoint_tolerance: float = 1e-8

    def __post_init__(self):
        if self.mode not in {"strict", "residual_checked", "always"}:
            raise ValueError("Unknown gradient policy")
        if self.adjoint_tolerance <= 0:
            raise ValueError("adjoint_tolerance must be positive")
