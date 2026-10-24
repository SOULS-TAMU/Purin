"""Repeated local correction with numerical primal/dual warm starts."""

from dataclasses import dataclass
from typing import NamedTuple
from purin.solvers.types import (
    ProjectionInfo,
    ProjectionResult,
    ProjectionTrainingOutput,
)


class CorrectionValues(NamedTuple):
    """Primal, multiplier and diagnostic values passed between correction layers."""

    y: object
    multipliers_eq: object
    multipliers_ineq: object
    gradient_eligible: object

    @property
    def y_tilde(self):
        """Return the corrected primal values using the shared output naming convention."""
        return self.y

    @property
    def forward_gradient_eligible(self):
        """Return the mask identifying instances eligible for projection gradients."""
        return self.gradient_eligible


def correction_layers(step, target, equality_duals, inequality_duals, layers):
    """Single repeated-correction execution used for every formulation."""
    results, eligible = [], None
    for _ in range(layers):
        result = step(target, equality_duals, inequality_duals)
        eligible = (
            result.gradient_eligible
            if eligible is None
            else eligible & result.gradient_eligible
        )
        result = result._replace(gradient_eligible=eligible)
        results.append(result)
        target, equality_duals, inequality_duals = (
            result.y,
            result.multipliers_eq,
            result.multipliers_ineq,
        )
    return tuple(results)


@dataclass(frozen=True)
class Sequential:
    """Definition of repeated local corrections with shared projection resources."""

    correction: object
    layers: int = 1

    def __post_init__(self):
        if not isinstance(self.layers, int) or self.layers < 1:
            raise ValueError("layers must be a positive integer")

    def prepare(self, **kwargs):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        return PreparedSequential(self.correction.prepare(**kwargs), self.layers)


@dataclass(frozen=True)
class PreparedSequential:
    """Prepared sequence of projection corrections and its differentiation policy."""

    correction: object
    layers: int

    def solve_layers(
        self, target, context, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Apply the configured correction layers and return the final correction values."""

        def step(target, multipliers_eq, multipliers_ineq):
            if hasattr(self.correction, "project_training"):
                r = self.correction.project_training(
                    target,
                    context,
                    multipliers_eq=multipliers_eq,
                    multipliers_ineq=multipliers_ineq,
                )
                result = ProjectionResult(
                    r.y_tilde,
                    r.multipliers_eq,
                    r.multipliers_ineq,
                    r.slacks,
                    r.forward_status_codes,
                    r.forward_iterations,
                    r.forward_residual_inf_norm,
                    r.forward_state_finite,
                    r.forward_gradient_eligible,
                )
            else:
                result = self.correction.solve(target, context)
            return result

        return correction_layers(
            step, target, multipliers_eq, multipliers_ineq, self.layers
        )

    def solve(self, target, context):
        """Return the final primal values after sequential correction."""
        return self.solve_layers(target, context)[-1]

    def __call__(self, target, context):
        return self.solve(target, context).y

    def with_info(self, target, context):
        """Evaluate the projection and return its primal values with solver diagnostics."""
        r = self.solve(target, context)
        return r.y, ProjectionInfo(
            r.status, r.iterations, r.residual_inf_norm, r.finite, r.gradient_eligible
        )

    def project_training(
        self, target, context, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        r = self.solve_layers(
            target,
            context,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
        )[-1]
        return ProjectionTrainingOutput(
            r.y,
            r.multipliers_eq,
            r.multipliers_ineq,
            r.slacks,
            r.status,
            r.iterations,
            r.residual_inf_norm,
            r.finite,
            r.gradient_eligible,
        )
