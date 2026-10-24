"""Shared projection statuses, results, and method identifiers."""

from __future__ import annotations
from enum import Enum, IntEnum
from typing import NamedTuple
import jax
from purin.linalg.sparse.types import LinearSolveResult as LinearSolveResult

Array = jax.Array


class ProjectionMethod(str, Enum):
    """Supported projection backends."""

    LINEAR_EQUALITY = "linear-equality"
    CHAMBOLLE_POCK = "chambolle-pock"
    ADMM = "admm"
    FISCHER_BURMEISTER_NEWTON = "fischer-burmeister-newton"
    FISCHER_BURMEISTER_REDUCED = "fischer-burmeister-reduced"
    PRIMAL_DUAL_INTERIOR_POINT = "primal-dual-interior-point"

    @classmethod
    def normalize(cls, value: "ProjectionMethod | str") -> "ProjectionMethod":
        """Resolve a method name or enum to its canonical projection method."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown projection method {value!r}. Supported methods: {supported}."
            ) from error


class SolveStatus(IntEnum):
    """Backend-neutral solver termination status."""

    CONVERGED = 0
    MAX_ITERATIONS = 1
    LINEAR_SOLVE_FAILURE = 2
    NONFINITE = 3
    LINE_SEARCH_FAILURE = 4


class ProjectionResult(NamedTuple):
    """Complete fixed-shape projection result."""

    y: Array
    multipliers_eq: Array
    multipliers_ineq: Array
    slacks: Array
    status: Array
    iterations: Array
    residual_inf_norm: Array
    finite: Array
    gradient_eligible: Array


class ProjectionInfo(NamedTuple):
    """Lightweight diagnostics returned by ``with_info``."""

    forward_status_codes: Array
    forward_iterations: Array
    forward_residual_inf_norm: Array
    forward_state_finite: Array
    gradient_eligible: Array

    @property
    def forward_converged(self):
        """Return the per-instance convergence flags from the forward solve."""
        return self.forward_status_codes == int(SolveStatus.CONVERGED)


class ProjectionTrainingOutput(NamedTuple):
    """Projected primal/latent-dual state used by the training frontend."""

    y_tilde: Array
    multipliers_eq: Array
    multipliers_ineq: Array
    slacks: Array
    forward_status_codes: Array
    forward_iterations: Array
    forward_residual_inf_norm: Array
    forward_state_finite: Array
    forward_gradient_eligible: Array


class ProjectionLayerOutput(NamedTuple):
    """Projection output retained for the custom-VJP training path."""

    y_tilde: Array
    forward_status_codes: Array
    forward_iterations: Array
    forward_residual_inf_norm: Array
    forward_state_finite: Array
    forward_gradient_eligible: Array
