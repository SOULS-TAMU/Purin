"""Compatibility view of per-row diagnostics from prepared projection layers."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectionInfo:
    """Per-instance projection status, residual and convergence diagnostics."""

    forward_status_codes: Any
    forward_iterations: Any
    forward_residual_inf_norm: Any
    forward_state_finite: Any
    gradient_eligible: Any

    @property
    def forward_converged(self):
        """Return the forward solver's per-instance convergence mask."""
        return self.forward_status_codes == 1
