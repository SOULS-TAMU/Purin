"""Explicit validation of problem/solver combinations."""

from dataclasses import dataclass
from .types import ProjectionMethod


@dataclass(frozen=True)
class SolverCapabilities:
    """Supported problem features and derivative behavior of a projection solver."""

    nonlinear_constraints: bool
    distance: bool = True
    diagonal_qp: bool = False
    reverse_mode: bool = True
    forward_mode: bool = False


def capabilities(method):
    """Look up the declared capabilities of a projection method."""
    name = ProjectionMethod.normalize(method)
    return SolverCapabilities(
        nonlinear_constraints=name
        in {
            ProjectionMethod.FISCHER_BURMEISTER_NEWTON,
            ProjectionMethod.FISCHER_BURMEISTER_REDUCED,
            ProjectionMethod.PRIMAL_DUAL_INTERIOR_POINT,
        },
        diagonal_qp=name is ProjectionMethod.CHAMBOLLE_POCK,
        forward_mode=name is ProjectionMethod.LINEAR_EQUALITY,
    )
