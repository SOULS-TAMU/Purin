"""NumPy orchestration and direct primal-dual solver components."""

from purin.solvers.primal_dual_interior_point.reference.bounds import BoundIndices
from purin.solvers.primal_dual_interior_point.reference.bounds import BoundSlacks
from purin.solvers.primal_dual_interior_point.reference.bounds import BoundValues
from purin.solvers.primal_dual_interior_point.reference.bounds import (
    compute_bound_slacks,
)
from purin.solvers.primal_dual_interior_point.reference.iteration import FixedMuOptions
from purin.solvers.primal_dual_interior_point.reference.iteration import FixedMuResult
from purin.solvers.primal_dual_interior_point.reference.iteration import (
    run_fixed_mu_iterations,
)
from purin.solvers.primal_dual_interior_point.reference.state import PrimalDualState


def solve(problem, **kwargs):
    """Solve a problem that exposes the DAEHN ``solve`` protocol."""
    if hasattr(problem, "solve"):
        return problem.solve(**kwargs)
    raise TypeError("problem does not provide solve()")


__all__ = [
    "BoundIndices",
    "BoundSlacks",
    "BoundValues",
    "compute_bound_slacks",
    "FixedMuOptions",
    "FixedMuResult",
    "run_fixed_mu_iterations",
    "PrimalDualState",
    "solve",
]
