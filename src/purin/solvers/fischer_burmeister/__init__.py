"""Full and sparse pivot-reduced Fischer--Burmeister backends."""

from purin.solvers.fischer_burmeister.common import ORIGIN_DERIVATIVE
from purin.solvers.fischer_burmeister.common import FischerBurmeisterState
from purin.solvers.fischer_burmeister.common import fischer_burmeister
from purin.solvers.fischer_burmeister.common import fischer_burmeister_partials
from purin.solvers.fischer_burmeister.config import FischerBurmeisterConfig
from purin.solvers.fischer_burmeister.newton import FullFischerBurmeisterProjection
from purin.solvers.fischer_burmeister.newton import (
    PreparedFullFischerBurmeisterProjection,
)
from purin.solvers.fischer_burmeister.operators import ReducedFBOperatorBundle
from purin.solvers.fischer_burmeister.operators import ReducedFBSchurBundle
from purin.solvers.fischer_burmeister.operators import build_primal_schur_operator
from purin.solvers.fischer_burmeister.operators import build_reduced_kkt_operator
from purin.solvers.fischer_burmeister.operators import recover_schur_multipliers
from purin.solvers.fischer_burmeister.reduced import (
    PreparedReducedFischerBurmeisterProjection,
)
from purin.solvers.fischer_burmeister.reduced import ReducedFischerBurmeisterProjection
from purin.solvers.fischer_burmeister.reduced import initial_pivots
from purin.solvers.fischer_burmeister.reduced import update_pivots
from purin.solvers.fischer_burmeister.solver import BatchedLinearSolveResult
from purin.solvers.fischer_burmeister.solver import batched_minres
from purin.solvers.fischer_burmeister.solver import batched_pcg
from purin.solvers.fischer_burmeister.solver import solve_symmetric_system

__all__ = [
    "FischerBurmeisterConfig",
    "FischerBurmeisterState",
    "ORIGIN_DERIVATIVE",
    "fischer_burmeister",
    "fischer_burmeister_partials",
    "FullFischerBurmeisterProjection",
    "PreparedFullFischerBurmeisterProjection",
    "ReducedFischerBurmeisterProjection",
    "PreparedReducedFischerBurmeisterProjection",
    "initial_pivots",
    "update_pivots",
    "ReducedFBOperatorBundle",
    "ReducedFBSchurBundle",
    "build_reduced_kkt_operator",
    "build_primal_schur_operator",
    "recover_schur_multipliers",
    "BatchedLinearSolveResult",
    "batched_minres",
    "batched_pcg",
    "solve_symmetric_system",
]
