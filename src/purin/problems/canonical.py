"""Immutable backend-neutral optimization-problem representations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CanonicalProblem:
    """Numerical functions and static sparsity shared by every backend.

    Public functions use argument order ``(theta, y)``. Jacobians returned by
    ``equality_jacobian`` and ``inequality_jacobian`` have the conventional
    row-major shapes ``(n_constraints, n_y)``; optimization backends transpose
    them once to obtain the documented ``A`` and ``C`` column orientation.
    """

    name: str
    parameter_size: int
    variable_size: int
    equality_size: int
    inequality_size: int
    objective: Callable
    equalities: Callable
    inequalities: Callable
    objective_gradient: Callable
    equality_jacobian: Callable
    inequality_jacobian: Callable
    lagrangian_hessian: Callable
    equality_sparsity: Any = None
    inequality_sparsity: Any = None
    equality_jacobian_values: Callable | None = None
    inequality_jacobian_values: Callable | None = None
    equality_local_derivatives: Callable | None = None
    inequality_local_derivatives: Callable | None = None
    constraint_hessian_vector_product: Callable | None = None
    metadata: Any = None


@dataclass(frozen=True)
class CanonicalQP(CanonicalProblem):
    """Canonical quadratic objective and affine constraints for projection preparation."""

    P: Any = None
    q0: Any = None
    q_parameter_map: Any = None
    A_eq: Any = None
    b_eq0: Any = None
    b_eq_parameter_map: Any = None
    A_ineq: Any = None
    b_ineq0: Any = None
    b_ineq_parameter_map: Any = None
