"""Conversions between the NumPy reference containers and JAX pytrees."""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from purin.solvers.primal_dual_interior_point.reference.bounds import (
    BoundIndices as NumpyBoundIndices,
)
from purin.solvers.primal_dual_interior_point.reference.bounds import (
    BoundSlacks as NumpyBoundSlacks,
)
from purin.solvers.primal_dual_interior_point.reference.bounds import (
    BoundValues as NumpyBoundValues,
)
from purin.solvers.primal_dual_interior_point.reference.residuals import (
    KKTResidual as NumpyKKTResidual,
)
from purin.solvers.primal_dual_interior_point.reference.residuals import (
    NLPQuantities as NumpyNLPQuantities,
)
from purin.solvers.primal_dual_interior_point.reference.state import (
    PrimalDualState as NumpyPrimalDualState,
)
from purin.solvers.primal_dual_interior_point.types import BoundIndices
from purin.solvers.primal_dual_interior_point.types import BoundSlacks
from purin.solvers.primal_dual_interior_point.types import BoundValues
from purin.solvers.primal_dual_interior_point.types import KKTResidual
from purin.solvers.primal_dual_interior_point.types import NLPQuantities
from purin.solvers.primal_dual_interior_point.types import PrimalDualState


def state_from_numpy(state: NumpyPrimalDualState) -> PrimalDualState:
    """Convert a host primal-dual state to the JAX state representation."""
    return PrimalDualState(*(jnp.asarray(block) for block in state.blocks()))


def state_to_numpy(state: PrimalDualState) -> NumpyPrimalDualState:
    """Copy a JAX primal-dual state into host NumPy arrays."""
    return NumpyPrimalDualState(*(np.asarray(block) for block in state.blocks()))


def indices_from_numpy(indices: NumpyBoundIndices) -> BoundIndices:
    """Convert host bound indices to their JAX representation."""
    return BoundIndices(
        *(jnp.asarray(block, dtype=jnp.int32) for _, block in indices.named_blocks())
    )


def bounds_from_numpy(bounds: NumpyBoundValues) -> BoundValues:
    """Convert host bound values to their JAX representation."""
    return BoundValues(
        jnp.asarray(bounds.x_lower),
        jnp.asarray(bounds.x_upper),
        jnp.asarray(bounds.s_lower),
        jnp.asarray(bounds.s_upper),
    )


def slacks_from_numpy(slacks: NumpyBoundSlacks) -> BoundSlacks:
    """Convert host bound-slack arrays to their JAX representation."""
    return BoundSlacks(
        jnp.asarray(slacks.x_lower),
        jnp.asarray(slacks.x_upper),
        jnp.asarray(slacks.s_lower),
        jnp.asarray(slacks.s_upper),
    )


def nlp_from_numpy(nlp: NumpyNLPQuantities) -> NLPQuantities:
    """Convert host objective and constraint quantities to JAX arrays."""
    objective = (
        None if nlp.objective_value is None else jnp.asarray(nlp.objective_value)
    )
    return NLPQuantities(
        grad_f=jnp.asarray(nlp.grad_f),
        c=jnp.asarray(nlp.c),
        d=jnp.asarray(nlp.d),
        jac_c=jnp.asarray(nlp.jac_c),
        jac_d=jnp.asarray(nlp.jac_d),
        hessian_lagrangian=jnp.asarray(nlp.hessian_lagrangian),
        objective_value=objective,
    )


def residual_from_numpy(residual: NumpyKKTResidual) -> KKTResidual:
    """Convert host KKT residual blocks to JAX arrays."""
    return KKTResidual(*(jnp.asarray(block) for block in residual.blocks()))
