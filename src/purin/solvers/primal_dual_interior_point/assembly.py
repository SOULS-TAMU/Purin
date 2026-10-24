"""Backend-specific sparse numerical-data assembly."""

from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
from purin.linalg.sparse.operators import local_jacobian_transpose_matvec
from purin.solvers.primal_dual_interior_point.newton import compute_bound_slacks
from purin.solvers.primal_dual_interior_point.newton import compute_sigma
from purin.solvers.primal_dual_interior_point.types import BoundIndices
from purin.solvers.primal_dual_interior_point.types import BoundValues
from purin.solvers.primal_dual_interior_point.types import KKTResidual
from purin.solvers.primal_dual_interior_point.types import PrimalDualState
from purin.solvers.primal_dual_interior_point.types import SparseNumericData

VectorFunction = Callable[[jax.Array, jax.Array], jax.Array]
LocalDerivativeFunction = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


def _residual_inf_norm(residual: KKTResidual) -> jax.Array:
    """Infinity norm without concatenating all residual blocks.

    This is algebraically identical to ``max(abs(residual.flatten()))`` but
    avoids materializing a large temporary vector in every Newton iteration.
    """

    block_norms = tuple(
        jnp.max(jnp.abs(block), initial=0.0) for block in residual.blocks()
    )
    return jnp.max(jnp.stack(block_norms), initial=0.0)


def _assemble_numeric_data(
    state: PrimalDualState,
    parameters: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_jacobian: jax.Array,
    inequality_jacobian: jax.Array,
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
) -> SparseNumericData:
    """Assemble residual and condensed-system data from derivative blocks."""

    equalities = equality_function(state.x, parameters)
    inequalities = inequality_function(state.x, parameters)
    return _assemble_numeric_data_from_constraint_values(
        state,
        parameters,
        equalities,
        inequalities,
        equality_jacobian,
        inequality_jacobian,
        equality_hessian,
        inequality_hessian,
        equality_support,
        equality_mask,
        inequality_support,
        inequality_mask,
        target_parameter_indices,
        indices,
        bounds,
        mu,
    )


def _assemble_numeric_data_from_constraint_values(
    state: PrimalDualState,
    parameters: jax.Array,
    equalities: jax.Array,
    inequalities: jax.Array,
    equality_jacobian: jax.Array,
    inequality_jacobian: jax.Array,
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
) -> SparseNumericData:
    """Assemble sparse residual data from already evaluated constraints."""

    slacks = compute_bound_slacks(state, indices, bounds)

    target = parameters[target_parameter_indices]
    residual_x = state.x - target
    residual_x = residual_x + local_jacobian_transpose_matvec(
        equality_jacobian, equality_support, equality_mask, state.y_c, state.x.shape[0]
    )
    residual_x = residual_x + local_jacobian_transpose_matvec(
        inequality_jacobian,
        inequality_support,
        inequality_mask,
        state.y_d,
        state.x.shape[0],
    )
    residual_x = residual_x.at[indices.x_lower].add(-state.z_L)
    residual_x = residual_x.at[indices.x_upper].add(state.z_U)

    residual_s = -state.y_d
    residual_s = residual_s.at[indices.s_lower].add(-state.v_L)
    residual_s = residual_s.at[indices.s_upper].add(state.v_U)

    residual = KKTResidual(
        x=residual_x,
        s=residual_s,
        y_c=equalities,
        y_d=inequalities - state.s,
        z_L=slacks.x_lower * state.z_L - mu,
        z_U=slacks.x_upper * state.z_U - mu,
        v_L=slacks.s_lower * state.v_L - mu,
        v_U=slacks.s_upper * state.v_U - mu,
    )
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    return SparseNumericData(
        residual=residual,
        slacks=slacks,
        sigma_x=sigma_x,
        sigma_s=sigma_s,
        equality_jacobian=equality_jacobian,
        inequality_jacobian=inequality_jacobian,
        equality_hessian=equality_hessian,
        inequality_hessian=inequality_hessian,
    )


def _evaluate_numeric_data(
    state: PrimalDualState,
    parameters: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_local_derivatives: LocalDerivativeFunction,
    inequality_local_derivatives: LocalDerivativeFunction,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
) -> SparseNumericData:
    equality_jacobian, equality_hessian = equality_local_derivatives(
        state.x, parameters
    )
    inequality_jacobian, inequality_hessian = inequality_local_derivatives(
        state.x, parameters
    )
    return _assemble_numeric_data(
        state,
        parameters,
        equality_function,
        inequality_function,
        equality_jacobian,
        inequality_jacobian,
        equality_hessian,
        inequality_hessian,
        equality_support,
        equality_mask,
        inequality_support,
        inequality_mask,
        target_parameter_indices,
        indices,
        bounds,
        mu,
    )
