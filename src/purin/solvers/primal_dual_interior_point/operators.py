"""Primal-dual condensed KKT operators and iterative linear solves."""

from __future__ import annotations
from collections.abc import Callable
import jax
import jax.numpy as jnp
import numpy as np
from purin.linalg.sparse.operators import _batched_local_constraint_hessian_diagonal
from purin.linalg.sparse.operators import _batched_local_constraint_hessian_matvec
from purin.linalg.sparse.operators import _batched_local_jacobian_matvec
from purin.linalg.sparse.operators import (
    _batched_local_jacobian_transpose_bucketed_matvec,
)
from purin.linalg.sparse.operators import (
    _batched_local_jacobian_transpose_gather_matvec,
)
from purin.linalg.sparse.operators import _batched_local_jacobian_transpose_matvec
from purin.linalg.sparse.operators import _batched_schur_diagonal
from purin.solvers.primal_dual_interior_point.assembly import _residual_inf_norm
from purin.solvers.primal_dual_interior_point.newton import (
    recover_bound_multiplier_directions,
)
from purin.solvers.primal_dual_interior_point.types import PrimalDualState
from purin.solvers.primal_dual_interior_point.types import SparseNumericData
from purin.solvers.primal_dual_interior_point.types import SparseProjectionAdjointResult
from purin.solvers.primal_dual_interior_point.types import SparseProjectionLinearization

try:  # Pallas is optional across JAX versions/installations.
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - depends on the installed JAX build.
    pl = None


def pallas_available() -> bool:
    """Return whether the installed JAX exposes Pallas."""

    return pl is not None


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""

    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _pad_second_axis(array, width: int, *, fill_value=0):
    """Pad a rank-two static sparse-layout array to ``width`` columns."""

    host = np.asarray(array)
    if host.ndim != 2:
        raise ValueError("Expected a rank-two sparse-layout array.")
    if host.shape[1] > width:
        raise ValueError("Requested padded width is smaller than the input width.")
    if host.shape[1] == width:
        return host
    pad = np.full((host.shape[0], width - host.shape[1]), fill_value, dtype=host.dtype)
    return np.concatenate((host, pad), axis=1)


def build_affine_constant_kkt_pallas_matvec(
    *,
    jacobian: jax.Array,
    row_support: jax.Array,
    row_mask: jax.Array,
    column_rows: jax.Array,
    column_slots: jax.Array,
    column_mask: jax.Array,
    n_variables: int,
    block_size: int = 64,
    interpret: bool = False,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Build a fused affine condensed-KKT product using Pallas.

    Parameters
    ----------
    jacobian:
        Constant combined equality/inequality local Jacobian with shape
        ``(n_constraints, row_width)``.
    row_support, row_mask:
        Row-oriented sparse layout used for ``J @ x``.
    column_rows, column_slots, column_mask:
        Column-oriented adjacency used for ``J.T @ y``.
    n_variables:
        Width of the primal block.
    block_size:
        Number of output entries produced by one Pallas program.
    interpret:
        Pallas interpreter mode, used by CPU unit tests only.  Production CUDA
        execution uses the Triton/Mosaic GPU lowering selected by JAX.

    Notes
    -----
    The mathematical product is exactly the affine condensed KKT operator::

        row_x = primal_diagonal * vector_x + J.T @ vector_dual
        row_d = J @ vector_x - dual_diagonal * vector_dual

    Nonlinear curvature and parameter-dependent Jacobians deliberately fall
    back to the standard JAX implementation.
    """

    if pl is None:
        raise RuntimeError("Pallas is not available in this JAX installation.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if jacobian.ndim != 2:
        raise ValueError("The Pallas affine KKT kernel requires a rank-two Jacobian.")

    n_constraints = int(jacobian.shape[0])
    row_width = int(jacobian.shape[1])
    max_column_degree = int(column_rows.shape[1])

    # Pallas' Triton lowering requires every vector/tile operation to have a
    # power-of-two element count.  The natural sparse widths are commonly not
    # powers of two (for example nnz=3 and maximum column degree=11).  Pad only
    # the static sparse-layout axes and mask the added entries so the numerical
    # operator is unchanged.
    pallas_block_size = _next_power_of_two(block_size)
    padded_row_width = _next_power_of_two(row_width)
    padded_column_degree = _next_power_of_two(max_column_degree)

    total_width = int(n_variables + n_constraints)
    padded_width = (
        (total_width + pallas_block_size - 1) // pallas_block_size
    ) * pallas_block_size

    jacobian_host = _pad_second_axis(jacobian, padded_row_width, fill_value=0.0)
    row_support_host = _pad_second_axis(row_support, padded_row_width, fill_value=0)
    row_mask_host = _pad_second_axis(row_mask, padded_row_width, fill_value=False)
    column_rows_host = _pad_second_axis(column_rows, padded_column_degree, fill_value=0)
    column_slots_host = _pad_second_axis(
        column_slots, padded_column_degree, fill_value=0
    )
    column_mask_host = _pad_second_axis(
        column_mask, padded_column_degree, fill_value=False
    )

    jacobian = jnp.asarray(jacobian_host)
    row_support = jnp.asarray(row_support_host, dtype=jnp.int32)
    row_mask = jnp.asarray(row_mask_host, dtype=bool)
    column_rows = jnp.asarray(column_rows_host, dtype=jnp.int32)
    column_slots = jnp.asarray(column_slots_host, dtype=jnp.int32)
    column_mask = jnp.asarray(column_mask_host, dtype=bool)

    def kernel(
        vector_ref,
        jacobian_ref,
        row_support_ref,
        row_mask_ref,
        column_rows_ref,
        column_slots_ref,
        column_mask_ref,
        primal_diagonal_ref,
        dual_diagonal_ref,
        output_ref,
    ):
        batch_index = pl.program_id(0)
        output_block = pl.program_id(1)
        lanes = jnp.arange(pallas_block_size, dtype=jnp.int32)
        columns = output_block * pallas_block_size + lanes
        valid = columns < total_width
        is_primal = columns < n_variables

        # Primal block: diagonal contribution plus J.T @ vector_dual.
        primal_columns = jnp.clip(columns, 0, max(n_variables - 1, 0))
        degree_slots = jnp.arange(padded_column_degree, dtype=jnp.int32)
        gathered_rows = column_rows_ref[primal_columns[:, None], degree_slots[None, :]]
        gathered_slots = column_slots_ref[
            primal_columns[:, None], degree_slots[None, :]
        ]
        gathered_mask = column_mask_ref[primal_columns[:, None], degree_slots[None, :]]
        primal_valid = valid[:, None] & is_primal[:, None] & gathered_mask
        transpose_values = jacobian_ref[gathered_rows, gathered_slots] * primal_valid
        transpose_duals = (
            vector_ref[batch_index, n_variables + gathered_rows] * primal_valid
        )
        transpose_product = jnp.sum(transpose_values * transpose_duals, axis=1)
        primal_vector = vector_ref[batch_index, primal_columns] * (valid & is_primal)
        primal_diagonal = primal_diagonal_ref[batch_index, primal_columns] * (
            valid & is_primal
        )
        primal_output = primal_diagonal * primal_vector + transpose_product

        # Constraint block: J @ vector_x minus the equality/inequality diagonal.
        constraint_rows = jnp.clip(columns - n_variables, 0, max(n_constraints - 1, 0))
        row_slots = jnp.arange(padded_row_width, dtype=jnp.int32)
        supports = row_support_ref[constraint_rows[:, None], row_slots[None, :]]
        support_mask = row_mask_ref[constraint_rows[:, None], row_slots[None, :]]
        dual_valid = valid[:, None] & (~is_primal[:, None]) & support_mask
        row_values = (
            jacobian_ref[constraint_rows[:, None], row_slots[None, :]] * dual_valid
        )
        primal_values = vector_ref[batch_index, supports] * dual_valid
        forward_product = jnp.sum(row_values * primal_values, axis=1)
        dual_vector = vector_ref[batch_index, n_variables + constraint_rows] * (
            valid & ~is_primal
        )
        dual_diagonal = dual_diagonal_ref[batch_index, constraint_rows] * (
            valid & ~is_primal
        )
        dual_output = forward_product - dual_diagonal * dual_vector

        output_ref[batch_index, columns] = jnp.where(
            valid,
            jnp.where(is_primal, primal_output, dual_output),
            jnp.asarray(0.0, dtype=vector_ref.dtype),
        )

    def matvec(
        vector: jax.Array, primal_diagonal: jax.Array, dual_diagonal: jax.Array
    ) -> jax.Array:
        if vector.ndim != 2:
            raise ValueError("vector must be rank two.")
        if int(vector.shape[1]) != total_width:
            raise ValueError(
                f"Expected vector width {total_width}, received {vector.shape[1]}."
            )
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (int(vector.shape[0]), padded_width), vector.dtype
            ),
            grid=(int(vector.shape[0]), padded_width // pallas_block_size),
            interpret=interpret,
            name="pyopt_affine_constant_kkt",
        )
        result = call(
            vector,
            jacobian,
            row_support,
            row_mask,
            column_rows,
            column_slots,
            column_mask,
            primal_diagonal,
            dual_diagonal,
        )
        return result[:, :total_width]

    return matvec


def _solve_condensed_direction_batch(
    state: PrimalDualState,
    numeric: SparseNumericData,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    indices,
    regularization,
    *,
    tolerance,
    absolute_tolerance,
    max_iterations,
    restart,
    use_preconditioner,
    constraints_affine=False,
    equality_full_support=False,
    inequality_full_support=False,
    combined_full_support=False,
    combined_support=None,
    combined_mask=None,
    column_rows=None,
    column_slots=None,
    column_mask=None,
    column_buckets=None,
    constant_combined_jacobian=None,
    constant_combined_jacobian_squared=None,
    affine_kkt_matvec=None,
    gmres_unroll=1,
    arnoldi_prefix_block=0,
):
    """Solve batched condensed KKT directions using the selected linear solver."""
    batch_size, n_variables = state.x.shape
    n_equalities = state.y_c.shape[1]

    # Build the reduced blocks directly instead of materializing the generic
    # full reduced-KKT RHS and slicing it again.  The scatter corrections are
    # exactly the same algebra as ``build_reduced_kkt_rhs``.
    rhs_x = -numeric.residual.x
    rhs_s = -numeric.residual.s
    rhs_x = rhs_x.at[:, indices.x_lower].add(
        -numeric.residual.z_L / numeric.slacks.x_lower
    )
    rhs_x = rhs_x.at[:, indices.x_upper].add(
        numeric.residual.z_U / numeric.slacks.x_upper
    )
    rhs_s = rhs_s.at[:, indices.s_lower].add(
        -numeric.residual.v_L / numeric.slacks.s_lower
    )
    rhs_s = rhs_s.at[:, indices.s_upper].add(
        numeric.residual.v_U / numeric.slacks.s_upper
    )
    rhs_c = -numeric.residual.y_c
    rhs_d = -numeric.residual.y_d

    slack_diagonal = numeric.sigma_s + regularization.delta_s
    inverse_slack = 1.0 / slack_diagonal
    rhs = jnp.concatenate((rhs_x, rhs_c, rhs_d + inverse_slack * rhs_s), axis=1)

    def split(vector):
        c_end_local = n_variables + n_equalities
        return (
            vector[:, :n_variables],
            vector[:, n_variables:c_end_local],
            vector[:, c_end_local:],
        )

    if constant_combined_jacobian is not None:
        combined_jacobian = constant_combined_jacobian
    elif n_equalities == 0:
        combined_jacobian = numeric.inequality_jacobian
        combined_full_support = inequality_full_support
    elif state.y_d.shape[1] == 0:
        combined_jacobian = numeric.equality_jacobian
        combined_full_support = equality_full_support
    else:
        equality_jacobian = numeric.equality_jacobian
        inequality_jacobian = numeric.inequality_jacobian
        if equality_jacobian.ndim != inequality_jacobian.ndim:
            if equality_jacobian.ndim == 2:
                equality_jacobian = jnp.broadcast_to(
                    equality_jacobian, (batch_size,) + equality_jacobian.shape
                )
            else:
                inequality_jacobian = jnp.broadcast_to(
                    inequality_jacobian, (batch_size,) + inequality_jacobian.shape
                )
        axis = 1 if equality_jacobian.ndim == 3 else 0
        combined_jacobian = jnp.concatenate(
            (equality_jacobian, inequality_jacobian), axis=axis
        )

    use_column_gather = (
        column_rows is not None and column_slots is not None and column_mask is not None
    )
    use_column_buckets = column_buckets is not None

    primal_kkt_diagonal = 1.0 + numeric.sigma_x + regularization.delta_x
    equality_dual_diagonal = jnp.full(
        (batch_size, n_equalities), regularization.delta_c, dtype=state.x.dtype
    )
    inequality_dual_diagonal = inverse_slack + regularization.delta_d
    dual_kkt_diagonal = jnp.concatenate(
        (equality_dual_diagonal, inequality_dual_diagonal), axis=1
    )

    def matvec(vector):
        with jax.named_scope("pyopt_kkt_matvec"):
            if affine_kkt_matvec is not None:
                return affine_kkt_matvec(vector, primal_kkt_diagonal, dual_kkt_diagonal)

            vector_x, vector_c, vector_d = split(vector)
            vector_dual = vector[:, n_variables:]
            if constraints_affine:
                constraint_hvp = jnp.zeros_like(vector_x)
            else:
                constraint_hvp = _batched_local_constraint_hessian_matvec(
                    numeric.equality_hessian,
                    numeric.inequality_hessian,
                    equality_support,
                    equality_mask,
                    inequality_support,
                    inequality_mask,
                    state.y_c,
                    state.y_d,
                    vector_x,
                )
            if use_column_buckets:
                transpose_product = _batched_local_jacobian_transpose_bucketed_matvec(
                    combined_jacobian, column_buckets, vector_dual, n_variables
                )
            elif use_column_gather:
                transpose_product = _batched_local_jacobian_transpose_gather_matvec(
                    combined_jacobian,
                    column_rows,
                    column_slots,
                    column_mask,
                    vector_dual,
                )
            else:
                transpose_product = _batched_local_jacobian_transpose_matvec(
                    combined_jacobian,
                    combined_support,
                    combined_mask,
                    vector_dual,
                    n_variables,
                    full_support=combined_full_support,
                )
            row_x = primal_kkt_diagonal * vector_x + constraint_hvp + transpose_product
            constraint_rows = _batched_local_jacobian_matvec(
                combined_jacobian,
                combined_support,
                combined_mask,
                vector_x,
                full_support=combined_full_support,
            )
            row_c = (
                constraint_rows[:, :n_equalities] - regularization.delta_c * vector_c
            )
            row_d = (
                constraint_rows[:, n_equalities:]
                - (inverse_slack + regularization.delta_d) * vector_d
            )
            return jnp.concatenate((row_x, row_c, row_d), axis=1)

    if use_preconditioner:
        if constraints_affine:
            hessian_diagonal = jnp.zeros_like(state.x)
        else:
            hessian_diagonal = _batched_local_constraint_hessian_diagonal(
                numeric.equality_hessian,
                numeric.inequality_hessian,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                state.y_c,
                state.y_d,
                n_variables,
            )
        primal_diagonal = jnp.maximum(
            jnp.abs(1.0 + numeric.sigma_x + regularization.delta_x + hessian_diagonal),
            1e-8,
        )
        inverse_primal = 1.0 / primal_diagonal
        equality_base = jnp.full(
            (batch_size, n_equalities),
            max(abs(float(regularization.delta_c)), 1e-8),
            dtype=state.x.dtype,
        )
        inequality_base = jnp.abs(inverse_slack + regularization.delta_d)
        combined_schur = jnp.maximum(
            _batched_schur_diagonal(
                combined_jacobian,
                combined_support,
                combined_mask,
                inverse_primal,
                jnp.concatenate((equality_base, inequality_base), axis=1),
                full_support=combined_full_support,
                jacobian_squared=constant_combined_jacobian_squared,
            ),
            1e-8,
        )
        equality_schur = combined_schur[:, :n_equalities]
        inequality_schur = combined_schur[:, n_equalities:]
        diagonal = jnp.concatenate(
            (primal_diagonal, equality_schur, inequality_schur), axis=1
        )

        def preconditioner(value):
            return value / diagonal
    else:

        def preconditioner(value):
            return value

    nonlinear_residual_inf = jax.vmap(_residual_inf_norm)(numeric.residual)
    forcing = jnp.minimum(0.1, jnp.sqrt(jnp.maximum(nonlinear_residual_inf, 0.0)))
    accepted_linear_residual = jnp.maximum(
        1e-10, forcing * jnp.maximum(nonlinear_residual_inf, 1e-16)
    )

    solution, info, linear_residual = _batched_restarted_gmres(
        matvec,
        rhs,
        preconditioner,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        accepted_residual_threshold=accepted_linear_residual,
        restart=min(restart, max_iterations),
        max_iterations=max_iterations,
        unroll=gmres_unroll,
        arnoldi_prefix_block=arnoldi_prefix_block,
    )
    delta_x, delta_y_c, delta_y_d = split(solution)
    delta_s = inverse_slack * (rhs_s + delta_y_d)
    recovered = jax.vmap(
        recover_bound_multiplier_directions, in_axes=(0, None, 0, 0, 0, 0)
    )(state, indices, numeric.slacks, numeric.residual, delta_x, delta_s)
    delta_z_L, delta_z_U, delta_v_L, delta_v_U = recovered
    direction = PrimalDualState(
        x=delta_x,
        s=delta_s,
        y_c=delta_y_c,
        y_d=delta_y_d,
        z_L=delta_z_L,
        z_U=delta_z_U,
        v_L=delta_v_L,
        v_U=delta_v_U,
    )
    regular = jnp.all(jnp.isfinite(inverse_slack), axis=1)
    return direction, info, jnp.where(regular, linear_residual, jnp.inf)


def _combine_projection_jacobians(
    state: PrimalDualState,
    numeric: SparseNumericData,
    *,
    constant_combined_jacobian=None,
):
    """Return one equality/inequality Jacobian block with a stable batch layout."""

    batch_size = state.x.shape[0]
    n_equalities = state.y_c.shape[1]
    n_inequalities = state.y_d.shape[1]
    if constant_combined_jacobian is not None:
        return constant_combined_jacobian
    if n_equalities == 0:
        return numeric.inequality_jacobian
    if n_inequalities == 0:
        return numeric.equality_jacobian
    equality_jacobian = numeric.equality_jacobian
    inequality_jacobian = numeric.inequality_jacobian
    if equality_jacobian.ndim != inequality_jacobian.ndim:
        if equality_jacobian.ndim == 2:
            equality_jacobian = jnp.broadcast_to(
                equality_jacobian, (batch_size,) + equality_jacobian.shape
            )
        else:
            inequality_jacobian = jnp.broadcast_to(
                inequality_jacobian, (batch_size,) + inequality_jacobian.shape
            )
    axis = 1 if equality_jacobian.ndim == 3 else 0
    return jnp.concatenate((equality_jacobian, inequality_jacobian), axis=axis)


def _build_sparse_projection_linearization(
    state: PrimalDualState,
    numeric: SparseNumericData,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    regularization,
    *,
    use_preconditioner,
    constraints_affine,
    equality_full_support,
    inequality_full_support,
    combined_full_support,
    combined_support,
    combined_mask,
    constant_combined_jacobian=None,
    constant_combined_jacobian_squared=None,
) -> SparseProjectionLinearization:
    """Build and cache all final-KKT data needed by repeated adjoint solves."""

    batch_size, n_variables = state.x.shape
    n_equalities = state.y_c.shape[1]
    combined_jacobian = _combine_projection_jacobians(
        state, numeric, constant_combined_jacobian=constant_combined_jacobian
    )
    inverse_slack = 1.0 / (numeric.sigma_s + regularization.delta_s)
    primal_kkt_diagonal = 1.0 + numeric.sigma_x + regularization.delta_x
    equality_dual_diagonal = jnp.full(
        (batch_size, n_equalities), regularization.delta_c, dtype=state.x.dtype
    )
    dual_kkt_diagonal = jnp.concatenate(
        (equality_dual_diagonal, inverse_slack + regularization.delta_d), axis=1
    )

    if use_preconditioner:
        if constraints_affine:
            hessian_diagonal = jnp.zeros_like(state.x)
        else:
            hessian_diagonal = _batched_local_constraint_hessian_diagonal(
                numeric.equality_hessian,
                numeric.inequality_hessian,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                state.y_c,
                state.y_d,
                n_variables,
            )
        primal_diagonal = jnp.maximum(
            jnp.abs(primal_kkt_diagonal + hessian_diagonal), 1e-8
        )
        inverse_primal = 1.0 / primal_diagonal
        equality_base = jnp.full(
            (batch_size, n_equalities),
            max(abs(float(regularization.delta_c)), 1e-8),
            dtype=state.x.dtype,
        )
        inequality_base = jnp.abs(inverse_slack + regularization.delta_d)
        combined_schur = jnp.maximum(
            _batched_schur_diagonal(
                combined_jacobian,
                combined_support,
                combined_mask,
                inverse_primal,
                jnp.concatenate((equality_base, inequality_base), axis=1),
                full_support=combined_full_support,
                jacobian_squared=constant_combined_jacobian_squared,
            ),
            1e-8,
        )
        preconditioner_diagonal = jnp.concatenate(
            (primal_diagonal, combined_schur), axis=1
        )
    else:
        preconditioner_diagonal = jnp.empty((batch_size, 0), dtype=state.x.dtype)

    return SparseProjectionLinearization(
        combined_jacobian=combined_jacobian,
        equality_hessian=numeric.equality_hessian,
        inequality_hessian=numeric.inequality_hessian,
        inverse_slack=inverse_slack,
        primal_kkt_diagonal=primal_kkt_diagonal,
        dual_kkt_diagonal=dual_kkt_diagonal,
        preconditioner_diagonal=preconditioner_diagonal,
    )


def _solve_condensed_yhat_adjoint_batch(
    state: PrimalDualState,
    linearization: SparseProjectionLinearization,
    grad_x: jax.Array,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    regularization,
    *,
    tolerance,
    absolute_tolerance,
    max_iterations,
    restart,
    use_preconditioner,
    constraints_affine=False,
    equality_full_support=False,
    inequality_full_support=False,
    combined_full_support=False,
    combined_support=None,
    combined_mask=None,
    column_rows=None,
    column_slots=None,
    column_mask=None,
    column_buckets=None,
    affine_kkt_matvec=None,
    gmres_unroll=1,
    arnoldi_prefix_block=0,
    solver_mode="gmres",
    initial_solution=None,
) -> SparseProjectionAdjointResult:
    """Solve the final condensed adjoint using saved forward linearization data."""

    batch_size, n_variables = state.x.shape
    n_equalities = state.y_c.shape[1]
    n_inequalities = state.y_d.shape[1]
    rhs = jnp.pad(grad_x, ((0, 0), (0, n_equalities + n_inequalities)))

    def split(vector):
        c_end_local = n_variables + n_equalities
        return (
            vector[:, :n_variables],
            vector[:, n_variables:c_end_local],
            vector[:, c_end_local:],
        )

    combined_jacobian = linearization.combined_jacobian
    inverse_slack = linearization.inverse_slack
    primal_kkt_diagonal = linearization.primal_kkt_diagonal
    dual_kkt_diagonal = linearization.dual_kkt_diagonal
    use_column_gather = (
        column_rows is not None and column_slots is not None and column_mask is not None
    )
    use_column_buckets = column_buckets is not None

    def matvec(vector):
        with jax.named_scope("pyopt_adjoint_kkt_matvec"):
            if affine_kkt_matvec is not None:
                return affine_kkt_matvec(vector, primal_kkt_diagonal, dual_kkt_diagonal)
            vector_x, vector_c, vector_d = split(vector)
            vector_dual = vector[:, n_variables:]
            if constraints_affine:
                constraint_hvp = jnp.zeros_like(vector_x)
            else:
                constraint_hvp = _batched_local_constraint_hessian_matvec(
                    linearization.equality_hessian,
                    linearization.inequality_hessian,
                    equality_support,
                    equality_mask,
                    inequality_support,
                    inequality_mask,
                    state.y_c,
                    state.y_d,
                    vector_x,
                )
            if use_column_buckets:
                transpose_product = _batched_local_jacobian_transpose_bucketed_matvec(
                    combined_jacobian, column_buckets, vector_dual, n_variables
                )
            elif use_column_gather:
                transpose_product = _batched_local_jacobian_transpose_gather_matvec(
                    combined_jacobian,
                    column_rows,
                    column_slots,
                    column_mask,
                    vector_dual,
                )
            else:
                transpose_product = _batched_local_jacobian_transpose_matvec(
                    combined_jacobian,
                    combined_support,
                    combined_mask,
                    vector_dual,
                    n_variables,
                    full_support=combined_full_support,
                )
            row_x = primal_kkt_diagonal * vector_x + constraint_hvp + transpose_product
            constraint_rows = _batched_local_jacobian_matvec(
                combined_jacobian,
                combined_support,
                combined_mask,
                vector_x,
                full_support=combined_full_support,
            )
            row_c = (
                constraint_rows[:, :n_equalities] - regularization.delta_c * vector_c
            )
            row_d = (
                constraint_rows[:, n_equalities:]
                - (inverse_slack + regularization.delta_d) * vector_d
            )
            return jnp.concatenate((row_x, row_c, row_d), axis=1)

    if use_preconditioner:
        diagonal = linearization.preconditioner_diagonal

        def preconditioner(value):
            return value / diagonal
    else:

        def preconditioner(value):
            return value

    if initial_solution is None:
        initial_solution = jnp.zeros_like(rhs)

    if solver_mode == "dense":
        width = rhs.shape[1]
        identity = jnp.eye(width, dtype=rhs.dtype)
        columns = jax.vmap(
            lambda column: matvec(jnp.broadcast_to(column, rhs.shape)), out_axes=2
        )(identity)
        solution = jax.vmap(jnp.linalg.solve)(columns, rhs)
        residual = matvec(solution) - rhs
        linear_residual = jnp.max(jnp.abs(residual), axis=1, initial=0.0)
        rhs_scale = jnp.max(jnp.abs(rhs), axis=1, initial=0.0)
        threshold = jnp.maximum(absolute_tolerance, tolerance * rhs_scale)
        finite = jnp.all(jnp.isfinite(solution), axis=1) & jnp.isfinite(linear_residual)
        info = jnp.where(
            ~finite, 2, jnp.where(linear_residual <= threshold, 0, 1)
        ).astype(jnp.int32)
        linear_iterations = jnp.ones((batch_size,), dtype=jnp.int32)
    else:
        solution, info, linear_residual, linear_iterations = _batched_restarted_gmres(
            matvec,
            rhs,
            preconditioner,
            tolerance=tolerance,
            absolute_tolerance=absolute_tolerance,
            accepted_residual_threshold=None,
            restart=min(restart, max_iterations),
            max_iterations=max_iterations,
            unroll=gmres_unroll,
            arnoldi_prefix_block=arnoldi_prefix_block,
            initial_solution=initial_solution,
            return_iterations=True,
        )

    lambda_x, lambda_c, lambda_d = split(solution)
    regular = jnp.all(jnp.isfinite(inverse_slack), axis=1)
    linear_residual = jnp.where(regular, linear_residual, jnp.inf)
    info = jnp.where(regular, info, jnp.asarray(2, dtype=info.dtype))
    return SparseProjectionAdjointResult(
        x=lambda_x,
        y_c=lambda_c,
        y_d=lambda_d,
        linear_status=info,
        linear_residual_inf_norm=linear_residual,
        linear_iterations=linear_iterations,
    )


def _batched_restarted_gmres(
    matvec,
    rhs,
    right_preconditioner,
    *,
    tolerance: float,
    absolute_tolerance: float,
    accepted_residual_threshold: jax.Array | None = None,
    restart: int,
    max_iterations: int,
    unroll: int = 1,
    arnoldi_prefix_block: int = 0,
    initial_solution: jax.Array | None = None,
    return_iterations: bool = False,
):
    """Restarted batched GMRES with incremental Givens rotations.

    The Krylov basis remains fixed-width for JAX compilation, but Arnoldi work
    stops as soon as every active row satisfies the linear residual estimate.
    The final restart block is also capped by the exact remaining iteration
    budget, so ``max_iterations`` is never exceeded.
    """

    batch_size, width = rhs.shape
    restart = int(min(restart, max_iterations, width))
    dtype = rhs.dtype
    epsilon = jnp.asarray(1e-30, dtype=dtype)
    breakdown_epsilon = jnp.asarray(1e-14, dtype=dtype)
    rhs_scale = jnp.max(jnp.abs(rhs), axis=1, initial=0.0)
    threshold = jnp.maximum(absolute_tolerance, tolerance * rhs_scale)
    if accepted_residual_threshold is not None:
        inexact_cap = jnp.maximum(1e-10, 1e-8 * rhs_scale)
        threshold = jnp.maximum(
            threshold, jnp.minimum(accepted_residual_threshold, inexact_cap)
        )

    prefix_block = int(arnoldi_prefix_block)
    if prefix_block <= 0 or prefix_block >= restart:
        prefix_sizes = (restart,)
    else:
        prefix_sizes = tuple(range(prefix_block, restart, prefix_block)) + (restart,)

    def orthogonalize_prefix(basis_, image, iteration):
        """Two-pass MGS over a small static prefix bucket.

        Future Krylov vectors are zero, so rounding the active prefix upward to
        a bucket boundary is algebraically identical while avoiding the full
        restart-width dot products used by the generic GPU graph.
        """

        def make_branch(prefix_size):
            def branch(operand):
                basis_value, image_value = operand
                active_basis = basis_value[:, :prefix_size, :]
                coefficients_prefix = jnp.einsum(
                    "bid,bd->bi", active_basis, image_value
                )
                image_value = image_value - jnp.einsum(
                    "bi,bid->bd", coefficients_prefix, active_basis
                )
                correction_prefix = jnp.einsum("bid,bd->bi", active_basis, image_value)
                coefficients_prefix = coefficients_prefix + correction_prefix
                image_value = image_value - jnp.einsum(
                    "bi,bid->bd", correction_prefix, active_basis
                )
                coefficients = jnp.pad(
                    coefficients_prefix, ((0, 0), (0, restart - prefix_size))
                )
                return coefficients, image_value

            return branch

        if len(prefix_sizes) == 1:
            return make_branch(prefix_sizes[0])((basis_, image))
        branch_index = jnp.minimum(iteration // prefix_block, len(prefix_sizes) - 1)
        return jax.lax.switch(
            branch_index,
            tuple(make_branch(size) for size in prefix_sizes),
            (basis_, image),
        )

    def apply_rotation_prefix(column, cosines, sines, iteration):
        def apply_previous_rotation(rotation_index, rotated_column):
            first = rotated_column[:, rotation_index]
            second = rotated_column[:, rotation_index + 1]
            cosine = cosines[:, rotation_index]
            sine = sines[:, rotation_index]
            rotated_first = cosine * first + sine * second
            rotated_second = -sine * first + cosine * second
            apply = rotation_index < iteration
            rotated_column = rotated_column.at[:, rotation_index].set(
                jnp.where(apply, rotated_first, first)
            )
            rotated_column = rotated_column.at[:, rotation_index + 1].set(
                jnp.where(apply, rotated_second, second)
            )
            return rotated_column

        def make_branch(prefix_size):
            return lambda value: jax.lax.fori_loop(
                0, prefix_size, apply_previous_rotation, value, unroll=unroll
            )

        if len(prefix_sizes) == 1:
            return make_branch(prefix_sizes[0])(column)
        branch_index = jnp.minimum(iteration // prefix_block, len(prefix_sizes) - 1)
        return jax.lax.switch(
            branch_index, tuple(make_branch(size) for size in prefix_sizes), column
        )

    def back_substitute(r_factor, projected_rhs, row_steps):
        coefficients = jnp.zeros((batch_size, restart), dtype=dtype)

        def body(reverse_index, values):
            row = restart - 1 - reverse_index
            trailing = jnp.sum(r_factor[:, row, :] * values, axis=1)
            numerator = projected_rhs[:, row] - trailing
            diagonal = r_factor[:, row, row]
            safe_diagonal = jnp.where(
                jnp.abs(diagonal) > epsilon, diagonal, jnp.ones_like(diagonal)
            )
            value = numerator / safe_diagonal
            value = jnp.where(row < row_steps, value, 0.0)
            return values.at[:, row].set(value)

        return jax.lax.fori_loop(0, restart, body, coefficients, unroll=unroll)

    def cycle_body(carry):
        completed_iterations, x, residual_inf, total_row_iterations = carry
        remaining_iterations = max_iterations - completed_iterations
        cycle_limit = jnp.minimum(
            jnp.asarray(restart, dtype=jnp.int32), remaining_iterations
        )

        residual = rhs - matvec(x)
        beta = jnp.linalg.norm(residual, axis=1)
        safe_beta = jnp.maximum(beta, epsilon)
        cycle_active = residual_inf > threshold

        basis = jnp.zeros((batch_size, restart + 1, width), dtype=dtype)
        basis = basis.at[:, 0, :].set(residual / safe_beta[:, None])
        preconditioned_basis = jnp.zeros((batch_size, restart, width), dtype=dtype)
        hessenberg = jnp.zeros((batch_size, restart + 1, restart), dtype=dtype)
        cosines = jnp.zeros((batch_size, restart), dtype=dtype)
        sines = jnp.zeros((batch_size, restart), dtype=dtype)
        projected_rhs = (
            jnp.zeros((batch_size, restart + 1), dtype=dtype).at[:, 0].set(beta)
        )
        row_steps = jnp.zeros((batch_size,), dtype=jnp.int32)
        arnoldi_iteration = jnp.asarray(0, dtype=jnp.int32)

        def arnoldi_condition(inner_carry):
            (
                iteration,
                _basis,
                _preconditioned,
                _hessenberg,
                _cosines,
                _sines,
                _projected_rhs,
                active,
                _row_steps,
            ) = inner_carry
            return (iteration < cycle_limit) & jnp.any(active)

        def arnoldi_body(inner_carry):
            (
                iteration,
                basis_,
                preconditioned_,
                hessenberg_,
                cosines_,
                sines_,
                projected_rhs_,
                active,
                row_steps_,
            ) = inner_carry

            vector = basis_[:, iteration, :]
            preconditioned_vector = right_preconditioner(vector)
            old_preconditioned = preconditioned_[:, iteration, :]
            preconditioned_ = preconditioned_.at[:, iteration, :].set(
                jnp.where(active[:, None], preconditioned_vector, old_preconditioned)
            )

            image = matvec(preconditioned_vector)
            with jax.named_scope("pyopt_gmres_orthogonalize"):
                coefficients, image = orthogonalize_prefix(basis_, image, iteration)

            next_norm = jnp.linalg.norm(image, axis=1)
            safe_norm = jnp.maximum(next_norm, epsilon)
            next_vector = image / safe_norm[:, None]
            old_next_vector = basis_[:, iteration + 1, :]
            basis_ = basis_.at[:, iteration + 1, :].set(
                jnp.where(active[:, None], next_vector, old_next_vector)
            )

            column = jnp.zeros((batch_size, restart + 1), dtype=dtype)
            column = column.at[:, :restart].set(coefficients)
            column = column.at[:, iteration + 1].set(next_norm)

            with jax.named_scope("pyopt_gmres_givens"):
                column = apply_rotation_prefix(column, cosines_, sines_, iteration)

            diagonal = column[:, iteration]
            subdiagonal = column[:, iteration + 1]
            rotation_norm = jnp.hypot(diagonal, subdiagonal)
            safe_rotation_norm = jnp.maximum(rotation_norm, epsilon)
            new_cosine = diagonal / safe_rotation_norm
            new_sine = subdiagonal / safe_rotation_norm
            new_cosine = jnp.where(
                rotation_norm > epsilon, new_cosine, jnp.ones_like(new_cosine)
            )
            new_sine = jnp.where(
                rotation_norm > epsilon, new_sine, jnp.zeros_like(new_sine)
            )

            old_cosine = cosines_[:, iteration]
            old_sine = sines_[:, iteration]
            cosines_ = cosines_.at[:, iteration].set(
                jnp.where(active, new_cosine, old_cosine)
            )
            sines_ = sines_.at[:, iteration].set(jnp.where(active, new_sine, old_sine))

            column = column.at[:, iteration].set(rotation_norm)
            column = column.at[:, iteration + 1].set(0.0)
            old_column = hessenberg_[:, :, iteration]
            hessenberg_ = hessenberg_.at[:, :, iteration].set(
                jnp.where(active[:, None], column, old_column)
            )

            projected_current = projected_rhs_[:, iteration]
            projected_next = projected_rhs_[:, iteration + 1]
            rotated_current = new_cosine * projected_current + new_sine * projected_next
            rotated_next = -new_sine * projected_current + new_cosine * projected_next
            projected_rhs_ = projected_rhs_.at[:, iteration].set(
                jnp.where(active, rotated_current, projected_current)
            )
            projected_rhs_ = projected_rhs_.at[:, iteration + 1].set(
                jnp.where(active, rotated_next, projected_next)
            )

            row_steps_ = jnp.where(active, iteration + 1, row_steps_)
            residual_estimate = jnp.abs(rotated_next)
            finite = (
                jnp.isfinite(residual_estimate)
                & jnp.isfinite(rotation_norm)
                & jnp.all(jnp.isfinite(next_vector), axis=1)
            )
            happy_breakdown = next_norm <= breakdown_epsilon
            active = (
                active & finite & ~happy_breakdown & (residual_estimate > threshold)
            )

            return (
                iteration + 1,
                basis_,
                preconditioned_,
                hessenberg_,
                cosines_,
                sines_,
                projected_rhs_,
                active,
                row_steps_,
            )

        (
            arnoldi_iterations,
            _basis,
            preconditioned_basis,
            hessenberg,
            _cosines,
            _sines,
            projected_rhs,
            _active,
            row_steps,
        ) = jax.lax.while_loop(
            arnoldi_condition,
            arnoldi_body,
            (
                arnoldi_iteration,
                basis,
                preconditioned_basis,
                hessenberg,
                cosines,
                sines,
                projected_rhs,
                cycle_active,
                row_steps,
            ),
        )

        r_factor = hessenberg[:, :restart, :]
        coefficients = back_substitute(r_factor, projected_rhs[:, :restart], row_steps)
        correction = jnp.einsum("bi,bid->bd", coefficients, preconditioned_basis)
        candidate = x + correction
        candidate_residual = rhs - matvec(candidate)
        candidate_norm = jnp.max(jnp.abs(candidate_residual), axis=1, initial=0.0)
        x = jnp.where(cycle_active[:, None], candidate, x)
        residual_inf = jnp.where(cycle_active, candidate_norm, residual_inf)
        return (
            completed_iterations + arnoldi_iterations,
            x,
            residual_inf,
            total_row_iterations + row_steps,
        )

    def cycle_condition(carry):
        completed_iterations, _x, residual_inf, _row_iterations = carry
        return (completed_iterations < max_iterations) & jnp.any(
            residual_inf > threshold
        )

    if initial_solution is None:
        initial_solution = jnp.zeros_like(rhs)
    initial_residual = rhs - matvec(initial_solution)
    initial_residual_inf = jnp.max(jnp.abs(initial_residual), axis=1, initial=0.0)
    _, solution, residual_inf, row_iterations = jax.lax.while_loop(
        cycle_condition,
        cycle_body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            initial_solution,
            initial_residual_inf,
            jnp.zeros((batch_size,), dtype=jnp.int32),
        ),
    )
    finite = jnp.all(jnp.isfinite(solution), axis=1) & jnp.isfinite(residual_inf)
    info = jnp.where(~finite, 2, jnp.where(residual_inf <= threshold, 0, 1)).astype(
        jnp.int32
    )
    if return_iterations:
        return solution, info, residual_inf, row_iterations
    return solution, info, residual_inf
