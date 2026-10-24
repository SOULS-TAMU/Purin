"""Reusable fixed-shape sparse and matrix-free linear operators."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax.experimental import sparse as jsparse
except Exception:  # pragma: no cover
    jsparse = None


def matvec(matrix: Any, vector):
    """Apply a dense, JAX sparse, or callable matrix without densification."""
    if callable(matrix):
        return matrix(vector)
    return matrix @ vector


def rmatvec(matrix: Any, vector):
    """Apply the transpose of a dense or JAX sparse matrix."""
    if hasattr(matrix, "T"):
        return matrix.T @ vector
    raise TypeError("rmatvec requires an object with a transpose operation.")


def diagonal_operator(diagonal) -> Callable:
    """Build the sparse operator representation of a diagonal array."""
    diagonal = jnp.asarray(diagonal)
    return lambda vector: diagonal * vector


def symmetric_block_operator(
    *,
    primal: Callable,
    A: Any,
    C: Any,
    retained_mask,
    multiplier_diagonal,
    equality_regularization,
):
    """Build the fixed-size symmetric reduced-KKT operator.

    Unknowns are ordered as ``(d_y, d_lambda, d_mu_retained)``.  ``A`` and
    ``C`` follow the package orientation ``(n_y, n_constraints)``.
    """
    retained_mask = jnp.asarray(retained_mask)
    multiplier_diagonal = jnp.asarray(multiplier_diagonal)
    equality_regularization = jnp.asarray(equality_regularization)
    n_y = int(A.shape[0]) if A.shape[1] else int(C.shape[0])
    n_eq = int(A.shape[1])
    n_ineq = int(C.shape[1])

    def operator(vector):
        d_y = vector[:n_y]
        d_lambda = vector[n_y : n_y + n_eq]
        d_mu = vector[n_y + n_eq : n_y + n_eq + n_ineq]
        retained_mu = retained_mask * d_mu
        primal_block = primal(d_y)
        if n_eq:
            primal_block = primal_block + matvec(A, d_lambda)
        if n_ineq:
            primal_block = primal_block + matvec(C, retained_mu)
        equality_block = (
            rmatvec(A, d_y) - equality_regularization * d_lambda
            if n_eq
            else jnp.empty((0,), dtype=d_y.dtype)
        )
        inequality_block = (
            retained_mask * rmatvec(C, d_y) + multiplier_diagonal * d_mu
            if n_ineq
            else jnp.empty((0,), dtype=d_y.dtype)
        )
        return jnp.concatenate((primal_block, equality_block, inequality_block))

    return operator


# Local-support sparse operators migrated from the primal-dual backend.


def _safe_support(support: jax.Array) -> jax.Array:
    """Return integer support indices (padding entries are expected to be zero)."""

    return jnp.asarray(support, dtype=jnp.int32)


def _is_jax_array(value) -> bool:
    return isinstance(value, jax.Array)


def local_jacobian_matvec(
    values: jax.Array,
    support: jax.Array,
    mask: jax.Array,
    vector: jax.Array,
    *,
    full_support: bool = False,
) -> jax.Array:
    """Apply a fixed-width row-sparse Jacobian to one vector."""

    if values.shape[0] == 0:
        return jnp.empty((0,), dtype=vector.dtype)
    gathered = vector[_safe_support(support)]
    if full_support:
        return jnp.sum(values * gathered, axis=-1)
    return jnp.sum(jnp.where(mask, values * gathered, 0.0), axis=-1)


def local_jacobian_transpose_matvec(
    values: jax.Array,
    support: jax.Array,
    mask: jax.Array,
    vector: jax.Array,
    n_variables: int,
    *,
    full_support: bool = False,
) -> jax.Array:
    """Apply the transpose of a fixed-width row-sparse Jacobian."""

    if values.shape[0] == 0:
        return jnp.zeros((n_variables,), dtype=vector.dtype)
    contributions = values * vector[:, None]
    if not full_support:
        contributions = jnp.where(mask, contributions, 0.0)
    result = jnp.zeros((n_variables,), dtype=values.dtype)
    return result.at[_safe_support(support)].add(contributions)


def local_jacobian_transpose_gather_matvec(
    values: jax.Array,
    column_rows: jax.Array,
    column_slots: jax.Array,
    column_mask: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Apply ``J.T`` using a precomputed column-oriented gather layout."""

    n_variables = int(column_rows.shape[0])
    if values.shape[0] == 0 or column_rows.shape[1] == 0:
        return jnp.zeros((n_variables,), dtype=vector.dtype)
    gathered_values = values[column_rows, column_slots]
    gathered_vector = vector[column_rows]
    return jnp.sum(
        jnp.where(column_mask, gathered_values * gathered_vector, 0.0), axis=-1
    )


def _build_column_gather_structure(
    support: np.ndarray, mask: np.ndarray, n_variables: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded column adjacency for gather/reduce transpose products."""

    entries: list[list[tuple[int, int]]] = [[] for _ in range(n_variables)]
    for row in range(support.shape[0]):
        for slot in range(support.shape[1]):
            if mask[row, slot]:
                entries[int(support[row, slot])].append((row, slot))
    width = max((len(column_entries) for column_entries in entries), default=0)
    rows = np.zeros((n_variables, width), dtype=np.int32)
    slots = np.zeros((n_variables, width), dtype=np.int32)
    valid = np.zeros((n_variables, width), dtype=bool)
    for variable, column_entries in enumerate(entries):
        for slot_index, (row, local_slot) in enumerate(column_entries):
            rows[variable, slot_index] = row
            slots[variable, slot_index] = local_slot
            valid[variable, slot_index] = True
    return rows, slots, valid


def _build_column_gather_buckets(
    support: np.ndarray, mask: np.ndarray, n_variables: int
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """Build power-of-two degree buckets for transpose gather/reduce.

    A single padded column layout can waste substantial bandwidth when one
    variable has a much larger degree than the rest.  Bucketing variables by
    their next power-of-two degree keeps GPU reads regular while sharply
    reducing padding.  Each tuple contains ``(variables, rows, slots, mask)``.
    """

    entries: list[list[tuple[int, int]]] = [[] for _ in range(n_variables)]
    for row in range(support.shape[0]):
        for slot in range(support.shape[1]):
            if mask[row, slot]:
                entries[int(support[row, slot])].append((row, slot))

    grouped: dict[int, list[int]] = {}
    for variable, column_entries in enumerate(entries):
        degree = len(column_entries)
        if degree == 0:
            continue
        width = 1 << (degree - 1).bit_length()
        grouped.setdefault(width, []).append(variable)

    buckets = []
    for width in sorted(grouped):
        variables = np.asarray(grouped[width], dtype=np.int32)
        rows = np.zeros((variables.size, width), dtype=np.int32)
        slots = np.zeros((variables.size, width), dtype=np.int32)
        valid = np.zeros((variables.size, width), dtype=bool)
        for local_variable, variable in enumerate(variables):
            for position, (row, slot) in enumerate(entries[int(variable)]):
                rows[local_variable, position] = row
                slots[local_variable, position] = slot
                valid[local_variable, position] = True
        buckets.append((variables, rows, slots, valid))
    return tuple(buckets)


def _column_bucket_padding_fraction(
    buckets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...],
) -> float:
    """Return the fraction of padded entries in degree buckets."""

    stored = sum(int(mask.size) for _, _, _, mask in buckets)
    valid = sum(int(np.count_nonzero(mask)) for _, _, _, mask in buckets)
    if stored == 0:
        return 0.0
    return float((stored - valid) / stored)


@dataclass(frozen=True)
class SparseTransposePlan:
    """Static row/column sparse metadata for one Jacobian transpose product.

    The plan is prepared once on the host and captured by JIT-compiled solver
    kernels.  ``mode`` is therefore a static Python choice; no runtime branch is
    inserted into Newton or Krylov iterations.
    """

    mode: str
    support: jax.Array
    mask: jax.Array
    n_variables: int
    full_support: bool
    column_rows: jax.Array | None = None
    column_slots: jax.Array | None = None
    column_mask: jax.Array | None = None
    column_buckets: tuple = ()


def _normalize_local_sparse_layout(support, mask) -> tuple[np.ndarray, np.ndarray]:
    support_np = np.asarray(support, dtype=np.int32)
    mask_np = np.asarray(mask, dtype=bool)
    if support_np.shape != mask_np.shape:
        raise ValueError(
            f"support and mask must have identical shape; got {support_np.shape} and {mask_np.shape}."
        )
    if support_np.ndim == 1 and support_np.size == 0:
        support_np = support_np.reshape((0, 0))
        mask_np = mask_np.reshape((0, 0))
    if support_np.ndim != 2:
        raise ValueError(f"support and mask must be rank two; got {support_np.shape}.")
    return support_np, mask_np


def _combine_local_sparse_layouts(
    equality_support, equality_mask, inequality_support, inequality_mask
) -> tuple[np.ndarray, np.ndarray]:
    eq_support, eq_mask = _normalize_local_sparse_layout(
        equality_support, equality_mask
    )
    ineq_support, ineq_mask = _normalize_local_sparse_layout(
        inequality_support, inequality_mask
    )
    width = max(
        int(eq_support.shape[1]) if eq_support.ndim == 2 else 0,
        int(ineq_support.shape[1]) if ineq_support.ndim == 2 else 0,
    )

    def widen(support_np, mask_np):
        if support_np.shape[1] == width:
            return support_np, mask_np
        padding = width - support_np.shape[1]
        return (
            np.pad(support_np, ((0, 0), (0, padding)), constant_values=0),
            np.pad(mask_np, ((0, 0), (0, padding)), constant_values=False),
        )

    eq_support, eq_mask = widen(eq_support, eq_mask)
    ineq_support, ineq_mask = widen(ineq_support, ineq_mask)
    if eq_support.shape[0] == 0:
        return ineq_support, ineq_mask
    if ineq_support.shape[0] == 0:
        return eq_support, eq_mask
    return (
        np.concatenate((eq_support, ineq_support), axis=0),
        np.concatenate((eq_mask, ineq_mask), axis=0),
    )


def _column_gather_padding_fraction(
    support: np.ndarray, mask: np.ndarray, n_variables: int
) -> float:
    if support.shape[0] == 0 or n_variables <= 0:
        return 0.0
    degrees = np.bincount(support[mask], minlength=n_variables)
    width = int(np.max(degrees, initial=0))
    capacity = int(n_variables) * width
    return 0.0 if capacity == 0 else float((capacity - int(np.sum(mask))) / capacity)


def select_sparse_transpose_mode(
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    n_variables: int,
    *,
    backend: str,
    device_kind: str = "",
    mode: str = "auto",
    padding_threshold: float = 0.25,
) -> str:
    """Select the sparse ``J.T @ v`` implementation using the PDIP policy.

    This is the shared policy used by PDIP and Fischer--Burmeister backends.
    The automatic behavior intentionally preserves the existing PDIP choices:
    CPU and NVIDIA T4 use scatter; newer accelerators use global gather when
    padding is small, then degree-bucketed gather, otherwise scatter.
    """

    requested = str(mode).strip().lower()
    if requested not in {"auto", "scatter", "gather", "bucketed"}:
        raise ValueError(
            "transpose_mode must be one of 'auto', 'scatter', 'gather', or 'bucketed'."
        )
    if requested != "auto":
        return requested

    platform = {"cuda": "gpu", "nvidia": "gpu"}.get(
        str(backend).strip().lower(), str(backend).strip().lower()
    )
    if platform == "cpu":
        return "scatter"

    support_np, mask_np = _combine_local_sparse_layouts(
        equality_support, equality_mask, inequality_support, inequality_mask
    )
    if "t4" in str(device_kind).lower():
        return "scatter"

    gather_padding = _column_gather_padding_fraction(
        support_np, mask_np, int(n_variables)
    )
    buckets = _build_column_gather_buckets(support_np, mask_np, int(n_variables))
    bucket_padding = _column_bucket_padding_fraction(buckets)
    if gather_padding <= float(padding_threshold):
        return "gather"
    if bucket_padding <= float(padding_threshold):
        return "bucketed"
    return "scatter"


def prepare_sparse_transpose_plan(
    support,
    mask,
    n_variables: int,
    *,
    mode: str = "scatter",
    build_all: bool = False,
) -> SparseTransposePlan:
    """Prepare reusable transpose metadata without changing sparse semantics."""

    selected = str(mode).strip().lower()
    if selected not in {"scatter", "gather", "bucketed"}:
        raise ValueError(
            "Prepared transpose mode must be 'scatter', 'gather', or 'bucketed'."
        )
    support_np, mask_np = _normalize_local_sparse_layout(support, mask)
    full_support = bool(np.all(mask_np))

    column_rows = column_slots = column_mask = None
    column_buckets = ()
    if build_all or selected == "gather":
        rows_np, slots_np, valid_np = _build_column_gather_structure(
            support_np, mask_np, int(n_variables)
        )
        column_rows = jnp.asarray(rows_np, dtype=jnp.int32)
        column_slots = jnp.asarray(slots_np, dtype=jnp.int32)
        column_mask = jnp.asarray(valid_np, dtype=bool)
    if build_all or selected == "bucketed":
        buckets_np = _build_column_gather_buckets(support_np, mask_np, int(n_variables))
        column_buckets = tuple(
            (
                jnp.asarray(variables, dtype=jnp.int32),
                jnp.asarray(rows, dtype=jnp.int32),
                jnp.asarray(slots, dtype=jnp.int32),
                jnp.asarray(valid, dtype=bool),
            )
            for variables, rows, slots, valid in buckets_np
        )

    return SparseTransposePlan(
        mode=selected,
        support=jnp.asarray(support_np, dtype=jnp.int32),
        mask=jnp.asarray(mask_np, dtype=bool),
        n_variables=int(n_variables),
        full_support=full_support,
        column_rows=column_rows,
        column_slots=column_slots,
        column_mask=column_mask,
        column_buckets=column_buckets,
    )


def apply_batched_sparse_transpose(plan: SparseTransposePlan, values, vector):
    """Apply one prepared batched sparse transpose plan."""

    if plan.mode == "gather":
        if (
            plan.column_rows is None
            or plan.column_slots is None
            or plan.column_mask is None
        ):
            raise RuntimeError("Gather transpose plan is missing column metadata.")
        return _batched_local_jacobian_transpose_gather_matvec(
            values, plan.column_rows, plan.column_slots, plan.column_mask, vector
        )
    if plan.mode == "bucketed":
        return _batched_local_jacobian_transpose_bucketed_matvec(
            values, plan.column_buckets, vector, plan.n_variables
        )
    return _batched_local_jacobian_transpose_matvec(
        values,
        plan.support,
        plan.mask,
        vector,
        plan.n_variables,
        full_support=plan.full_support,
    )


def local_constraint_hessian_matvec(
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    equality_multipliers: jax.Array,
    inequality_multipliers: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Apply the weighted sum of local constraint Hessians."""

    result = jnp.zeros_like(vector)
    if equality_hessian.shape[0]:
        local_vector = vector[_safe_support(equality_support)]
        local_result = jnp.einsum("mij,mj->mi", equality_hessian, local_vector)
        local_result = jnp.where(
            equality_mask, local_result * equality_multipliers[:, None], 0.0
        )
        result = result.at[_safe_support(equality_support)].add(local_result)
    if inequality_hessian.shape[0]:
        local_vector = vector[_safe_support(inequality_support)]
        local_result = jnp.einsum("mij,mj->mi", inequality_hessian, local_vector)
        local_result = jnp.where(
            inequality_mask, local_result * inequality_multipliers[:, None], 0.0
        )
        result = result.at[_safe_support(inequality_support)].add(local_result)
    return result


def local_constraint_hessian_diagonal(
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    equality_multipliers: jax.Array,
    inequality_multipliers: jax.Array,
    n_variables: int,
) -> jax.Array:
    """Scatter the diagonal of the weighted constraint Hessian."""

    dtype = (
        equality_hessian.dtype
        if equality_hessian.shape[0]
        else inequality_hessian.dtype
    )
    result = jnp.zeros((n_variables,), dtype=dtype)
    if equality_hessian.shape[0]:
        diagonal = jnp.diagonal(equality_hessian, axis1=-2, axis2=-1)
        contributions = jnp.where(
            equality_mask, diagonal * equality_multipliers[:, None], 0.0
        )
        result = result.at[_safe_support(equality_support)].add(contributions)
    if inequality_hessian.shape[0]:
        diagonal = jnp.diagonal(inequality_hessian, axis1=-2, axis2=-1)
        contributions = jnp.where(
            inequality_mask, diagonal * inequality_multipliers[:, None], 0.0
        )
        result = result.at[_safe_support(inequality_support)].add(contributions)
    return result


def _jacobian_row_count(values: jax.Array) -> int:
    return int(values.shape[-2])


def _batched_local_jacobian_matvec(
    values, support, mask, vector, *, full_support=False
):
    """Apply row-sparse Jacobians with batched or shared coefficients."""

    if _jacobian_row_count(values) == 0:
        return jnp.empty((vector.shape[0], 0), dtype=vector.dtype)
    gathered = vector[:, _safe_support(support)]
    coefficients = values if values.ndim == 3 else values[None, :, :]
    if full_support:
        return jnp.sum(coefficients * gathered, axis=-1)
    return jnp.sum(jnp.where(mask[None, :, :], coefficients * gathered, 0.0), axis=-1)


def _batched_local_jacobian_transpose_matvec(
    values, support, mask, vector, n_variables, *, full_support=False
):
    """Scatter transpose product with batched or shared coefficients."""

    if _jacobian_row_count(values) == 0:
        return jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    coefficients = values if values.ndim == 3 else values[None, :, :]
    contributions = coefficients * vector[:, :, None]
    if not full_support:
        contributions = jnp.where(mask[None, :, :], contributions, 0.0)
    result = jnp.zeros((vector.shape[0], n_variables), dtype=values.dtype)
    return result.at[:, _safe_support(support)].add(contributions)


def _batched_local_jacobian_transpose_gather_matvec(
    values, column_rows, column_slots, column_mask, vector
):
    """Global padded gather/reduce transpose product."""

    n_variables = int(column_rows.shape[0])
    if _jacobian_row_count(values) == 0 or column_rows.shape[1] == 0:
        return jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    if values.ndim == 3:
        gathered_values = values[:, column_rows, column_slots]
    else:
        gathered_values = values[column_rows, column_slots][None, :, :]
    gathered_vector = vector[:, column_rows]
    return jnp.sum(
        jnp.where(column_mask[None, :, :], gathered_values * gathered_vector, 0.0),
        axis=-1,
    )


def _batched_local_jacobian_transpose_bucketed_matvec(
    values, column_buckets, vector, n_variables
):
    """Degree-bucketed gather/reduce transpose product for GPUs."""

    if _jacobian_row_count(values) == 0:
        return jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    result = jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    for variable_indices, rows, slots, valid in column_buckets:
        if values.ndim == 3:
            gathered_values = values[:, rows, slots]
        else:
            gathered_values = values[rows, slots][None, :, :]
        gathered_vector = vector[:, rows]
        contributions = jnp.sum(
            jnp.where(valid[None, :, :], gathered_values * gathered_vector, 0.0),
            axis=-1,
        )
        result = result.at[:, variable_indices].set(contributions)
    return result


def _batched_local_constraint_hessian_matvec(
    equality_hessian,
    inequality_hessian,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    equality_multipliers,
    inequality_multipliers,
    vector,
):
    result = jnp.zeros_like(vector)
    if equality_hessian.shape[1]:
        local_vector = vector[:, _safe_support(equality_support)]
        local_result = jnp.einsum("bmij,bmj->bmi", equality_hessian, local_vector)
        local_result = jnp.where(
            equality_mask[None, :, :],
            local_result * equality_multipliers[:, :, None],
            0.0,
        )
        result = result.at[:, _safe_support(equality_support)].add(local_result)
    if inequality_hessian.shape[1]:
        local_vector = vector[:, _safe_support(inequality_support)]
        local_result = jnp.einsum("bmij,bmj->bmi", inequality_hessian, local_vector)
        local_result = jnp.where(
            inequality_mask[None, :, :],
            local_result * inequality_multipliers[:, :, None],
            0.0,
        )
        result = result.at[:, _safe_support(inequality_support)].add(local_result)
    return result


def _batched_local_constraint_hessian_diagonal(
    equality_hessian,
    inequality_hessian,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    equality_multipliers,
    inequality_multipliers,
    n_variables,
):
    result = jnp.zeros(
        (equality_multipliers.shape[0], n_variables), dtype=equality_hessian.dtype
    )
    if equality_hessian.shape[1]:
        diagonal = jnp.diagonal(equality_hessian, axis1=-2, axis2=-1)
        contribution = jnp.where(
            equality_mask[None, :, :], diagonal * equality_multipliers[:, :, None], 0.0
        )
        result = result.at[:, _safe_support(equality_support)].add(contribution)
    if inequality_hessian.shape[1]:
        diagonal = jnp.diagonal(inequality_hessian, axis1=-2, axis2=-1)
        contribution = jnp.where(
            inequality_mask[None, :, :],
            diagonal * inequality_multipliers[:, :, None],
            0.0,
        )
        result = result.at[:, _safe_support(inequality_support)].add(contribution)
    return result


def _batched_schur_diagonal(
    jacobian,
    support,
    mask,
    inverse_primal,
    base,
    *,
    full_support=False,
    jacobian_squared=None,
):
    if _jacobian_row_count(jacobian) == 0:
        return jnp.empty((inverse_primal.shape[0], 0), dtype=inverse_primal.dtype)
    local_inverse = inverse_primal[:, _safe_support(support)]
    if jacobian_squared is None:
        squared = jacobian * jacobian
    else:
        squared = jacobian_squared
    squared = squared if squared.ndim == 3 else squared[None, :, :]
    contributions = squared * local_inverse
    if not full_support:
        contributions = jnp.where(mask[None, :, :], contributions, 0.0)
    correction = jnp.sum(contributions, axis=-1)
    return base + correction
