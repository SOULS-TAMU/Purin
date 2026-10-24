"""Sparse row/transpose layouts agree with assembled dense Jacobians and Hessians."""

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin.linalg.sparse import operators as op
from purin.problems.sparsity import (
    transpose_gather_padding_fraction,
    transpose_bucket_padding_fraction,
)


@pytest.mark.parametrize("mode", ["scatter", "gather", "bucketed"])
@pytest.mark.parametrize("rows", [((0, 1), (1, 2)), ((0,), (0, 1, 2), (2,)), ()])
def test_sparse_transpose_layouts_match_dense(mode, rows):
    from purin.solvers.chambolle_pock.kernels import layout

    support, mask, _ = layout(rows, 3)
    values = (
        jnp.arange(2 * support.size, dtype=jnp.float64).reshape((2, *support.shape)) + 1
    )
    values = values * mask
    vector = jnp.arange(2 * len(rows), dtype=jnp.float64).reshape((2, len(rows))) + 1
    dense = np.zeros((2, len(rows), 3))
    for row, indices in enumerate(rows):
        for slot, column in enumerate(indices):
            dense[:, row, column] = np.asarray(values[:, row, slot])
    expected = np.einsum("bmn,bm->bn", dense, vector)
    plan = op.prepare_sparse_transpose_plan(support, mask, 3, mode=mode, build_all=True)
    np.testing.assert_allclose(
        jax.jit(lambda v, x: op.apply_batched_sparse_transpose(plan, v, x))(
            values, vector
        ),
        expected,
    )
    v = jnp.arange(3, dtype=jnp.float64)
    np.testing.assert_allclose(
        op.local_jacobian_matvec(values[0], support, mask, v), dense[0] @ v
    )
    np.testing.assert_allclose(
        op.local_jacobian_transpose_matvec(values[0], support, mask, vector[0], 3),
        expected[0],
    )
    np.testing.assert_allclose(
        op.local_jacobian_transpose_gather_matvec(
            values[0], plan.column_rows, plan.column_slots, plan.column_mask, vector[0]
        ),
        expected[0],
    )
    for fraction in [
        transpose_gather_padding_fraction,
        transpose_bucket_padding_fraction,
    ]:
        padding = fraction(support, mask, support[:0], mask[:0], 3)
        assert 0 <= padding <= 1
        assert fraction(support[:0], mask[:0], support[:0], mask[:0], 0) == 0
    for backend, kind in [
        ("cpu", ""),
        ("gpu", "NVIDIA T4"),
        ("gpu", "NVIDIA A100"),
        ("cuda", ""),
    ]:
        selected = op.select_sparse_transpose_mode(
            support, mask, support[:0], mask[:0], 3, backend=backend, device_kind=kind
        )
        assert selected in {"scatter", "gather", "bucketed"}
    if mode == "gather":
        with pytest.raises(RuntimeError):
            op.apply_batched_sparse_transpose(
                replace(plan, column_rows=None), values, vector
            )


@pytest.mark.parametrize("both", [False, True])
def test_local_hessian_products_match_assembled_matrix(both):
    support = jnp.array([[0, 2]])
    mask = jnp.array([[True, True]])
    eq = jnp.array([[[2.0, 1.0], [1.0, 3.0]]])
    iq = 2 * eq if both else jnp.empty((0, 2, 2))
    iq_support = support if both else support[:0]
    iq_mask = mask if both else mask[:0]
    lam = jnp.array([2.0])
    mu = jnp.array([3.0]) if both else jnp.empty(0)
    dense = np.zeros((3, 3))
    dense[np.ix_([0, 2], [0, 2])] = np.asarray(eq[0]) * (8 if both else 2)
    vector = jnp.array([1.0, 2.0, 3.0])
    args = (eq, iq, support, mask, iq_support, iq_mask, lam, mu)
    np.testing.assert_allclose(
        op.local_constraint_hessian_matvec(*args, vector), dense @ vector
    )
    np.testing.assert_allclose(
        op.local_constraint_hessian_diagonal(*args, 3), np.diag(dense)
    )
    batched = (
        eq[None],
        iq[None],
        support,
        mask,
        iq_support,
        iq_mask,
        lam[None],
        mu[None],
    )
    np.testing.assert_allclose(
        op._batched_local_constraint_hessian_matvec(*batched, vector[None]),
        (dense @ vector)[None],
    )
    np.testing.assert_allclose(
        op._batched_local_constraint_hessian_diagonal(*batched, 3), np.diag(dense)[None]
    )


def test_invalid_transpose_layouts():
    for mode in ["bad", "auto"]:
        with pytest.raises(ValueError):
            op.prepare_sparse_transpose_plan([[0]], [[True]], 1, mode=mode)
    with pytest.raises(ValueError):
        op.select_sparse_transpose_mode(
            [[0]], [[True]], [], [], 1, backend="cpu", mode="bad"
        )
    for support, mask in [([0], [True]), ([[0]], [[True, False]])]:
        with pytest.raises(ValueError):
            op.prepare_sparse_transpose_plan(support, mask, 1)
