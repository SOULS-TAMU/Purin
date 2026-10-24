"""Linear algebra kernels checked against independent dense products and solves."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin.linalg.direct import dense_linear_solve, device_summary
from purin.linalg.operators import DenseOperator, DiagonalOperator, SparseOperator
from purin.linalg.sparse.minres import minres
from purin.linalg.sparse.assembly import (
    pattern_indices,
    FixedSparseMatrixFactory,
    masked_diagonal,
)
from purin.linalg.equilibration import ruiz_equilibrate, max_abs_rows, max_abs_columns
from purin.execution.runtime import (
    resolve_device,
    resolve_jax_dtype,
    aot_compile,
    select_device,
)
from purin._utils.validation import (
    positive_integer,
    positive_float,
    nonnegative_float,
    open_unit_interval,
    validate_metric_diagonal,
)
from purin.solvers.capabilities import capabilities
from purin.differentiation.policies import GradientPolicy


@pytest.mark.parametrize("symmetric", [False, True])
def test_direct_solve_and_transpose_derivative(symmetric):
    matrix = (
        jnp.array([[3.0, 1.0], [1.0, 2.0]])
        if symmetric
        else jnp.array([[3.0, 2.0], [1.0, 2.0]])
    )
    rhs = jnp.array([1.0, 3.0])
    np.testing.assert_allclose(
        dense_linear_solve(matrix, rhs, symmetric=symmetric),
        np.linalg.solve(matrix, rhs),
        atol=1e-12,
    )
    actual = jax.jacrev(lambda b: dense_linear_solve(matrix, b, symmetric=symmetric))(
        rhs
    )
    np.testing.assert_allclose(actual, np.linalg.inv(matrix), atol=1e-12)


@pytest.mark.parametrize("preconditioned", [False, True])
def test_minres_indefinite_operator(preconditioned):
    matrix = jnp.array([[2.0, 1.0], [1.0, -2.0]])
    rhs = jnp.array([1.0, 2.0])
    result = minres(
        lambda v: matrix @ v,
        rhs,
        preconditioner=(lambda v: v / 2) if preconditioned else None,
        tolerance=1e-10,
    )
    np.testing.assert_allclose(
        result.solution, np.linalg.solve(matrix, rhs), atol=1e-10
    )
    assert result.converged
    exact = minres(lambda v: matrix @ v, rhs, initial=result.solution)
    assert exact.converged


def test_operators_and_sparse_factory():
    matrix = jnp.array([[2.0, 0.0, 3.0], [0.0, 4.0, 0.0]])
    vector = jnp.array([1.0, 2.0, 3.0])
    cot = jnp.array([4.0, 5.0])
    dense = DenseOperator(matrix)
    np.testing.assert_allclose(dense.mv(vector), matrix @ vector)
    np.testing.assert_allclose(dense.tmv(cot), matrix.T @ cot)
    diagonal = DiagonalOperator(jnp.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(diagonal.mv(vector), [2.0, 6.0, 12.0])
    np.testing.assert_array_equal(diagonal.tmv(vector), diagonal.mv(vector))
    sparse = SparseOperator(
        jnp.array([[[2.0, 3.0], [4.0, 0.0]]]),
        jnp.array([[0, 2], [1, 0]]),
        jnp.array([[True, True], [True, False]]),
        3,
    )
    np.testing.assert_allclose(sparse.mv(vector[None, :]), (matrix @ vector)[None, :])
    np.testing.assert_allclose(sparse.tmv(cot[None, :]), (matrix.T @ cot)[None, :])
    indices = pattern_indices(((0, 2), (1,)), row_count=2, column_count=3)
    factory = FixedSparseMatrixFactory(indices, (2, 3))
    values = factory.gather_values(matrix)
    np.testing.assert_array_equal(factory.build_from_values(values).todense(), matrix)
    np.testing.assert_array_equal(
        pattern_indices(None, row_count=1, column_count=2, transpose=True),
        [[0, 0], [1, 0]],
    )
    empty = FixedSparseMatrixFactory(
        pattern_indices((), row_count=0, column_count=2), (0, 2)
    )
    assert empty.gather_values(jnp.empty((0, 2))).size == 0
    np.testing.assert_array_equal(
        masked_diagonal([1.0, 2.0], [True, False]), [1.0, -1.0]
    )


def test_equilibration_preserves_qp_coordinates():
    q = jnp.array([[2.0, 8.0]])
    c = jnp.array([[3.0, 4.0]])
    A = jnp.array([[[2.0, 3.0]]])
    C = jnp.array([[[-4.0, 1.0]]])
    b = jnp.array([[1.0]])
    d = jnp.array([[2.0]])
    lo = jnp.array([[-2.0, -3.0]])
    hi = -lo
    qs, cs, As, bs, Cs, ds, ls, us, column, eq, iq = ruiz_equilibrate(
        q, c, A, b, C, d, lo, hi
    )
    np.testing.assert_allclose(qs, q * column**2)
    np.testing.assert_allclose(cs, c * column)
    np.testing.assert_allclose(As, eq[:, :, None] * A * column[:, None, :])
    np.testing.assert_allclose(bs, b * eq)
    np.testing.assert_allclose(Cs, iq[:, :, None] * C * column[:, None, :])
    np.testing.assert_allclose(ds, d * iq)
    np.testing.assert_allclose(ls, lo / column)
    np.testing.assert_allclose(us, hi / column)
    assert max_abs_rows(jnp.empty((1, 0, 2))).shape == (1, 0)
    np.testing.assert_array_equal(max_abs_columns(jnp.empty((1, 0, 2))), [[0.0, 0.0]])


def test_runtime_resolution_and_validation():
    cpu = jax.devices("cpu")[0]
    assert resolve_device(None) is None
    assert resolve_device(cpu) is cpu
    assert resolve_device("cpu:0") == cpu and select_device("cpu") == cpu
    assert device_summary()
    np.testing.assert_array_equal(
        aot_compile(lambda x: x + 1, jnp.ones(2))(jnp.ones(2)), [2.0, 2.0]
    )
    for value in ["bad", "cpu:bad", "cpu:99999", "gpu:99999"]:
        with pytest.raises(ValueError):
            resolve_device(value)
    for value in ["float32", "fp32", "32", "float64", "fp64", "64"]:
        assert resolve_jax_dtype(value) in (jnp.float32, jnp.float64)
    with pytest.raises(ValueError):
        resolve_jax_dtype("int32")
    for fn, good, bad in [
        (positive_integer, 2, 0),
        (positive_float, 1.0, 0.0),
        (nonnegative_float, 0.0, -1.0),
        (open_unit_interval, 0.5, 1.0),
    ]:
        assert fn(good, "value") == good
        with pytest.raises(ValueError):
            fn(bad, "value")
    assert validate_metric_diagonal(None) is None
    assert validate_metric_diagonal([1.0, 2.0]) == (1.0, 2.0)
    for bad in [[], [0.0], [np.inf], [[1.0]]]:
        with pytest.raises(ValueError):
            validate_metric_diagonal(bad)
    assert capabilities("chambolle-pock").diagonal_qp
    assert capabilities("linear-equality").forward_mode
    assert capabilities("primal-dual-interior-point").nonlinear_constraints
    assert GradientPolicy().mode == "residual_checked"
    for kwargs in [dict(mode="bad"), dict(adjoint_tolerance=0)]:
        with pytest.raises(ValueError):
            GradientPolicy(**kwargs)
