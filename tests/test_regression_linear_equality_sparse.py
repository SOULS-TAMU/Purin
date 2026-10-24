"""Sparse linear projection feasibility, dense-reference and gradient checks."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from purin import Projection, prepare_projection
from purin.solvers.linear_equality.solver import LinearEqualityProjection
from purin.solvers.types import SolveStatus


def reference(matrix, rhs, values):
    return (
        values
        - jnp.linalg.solve(matrix @ matrix.T, (values @ matrix.T - rhs).T).T @ matrix
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_explicit_matches_dense_values_and_gradients(sparse):
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(8, 24))
    matrix[np.abs(matrix) < 0.8] = 0
    values = jnp.asarray(rng.normal(size=(4, 24)))
    theta = jnp.asarray(rng.normal(size=(4, 2)))
    mapping = jnp.asarray(rng.normal(size=(8, 2)))
    rhs = 1 + theta @ mapping.T
    layer = LinearEqualityProjection(
        csr_matrix(matrix) if sparse else matrix, b0=np.ones(8), b_parameter_map=mapping
    ).prepare(batch_size=4)
    result = jax.jit(layer.solve)(values, theta)
    np.testing.assert_allclose(
        result.y, reference(matrix, rhs, values), atol=1e-10, rtol=1e-10
    )
    assert np.max(result.residual_inf_norm) <= 1e-10
    assert np.all(result.status == int(SolveStatus.CONVERGED))
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.sum(layer(x, theta) ** 2))(values),
        jax.grad(lambda x: jnp.sum(reference(matrix, rhs, x) ** 2))(values),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        jax.grad(lambda t: jnp.sum(layer(values, t) ** 2))(theta),
        jax.grad(lambda t: jnp.sum(reference(matrix, 1 + t @ mapping.T, values) ** 2))(
            theta
        ),
        atol=1e-9,
        rtol=1e-9,
    )


def test_syngas_symbolic_projection():
    p = Projection(name="syngas", method="linear-equality")
    t = p.add_parameter(["a", "P", "T"])
    y = p.add_variable(["CO2", "H2", "CO", "CH4", "H2O", "Util"])
    p.constraints.add(
        y.CO2 + y.CO + y.CH4 == 1,
        2 * y.CO2 + y.CO + y.H2O == 2,
        4 * y.CH4 + 2 * y.H2 + 2 * y.H2O == 2 * t.a,
    )
    p.objective = 0.0
    layer = prepare_projection(p, method="linear-equality", batch_size=50)
    rng = np.random.default_rng(42)
    values = jnp.asarray(rng.normal(size=(50, 6)))
    theta = jnp.asarray(np.column_stack([rng.uniform(0, 3, 50), np.ones((50, 2))]))
    matrix = jnp.asarray(
        [[1, 0, 1, 1, 0, 0], [2, 0, 1, 0, 1, 0], [0, 2, 0, 4, 2, 0]], dtype=jnp.float64
    )
    rhs = jnp.column_stack([jnp.ones(50), 2 * jnp.ones(50), 2 * theta[:, 0]])
    result = jax.jit(layer.solve)(values, theta)
    np.testing.assert_allclose(
        result.y, reference(matrix, rhs, values), atol=1e-10, rtol=1e-10
    )
    assert np.max(result.residual_inf_norm) <= 1e-10
    np.testing.assert_allclose(
        jax.grad(lambda t: jnp.sum(layer(values, t) ** 2))(theta),
        jax.grad(
            lambda t: jnp.sum(
                reference(
                    matrix,
                    jnp.column_stack([jnp.ones(50), 2 * jnp.ones(50), 2 * t[:, 0]]),
                    values,
                )
                ** 2
            )
        )(theta),
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize("shape", [(0, 4), (1, 1), (40, 120)])
def test_sparse_shapes(shape):
    m, n = shape
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=shape)
    matrix[np.abs(matrix) < 1] = 0
    matrix[:, :m] += 3 * np.eye(m)
    layer = LinearEqualityProjection(csr_matrix(matrix), b=np.ones(m)).prepare(
        batch_size=2
    )
    out = jax.jit(layer.solve)(rng.normal(size=(2, n)))
    assert np.max(out.residual_inf_norm) <= 1e-10
    assert np.all(out.gradient_eligible)


def test_rank_deficiency_rejected():
    with pytest.raises(ValueError, match="full row rank"):
        LinearEqualityProjection(A=np.ones((2, 3))).prepare(batch_size=1)


def test_regularization_keeps_values_but_reports_infeasibility():
    layer = LinearEqualityProjection(
        A=np.ones((1, 2)), b=np.ones(1), regularization=0.1
    ).prepare(batch_size=1)
    out = layer.solve(jnp.zeros((1, 2)))
    np.testing.assert_allclose(out.y, [[1 / 2.1, 1 / 2.1]], atol=1e-12)
    assert int(out.status[0]) == int(SolveStatus.LINEAR_SOLVE_FAILURE)
    assert not bool(out.gradient_eligible[0])


def test_parameter_dependent_matrix_and_float32():
    p = Projection(method="linear-equality")
    t = p.add_parameter(["a"])
    y = p.add_variable(["u", "v"])
    p.constraints.add(t.a * y.u + y.v == 1)
    p.objective = 0.0
    layer = prepare_projection(p, method="linear-equality", batch_size=2)
    theta = jnp.array([[2.0], [3.0]])
    x = jnp.array([[0.2, 0.3], [0.5, -0.2]])
    expected = lambda t: jax.vmap(
        lambda a, z: (
            z - jnp.array([a[0], 1.0]) * ((a[0] * z[0] + z[1] - 1) / (a[0] ** 2 + 1))
        )
    )(t, x)
    np.testing.assert_allclose(layer(x, theta), expected(theta), atol=1e-10)
    np.testing.assert_allclose(
        jax.jacrev(lambda t: layer(x, t))(theta), jax.jacrev(expected)(theta), atol=1e-9
    )
    f32 = LinearEqualityProjection(A=np.ones((1, 2)), b=np.ones(1)).prepare(
        batch_size=1, dtype=jnp.float32
    )
    out = f32.solve(jnp.array([[0.2, 0.3]], dtype=jnp.float32))
    assert out.y.dtype == jnp.float32
    assert np.max(out.residual_inf_norm) < 1e-6
