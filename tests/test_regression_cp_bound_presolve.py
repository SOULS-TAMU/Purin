"""Bound presolve equivalence, original dual recovery, and parameter gradients."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from purin import Projection, prepare_projection, ChambollePockConfig
from purin import ImplicitGradientConfig, SolveStatus


def test_tied_scaled_bounds_restore_duals_and_slacks():
    p = Projection()
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(y.u >= 0, 2 * y.u >= 0, y.u <= 2, y.v <= 1)
    layer = prepare_projection(p, method="chambolle-pock", batch_size=1)
    out = jax.jit(layer.solve)(jnp.array([[-3.0, 4.0]]))
    np.testing.assert_allclose(out.y, [[0.0, 1.0]], atol=1e-10)
    np.testing.assert_allclose(out.multipliers_ineq, [[1.5, 0.75, 0.0, 3.0]], atol=1e-9)
    np.testing.assert_allclose(out.slacks, [[0.0, 0.0, 2.0, 0.0]], atol=1e-10)
    assert int(out.status[0]) == int(SolveStatus.CONVERGED)
    assert len(layer._bound_projection.rows) == 0
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.sum(layer(x)))(jnp.array([[-3.0, 4.0]])), 0.0, atol=1e-9
    )


def test_parameter_dependent_bound_sign_zero_and_derivative():
    p = Projection()
    t = p.add_parameter(["a", "b"])
    y = p.add_variable(["u"])
    p.objective = 0.0
    p.constraints.add(t.a * y.u <= t.b)
    cfg = ChambollePockConfig(
        implicit_gradient=ImplicitGradientConfig(differentiate_problem_parameters=True)
    )
    layer = prepare_projection(p, method="chambolle-pock", config=cfg, batch_size=1)
    x = jnp.array([[3.0]])
    theta = jnp.array([[2.0, 1.0]])
    np.testing.assert_allclose(layer(x, theta), [[0.5]], atol=1e-10)
    jac = jax.jacrev(lambda z: layer(x, z))(theta)
    np.testing.assert_allclose(jac[0, 0, 0], [-0.25, 0.5], atol=1e-8)
    np.testing.assert_allclose(
        layer(jnp.array([[-3.0]]), jnp.array([[-2.0, 1.0]])), [[-0.5]], atol=1e-10
    )
    out = layer.solve(x, jnp.array([[0.0, -1.0]]))
    assert not bool(out.gradient_eligible[0])
    np.testing.assert_allclose(
        jax.grad(lambda z: jnp.sum(layer(z, jnp.array([[0.0, -1.0]]))))(x), 0.0
    )
    np.testing.assert_allclose(layer(x, jnp.array([[0.0, 1.0]])), x)


@pytest.mark.parametrize("seed", [12, 42])
def test_coupled_qp_matches_osqp_and_target_finite_difference(seed):
    import osqp
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    p = Projection()
    y = p.add_variable(["u", "v", "w"])
    p.objective = 0.0
    p.constraints.add(
        y.u + y.v + y.w == 1, y.u >= 0, y.v >= -0.5, y.w <= 0.7, 2 * y.u - y.v <= 0.9
    )
    layer = prepare_projection(p, method="chambolle-pock", batch_size=1)
    x = rng.normal(size=(1, 3))
    matrix = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, -1.0, 0.0],
        ]
    )
    upper = np.array([1.0, 0.0, 0.5, 0.7, 0.9])
    qp = osqp.OSQP()
    qp.setup(
        P=sp.eye(3, format="csc"),
        q=-x[0],
        A=sp.csc_matrix(matrix),
        l=np.r_[1.0, np.full(4, -np.inf)],
        u=upper,
        eps_abs=1e-11,
        eps_rel=1e-11,
        polishing=True,
        verbose=False,
    )
    ref = qp.solve()
    assert ref.info.status_val == 1
    out = jax.jit(layer.solve)(jnp.asarray(x))
    np.testing.assert_allclose(out.y[0], ref.x, atol=2e-8)
    np.testing.assert_allclose(
        (out.y[0] - x[0])
        + matrix.T @ np.r_[out.multipliers_eq[0], out.multipliers_ineq[0]],
        0.0,
        atol=1e-9,
    )
    assert bool(out.gradient_eligible[0])
    g = rng.normal(size=(1, 3))
    analytic = np.asarray(jax.grad(lambda z: jnp.sum(layer(z) * g))(jnp.asarray(x)))
    for i in range(3):
        d = np.zeros_like(x)
        d[0, i] = 1e-4
        fd = float(jnp.sum((layer(x + d) - layer(x - d)) * g) / (2e-4))
        np.testing.assert_allclose(analytic[0, i], fd, atol=2e-5)


def test_admm_keeps_original_rows():
    p = Projection()
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(y.u >= 0, y.v <= 1)
    layer = prepare_projection(p, method="admm", batch_size=1)
    assert getattr(layer, "_bound_projection", None) is None
    assert layer._support.shape[0] == 2
