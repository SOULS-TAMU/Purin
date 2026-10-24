"""Sparse CP projection values, gradients, diagnostics and method integration."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from purin import Projection, prepare_projection, ChambollePockConfig
from purin import (
    ImplicitGradientConfig,
    SolveStatus,
)


def problem():
    p = Projection(method="chambolle-pock")
    t = p.add_parameter(["b"])
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(y.u + y.v == t.b, y.u >= 0)
    return p


@pytest.mark.parametrize("accelerated", [False, True])
def test_affine_projection_and_gradients(accelerated):
    cfg = ChambollePockConfig(
        accelerated=accelerated,
        implicit_gradient=ImplicitGradientConfig(differentiate_problem_parameters=True),
    )
    layer = prepare_projection(
        problem(), method="chambolle-pock", config=cfg, batch_size=2
    )
    values = jnp.array([[2.0, 0.0], [-2.0, 0.0]])
    theta = jnp.ones((2, 1))
    out = jax.jit(layer.solve)(values, theta)
    np.testing.assert_allclose(out.y, [[1.5, -0.5], [0.0, 1.0]], atol=1e-8)
    assert np.all(out.status == int(SolveStatus.CONVERGED))
    assert np.max(out.residual_inf_norm) <= cfg.tolerance
    jac = np.asarray(jax.jacrev(lambda x: layer(x, theta))(values))
    np.testing.assert_allclose(jac[0, :, 0, :], [[0.5, -0.5], [-0.5, 0.5]], atol=1e-8)
    np.testing.assert_allclose(jac[1, :, 1, :], 0, atol=1e-8)
    np.testing.assert_allclose(jac[0, :, 1, :], 0, atol=1e-8)
    derivative = np.asarray(jax.jacrev(lambda t: layer(values, t))(theta))
    np.testing.assert_allclose(derivative[0, :, 0, 0], [0.5, 0.5], atol=1e-8)
    np.testing.assert_allclose(derivative[1, :, 1, 0], [0.0, 1.0], atol=1e-8)
    eps = 1e-4
    np.testing.assert_allclose(
        (
            np.asarray(layer(values, theta + eps))
            - np.asarray(layer(values, theta - eps))
        )
        / (2 * eps),
        [[0.5, 0.5], [0.0, 1.0]],
        atol=1e-5,
    )


def test_rejects_nonlinear_constraints():
    p = Projection()
    y = p.add_variable(["u"])
    p.objective = 0.0
    p.constraints.add(y.u**2 <= 1)
    with pytest.raises(ValueError, match="affine"):
        prepare_projection(p, method="chambolle-pock", batch_size=1)


def test_iteration_limit_and_nonfinite_diagnostics():
    layer = prepare_projection(
        problem(),
        method="chambolle-pock",
        config=ChambollePockConfig(max_iterations=1),
        batch_size=1,
    )
    out = layer.solve(jnp.array([[-2.0, 0.0]]), jnp.ones((1, 1)))
    assert int(out.status[0]) == int(SolveStatus.MAX_ITERATIONS)
    assert not bool(out.gradient_eligible[0])
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.sum(layer(x, jnp.ones((1, 1)))))(
            jnp.array([[-2.0, 0.0]])
        ),
        0,
    )
    out = layer.solve(jnp.array([[jnp.nan, 0.0]]), jnp.ones((1, 1)))
    assert int(out.status[0]) == int(SolveStatus.NONFINITE)


def test_no_constraints_and_zero_initial_gradient():
    p = Projection()
    p.add_variable(["u", "v"])
    p.objective = 0.0
    layer = prepare_projection(p, method="chambolle-pock", batch_size=1)
    x = jnp.array([[0.2, 0.7]])
    np.testing.assert_allclose(layer(x), x, atol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda v: jnp.sum(layer(v)))(x), 1, atol=1e-9)


def test_inequality_only_and_warm_start_gradients():
    p = Projection()
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(y.u >= 0, y.v <= 1)
    layer = prepare_projection(p, method="chambolle-pock", batch_size=1)
    x = jnp.array([[-2.0, 3.0]])
    out = layer.project_training(x, multipliers_ineq=jnp.array([[0.2, 0.3]]))
    np.testing.assert_allclose(out.y_tilde, [[0.0, 1.0]], atol=1e-8)
    assert bool(out.forward_gradient_eligible[0])
    np.testing.assert_allclose(
        jax.grad(
            lambda d: jnp.sum(layer.project_training(x, multipliers_ineq=d).y_tilde)
        )(jnp.array([[0.2, 0.3]])),
        0,
    )


def test_parameter_dependent_sparse_coefficients():
    p = Projection()
    t = p.add_parameter(["a"])
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(t.a * y.u + y.v == 1)
    cfg = ChambollePockConfig(
        implicit_gradient=ImplicitGradientConfig(differentiate_problem_parameters=True)
    )
    layer = prepare_projection(p, method="chambolle-pock", config=cfg, batch_size=1)
    x = jnp.array([[0.2, 0.3]])
    theta = jnp.array([[2.0]])
    expected = lambda t: (
        x
        - jnp.array([[t[0, 0], 1.0]])
        * (jnp.sum(jnp.array([[t[0, 0], 1.0]]) * x) - 1)
        / (t[0, 0] ** 2 + 1)
    )
    np.testing.assert_allclose(layer(x, theta), expected(theta), atol=1e-8)
    np.testing.assert_allclose(
        jax.jacrev(lambda t: layer(x, t))(theta), jax.jacrev(expected)(theta), atol=1e-7
    )


def test_infeasible_problem_never_claims_convergence():
    p = Projection()
    y = p.add_variable(["u"])
    p.objective = 0.0
    p.constraints.add(y.u <= 0, y.u >= 1)
    layer = prepare_projection(
        p,
        method="chambolle-pock",
        config=ChambollePockConfig(max_iterations=100),
        batch_size=1,
    )
    out = layer.solve(jnp.zeros((1, 1)))
    assert int(out.status[0]) == int(SolveStatus.MAX_ITERATIONS)
    assert not bool(out.gradient_eligible[0])


def test_scaled_constraints_report_original_units():
    p = Projection()
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(1000 * y.u + 1000 * y.v == 1000, y.u >= 0)
    layer = prepare_projection(p, method="chambolle-pock", batch_size=1)
    out = layer.solve(jnp.array([[2.0, 0.0]]))
    assert bool(out.gradient_eligible[0])
    assert abs(float(jnp.sum(out.y) * 1000 - 1000)) <= layer.config.tolerance
