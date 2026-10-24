"""Original API, multiplier signs and dynamic row-pair validation."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from purin import Projection, prepare_projection, ADMMConfig
from purin import ImplicitGradientConfig, SolveStatus


def test_two_sided_box_original_multipliers_slacks_and_gradient():
    p = Projection()
    y = p.add_variable(["x", "y"])
    p.objective = 0.0
    p.constraints.add(y.x <= 1, -y.x <= 0, y.y <= 2, -y.y <= 1)
    layer = prepare_projection(p, method="admm", batch_size=3)
    assert len(layer._paired_rows.keep) == 2
    targets = jnp.array([[3.0, 4.0], [-2.0, -3.0], [0.5, 0.5]])
    out = jax.jit(layer.solve)(targets)
    expected = jnp.clip(targets, jnp.array([0.0, -1.0]), jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(out.y, expected, atol=2e-9)
    np.testing.assert_allclose(
        out.multipliers_ineq,
        [[2.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
        atol=2e-9,
    )
    np.testing.assert_allclose(
        out.slacks,
        [[0.0, 1.0, 0.0, 3.0], [1.0, 0.0, 3.0, 0.0], [0.5, 0.5, 1.5, 1.5]],
        atol=2e-9,
    )
    np.testing.assert_allclose(
        jax.grad(lambda t: jnp.sum(layer(t)))(targets),
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        atol=1e-8,
    )
    assert np.all(out.status == int(SolveStatus.CONVERGED))


def test_parameter_changes_invalidate_candidate_pairs():
    p = Projection()
    t = p.add_parameter(["a"])
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(y.u + y.v <= 1, -(1 + t.a) * y.u - y.v <= 0)
    cfg = ADMMConfig(
        implicit_gradient=ImplicitGradientConfig(differentiate_problem_parameters=True)
    )
    layer = prepare_projection(p, method="admm", config=cfg, batch_size=1)
    assert layer._paired_rows is not None
    target = jnp.array([[-2.0, 0.0]])
    run = jax.jit(layer.solve)
    for a in [0.0, 1.0, -0.5]:
        theta = jnp.array([[a]])
        normal = jnp.array([[1 + a, 1.0]])
        expected = target - normal * jnp.sum(normal * target) / jnp.sum(normal**2)
        out = run(target, theta)
        np.testing.assert_allclose(out.y, expected, atol=1e-8)
        assert bool(out.gradient_eligible[0])
        eps = 1e-4
        fd = (run(target, theta + eps).y - run(target, theta - eps).y) / (2 * eps)
        derivative = jax.jacrev(lambda th: layer(target, th))(theta)[0, :, 0, 0]
        np.testing.assert_allclose(derivative, fd[0], atol=2e-5)


def test_contradictory_opposite_rows_remain_ineligible():
    p = Projection()
    y = p.add_variable(["u"])
    p.objective = 0.0
    p.constraints.add(y.u <= 0, -y.u <= -1)
    layer = prepare_projection(
        p, method="admm", config=ADMMConfig(max_iterations=100), batch_size=1
    )
    out = layer.solve(jnp.array([[2.0]]))
    assert not bool(out.gradient_eligible[0])
    assert int(out.status[0]) == int(SolveStatus.MAX_ITERATIONS)


def test_zero_width_rows_keep_original_warm_duals_and_gradient():
    p = Projection()
    y = p.add_variable(["u"])
    p.objective = 0.0
    p.constraints.add(y.u <= 0, -y.u <= 0)
    layer = prepare_projection(p, method="admm", batch_size=1)
    warm = jnp.ones((1, 2))
    target = jnp.zeros((1, 1))
    out = layer.project_training(target, multipliers_ineq=warm)
    np.testing.assert_allclose(out.y_tilde, 0.0, atol=1e-12)
    np.testing.assert_allclose(out.multipliers_ineq, warm, atol=1e-12)
    assert bool(out.forward_gradient_eligible[0])
    derivative = jax.grad(
        lambda t: jnp.sum(layer.project_training(t, multipliers_ineq=warm).y_tilde)
    )(target)
    np.testing.assert_allclose(derivative, 0.0, atol=1e-9)
