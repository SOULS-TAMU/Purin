"""Implicit distance and correction adapters preserve shared training outputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin.problems.functional import FunctionalProblem
from purin.projections.implicit_distance import SparseProjection
from purin.projections.adapters import (
    WithInfoProjection,
    ImplicitDistance,
    CallableCorrection,
)


@pytest.mark.parametrize("parameterized", [False, True])
def test_implicit_distance_values_gradients_and_adapters(parameterized):
    problem = FunctionalProblem(
        parameter_size=int(parameterized),
        variable_size=2,
        objective=lambda p, y: 0.5 * jnp.sum(y**2),
        equalities=lambda p, y: jnp.array(
            [jnp.sum(y) - (p[0] if parameterized else 1.0)]
        ),
        equality_size=1,
    )
    project = SparseProjection(problem).prepare(batch_size=1)
    context = jnp.ones((1, 1)) if parameterized else None
    target = jnp.array([[0.8, 0.7]])
    expected = jnp.array([[0.55, 0.45]])
    np.testing.assert_allclose(project(target, context), expected, atol=1e-8)
    np.testing.assert_allclose(
        project.with_info(target, context)[0], expected, atol=1e-8
    )
    derivative = jax.jacrev(lambda y: project(y[None], context)[0])(target[0])
    np.testing.assert_allclose(derivative, [[0.5, -0.5], [-0.5, 0.5]], atol=1e-8)
    packed = jnp.ones((1, 1)) if parameterized else jnp.empty((1, 0))
    for adapter in [
        WithInfoProjection(project),
        ImplicitDistance(project._layer),
        ImplicitDistance(project._layer, use_dual_warm_start=True),
    ]:
        output = adapter.project_training(
            target,
            packed,
            multipliers_eq=jnp.zeros((1, 1)),
            multipliers_ineq=jnp.empty((1, 0)),
        )
        np.testing.assert_allclose(output.y_tilde, expected, atol=1e-8)
        assert output.forward_gradient_eligible[0]
    with pytest.raises(ValueError):
        SparseProjection(problem).prepare(batch_size=0)
    with pytest.raises(TypeError):
        SparseProjection(object()).prepare(batch_size=1)


def test_callable_correction_repeats_and_transports_duals():
    step = lambda p, y, eq, iq: (y / 2, eq + 1, iq + 2)
    correction = CallableCorrection(step, layers=3)
    output = correction.project_training(
        jnp.ones((1, 2)) * 8,
        None,
        multipliers_eq=jnp.zeros((1, 1)),
        multipliers_ineq=jnp.zeros((1, 1)),
    )
    np.testing.assert_array_equal(output.y_tilde, [[1.0, 1.0]])
    np.testing.assert_array_equal(output.multipliers_eq, [[3.0]])
    np.testing.assert_array_equal(output.multipliers_ineq, [[6.0]])
    assert output.forward_gradient_eligible[0]
