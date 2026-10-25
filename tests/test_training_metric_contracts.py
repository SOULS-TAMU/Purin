"""Training projection pullbacks and weighted metrics retain analytic solutions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin import Distance, WeightedDistance, Projection
from purin.solvers.primal_dual_interior_point.solver import (
    run_batched_fixed_mu_iterations,
)
from purin.solvers.primal_dual_interior_point.types import (
    BoundIndices,
    BoundValues,
    PrimalDualState,
)
from test_projections import make_problem, config


@pytest.mark.parametrize(
    "method", ["fischer-burmeister-newton", "fischer-burmeister-reduced"]
)
def test_training_output_pullback_and_dual_warm_start(method):
    prepared = Distance(
        make_problem(inequalities=True), method, config(method)
    ).prepare(batch_size=1)
    target = jnp.array([[0.8, 0.7]])
    context = jnp.ones((1, 1))
    eq, iq = jnp.zeros((1, 1)), jnp.ones((1, 2))

    def training(y, equality, inequality):
        return prepared.project_training(
            y, context, multipliers_eq=equality, multipliers_ineq=inequality
        ).y_tilde

    projected = training(target, eq, iq)
    np.testing.assert_allclose(projected, [[0.55, 0.45]], atol=2e-6)
    grad_y, grad_eq, grad_iq = jax.grad(
        lambda y, a, b: jnp.sum(training(y, a, b) * jnp.array([[2.0, -1.0]])),
        argnums=(0, 1, 2),
    )(target, eq, iq)
    np.testing.assert_allclose(grad_y, [[1.5, -1.5]], atol=2e-5)
    np.testing.assert_allclose(grad_eq, 0, atol=1e-12)
    np.testing.assert_allclose(grad_iq, 0, atol=1e-12)
    for params, y in [
        (None, target),
        (jnp.ones((2, 1)), target),
        (context, jnp.ones((2, 2))),
    ]:
        with pytest.raises(ValueError):
            prepared.solve(y, params)
    with pytest.raises(ValueError, match="initial_state"):
        prepared.solve(target, context, initial_state=jnp.zeros((1, 1)))
    for equality, inequality in [(None, iq), (eq, None)]:
        np.testing.assert_allclose(
            prepared.project_training(
                target, context, multipliers_eq=equality, multipliers_ineq=inequality
            ).y_tilde,
            projected,
            atol=2e-6,
        )


def test_weighted_metric_protocol_and_analytic_gradient():
    from purin.solvers.metric import _scaled_problem

    prepared = WeightedDistance(
        make_problem(), [1.0, 4.0], method="chambolle-pock"
    ).prepare(batch_size=1)
    target, context = jnp.array([[2.0, 0.0]]), jnp.ones((1, 1))
    expected = [[1.2, -0.2]]
    with pytest.raises(ValueError, match="warm starts"):
        prepared.solve(target, context, initial_state=jnp.array([[1.2, -0.2]]))
    result = prepared.solve(target, context)
    np.testing.assert_allclose(result.y, expected, atol=2e-5)
    np.testing.assert_allclose(
        prepared.project_training(target, context).y_tilde, expected, atol=2e-5
    )
    np.testing.assert_allclose(
        prepared.with_info(target, context)[0], expected, atol=2e-5
    )
    gradient = jax.jacrev(lambda x: prepared(x[None], context)[0])(target[0])
    np.testing.assert_allclose(gradient, [[0.2, -0.8], [-0.2, 0.8]], atol=2e-5)
    with pytest.raises(TypeError, match="symbolic"):
        _scaled_problem(object(), np.ones(2))
    with pytest.raises(ValueError, match="entries"):
        _scaled_problem(make_problem(), np.ones(1))
    # Target coefficients cannot simultaneously be physical constraint parameters.
    problem = Projection()
    p = problem.add_parameter(["target"])
    y = problem.add_variable(["y"])
    problem.set_projection_objective(variables=y, target=[p.target])
    problem.constraints.add(y.y >= p.target)
    with pytest.raises(ValueError, match="projection-target"):
        _scaled_problem(problem, np.ones(1))


@pytest.mark.parametrize(
    "options",
    [
        {"max_iterations": -1},
        {"max_iterations": 0.5},
        {"residual_tolerance": 0},
        {"tau": 1},
        {"step_scale": 0},
        {"minimum_step": -1},
        {"parameter_values": [[np.nan]]},
        {"parameter_values": [1.0]},
        {"mu": -1},
        {"mu": [1.0, 2.0]},
    ],
)
def test_batched_ipm_rejects_invalid_solver_inputs(options):
    empty = jnp.empty((1, 0))
    state = PrimalDualState(jnp.ones((1, 1)), *([empty] * 7))
    indices = BoundIndices(*([jnp.empty(0, dtype=jnp.int32)] * 4))
    bounds = BoundValues(*([jnp.empty(0)] * 4))
    kwargs = dict(
        initial_state=state,
        parameter_values=[[1.0]],
        evaluate_nlp=None,
        indices=indices,
        bounds=bounds,
        mu=0.1,
    )
    kwargs.update(options)
    with pytest.raises(ValueError):
        run_batched_fixed_mu_iterations(**kwargs)


def test_batched_ipm_rejects_mismatched_state_batch():
    empty = jnp.empty((1, 0))
    state = PrimalDualState(jnp.ones((2, 1)), *([empty] * 7))
    with pytest.raises(ValueError, match="Initial-state"):
        run_batched_fixed_mu_iterations(state, [[1.0]], None, None, None, 0.1)
