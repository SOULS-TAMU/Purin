"""Diagonal QPs and local approximations checked against analytic constrained minima."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin import ChambollePockConfig
from purin.solvers.diagonal_qp import QPStructure, dense_qp_data
from purin.problems.functional import FunctionalProblem
from purin.projections.local_quadratic import LocalQuadratic


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("scaling", [False, True])
def test_diagonal_qp_value_and_coefficient_gradient(accelerated, scaling):
    structure, data = dense_qp_data(
        jnp.array([[2.0, 4.0]]),
        jnp.array([[-2.0, -4.0]]),
        jnp.ones((1, 2)),
        jnp.array([1.0]),
        jnp.empty((0, 2)),
        jnp.empty(0),
        -jnp.inf,
        jnp.inf,
    )
    config = ChambollePockConfig(
        accelerated=accelerated,
        row_scaling=scaling,
        tolerance=1e-9,
        max_iterations=3000,
    )
    solver = structure.prepare(batch_size=1, config=config)
    result = solver.solve(data)
    np.testing.assert_allclose(result.y, [[1 / 3, 2 / 3]], atol=1e-8)
    assert result.residual_inf_norm[0] < 1e-8
    # d y / d c = -Q^-1 + Q^-1 a (a.T Q^-1 a)^-1 a.T Q^-1.
    expected = -np.diag([0.5, 0.25]) + np.outer([0.5, 0.25], [0.5, 0.25]) / 0.75
    actual = jax.jacrev(lambda c: solver.solve(data._replace(c=c[None, :])).y[0])(
        data.c[0]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    # Warm starts are numerical aids and must not change the solution.
    warm = solver.solve(data, initial_y=result.y, initial_dual=result.multipliers_eq)
    np.testing.assert_allclose(warm.y, result.y, atol=1e-8)
    with pytest.raises(ValueError):
        solver.solve(data._replace(q=jnp.ones((2, 2))))
    with pytest.raises(ValueError):
        solver.solve(data, initial_y=jnp.zeros((2, 2)))
    with pytest.raises(ValueError):
        solver.solve(data, initial_dual=jnp.zeros((2, 1)))


def test_local_quadratic_coefficients_and_training_protocol():
    problem = FunctionalProblem(
        parameter_size=1,
        variable_size=2,
        objective=lambda p, y: 0.5 * jnp.sum(jnp.array([2.0, 4.0]) * (y - 1) ** 2),
        equalities=lambda p, y: jnp.array([jnp.sum(y) - p[0]]),
        equality_size=1,
        inequalities=lambda p, y: jnp.array([-y[0]]),
        inequality_size=1,
    )
    definition = LocalQuadratic(
        problem,
        lower=lambda p: jnp.zeros(2),
        upper=2.0,
        config=ChambollePockConfig(tolerance=1e-9, max_iterations=3000),
    )
    compiled = definition.compile()
    context = jnp.ones((1, 1))
    anchor = jnp.array([[0.4, 0.5]])
    data = compiled.build(context, anchor)
    np.testing.assert_allclose(data.q, [[2.0, 4.0]])
    np.testing.assert_allclose(data.c, [[-2.0, -4.0]])
    np.testing.assert_allclose(data.rhs, [[1.0, 0.0]])
    prepared = definition.prepare(batch_size=1)
    result = prepared.solve(anchor, context)
    np.testing.assert_allclose(result.y, [[1 / 3, 2 / 3]], atol=1e-8)
    np.testing.assert_allclose(prepared(anchor, context), result.y, atol=1e-8)
    np.testing.assert_allclose(
        prepared.with_info(anchor, context)[0], result.y, atol=1e-8
    )
    for kwargs in [
        {},
        {
            "multipliers_eq": result.multipliers_eq,
            "multipliers_ineq": result.multipliers_ineq,
        },
    ]:
        np.testing.assert_allclose(
            prepared.project_training(anchor, context, **kwargs).y_tilde,
            result.y,
            atol=1e-8,
        )
    for badcontext, badanchor in [
        (context[0], anchor),
        (context, anchor[0]),
        (jnp.ones((2, 1)), anchor),
        (jnp.ones((1, 2)), anchor),
        (context, jnp.ones((1, 3))),
    ]:
        with pytest.raises(ValueError):
            compiled.build(badcontext, badanchor)
    for kwargs in [dict(rho=0), dict(diagonal_floor=-1), dict(rho=np.inf)]:
        with pytest.raises(ValueError):
            LocalQuadratic(problem, **kwargs)


def test_invalid_qp_structures():
    for args in [(0,), (2, ((0, 0),)), (2, ((2,),)), (2, ((-1,),))]:
        with pytest.raises(ValueError):
            QPStructure(*args)
    structure = QPStructure(2)
    with pytest.raises(ValueError):
        structure.prepare(batch_size=0)
    with pytest.raises(ValueError):
        structure.prepare(
            batch_size=1, config=ChambollePockConfig(metric_diagonal=(1.0, 1.0))
        )
    with pytest.raises(ValueError):
        dense_qp_data(
            jnp.ones(2),
            jnp.ones(2),
            jnp.empty((0, 2)),
            [],
            jnp.empty((0, 2)),
            [],
            -1,
            1,
        )
