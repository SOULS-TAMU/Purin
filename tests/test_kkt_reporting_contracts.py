"""Matrix-free KKT products and named result diagnostics agree with dense data."""

from dataclasses import replace
import jax.numpy as jnp
import numpy as np
import pytest
from purin.linalg.sparse.operators import (
    matvec,
    rmatvec,
    diagonal_operator,
    symmetric_block_operator,
)
from test_symbolic_workflow import distance_problem


@pytest.mark.parametrize("equalities,inequalities", [(0, 0), (1, 0), (0, 2), (1, 2)])
def test_symmetric_kkt_operator_matches_explicit_block_matrix(equalities, inequalities):
    h = np.array([[3.0, 0.5], [0.5, 2.0]])
    a = np.arange(2 * equalities, dtype=float).reshape(2, equalities) + 1
    c = np.arange(2 * inequalities, dtype=float).reshape(2, inequalities) + 1
    retained = np.arange(inequalities) % 2 == 0
    multiplier = np.full(inequalities, -0.2)
    regularization = 0.1
    dense = np.block(
        [
            [h, a, c * retained],
            [
                a.T,
                -regularization * np.eye(equalities),
                np.zeros((equalities, inequalities)),
            ],
            [
                (c * retained).T,
                np.zeros((inequalities, equalities)),
                np.diag(multiplier),
            ],
        ]
    )
    operator = symmetric_block_operator(
        primal=lambda y: jnp.asarray(h) @ y,
        A=jnp.asarray(a),
        C=jnp.asarray(c),
        retained_mask=retained,
        multiplier_diagonal=multiplier,
        equality_regularization=regularization,
    )
    vector = jnp.arange(2 + equalities + inequalities, dtype=jnp.float64) + 1
    np.testing.assert_allclose(operator(vector), dense @ vector, atol=1e-12)
    np.testing.assert_allclose(matvec(lambda x: 2 * x, vector), 2 * vector)
    np.testing.assert_allclose(
        diagonal_operator(jnp.ones(vector.size) * 3)(vector), 3 * vector
    )
    np.testing.assert_allclose(rmatvec(jnp.asarray(h), vector[:2]), h.T @ vector[:2])
    with pytest.raises(TypeError, match="transpose"):
        rmatvec(object(), vector)


def test_named_single_and_batch_diagnostics_match_solver_results():
    problem = distance_problem()
    values = {"b": 1.0, "target_u": 0.8, "target_v": 0.7}
    one = problem.solve(parameters=values, mu=1e-8)
    assert one.converged and one.status == str(one.raw_result.status)
    assert one.iterations == int(one.raw_result.iterations)
    assert one.residual_inf_norm < 1e-7
    assert one.state is one.raw_result.state
    assert one["u"] == pytest.approx(0.55, abs=1e-7)
    batch = problem.solve_batch(
        parameters=values, batch_size=2, mu=1e-8, projection_mode="sparse_iterative"
    )
    assert batch.batch_size == 2 and batch.all_converged
    assert batch.status == ("converged", "converged")
    assert batch.state is batch.raw_result.state
    np.testing.assert_allclose(batch["u"], [0.55, 0.55], atol=1e-7)
    np.testing.assert_array_equal(batch.objective, batch.objectives)
    host = replace(
        batch,
        status_codes_host=np.array([2, 4]),
        converged_host=np.array([False, False]),
        iterations_host=np.array([5, 6]),
        residual_inf_norm_host=np.array([0.1, 0.2]),
    )
    assert host.status == ("maximum_iterations", "numerical_failure")
    assert not host.all_converged
    np.testing.assert_array_equal(host.iterations, [5, 6])
    np.testing.assert_array_equal(host.residual_inf_norm, [0.1, 0.2])
    # Clear host caches to verify the JAX-backed fallback views as well.
    uncached = replace(
        batch,
        status_codes_host=None,
        converged_host=None,
        iterations_host=None,
        residual_inf_norm_host=None,
    )
    assert uncached.status == batch.status
    np.testing.assert_array_equal(uncached.iterations, batch.iterations)
    np.testing.assert_array_equal(uncached.residual_inf_norm, batch.residual_inf_norm)
    np.testing.assert_array_equal(uncached.converged, batch.converged)


def test_sparse_adjoint_recompute_cache_and_low_level_validation():
    problem = distance_problem()
    parameters = jnp.array([[1.0, 0.8, 0.7]])
    prepared = problem.prepare_sparse_projection_batch_solver(
        parameters=parameters, batch_size=1
    )
    forward = prepared.solve_with_linearization(parameters, mu=1e-8)
    solver = prepared.sparse_solver
    gradient = jnp.array([[2.0, -1.0]])
    for cached in [False, True]:
        output = solver.solve_yhat_adjoint(
            final_state=forward.state,
            grad_x=gradient,
            final_linearization=forward.final_linearization if cached else None,
            parameter_values=None if cached else parameters,
            mu=1e-8,
            initial_solution=jnp.zeros((1, 3)),
        )
        np.testing.assert_allclose(output.x, [[1.5, -1.5]], atol=1e-8)
    for values, mu in [
        (jnp.ones(3), 0.1),
        (jnp.ones((2, 3)), 0.1),
        (parameters, -1),
        (parameters, [np.nan]),
        (parameters, [0.1, 0.2]),
    ]:
        with pytest.raises(ValueError):
            solver.solve(initial_state=forward.state, parameter_values=values, mu=mu)
    for changes in [
        {"grad_x": jnp.ones((1, 1))},
        {"initial_solution": jnp.ones((1, 1))},
        {"parameter_values": None},
        {"parameter_values": jnp.ones(3)},
        {"parameter_values": jnp.ones((2, 3))},
    ]:
        kwargs = dict(
            final_state=forward.state,
            grad_x=gradient,
            parameter_values=parameters,
            mu=1e-8,
        )
        kwargs.update(changes)
        with pytest.raises(ValueError):
            solver.solve_yhat_adjoint(**kwargs)
