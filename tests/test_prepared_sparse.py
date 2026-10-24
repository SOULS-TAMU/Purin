"""Prepared sparse solves reuse linearizations without changing analytic gradients."""

import jax.numpy as jnp
import numpy as np
import pytest
from purin.solvers.primal_dual_interior_point.reference import FixedMuOptions
from test_symbolic_workflow import distance_problem


@pytest.mark.parametrize("fused", [False, True])
def test_prepared_sparse_forward_adjoint_and_loss(fused):
    problem = distance_problem()
    params = jnp.array([[1.0, 0.8, 0.7]])
    solver = problem.prepare_sparse_projection_batch_solver(
        parameters=params,
        batch_size=1,
        options=FixedMuOptions(max_iterations=15, residual_tolerance=1e-9),
        fuse_initialization=fused,
        transpose_mode="gather" if fused else "bucketed",
        adjoint_solver_mode="dense" if fused else "gmres",
    )
    assert solver.backend == "cpu"
    np.testing.assert_array_equal(
        solver.pack_projection_parameters_jax(
            jnp.ones((1, 1)), jnp.array([[0.8, 0.7]])
        ),
        params,
    )
    expected = jnp.array([[0.55, 0.45]])
    initial = jnp.array([[0.0, 0.0]])
    for values in [None, initial]:
        result = solver.solve(params, mu=1e-8, initial_values=values)
        np.testing.assert_allclose(result.state.x, expected, atol=1e-8)
        forward = solver.solve_with_linearization(
            params, mu=1e-8, initial_values=values
        )
        assert forward.converged[0]
        for linearization in [None, forward.final_linearization]:
            adjoint = solver.solve_yhat_adjoint(
                forward.state,
                params,
                [[2.0, -1.0]],
                mu=1e-8,
                final_linearization=linearization,
            )
            np.testing.assert_allclose(adjoint.x, [[1.5, -1.5]], atol=1e-8)
            warm = solver.solve_yhat_adjoint(
                forward.state,
                params,
                [[2.0, -1.0]],
                mu=1e-8,
                final_linearization=linearization,
                initial_adjoint=adjoint,
            )
            np.testing.assert_allclose(warm.x, adjoint.x, atol=1e-8)
        for mode, q in [
            ("identity", jnp.ones(2)),
            ("diagonal", jnp.array([2.0, 3.0])),
            ("dense", jnp.diag(jnp.array([2.0, 3.0]))),
        ]:
            output = solver.solve_loss_yhat_gradient(
                params,
                q,
                q_mode=mode,
                initial_values=values,
                mu=1e-8,
                valid_mask=[True],
            )
            grad = 2 * expected[0] * (q if mode != "dense" else jnp.diag(q))
            expected_grad = grad - jnp.mean(grad)
            np.testing.assert_allclose(output.y_tilde, expected, atol=1e-8)
            np.testing.assert_allclose(
                output.grad_y_hat, expected_grad[None], atol=1e-8
            )
    for bad in [np.ones(3), np.ones((2, 3)), np.ones((1, 4))]:
        with pytest.raises(ValueError):
            solver.put_parameters(bad)
    with pytest.raises(ValueError):
        solver.put_mu([1.0, 2.0])
    with pytest.raises(ValueError):
        solver.solve(params, initial_values=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        solver.solve_yhat_adjoint(forward.state, params, [[1.0]], mu=0.0)
    with pytest.raises(ValueError):
        solver.solve_yhat_adjoint(
            forward.state, params, [[1.0, 1.0]], initial_adjoint=np.zeros((1, 2))
        )
    for kwargs in [
        dict(q_mode="bad"),
        dict(valid_mask=[True, False]),
        dict(initial_adjoint=np.zeros((1, 2))),
    ]:
        with pytest.raises(ValueError):
            solver.solve_loss_yhat_gradient(params, jnp.ones(2), **kwargs)
    with pytest.raises(ValueError):
        solver.solve_loss_yhat_gradient(params, jnp.ones(3))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(transpose_mode="bad"),
        dict(gpu_kkt_mode="bad"),
        dict(gmres_unroll=0),
        dict(arnoldi_prefix_block=-1),
        dict(pallas_kkt_block_size=0),
        dict(adjoint_solver_mode="bad"),
        dict(adjoint_gmres_tolerance=0),
        dict(adjoint_gmres_restart=0),
        dict(adjoint_gmres_maxiter=0),
        dict(adjoint_gmres_unroll=0),
        dict(adjoint_arnoldi_prefix_block=-1),
        dict(gmres_tolerance=0),
        dict(gmres_restart=0),
        dict(gmres_maxiter=0),
        dict(max_backtracking_steps=0),
        dict(backtracking_factor=1.0),
    ],
)
def test_invalid_sparse_solver_configuration(kwargs):
    problem = distance_problem()
    with pytest.raises(ValueError):
        problem.prepare_sparse_projection_batch_solver(
            parameters=np.array([[1.0, 0.8, 0.7]]), batch_size=1, **kwargs
        )
