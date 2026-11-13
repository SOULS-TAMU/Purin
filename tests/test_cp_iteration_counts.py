"""Check opt-in counts for dense and packed Chambolle--Pock solves."""

import jax.numpy as jnp
import numpy as np
import pytest

from purin.solvers.chambolle_pock.dense_layer import solve_dense_implicit


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("support", [None, ((0,),)])
@pytest.mark.parametrize("equilibrate", [False, True])
def test_iteration_counts_preserve_solution(accelerated, support, equilibrate):
    arguments = (
        jnp.ones((2, 1)),
        jnp.array([[0.0], [-2.0]]),
        jnp.empty((0, 1)),
        jnp.empty((0,)),
        jnp.ones((1, 1)),
        jnp.ones((1,)),
        jnp.array([-jnp.inf]),
        jnp.array([jnp.inf]),
    )
    options = dict(
        accelerated=accelerated,
        inequality_support=support,
        equilibrate=equilibrate,
        max_iterations=8,
        tolerance=1e-12,
    )
    expected = solve_dense_implicit(*arguments, **options)
    *actual, iterations = solve_dense_implicit(
        *arguments, **options, return_iterations=True
    )
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)
    # The dense adapter advances all rows until the shared batch loop exits.
    np.testing.assert_array_equal(iterations, [8, 8])
    singleton = tuple(
        value[:1] if i < 2 else value for i, value in enumerate(arguments)
    )
    *_, early = solve_dense_implicit(*singleton, **options, return_iterations=True)
    np.testing.assert_array_equal(early, [1])
