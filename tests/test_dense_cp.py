"""Dense CP adapters retain primal/dual values and implicit coefficient derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin.solvers.chambolle_pock.dense_layer import solve_dense_implicit
from purin.solvers.chambolle_pock.legacy import (
    CP_fixed,
    CP_accelerated,
    canonicalize_constraints,
    estimate_AC_norm,
    _init_x,
    _primal_residual,
    _primal_dual_gap,
    _prepare_scaled_problem,
)


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("equilibrate", [False, True])
def test_dense_cp_matches_analytic_equality_qp(accelerated, equilibrate):
    q = jnp.array([[2.0, 4.0]])
    c = jnp.array([[-2.0, -4.0]])
    A = jnp.ones((1, 1, 2))
    b = jnp.ones((1, 1))
    C = jnp.array([[[-1.0, 0.0]]])
    d = jnp.zeros((1, 1))
    lo = jnp.zeros((1, 2))
    hi = jnp.ones((1, 2)) * 2
    args = (q, c, A, b, C, d, lo, hi)
    kwargs = dict(
        accelerated=accelerated,
        equilibrate=equilibrate,
        tolerance=1e-13,
        max_iterations=20000,
    )
    actual = solve_dense_implicit(*args, **kwargs)
    np.testing.assert_allclose(actual[0], [[1 / 3, 2 / 3]], atol=1e-6)
    np.testing.assert_allclose(
        solve_dense_implicit(
            *args, y0=actual[0], lambda0=actual[1], mu0=actual[2], **kwargs
        )[0],
        actual[0],
        atol=1e-6,
    )
    derivative = jax.jacrev(
        lambda linear: solve_dense_implicit(q, linear, A, b, C, d, lo, hi, **kwargs)[0]
    )(c)
    expected = -np.diag([0.5, 0.25]) + np.outer([0.5, 0.25], [0.5, 0.25]) / 0.75
    np.testing.assert_allclose(derivative[0, :, 0, :], expected, atol=1e-6)
    legacy = CP_accelerated if accelerated else CP_fixed
    answer = legacy(*args, use_ruiz=equilibrate, max_iter=20000, tol=1e-13)
    np.testing.assert_allclose(answer[0], actual[0], atol=1e-6)
    assert jnp.all(_primal_residual(actual[0], A, b, C, d, lo, hi) < 1e-7)
    assert jnp.all(jnp.abs(_primal_dual_gap(*args, *actual)) < 1e-7)
    assert jnp.all(estimate_AC_norm(A, C, jax.random.PRNGKey(0)) > 0)
    assert len(_prepare_scaled_problem(*args, use_ruiz=equilibrate, ruiz_iters=2)) == 11


def test_dense_empty_constraints():
    q = jnp.ones((1, 2))
    c = jnp.array([[-1.0, -2.0]])
    A, b, C, d = canonicalize_constraints(q, None, None, None, None)
    assert A.shape == C.shape == (1, 0, 2)
    actual = solve_dense_implicit(
        q,
        c,
        A,
        b,
        C,
        d,
        jnp.zeros((1, 2)),
        jnp.ones((1, 2)) * 3,
        equilibrate=True,
        tolerance=1e-14,
    )
    np.testing.assert_allclose(actual[0], [[1.0, 2.0]], atol=1e-6)
    norms = jnp.linalg.norm(_init_x(jax.random.PRNGKey(0), 2, 3, jnp.float64), axis=1)
    np.testing.assert_allclose(norms, 1, atol=1e-10)
