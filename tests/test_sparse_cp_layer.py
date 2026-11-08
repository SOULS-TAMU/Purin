"""Dense/sparse CP equivalence for primal iterates and implicit derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from purin.solvers.chambolle_pock.dense_layer import solve_dense_implicit


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("equilibrate", [False, True])
@pytest.mark.parametrize("concentrated", [False, True])
def test_sparse_forward_and_structural_gradients(
    accelerated, equilibrate, concentrated
):
    rng = np.random.default_rng(7)
    n, me, mi, batch = 8, 2, 3, 2
    rows = ((0, 2), (), (1, 4, 7))
    if concentrated:
        n = 32
        rows = ((0, 1),) * mi
    mask = np.zeros((mi, n))
    for i, row in enumerate(rows):
        mask[i, list(row)] = 1
    q = jnp.asarray(rng.uniform(1, 2, (batch, n)))
    c = jnp.asarray(rng.normal(size=(batch, n)))
    a = jnp.asarray(rng.normal(size=(batch, me, n)))
    b = jnp.zeros((batch, me))
    matrix = jnp.asarray(rng.normal(size=(batch, mi, n)))
    d = jnp.asarray(np.broadcast_to([0.05, 1, 0.05], (batch, mi)))
    lower, upper = -jnp.ones_like(q), jnp.ones_like(q)
    parameters = (q, c, a, b, matrix, d, lower, upper)

    def solve(params, sparse):
        q, c, a, b, matrix, d, lower, upper = params
        return solve_dense_implicit(
            q,
            c,
            a,
            b,
            matrix * mask,
            d,
            lower,
            upper,
            accelerated=accelerated,
            equilibrate=equilibrate,
            max_iterations=500,
            norm_iterations=25,
            # Compare converged adjoints across platform reduction orders.
            adjoint_iterations=100,
            inequality_support=rows if sparse else None,
        )

    dense, sparse = solve(parameters, False), solve(parameters, True)
    for x, y in zip(dense, sparse):
        np.testing.assert_allclose(x, y, atol=2e-7, rtol=2e-6)
    weights = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in dense)

    def loss(params, sparse):
        return sum(jnp.sum(w * y) for w, y in zip(weights, solve(params, sparse)))

    dense_grad = jax.grad(loss)(parameters, False)
    sparse_grad = jax.grad(loss)(parameters, True)
    for x, y in zip(dense_grad, sparse_grad):
        # The relative residual tolerance is 1e-6, so near-zero gradient
        # components can differ across reduction orders by a few micro-units.
        np.testing.assert_allclose(x, y, atol=5e-6, rtol=2e-5)
        assert np.linalg.norm(x - y) <= 1e-5 * max(1.0, np.linalg.norm(x))


@pytest.mark.parametrize(
    "rows", [((0,),), ((0,), (8,)), ((0,), (1, 1)), ((0,), (0.5,))]
)
def test_invalid_support_rejected(rows):
    from purin.solvers.chambolle_pock.sparse_layer import packed_implicit_solver

    with pytest.raises(ValueError):
        packed_implicit_solver(rows, 8, 2)


def test_zero_constraint_blocks_and_diagonal_proximal():
    q = jnp.asarray([[1.0, 2.0, 3.0]])
    c = jnp.asarray([[0.2, -0.6, 0.9]])
    empty = jnp.zeros((1, 0, 3))
    rhs = jnp.zeros((1, 0))
    y, eq, iq = solve_dense_implicit(
        q,
        c,
        empty,
        rhs,
        empty,
        rhs,
        -jnp.ones_like(q),
        jnp.ones_like(q),
        inequality_support=(),
        max_iterations=500,
    )
    dense = solve_dense_implicit(
        q,
        c,
        empty,
        rhs,
        empty,
        rhs,
        -jnp.ones_like(q),
        jnp.ones_like(q),
        max_iterations=500,
    )
    np.testing.assert_allclose(y, dense[0], atol=1e-12)
    from purin.solvers.chambolle_pock.kernels import proximal_coefficients

    tau = jnp.asarray([0.2])
    inverse, weight, offset = proximal_coefficients(q, c, tau)
    assert inverse.shape == weight.shape == offset.shape == q.shape
    vector = jnp.asarray([[0.3, 0.4, -0.2]])
    actual = inverse * vector + offset
    expected = np.linalg.solve(
        np.diag(1 + 0.2 * np.asarray(q[0])), np.asarray(vector[0] - 0.2 * c[0])
    )
    np.testing.assert_allclose(actual[0], expected, atol=1e-14)
    assert eq.shape == iq.shape == (1, 0)
