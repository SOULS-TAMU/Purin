"""Cached diagonal-Q proximal updates preserve values and derivatives."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from purin.solvers.chambolle_pock.kernels import cp_step, proximal_coefficients


def test_diagonal_proximal_values_and_derivatives():
    rng = np.random.default_rng(42)
    q = jnp.asarray(rng.uniform(0.1, 5.0, (3, 7)))
    c = jnp.asarray(rng.normal(size=(3, 7)))
    tau = jnp.asarray([0.1, 0.4, 0.7])
    y = jnp.asarray(rng.normal(size=(3, 7)))
    a, d, offset = proximal_coefficients(q, c, tau)
    actual = a * y - d * (2 * y) + offset
    expected = (y - tau[:, None] * (2 * y + c)) / (1 + tau[:, None] * q)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    cached = lambda x: sum(proximal_coefficients(q, x, tau)[2].ravel())
    original = lambda x: jnp.sum(-tau[:, None] * x / (1 + tau[:, None] * q))
    np.testing.assert_allclose(jax.grad(cached)(c), jax.grad(original)(c), atol=1e-14)


def test_step_cached_and_uncached_match_with_zero_diagonal():
    q = jnp.array([[0.0, 0.3, 2.0]])
    c = jnp.array([[1.0, -1.0, 0.5]])
    y = jnp.array([[0.3, 0.2, -0.1]])
    dual = jnp.array([[0.2, 0.3, 0.4]])
    tau = jnp.array([0.3])
    sigma = jnp.array([0.2])
    theta = jnp.ones(1)
    args = (
        (y, dual, y),
        q,
        c,
        jnp.zeros_like(y),
        jnp.full_like(y, -2.0),
        jnp.full_like(y, 2.0),
        lambda x: x,
        lambda x: x,
        1,
        tau,
        sigma,
        theta,
    )
    uncached = cp_step(*args)
    cached = cp_step(*args, proximal=proximal_coefficients(q, c, tau))
    for a, b in zip(cached, uncached):
        np.testing.assert_array_equal(a, b)
