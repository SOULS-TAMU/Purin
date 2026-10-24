"""Common Fischer--Burmeister value and generalized derivative policies."""

import jax.numpy as jnp


def radius(a, b, smoothing, *, smoothing_weight=1.0):
    """Compute the smoothed complementarity radius for paired primal and dual values."""
    return jnp.sqrt(a * a + b * b + smoothing_weight * smoothing * smoothing)


def value(a, b, smoothing, *, smoothing_weight=1.0):
    """Evaluate the smoothed Fischer--Burmeister complementarity function."""
    return radius(a, b, smoothing, smoothing_weight=smoothing_weight) - a - b


def partials(a, b, smoothing, *, smoothing_weight=1.0, origin=-1.0):
    """Evaluate the complementarity function's partial derivatives."""
    r = radius(a, b, smoothing, smoothing_weight=smoothing_weight)
    regular = r > 0.0
    return (
        jnp.where(regular, a / r - 1.0, jnp.asarray(origin, r.dtype)),
        jnp.where(regular, b / r - 1.0, jnp.asarray(origin, r.dtype)),
    )
