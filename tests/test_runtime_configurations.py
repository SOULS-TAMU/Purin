"""Equality projection preserves supported dtype, JIT and batch contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from purin import Distance, Projection


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_projection_runtime_configuration(dtype, compiled, batch_size):
    problem = Projection()
    p = problem.add_parameter(["b"])
    y = problem.add_variable(["u", "v"])
    problem.objective = 0.0
    problem.constraints.add(y.u + y.v == p.b)
    project = Distance(problem, "linear-equality").prepare(
        batch_size=batch_size, dtype=dtype
    )
    target = jnp.tile(jnp.array([[2.0, 0.0]], dtype=dtype), (batch_size, 1))
    context = jnp.ones((batch_size, 1), dtype=dtype)
    if compiled:
        actual = jax.jit(project)(target, context)
    else:
        with jax.disable_jit():
            actual = project(target, context)
    assert actual.dtype == dtype
    np.testing.assert_allclose(
        actual, np.tile([1.5, -0.5], (batch_size, 1)), atol=1e-6, rtol=0
    )
