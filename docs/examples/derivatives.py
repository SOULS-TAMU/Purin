"""Check an affine projection's analytic Jacobian."""

import jax
import jax.numpy as jnp
import numpy as np
from purin import Distance, Projection

jax.config.update("jax_enable_x64", True)
problem = Projection()
p = problem.add_parameter(["b"])
y = problem.add_variable(["u", "v"])
problem.objective = 0.0
problem.constraints.add(y.u + y.v == p.b)
layer = Distance(problem, "linear-equality").prepare(batch_size=1)
theta = jnp.array([[1.0]])


def project_one(candidate):
    return layer(candidate[None, :], theta)[0]


jacobian = jax.jacrev(project_one)(jnp.array([2.0, 0.0]))
np.testing.assert_allclose(jacobian, [[0.5, -0.5], [-0.5, 0.5]], atol=1e-9)
print(jacobian)
