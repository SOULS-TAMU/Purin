import jax
import jax.numpy as jnp
from purin import Distance, Projection

jax.config.update("jax_enable_x64", True)

problem = Projection()
p = problem.add_parameter(["b"])
y = problem.add_variable(["u", "v"])
problem.objective = 0.0
problem.constraints.add(y.u + y.v == p.b)

project = Distance(problem, "linear-equality").prepare(batch_size=1)
candidate = jnp.array([[2.0, 0.0]])  # Shape: (batch, variables).
parameters = jnp.array([[1.0]])  # Shape: (batch, parameters).
projected = project(candidate, parameters)
print(projected)  # [[ 1.5 -0.5]]

# Check the declared physical/output contract and model transport.

import numpy as np

np.testing.assert_allclose(projected, [[1.5, -0.5]], atol=1e-9)
