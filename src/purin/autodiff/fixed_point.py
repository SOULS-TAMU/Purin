"""Matrix-free implicit differentiation of a stationary iteration."""

import jax
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import bicgstab, gmres


def fixed_point_pullback(
    step,
    state,
    parameters,
    cotangent,
    *,
    method="bicgstab",
    tolerance=1e-6,
    max_iterations=50,
    restart=50,
):
    """Solve (I - dF/dstate).T v = cotangent and return v.T dF/dparameters."""
    _, pullback = jax.vjp(step, state, parameters)
    rhs, unravel = ravel_pytree(cotangent)

    def operator(flat):
        tangent, _ = pullback(unravel(flat))
        return flat - ravel_pytree(tangent)[0]

    if method == "bicgstab":
        solution, _ = bicgstab(
            operator, rhs, tol=tolerance, atol=0.0, maxiter=max_iterations
        )
    elif method == "gmres":
        solution, _ = gmres(
            operator,
            rhs,
            tol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
            restart=restart,
        )
    else:
        raise ValueError(f"Unknown adjoint method {method!r}")
    return pullback(unravel(solution))[1]
