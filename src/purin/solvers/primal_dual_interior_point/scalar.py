"""Compile one binding for the optional NumPy reference solver."""

import jax
import jax.numpy as jnp
import numpy as np
from .derivatives import CompiledProblem


def compile_scalar(
    problem, *, parameters=None, initial_values=None, interior_margin=1.0
):
    """Prepare derivative and solver resources for a scalar problem instance."""
    from .reference.bounds import BoundIndices, BoundValues
    from .reference.state import PrimalDualState
    from .reference.residuals import NLPQuantities

    values = problem._bind_parameters(parameters)
    initial = problem._bind_initial_values(initial_values)
    margin = float(interior_margin)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError(f"interior_margin must be finite and positive; got {margin}.")
    objective, equalities, inequalities = problem._build_functions()
    structure = problem._build_projection_structure()
    if structure is not None and initial_values is None:
        initial = values[
            np.asarray(structure.target_parameter_indices, dtype=np.int64)
        ].copy()
    equality_jacobian = jax.jacobian(equalities, argnums=0)
    inequality_jacobian = jax.jacobian(inequalities, argnums=0)
    if structure is None:
        gradient = jax.grad(objective, argnums=0)

        def lagrangian(y, p, equality, inequality):
            return (
                objective(y, p)
                + jnp.vdot(equality, equalities(y, p))
                + jnp.vdot(inequality, inequalities(y, p))
            )

        hessian = jax.hessian(lagrangian, argnums=0)
    else:
        target_indices = jnp.asarray(
            structure.target_parameter_indices, dtype=jnp.int32
        )

        def gradient(y, p):
            return y - p[target_indices]

        def constraint_lagrangian(y, p, equality, inequality):
            return jnp.vdot(equality, equalities(y, p)) + jnp.vdot(
                inequality, inequalities(y, p)
            )

        constraint_hessian = jax.hessian(constraint_lagrangian, argnums=0)

        def hessian(y, p, equality, inequality):
            return jnp.eye(y.shape[0], dtype=y.dtype) + constraint_hessian(
                y, p, equality, inequality
            )

    bound_parameters = jnp.asarray(values, dtype=jnp.float64)

    def evaluate(state):
        y, equality, inequality = (
            jnp.asarray(v, dtype=jnp.float64) for v in (state.x, state.y_c, state.y_d)
        )
        return NLPQuantities(
            grad_f=np.asarray(gradient(y, bound_parameters), dtype=np.float64),
            c=np.asarray(equalities(y, bound_parameters), dtype=np.float64),
            d=np.asarray(inequalities(y, bound_parameters), dtype=np.float64),
            jac_c=np.asarray(
                equality_jacobian(y, bound_parameters), dtype=np.float64
            ).reshape((state.y_c.size, state.x.size)),
            jac_d=np.asarray(
                inequality_jacobian(y, bound_parameters), dtype=np.float64
            ).reshape((state.y_d.size, state.x.size)),
            hessian_lagrangian=np.asarray(
                hessian(y, bound_parameters, equality, inequality), dtype=np.float64
            ),
            objective_value=float(np.asarray(objective(y, bound_parameters))),
        )

    initial_jax = jnp.asarray(initial, dtype=jnp.float64)
    eq = np.asarray(equalities(initial_jax, bound_parameters), dtype=np.float64)
    iq = np.asarray(inequalities(initial_jax, bound_parameters), dtype=np.float64)
    empty, empty_index = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    indices = BoundIndices(
        empty_index.copy(),
        empty_index.copy(),
        empty_index.copy(),
        np.arange(iq.size, dtype=np.int64),
    )
    bounds = BoundValues(
        empty.copy(), empty.copy(), empty.copy(), np.zeros(iq.size, dtype=np.float64)
    )
    state = PrimalDualState(
        initial,
        np.minimum(iq, -margin),
        np.zeros(eq.size, dtype=np.float64),
        np.ones(iq.size, dtype=np.float64),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        np.ones(iq.size, dtype=np.float64),
    )
    return CompiledProblem(
        problem.name,
        tuple(problem.variable_names),
        tuple(problem.parameter_names),
        values,
        state,
        indices,
        bounds,
        evaluate,
        objective,
        equalities,
        inequalities,
    )


def solve_scalar(
    problem,
    *,
    parameters=None,
    initial_values=None,
    mu=1e-4,
    options=None,
    interior_margin=1.0,
    use_filter_line_search=True,
    raise_on_failure=False,
):
    """Solve one problem instance through the prepared scalar solver pathway."""
    from .reference.iteration import FixedMuOptions, run_fixed_mu_iterations
    from .reference.line_search import FilterLineSearchOptions
    from .types import ProblemSolution
    from purin.problems.types import NamedValues

    compiled = compile_scalar(
        problem,
        parameters=parameters,
        initial_values=initial_values,
        interior_margin=interior_margin,
    )
    if options is None:
        options = FixedMuOptions(
            line_search=FilterLineSearchOptions() if use_filter_line_search else None
        )
    result = run_fixed_mu_iterations(
        initial_state=compiled.initial_state,
        evaluate_nlp=compiled.evaluate_nlp,
        indices=compiled.indices,
        bounds=compiled.bounds,
        mu=mu,
        options=options,
    )
    if raise_on_failure and not result.converged:
        raise RuntimeError(
            f"Problem {problem.name!r} did not converge: status={result.status}, residual={result.final_residual_inf_norm:.3e}."
        )
    variables = result.state.x
    return ProblemSolution(
        problem.name,
        NamedValues(problem.variable_names, variables),
        NamedValues(problem.parameter_names, compiled.parameter_values),
        compiled.objective(variables),
        compiled.equalities(variables),
        compiled.inequalities(variables),
        result,
    )
