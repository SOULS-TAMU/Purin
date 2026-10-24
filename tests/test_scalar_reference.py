"""Scalar reference compilation agrees with batch production solving."""

import numpy as np
import pytest
from purin import Projection
from purin.solvers.primal_dual_interior_point.scalar import compile_scalar, solve_scalar
from purin.solvers.primal_dual_interior_point.reference import (
    FixedMuOptions,
    compute_bound_slacks,
)
from purin.solvers.primal_dual_interior_point.reference.residuals import (
    compute_kkt_residual,
)
from purin.solvers.primal_dual_interior_point import conversion
from test_symbolic_workflow import distance_problem


@pytest.mark.parametrize("distance", [False, True])
def test_scalar_compilation_and_solution(distance):
    if distance:
        problem = distance_problem()
        parameters = [1.0, 0.8, 0.7]
        expected = [0.55, 0.45]
    else:
        problem = Projection()
        y = problem.add_variable(["u", "v"])
        problem.objective = (y.u - 1) ** 2 + (y.v - 2) ** 2
        problem.constraints.add(y.u + y.v == 3, y.u >= 0)
        parameters = None
        expected = [1.0, 2.0]
    compiled = compile_scalar(problem, parameters=parameters)
    nlp = compiled.evaluate_nlp(compiled.initial_state)
    jstate = conversion.state_from_numpy(compiled.initial_state)
    np.testing.assert_array_equal(
        conversion.state_to_numpy(jstate).flatten(), compiled.initial_state.flatten()
    )
    ji = conversion.indices_from_numpy(compiled.indices)
    jb = conversion.bounds_from_numpy(compiled.bounds)
    np.testing.assert_array_equal(ji.s_upper, compiled.indices.s_upper)
    np.testing.assert_array_equal(jb.s_upper, compiled.bounds.s_upper)
    jnlp = conversion.nlp_from_numpy(nlp)
    np.testing.assert_allclose(jnlp.grad_f, nlp.grad_f)
    slacks = compute_bound_slacks(
        compiled.initial_state, compiled.indices, compiled.bounds
    )
    np.testing.assert_allclose(
        conversion.slacks_from_numpy(slacks).s_upper, slacks.s_upper
    )
    residual = compute_kkt_residual(
        compiled.initial_state, nlp, compiled.indices, slacks, 1e-8
    )
    np.testing.assert_allclose(
        conversion.residual_from_numpy(residual).flatten(), residual.flatten()
    )
    result = solve_scalar(
        problem,
        parameters=parameters,
        mu=1e-8,
        raise_on_failure=True,
        use_filter_line_search=False,
    )
    np.testing.assert_allclose(result.variables.vector(), expected, atol=1e-7)
    assert compiled.objective(result.variables.vector()) >= 0
    compiled.equalities(result.variables.vector())
    compiled.inequalities(result.variables.vector())
    if not distance:
        with pytest.raises(RuntimeError):
            solve_scalar(
                problem,
                mu=1e-8,
                raise_on_failure=True,
                options=FixedMuOptions(max_iterations=0),
            )
    with pytest.raises(ValueError):
        compile_scalar(problem, parameters=parameters, interior_margin=0)
