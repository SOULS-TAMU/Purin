"""Dense reference IPM checked against full KKT systems and analytic optima."""

import numpy as np
import pytest
from purin.solvers.primal_dual_interior_point.reference import (
    BoundIndices,
    BoundValues,
    PrimalDualState,
    compute_bound_slacks,
    FixedMuOptions,
    run_fixed_mu_iterations,
)
from purin.solvers.primal_dual_interior_point.reference.residuals import (
    NLPQuantities,
    compute_kkt_residual,
)
from purin.solvers.primal_dual_interior_point.reference.newton import (
    solve_primal_dual_newton_system,
    build_full_kkt_matrix,
    NewtonRegularization,
)
from purin.solvers.primal_dual_interior_point.reference.line_search import (
    FilterLineSearchOptions,
)
from purin.solvers.primal_dual_interior_point.reference.step import (
    fraction_to_boundary,
    compute_fraction_to_boundary_steps,
)
from purin.solvers.primal_dual_interior_point.reference.filter import (
    Filter,
    FilterEntry,
)


def example(bounded):
    if bounded:
        state = PrimalDualState(
            [0.4, 0.8], [0.4], [0.0], [0.0], [1.0, 1.0], [1.0, 1.0], [1.0], [1.0]
        )
        indices = BoundIndices([0, 1], [0, 1], [0], [0])
        bounds = BoundValues([0.0, 0.0], [2.0, 2.0], [0.0], [2.0])
    else:
        state = PrimalDualState([3.0, -2.0], [], [], [], [], [], [], [])
        indices = BoundIndices.empty()
        bounds = BoundValues.empty()

    def evaluate(state):
        x = state.x
        return NLPQuantities(
            x - np.array([0.7, 0.3]),
            np.array([x.sum() - 1]) if bounded else np.empty(0),
            np.array([x[0]]) if bounded else np.empty(0),
            np.ones((1, 2)) if bounded else np.empty((0, 2)),
            np.array([[1.0, 0.0]]) if bounded else np.empty((0, 2)),
            np.eye(2),
            0.5 * float(np.sum((x - [0.7, 0.3]) ** 2)),
        )

    return state, indices, bounds, evaluate


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("line_search", [False, True])
def test_reference_iterations_converge(bounded, line_search):
    state, indices, bounds, evaluate = example(bounded)
    options = FixedMuOptions(
        max_iterations=60,
        residual_tolerance=1e-10,
        check_newton_residual=True,
        line_search=FilterLineSearchOptions() if line_search else None,
    )
    result = run_fixed_mu_iterations(
        state, evaluate, indices, bounds, 1e-6, options=options
    )
    assert result.status == "converged"
    assert result.final_residual_inf_norm < 1e-10
    np.testing.assert_allclose(result.state.x, [0.7, 0.3], atol=2e-5)
    assert result.residual_history[0] > result.residual_history[-1]
    limited = run_fixed_mu_iterations(
        state, evaluate, indices, bounds, 1e-6, options=FixedMuOptions(max_iterations=0)
    )
    assert limited.status == "maximum_iterations"


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("regularized", [False, True])
def test_reduced_newton_matches_full_system(bounded, regularized):
    state, indices, bounds, evaluate = example(bounded)
    nlp = evaluate(state)
    regularization = NewtonRegularization() if regularized else None
    result = solve_primal_dual_newton_system(
        state, nlp, indices, bounds, 0.01, regularization=regularization
    )
    slacks = compute_bound_slacks(state, indices, bounds)
    residual = compute_kkt_residual(state, nlp, indices, slacks, 0.01)
    full = build_full_kkt_matrix(
        state, nlp, indices, slacks, regularization=regularization
    )
    np.testing.assert_allclose(
        full @ result.direction.flatten(), -residual.flatten(), atol=1e-12
    )
    np.testing.assert_allclose(
        result.direction.flatten(),
        np.linalg.solve(full, -residual.flatten()),
        atol=1e-12,
    )
    assert residual.block_sizes == state.block_sizes
    assert residual.size == state.size
    np.testing.assert_allclose(residual.as_state().flatten(), residual.flatten())
    np.testing.assert_allclose(
        residual.squared_l2_norm(), residual.flatten() @ residual.flatten()
    )
    assert max(residual.block_infinity_norms().values()) == residual.infinity_norm()
    residual.validate_against_state(state)
    copied = state.copy()
    copied.x[0] += 1
    assert copied.x[0] != state.x[0]
    assert len(dict(state.named_blocks())) == 8
    steps = compute_fraction_to_boundary_steps(
        state, result.direction, indices, bounds, 0.995
    )
    assert 0 < steps.alpha_primal <= 1 and 0 < steps.alpha_dual <= 1


def test_filter_dominance_and_invalid_pairs():
    filter_ = Filter()
    assert len(filter_) == 0
    assert filter_.acceptable(1.0, 1.0)
    filter_.add(1.0, 1.0, 0)
    assert filter_.acceptable(0.5, 2.0) and filter_.acceptable(2.0, 0.5)
    assert not filter_.acceptable(2.0, 2.0)
    assert not filter_.acceptable(np.nan, 1.0)
    assert not filter_.acceptable(1.0, -1.0)
    filter_.add(0.5, 0.5, 1)
    assert len(filter_) == 1
    filter_.add_reference_with_margin(
        barrier_objective=0.5,
        constraint_violation=0.5,
        gamma_phi=0.1,
        gamma_theta=0.1,
        iteration=2,
    )
    assert len(filter_.entries) == 1
    filter_.clear()
    assert len(filter_) == 0
    for args in [(np.inf, 1.0, 0), (1.0, -1.0, 0), (1.0, 1.0, -1)]:
        with pytest.raises(ValueError):
            FilterEntry(*args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": -1},
        {"max_iterations": 1.2},
        {"residual_tolerance": 0},
        {"tau": 0},
        {"tau": 1},
        {"step_scale": 0},
        {"minimum_step": -1},
        {"positivity_tolerance": -1},
        {"tau": np.nan},
    ],
)
def test_invalid_iteration_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        FixedMuOptions(**kwargs)


def test_fraction_to_boundary():
    assert (
        fraction_to_boundary(np.array([1.0, 2.0]), np.array([-2.0, 1.0]), 0.9) == 0.45
    )
    assert fraction_to_boundary(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0.9) == 1
    assert fraction_to_boundary(np.array([]), np.array([]), 0.9) == 1
    for values, direction, tau in [
        ([1.0], [-1.0], 0),
        ([1.0], [-1.0], 1),
        ([0.0], [-1.0], 0.9),
        ([1.0], [1.0, 2.0], 0.9),
        ([[1.0]], [[1.0]], 0.9),
    ]:
        with pytest.raises(ValueError):
            fraction_to_boundary(values, direction, tau)
