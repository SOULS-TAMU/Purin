"""Numerical configurations and reference containers reject malformed resources."""

from dataclasses import replace
import numpy as np
import pytest
from purin.solvers.config import (
    LinearSolverConfig,
    ImplicitGradientConfig,
    ProjectionConfig,
    LinearEqualityConfig,
    FischerBurmeisterConfig,
)
from purin.solvers.semismooth import SemismoothNewtonConfig
from purin.solvers.primal_dual_interior_point.reference.line_search import (
    FilterLineSearchOptions,
)
from purin.solvers.primal_dual_interior_point.reference.newton import (
    NewtonRegularization,
    ReducedKKTSystem,
)
from purin.solvers.primal_dual_interior_point.reference.bounds import (
    BoundIndices,
    BoundValues,
)
from purin.solvers.primal_dual_interior_point.reference.state import PrimalDualState
from test_reference_ipm import example

INVALID = [
    (LinearSolverConfig, dict(tolerance=0)),
    (LinearSolverConfig, dict(atol=-1)),
    (LinearSolverConfig, dict(restart=0)),
    (LinearSolverConfig, dict(max_iterations=0)),
    (ImplicitGradientConfig, dict(policy="bad")),
    (ImplicitGradientConfig, dict(invalid_batch_policy="bad")),
    (ImplicitGradientConfig, dict(inexact_tolerance=-1)),
    (ImplicitGradientConfig, dict(adjoint_tolerance=-1)),
    (ProjectionConfig, dict(max_iterations=-1)),
    (ProjectionConfig, dict(tolerance=np.nan)),
    (ProjectionConfig, dict(mu=-1)),
    (ProjectionConfig, dict(max_backtracking_steps=-1)),
    (ProjectionConfig, dict(backtracking_factor=1)),
    (LinearEqualityConfig, dict(regularization=-1)),
    (FischerBurmeisterConfig, dict(tolerance=0)),
    (FischerBurmeisterConfig, dict(smoothing=-1)),
    (FischerBurmeisterConfig, dict(max_iterations=0)),
    (FischerBurmeisterConfig, dict(maximum_regularization_attempts=0)),
    (FischerBurmeisterConfig, dict(maximum_backtracking_steps=0)),
    (FischerBurmeisterConfig, dict(armijo_constant=0)),
    (FischerBurmeisterConfig, dict(backtracking_factor=1)),
    (FischerBurmeisterConfig, dict(regularization_growth=0.5)),
    (FischerBurmeisterConfig, dict(transpose_mode="bad")),
    (SemismoothNewtonConfig, dict(tolerance=-1)),
    (SemismoothNewtonConfig, dict(max_iterations=0)),
    (SemismoothNewtonConfig, dict(regularization=-1)),
    (SemismoothNewtonConfig, dict(max_line_search_steps=0)),
    (SemismoothNewtonConfig, dict(line_search_factor=1)),
]


@pytest.mark.parametrize("factory,kwargs", INVALID)
def test_invalid_solver_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alpha_reduction_factor=1),
        dict(maximum_backtracks=-1),
        dict(maximum_backtracks=1.2),
        dict(minimum_alpha=-1),
        dict(gamma_phi=0),
        dict(gamma_theta=1),
        dict(eta_phi=0),
        dict(delta=0),
        dict(s_phi=1),
        dict(s_theta=1),
        dict(theta_min_factor=0),
        dict(theta_max_factor=1e-6),
        dict(objective_max_increase_orders=0),
        dict(minimum_alpha=np.nan),
    ],
)
def test_invalid_line_search_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        FilterLineSearchOptions(**kwargs)


def test_regularization_and_reduced_system_validation():
    for key in ["delta_x", "delta_s", "delta_c", "delta_d"]:
        for value in [-1.0, np.inf]:
            with pytest.raises(ValueError):
                NewtonRegularization(**{key: value})
    values = dict(
        matrix=np.eye(2), rhs=np.ones(2), sigma_x=np.ones(1), sigma_s=np.ones(1)
    )
    system = ReducedKKTSystem(**values)
    assert system.dimension == 2 and system.symmetry_error() == 0
    np.testing.assert_array_equal(system.linear_residual(np.ones(2)), [0, 0])
    with pytest.raises(ValueError):
        system.linear_residual(np.ones(3))
    for key, bad in [
        ("matrix", np.ones((2, 3))),
        ("rhs", np.ones(3)),
        ("sigma_x", np.ones((1, 1))),
        ("sigma_s", np.ones((1, 1))),
        ("matrix", np.full((2, 2), np.nan)),
    ]:
        with pytest.raises(ValueError):
            ReducedKKTSystem(**{**values, key: bad})
    assert (
        ReducedKKTSystem(
            np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0)
        ).symmetry_error()
        == 0
    )


def test_reference_state_bounds_and_derivative_validation():
    state, indices, bounds, evaluate = example(True)
    for bad in [np.array([[0]]), np.array([0.0]), np.array([-1]), np.array([0, 0])]:
        with pytest.raises((ValueError, TypeError)):
            BoundIndices(
                bad,
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
            )
    for bad in [[[0.0]], [np.nan]]:
        with pytest.raises(ValueError):
            BoundValues(bad, [], [], [])
    with pytest.raises(ValueError):
        bounds.validate_against_indices(BoundIndices.empty())
    with pytest.raises(ValueError):
        replace(bounds, x_lower=np.array([3.0, 3.0])).validate_two_sided_bounds(indices)
    with pytest.raises(ValueError):
        replace(bounds, s_lower=np.array([3.0])).validate_two_sided_bounds(indices)
    with pytest.raises(IndexError):
        replace(indices, x_lower=np.array([0, 9])).validate_against_state(state)
    with pytest.raises(ValueError):
        replace(state, x=np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        replace(state, y_d=np.empty(0))
    with pytest.raises(ValueError):
        replace(state, x=np.array([np.nan, 1.0])).validate_finite()
    empty = PrimalDualState.zeros(
        n_x=0, n_s=0, n_c=0, n_x_lower=0, n_x_upper=0, n_s_lower=0, n_s_upper=0
    )
    assert empty.flatten().size == 0
    with pytest.raises(ValueError):
        PrimalDualState.from_flat(np.ones(1), template=state)
    nlp = evaluate(state)
    for key in ["grad_f", "c", "d", "jac_c", "jac_d", "hessian_lagrangian"]:
        bad = np.full_like(getattr(nlp, key), np.nan)
        with pytest.raises(ValueError):
            replace(nlp, **{key: bad})
    with pytest.raises(ValueError):
        replace(nlp, objective_value=np.inf)
    with pytest.raises(ValueError):
        replace(nlp, grad_f=np.ones(3)).validate_against_state(state)
    with pytest.raises(ValueError):
        replace(
            nlp, hessian_lagrangian=np.array([[1.0, 2.0], [0.0, 1.0]])
        ).validate_hessian_symmetry()
