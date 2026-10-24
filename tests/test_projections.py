"""Numerical projections checked against analytic solutions and Jacobians."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from purin import (
    ADMMConfig,
    ChambollePockConfig,
    Distance,
    FischerBurmeisterConfig,
    LinearEqualityConfig,
    Projection,
    ProjectionConfig,
    Sequential,
    WeightedDistance,
    registered_methods,
)
from purin.projections.closed_form import AffineProjection, BoxProjection


METHODS = (
    "linear-equality",
    "chambolle-pock",
    "admm",
    "fischer-burmeister-newton",
    "fischer-burmeister-reduced",
    "primal-dual-interior-point",
)


def make_problem(*, inequalities=False, nonlinear=False):
    problem = Projection()
    p = problem.add_parameter(["b"])
    y = problem.add_variable(["u", "v"])
    problem.objective = 0.0
    if nonlinear:
        problem.constraints.add(y.u**2 + y.v**2 <= p.b)
    else:
        problem.constraints.add(y.u + y.v == p.b)
        if inequalities:
            problem.constraints.add(y.u >= 0)
            problem.constraints.add(y.v >= 0)
    return problem


def config(method):
    if method == "linear-equality":
        return LinearEqualityConfig()
    if method == "chambolle-pock":
        return ChambollePockConfig(tolerance=1e-9, max_iterations=3000)
    if method == "admm":
        return ADMMConfig(tolerance=1e-9, max_iterations=3000)
    if method.startswith("fischer"):
        return FischerBurmeisterConfig(tolerance=1e-9, max_iterations=60)
    return ProjectionConfig(tolerance=1e-9, mu=1e-10, max_iterations=100)


def test_registered_algorithms():
    assert set(registered_methods()) == set(METHODS)


@pytest.mark.parametrize("method", METHODS)
def test_affine_projection_and_analytic_jacobian(method):
    project = Distance(make_problem(), method, config(method)).prepare(batch_size=1)
    target = jnp.array([[0.8, 0.7]])
    context = jnp.ones((1, 1))
    values = project(target, context)
    np.testing.assert_allclose(values, [[0.55, 0.45]], atol=2e-6, rtol=0)
    derivative = jax.jacrev(lambda x: project(x[None, :], context)[0])(target[0])
    np.testing.assert_allclose(
        derivative, [[0.5, -0.5], [-0.5, 0.5]], atol=2e-5, rtol=0
    )


@pytest.mark.parametrize("method", METHODS[1:])
def test_projection_with_active_inequality(method):
    project = Distance(make_problem(inequalities=True), method, config(method)).prepare(
        batch_size=1
    )
    values = project(jnp.array([[2.0, -1.0]]), jnp.ones((1, 1)))
    np.testing.assert_allclose(values, [[1.0, 0.0]], atol=2e-5, rtol=0)


@pytest.mark.parametrize("accelerated", [False, True])
def test_cp_variants(accelerated):
    cfg = ChambollePockConfig(
        accelerated=accelerated, tolerance=1e-9, max_iterations=3000
    )
    project = Distance(make_problem(inequalities=True), "chambolle-pock", cfg).prepare(
        batch_size=1
    )
    np.testing.assert_allclose(
        project(jnp.array([[2.0, -1.0]]), jnp.ones((1, 1))), [[1.0, 0.0]], atol=2e-6
    )


@pytest.mark.parametrize("method", METHODS[3:])
def test_nonlinear_disk_projection(method):
    project = Distance(make_problem(nonlinear=True), method, config(method)).prepare(
        batch_size=1
    )
    values = project(jnp.array([[2.0, 0.0]]), jnp.ones((1, 1)))
    np.testing.assert_allclose(values, [[1.0, 0.0]], atol=2e-5, rtol=0)


@pytest.mark.parametrize("method", METHODS[:3])
def test_affine_only_methods_reject_nonlinear_constraints(method):
    with pytest.raises((ValueError, TypeError), match="affine|linear"):
        Distance(make_problem(nonlinear=True), method).prepare(batch_size=1)


def test_linear_equality_rejects_inequalities():
    with pytest.raises(ValueError, match="inequalities"):
        Distance(make_problem(inequalities=True), "linear-equality").prepare(
            batch_size=1
        )


def test_weighted_projection_matches_diagonal_metric_solution():
    project = WeightedDistance(make_problem(), [1.0, 4.0]).prepare(batch_size=1)
    values = project(jnp.array([[2.0, 0.0]]), jnp.ones((1, 1)))
    np.testing.assert_allclose(values, [[1.2, -0.2]], atol=2e-6, rtol=0)


def test_sequential_projection_is_idempotent():
    project = Sequential(Distance(make_problem(), "linear-equality"), layers=3).prepare(
        batch_size=1
    )
    np.testing.assert_allclose(
        project(jnp.array([[2.0, 0.0]]), jnp.ones((1, 1))), [[1.5, -0.5]], atol=1e-12
    )


def test_closed_form_projections():
    box = BoxProjection(jnp.array([0.0, -1.0]), jnp.array([1.0, 1.0]))
    np.testing.assert_array_equal(box(jnp.array([[-2.0, 3.0]])), [[0.0, 1.0]])
    affine = AffineProjection(jnp.array([[1.0, 1.0]]), jnp.array([1.0]))
    np.testing.assert_allclose(
        affine(jnp.array([[2.0, 0.0]])), [[1.5, -0.5]], atol=1e-12
    )


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (LinearEqualityConfig, {"regularization": -1}),
        (ChambollePockConfig, {"max_iterations": 0}),
        (ProjectionConfig, {"tolerance": 0}),
        (FischerBurmeisterConfig, {"max_iterations": 0}),
    ],
)
def test_invalid_solver_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
