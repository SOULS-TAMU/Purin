"""Functional and matrix formulations agree on objective, constraints and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from purin.problems.functional import FunctionalProblem, functional_problem
from purin.problems.qp import ParametricQP
from purin.solvers.fischer_burmeister.dense_semismooth import (
    FischerBurmeisterProjection,
)


@pytest.mark.parametrize("sparse", [False, True])
def test_matrix_qp_compilation_and_bounds(sparse):
    convert = csr_matrix if sparse else np.asarray
    problem = ParametricQP(
        P=convert(np.diag([2.0, 3.0])),
        q0=[-2.0, -3.0],
        q_parameter_map=convert([[1.0], [2.0]]),
        A_eq=convert([[1.0, 1.0]]),
        b_eq0=[1.0],
        b_eq_parameter_map=convert([[1.0]]),
        A_ineq=convert([[-1.0, 0.0]]),
        b_ineq0=[0.0],
    )
    p = jnp.array([0.2])
    y = jnp.array([0.4, 0.8])
    canonical = problem.compile()
    assert (
        problem.variable_size,
        problem.parameter_size,
        problem.equality_size,
        problem.inequality_size,
    ) == (2, 1, 1, 1)
    np.testing.assert_allclose(
        canonical.objective_gradient(p, y),
        np.array([2.0, 3.0]) * y + np.array([-2.0, -3.0]) + np.array([1.0, 2.0]) * 0.2,
    )
    np.testing.assert_allclose(
        canonical.lagrangian_hessian(p, y, jnp.ones(1), jnp.ones(1)),
        np.diag([2.0, 3.0]),
    )
    for symbolic in [problem.to_symbolic_problem(), problem.as_projection_problem()]:
        prepared = symbolic.compile()
        assert prepared is not None
    assert problem.canonicalize().variable_size == 2
    bound = ParametricQP.from_bounds(
        P=convert(np.eye(2)),
        q0=[0.0, 0.0],
        A=convert([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        lower0=[0.0, -np.inf, 1.0],
        upper0=[2.0, 3.0, 1.0],
        lower_parameter_map=np.zeros((3, 1)),
        upper_parameter_map=np.ones((3, 1)),
    )
    assert bound.equality_size == 1 and bound.inequality_size == 3
    empty = ParametricQP.from_bounds(
        P=convert(np.eye(2)),
        q0=[0.0, 0.0],
        A=convert(np.empty((0, 2))),
        lower0=[],
        upper0=[],
    )
    assert empty.inequality_size == 0
    with pytest.raises(ValueError):
        ParametricQP.from_bounds(
            P=np.eye(2), q0=[0, 0], A=np.eye(2), lower0=[0], upper0=[1, 2]
        )


@pytest.mark.parametrize("inequality", [False, True])
def test_functional_derivatives_and_dense_projection(inequality):
    problem = FunctionalProblem(
        parameter_size=1,
        variable_size=2,
        objective=lambda p, y: 0.5 * jnp.sum(y**2),
        equalities=lambda p, y: jnp.array([y.sum() - p[0]]),
        equality_size=1,
        inequalities=(lambda p, y: -y) if inequality else (lambda p, y: jnp.empty(0)),
        inequality_size=2 if inequality else 0,
        equality_sparsity=((0, 1),),
    )
    canonical = problem.canonicalize()
    np.testing.assert_array_equal(
        canonical.objective_gradient(jnp.ones(1), jnp.ones(2)), [1, 1]
    )
    np.testing.assert_array_equal(
        canonical.lagrangian_hessian(
            jnp.ones(1),
            jnp.ones(2),
            jnp.ones(1),
            jnp.ones(2) if inequality else jnp.empty(0),
        ),
        np.eye(2),
    )
    assert problem.to_symbolic_problem().compile() is not None
    assert problem.as_projection_problem().compile() is not None
    project = FischerBurmeisterProjection(problem, tolerance=1e-9).prepare(batch_size=1)
    target = jnp.array([[2.0, -1.0]]) if inequality else jnp.array([[0.8, 0.7]])
    expected = [[1.0, 0.0]] if inequality else [[0.55, 0.45]]
    actual, info = project.with_info(target, jnp.ones((1, 1)))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_allclose(
        project(target[0], jnp.ones((1, 1))), expected[0], atol=1e-7
    )
    derivative = jax.jacrev(lambda x: project(x[None, :], jnp.ones((1, 1)))[0])(
        target[0]
    )
    assert jnp.all(jnp.isfinite(derivative))
    with pytest.raises(ValueError):
        project(jnp.zeros((2, 2)), jnp.ones((2, 1)))
    with pytest.raises(ValueError):
        project.with_info(jnp.zeros((2, 2)), jnp.ones((2, 1)))


def test_functional_factory_and_validation():
    spec = dict(parameter_size=0, variable_size=1)
    objective = lambda p, y: jnp.sum(y**2)

    @functional_problem(**spec)
    def factory():
        return dict(objective=objective)

    assert isinstance(factory, FunctionalProblem)

    @functional_problem()
    def instance():
        return factory

    assert instance is factory
    with pytest.raises(TypeError):
        functional_problem(**spec)(lambda: 3)
    for values in [
        dict(parameter_size=-1, variable_size=1),
        dict(parameter_size=0, variable_size=0),
        dict(**spec, equality_size=-1),
    ]:
        with pytest.raises(ValueError):
            FunctionalProblem(**values, objective=objective)
    with pytest.raises(TypeError):
        FunctionalProblem(**spec, objective=None)
    for sparsity in [(), ((2,),)]:
        with pytest.raises(ValueError):
            FunctionalProblem(
                **spec, objective=objective, equality_size=1, equality_sparsity=sparsity
            ).compile()
    with pytest.raises(TypeError):
        FischerBurmeisterProjection(object()).prepare(batch_size=1)
    with pytest.raises(ValueError):
        FischerBurmeisterProjection(factory).prepare(batch_size=0)
    with pytest.raises(ValueError):
        FischerBurmeisterProjection(factory, gradient_policy="bad").prepare(
            batch_size=1
        )
