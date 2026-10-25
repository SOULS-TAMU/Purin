"""Sparse and barrier kernels checked against independent dense algebra."""

from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin.solvers.primal_dual_interior_point import operators
from purin.solvers.primal_dual_interior_point.types import (
    BoundIndices,
    BoundSlacks,
    NLPQuantities,
    PrimalDualState,
)
from purin.solvers.primal_dual_interior_point.filter_metrics import (
    compute_constraint_violation,
    compute_barrier_objective,
    compute_barrier_gradient,
    compute_barrier_directional_derivative,
)


def test_barrier_value_gradient_and_direction_against_autodiff():
    empty = jnp.empty(0)
    x, s = jnp.array([0.4, 0.7]), jnp.array([0.6])
    state = PrimalDualState(x, s, empty, empty, empty, empty, empty, empty)
    direction = state._replace(x=jnp.array([0.2, -0.1]), s=jnp.array([0.3]))
    indices = BoundIndices(
        jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([0]), jnp.array([0])
    )
    slacks = BoundSlacks(x, 2 - x, s, 3 - s)
    nlp = NLPQuantities(
        x,
        jnp.array([0.2, -0.3]),
        jnp.array([0.9]),
        jnp.empty((2, 2)),
        jnp.empty((1, 2)),
        jnp.eye(2),
        jnp.sum(x * x) / 2,
    )
    mu = 0.1

    def objective(x, s):
        return jnp.sum(x * x) / 2 - mu * (
            jnp.log(x).sum()
            + jnp.log(2 - x).sum()
            + jnp.log(s).sum()
            + jnp.log(3 - s).sum()
        )

    np.testing.assert_allclose(compute_constraint_violation(state, nlp), 0.8)
    np.testing.assert_allclose(
        compute_barrier_objective(nlp, slacks, mu), objective(x, s)
    )
    actual = compute_barrier_gradient(state, nlp, indices, slacks, mu)
    expected = jax.grad(objective, argnums=(0, 1))(x, s)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-12)
    slope = expected[0] @ direction.x + expected[1] @ direction.s
    np.testing.assert_allclose(
        compute_barrier_directional_derivative(
            state, direction, nlp, indices, slacks, mu
        ),
        slope,
    )
    with pytest.raises(ValueError, match="objective_value"):
        compute_barrier_objective(nlp._replace(objective_value=None), slacks, mu)
    np.testing.assert_allclose(state.axpy(0.5, direction).x, x + 0.5 * direction.x)


def test_pallas_affine_kkt_interpreter_matches_dense_operator(monkeypatch):
    if not operators.pallas_available():
        pytest.skip("Pallas is not available in this JAX build")
    jac = jnp.array([[1.0, 2.0], [-1.0, 3.0]])
    kwargs = dict(
        jacobian=jac,
        row_support=jnp.array([[0, 1], [0, 1]]),
        row_mask=jnp.ones((2, 2), bool),
        column_rows=jnp.array([[0, 1], [0, 1]]),
        column_slots=jnp.array([[0, 0], [1, 1]]),
        column_mask=jnp.ones((2, 2), bool),
        n_variables=2,
        block_size=3,
        interpret=True,
    )
    kernel = operators.build_affine_constant_kkt_pallas_matvec(**kwargs)
    vectors = jnp.array([[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, -1.0]])
    primal = jnp.array([[2.0, 4.0], [1.0, 3.0]])
    dual = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    expected = np.concatenate(
        (
            primal * np.asarray(vectors[:, :2])
            + np.asarray(vectors[:, 2:]) @ np.asarray(jac),
            np.asarray(vectors[:, :2]) @ np.asarray(jac).T
            - dual * np.asarray(vectors[:, 2:]),
        ),
        axis=1,
    )
    np.testing.assert_allclose(kernel(vectors, primal, dual), expected, atol=1e-12)
    for vector in [jnp.ones(4), jnp.ones((2, 3))]:
        with pytest.raises(ValueError):
            kernel(vector, primal, dual)
    for changes in [{"block_size": 0}, {"jacobian": jnp.ones(2)}]:
        with pytest.raises(ValueError):
            operators.build_affine_constant_kkt_pallas_matvec(**{**kwargs, **changes})
    monkeypatch.setattr(operators, "pl", None)
    assert not operators.pallas_available()
    with pytest.raises(RuntimeError, match="Pallas"):
        operators.build_affine_constant_kkt_pallas_matvec(**kwargs)


def test_sparse_padding_and_power_of_two():
    assert [operators._next_power_of_two(n) for n in [0, 1, 2, 3, 7]] == [1, 1, 2, 4, 8]
    np.testing.assert_array_equal(
        operators._pad_second_axis([[1, 2]], 3, fill_value=-1), [[1, 2, -1]]
    )
    np.testing.assert_array_equal(operators._pad_second_axis([[1, 2]], 2), [[1, 2]])
    for array, width in [([1, 2], 3), ([[1, 2]], 1)]:
        with pytest.raises(ValueError):
            operators._pad_second_axis(array, width)


@pytest.mark.parametrize("prefix", [0, 1])
def test_batched_gmres_dense_reference_and_exact_warm_start(prefix):
    matrix = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rhs = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs).T).T
    kwargs = dict(
        tolerance=1e-11,
        absolute_tolerance=1e-12,
        restart=3,
        max_iterations=6,
        arnoldi_prefix_block=prefix,
        return_iterations=True,
    )
    for initial in [None, jnp.asarray(expected)]:
        result, status, residual, iterations = operators._batched_restarted_gmres(
            lambda x: x @ matrix.T,
            rhs,
            lambda x: x / jnp.diag(matrix),
            initial_solution=initial,
            **kwargs,
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)
        np.testing.assert_array_equal(status, [0, 0])
        assert np.max(residual) < 1e-10
        assert np.max(iterations) <= 6
        if initial is not None:
            np.testing.assert_array_equal(iterations, [0, 0])


@pytest.mark.parametrize("equality_batched", [False, True])
def test_mixed_jacobian_layouts_preserve_batch_order(equality_batched):
    eq = jnp.array([[1.0, 2.0]])
    iq = jnp.array([[3.0, 4.0]])
    state = SimpleNamespace(
        x=jnp.zeros((2, 2)), y_c=jnp.zeros((2, 1)), y_d=jnp.zeros((2, 1))
    )
    numeric = SimpleNamespace(
        equality_jacobian=jnp.broadcast_to(eq, (2, 1, 2)) if equality_batched else eq,
        inequality_jacobian=iq if equality_batched else jnp.broadcast_to(iq, (2, 1, 2)),
    )
    combined = operators._combine_projection_jacobians(state, numeric)
    np.testing.assert_array_equal(
        combined, np.broadcast_to([[1.0, 2.0], [3.0, 4.0]], (2, 2, 2))
    )
    constant = jnp.ones((2, 2))
    assert (
        operators._combine_projection_jacobians(
            state, numeric, constant_combined_jacobian=constant
        )
        is constant
    )


def test_sparse_constraint_derivatives_match_analytic_polynomial():
    from purin import Projection

    problem = Projection()
    p = problem.add_parameter(["p", "target_u", "target_v"])
    y = problem.add_variable(["u", "v"])
    problem.set_projection_objective(variables=y, target=[p.target_u, p.target_v])
    problem.constraints.add(y.u**2 + y.u * y.v == p.p, y.v**3 <= 4, p.p >= 0)
    structure = problem._build_projection_structure()
    eq, iq, hvp = problem._build_sparse_projection_derivatives(structure)
    variables, parameters = jnp.array([2.0, 3.0]), jnp.array([10.0, 0.0, 0.0])
    np.testing.assert_allclose(eq(variables, parameters), [7.0, 2.0])
    np.testing.assert_allclose(iq(variables, parameters), [27.0])
    # 2*H(u^2+uv)@[1,2] + 3*H(v^3)@[1,2].
    np.testing.assert_allclose(
        hvp(
            variables,
            parameters,
            jnp.array([2.0]),
            jnp.array([3.0, 5.0]),
            jnp.array([1.0, 2.0]),
        ),
        [8.0, 110.0],
    )
    local_eq, local_iq, *_ = problem._build_local_projection_derivatives(structure)
    eq_jac, eq_hess = local_eq(variables, parameters)
    iq_jac, iq_hess = local_iq(variables, parameters)
    np.testing.assert_allclose(eq_jac, [[7.0, 2.0]])
    np.testing.assert_allclose(eq_hess, [[[2.0, 1.0], [1.0, 0.0]]])
    np.testing.assert_allclose(iq_jac, [[27.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(iq_hess[0], [[18.0, 0.0], [0.0, 0.0]])


def test_affine_expression_without_coefficient_metadata_uses_correct_derivatives():
    from purin import Projection
    from purin.problems.expressions import Constraint, Expression

    problem = Projection()
    p = problem.add_parameter(["p", "target_u", "target_v"])
    y = problem.add_variable(["u", "v"])
    problem.set_projection_objective(variables=y, target=[p.target_u, p.target_v])
    residual = Expression(
        lambda context: (
            2 * context.variables[0] + context.variables[1] - context.parameters[0]
        ),
        "2*u + v - p",
        variable_dependencies={"u", "v"},
        parameter_dependencies={"p"},
        affine_in_variables=True,
    )
    problem.constraints.add(Constraint(residual, "eq", "2*u + v == p"))
    structure = problem._build_projection_structure()
    eq, iq, eq_offset, iq_offset, eq_constant, iq_constant = (
        problem._build_local_projection_derivatives(structure)
    )
    variables, parameters = jnp.array([1.0, 2.0]), jnp.array([4.0, 0.0, 0.0])
    jacobian, hessian = eq(variables, parameters)
    np.testing.assert_array_equal(jacobian, [[2.0, 1.0]])
    np.testing.assert_array_equal(hessian, np.zeros((1, 2, 2)))
    assert eq_offset is None and eq_constant is None
    assert iq(variables, parameters)[0].shape == (0, 2)
    assert iq_offset(variables, parameters).shape == (0,)
    assert iq_constant.shape == (0, 2)
