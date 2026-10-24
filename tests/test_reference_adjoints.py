"""Implicit adjoints checked against full transpose solves and analytic VJPs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_reference_ipm import example
from purin.solvers.primal_dual_interior_point import implicit_compat_numpy as host
from purin.solvers.primal_dual_interior_point import implicit_compat_jax as device
from purin.solvers.primal_dual_interior_point import types


@pytest.mark.parametrize("bounded", [False, True])
def test_numpy_and_jax_adjoint_agree_with_full_kkt(bounded):
    state, indices, bounds, evaluate = example(bounded)
    nlp = evaluate(state)
    cot = host.primal_output_cotangent(state, np.array([2.0, -1.0]))
    result = host.solve_implicit_adjoint(state, nlp, indices, bounds, cot)
    full = host.solve_full_adjoint_direct(state, nlp, indices, bounds, cot)
    np.testing.assert_allclose(result.adjoint.flatten(), full.flatten(), atol=1e-12)
    assert result.full_residual_inf_norm < 1e-12
    multi = host.solve_implicit_adjoint_multi_rhs(
        state, nlp, indices, bounds, [cot, cot]
    )
    np.testing.assert_allclose(
        multi.adjoint_matrix, np.column_stack([full.flatten()] * 2), atol=1e-12
    )
    with pytest.raises(ValueError):
        host.solve_implicit_adjoint_multi_rhs(state, nlp, indices, bounds, [])
    js = types.PrimalDualState(*(jnp.asarray(x) for x in state.blocks()))
    ji = types.BoundIndices(
        *(
            jnp.asarray(getattr(indices, k))
            for k in ["x_lower", "x_upper", "s_lower", "s_upper"]
        )
    )
    jb = types.BoundValues(
        *(
            jnp.asarray(getattr(bounds, k))
            for k in ["x_lower", "x_upper", "s_lower", "s_upper"]
        )
    )
    jn = types.NLPQuantities(
        *(
            jnp.asarray(getattr(nlp, k))
            for k in ["grad_f", "c", "d", "jac_c", "jac_d", "hessian_lagrangian"]
        )
    )
    jc = device.primal_output_cotangent(js, jnp.array([2.0, -1.0]))
    actual = device.solve_implicit_adjoint(js, jn, ji, jb, jc)
    np.testing.assert_allclose(actual.adjoint.flatten(), full.flatten(), atol=1e-12)
    many = device.solve_implicit_adjoint_multi_rhs(
        js, jn, ji, jb, jnp.column_stack([jc.flatten()] * 2)
    )
    np.testing.assert_allclose(many.adjoint_matrix, multi.adjoint_matrix, atol=1e-12)
    np.testing.assert_array_equal(
        device.state_from_flat(js.flatten(), js).flatten(), js.flatten()
    )


def test_parameter_vjp_callbacks_and_finite_difference():
    state, indices, bounds, evaluate = example(False)
    adjoint = host.solve_implicit_adjoint(
        state,
        evaluate(state),
        indices,
        bounds,
        host.primal_output_cotangent(state, [2.0, -1.0]),
    ).adjoint
    matrix = np.array([[2.0, 1.0], [-1.0, 3.0]])
    expected = -matrix.T @ adjoint.flatten()
    np.testing.assert_allclose(
        host.parameter_vjp_from_jacobian(adjoint, matrix), expected
    )
    np.testing.assert_allclose(
        host.parameter_vjp_from_callback(adjoint, lambda seed: matrix.T @ seed),
        expected,
    )
    for fn, arg in [
        (host.parameter_vjp_from_jacobian, matrix),
        (host.parameter_vjp_from_callback, lambda seed: matrix.T @ seed),
    ]:
        np.testing.assert_allclose(
            fn(adjoint, arg, direct_parameter_gradient=np.ones(2)), expected + 1
        )
        with pytest.raises(ValueError):
            fn(adjoint, arg, direct_parameter_gradient=np.ones(3))
    np.testing.assert_allclose(
        host.finite_difference_residual_parameter_jacobian(
            lambda p: matrix @ p, np.ones(2)
        ),
        matrix,
        atol=1e-9,
    )
    for bad in [np.ones(2), np.ones((3, 2))]:
        with pytest.raises(ValueError):
            host.parameter_vjp_from_jacobian(adjoint, bad)
    with pytest.raises(TypeError):
        host.parameter_vjp_from_callback(adjoint, None)
    with pytest.raises(ValueError):
        host.parameter_vjp_from_callback(adjoint, lambda seed: np.ones((2, 2)))
    for fn, params, step in [
        (None, np.ones(2), 1e-6),
        (lambda x: x, np.ones((2, 2)), 1e-6),
        (lambda x: x, np.ones(2), 0),
        (lambda x: x[:, None], np.ones(2), 1e-6),
    ]:
        with pytest.raises((ValueError, TypeError)):
            host.finite_difference_residual_parameter_jacobian(
                fn, params, relative_step=step
            )


def test_jax_final_kkt_vjp_is_analytic():
    indices = types.BoundIndices(*(jnp.empty(0, dtype=jnp.int32) for _ in range(4)))
    bounds = types.BoundValues(*(jnp.empty(0) for _ in range(4)))

    def forward(theta):
        return types.PrimalDualState(theta**2, *(jnp.empty(0) for _ in range(7)))

    def quantities(state, theta):
        return types.NLPQuantities(
            state.x - theta**2,
            jnp.empty(0),
            jnp.empty(0),
            jnp.empty((0, 2)),
            jnp.empty((0, 2)),
            jnp.eye(2),
        ), bounds

    solve = device.make_implicit_differentiable_solver(
        forward, kkt_data_function=quantities, indices=indices, mu=0.0
    )
    theta = jnp.array([2.0, 3.0])
    np.testing.assert_array_equal(solve(theta).x, theta**2)
    np.testing.assert_allclose(
        jax.jacrev(lambda t: solve(t).x)(theta), np.diag([4.0, 6.0])
    )
    for fn, data in [(None, quantities), (forward, None)]:
        with pytest.raises(TypeError):
            device.make_implicit_differentiable_solver(
                fn, kkt_data_function=data, indices=indices, mu=0.0
            )
