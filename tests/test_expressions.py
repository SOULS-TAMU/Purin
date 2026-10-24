"""Symbolic arithmetic agrees with explicit JAX values and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from purin import Projection
from purin.problems.expressions import (
    VectorExpression,
    Constraint,
    _as_expression,
    _EvaluationContext,
)


def test_vector_algebra_and_matrix_builders():
    problem = Projection()
    p = problem.add_parameter(["a", "b"])
    y = problem.add_variable(["u", "v"])
    v = y.vector()
    context = _EvaluationContext(
        jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0]), {"u": 0, "v": 1}, {"a": 0, "b": 1}
    )
    pairs = [
        (v + 1, [3.0, 4.0]),
        (1 + v, [3.0, 4.0]),
        (v - p, [-2.0, -2.0]),
        (1 - v, [-1.0, -2.0]),
        (v * 2, [4.0, 6.0]),
        (2 * v, [4.0, 6.0]),
        (v / [2.0, 3.0], [1.0, 1.0]),
        (-v, [-2.0, -3.0]),
    ]
    for expr, expected in pairs:
        np.testing.assert_array_equal(expr.evaluate(context), expected)
    assert v.sum().evaluate(context) == 5
    assert len(y) == 2 and len(list(y)) == 2 and y[0].evaluate(context) == 2
    assert "VectorExpression" in repr(v) and "Expression" in repr(y.u)
    assert VectorExpression([], "empty").evaluate(context).shape == (0,)
    for constraints in [v <= 4, v >= 1, v == [2, 3]]:
        problem.constraints.add(constraints)
    problem.constraints.equality(y.u - 2, name="u")
    problem.constraints.inequality(y.v - 4)
    assert len(problem.constraints) == 8
    matrix = np.array([[1.0, 2.0], [0.0, 3.0]])
    np.testing.assert_allclose(
        problem.lin(matrix, y).evaluate(context), matrix @ context.variables
    )
    np.testing.assert_allclose(problem.lin([1.0, 2.0], v).evaluate(context), [8.0])
    np.testing.assert_allclose(
        problem.quad(matrix, y).evaluate(context),
        context.variables @ matrix @ context.variables,
    )
    for method, good, bad in [
        (problem.matrix, np.ones((2, 2)), [1]),
        (problem.vector, [1, 2], [[1, 2]]),
        (problem.tensor, np.ones((1, 2, 3)), [[1, 2]]),
    ]:
        np.testing.assert_array_equal(method(good), good)
        with pytest.raises(ValueError):
            method(bad)
    for method in [problem.lin, problem.quad]:
        with pytest.raises(TypeError):
            method([[1]], 3)
        with pytest.raises(ValueError):
            method([[1]], y)
    with pytest.raises(TypeError):
        bool(y.u)
    with pytest.raises(TypeError):
        _as_expression([1, 2])
    assert _as_expression(np.array(2.0)).evaluate(context) == 2
    for bad in [[1.0], np.ones((2, 2)), VectorExpression([y.u], "one")]:
        with pytest.raises(ValueError):
            v + bad
    with pytest.raises(AttributeError):
        y.unknown
    with pytest.raises(KeyError):
        context.value("unknown", "u")
    with pytest.raises(ValueError):
        Constraint(y.u, "bad", "bad")
    with pytest.raises(TypeError):
        problem.constraints.add(True)


def test_nonlinear_arithmetic_and_derivatives():
    problem = Projection()
    p = problem.add_parameter("a")
    y = problem.add_variable("u")
    expression = (
        (3 / y.u)
        + (y.u**p.a)
        + (2**y.u)
        + problem.sin(y.u)
        + problem.cos(y.u)
        + problem.exp(y.u)
        + problem.log(y.u)
        + problem.sqrt(y.u)
        + problem.abs(y.u)
    )
    problem.objective = expression
    actual = problem.compile()
    theta = jnp.array([2.0])
    values = jnp.array([1.5])
    reference = lambda u: (
        3 / u
        + u**2
        + 2**u
        + jnp.sin(u)
        + jnp.cos(u)
        + jnp.exp(u)
        + jnp.log(u)
        + jnp.sqrt(u)
        + jnp.abs(u)
    )
    np.testing.assert_allclose(actual.objective(theta, values), reference(values[0]))
    np.testing.assert_allclose(
        actual.objective_gradient(theta, values), [jax.grad(reference)(values[0])]
    )
    # Power expressions retain their numerical derivative contract.
    power = Projection()
    z = power.add_variable("z")
    power.objective = z.z**0 + z.z**1
    assert power.compile().objective(jnp.empty(0), jnp.array([2.0])) == 3
    assert not (y.u**2).affine_in_variables
    assert not (y.u / y.u).affine_in_variables
    for names in [[], [""], ["a", "a"]]:
        with pytest.raises(ValueError):
            Projection().add_parameter(names)
    with pytest.raises(ValueError):
        problem.add_variable("a")
    with pytest.raises(ValueError):
        Projection().compile()
    empty = Projection()
    empty.add_variable("u")
    with pytest.raises(ValueError):
        empty.compile()


def test_projection_definition_validation():
    from purin import ChambollePockConfig

    assert Projection(config=ChambollePockConfig()).config["accelerated"] is False
    assert Projection(config={"a": 1}).config == {"a": 1}
    with pytest.raises(TypeError):
        Projection(config=3)
    p = Projection().solver("admm", tolerance=1e-8)
    assert p.method == "admm" and p.config["tolerance"] == 1e-8
    q = p.add_parameter(["a", "b"])
    y = p.add_variable(["u", "v"])
    for target in [[q.a], 3, [y.u, y.v]]:
        with pytest.raises((ValueError, TypeError)):
            p.set_projection_objective(variables=y, target=target)
    with pytest.raises(ValueError):
        p.set_projection_objective(variables=[y.u + 1, y.v], target=[q.a, q.b])
