"""Public symbolic solve modes and input bindings preserve the same physical solution."""

import numpy as np
import pytest
from purin import Projection
from purin.problems.types import NamedValues, BatchNamedValues
from purin.solvers.primal_dual_interior_point.reference.iteration import FixedMuOptions


def distance_problem():
    problem = Projection()
    p = problem.add_parameter(["b", "target_u", "target_v"])
    y = problem.add_variable(["u", "v"])
    problem.set_projection_objective(variables=y, target=[p.target_u, p.target_v])
    problem.constraints.add(y.u + y.v == p.b)
    return problem


@pytest.mark.parametrize(
    "mode",
    ["auto", "shared_linear", "dense_batched", "sparse_iterative", "generic_dense"],
)
def test_distance_solve_modes(mode):
    problem = distance_problem()
    kwargs = dict(
        parameters={"b": [1.0, 2.0], "target_u": 0.8, "target_v": 0.7},
        batch_size=2,
        initial_values={"u": [0.0, 0.0]},
        projection_mode=mode,
        mu=1e-8,
        options=FixedMuOptions(max_iterations=15, residual_tolerance=1e-8),
    )
    result = problem.solve_batch(**kwargs)
    np.testing.assert_allclose(
        result.variables.matrix(), [[0.55, 0.45], [1.05, 0.95]], atol=1e-7
    )
    assert result.variables.batch_size == 2
    np.testing.assert_allclose(result.variables.u, [0.55, 1.05], atol=1e-7)
    if mode == "sparse_iterative":
        other = problem.solve_batch(**kwargs)
        np.testing.assert_array_equal(
            other.variables.matrix(), result.variables.matrix()
        )
        info = problem.batch_solver_cache_info()
        assert sum(info.values()) > 0
    # Every projection compiler exposes matching static sparse supports.
    canonical = problem.compile()
    structure = problem._build_projection_structure()
    assert structure.jacobian_nonzeros == 2
    problem.set_sparsity_override(
        canonical.equality_sparsity, canonical.inequality_sparsity
    )
    assert problem.compile().equality_sparsity == ((0, 1),)
    problem.set_sparsity_override(((0,),), ())
    with pytest.raises(ValueError):
        problem.compile()


def test_general_nlp_single_and_batch_solve():
    problem = Projection()
    y = problem.add_variable(["u", "v"])
    problem.objective = (y.u - 1) ** 2 + (y.v - 2) ** 2
    problem.constraints.add(y.u + y.v == 3, y.u >= 0)
    result = problem.solve(
        initial_values={"u": 0.2, "v": 2.8},
        mu=1e-8,
        options=FixedMuOptions(max_iterations=40, residual_tolerance=1e-8),
    )
    np.testing.assert_allclose(result.variables.vector(), [1.0, 2.0], atol=1e-7)
    assert result.variables.u == pytest.approx(1.0, abs=1e-7)
    assert len(result.parameters) == 0
    np.testing.assert_allclose(
        problem.optimize(mu=1e-8).variables.vector(), [1.0, 2.0], atol=1e-6
    )
    for mode in ["generic_dense", "dense_batched", "auto"]:
        batch = problem.optimize_batch(
            parameters=None,
            batch_size=2,
            initial_values=np.array([0.5, 2.5]),
            projection_mode=mode,
            mu=1e-8,
        )
        np.testing.assert_allclose(
            batch.variables.matrix(), [[1.0, 2.0]] * 2, atol=1e-6
        )
    for mode in ["bad", "shared_linear", "sparse_iterative"]:
        with pytest.raises(ValueError):
            problem.solve_batch(parameters=None, batch_size=1, projection_mode=mode)


def test_parameter_and_initial_value_bindings():
    problem = Projection()
    problem.add_parameter(["a", "b"])
    y = problem.add_variable(["u", "v"])
    problem.objective = y.u**2 + y.v**2
    np.testing.assert_array_equal(
        problem._bind_parameters({"b": 2, "a": 1}), [1.0, 2.0]
    )
    np.testing.assert_array_equal(problem._bind_initial_values({"v": 2}), [0.0, 2.0])
    cases = [
        ([{"a": 1, "b": 2}, {"a": 3, "b": 4}], None),
        ({"a": [1, 3], "b": [2, 4]}, None),
        ({"a": 1, "b": 2}, 2),
        (np.array([1, 2]), 2),
        (np.array([[1, 2], [3, 4]]), 2),
    ]
    for values, size in cases:
        actual = problem._bind_parameter_batch(values, batch_size=size)
        assert actual.shape == (2, 2)
    for values in [None, {}, [1.0], {"a": 1, "b": 2, "extra": 3}]:
        with pytest.raises(ValueError):
            problem._bind_parameters(values)
    for values in [{"bad": 1}, [1.0]]:
        with pytest.raises(ValueError):
            problem._bind_initial_values(values)
    for values, size in [
        (None, 1),
        ({}, 1),
        ({"a": [[1]], "b": 1}, 1),
        ({"a": [1, 2], "b": [1, 2, 3]}, None),
        ({"a": [1, 2], "b": [1, 2]}, 3),
        ([{"a": 1, "b": 2}], 2),
        (np.ones((2, 3)), 2),
        (np.ones((2, 2)), 3),
        (np.ones((1, 2, 2)), 2),
        (np.ones(2), 0),
    ]:
        with pytest.raises(ValueError):
            problem._bind_parameter_batch(values, batch_size=size)
    for values in [None, {"u": 1}, np.array([1, 2]), np.ones((2, 2))]:
        assert problem._bind_initial_value_batch(values, 2).shape == (2, 2)
    for values in [
        {"bad": 1},
        {"u": [[1]]},
        np.ones((2, 3)),
        np.ones((3, 2)),
        np.ones((1, 2, 2)),
        np.ones(3),
    ]:
        with pytest.raises(ValueError):
            problem._bind_initial_value_batch(values, 2)


def test_named_value_containers():
    single = NamedValues(["a", "b"], [1, 2])
    assert dict(single) == {"a": 1.0, "b": 2.0}
    assert len(single) == 2 and single.a == 1 and "NamedValues" in repr(single)
    with pytest.raises(AttributeError):
        single.missing
    with pytest.raises(ValueError):
        NamedValues(["a"], [1, 2])
    batch = BatchNamedValues(["a", "b"], [[1, 2], [3, 4]])
    assert list(batch) == ["a", "b"] and len(batch) == 2 and batch.batch_size == 2
    assert "BatchNamedValues" in repr(batch)
    np.testing.assert_array_equal(batch.row(1).vector(), [3, 4])
    np.testing.assert_array_equal(batch.vector(), [[1, 2], [3, 4]])
    copy = batch.matrix()
    copy[0, 0] = 99
    assert batch.a[0] == 1
    assert not batch.matrix(copy=False).flags.writeable
    with pytest.raises(AttributeError):
        batch.missing
    with pytest.raises(ValueError):
        BatchNamedValues(["a"], [[1, 2]])
