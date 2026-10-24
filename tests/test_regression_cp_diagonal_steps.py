"""Diagonal CP metrics: stability bound, sparse QP values, and gradients."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from purin import Projection, prepare_projection, ChambollePockConfig
from purin.solvers.chambolle_pock.kernels import diagonal_step_sizes, layout, sparse_cp


def test_metric_bound_with_heterogeneous_rows_and_zero_columns():
    matrix = np.array(
        [[100.0, 0.0, 0.2, 0.0], [0.0, 0.01, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    support, mask, _ = layout([tuple(np.flatnonzero(r)) for r in matrix], 4)
    values = jnp.asarray(np.where(mask, matrix[np.arange(3)[:, None], support], 0.0))
    tau, sigma = diagonal_step_sizes(jnp.ones((2, 4)), values, support, mask, 0.95)
    assert np.all(np.isfinite(tau)) and np.all(tau > 0)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)
    for t, s in zip(np.asarray(tau), np.asarray(sigma)):
        norm = np.linalg.norm(np.sqrt(s)[:, None] * matrix * np.sqrt(t)[None, :], 2)
        assert norm <= 0.95 + 1e-14


@pytest.mark.parametrize("use_diagonal", [False, True])
def test_general_diagonal_q_with_coupled_inequalities_matches_osqp(use_diagonal):
    import osqp
    import scipy.sparse as sp

    rng = np.random.default_rng(123)
    n, me, mi, batch = 6, 2, 8, 3
    matrix = rng.normal(size=(me + mi, n))
    matrix[abs(matrix) < 0.6] = 0
    support, mask, _ = layout([tuple(np.flatnonzero(r)) for r in matrix], n)
    data = jnp.asarray(
        np.where(mask, matrix[np.arange(me + mi)[:, None], support], 0.0)
    )
    q = jnp.asarray(rng.uniform(0.5, 3.0, (batch, n)))
    c = jnp.asarray(rng.normal(size=(batch, n)))
    feasible = rng.uniform(-0.3, 0.3, (batch, n))
    rhs = feasible @ matrix.T
    rhs[:, me:] += rng.uniform(0.3, 1.0, (batch, mi))
    out = jax.jit(
        lambda: sparse_cp(
            q,
            c,
            data,
            jnp.asarray(rhs),
            support,
            mask,
            me,
            -jnp.ones_like(q),
            jnp.ones_like(q),
            tolerance=1e-9,
            max_iterations=20000,
            freeze_converged=True,
            check_stationarity=True,
            norm_bound=True,
            diagonal_preconditioning=use_diagonal,
        )
    )()
    assert np.all(out.converged)
    for i in range(batch):
        qp = osqp.OSQP()
        qp.setup(
            P=sp.diags(np.asarray(q[i]), format="csc"),
            q=np.asarray(c[i]),
            A=sp.csc_matrix(np.r_[matrix, np.eye(n)]),
            l=np.r_[rhs[i, :me], np.full(mi, -np.inf), -np.ones(n)],
            u=np.r_[rhs[i], np.ones(n)],
            eps_abs=1e-11,
            eps_rel=1e-11,
            polishing=True,
            verbose=False,
        )
        ref = qp.solve()
        assert ref.info.status_val == 1
        np.testing.assert_allclose(out.y[i], ref.x, atol=3e-8)


def test_scalar_opt_out_and_no_bound_parameter_gradient():
    p = Projection()
    t = p.add_parameter(["a"])
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(t.a * y.u + y.v == 1)
    from purin import ImplicitGradientConfig

    for diagonal in (False, True):
        cfg = ChambollePockConfig(
            diagonal_preconditioning=diagonal,
            implicit_gradient=ImplicitGradientConfig(
                differentiate_problem_parameters=True
            ),
        )
        layer = prepare_projection(p, method="chambolle-pock", config=cfg, batch_size=1)
        x = jnp.array([[0.2, 0.3]])
        theta = jnp.array([[2.0]])
        expected = lambda a: (
            x
            - jnp.array([[a[0, 0], 1.0]])
            * (a[0, 0] * 0.2 + 0.3 - 1)
            / (1 + a[0, 0] ** 2)
        )
        np.testing.assert_allclose(layer(x, theta), expected(theta), atol=1e-8)
        np.testing.assert_allclose(
            jax.jacrev(lambda a: layer(x, a))(theta),
            jax.jacrev(expected)(theta),
            atol=2e-7,
        )
