"""Periodic diagnostics preserve strict tolerances and always check the final step."""

from dataclasses import replace
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from purin import Projection, prepare_projection, ADMMConfig
from purin import SolveStatus
from purin.solvers.admm.kernels import sparse_admm
from purin.solvers.chambolle_pock.kernels import layout


def make_problem():
    p = Projection()
    y = p.add_variable(["u", "v"])
    p.objective = 0.0
    p.constraints.add(y.u + y.v == 1, y.u >= 0, y.u <= 1)
    return p


@pytest.mark.parametrize("adaptive", [False, True])
def test_periodic_acceptance_and_adaptive_rho(adaptive):
    target = jnp.array([[3.0, -1.0], [-2.0, 2.0], [0.2, 0.4]])
    cfg = ADMMConfig(adaptive_rho=adaptive, rho_update_interval=7)
    outputs = []
    for interval in [1, 25]:
        layer = prepare_projection(
            make_problem(),
            method="admm",
            config=replace(cfg, check_interval=interval),
            batch_size=3,
        )
        out = jax.jit(layer.solve)(target)
        assert np.all(out.gradient_eligible)
        assert np.max(out.residual_inf_norm) <= cfg.tolerance
        outputs.append(out.y)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=3e-9)
    np.testing.assert_allclose(
        outputs[1], [[1.0, 0.0], [0.0, 1.0], [0.4, 0.6]], atol=3e-9
    )


def test_final_step_is_checked_between_intervals():
    cfg = ADMMConfig(tolerance=1e-14, max_iterations=17, check_interval=25)
    layer = prepare_projection(make_problem(), method="admm", config=cfg, batch_size=1)
    out = layer.solve(jnp.array([[3.0, -1.0]]))
    assert int(out.iterations[0]) == 17
    assert np.isfinite(out.residual_inf_norm[0])
    assert int(out.status[0]) == int(SolveStatus.MAX_ITERATIONS)
    assert not bool(out.gradient_eligible[0])


def test_periodic_pcg_diagonal_q_matches_osqp():
    import osqp
    import scipy.sparse as sp

    rng = np.random.default_rng(53)
    n, m = 12, 16
    a = rng.normal(size=(m, n))
    q = rng.uniform(0.5, 2.0, size=n)
    c = rng.normal(size=n)
    lo = -rng.uniform(0.2, 1.0, size=m)
    hi = rng.uniform(0.2, 1.0, size=m)
    support, mask, _ = layout([tuple(range(n))] * m, n)
    reference = osqp.OSQP()
    reference.setup(
        P=sp.diags(q, format="csc"),
        q=c,
        A=sp.csc_matrix(a),
        l=lo,
        u=hi,
        eps_abs=1e-11,
        eps_rel=1e-11,
        polishing=True,
        verbose=False,
    )
    expected = reference.solve().x
    for interval in [1, 25]:
        out = jax.jit(
            lambda: sparse_admm(
                jnp.asarray(q[None]),
                jnp.asarray(c[None]),
                jnp.asarray(a[None]),
                jnp.asarray(lo[None]),
                jnp.asarray(hi[None]),
                support,
                mask,
                check_interval=interval,
            )
        )()
        assert bool(out.converged[0])
        assert float(out.linear_residual[0]) <= 1e-9
        assert float(out.primal_residual[0]) <= 1e-9
        assert float(out.dual_residual[0]) <= 1e-9
        np.testing.assert_allclose(out.x[0], expected, atol=2e-8)


@pytest.mark.parametrize("interval", [0, -1, 1.5])
def test_invalid_check_interval(interval):
    with pytest.raises(ValueError, match="check_interval"):
        ADMMConfig(check_interval=interval)
