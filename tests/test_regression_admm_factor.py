"""Sparse band factors: independent dense references and invalidation inputs."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from purin.solvers.admm.factor import prepare_banded
from purin.solvers.chambolle_pock.kernels import layout


@pytest.mark.parametrize("n,width", [(1, 0), (8, 1), (12, 3), (18, 6)])
def test_sparse_factor_matches_dense(n, width):
    rng = np.random.default_rng(n)
    rows = [tuple(range(i, min(i + width + 1, n))) for i in range(n)]
    support, mask, _ = layout(rows, n)
    plan = prepare_banded(support, mask, n)
    assert plan is not None
    values = rng.normal(size=(3, *support.shape)) * np.asarray(mask)
    diagonal = rng.uniform(0.2, 2.0, size=(3, n))
    rhs = rng.normal(size=(3, n))
    solve = jax.jit(lambda d, v, r, p: plan.solve(plan.factor(d, v, p, 1e-6), r))
    for scale in [1.0, 2.3]:
        penalty = rng.uniform(0.1, 4.0, size=(3, n)) * scale
        actual = solve(
            jnp.asarray(diagonal),
            jnp.asarray(values),
            jnp.asarray(rhs),
            jnp.asarray(penalty),
        )
        for b in range(3):
            a = np.zeros((n, n))
            for i, row in enumerate(rows):
                a[i, list(row)] = values[b, i, : len(row)]
            matrix = np.diag(diagonal[b] + 1e-6) + a.T @ (penalty[b, :, None] * a)
            np.testing.assert_allclose(
                actual[b], np.linalg.solve(matrix, rhs[b]), atol=2e-12
            )


def test_wide_structure_uses_pcg_fallback():
    support, mask, _ = layout([tuple(range(20))], 20)
    assert prepare_banded(support, mask, 20) is None
