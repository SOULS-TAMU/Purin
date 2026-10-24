"""Dense CP boundary using shared iteration and fixed-point differentiation.

The frozen proximal policy preserves the historical NLPOPTNet derivative.
It intentionally omits derivatives through the quadratic proximal inverse.
"""

import jax
import jax.numpy as jnp
from purin.autodiff.fixed_point import fixed_point_pullback
from .dense import dense_operators, solve_dense
from .kernels import cp_step, norm_estimate


def _forward(accelerated):
    def solve(
        q,
        c,
        a,
        b,
        matrix,
        d,
        lower,
        upper,
        *,
        safety=0.95,
        knorm_iters=20,
        knorm_seed=42,
        max_iter=5000,
        tol=1e-9,
        y0=None,
        lam0=None,
        mu0=None,
    ):
        return solve_dense(
            q,
            c,
            a,
            b,
            matrix,
            d,
            lower,
            upper,
            accelerated=accelerated,
            safety=safety,
            norm_iterations=knorm_iters,
            norm_seed=knorm_seed,
            max_iterations=max_iter,
            tolerance=tol,
            y0=y0,
            lambda0=lam0,
            mu0=mu0,
        )

    return solve


def _cp_step_fixed(z, params, steps):
    y, equality, inequality, extrapolated = z
    q, c, a, b, matrix, d, lower, upper = params
    tau, sigma, theta, inverse = steps
    mv, tmv = dense_operators(a, matrix)
    next_y, dual, next_bar = cp_step(
        (y, jnp.concatenate((equality, inequality), axis=1), extrapolated),
        q,
        c,
        jnp.concatenate((b, d), axis=1),
        lower,
        upper,
        mv,
        tmv,
        a.shape[1],
        tau,
        sigma,
        theta,
        proximal=(inverse, None, None),
        factored_proximal=True,
    )
    return next_y, dual[:, : a.shape[1]], dual[:, a.shape[1] :], next_bar


def _steps_fixed(Q_diag, A, C, *, safety, knorm_iters, knorm_seed):
    dtype = jnp.result_type(Q_diag.dtype, A.dtype, C.dtype)
    q, a, c = (jnp.asarray(value, dtype=dtype) for value in (Q_diag, A, C))
    mv, tmv = dense_operators(a, c)
    norm = norm_estimate(
        mv, tmv, q.shape[0], q.shape[1], dtype, knorm_iters, knorm_seed
    )
    lipschitz = jnp.max(q, axis=1) if q.shape[1] else jnp.zeros(q.shape[0], dtype)
    bound = jnp.maximum(norm, 1e-12)
    sigma = jnp.asarray(safety, dtype) / bound
    tau = jnp.asarray(safety, dtype) / (lipschitz + bound)
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        (tau, sigma, jnp.ones(q.shape[0], dtype), 1.0 / (1.0 + tau[:, None] * q)),
    )


_steps_for_accel_backward = _steps_fixed


def _implicit(forward):
    @jax.custom_vjp
    def solve(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        safety=0.95,
        knorm_iters=20,
        knorm_seed=42,
        max_iter=5000,
        tol=1e-9,
        y0=None,
        lam0=None,
        mu0=None,
        adjoint_iters: int = 50,
    ):
        if y0 is None or lam0 is None or mu0 is None:
            raise ValueError(
                "Pass warm-start arrays y0/lam0/mu0 (no None). "
                "Use opt.projection.solve_cp_implicit to canonicalize them."
            )
        return forward(
            Q_diag,
            c,
            A,
            b,
            C,
            d,
            l,
            u,
            safety=safety,
            knorm_iters=knorm_iters,
            knorm_seed=knorm_seed,
            max_iter=max_iter,
            tol=tol,
            y0=y0,
            lam0=lam0,
            mu0=mu0,
        )

    def fwd(
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        safety,
        knorm_iters,
        knorm_seed,
        max_iter,
        tol,
        y0,
        lam0,
        mu0,
        adjoint_iters,
    ):
        result = forward(
            Q_diag,
            c,
            A,
            b,
            C,
            d,
            l,
            u,
            safety=safety,
            knorm_iters=knorm_iters,
            knorm_seed=knorm_seed,
            max_iter=max_iter,
            tol=tol,
            y0=y0,
            lam0=lam0,
            mu0=mu0,
        )
        y, equality, inequality = result
        steps = _steps_fixed(
            Q_diag, A, C, safety=safety, knorm_iters=knorm_iters, knorm_seed=knorm_seed
        )
        frozen = jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            (
                (y, equality, inequality, y),
                tuple(map(jnp.asarray, (Q_diag, c, A, b, C, d, l, u))),
                steps,
            ),
        )
        return result, (*frozen, adjoint_iters)

    def bwd(residual, cotangent):
        state, params, steps, iterations = residual
        cotangent = (*cotangent, jnp.zeros_like(cotangent[0]))
        gradients = fixed_point_pullback(
            lambda z, p: _cp_step_fixed(z, p, steps),
            state,
            params,
            cotangent,
            max_iterations=iterations,
        )
        return (*gradients, *((None,) * 9))

    solve.defvjp(fwd, bwd)
    return solve, fwd, bwd


CP_fixed_implicit, _CP_fixed_implicit_fwd, _CP_fixed_implicit_bwd = _implicit(
    _forward(False)
)
CP_accelerated_implicit, _CP_accelerated_implicit_fwd, _CP_accelerated_implicit_bwd = (
    _implicit(_forward(True))
)
_STATIC = ("safety", "knorm_iters", "knorm_seed", "max_iter", "tol", "adjoint_iters")
CP_fixed_implicit_jit = jax.jit(CP_fixed_implicit, static_argnames=_STATIC)
CP_accelerated_implicit_jit = jax.jit(CP_accelerated_implicit, static_argnames=_STATIC)
