"""General positive diagonal QP layer on the extracted sparse CP kernels.

Unlike the historical distance wrapper, all q/c/constraint/bound coefficients
are differentiable. The proximal inverse is recomputed inside the VJP map,
so freezing step sizes does not discard the derivative with respect to q.
"""

from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .chambolle_pock.kernels import (
    sparse_cp,
    cp_step,
    operators,
    norm_estimate,
    diagonal_step_sizes,
    layout,
)
from .chambolle_pock.config import ChambollePockConfig
from .primal_dual_interior_point.operators import _batched_restarted_gmres
from .types import ProjectionResult, ProjectionInfo as ProjectionInfo, SolveStatus


class QPData(NamedTuple):
    """Batched diagonal objective, constraint coefficients and variable bounds."""

    q: object
    c: object
    values: object
    rhs: object
    lower: object
    upper: object


@dataclass(frozen=True)
class QPStructure:
    """Variable dimension and fixed row supports for a family of diagonal QPs."""

    variable_size: int
    equality_rows: tuple = ()
    inequality_rows: tuple = ()

    def __post_init__(self):
        if self.variable_size < 1:
            raise ValueError("variable_size must be positive")
        for key in ("equality_rows", "inequality_rows"):
            rows = tuple(tuple(int(i) for i in row) for row in getattr(self, key))
            if any(
                len(set(row)) != len(row)
                or any(i < 0 or i >= self.variable_size for i in row)
                for row in rows
            ):
                raise ValueError(
                    "constraint supports must contain unique valid indices"
                )
            object.__setattr__(self, key, rows)

    @property
    def rows(self):
        """Return equality rows followed by inequality rows in solver order."""
        return self.equality_rows + self.inequality_rows

    def prepare(self, *, batch_size, config=None, dtype=jnp.float64):
        """Prepare a solver for this fixed sparse structure and batch shape."""
        return PreparedDiagonalQP(
            self, int(batch_size), config or ChambollePockConfig(), dtype
        )


@dataclass(frozen=True, eq=False)
class PreparedDiagonalQP:
    """Prepared diagonal-QP solve using shared sparse Chambolle--Pock kernels."""

    structure: QPStructure
    batch_size: int
    config: ChambollePockConfig
    dtype: object
    differentiate_duals: bool = True

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.config.metric_diagonal is not None:
            raise ValueError("QPData.q already defines the quadratic metric")
        support, mask, _ = layout(self.structure.rows, self.structure.variable_size)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "_run", jax.jit(self._make_solve()))

    def _make_solve(self):
        """Build the reusable diagonal-QP solve and its custom implicit derivative."""
        cfg = self.config
        me = len(self.structure.equality_rows)
        n = self.structure.variable_size
        bsize = self.batch_size
        support, mask = self.support, self.mask

        def forward(data, y0, dual0, solve_tolerance):
            out = sparse_cp(
                *data[:4],
                support,
                mask,
                me,
                data.lower,
                data.upper,
                accelerated=cfg.accelerated,
                safety=cfg.safety,
                max_iterations=cfg.max_iterations,
                tolerance=solve_tolerance,
                norm_iterations=cfg.norm_iterations,
                norm_seed=cfg.norm_seed,
                y0=y0,
                dual0=dual0,
                freeze_converged=True,
                check_stationarity=True,
                norm_bound=cfg.norm_bound,
                diagonal_preconditioning=cfg.diagonal_preconditioning,
            )
            valid = jnp.all(data.q > 0, axis=1) & jnp.all(
                data.lower <= data.upper, axis=1
            )
            return out._replace(converged=out.converged & valid)

        @jax.custom_vjp
        def solve(data, y0, dual0, solve_tolerance):
            return forward(data, y0, dual0, solve_tolerance)

        def fwd(data, y0, dual0, solve_tolerance):
            out = forward(data, y0, dual0, solve_tolerance)
            return out, (out, data)

        def bwd(saved, cot):
            out, data = saved
            mv, tmv = operators(data.values, support, mask, n)
            if cfg.diagonal_preconditioning and not cfg.accelerated:
                tau, sigma = diagonal_step_sizes(
                    data.q, data.values, support, mask, cfg.safety
                )
            else:
                if cfg.norm_bound:
                    amv, atmv = operators(abs(data.values), support, mask, n)
                    L = jnp.sqrt(
                        jnp.max(amv(jnp.ones_like(data.q)), axis=1, initial=0.0)
                        * jnp.max(atmv(jnp.ones_like(data.rhs)), axis=1, initial=0.0)
                    )
                else:
                    L = norm_estimate(
                        mv,
                        tmv,
                        bsize,
                        n,
                        data.q.dtype,
                        cfg.norm_iterations,
                        cfg.norm_seed,
                    )
                L = jnp.maximum(L, 1e-12)
                tau = jax.lax.stop_gradient(cfg.safety / (jnp.max(data.q, axis=1) + L))
                sigma = jax.lax.stop_gradient(cfg.safety / L)
            m = data.rhs.shape[1]

            def pack(z):
                return jnp.concatenate(z, axis=1)

            def unpack(z):
                return z[:, :n], z[:, n : n + m], z[:, n + m :]

            def step(z, d):
                apply, transpose = operators(d.values, support, mask, n)
                return pack(
                    cp_step(
                        unpack(z),
                        d.q,
                        d.c,
                        d.rhs,
                        d.lower,
                        d.upper,
                        apply,
                        transpose,
                        me,
                        tau,
                        sigma,
                        jnp.ones(bsize, d.q.dtype),
                    )
                )

            state = pack((out.y, out.dual, out.y))
            _, pullback = jax.vjp(step, state, data)
            operator = lambda v: v - pullback(v)[0]
            dual_cot = (
                cot.dual if self.differentiate_duals else jnp.zeros_like(out.dual)
            )
            g = pack((cot.y, dual_cot, jnp.zeros_like(out.y)))
            g = jnp.where(out.converged[:, None], g, 0)
            linear = cfg.linear_solver
            adj, code, residual = _batched_restarted_gmres(
                operator,
                g,
                lambda v: v,
                tolerance=linear.tolerance,
                absolute_tolerance=linear.atol,
                restart=linear.restart,
                max_iterations=linear.max_iterations,
            )
            valid = (
                out.converged
                & jnp.all(jnp.isfinite(adj), axis=1)
                & jnp.isfinite(residual)
            )
            valid &= (code == 0) | (residual <= cfg.implicit_gradient.adjoint_tolerance)
            _, gradients = pullback(jnp.where(valid[:, None], adj, 0))
            gradients = jax.tree_util.tree_map(
                lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), gradients
            )
            return (
                gradients,
                jnp.zeros_like(out.y),
                jnp.zeros_like(out.dual),
                jnp.zeros((bsize,), out.y.dtype),
            )

        solve.defvjp(fwd, bwd)
        return solve

    def solve(self, data, *, initial_y=None, initial_dual=None):
        """Solve the batched diagonal QP with the supplied coefficients and warm start."""
        data = QPData(*(jnp.asarray(x, dtype=self.dtype) for x in data))
        b, n = self.batch_size, self.structure.variable_size
        m, width = self.support.shape
        expected = ((b, n), (b, n), (b, m, width), (b, m), (b, n), (b, n))
        for key, value, shape in zip(QPData._fields, data, expected):
            if value.shape != shape:
                raise ValueError(f"{key} must have shape {shape}; got {value.shape}")
        y0 = (
            jnp.zeros((b, n), self.dtype)
            if initial_y is None
            else jnp.asarray(initial_y, self.dtype)
        )
        dual0 = (
            jnp.zeros((b, m), self.dtype)
            if initial_dual is None
            else jnp.asarray(initial_dual, self.dtype)
        )
        if y0.shape != (b, n) or dual0.shape != (b, m):
            raise ValueError("initial primal/dual shapes do not match the prepared QP")
        scale = jnp.ones_like(data.rhs)
        if self.config.row_scaling:
            norms = jnp.sqrt(
                jnp.sum(jnp.where(self.mask[None, :, :], data.values**2, 0), axis=2)
            )
            scale = jax.lax.stop_gradient(
                jnp.where(norms > 0, 1 / jnp.maximum(norms, 1e-30), 1.0)
            )
        scaled = data._replace(
            values=data.values * scale[:, :, None], rhs=data.rhs * scale
        )
        tolerance = self.config.tolerance * jnp.minimum(
            1.0, jnp.min(scale, axis=1, initial=1.0)
        )
        out = self._run(
            scaled,
            jax.lax.stop_gradient(y0),
            jax.lax.stop_gradient(dual0 / scale),
            jax.lax.stop_gradient(tolerance),
        )
        out = out._replace(dual=out.dual * scale)
        apply, transpose = operators(data.values, self.support, self.mask, n)
        residuals = apply(out.y) - data.rhs
        me = len(self.structure.equality_rows)
        eq = jnp.max(abs(residuals[:, :me]), axis=1, initial=0.0)
        iq = jnp.max(jnp.maximum(residuals[:, me:], 0), axis=1, initial=0.0)
        stationarity = jnp.max(
            abs(
                out.y
                - jnp.clip(
                    out.y - (data.q * out.y + data.c + transpose(out.dual)),
                    data.lower,
                    data.upper,
                )
            ),
            axis=1,
        )
        original_residual = jnp.maximum(jnp.maximum(eq, iq), stationarity)
        out = out._replace(
            residual=original_residual,
            converged=out.converged & (original_residual <= self.config.tolerance),
        )
        finite = (
            jnp.all(jnp.isfinite(out.y), axis=1)
            & jnp.all(jnp.isfinite(out.dual), axis=1)
            & jnp.isfinite(out.residual)
        )
        status = jnp.where(
            out.converged,
            int(SolveStatus.CONVERGED),
            jnp.where(
                finite, int(SolveStatus.MAX_ITERATIONS), int(SolveStatus.NONFINITE)
            ),
        )
        me = len(self.structure.equality_rows)
        mv, _ = operators(data.values, self.support, self.mask, n)
        slacks = jnp.maximum(data.rhs[:, me:] - mv(out.y)[:, me:], 0)
        stop = jax.lax.stop_gradient
        return ProjectionResult(
            out.y,
            out.dual[:, :me],
            out.dual[:, me:],
            slacks,
            stop(status),
            stop(out.iterations),
            stop(out.residual),
            stop(finite),
            stop(out.converged),
        )


def dense_qp_data(q, c, A, b, C, d, lower, upper):
    """Convenience adapter; production callers may provide sparse QPData directly."""
    q, c = jnp.asarray(q), jnp.asarray(c)
    if q.ndim != 2 or q.shape != c.shape:
        raise ValueError("q and c must be matching batched matrices")
    batch, n = q.shape
    A, C = jnp.asarray(A), jnp.asarray(C)
    A = jnp.broadcast_to(A, (batch,) + A.shape[-2:])
    C = jnp.broadcast_to(C, (batch,) + C.shape[-2:])
    rows = tuple(tuple(range(n)) for _ in range(A.shape[1]))
    iq = tuple(tuple(range(n)) for _ in range(C.shape[1]))
    structure = QPStructure(n, rows, iq)
    support, mask, _ = layout(structure.rows, n)
    matrix = jnp.concatenate((A, C), axis=1)
    values = jnp.where(
        mask[None, :, :],
        matrix[:, jnp.arange(len(structure.rows))[:, None], support],
        0,
    )
    return structure, QPData(
        q,
        c,
        values,
        jnp.concatenate(
            (
                jnp.broadcast_to(b, (batch, A.shape[1])),
                jnp.broadcast_to(d, (batch, C.shape[1])),
            ),
            axis=1,
        ),
        jnp.broadcast_to(lower, q.shape),
        jnp.broadcast_to(upper, q.shape),
    )
