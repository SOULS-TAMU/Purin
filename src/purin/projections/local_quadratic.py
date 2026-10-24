"""NLPOPTNet-style local objective approximation and constraint linearization."""

from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from purin.compilation import compile_problem
from purin.solvers.diagonal_qp import QPStructure, QPData
from purin.solvers.chambolle_pock.kernels import layout
from purin.solvers.types import ProjectionInfo


@dataclass(frozen=True)
class LocalQuadratic:
    """Local diagonal objective approximation with linearized nonlinear constraints."""

    problem: object
    rho: float = 1.0
    diagonal_floor: float = 1e-8
    lower: object = None
    upper: object = None
    config: object = None

    def __post_init__(self):
        if (
            not math.isfinite(self.rho)
            or self.rho <= 0
            or not math.isfinite(self.diagonal_floor)
            or self.diagonal_floor <= 0
        ):
            raise ValueError("rho and diagonal_floor must be positive and finite")

    def compile(self):
        """Compile objective derivatives, constraint structure and approximation resources."""
        return CompiledLocalQuadratic(self, compile_problem(self.problem))

    def prepare(self, *, batch_size, dtype=jnp.float64, backend="cpu"):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        from purin.execution.runtime import resolve_device

        device = resolve_device(backend)
        compiled = self.compile()
        solver = compiled.structure.prepare(
            batch_size=batch_size, config=self.config, dtype=dtype
        )
        return PreparedLocalQuadratic(compiled, solver, device)


@dataclass(frozen=True, eq=False)
class CompiledLocalQuadratic:
    """Compiled derivative resources and fixed QP structure for local approximation."""

    definition: LocalQuadratic
    canonical: object

    def __post_init__(self):
        c = self.canonical
        eq = (
            c.equality_sparsity
            if c.equality_sparsity is not None
            else tuple(tuple(range(c.variable_size)) for _ in range(c.equality_size))
        )
        iq = (
            c.inequality_sparsity
            if c.inequality_sparsity is not None
            else tuple(tuple(range(c.variable_size)) for _ in range(c.inequality_size))
        )
        structure = QPStructure(c.variable_size, eq, iq)
        support, mask, _ = layout(structure.rows, c.variable_size)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "_gradient", jax.grad(c.objective, argnums=1))

    def one(self, context, anchor):
        """Build one diagonal QP around the supplied context and primal anchor."""
        c, definition = self.canonical, self.definition
        gradient = self._gradient(context, anchor)

        # Compute diagonal Hessian entries without storing a dense Hessian.
        def diagonal_entry(i):
            direction = jax.nn.one_hot(i, c.variable_size, dtype=anchor.dtype)
            return jax.jvp(
                lambda y: self._gradient(context, y), (anchor,), (direction,)
            )[1][i]

        diagonal = jax.lax.map(diagonal_entry, jnp.arange(c.variable_size))
        q = definition.rho * jnp.maximum(diagonal, definition.diagonal_floor)
        linear = gradient - q * anchor
        blocks = []
        for row_offset, count, value_function, jacobian in (
            (0, c.equality_size, c.equality_jacobian_values, c.equality_jacobian),
            (
                c.equality_size,
                c.inequality_size,
                c.inequality_jacobian_values,
                c.inequality_jacobian,
            ),
        ):
            support = self.support[row_offset : row_offset + count]
            mask = self.mask[row_offset : row_offset + count]
            if value_function is None:
                matrix = jacobian(context, anchor).reshape((count, c.variable_size))
                block = jnp.where(mask, matrix[jnp.arange(count)[:, None], support], 0)
            else:
                # Canonical sparse values follow the declared row/support order.
                rows = self.structure.rows[row_offset : row_offset + count]
                positions = [
                    i * self.support.shape[1] + j
                    for i, row in enumerate(rows)
                    for j in range(len(row))
                ]
                block = (
                    jnp.zeros(count * self.support.shape[1], anchor.dtype)
                    .at[jnp.asarray(positions, dtype=jnp.int32)]
                    .set(value_function(context, anchor))
                    .reshape(support.shape)
                )
            blocks.append(block)
        values = jnp.concatenate(blocks, axis=0)
        residuals = jnp.concatenate(
            (c.equalities(context, anchor), c.inequalities(context, anchor))
        )
        rhs = jnp.sum(values * anchor[self.support] * self.mask, axis=1) - residuals

        def bound(value, default):
            return jnp.broadcast_to(
                jnp.asarray(
                    default
                    if value is None
                    else value(context)
                    if callable(value)
                    else value,
                    dtype=anchor.dtype,
                ),
                anchor.shape,
            )

        return QPData(
            q,
            linear,
            values,
            rhs,
            bound(definition.lower, -jnp.inf),
            bound(definition.upper, jnp.inf),
        )

    def build(self, context, anchor):
        """Build batched local QP coefficients around the supplied primal anchors."""
        if (
            context.ndim != 2
            or anchor.ndim != 2
            or context.shape[0] != anchor.shape[0]
            or context.shape[1] != self.canonical.parameter_size
            or anchor.shape[1] != self.canonical.variable_size
        ):
            raise ValueError("context/anchor shapes do not match the compiled problem")
        return jax.vmap(self.one)(context, anchor)


@dataclass(frozen=True)
class PreparedLocalQuadratic:
    """Prepared local-QP builder coupled to its reusable projection solver."""

    builder: CompiledLocalQuadratic
    solver: object
    device: object

    def solve(self, anchor, context, *, initial_dual=None):
        """Build and solve the local QPs for the supplied contexts and anchors."""
        anchor = jax.device_put(
            jnp.asarray(anchor, dtype=self.solver.dtype), self.device
        )
        context = jax.device_put(
            jnp.asarray(context, dtype=self.solver.dtype), self.device
        )
        data = self.builder.build(context, anchor)
        return self.solver.solve(data, initial_y=anchor, initial_dual=initial_dual)

    def __call__(self, anchor, context):
        return self.solve(anchor, context).y

    def with_info(self, anchor, context):
        """Evaluate the projection and return its primal values with solver diagnostics."""
        r = self.solve(anchor, context)
        return r.y, ProjectionInfo(
            r.status, r.iterations, r.residual_inf_norm, r.finite, r.gradient_eligible
        )

    def project_training(
        self, anchor, context, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        from purin.solvers.types import ProjectionTrainingOutput

        b = self.solver.batch_size
        me = len(self.builder.structure.equality_rows)
        mi = len(self.builder.structure.inequality_rows)
        eq = (
            jnp.zeros((b, me), self.solver.dtype)
            if multipliers_eq is None
            else jnp.asarray(multipliers_eq, self.solver.dtype)
        )
        iq = (
            jnp.zeros((b, mi), self.solver.dtype)
            if multipliers_ineq is None
            else jnp.asarray(multipliers_ineq, self.solver.dtype)
        )
        r = self.solve(anchor, context, initial_dual=jnp.concatenate((eq, iq), axis=1))
        return ProjectionTrainingOutput(
            r.y,
            r.multipliers_eq,
            r.multipliers_ineq,
            r.slacks,
            r.status,
            r.iterations,
            r.residual_inf_norm,
            r.finite,
            r.gradient_eligible,
        )
