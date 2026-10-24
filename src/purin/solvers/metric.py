"""Optional constant positive diagonal projection metric.

z=sqrt(D)*y converts 0.5*(y-target).T*D*(y-target) exactly to the
existing identity projection. Constraint supports and dual/slack units are
unchanged. Chain-rule differentiation gives the weighted KKT derivative.
"""

from dataclasses import replace
import numpy as np


from purin._utils.validation import validate_metric_diagonal


def _scaled_problem(problem, inverse):
    """Transform the problem to Euclidean coordinates using the diagonal metric."""
    import jax.numpy as jnp
    from purin.problems.problem import Problem
    from purin.problems.symbolic import SymbolicProblem
    from purin.problems.expressions import Constraint
    from purin.problems.expressions import Expression
    from purin.problems.expressions import _AffineScalar

    source = (
        problem
        if isinstance(problem, Problem)
        else (
            problem.to_symbolic_problem()
            if hasattr(problem, "to_symbolic_problem")
            else None
        )
    )
    if source is None:
        raise TypeError(
            "weighted projection requires a symbolic problem or a frontend providing to_symbolic_problem()"
        )
    # CoreProblem target parameters are preparation details, not physical inputs.
    target_names = set(getattr(source, "_projection_target_names", None) or ())
    if any(
        c.residual.parameter_dependencies & target_names for c in source.constraints
    ):
        raise ValueError(
            "weighted projection constraints must not depend on projection-target parameters"
        )
    n = len(source.variable_names)
    if len(inverse) != n:
        raise ValueError(f"metric_diagonal must have {n} entries; got {len(inverse)}")
    result = SymbolicProblem(source.name + "_diagonal_metric")
    physical = [name for name in source.parameter_names if name not in target_names]
    if physical:
        result.add_parameter(physical)
    result.add_variable(source.variable_names)
    result.objective = 0.0
    inv = jnp.asarray(inverse)
    indices = {name: i for i, name in enumerate(source.variable_names)}
    for constraint in source.constraints:
        residual = constraint.residual
        coefficients = None
        if residual.affine_coefficients is not None:
            coefficients = {
                name: _AffineScalar(
                    lambda ctx, c=coefficient, scale=float(inverse[indices[name]]): (
                        c.evaluate(ctx) * scale
                    ),
                    coefficient.parameter_dependencies,
                )
                for name, coefficient in residual.affine_coefficients.items()
            }
        expression = Expression(
            lambda ctx, r=residual: r.evaluate(
                replace(ctx, variables=ctx.variables * inv.astype(ctx.variables.dtype))
            ),
            residual.text,
            variable_dependencies=residual.variable_dependencies,
            parameter_dependencies=residual.parameter_dependencies,
            affine_in_variables=residual.affine_in_variables,
            affine_coefficients=coefficients,
            affine_offset=residual.affine_offset,
        )
        result.constraints.add(Constraint(expression, constraint.kind, constraint.text))
    override = getattr(source, "_sparsity_override", None)
    if override is not None:
        result.set_sparsity_override(*override)
    return result


class PreparedDiagonalMetricProjection:
    """Same projection methods/results, expressed in original coordinates."""

    def __init__(self, base, diagonal, config):
        import jax.numpy as jnp

        self.base = base
        self.config = config
        self.metric_diagonal = tuple(diagonal)
        self._sqrt = jnp.sqrt(jnp.asarray(diagonal, dtype=base.dtype))
        self._residual_factor = max(1.0, float(np.sqrt(np.max(diagonal))))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, y_hat, parameters=None):
        import jax.numpy as jnp

        return (
            self.base(jnp.asarray(y_hat, dtype=self.dtype) * self._sqrt, parameters)
            / self._sqrt
        )

    def solve(self, y_hat, parameters=None, initial_state=None):
        """Solve the prepared projection with the supplied targets and problem context."""
        import jax.numpy as jnp

        initial = (
            None if initial_state is None else jnp.asarray(initial_state) * self._sqrt
        )
        out = self.base.solve(
            jnp.asarray(y_hat, dtype=self.dtype) * self._sqrt,
            parameters,
            initial_state=initial,
        )
        return out._replace(
            y=out.y / self._sqrt,
            residual_inf_norm=out.residual_inf_norm * self._residual_factor,
        )

    def project_training(
        self, y_hat, parameters=None, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        import jax.numpy as jnp

        out = self.base.project_training(
            jnp.asarray(y_hat, dtype=self.dtype) * self._sqrt,
            parameters,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
        )
        return out._replace(
            y_tilde=out.y_tilde / self._sqrt,
            forward_residual_inf_norm=out.forward_residual_inf_norm
            * self._residual_factor,
        )

    def with_info(self, y_hat, parameters=None):
        """Evaluate the projection and return its primal values with solver diagnostics."""
        import jax.numpy as jnp

        y, info = self.base.with_info(
            jnp.asarray(y_hat, dtype=self.dtype) * self._sqrt, parameters
        )
        return y / self._sqrt, info._replace(
            forward_residual_inf_norm=info.forward_residual_inf_norm
            * self._residual_factor
        )


def prepare_diagonal_metric(problem, config, definition, *, batch_size, dtype, backend):
    """Prepare once; identity defaults never enter this function."""
    diagonal = validate_metric_diagonal(config.metric_diagonal)
    scaled = _scaled_problem(problem, 1 / np.sqrt(np.asarray(diagonal)))
    factor = max(1.0, float(np.sqrt(max(diagonal))))
    base_config = replace(
        config, metric_diagonal=None, tolerance=config.tolerance / factor
    )
    if hasattr(config, "implicit_gradient"):
        base_config = replace(
            base_config,
            implicit_gradient=replace(
                config.implicit_gradient,
                inexact_tolerance=config.implicit_gradient.inexact_tolerance / factor,
            ),
        )
    # Explicit all-ones metrics also use the original problem and arithmetic.
    if all(v == 1.0 for v in diagonal):
        return definition(problem, replace(config, metric_diagonal=None)).prepare(
            batch_size=batch_size, dtype=dtype, backend=backend
        )
    base = definition(scaled, base_config).prepare(
        batch_size=batch_size, dtype=dtype, backend=backend
    )
    return PreparedDiagonalMetricProjection(base, diagonal, config)
