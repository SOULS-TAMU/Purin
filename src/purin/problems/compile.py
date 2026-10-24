"""Backend-neutral compilation of symbolic problems.

Automatic-differentiation transformations are created once here, outside all
backend nonlinear iterations. The canonical sign convention is ``h=0`` and
``g<=0``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from purin.problems.canonical import CanonicalProblem

if TYPE_CHECKING:
    from purin.problems.problem import Problem


def compile_symbolic_problem(problem: "Problem") -> CanonicalProblem:
    """Compile one symbolic frontend object into immutable numerical callables."""

    objective_y_theta, equality_y_theta, inequality_y_theta = problem._build_functions()
    equality_count = sum(item.kind == "eq" for item in problem.constraints)
    inequality_count = sum(item.kind == "ineq" for item in problem.constraints)
    variable_count = len(problem.variable_names)
    parameter_count = len(problem.parameter_names)
    structure = problem._build_projection_structure(require_projection_target=False)
    equality_jacobian_values = None
    inequality_jacobian_values = None
    equality_local_derivatives = None
    inequality_local_derivatives = None
    constraint_hessian_vector_product = None
    if structure is not None:
        (equality_values_y_theta, inequality_values_y_theta, constraint_hvp_y_theta) = (
            problem._build_sparse_projection_derivatives(structure)
        )

        def equality_jacobian_values(theta, y):
            return equality_values_y_theta(y, theta)

        def inequality_jacobian_values(theta, y):
            return inequality_values_y_theta(y, theta)

        def constraint_hessian_vector_product(theta, y, lambda_eq, mu_ineq, vector):
            return constraint_hvp_y_theta(y, theta, lambda_eq, mu_ineq, vector)

        (
            equality_local_y_theta,
            inequality_local_y_theta,
            _equality_offsets,
            _inequality_offsets,
            _equality_constant,
            _inequality_constant,
        ) = problem._build_local_projection_derivatives(structure)

        def equality_local_derivatives(theta, y):
            return equality_local_y_theta(y, theta)

        def inequality_local_derivatives(theta, y):
            return inequality_local_y_theta(y, theta)

    def objective(theta, y):
        return objective_y_theta(y, theta)

    def equalities(theta, y):
        return equality_y_theta(y, theta)

    def inequalities(theta, y):
        return inequality_y_theta(y, theta)

    objective_gradient = jax.grad(objective, argnums=1)
    equality_jacobian = jax.jacfwd(equalities, argnums=1)
    inequality_jacobian = jax.jacfwd(inequalities, argnums=1)

    def lagrangian(theta, y, lambda_eq, mu_ineq):
        value = objective(theta, y)
        if equality_count:
            value = value + jnp.vdot(lambda_eq, equalities(theta, y))
        if inequality_count:
            value = value + jnp.vdot(mu_ineq, inequalities(theta, y))
        return value

    lagrangian_hessian = jax.hessian(lagrangian, argnums=1)
    if structure is None:
        equality_sparsity = tuple(
            tuple(range(variable_count)) for _ in range(equality_count)
        )
        inequality_sparsity = tuple(
            tuple(range(variable_count)) for _ in range(inequality_count)
        )
    else:
        equality_sparsity = structure.equality_variable_indices
        inequality_sparsity = structure.inequality_variable_indices

    saved_override = getattr(problem, "_sparsity_override", None)
    if saved_override is not None:
        saved_equality, saved_inequality = saved_override
        if tuple(saved_equality) != tuple(equality_sparsity):
            raise ValueError(
                "Saved equality sparsity does not match the reconstructed symbolic problem."
            )
        if tuple(saved_inequality) != tuple(inequality_sparsity):
            raise ValueError(
                "Saved inequality sparsity does not match the reconstructed symbolic problem."
            )
        equality_sparsity = tuple(saved_equality)
        inequality_sparsity = tuple(saved_inequality)

    return CanonicalProblem(
        name=problem.name,
        parameter_size=parameter_count,
        variable_size=variable_count,
        equality_size=equality_count,
        inequality_size=inequality_count,
        objective=objective,
        equalities=equalities,
        inequalities=inequalities,
        objective_gradient=objective_gradient,
        equality_jacobian=equality_jacobian,
        inequality_jacobian=inequality_jacobian,
        lagrangian_hessian=lagrangian_hessian,
        equality_sparsity=equality_sparsity,
        inequality_sparsity=inequality_sparsity,
        equality_jacobian_values=equality_jacobian_values,
        inequality_jacobian_values=inequality_jacobian_values,
        equality_local_derivatives=equality_local_derivatives,
        inequality_local_derivatives=inequality_local_derivatives,
        constraint_hessian_vector_product=constraint_hessian_vector_product,
        metadata={
            "kind": "symbolic",
            "variable_names": tuple(problem.variable_names),
            "parameter_names": tuple(problem.parameter_names),
            "equalities_affine_in_variables": all(
                item.residual.affine_in_variables
                for item in problem.constraints
                if item.kind == "eq"
            ),
            "inequalities_affine_in_variables": all(
                item.residual.affine_in_variables
                for item in problem.constraints
                if item.kind == "ineq"
            ),
        },
    )
