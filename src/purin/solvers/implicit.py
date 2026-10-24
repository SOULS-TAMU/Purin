"""Backend-neutral implicit projection custom-VJP orchestration."""

from __future__ import annotations
from dataclasses import dataclass as _dataclass
from typing import Callable as _Callable
import jax as _jax
import jax.numpy as _jnp
import numpy as _np
from purin.solvers.primal_dual_interior_point.types import BATCH_STATUS_CONVERGED
from purin.solvers.primal_dual_interior_point.types import PrimalDualState
from purin.solvers.types import ProjectionLayerOutput
from purin.solvers.types import ProjectionTrainingOutput


@_dataclass(frozen=True)
class ImplicitProjectionLayer:
    """Projection boundary exposing solver-defined implicit differentiation."""

    project: _Callable
    project_with_info: _Callable
    project_training: _Callable
    pack_parameters: _Callable
    physical_parameter_count: int
    output_dim: int


def _normalize_projection_gradient_policy(value: str) -> str:
    selected = str(value).strip().lower()
    if selected not in {"strict", "residual_checked", "always"}:
        raise ValueError("gradient_policy must be strict, residual_checked, or always.")
    return selected


def make_implicit_projection_layer(
    prepared,
    *,
    mu: float = 1e-4,
    differentiate_problem_parameters: bool = False,
    gradient_policy: str = "residual_checked",
    inexact_gradient_tolerance: float = 1e-6,
    adjoint_gradient_tolerance: float = 1e-8,
) -> ImplicitProjectionLayer:
    """Create the preserved sparse final-KKT custom VJP projection layer.

    Learned duals may be supplied as numerical initialization only. Their
    cotangents are zero because differentiation is through the final KKT root,
    not through the iterative path or its initial state.
    """
    selected_gradient_policy = _normalize_projection_gradient_policy(gradient_policy)
    inexact_gradient_tolerance = float(inexact_gradient_tolerance)
    adjoint_gradient_tolerance = float(adjoint_gradient_tolerance)
    if not _np.isfinite(inexact_gradient_tolerance) or inexact_gradient_tolerance < 0.0:
        raise ValueError("inexact_gradient_tolerance must be finite and nonnegative.")
    if not _np.isfinite(adjoint_gradient_tolerance) or adjoint_gradient_tolerance < 0.0:
        raise ValueError("adjoint_gradient_tolerance must be finite and nonnegative.")

    physical_indices = prepared.physical_parameter_indices
    target_indices = prepared.target_parameter_indices
    batch_size = int(prepared.batch_size)
    parameter_count = int(prepared.parameter_count)
    output_dim = int(len(prepared.compiled.variable_names))
    physical_count = int(parameter_count - output_dim)
    equality_count = int(prepared.compiled.initial_state.y_c.shape[1])
    inequality_count = int(prepared.compiled.initial_state.y_d.shape[1])
    mu_values = _jnp.full((batch_size,), float(mu), dtype=_jnp.float64)

    def pack_parameters(problem_parameters, y_hat):
        problem_parameters = _jnp.asarray(problem_parameters, dtype=_jnp.float64)
        y_hat = _jnp.asarray(y_hat, dtype=_jnp.float64)
        packed = _jnp.zeros((batch_size, parameter_count), dtype=y_hat.dtype)
        packed = packed.at[:, target_indices].set(y_hat)
        return packed.at[:, physical_indices].set(problem_parameters)

    def state_is_finite(state):
        blocks = (state.x, state.s, state.y_c, state.y_d, state.z_L, state.z_U)
        finite = _jnp.ones((batch_size,), dtype=_jnp.bool_)
        for block in blocks:
            finite = finite & _jnp.all(_jnp.isfinite(block), axis=1)
        return finite

    def forward_eligible(status, residual, state_finite):
        residual_finite = _jnp.isfinite(residual)
        if selected_gradient_policy == "strict":
            policy_ok = status == BATCH_STATUS_CONVERGED
        elif selected_gradient_policy == "residual_checked":
            policy_ok = _jnp.logical_or(
                status == BATCH_STATUS_CONVERGED, residual <= inexact_gradient_tolerance
            )
        else:
            policy_ok = _jnp.ones_like(state_finite)
        return state_finite & residual_finite & policy_ok

    def make_output(forward):
        state_finite = state_is_finite(forward.state)
        gradient_eligible = forward_eligible(
            forward.status_codes, forward.final_residual_inf_norm, state_finite
        )
        basic = ProjectionLayerOutput(
            y_tilde=forward.state.x,
            forward_status_codes=forward.status_codes,
            forward_iterations=forward.iterations,
            forward_residual_inf_norm=forward.final_residual_inf_norm,
            forward_state_finite=state_finite,
            forward_gradient_eligible=gradient_eligible,
        )
        training = ProjectionTrainingOutput(
            y_tilde=forward.state.x,
            multipliers_eq=_jax.lax.stop_gradient(forward.state.y_c),
            multipliers_ineq=_jax.lax.stop_gradient(forward.state.y_d),
            slacks=_jax.lax.stop_gradient(forward.state.s),
            forward_status_codes=_jax.lax.stop_gradient(forward.status_codes),
            forward_iterations=_jax.lax.stop_gradient(forward.iterations),
            forward_residual_inf_norm=_jax.lax.stop_gradient(
                forward.final_residual_inf_norm
            ),
            forward_state_finite=_jax.lax.stop_gradient(state_finite),
            forward_gradient_eligible=_jax.lax.stop_gradient(gradient_eligible),
        )
        return basic, training

    def solve_default(problem_parameters, y_hat):
        packed = pack_parameters(problem_parameters, y_hat)
        forward = prepared.solve_with_linearization_jax(packed, mu_values)
        return make_output(forward), packed, forward

    def build_initial_state(packed, y_hat, lambda_eq, mu_ineq):
        margin = _jnp.asarray(prepared.interior_margin, dtype=y_hat.dtype)
        state = prepared.compiled.initialize_state_function(y_hat, packed, margin)
        lambda_eq = _jnp.asarray(lambda_eq, dtype=y_hat.dtype)
        mu_ineq = _jnp.asarray(mu_ineq, dtype=y_hat.dtype)
        if lambda_eq.shape != (batch_size, equality_count):
            raise ValueError(
                f"lambda_eq must have shape {(batch_size, equality_count)}; got {lambda_eq.shape}."
            )
        if mu_ineq.shape != (batch_size, inequality_count):
            raise ValueError(
                f"mu_ineq must have shape {(batch_size, inequality_count)}; got {mu_ineq.shape}."
            )
        # Keep the initial slack stationarity block -y_d + v_U = 0.
        return PrimalDualState(
            x=state.x,
            s=state.s,
            y_c=lambda_eq,
            y_d=mu_ineq,
            z_L=state.z_L,
            z_U=state.z_U,
            v_L=state.v_L,
            v_U=mu_ineq,
        )

    def solve_initialized(problem_parameters, y_hat, lambda_eq, mu_ineq):
        packed = pack_parameters(problem_parameters, y_hat)
        initial_state = build_initial_state(packed, y_hat, lambda_eq, mu_ineq)
        forward = prepared.solve_with_initial_state_jax(
            initial_state, packed, mu_values
        )
        return make_output(forward), packed, forward

    def backward_from_saved(saved, cotangent_y):
        (
            final_state,
            final_linearization,
            forward_status,
            forward_residual,
            forward_state_finite,
            packed,
        ) = saved
        adjoint = prepared.solve_yhat_adjoint_jax(
            final_state, final_linearization, cotangent_y
        )
        forward_valid = forward_eligible(
            forward_status, forward_residual, forward_state_finite
        )
        adjoint_finite = (
            _jnp.all(_jnp.isfinite(adjoint.x), axis=1)
            & _jnp.all(_jnp.isfinite(adjoint.y_c), axis=1)
            & _jnp.all(_jnp.isfinite(adjoint.y_d), axis=1)
            & _jnp.isfinite(adjoint.linear_residual_inf_norm)
        )
        if selected_gradient_policy == "strict":
            adjoint_valid = adjoint.linear_status == 0
        elif selected_gradient_policy == "residual_checked":
            adjoint_valid = _jnp.logical_or(
                adjoint.linear_status == 0,
                adjoint.linear_residual_inf_norm <= adjoint_gradient_tolerance,
            )
        else:
            adjoint_valid = _jnp.ones_like(adjoint_finite)
        valid = forward_valid & adjoint_finite & adjoint_valid
        grad_y_hat = adjoint.x
        if differentiate_problem_parameters:
            full_parameter_gradient = prepared.parameter_vjp_from_adjoint_jax(
                final_state, packed, adjoint
            )
            grad_problem_parameters = full_parameter_gradient[:, physical_indices]
        else:
            grad_problem_parameters = _jnp.zeros(
                (batch_size, physical_count), dtype=grad_y_hat.dtype
            )
        grad_y_hat = _jnp.nan_to_num(grad_y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        grad_problem_parameters = _jnp.nan_to_num(
            grad_problem_parameters, nan=0.0, posinf=0.0, neginf=0.0
        )
        return (
            _jnp.where(valid[:, None], grad_problem_parameters, 0.0),
            _jnp.where(valid[:, None], grad_y_hat, 0.0),
        )

    @_jax.custom_vjp
    def project_with_info(problem_parameters, y_hat):
        (basic, _training), _packed, _forward = solve_default(problem_parameters, y_hat)
        return basic

    def project_forward(problem_parameters, y_hat):
        (basic, _training), packed, forward = solve_default(problem_parameters, y_hat)
        return basic, (
            forward.state,
            forward.final_linearization,
            forward.status_codes,
            forward.final_residual_inf_norm,
            basic.forward_state_finite,
            packed,
        )

    def project_backward(saved, output_cotangent):
        grad_parameters, grad_y_hat = backward_from_saved(
            saved, output_cotangent.y_tilde
        )
        return grad_parameters, grad_y_hat

    project_with_info.defvjp(project_forward, project_backward)

    @_jax.custom_vjp
    def project_training(problem_parameters, y_hat, lambda_eq, mu_ineq):
        (_basic, training), _packed, _forward = solve_initialized(
            problem_parameters, y_hat, lambda_eq, mu_ineq
        )
        return training

    def training_forward(problem_parameters, y_hat, lambda_eq, mu_ineq):
        (_basic, training), packed, forward = solve_initialized(
            problem_parameters, y_hat, lambda_eq, mu_ineq
        )
        return training, (
            forward.state,
            forward.final_linearization,
            forward.status_codes,
            forward.final_residual_inf_norm,
            training.forward_state_finite,
            packed,
            lambda_eq,
            mu_ineq,
        )

    def training_backward(saved, output_cotangent):
        core_saved = saved[:6]
        lambda_eq, mu_ineq = saved[6], saved[7]
        grad_parameters, grad_y_hat = backward_from_saved(
            core_saved, output_cotangent.y_tilde
        )
        return (
            grad_parameters,
            grad_y_hat,
            _jnp.zeros_like(lambda_eq),
            _jnp.zeros_like(mu_ineq),
        )

    project_training.defvjp(training_forward, training_backward)

    def project(problem_parameters, y_hat):
        return project_with_info(problem_parameters, y_hat).y_tilde

    return ImplicitProjectionLayer(
        project=project,
        project_with_info=project_with_info,
        project_training=project_training,
        pack_parameters=pack_parameters,
        physical_parameter_count=physical_count,
        output_dim=output_dim,
    )


__all__ = [
    "ImplicitProjectionLayer",
    "ProjectionLayerOutput",
    "ProjectionTrainingOutput",
    "make_implicit_projection_layer",
]
