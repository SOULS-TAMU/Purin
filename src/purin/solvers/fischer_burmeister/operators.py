"""Matrix-free operators for pivot-reduced Fischer--Burmeister systems."""

from __future__ import annotations
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp


class ReducedFBOperatorBundle(NamedTuple):
    """Matrix-free operators for the reduced Fischer--Burmeister Newton system."""

    operator: Callable
    rhs: jax.Array
    multiplier_diagonal: jax.Array


class ReducedFBSchurBundle(NamedTuple):
    """Schur-complement operator resources for reduced Fischer--Burmeister solves."""

    operator: Callable
    rhs: jax.Array
    inverse_diagonal: jax.Array


def build_reduced_kkt_operator(
    *,
    split_reduced: Callable,
    combined_jv: Callable,
    combined_jtv: Callable,
    weighted_hessian_matvec: Callable,
    combined_values,
    mask_p,
    ratio_q,
    ratio_p,
    regularization,
    primal_rhs,
    equality_rhs,
    multiplier_rhs,
) -> ReducedFBOperatorBundle:
    """Build the symmetric pivot-reduced FB KKT operator and RHS."""
    dtype = regularization.dtype
    multiplier_diagonal = jnp.where(
        mask_p.astype(bool),
        -ratio_p - regularization[:, None],
        jnp.asarray(-1.0, dtype),
    )

    def operator(reduced_vector):
        d_y, d_lambda, d_mu = split_reduced(reduced_vector)
        combined_forward = combined_jv(combined_values, d_y)
        equality_size = d_lambda.shape[1]
        a_dy = combined_forward[:, :equality_size]
        c_dy = combined_forward[:, equality_size:]
        inequality_dual = mask_p * d_mu + ratio_q * c_dy
        combined_dual = jnp.concatenate((d_lambda, inequality_dual), axis=1)
        primal = weighted_hessian_matvec(d_y)
        primal = primal + combined_jtv(combined_values, combined_dual)
        primal = primal + regularization[:, None] * d_y
        equality = a_dy - regularization[:, None] * d_lambda
        inequality = mask_p * c_dy + multiplier_diagonal * d_mu
        return jnp.concatenate((primal, equality, inequality), axis=1)

    rhs = jnp.concatenate((primal_rhs, equality_rhs, multiplier_rhs), axis=1)
    return ReducedFBOperatorBundle(operator, rhs, multiplier_diagonal)


def build_primal_schur_operator(
    *,
    equality_jv: Callable,
    equality_jtv: Callable,
    inequality_jv: Callable,
    inequality_jtv: Callable,
    weighted_hessian_matvec: Callable,
    equality_jacobian,
    inequality_jacobian,
    mask_p,
    ratio_q,
    multiplier_diagonal,
    regularization,
    primal_rhs,
    equality_rhs,
    multiplier_rhs,
    hessian_diagonal,
    equality_jacobian_squared,
    inequality_jacobian_squared,
    diagonal_floor: float,
) -> ReducedFBSchurBundle:
    """Build the SPD primal Schur complement for affine/convex FB systems.

    Equality and retained multiplier directions are eliminated from the
    symmetric reduced KKT system.  ``multiplier_diagonal`` is negative on
    retained rows, making ``-1 / multiplier_diagonal`` positive.
    """
    dtype = primal_rhs.dtype
    floor = jnp.asarray(diagonal_floor, dtype)
    safe_regularization = jnp.maximum(regularization, floor)
    safe_multiplier = jnp.where(
        jnp.abs(multiplier_diagonal) > floor, multiplier_diagonal, -floor
    )
    retained_weight = mask_p * (-1.0 / safe_multiplier)

    def operator(d_y):
        a_dy = equality_jv(equality_jacobian, d_y)
        c_dy = inequality_jv(inequality_jacobian, d_y)
        result = weighted_hessian_matvec(d_y)
        result = result + regularization[:, None] * d_y
        result = result + inequality_jtv(inequality_jacobian, ratio_q * c_dy)
        result = result + equality_jtv(
            equality_jacobian, a_dy / safe_regularization[:, None]
        )
        result = result + inequality_jtv(inequality_jacobian, retained_weight * c_dy)
        return result

    rhs = primal_rhs
    rhs = rhs + equality_jtv(
        equality_jacobian, equality_rhs / safe_regularization[:, None]
    )
    rhs = rhs - inequality_jtv(
        inequality_jacobian, mask_p * multiplier_rhs / safe_multiplier
    )

    diagonal = hessian_diagonal + regularization[:, None]
    diagonal = diagonal + inequality_jtv(inequality_jacobian_squared, jnp.abs(ratio_q))
    diagonal = diagonal + equality_jtv(
        equality_jacobian_squared, 1.0 / safe_regularization[:, None]
    )
    diagonal = diagonal + inequality_jtv(inequality_jacobian_squared, retained_weight)
    inverse_diagonal = 1.0 / jnp.maximum(jnp.abs(diagonal), floor)
    return ReducedFBSchurBundle(operator, rhs, inverse_diagonal)


def recover_schur_multipliers(
    *,
    d_y,
    equality_jv: Callable,
    inequality_jv: Callable,
    equality_jacobian,
    inequality_jacobian,
    mask_p,
    regularization,
    multiplier_diagonal,
    equality_rhs,
    multiplier_rhs,
    diagonal_floor: float,
):
    """Recover equality and retained FB multiplier directions from ``d_y``."""
    dtype = d_y.dtype
    floor = jnp.asarray(diagonal_floor, dtype)
    safe_regularization = jnp.maximum(regularization, floor)
    safe_multiplier = jnp.where(
        jnp.abs(multiplier_diagonal) > floor, multiplier_diagonal, -floor
    )
    a_dy = equality_jv(equality_jacobian, d_y)
    c_dy = inequality_jv(inequality_jacobian, d_y)
    d_lambda = (a_dy - equality_rhs) / safe_regularization[:, None]
    retained_d_mu = mask_p * (multiplier_rhs - c_dy) / safe_multiplier
    return d_lambda, retained_d_mu


__all__ = [
    "ReducedFBOperatorBundle",
    "ReducedFBSchurBundle",
    "build_reduced_kkt_operator",
    "build_primal_schur_operator",
    "recover_schur_multipliers",
]
