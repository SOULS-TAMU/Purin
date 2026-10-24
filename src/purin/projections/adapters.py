"""Protocol adapters for prepared problem/correction representations."""

from dataclasses import dataclass
from typing import NamedTuple


class WithInfoResult(NamedTuple):
    """Adapter output carrying primal values and projection diagnostics."""

    y_tilde: object
    forward_gradient_eligible: object
    info: object


@dataclass(frozen=True)
class WithInfoProjection:
    """Adapt the historical target/context/diagnostics interface."""

    layer: object
    accept_empty_duals = True

    def project_training(
        self, target, context, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        y, info = self.layer.with_info(target, context)
        return WithInfoResult(y, info.gradient_eligible, info)


@dataclass(frozen=True)
class ImplicitDistance:
    """Adapt an implicit distance-projection callable to the training contract."""

    layer: object
    use_dual_warm_start: bool = False
    accept_empty_duals = True

    def project_training(
        self, target, context, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        if self.use_dual_warm_start:
            return self.layer.project_training(
                context, target, multipliers_eq, multipliers_ineq
            )
        return self.layer.project_with_info(context, target)


@dataclass(frozen=True)
class CallableCorrection:
    """Normalize an objective-specific local builder to the common pathway."""

    step: object
    layers: int = 1
    accept_empty_duals = True

    def project_training(
        self, target, context, *, multipliers_eq=None, multipliers_ineq=None
    ):
        """Adapt the projection output and gradient eligibility to the shared training contract."""
        from .sequential import correction_layers, CorrectionValues
        import jax.numpy as jnp

        def apply(y, eq, iq):
            y, eq, iq = self.step(context, y, eq, iq)
            return CorrectionValues(y, eq, iq, jnp.ones(y.shape[0], dtype=bool))

        return correction_layers(
            apply, target, multipliers_eq, multipliers_ineq, self.layers
        )[-1]
