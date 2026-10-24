"""Single public preparation entry point for all projection backends."""

from __future__ import annotations
import warnings
import jax.numpy as jnp
from purin.solvers.admm import ADMMConfig
from purin.solvers.admm import ADMMProjection
from purin.solvers.chambolle_pock import ChambollePockConfig
from purin.solvers.chambolle_pock import ChambollePockProjection
from purin.solvers.config import FischerBurmeisterConfig
from purin.solvers.config import LinearEqualityConfig
from purin.solvers.config import ProjectionConfig
from purin.solvers.fischer_burmeister import FullFischerBurmeisterProjection
from purin.solvers.fischer_burmeister import ReducedFischerBurmeisterProjection
from purin.solvers.linear_equality import CanonicalLinearEqualityProjection
from purin.solvers.linear_equality import LinearEqualityProjection
from purin.solvers.primal_dual_interior_point import PrimalDualInteriorPointProjection
from purin.solvers.registry import projection_factory
from purin.solvers.registry import registered_methods as custom_registered_methods
from purin.solvers.types import ProjectionMethod


def _linear_factory(problem, config):
    if isinstance(
        problem, (LinearEqualityProjection, CanonicalLinearEqualityProjection)
    ):
        return problem
    canonical = problem.canonicalize() if hasattr(problem, "canonicalize") else problem
    if canonical.inequality_size:
        message = (
            "linear-equality projection is valid only when every constraint is "
            "a linear equality; this problem contains inequalities. Select "
            "fischer-burmeister-newton, fischer-burmeister-reduced, or "
            "primal-dual-interior-point instead."
        )
        warnings.warn(message, UserWarning, stacklevel=3)
        raise ValueError(message)

    metadata = canonical.metadata if isinstance(canonical.metadata, dict) else {}
    affine = metadata.get("equalities_affine_in_variables")
    if affine is False:
        message = (
            "linear-equality projection is valid only when every constraint is "
            "affine in the decision variables; nonlinear equalities were found."
        )
        warnings.warn(message, UserWarning, stacklevel=3)
        raise ValueError(message)

    required = ("A_eq", "b_eq0", "b_eq_parameter_map")
    if all(hasattr(canonical, name) for name in required):
        return LinearEqualityProjection(
            A=canonical.A_eq,
            b0=canonical.b_eq0,
            b_parameter_map=canonical.b_eq_parameter_map,
            config=config,
        )
    if affine is True:
        return CanonicalLinearEqualityProjection(canonical=canonical, config=config)

    message = (
        "The linear-equality backend could not verify that the equalities are "
        "linear. Define the problem with the symbolic frontend or use an "
        "explicit LinearEqualityProjection(A=..., b=...)."
    )
    warnings.warn(message, UserWarning, stacklevel=3)
    raise TypeError(message)


def _full_fb_factory(problem, config):
    return FullFischerBurmeisterProjection(problem, config)


def _reduced_fb_factory(problem, config):
    return ReducedFischerBurmeisterProjection(problem, config)


def _pdip_factory(problem, config):
    return PrimalDualInteriorPointProjection(problem, config)


_DEFAULT_FACTORIES = {
    ProjectionMethod.LINEAR_EQUALITY: _linear_factory,
    ProjectionMethod.CHAMBOLLE_POCK: ChambollePockProjection,
    ProjectionMethod.ADMM: ADMMProjection,
    ProjectionMethod.FISCHER_BURMEISTER_NEWTON: _full_fb_factory,
    ProjectionMethod.FISCHER_BURMEISTER_REDUCED: _reduced_fb_factory,
    ProjectionMethod.PRIMAL_DUAL_INTERIOR_POINT: _pdip_factory,
}


def registered_methods() -> tuple[str, ...]:
    """Return built-in and explicitly registered projection methods.

    Built-in methods are represented by an immutable module-level mapping; no
    registry mutation occurs during import. Optional third-party registrations
    remain explicit through :func:`kkthn.opt.registry.register_projection`.
    """
    custom = set(custom_registered_methods())
    return tuple(
        method.value
        for method in ProjectionMethod
        if method in _DEFAULT_FACTORIES or method.value in custom
    )


def prepare_projection(
    problem,
    *,
    method: ProjectionMethod | str,
    config=None,
    batch_size: int,
    dtype=jnp.float64,
    backend: str = "cpu",
):
    """Prepare one fixed-shape projection backend.

    Preparation performs validation, symbolic compilation, and sparse-pattern
    construction once. Runtime calls update numerical values only.
    """
    selected = ProjectionMethod.normalize(method)
    if config is None:
        if selected is ProjectionMethod.ADMM:
            config = ADMMConfig()
        elif selected is ProjectionMethod.CHAMBOLLE_POCK:
            config = ChambollePockConfig()
        elif selected is ProjectionMethod.LINEAR_EQUALITY:
            config = LinearEqualityConfig()
        elif selected in {
            ProjectionMethod.FISCHER_BURMEISTER_NEWTON,
            ProjectionMethod.FISCHER_BURMEISTER_REDUCED,
        }:
            config = FischerBurmeisterConfig()
        else:
            config = ProjectionConfig()
    factory = _DEFAULT_FACTORIES.get(selected)
    if factory is None:
        factory = projection_factory(selected)
    definition = factory(problem, config)
    return definition.prepare(
        batch_size=int(batch_size), dtype=dtype, backend=str(backend)
    )
