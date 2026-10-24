"""Static registry for interchangeable projection backends."""

from __future__ import annotations
from collections.abc import Callable
from purin.solvers.types import ProjectionMethod

_REGISTRY: dict[ProjectionMethod, Callable] = {}


def register_projection(method: ProjectionMethod | str, factory: Callable) -> None:
    """Register a projection implementation under its canonical method name."""
    selected = ProjectionMethod.normalize(method)
    if selected in _REGISTRY:
        raise ValueError(f"Projection method {selected.value!r} is already registered.")
    _REGISTRY[selected] = factory


def projection_factory(method: ProjectionMethod | str) -> Callable:
    """Resolve the registered constructor for a requested projection method."""
    selected = ProjectionMethod.normalize(method)
    try:
        return _REGISTRY[selected]
    except KeyError as error:
        raise RuntimeError(
            f"Projection method {selected.value!r} has not been registered."
        ) from error


def registered_methods() -> tuple[str, ...]:
    """Return the available projection method names."""
    return tuple(item.value for item in ProjectionMethod if item in _REGISTRY)
