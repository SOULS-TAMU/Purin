"""Differentiation contracts and preserved solver-specific adjoints."""

from purin.solvers.config import ImplicitGradientConfig as ImplicitGradientConfig
from purin.solvers.implicit import (
    ImplicitProjectionLayer as ImplicitProjectionLayer,
    make_implicit_projection_layer as make_implicit_projection_layer,
)
from .policies import GradientPolicy as GradientPolicy
