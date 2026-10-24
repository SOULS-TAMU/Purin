"""Projection definitions separate problem construction from solver selection."""

from dataclasses import dataclass, replace
import jax.numpy as jnp
from purin.solvers import prepare_projection


@dataclass(frozen=True)
class Distance:
    """Euclidean nearest-feasible-point projection definition."""

    problem: object
    method: str = "fischer-burmeister-newton"
    config: object = None

    def prepare(self, *, batch_size, dtype=jnp.float64, backend="cpu"):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        problem = self.problem
        if not hasattr(problem, "as_projection_problem") and hasattr(
            problem, "variable_names"
        ):
            from purin.problems.symbolic import SymbolicProblem

            wrapper = SymbolicProblem()
            object.__setattr__(wrapper, "_core", problem)
            problem = wrapper
        return prepare_projection(
            problem,
            method=self.method,
            config=self.config,
            batch_size=batch_size,
            dtype=dtype,
            backend=backend,
        )


@dataclass(frozen=True)
class WeightedDistance:
    """Nearest-feasible-point projection with a positive diagonal metric."""

    problem: object
    diagonal: object
    method: str = "chambolle-pock"
    config: object = None

    def prepare(self, *, batch_size, dtype=jnp.float64, backend="cpu"):
        """Prepare reusable projection resources for a fixed batch size, dtype and backend."""
        from purin.solvers import ChambollePockConfig, ADMMConfig, ProjectionConfig

        configs = {
            "chambolle-pock": ChambollePockConfig,
            "admm": ADMMConfig,
            "primal-dual-interior-point": ProjectionConfig,
        }
        if self.method not in configs:
            raise ValueError("weighted projection supports CP, ADMM, or interior-point")
        cfg = self.config or configs[self.method]()
        cfg = replace(cfg, metric_diagonal=tuple(self.diagonal))
        return Distance(self.problem, self.method, cfg).prepare(
            batch_size=batch_size, dtype=dtype, backend=backend
        )
