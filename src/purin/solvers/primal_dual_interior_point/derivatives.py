"""Compiled derivative containers used by the primal-dual backend."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from purin.problems.sparsity import ProjectionStructure
from purin.solvers.primal_dual_interior_point.types import BoundIndices
from purin.solvers.primal_dual_interior_point.types import BoundValues
from purin.solvers.primal_dual_interior_point.types import NLPQuantities
from purin.solvers.primal_dual_interior_point.types import PrimalDualState

JaxNLPQuantities = NLPQuantities
JaxPrimalDualState = PrimalDualState

ScalarFunction = Callable[[jnp.ndarray, jnp.ndarray], jax.Array]
VectorFunction = Callable[[jnp.ndarray, jnp.ndarray], jax.Array]


@dataclass(frozen=True)
class CompiledProblem:
    """Low-level objects generated from one parameter binding."""

    name: str
    variable_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    initial_state: PrimalDualState
    indices: BoundIndices
    bounds: BoundValues
    evaluate_nlp: Callable[[PrimalDualState], NLPQuantities]
    objective_function: ScalarFunction
    equality_function: VectorFunction
    inequality_function: VectorFunction

    def objective(self, variables: Any) -> float:
        """Evaluate the compiled scalar objective at the supplied primal values."""
        value = self.objective_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return float(np.asarray(value))

    def equalities(self, variables: Any) -> np.ndarray:
        """Evaluate the compiled equality residual vector."""
        value = self.equality_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(value, dtype=np.float64)

    def inequalities(self, variables: Any) -> np.ndarray:
        """Evaluate the compiled inequality residual vector."""
        value = self.inequality_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class CompiledBatchProblem:
    """JAX-compiled same-structure batch of parameterized problems."""

    name: str
    variable_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    initial_state: JaxPrimalDualState
    indices: Any
    bounds: Any
    evaluate_nlp_single: Callable[[JaxPrimalDualState, jax.Array], JaxNLPQuantities]
    objective_function: ScalarFunction
    equality_function: VectorFunction
    inequality_function: VectorFunction
    initialize_state_function: Callable[
        [jax.Array, jax.Array, jax.Array], JaxPrimalDualState
    ]
    objective_batch_function: Callable[[jax.Array, jax.Array], jax.Array]
    equality_batch_function: Callable[[jax.Array, jax.Array], jax.Array]
    inequality_batch_function: Callable[[jax.Array, jax.Array], jax.Array]
    equality_jacobian_function: Callable[[jax.Array, jax.Array], jax.Array]
    inequality_jacobian_function: Callable[[jax.Array, jax.Array], jax.Array]
    equality_sparse_jacobian_values_function: Callable | None = None
    inequality_sparse_jacobian_values_function: Callable | None = None
    constraint_hvp_function: Callable | None = None
    equality_local_derivatives_function: Callable | None = None
    inequality_local_derivatives_function: Callable | None = None
    equality_affine_offset_function: Callable | None = None
    inequality_affine_offset_function: Callable | None = None
    equality_constant_local_jacobian: np.ndarray | None = None
    inequality_constant_local_jacobian: np.ndarray | None = None
    projection_structure: ProjectionStructure | None = None

    @property
    def batch_size(self) -> int:
        """Return the number of problem instances in the compiled batch."""
        return int(self.parameter_values.shape[0])

    def evaluate_nlp(self, state: JaxPrimalDualState) -> JaxNLPQuantities:
        """Evaluate objective derivatives and constraint resources for the current state."""
        return jax.vmap(self.evaluate_nlp_single, in_axes=(0, 0))(
            state, jnp.asarray(self.parameter_values, dtype=jnp.float64)
        )

    def objectives(self, variables: Any) -> np.ndarray:
        """Evaluate objective values for every instance in the batch."""
        values = self.objective_batch_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=np.float64)

    def equalities(self, variables: Any) -> np.ndarray:
        """Evaluate equality residuals for every instance in the batch."""
        values = self.equality_batch_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=np.float64)

    def inequalities(self, variables: Any) -> np.ndarray:
        """Evaluate inequality residuals for every instance in the batch."""
        values = self.inequality_batch_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=np.float64)
