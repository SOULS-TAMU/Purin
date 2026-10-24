"""Backend-neutral immutable sparse-pattern metadata."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, NamedTuple


@dataclass(frozen=True)
class SparsePattern:
    """Static COO sparsity pattern prepared before JAX tracing."""

    shape: tuple[int, int]
    rows: Any
    cols: Any

    @property
    def nnz(self) -> int:
        """Return the number of stored positions in this sparsity pattern."""
        return int(len(self.rows))


class LinearSolveResult(NamedTuple):
    """Result of a fixed-shape iterative sparse linear solve."""

    solution: __import__("jax").Array
    residual_norm: __import__("jax").Array
    iterations: __import__("jax").Array
    converged: __import__("jax").Array
    numerical_failure: __import__("jax").Array
