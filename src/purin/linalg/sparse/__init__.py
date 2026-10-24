"""Shared sparse structures, operators, assembly, and iterative solvers."""

from purin.linalg.sparse.assembly import FixedSparseMatrixFactory
from purin.linalg.sparse.assembly import masked_diagonal
from purin.linalg.sparse.assembly import pattern_indices
from purin.linalg.sparse.assembly import transposed_jacobian_factories
from purin.linalg.sparse.conversion import as_bcoo
from purin.linalg.sparse.conversion import host_dense_array
from purin.linalg.sparse.minres import minres
from purin.linalg.sparse.operators import diagonal_operator
from purin.linalg.sparse.operators import matvec
from purin.linalg.sparse.operators import rmatvec
from purin.linalg.sparse.operators import symmetric_block_operator
from purin.linalg.sparse.types import LinearSolveResult
from purin.linalg.sparse.types import SparsePattern

__all__ = [
    "SparsePattern",
    "LinearSolveResult",
    "as_bcoo",
    "host_dense_array",
    "matvec",
    "rmatvec",
    "diagonal_operator",
    "symmetric_block_operator",
    "masked_diagonal",
    "FixedSparseMatrixFactory",
    "pattern_indices",
    "transposed_jacobian_factories",
    "minres",
]
