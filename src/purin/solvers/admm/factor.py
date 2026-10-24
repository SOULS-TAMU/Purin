"""Structure-only band ordering and reusable sparse Cholesky solves.

No dense normal matrix is assembled. Numerical factors are built once per
projection call (and after a rho change), never once per ADMM iteration.
"""

from dataclasses import dataclass
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True, eq=False)
class BandedPlan:
    """Fixed index layout used to assemble and solve a banded ADMM system."""

    permutation: object
    inverse: object
    rows: object
    left: object
    right: object
    destination: object
    n: int
    width: int

    def factor(self, diagonal, values, penalty, sigma):
        """Assemble only the lower band, then factor in O(n*w*w)."""
        n, w = self.n, self.width
        band = jnp.zeros((values.shape[0], n * (w + 1)), values.dtype)
        products = (
            penalty[:, self.rows]
            * values[:, self.rows, self.left]
            * values[:, self.rows, self.right]
        )
        band = band.at[:, self.destination].add(products).reshape((-1, n, w + 1))
        band = band.at[:, :, 0].add(diagonal[:, self.permutation] + sigma)

        # Column Cholesky: subtract already computed overlapping band entries.
        def column(j, factor):
            offsets = jnp.arange(1, w + 1)
            previous = j - offsets
            lj = jnp.where(previous[None, :] >= 0, factor[:, j, 1:], 0.0)
            pivot = jnp.sqrt(factor[:, j, 0] - jnp.sum(lj * lj, axis=1))
            factor = factor.at[:, j, 0].set(pivot)
            for d in range(1, w + 1):
                i = jnp.minimum(j + d, n - 1)
                ds = offsets + d
                overlap = (previous >= 0) & (ds <= w)
                li = factor[:, i, jnp.minimum(ds, w)]
                entry = (
                    factor[:, i, d]
                    - jnp.sum(jnp.where(overlap[None, :], li * lj, 0.0), axis=1)
                ) / pivot
                factor = factor.at[:, i, d].set(
                    jnp.where(j + d < n, entry, factor[:, i, d])
                )
            return factor

        return jax.lax.fori_loop(0, n, column, band)

    def solve(self, factor, rhs):
        """Two triangular band solves, O(n*w) work and O(n*w) storage."""
        n, w = self.n, self.width
        offsets = jnp.arange(1, w + 1)
        b = rhs[:, self.permutation]

        def forward(i, x):
            indices = jnp.maximum(i - offsets, 0)
            dot = jnp.sum(
                jnp.where(
                    (i >= offsets)[None, :], factor[:, i, 1:] * x[:, indices], 0.0
                ),
                axis=1,
            )
            return x.at[:, i].set((b[:, i] - dot) / factor[:, i, 0])

        y = jax.lax.fori_loop(0, n, forward, jnp.zeros_like(b))

        def backward(k, x):
            i = n - 1 - k
            indices = jnp.minimum(i + offsets, n - 1)
            dot = jnp.sum(
                jnp.where(
                    (i + offsets < n)[None, :],
                    factor[:, indices, offsets] * x[:, indices],
                    0.0,
                ),
                axis=1,
            )
            return x.at[:, i].set((y[:, i] - dot) / factor[:, i, 0])

        x = jax.lax.fori_loop(0, n, backward, jnp.zeros_like(b))
        return x[:, self.inverse]


def prepare_banded(support, mask, n):
    """Reverse Cuthill-McKee graph ordering; cap bandwidth to retain sparsity."""
    support, mask = np.asarray(support), np.asarray(mask)
    graph = [set() for _ in range(n)]
    for indices, valid in zip(support, mask):
        row = indices[valid].tolist()
        for i in row:
            graph[i].update(j for j in row if j != i)
    visited, order = set(), []
    while len(order) < n:
        root = min(
            (i for i in range(n) if i not in visited), key=lambda i: (len(graph[i]), i)
        )
        queue = deque([root])
        visited.add(root)
        while queue:
            i = queue.popleft()
            order.append(i)
            for j in sorted(graph[i] - visited, key=lambda j: (len(graph[j]), j)):
                if j not in visited:
                    visited.add(j)
                    queue.append(j)
    permutation = np.array(order[::-1], np.int32)
    inverse = np.argsort(permutation).astype(np.int32)
    width = max(
        (abs(int(inverse[i]) - int(inverse[j])) for i in range(n) for j in graph[i]),
        default=0,
    )
    if width > 8:
        return None
    rows, left, right, dest = [], [], [], []
    for r, (indices, valid) in enumerate(zip(support, mask)):
        for a in np.flatnonzero(valid):
            for b in np.flatnonzero(valid):
                i, j = int(inverse[indices[a]]), int(inverse[indices[b]])
                if i >= j:
                    rows.append(r)
                    left.append(a)
                    right.append(b)
                    dest.append(i * (width + 1) + i - j)
    return BandedPlan(
        *(
            jnp.asarray(x, dtype=jnp.int32)
            for x in (permutation, inverse, rows, left, right, dest)
        ),
        n,
        width,
    )
