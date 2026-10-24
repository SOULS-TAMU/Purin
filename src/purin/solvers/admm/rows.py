"""Verified two-sided row coalescing with original multiplier reconstruction."""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from purin.solvers.admm.factor import prepare_banded


@dataclass(frozen=True, eq=False)
class PairedRows:
    """Row pairing metadata for compacting repeated affine constraint structure."""

    keep: object
    first: object
    second: object
    positions: object
    support: object
    mask: object
    factor_plan: object

    def verified(self, values, rhs):
        # Exact equality, no approximate row merging. Validate every call and
        # sample; parameter-dependent coefficients may invalidate the plan.
        # Retain zero-width intervals: opposite original multipliers can both
        # be positive there, and canceling them would change active-set VJPs.
        """Report whether the row-pairing assumptions were verified."""
        return jnp.all(values[:, self.first] == -values[:, self.second]) & jnp.all(
            -rhs[:, self.second] < rhs[:, self.first]
        )

    def compact(self, rhs, initial, equality_size):
        """Map full row values into the verified compact row representation."""
        upper = rhs[:, self.keep]
        lower = (
            jnp.full_like(upper, -jnp.inf)
            .at[:, :equality_size]
            .set(upper[:, :equality_size])
        )
        lower = lower.at[:, self.positions].set(-rhs[:, self.second])
        dual = initial[:, self.keep].at[:, self.positions].add(-initial[:, self.second])
        return lower, upper, dual

    def restore(self, compact_dual, original_rhs):
        """Expand compact row values back to the original constraint order."""
        dual = jnp.zeros_like(original_rhs).at[:, self.keep].set(compact_dual)
        signed = compact_dual[:, self.positions]
        dual = dual.at[:, self.first].set(jnp.maximum(signed, 0.0))
        return dual.at[:, self.second].set(jnp.maximum(-signed, 0.0))


def prepare_pairs(owner):
    """Use a reference only to nominate pairs, never to certify equivalence."""
    values = np.asarray(
        owner._numeric_one(jnp.zeros(owner.parameter_size, owner.dtype))[0]
    )
    if owner.config.row_scaling:
        norm = np.sqrt(np.sum(values**2, axis=1))
        values = values * np.where(norm > 0, 1 / np.maximum(norm, 1e-30), 1.0)[:, None]
    support, mask = np.asarray(owner._support), np.asarray(owner._mask)
    waiting, pairs = {}, []
    for row in range(owner.equality_size, len(values)):
        v = values[row]
        if not np.all(np.isfinite(v)) or not np.any(v):
            continue
        columns = tuple(support[row].tolist())
        key = (columns, tuple(v.tolist()))
        opposite = (columns, tuple((-v).tolist()))
        if waiting.get(opposite):
            pairs.append((waiting[opposite].pop(), row))
        else:
            waiting.setdefault(key, []).append(row)
    if not pairs:
        return None
    removed = {j for _, j in pairs}
    keep = [i for i in range(len(values)) if i not in removed]
    index = {r: i for i, r in enumerate(keep)}
    first, second = zip(*pairs)
    positions = [index[i] for i in first]
    return PairedRows(
        *(jnp.asarray(x, dtype=jnp.int32) for x in (keep, first, second, positions)),
        jnp.asarray(support[keep]),
        jnp.asarray(mask[keep]),
        prepare_banded(support[keep], mask[keep], owner.output_size),
    )
