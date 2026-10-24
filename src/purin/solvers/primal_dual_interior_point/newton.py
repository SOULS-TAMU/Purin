"""Sparse primal-dual Newton algebra, bound slacks, and step updates."""

from __future__ import annotations
import jax.numpy as jnp
from purin.solvers.primal_dual_interior_point.residuals import compute_kkt_residual
from purin.solvers.primal_dual_interior_point.types import BoundIndices
from purin.solvers.primal_dual_interior_point.types import BoundSlacks
from purin.solvers.primal_dual_interior_point.types import BoundValues
from purin.solvers.primal_dual_interior_point.types import FractionToBoundarySteps
from purin.solvers.primal_dual_interior_point.types import KKTResidual
from purin.solvers.primal_dual_interior_point.types import NewtonRegularization
from purin.solvers.primal_dual_interior_point.types import NewtonSolveResult
from purin.solvers.primal_dual_interior_point.types import NLPQuantities
from purin.solvers.primal_dual_interior_point.types import PrimalDualState
from purin.solvers.primal_dual_interior_point.types import ReducedKKTSystem


def compute_bound_slacks(
    state: PrimalDualState, indices: BoundIndices, bounds: BoundValues
) -> BoundSlacks:
    """Compute distances from ``x`` and ``s`` to all finite bounds."""
    return BoundSlacks(
        x_lower=state.x[indices.x_lower] - bounds.x_lower,
        x_upper=bounds.x_upper - state.x[indices.x_upper],
        s_lower=state.s[indices.s_lower] - bounds.s_lower,
        s_upper=bounds.s_upper - state.s[indices.s_upper],
    )


def compute_sigma(state: PrimalDualState, indices: BoundIndices, slacks: BoundSlacks):
    """Form barrier curvature terms from bound slacks and multipliers."""
    sigma_x = jnp.zeros_like(state.x)
    sigma_s = jnp.zeros_like(state.s)

    sigma_x = sigma_x.at[indices.x_lower].add(state.z_L / slacks.x_lower)
    sigma_x = sigma_x.at[indices.x_upper].add(state.z_U / slacks.x_upper)
    sigma_s = sigma_s.at[indices.s_lower].add(state.v_L / slacks.s_lower)
    sigma_s = sigma_s.at[indices.s_upper].add(state.v_U / slacks.s_upper)
    return sigma_x, sigma_s


def build_reduced_kkt_rhs(
    residual: KKTResidual, indices: BoundIndices, slacks: BoundSlacks
):
    """Assemble the condensed primal-dual Newton right-hand side."""
    b_x = -residual.x
    b_s = -residual.s

    b_x = b_x.at[indices.x_lower].add(-residual.z_L / slacks.x_lower)
    b_x = b_x.at[indices.x_upper].add(residual.z_U / slacks.x_upper)
    b_s = b_s.at[indices.s_lower].add(-residual.v_L / slacks.s_lower)
    b_s = b_s.at[indices.s_upper].add(residual.v_U / slacks.s_upper)

    return jnp.concatenate((b_x, b_s, -residual.y_c, -residual.y_d))


def build_reduced_kkt_matrix(
    state: PrimalDualState,
    nlp: NLPQuantities,
    sigma_x,
    sigma_s,
    *,
    regularization: NewtonRegularization | None = None,
):
    """Assemble the reduced KKT matrix with the requested regularization."""
    reg = regularization or NewtonRegularization()

    n_x = state.x.shape[0]
    n_s = state.s.shape[0]
    n_c = state.y_c.shape[0]
    n_d = state.y_d.shape[0]
    dtype = state.x.dtype

    xx = nlp.hessian_lagrangian + jnp.diag(sigma_x + reg.delta_x)
    xs = jnp.zeros((n_x, n_s), dtype=dtype)
    sc = jnp.zeros((n_s, n_c), dtype=dtype)
    cs = jnp.zeros((n_c, n_s), dtype=dtype)
    cd = jnp.zeros((n_c, n_d), dtype=dtype)
    dc = jnp.zeros((n_d, n_c), dtype=dtype)

    ss = jnp.diag(sigma_s + reg.delta_s)
    cc = -reg.delta_c * jnp.eye(n_c, dtype=dtype)
    dd = -reg.delta_d * jnp.eye(n_d, dtype=dtype)
    minus_identity = -jnp.eye(n_s, dtype=dtype)

    row_x = jnp.concatenate((xx, xs, nlp.jac_c.T, nlp.jac_d.T), axis=1)
    row_s = jnp.concatenate((xs.T, ss, sc, minus_identity), axis=1)
    row_c = jnp.concatenate((nlp.jac_c, cs, cc, cd), axis=1)
    row_d = jnp.concatenate((nlp.jac_d, minus_identity, dc, dd), axis=1)
    return jnp.concatenate((row_x, row_s, row_c, row_d), axis=0)


def assemble_reduced_kkt_system(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    *,
    regularization: NewtonRegularization | None = None,
) -> ReducedKKTSystem:
    """Build the reduced Newton matrix, right-hand side and elimination resources."""
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    matrix = build_reduced_kkt_matrix(
        state, nlp, sigma_x, sigma_s, regularization=regularization
    )
    rhs = build_reduced_kkt_rhs(residual, indices, slacks)
    return ReducedKKTSystem(matrix, rhs, sigma_x, sigma_s)


def split_reduced_solution(reduced_solution, state: PrimalDualState):
    """Unpack the reduced Newton solution into its state-variable directions."""
    n_x = state.x.shape[0]
    n_s = state.s.shape[0]
    n_c = state.y_c.shape[0]

    x_end = n_x
    s_end = x_end + n_s
    c_end = s_end + n_c
    return (
        reduced_solution[:x_end],
        reduced_solution[x_end:s_end],
        reduced_solution[s_end:c_end],
        reduced_solution[c_end:],
    )


def recover_bound_multiplier_directions(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    delta_x,
    delta_s,
):
    """Recover eliminated bound-multiplier directions from the reduced solution."""
    delta_z_L = (-residual.z_L - state.z_L * delta_x[indices.x_lower]) / slacks.x_lower
    delta_z_U = (-residual.z_U + state.z_U * delta_x[indices.x_upper]) / slacks.x_upper
    delta_v_L = (-residual.v_L - state.v_L * delta_s[indices.s_lower]) / slacks.s_lower
    delta_v_U = (-residual.v_U + state.v_U * delta_s[indices.s_upper]) / slacks.s_upper
    return delta_z_L, delta_z_U, delta_v_L, delta_v_U


def build_full_kkt_matrix(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    *,
    regularization: NewtonRegularization | None = None,
):
    """Assemble the full primal-dual KKT matrix for residual verification."""
    reg = regularization or NewtonRegularization()
    sizes = tuple(block.shape[0] for block in state)
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)

    (x0, s0, yc0, yd0, zl0, zu0, vl0, vu0, end) = offsets
    n_x, n_s, n_c, n_d, n_zl, n_zu, n_vl, n_vu = sizes
    dtype = state.x.dtype

    matrix = jnp.zeros((end, end), dtype=dtype)

    matrix = matrix.at[x0:s0, x0:s0].set(
        nlp.hessian_lagrangian + reg.delta_x * jnp.eye(n_x, dtype=dtype)
    )
    matrix = matrix.at[x0:s0, yc0:yd0].set(nlp.jac_c.T)
    matrix = matrix.at[x0:s0, yd0:zl0].set(nlp.jac_d.T)

    if n_zl:
        cols = zl0 + jnp.arange(n_zl)
        rows = x0 + indices.x_lower
        matrix = matrix.at[rows, cols].set(-1.0)
    if n_zu:
        cols = zu0 + jnp.arange(n_zu)
        rows = x0 + indices.x_upper
        matrix = matrix.at[rows, cols].set(1.0)

    matrix = matrix.at[s0:yc0, s0:yc0].set(reg.delta_s * jnp.eye(n_s, dtype=dtype))
    matrix = matrix.at[s0:yc0, yd0:zl0].set(-jnp.eye(n_s, dtype=dtype))
    if n_vl:
        cols = vl0 + jnp.arange(n_vl)
        rows = s0 + indices.s_lower
        matrix = matrix.at[rows, cols].set(-1.0)
    if n_vu:
        cols = vu0 + jnp.arange(n_vu)
        rows = s0 + indices.s_upper
        matrix = matrix.at[rows, cols].set(1.0)

    matrix = matrix.at[yc0:yd0, x0:s0].set(nlp.jac_c)
    matrix = matrix.at[yd0:zl0, x0:s0].set(nlp.jac_d)
    matrix = matrix.at[yd0:zl0, s0:yc0].set(-jnp.eye(n_s, dtype=dtype))
    matrix = matrix.at[yc0:yd0, yc0:yd0].set(-reg.delta_c * jnp.eye(n_c, dtype=dtype))
    matrix = matrix.at[yd0:zl0, yd0:zl0].set(-reg.delta_d * jnp.eye(n_d, dtype=dtype))

    if n_zl:
        rows = zl0 + jnp.arange(n_zl)
        matrix = matrix.at[rows, x0 + indices.x_lower].set(state.z_L)
        matrix = matrix.at[rows, rows].set(slacks.x_lower)
    if n_zu:
        rows = zu0 + jnp.arange(n_zu)
        matrix = matrix.at[rows, x0 + indices.x_upper].set(-state.z_U)
        matrix = matrix.at[rows, rows].set(slacks.x_upper)
    if n_vl:
        rows = vl0 + jnp.arange(n_vl)
        matrix = matrix.at[rows, s0 + indices.s_lower].set(state.v_L)
        matrix = matrix.at[rows, rows].set(slacks.s_lower)
    if n_vu:
        rows = vu0 + jnp.arange(n_vu)
        matrix = matrix.at[rows, s0 + indices.s_upper].set(-state.v_U)
        matrix = matrix.at[rows, rows].set(slacks.s_upper)

    return matrix


def compute_full_newton_residual(
    state: PrimalDualState,
    direction: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    *,
    regularization: NewtonRegularization | None = None,
):
    """Evaluate the full KKT residual of a candidate Newton direction."""
    matrix = build_full_kkt_matrix(
        state, nlp, indices, slacks, regularization=regularization
    )
    return matrix @ direction.flatten() + residual.flatten()


def solve_primal_dual_newton_system(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: float,
    *,
    regularization: NewtonRegularization | None = None,
    check_full_residual: bool = True,
) -> NewtonSolveResult:
    """Solve the reduced KKT system and recover the full Newton direction."""
    slacks = compute_bound_slacks(state, indices, bounds)
    residual = compute_kkt_residual(state, nlp, indices, slacks, mu)
    system = assemble_reduced_kkt_system(
        state, nlp, indices, slacks, residual, regularization=regularization
    )
    solution = jnp.linalg.solve(system.matrix, system.rhs)
    delta_x, delta_s, delta_y_c, delta_y_d = split_reduced_solution(solution, state)
    delta_z_L, delta_z_U, delta_v_L, delta_v_U = recover_bound_multiplier_directions(
        state, indices, slacks, residual, delta_x, delta_s
    )
    direction = PrimalDualState(
        delta_x,
        delta_s,
        delta_y_c,
        delta_y_d,
        delta_z_L,
        delta_z_U,
        delta_v_L,
        delta_v_U,
    )
    reduced_linear_residual = system.matrix @ solution - system.rhs
    reduced_norm = jnp.max(jnp.abs(reduced_linear_residual), initial=0.0)

    if check_full_residual:
        full_linear_residual = compute_full_newton_residual(
            state,
            direction,
            nlp,
            indices,
            slacks,
            residual,
            regularization=regularization,
        )
        full_norm = jnp.max(jnp.abs(full_linear_residual), initial=0.0)
    else:
        full_norm = jnp.asarray(jnp.nan, dtype=state.x.dtype)

    return NewtonSolveResult(
        direction, residual, slacks, system, solution, reduced_norm, full_norm
    )


def fraction_to_boundary(values, directions, tau: float):
    """Largest step in ``[0, 1]`` retaining a ``1 - tau`` fraction."""
    moving = directions < 0.0
    safe_denominator = jnp.where(moving, directions, -jnp.ones_like(directions))
    candidates = jnp.where(moving, -tau * values / safe_denominator, jnp.inf)
    return jnp.minimum(1.0, jnp.min(candidates, initial=jnp.inf))


def compute_fraction_to_boundary_steps(
    state: PrimalDualState,
    direction: PrimalDualState,
    indices: BoundIndices,
    bounds: BoundValues,
    tau: float,
) -> FractionToBoundarySteps:
    """Bound primal and dual step lengths to preserve strict interior feasibility."""
    slacks = compute_bound_slacks(state, indices, bounds)
    return FractionToBoundarySteps(
        x_lower=fraction_to_boundary(slacks.x_lower, direction.x[indices.x_lower], tau),
        x_upper=fraction_to_boundary(
            slacks.x_upper, -direction.x[indices.x_upper], tau
        ),
        s_lower=fraction_to_boundary(slacks.s_lower, direction.s[indices.s_lower], tau),
        s_upper=fraction_to_boundary(
            slacks.s_upper, -direction.s[indices.s_upper], tau
        ),
        z_L=fraction_to_boundary(state.z_L, direction.z_L, tau),
        z_U=fraction_to_boundary(state.z_U, direction.z_U, tau),
        v_L=fraction_to_boundary(state.v_L, direction.v_L, tau),
        v_U=fraction_to_boundary(state.v_U, direction.v_U, tau),
    )


def apply_primal_dual_step(
    state: PrimalDualState,
    direction: PrimalDualState,
    *,
    alpha_primal,
    alpha_dual,
    alpha_equality=None,
) -> PrimalDualState:
    """Advance primal and dual blocks with their respective step lengths."""
    if alpha_equality is None:
        alpha_equality = alpha_dual

    return PrimalDualState(
        x=state.x + alpha_primal * direction.x,
        s=state.s + alpha_primal * direction.s,
        y_c=state.y_c + alpha_equality * direction.y_c,
        y_d=state.y_d + alpha_equality * direction.y_d,
        z_L=state.z_L + alpha_dual * direction.z_L,
        z_U=state.z_U + alpha_dual * direction.z_U,
        v_L=state.v_L + alpha_dual * direction.v_L,
        v_U=state.v_U + alpha_dual * direction.v_U,
    )


def compute_primal_dual_newton_direction(
    state, nlp, indices, bounds, mu, *, regularization=None
):
    """Evaluate KKT resources and compute one primal-dual Newton direction."""
    return solve_primal_dual_newton_system(
        state,
        nlp,
        indices,
        bounds,
        mu,
        regularization=regularization,
        check_full_residual=False,
    ).direction
