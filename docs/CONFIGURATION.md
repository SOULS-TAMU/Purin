# Differentiation and sparse operators

## Implicit differentiation

Projection layers participate in JAX differentiation. Implicit derivatives use
local optimality equations instead of storing every forward iteration. For
an optimality residual $F(z,\theta)=0$:

$$
F_z\,dz = -F_\theta\,d\theta.
$$

The adjoint solves a transposed system for vector–Jacobian products. A suitable
local linearization and accurate forward solution are needed. Active-set changes,
degeneracy and nonconvergence can invalidate a naive smooth interpretation.

`ImplicitGradientConfig` controls policy and residual/adjoint criteria. Check
forward convergence and gradient eligibility separately. The
[Jacobian example](examples/derivatives.rst) compares a projection derivative with
its exact analytic matrix.

## Sparse computation and diagnostics

Declared sparsity supports sparse operators, Jacobian products and KKT/adjoint
linear algebra. Reuse prepared structure. Dense, diagonal and sparse interfaces
are in `purin.linalg.operators`. Specialized kernels depend on backend and problem
structure; a sparse formulation does not imply every intermediate is sparse.

Check physical equality residuals, positive inequality violation, iteration count
and status. Scaled solver residuals differ from physical violation. Nonlinear
KKT convergence gives a local condition, not global nearest-point optimality
for a nonconvex set.
