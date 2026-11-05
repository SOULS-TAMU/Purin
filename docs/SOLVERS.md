# Choose a projection solver


Choose a solver for the constraint structure of your projection problem:

| Algorithm / method name | Suitable problems | Implementation notes |
| --- | --- | --- |
| Linear equality (`linear-equality`) | Euclidean projection onto affine equalities, such as conservation equations or linear balance constraints. | Direct equality-system solve. Requires verified affine constraints; inequalities and nonlinear equalities are rejected. |
| Chambolle–Pock (`chambolle-pock`) | Convex distance projections with affine equalities, affine inequalities and bounds; useful for sparse polyhedral constraints. | First-order primal-dual splitting with fixed-step or accelerated updates through `ChambollePockConfig(accelerated=...)`. Both variants use shared sparse kernels. |
| ADMM (`admm`) | Convex distance projections with affine equalities, affine inequalities and bounds, including structured linear constraints. | OSQP-style operator splitting with configurable penalty and linear solves. Requires verified affine constraints. |
| Full Fischer–Burmeister Newton (`fischer-burmeister-newton`) | Distance projections with smooth nonlinear equalities and inequalities, as well as affine constraints. | Solves the full stationarity, feasibility and complementarity system using a Fischer–Burmeister reformulation. |
| Reduced Fischer–Burmeister (`fischer-burmeister-reduced`) | The same affine or smooth nonlinear constrained projection class, especially when a reduced sparse Newton system is useful. | Uses pivot reduction and sparse linear-system operators for the complementarity equations. |
| Primal-dual interior point (`primal-dual-interior-point`) | Distance projections with smooth equality and inequality constraints, including nonlinear formulations. | Uses primal-dual barrier iterations and Newton systems with positive slack and inequality-multiplier iterates. |

`Distance` defines Euclidean projection; `WeightedDistance` supplies a positive
diagonal metric for the supported CP, ADMM and interior-point interfaces.
`LocalQuadratic` builds a positive diagonal quadratic objective and linearizes
constraints around an anchor, then solves the local QP with CP. Its nonlinear
constraints are approximated in that subproblem. `Sequential` composes
corrections; it is not a separate numerical solver.

Lower-level closed-form `BoxProjection` (bound clipping) and `AffineProjection`
(affine equality projection) utilities are available in
`purin.projections.closed_form`. Use `registered_methods()` to list the six
solver method names above; these utilities are not additional registry entries.

For nonlinear, nonconvex constraints, convergence to a local stationary point
does not establish a globally nearest feasible point. Check solver diagnostics
and residuals, and choose precision and tolerances for your problem.


## Configure and check a run

Prepare outside a repeated prediction loop. Changing batch size, shape, dtype or
problem structure can trigger preparation/compilation again. Both fixed and
accelerated CP updates are selected through `ChambollePockConfig`; they are not
separate frontend implementations.

For nonlinear constraints choose suitable initialization, tolerances and iteration
limits. Inspect status and physical feasibility before trusting an output or its
gradient. See the [API configuration pages](api/solvers.rst) for exact defaults.
