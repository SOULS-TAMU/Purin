# Define a projection problem

## Distance projection

PURIN corrects a candidate neural output by solving a constrained problem:

$$
y^* = \arg\min_y \tfrac12\|y-\hat y\|_2^2,
\qquad h(\theta,y)=0,\quad g(\theta,y)\le0.
$$

`Projection`/`SymbolicProblem` registers named parameters and variables. Add
symbolic equalities, inequalities and bounds through `constraints.add(...)`.
`Distance` defines the correction objective. Prepare once for the intended batch
size, dtype and backend, then call with candidate and problem-parameter arrays.

The [complete example](examples/quickstart.rst) projects `[2, 0]` onto `u+v=1`,
returning `[1.5, -0.5]`. Nonnegativity is not implied by an equality; declare it
when the application requires it.

## Other problem suppliers

`FunctionalProblem` provides JAX-compatible functions and size/sparsity metadata.
`MatrixProblem`, `ParametricQP` and `ParametricLP` supply matrix-oriented problems.
Keep declared dimensions, parameter ordering and sparsity consistent with the
numeric functions.

| Definition | Interpretation |
|---|---|
| `Distance` | Identity-metric Euclidean correction |
| `WeightedDistance` | Positive diagonal metric for supported methods |
| `LocalQuadratic` | Positive diagonal quadratic objective and locally linearized constraints |
| `Sequential` | Composition of prepared correction stages |

A local QP approximates nonlinear constraints at an anchor. Solving it is not a
global solution of the original nonlinear problem. Check the
[solver requirements](SOLVERS.md) before selecting a backend.
