<p align="center">
  <img src="assets/purin-logo-v2.png" alt="PURIN logo" width="220">
</p>

# PURIN

[![CI](https://github.com/SOULS-TAMU/Purin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Purin/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/Purin/codex/coverage/coverage.svg)](https://github.com/SOULS-TAMU/Purin/tree/codex/coverage)
[![CodeQL](https://github.com/SOULS-TAMU/Purin/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Purin/actions/workflows/codeql.yml)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Powered by JAX](https://img.shields.io/badge/JAX-powered-8A2BE2)](https://docs.jax.dev/)
[![License: Apache-2.0](https://img.shields.io/github/license/SOULS-TAMU/Purin?color=blue)](LICENSE)
[![Citation: BibTeX](https://img.shields.io/badge/Citation-BibTeX-6F4E37)](CITATION.bib)

**Differentiable constrained projections and sparse numerical kernels in JAX.**

PURIN turns problem definitions into reusable projection layers. Use it on
its own or as the numerical backend for OHAGI and the KKTHN, DAEHN and
NLPOPTNet frontends.

[Installation](#installation) · [Quick start](#quick-start) ·
[Projection methods](#projection-methods) · [Architecture](#architecture) ·
[Contributing](CONTRIBUTING.md)

- **Composable projections:** Euclidean distance, diagonal weighted distance
  and objective-aware local quadratic corrections.
- **Shared solvers:** closed-form equality projection, Chambolle–Pock, ADMM,
  Fischer–Burmeister variants and primal-dual interior point methods.
- **JAX integration:** sparse operators and differentiable projection layers
  for constrained learning workflows.

## Installation

Install from the GitHub source repository:

```sh
git clone https://github.com/SOULS-TAMU/Purin.git
cd Purin
python -m pip install .
```

For a local HardNet stack checkout, run `python -m pip install ./purin`
from the stack directory.

Python 3.10 or newer is required. Install JAX for your device using the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
PURIN does not depend on OHAGI or any frontend package.

## Quick start

Project a two-dimensional point onto the equality constraint `u + v = b`:

```python
import jax
import jax.numpy as jnp
from purin import Distance, Projection

jax.config.update("jax_enable_x64", True)

problem = Projection()
p = problem.add_parameter(["b"])
y = problem.add_variable(["u", "v"])
problem.objective = 0.0
problem.constraints.add(y.u + y.v == p.b)

project = Distance(problem, "linear-equality").prepare(batch_size=1)
candidate = jnp.array([[2.0, 0.0]])  # Shape: (batch, variables).
parameters = jnp.array([[1.0]])     # Shape: (batch, parameters).
projected = project(candidate, parameters)
print(projected)  # [[ 1.5 -0.5]]
```

Prepare the layer once and reuse it for inputs with the configured batch
size and dtype. Choose precision, solver settings and tolerances for your
problem; iterative solver convergence depends on that configuration.

## Projection methods

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

## Architecture

PURIN is the numerical projection library in the HardNet stack. It can be used
independently or through OHAGI's shared learning workflow, with responsibilities
divided across three layers:

- **KKTHN, DAEHN and NLPOPTNet** supply framework-specific problem formulations
  and processing through their frontend APIs.
- **OHAGI** builds and trains projected neural models, then handles checkpoints,
  JAX model export and inference.
- **PURIN** compiles projection problems, prepares reusable solver resources,
  and computes projections and their derivatives using shared numerical kernels.
  It does not depend on OHAGI or the frontend libraries.

## Project information

This is the initial public source distribution. Version 0.1.0 is in
preparation; no tagged release is published yet.

[Issues](https://github.com/SOULS-TAMU/Purin/issues) ·
[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) ·
[BibTeX citation](CITATION.bib)

## Cross-platform compatibility

CPU compatibility is verified in GitHub Actions:

| Platform | Python versions |
| --- | --- |
| Linux x86-64 | 3.10–3.14 |
| Windows x86-64 | 3.10–3.14 |
| macOS Apple Silicon | 3.10–3.14 |
| Linux ARM64 | 3.13 |

CI checks current compatible JAX releases and JAX 0.6.2 on Python 3.12
across all three OSes. See the [CI matrix](CONTRIBUTING.md#automated-checks).

## License

PURIN is licensed under [Apache-2.0](LICENSE). Attribution is recorded in
[AUTHORS](AUTHORS) and [NOTICE](NOTICE).
