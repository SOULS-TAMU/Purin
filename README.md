<p align="center">
  <img src="assets/purin-logo-v2.png" alt="PURIN logo" width="200">
</p>

# PURIN

**Differentiable constraint projections and sparse numerical kernels in JAX.**

[![CI](https://github.com/SOULS-TAMU/Purin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Purin/actions/workflows/ci.yml?query=branch%3Amain)
[![Validation](https://github.com/SOULS-TAMU/Purin/actions/workflows/validation.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Purin/actions/workflows/validation.yml?query=branch%3Amain)
[![CodeQL](https://github.com/SOULS-TAMU/Purin/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Purin/actions/workflows/codeql.yml?query=branch%3Amain)
[![Dependency review](https://github.com/SOULS-TAMU/Purin/actions/workflows/dependency-review.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/Purin/actions/workflows/dependency-review.yml?query=branch%3Amain)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/Purin/codex/coverage/coverage.svg)](https://github.com/SOULS-TAMU/Purin/tree/codex/coverage)

[![Python 3.10–3.14](https://img.shields.io/badge/Python-3.10%E2%80%933.14-3776AB?logo=python&logoColor=white)](.github/workflows/ci.yml)
[![CPU platforms](https://img.shields.io/badge/CPU-Linux%20%7C%20Windows%20%7C%20macOS-555555)](.github/workflows/ci.yml)
[![JAX](https://img.shields.io/badge/Powered_by-JAX-8A2BE2)](https://docs.jax.dev/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Citation: BibTeX](https://img.shields.io/badge/Cite-BibTeX-6F4E37)](CITATION.bib)

Turn a parameterized constraint problem into a reusable projection layer.
Use PURIN independently or as the numerical backend for OHAGI and the HardNet
frontends.

- **Projection formulations:** Euclidean distance, diagonal weighted distance
  and objective-aware local quadratic corrections.
- **Six solver methods:** affine equality projection, Chambolle–Pock, ADMM,
  full/reduced Fischer–Burmeister and primal-dual interior point.
- **Sparse computation:** reusable operators and kernels, including matrix-free
  paths where supported by the selected solver.
- **Differentiable layers:** solver-specific implicit differentiation and
  numerical diagnostics for constrained learning.

[Installation](#installation) · [Basic API](#basic-api) · [Projection methods](#projection-methods) · [Documentation](#documentation) · [Contributing](#contributing) · [Citation](#citation)

## Installation

This README documents the **main branch (version 1.0.0)**. Install its source packages
in dependency order with Python 3.10 or newer:

```shell
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@main"
```

From a local stack checkout, use `python -m pip install ./purin`.
PURIN does not depend on OHAGI or any frontend package.

## Basic API

Project a point onto the affine equality **u + v = b**:

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

Prepare once and reuse the layer for its configured shapes and dtype. The same
problem interface accepts other registered solver methods when their constraint
requirements are satisfied.

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

## The HardNet stack

- **[KKTHN](https://github.com/SOULS-TAMU/kkt-hardnet/tree/main),
  [DAEHN](https://github.com/SOULS-TAMU/dae-hardnet/tree/main) and
  [NLPOPTNet](https://github.com/SOULS-TAMU/nlpoptnet)** supply problem formulations
  and framework-specific processing.
- **[OHAGI](https://github.com/SOULS-TAMU/Ohagi/tree/main)** composes models,
  runs the shared learning engine, and handles checkpoints and JAX deployment.
- **[PURIN](https://github.com/SOULS-TAMU/Purin/tree/main)** owns projection
  problems, numerical solvers, sparse kernels and projection differentiation.

## Compatibility and validation

The CPU CI matrix covers **Python 3.10–3.14** on Linux, Windows and macOS,
plus Linux ARM64 on Python 3.13. It exercises current compatible JAX releases
and JAX 0.6.2 on Python 3.12. See the [workflow](.github/workflows/ci.yml)
for exact environments; accelerator installation follows the
[JAX guide](https://docs.jax.dev/en/latest/installation.html).

- **CI:** tests, distribution builds and installation checks.
- **Validation:** public APIs, numerical references, documentation and code quality.
- **Coverage:** at least **90% combined statement and branch coverage**; the badge
  links to the measured report for this branch.
- **Security:** CodeQL analysis and dependency review.

## Documentation

**[Read the full documentation](https://purin.readthedocs.io/en/latest/)** — installation, tutorials, user guide and API reference.

The full Sphinx guide follows the Read the Docs style: [contents](docs/index.rst),
[runnable examples](docs/examples/index.rst), [API reference](docs/api/index.rst)
and [build instructions](docs/BUILD.md).

| Resource | Contents |
| --- | --- |
| [Projection methods](#projection-methods) | Algorithm names, suitable problems and formulation choices |
| [Contribution guide](CONTRIBUTING.md) | Solver boundaries, numerical validation and packaging |
| [Validation policy](.github/validation/README.md) | API, gradient, sparse-kernel and quality checks |
| [Changelog](CHANGELOG.md) | Initial release scope and changes |

## Contributing

Bug reports, examples and focused pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, numerical checks and compatibility
requirements. Keep generated results, environments and checkpoints out of Git.

## Citation

Use [CITATION.bib](CITATION.bib) for a copyable BibTeX entry or GitHub's
**Cite this repository** menu for [citation metadata](CITATION.cff).

## License

[Apache License 2.0](LICENSE). Authorship and upstream attribution are recorded
in [AUTHORS](AUTHORS) and [NOTICE](NOTICE).

[Changelog](CHANGELOG.md) · [Report an issue](https://github.com/SOULS-TAMU/Purin/issues)
