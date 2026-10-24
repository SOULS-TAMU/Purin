# Contributing to PURIN

Thank you for helping improve PURIN. Bug reports, documentation fixes,
examples and numerical improvements are welcome. This guide accompanies
the first public version of the project.

## Issues and proposals

For small fixes, open a focused
[pull request](https://github.com/SOULS-TAMU/Purin/pulls). Discuss new solvers,
public APIs, dependencies or changes to numerical behavior in an
[issue](https://github.com/SOULS-TAMU/Purin/issues) first.

For a bug report, provide a minimal reproducer, expected and observed
behavior, package versions, device, dtype, random seed and solver settings.
Include the problem dimensions and constraint residuals when relevant.
Remove private data and credentials from examples and logs.

## Development setup

Use Python 3.10 or newer in an isolated environment. Fork the repository
on GitHub for contributions, then clone your fork. For a local checkout of
the upstream source:

```sh
git clone https://github.com/SOULS-TAMU/Purin.git
cd Purin
python -m pip install -e ".[build]"
python -m pip install pytest ruff
```

Create a focused branch for your changes and submit the pull request against
`main`.

Follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for your device. Install OHAGI and the relevant frontend in editable mode
only when checking their integration. PURIN itself must remain usable
without those packages.

## Library boundaries

PURIN owns projection problems, composition, solvers, sparse operators and
projection differentiation. Reuse its canonical solver implementations.
Keep training, neural network construction, checkpointing and model export
in OHAGI; keep framework-specific formulation policies in the frontends.

Do not introduce imports from OHAGI, KKTHN, DAEHN or NLPOPTNet into PURIN.
Avoid duplicating numerical kernels to support a new adapter.

## Code and documentation

Use descriptive names, focused functions and Google-style docstrings for
public interfaces. Document shapes, axis order, coordinate conventions,
constraint signs, metric assumptions and gradient behavior where relevant.
Explain mathematical choices and invariants rather than restating Python.

Run these checks from the PURIN directory:

```sh
python -m ruff format --check src
python -m ruff check --select F,E9 src
```

Use `python -m ruff format src` to apply formatting. Keep unrelated
formatting changes separate and leave benchmark scripts unchanged unless
the proposed change specifically concerns them.

## Numerical validation

Include focused regression tests or a reproducible validation script with
the pull request. Record commands and results. Run the README quick start
when changing the public projection workflow.

| Change | Validation |
| --- | --- |
| Solver or projection | Compare projected values, feasibility residuals, objective values and available convergence diagnostics. |
| Differentiation | Compare parameter and input gradients with the previous implementation; use an independent check on suitable smooth problems. |
| Sparse kernels | Compare with an equivalent small dense calculation and exercise supported shapes and dtypes. |
| Public interfaces | Check signatures, configuration defaults and the relevant unchanged frontend benchmarks. |
| Restructuring | Compare outputs and gradients before and after the change under identical settings. |

Use identical inputs, seeds, devices, JAX versions, precision and solver
settings for comparisons. State tolerances explicitly and investigate any
difference. Cover applicable boundary cases, including empty constraint
sets, degenerate systems, iteration limits and warm starts. At nonsmooth
points, check the documented differentiation convention rather than
assuming a unique derivative.

Separate intentional numerical changes from restructuring. For timing
comparisons, warm up compilation and synchronize device execution.

## Packaging

From the PURIN directory:

```sh
python -m build
python -m twine check dist/*
```

Install the built wheel in a separate environment and run the quick start.
Keep environments, caches, build output, generated results and model
artifacts out of version control.

## Pull request checklist

- Explain the problem, resulting behavior and compatibility implications.
- Keep changes focused and preserve public APIs and benchmark call signatures.
- Include reproducible numerical checks and explicit tolerances.
- Update documentation and the appropriate unreleased changelog entry.
- Preserve the dependency boundary between PURIN and its consumers.

Contributions are provided under the project's [Apache-2.0 license](LICENSE).
Retain relevant attribution in [AUTHORS](AUTHORS) and [NOTICE](NOTICE).
Keep [CITATION.bib](CITATION.bib) and [CITATION.cff](CITATION.cff) consistent.
The CFF metadata supports GitHub's citation panel; BibTeX remains available directly.
