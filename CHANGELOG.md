# Changelog

Notable changes to PURIN are recorded here. This history starts with the
initial public version; internal development steps are not separate releases.
Release dates follow the PyPI publication date in America/Chicago.

## 1.0.2 — 2026-11-13

- Add opt-in Chambolle–Pock iteration counts for dense and sparse projections.
- Check sparse implicit adjoints at cross-platform solver precision.

## 1.0.1 — 2026-11-07

- Add optional structural sparse inequality products to the shared CP forward and implicit adjoint; retain dense equality operations and the existing vector diagonal proximal map. Dense behavior remains the default for library callers.

- Improve repository releases, publishing checks and contribution workflows; runtime APIs and numerical implementations are unchanged.

## 1.0.0 — 2026-11-05

- Promote the shared-stack implementation to `main`; preserve the previous main in `v0.1.0`.
- Align main-branch installation links, badges and citation metadata.

### Added

- Added dense-reference KKT/Pallas-interpreter checks, barrier derivatives,
  sparse adjoint validation and full-state projection pullback tests.

- API, numerical, frontend integration, dependency review, documentation,
  quality regression and informational performance workflows.

- Expanded analytic, dense-reference, sparse-kernel and implicit-gradient tests,
  with an inclusive 90% combined statement-and-branch coverage gate.

- Cross-platform CPU CI for Linux, Windows and Apple Silicon macOS, with
  baseline/current JAX, runtime configuration artifacts and ARM64 coverage.

- GitHub Actions CI for numerical tests, package checks and Ruff, with measured
  branch coverage reporting and scheduled Python CodeQL analysis.

- A standalone JAX library for constrained projections and sparse numerical
  kernels, usable independently of OHAGI and the frontend frameworks.
- Projection problem interfaces and reusable preparation utilities.
- Euclidean distance, diagonal weighted-distance, local quadratic and
  sequential projection interfaces.
- Linear equality, Chambolle–Pock, ADMM, Fischer–Burmeister and primal-dual
  interior point projection methods.
- Sparse operators and projection differentiation utilities, including
  implicit differentiation support.
- Shared projection infrastructure for OHAGI, KKTHN, DAEHN and NLPOPTNet.
- Python packaging, a runnable quick start, contribution guidance and a
  BibTeX software citation.
- Apache-2.0 licensing and attribution notices.
