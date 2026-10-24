# Changelog

Notable changes to PURIN are recorded here. This history starts with the
initial public version; internal development steps are not separate releases.
The first GitHub publication contains the initial source distribution.
Version 0.1.0 remains unreleased until a release is explicitly published;
no release date is assigned.

## 0.1.0 — Unreleased

### Added

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
