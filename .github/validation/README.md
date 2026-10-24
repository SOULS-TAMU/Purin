# Additional validation workflows

`validation.yml` runs on pull requests, pushes to main, manual dispatch and a
weekly schedule. `dependency-review.yml` separately checks newly introduced
dependencies for moderate-or-higher known vulnerabilities. These workflows use
read-only repository permissions and do not post pull-request comments.

| Check | Policy |
| --- | --- |
| Public API | Compare existing public functions, methods, fields and lazy exports against `api.json`; additions are allowed. |
| Numerical references | Check all six solver methods against analytic values and Jacobians in float32/float64. Float32 uses solver/adjoint tolerances appropriate to its precision. |
| Frontend integration | Train frozen KKTHN, DAEHN and NLPOPTNet consumers briefly; test differential processing, fixed/accelerated CP, full-model loading and fresh-process inference exports. |
| Cognitive complexity | Complexipy 8.1.0; new functions must score at most 15; existing functions cannot exceed their recorded score or 15. |
| Duplication | jscpd 5.4.2, exact clones of at least 80 tokens and six lines; no new clone fingerprints or increased duplicated-line count. |
| Type checking | mypy 2.4.0, Python 3.12/Linux target, imports skipped; no new error fingerprints or increased counts. This is a regression baseline, not a clean type-check claim. |
| Dependency boundaries | OHAGI cannot import frontends; PURIN cannot import OHAGI/frontends; no new static import cycles. Static cycles include deferred and type-only imports. |
| Documentation | Validate local files/headings, parse BibTeX/CFF, build downloadable HTML, and check external content links with retries. Dynamic status badges are excluded; HTTP 429 is treated as rate limiting. |
| Performance | Compare base and candidate source in separate processes on the same runner/dependencies. Record compilation, synchronized singleton latency, batch throughput and memory; timing ratios are informational. |

The API and quality snapshots record the current state, including existing type
errors, complex functions and static cycles. Improvements are welcome. Baseline
updates require explicit maintainer review and a compatibility explanation; CI
never updates them automatically. They are not substitutes for numerical tests.

Performance workloads are small synthetic CPU cases, not application/MPC
benchmarks. The report distinguishes device-array and host-array inference,
keeps loading separate, and records Python allocation and process peak memory.
Shared-runner timings do not justify a strict latency regression threshold.
Artifacts retain both base/candidate observations for 90 days. Pull requests use
the base SHA; pushes use the previous SHA; scheduled/manual runs compare the same
source to show runner noise.

To reproduce checks, install `.github/validation/requirements.txt`, run
`python .github/scripts/api_contract.py`, then generate the jscpd JSON report
with the command in the workflow and run
`python .github/scripts/quality_metrics.py quality-duplication/jscpd-report.json`.
Run `documentation.py`, `numerical_regression.py` and `performance.py` from the
repository root with the package/dependencies installed. The frontend script
lives in OHAGI and validates fixture hashes before running.
