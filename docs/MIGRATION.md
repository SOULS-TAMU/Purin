# Versions and migration

This guide describes **PURIN 1.0.0** on the current main branch. The previous
source is preserved in the [`v0.1.0` branch](https://github.com/SOULS-TAMU/Purin/tree/v0.1.0).

Problem suppliers (KKTHN/DAEHN), learning execution (OHAGI) and projection
mathematics (PURIN) have separate ownership. Benchmark signatures are retained;
the packages now use a `src/` layout. Use these public API pages rather than
copying removed internal module paths.

## Solver compatibility

Use public problem, projection and solver-configuration interfaces. The six
registered method names are stable entry points; internal filenames are not a
second frontend API. Record solver settings, precision and gradient policy when
comparing old and new runs.
