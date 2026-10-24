"""Use CPU float64 for deterministic numerical regression checks."""

import jax

jax.config.update("jax_enable_x64", True)

# Plotting tests must run without a window or display server.
import os

os.environ.setdefault("MPLBACKEND", "Agg")
