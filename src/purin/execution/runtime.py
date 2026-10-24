"""Explicit JAX runtime configuration and ahead-of-time compilation helpers."""

from __future__ import annotations
from pathlib import Path
import jax


def configure_cache(path):
    """Configure JAX's persistent cache when explicitly requested."""
    if path is None:
        return
    cache_path = Path(path).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def select_device(backend="cpu"):
    """Resolve a device using the shared runtime implementation."""
    return resolve_device(backend)


def aot_compile(function, *example_args, donate_argnums=()):
    """Lower and compile a function through an explicit host-side call."""
    return (
        jax.jit(function, donate_argnums=tuple(donate_argnums))
        .lower(*example_args)
        .compile()
    )


def resolve_device(device=None):
    """Resolve CPU/GPU/CUDA aliases to one concrete local JAX device."""
    if device is None:
        return None
    if hasattr(device, "platform") and hasattr(device, "id"):
        return device
    requested_text = str(device).strip().lower()
    platform_text, separator, device_id_text = requested_text.partition(":")
    requested_platform = {"cuda": "gpu", "nvidia": "gpu"}.get(
        platform_text, platform_text
    )
    if requested_platform not in {"cpu", "gpu", "tpu"}:
        raise ValueError(
            f"Unknown JAX backend/device {device!r}; expected cpu, gpu, cuda, "
            "gpu:<id>, or cuda:<id>."
        )
    requested_id = None
    if separator:
        try:
            requested_id = int(device_id_text)
        except ValueError as error:
            raise ValueError(f"Device id in {device!r} must be an integer.") from error
    try:
        candidates = list(jax.devices(requested_platform))
    except RuntimeError:
        candidates = []
    if requested_id is not None:
        candidates = [
            candidate for candidate in candidates if candidate.id == requested_id
        ]
    if not candidates:
        visible = [
            (candidate.platform, candidate.device_kind, candidate.id)
            for candidate in jax.devices()
        ]
        extra = ""
        if requested_platform == "gpu":
            extra = (
                " JAX must be installed with NVIDIA GPU support. Native Windows "
                "CUDA execution is not supported by JAX; use Linux or WSL2."
            )
        raise ValueError(
            f"Requested JAX device {device!r} is unavailable; visible devices={visible}."
            + extra
        )
    return candidates[0]


def resolve_jax_dtype(dtype):
    """Resolve a requested floating dtype and reject silent x64 truncation."""
    import jax.numpy as jnp

    aliases = {"fp64": "float64", "64": "float64", "fp32": "float32", "32": "float32"}
    if isinstance(dtype, str):
        dtype = aliases.get(dtype.strip().lower(), dtype.strip().lower())
    resolved = jnp.dtype(dtype)
    if resolved not in {jnp.dtype("float32"), jnp.dtype("float64")}:
        raise ValueError(f"Unsupported floating dtype {dtype!r}.")
    if resolved == jnp.dtype(jnp.float64) and not bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "KKT-HardNet was configured with dtype='float64', but JAX x64 "
            "support is disabled. Restart Python/Jupyter and enable x64 before "
            "importing or training:\n\n"
            "    export JAX_ENABLE_X64=True\n\n"
            "or in the first notebook cell:\n\n"
            "    import os\n"
            "    os.environ['JAX_ENABLE_X64'] = 'True'\n"
            "    import jax\n"
            "    jax.config.update('jax_enable_x64', True)"
        )
    return resolved.type
