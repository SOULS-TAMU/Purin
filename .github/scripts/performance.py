"""Synchronized synthetic CPU benchmarks; report timings without a noisy speed gate."""

import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import sys
import tempfile
import time
import tracemalloc


def samples(function, argument, jax, repeats=150):
    for _ in range(10):
        jax.block_until_ready(function(argument))
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        jax.block_until_ready(function(argument))
        values.append((time.perf_counter_ns() - start) / 1000)
    return {
        "median_us": statistics.median(values),
        "p95_us": sorted(values)[int(0.95 * (len(values) - 1))],
        "calls": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, default=Path("performance-results.json"))
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from purin import Projection, Distance, ChambollePockConfig

    jax.config.update("jax_enable_x64", True)
    package = "ohagi" if (args.source / "src/ohagi").exists() else "purin"
    metrics = {}
    if package == "ohagi":
        from ohagi.nn.mlp import MLPBackbone
        from ohagi.inference import export, load

        backbone = MLPBackbone((16, 16))
        params = backbone.init(jax.random.PRNGKey(7), 4, 2)

        def graph(x):
            y = backbone.apply(params, x)
            return y - (jnp.sum(y, axis=1, keepdims=True) - 1) / 2

        argument = jnp.ones((1, 4))
        start = time.perf_counter()
        compiled = jax.jit(graph).lower(argument).compile()
        metrics["compile_seconds"] = time.perf_counter() - start
        metrics["compiled_single"] = samples(compiled, argument, jax)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            export(path, graph, input_size=4, output_size=2)
            start = time.perf_counter()
            predictor = load(path)
            metrics["load_and_warmup_seconds"] = time.perf_counter() - start
            expected = np.asarray(graph(argument))[0]
            np.testing.assert_allclose(
                predictor.predict(np.ones(4)), expected, atol=1e-12
            )
            metrics["loaded_device_single"] = samples(
                predictor.predict_jax, jnp.ones(4), jax
            )
            metrics["loaded_host_single"] = samples(predictor.predict, np.ones(4), jax)
        metrics["batch_16"] = samples(jax.jit(graph), jnp.ones((16, 4)), jax)
        metrics["batch_16"]["instances_per_second"] = (
            16e6 / metrics["batch_16"]["median_us"]
        )
    else:
        problem = Projection()
        p = problem.add_parameter(["b"])
        y = problem.add_variable(["u", "v"])
        problem.objective = 0.0
        problem.constraints.add(y.u + y.v == p.b)
        for batch in (1, 16):
            context = jnp.ones((batch, 1))
            argument = jnp.tile(jnp.array([[0.8, 0.7]]), (batch, 1))
            for method in ("linear-equality", "chambolle-pock"):
                config = (
                    ChambollePockConfig(max_iterations=1000, tolerance=1e-9)
                    if method == "chambolle-pock"
                    else None
                )
                start = time.perf_counter()
                layer = Distance(problem, method, config).prepare(batch_size=batch)
                compiled = (
                    jax.jit(lambda x: layer(x, context)).lower(argument).compile()
                )
                metrics[f"{method}/batch_{batch}/compile_seconds"] = (
                    time.perf_counter() - start
                )
                np.testing.assert_allclose(
                    compiled(argument), np.tile([[0.55, 0.45]], (batch, 1)), atol=1e-7
                )
                measurements = samples(compiled, argument, jax)
                measurements["instances_per_second"] = (
                    batch * 1e6 / measurements["median_us"]
                )
                metrics[f"{method}/batch_{batch}"] = measurements
    # Measure Python allocation separately so instrumentation does not distort timings.
    tracemalloc.start()
    jax.block_until_ready(compiled(argument))
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        rss = None
    report = {
        "workload": "Synthetic CPU baseline; not an application/MPC benchmark",
        "package": package,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "python_peak_bytes_for_one_compiled_call": python_peak,
        "process_peak_rss_bytes": rss,
        "metrics": metrics,
    }
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        report["median_ratios_current_over_base"] = {
            name: value["median_us"] / baseline["metrics"][name]["median_us"]
            for name, value in metrics.items()
            if isinstance(value, dict) and name in baseline["metrics"]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(
                "## Performance observation\n\nSynthetic, synchronized CPU workloads; timings are informational.\n\n```json\n"
                + json.dumps(report, indent=2)
                + "\n```\n"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
