"""Fixed analytic solver/gradient oracles shared by OHAGI and PURIN CI."""

from dataclasses import replace
import json
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from purin import (
    Distance,
    Projection,
    ChambollePockConfig,
    ADMMConfig,
    FischerBurmeisterConfig,
    ProjectionConfig,
    LinearEqualityConfig,
    ImplicitGradientConfig,
    LinearSolverConfig,
)


def main():
    jax.config.update("jax_enable_x64", True)
    cases = {}
    for dtype in (jnp.float32, jnp.float64):
        tolerance = 5e-5 if dtype == jnp.float32 else 2e-6
        for method in (
            "linear-equality",
            "chambolle-pock",
            "admm",
            "fischer-burmeister-newton",
            "fischer-burmeister-reduced",
            "primal-dual-interior-point",
        ):
            problem = Projection()
            p = problem.add_parameter(["b"])
            y = problem.add_variable(["u", "v"])
            problem.objective = 0.0
            problem.constraints.add(y.u + y.v == p.b)
            if method == "linear-equality":
                config = LinearEqualityConfig()
            elif method == "chambolle-pock":
                config = ChambollePockConfig(
                    max_iterations=3000,
                    tolerance=1e-6 if dtype == jnp.float32 else 1e-10,
                )
            elif method == "admm":
                config = ADMMConfig(
                    max_iterations=3000,
                    tolerance=1e-6 if dtype == jnp.float32 else 1e-10,
                )
            elif method.startswith("fischer"):
                config = FischerBurmeisterConfig(
                    max_iterations=40, tolerance=1e-6 if dtype == jnp.float32 else 1e-10
                )
            else:
                config = ProjectionConfig(
                    max_iterations=40, tolerance=1e-6 if dtype == jnp.float32 else 1e-10
                )
            if dtype == jnp.float32 and method != "linear-equality":
                settings = {
                    "implicit_gradient": ImplicitGradientConfig(
                        adjoint_tolerance=1e-5, inexact_tolerance=1e-5
                    )
                }
                if hasattr(config, "linear_solver"):
                    settings["linear_solver"] = LinearSolverConfig(tolerance=1e-6)
                config = replace(config, **settings)
            layer = Distance(problem, method, config).prepare(batch_size=1, dtype=dtype)
            target = jnp.array([[0.8, 0.7]], dtype=dtype)
            context = jnp.ones((1, 1), dtype=dtype)
            actual = layer(target, context)
            np.testing.assert_allclose(actual, [[0.55, 0.45]], atol=tolerance, rtol=0)
            jacobian = jax.jacrev(lambda x: layer(x[None], context)[0])(target[0])
            np.testing.assert_allclose(
                jacobian,
                [[0.5, -0.5], [-0.5, 0.5]],
                atol=tolerance,
                rtol=0,
                err_msg=f"{dtype}/{method}",
            )
            cases[f"{np.dtype(dtype).name}/{method}"] = {
                "values": np.asarray(actual).tolist(),
                "jacobian": np.asarray(jacobian).tolist(),
                "equality_residual": float(jnp.max(abs(actual.sum(1) - 1))),
            }
    root = Path(__file__).resolve().parents[2]
    if (root / "src/ohagi").exists():
        from ohagi.inference import load
        from ohagi.serialization.bundle import save_model, load_parameters

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            save_model(
                path,
                framework="custom",
                parameters={"weights": jnp.eye(2)},
                inference_spec=dict(
                    predict_fn=lambda x: 2 * x,
                    input_size=2,
                    output_size=2,
                    dtype="float64",
                ),
            )
            data = json.loads(path.read_text())
            schema = json.loads(
                (root / ".github/validation/metadata-schema.json").read_text()
            )
            for section, required in schema["required"].items():
                value = data
                for key in section.split(".") if section else []:
                    value = value[key]
                assert set(required) <= set(value), (section, required, value)
            assert data["format"] == schema["formats"]["bundle"]
            assert data["parameters"]["format"] == schema["formats"]["parameters"]
            assert data["inference"]["format"] == schema["formats"]["inference"]
            assert (
                data["inference"]["portable"]["format"] == schema["formats"]["program"]
            )
            np.testing.assert_array_equal(load_parameters(path)["weights"], np.eye(2))
            predictor = load(path)
            for _ in range(25):
                np.testing.assert_array_equal(predictor([1.0, 2.0]), [2.0, 4.0])
            data["format"] = "unsupported"
            path.write_text(json.dumps(data))
            try:
                load(path)
            except ValueError:
                pass
            else:
                raise AssertionError("Unknown schema accepted")
        cases["metadata"] = "schema and repeated inference passed"
    (root / "numerical-results.json").write_text(
        json.dumps(cases, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Passed {len(cases)} independent numerical/schema cases.")


if __name__ == "__main__":
    main()
