# Installation

Use Python 3.10 or newer and an isolated environment. The distribution name is
`purin`; the Python import is `purin`.

```shell
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install purin
python -c "import purin; print(purin.__version__)"
```

PyPI resolves the shared-stack dependencies automatically. For development from a
local checkout, install the sibling source packages in dependency order using
`python -m pip install -e <folder>`: PURIN, OHAGI, then the frontend.
Start with CPU execution and choose supported JAX/CUDA dependencies if needed.

PURIN depends on JAX, NumPy and SciPy, not on OHAGI or a frontend.

The production release installation checks cover Python 3.10–3.15 on Linux,
Windows and macOS, plus Linux ARM64. See [publishing and verification](PUBLISHING.md).
