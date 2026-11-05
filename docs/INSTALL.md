# Installation

Use Python 3.10 or newer and an isolated environment. The distribution name is
`purin`; the Python import is `purin`.

```shell
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@main"
python -c "import purin; print(purin.__version__)"
```

These source commands select the current shared-stack implementation. Package-index
publishing is a separate release step. For a local editable installation, install
the sibling packages in dependency order using `python -m pip install -e <folder>`.
Start with CPU execution and choose supported JAX/CUDA dependencies if needed.

PURIN depends on JAX, NumPy and SciPy, not on OHAGI or a frontend.
