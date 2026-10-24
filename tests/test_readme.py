"""Keep the documented quick start executable."""

from pathlib import Path
import re

import numpy as np


def test_readme_quick_start():
    text = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    example = re.search(r"```python\n(.*?)```", text, flags=re.S).group(1)
    namespace = {}
    exec(compile(example, "README.md", "exec"), namespace)
    np.testing.assert_allclose(namespace["projected"], [[1.5, -0.5]], atol=1e-12)
