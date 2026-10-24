"""User-facing projection problem specification."""

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping
from purin.problems.problem import Problem


class Projection(Problem):
    """Symbolic constraint system plus one selected projection backend."""

    def __init__(
        self,
        name: str = "projection",
        *,
        method: str = "fischer-burmeister-newton",
        config: Mapping[str, Any] | Any | None = None,
    ) -> None:
        super().__init__(name=name)
        self.method = str(method)
        if config is None:
            self.config: dict[str, Any] = {}
        elif is_dataclass(config):
            self.config = asdict(config)
        elif isinstance(config, Mapping):
            self.config = dict(config)
        else:
            raise TypeError("projection config must be a mapping or dataclass.")

    def solver(self, method: str, **config) -> "Projection":
        """Resolve the solver definition attached to this projection specification."""
        self.method = str(method)
        self.config = dict(config)
        return self
