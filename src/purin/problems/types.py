"""Backend-neutral named value containers used by the public frontend."""

from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
import numpy as np
from purin._utils.validation import strict_zip

ArrayLike = Any


class NamedValues(Mapping[str, float]):
    """Read-only mapping with attribute access for named numerical values."""

    def __init__(self, names: Sequence[str], values: ArrayLike) -> None:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != len(names):
            raise ValueError(
                f"Expected {len(names)} values for {list(names)}; received {vector.size}."
            )
        self._data = dict(strict_zip(names, vector.tolist()))

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> float:
        try:
            return self._data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def vector(self) -> np.ndarray:
        """Pack named values into a vector using the declared name order."""
        return np.asarray(list(self._data.values()), dtype=np.float64)

    def __repr__(self) -> str:
        return f"NamedValues({self._data!r})"


class BatchNamedValues(Mapping[str, np.ndarray]):
    """Read-only named columns for a batch of numerical values."""

    def __init__(self, names: Sequence[str], values: ArrayLike) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError(
                "Expected a matrix with shape (batch, number_of_names); "
                f"received {matrix.shape} for {len(names)} names."
            )
        self._names = tuple(names)
        self._matrix = matrix
        self._index = {name: index for index, name in enumerate(self._names)}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._matrix[:, self._index[key]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def batch_size(self) -> int:
        """Return the number of instances represented by these named arrays."""
        return int(self._matrix.shape[0])

    def matrix(self, *, copy: bool = True) -> np.ndarray:
        """Return the batch matrix, optionally as a zero-copy read-only view."""
        if copy:
            return self._matrix.copy()
        view = self._matrix.view()
        view.flags.writeable = False
        return view

    def vector(self) -> np.ndarray:
        """Return the full ``(batch, width)`` matrix for API symmetry."""
        return self.matrix()

    def row(self, index: int) -> NamedValues:
        """Select one instance while preserving the declared symbol names."""
        return NamedValues(self._names, self._matrix[index])

    def __repr__(self) -> str:
        return f"BatchNamedValues(names={self._names!r}, shape={self._matrix.shape!r})"
