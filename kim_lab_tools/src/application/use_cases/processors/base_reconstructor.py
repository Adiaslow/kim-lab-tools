"""
# src/kim_lab_tools/reconstructors/base_reconstructor.py

Base classes for reconstructors.
"""

from ..core.interfaces import Reconstructor
from typing import Any


class BaseReconstructor(Reconstructor):
    """Base class for all reconstructors."""

    def reconstruct(self, data: Any) -> Any:
        """Reconstruct data."""
        self._validate_data(data)
        return self._reconstruct_implementation(data)

    def _validate_data(self, data: Any) -> None:
        """Validate input data."""
        raise NotImplementedError

    def _reconstruct_implementation(self, data: Any) -> Any:
        """Implementation of reconstruction logic."""
        raise NotImplementedError
