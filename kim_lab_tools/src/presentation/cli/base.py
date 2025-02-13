"""
# src/kim_lab_tools/cli/base.py

Base classes for CLI commands.
"""

import click
from typing import Any
from pathlib import Path


class BaseCommand:
    """Base class for CLI commands."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def validate_inputs(self) -> None:
        """Validate command inputs."""
        raise NotImplementedError

    def execute(self) -> Any:
        """Execute the command."""
        raise NotImplementedError
