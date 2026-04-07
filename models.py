"""
Data models for the CSV Cleaner Environment.

The CSV Cleaner environment simulates real-world data cleaning tasks
where an AI agent must clean messy CSV datasets using structured commands.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class CsvCleanerAction(Action):
    """Action for the CSV Cleaner environment — a cleaning command with parameters."""

    command: str = Field(
        ...,
        description=(
            "Cleaning command to execute. One of: rename_column, cast_column, "
            "fill_missing, drop_missing, drop_duplicates, filter_rows, "
            "strip_whitespace, replace_values"
        ),
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Command-specific parameters (see README for each command's params)",
    )


class CsvCleanerObservation(Observation):
    """Observation from the CSV Cleaner environment — current dataset state."""

    columns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Column metadata: name, dtype, null_count, unique_count, sample_values",
    )
    row_count: int = Field(default=0, ge=0, description="Current number of rows")
    duplicate_count: int = Field(default=0, ge=0, description="Number of duplicate rows")
    task_description: str = Field(default="", description="Description of the cleaning objective")
    last_action_result: str = Field(default="", description="Result of the last action (success/error)")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress toward target (0.0-1.0)")
