"""
CSV Cleaner Environment — A real-world data cleaning environment for OpenEnv.

This environment exposes data cleaning tools through MCP:
- rename_column, cast_column, fill_missing, drop_missing,
  drop_duplicates, filter_rows, strip_whitespace, replace_values

Example:
    >>> from csv_cleaner_env import CsvCleanerEnv
    >>>
    >>> with CsvCleanerEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("cast_column", column="age", dtype="int")
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import CsvCleanerEnv

__all__ = ["CsvCleanerEnv", "CallToolAction", "ListToolsAction"]
