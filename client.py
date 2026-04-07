"""
CSV Cleaner Environment Client.

Provides the client for connecting to a CSV Cleaner Environment server.
CsvCleanerEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with CsvCleanerEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("cast_column", column="age", dtype="int")
    ...     print(result)

Example with Docker:
    >>> env = CsvCleanerEnv.from_docker_image("csv-cleaner-env:latest")
    >>> try:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("fill_missing", column="salary", strategy="mean")
    ... finally:
    ...     env.close()
"""

from openenv.core.mcp_client import MCPToolClient


class CsvCleanerEnv(MCPToolClient):
    """
    Client for the CSV Cleaner Environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available cleaning tools
    - call_tool(name, **kwargs): Call a cleaning tool by name
    - reset(**kwargs): Reset with a new messy dataset
    - step(action): Execute a cleaning action
    """

    pass
