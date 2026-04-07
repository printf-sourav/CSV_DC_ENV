"""
CSV Cleaning Environment Implementation.

A real-world data cleaning environment where an AI agent must clean messy CSV
datasets using structured commands. Exposes cleaning tools through MCP.

Supported tools:
- rename_column(old_name, new_name)
- cast_column(column, dtype)
- fill_missing(column, strategy, value?)
- drop_missing(column?)
- drop_duplicates(columns?)
- filter_rows(column, operator, value)
- strip_whitespace(column)
- replace_values(column, old_value, new_value)
"""

import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

from .tasks import TASKS, TaskDefinition


class CsvCleaningEnvironment(MCPEnvironment):
    """
    A data cleaning environment where agents fix messy CSV data.

    The environment generates a messy dataset for the selected task.
    Each step, the agent issues a cleaning command via MCP tools.
    The environment applies the command, updates the dataset, and
    returns reward based on progress toward the target clean dataset.
    """

    def __init__(self):
        """Initialize with MCP server and cleaning tools."""
        mcp = FastMCP("csv_cleaner_env")
        self._df: Optional[pd.DataFrame] = None
        self._target: Optional[pd.DataFrame] = None
        self._task: Optional[TaskDefinition] = None
        self._last_result: str = ""
        self._prev_score: float = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._env_ref = self  # capture for closures

        # ---- MCP Tools ----

        @mcp.tool
        def rename_column(old_name: str, new_name: str) -> str:
            """Rename a column in the dataset."""
            return self._exec_rename_column(old_name, new_name)

        @mcp.tool
        def cast_column(column: str, dtype: str) -> str:
            """Cast a column to a new type. dtype: int, float, str, datetime."""
            return self._exec_cast_column(column, dtype)

        @mcp.tool
        def fill_missing(column: str, strategy: str, value: str = "") -> str:
            """Fill missing values. strategy: mean, median, mode, constant. value used if strategy=constant."""
            return self._exec_fill_missing(column, strategy, value)

        @mcp.tool
        def drop_missing(column: str = "") -> str:
            """Drop rows with missing values. If column empty, drops rows with any null."""
            return self._exec_drop_missing(column)

        @mcp.tool
        def drop_duplicates(columns: str = "") -> str:
            """Remove duplicate rows. columns: comma-separated list or empty for all."""
            return self._exec_drop_duplicates(columns)

        @mcp.tool
        def filter_rows(column: str, operator: str, value: str) -> str:
            """Filter rows. operator: ==, !=, >, <, >=, <=, contains."""
            return self._exec_filter_rows(column, operator, value)

        @mcp.tool
        def strip_whitespace(column: str) -> str:
            """Strip leading/trailing whitespace from a string column."""
            return self._exec_strip_whitespace(column)

        @mcp.tool
        def replace_values(column: str, old_value: str, new_value: str) -> str:
            """Replace occurrences of old_value with new_value in a column."""
            return self._exec_replace_values(column, old_value, new_value)

        @mcp.tool
        def get_dataset_info() -> str:
            """Get current dataset info: columns, types, null counts, sample rows."""
            return self._exec_get_info()

        super().__init__(mcp)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _exec_rename_column(self, old_name: str, new_name: str) -> str:
        if self._df is None:
            return "Error: No dataset loaded. Call reset() first."
        if old_name not in self._df.columns:
            self._last_result = f"Error: Column '{old_name}' not found. Available: {list(self._df.columns)}"
            return self._last_result
        self._df = self._df.rename(columns={old_name: new_name})
        self._last_result = f"Renamed '{old_name}' to '{new_name}'"
        return self._last_result

    def _exec_cast_column(self, column: str, dtype: str) -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        if column not in self._df.columns:
            self._last_result = f"Error: Column '{column}' not found."
            return self._last_result
        try:
            if dtype == "int":
                self._df[column] = pd.to_numeric(self._df[column], errors="coerce").astype("Int64")
            elif dtype == "float":
                self._df[column] = pd.to_numeric(self._df[column].astype(str).str.replace("$", "", regex=False), errors="coerce")
            elif dtype == "str":
                self._df[column] = self._df[column].astype(str)
            elif dtype in ("datetime", "date"):
                self._df[column] = pd.to_datetime(self._df[column], errors="coerce")
            else:
                self._last_result = f"Error: Unknown dtype '{dtype}'. Use: int, float, str, datetime."
                return self._last_result
            self._last_result = f"Cast '{column}' to {dtype}"
        except Exception as e:
            self._last_result = f"Error casting '{column}' to {dtype}: {e}"
        return self._last_result

    def _exec_fill_missing(self, column: str, strategy: str, value: str = "") -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        if column not in self._df.columns:
            self._last_result = f"Error: Column '{column}' not found."
            return self._last_result
        try:
            null_before = int(self._df[column].isnull().sum())
            if strategy == "mean":
                fill_val = self._df[column].mean()
                self._df[column] = self._df[column].fillna(fill_val)
            elif strategy == "median":
                fill_val = self._df[column].median()
                self._df[column] = self._df[column].fillna(fill_val)
            elif strategy == "mode":
                mode_vals = self._df[column].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else ""
                self._df[column] = self._df[column].fillna(fill_val)
            elif strategy == "constant":
                self._df[column] = self._df[column].fillna(value)
            elif strategy == "zero":
                self._df[column] = self._df[column].fillna(0)
            else:
                self._last_result = f"Error: Unknown strategy '{strategy}'. Use: mean, median, mode, constant, zero."
                return self._last_result
            null_after = int(self._df[column].isnull().sum())
            self._last_result = f"Filled {null_before - null_after} nulls in '{column}' using {strategy}"
        except Exception as e:
            self._last_result = f"Error filling missing in '{column}': {e}"
        return self._last_result

    def _exec_drop_missing(self, column: str = "") -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        before = len(self._df)
        try:
            if column and column in self._df.columns:
                self._df = self._df.dropna(subset=[column]).reset_index(drop=True)
            else:
                self._df = self._df.dropna().reset_index(drop=True)
            after = len(self._df)
            self._last_result = f"Dropped {before - after} rows with missing values"
        except Exception as e:
            self._last_result = f"Error dropping missing: {e}"
        return self._last_result

    def _exec_drop_duplicates(self, columns: str = "") -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        before = len(self._df)
        try:
            if columns:
                col_list = [c.strip() for c in columns.split(",")]
                valid_cols = [c for c in col_list if c in self._df.columns]
                if valid_cols:
                    self._df = self._df.drop_duplicates(subset=valid_cols).reset_index(drop=True)
                else:
                    self._last_result = f"Error: None of {col_list} found in columns."
                    return self._last_result
            else:
                self._df = self._df.drop_duplicates().reset_index(drop=True)
            after = len(self._df)
            self._last_result = f"Removed {before - after} duplicate rows"
        except Exception as e:
            self._last_result = f"Error removing duplicates: {e}"
        return self._last_result

    def _exec_filter_rows(self, column: str, operator: str, value: str) -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        if column not in self._df.columns:
            self._last_result = f"Error: Column '{column}' not found."
            return self._last_result
        before = len(self._df)
        try:
            col_data = self._df[column]
            if operator == "==":
                mask = col_data.astype(str) == value
            elif operator == "!=":
                mask = col_data.astype(str) != value
            elif operator == ">":
                mask = pd.to_numeric(col_data, errors="coerce") > float(value)
            elif operator == "<":
                mask = pd.to_numeric(col_data, errors="coerce") < float(value)
            elif operator == ">=":
                mask = pd.to_numeric(col_data, errors="coerce") >= float(value)
            elif operator == "<=":
                mask = pd.to_numeric(col_data, errors="coerce") <= float(value)
            elif operator == "contains":
                mask = col_data.astype(str).str.contains(value, na=False)
            else:
                self._last_result = f"Error: Unknown operator '{operator}'."
                return self._last_result
            self._df = self._df[mask].reset_index(drop=True)
            after = len(self._df)
            self._last_result = f"Filtered: kept {after} rows ({before - after} removed)"
        except Exception as e:
            self._last_result = f"Error filtering: {e}"
        return self._last_result

    def _exec_strip_whitespace(self, column: str) -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        if column not in self._df.columns:
            self._last_result = f"Error: Column '{column}' not found."
            return self._last_result
        try:
            self._df[column] = self._df[column].astype(str).str.strip()
            self._last_result = f"Stripped whitespace from '{column}'"
        except Exception as e:
            self._last_result = f"Error stripping whitespace: {e}"
        return self._last_result

    def _exec_replace_values(self, column: str, old_value: str, new_value: str) -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        if column not in self._df.columns:
            self._last_result = f"Error: Column '{column}' not found."
            return self._last_result
        try:
            count = int((self._df[column].astype(str) == old_value).sum())
            self._df[column] = self._df[column].astype(str).str.replace(old_value, new_value, regex=False)
            self._last_result = f"Replaced {count} occurrences of '{old_value}' with '{new_value}' in '{column}'"
        except Exception as e:
            self._last_result = f"Error replacing values: {e}"
        return self._last_result

    def _exec_get_info(self) -> str:
        if self._df is None:
            return "Error: No dataset loaded."
        obs_data = self._get_observation_dict()
        info = {
            "row_count": obs_data["row_count"],
            "duplicate_count": obs_data["duplicate_count"],
            "columns": obs_data["columns"],
            "task_description": obs_data["task_description"],
            "last_action_result": obs_data["last_action_result"],
            "progress": obs_data["progress"],
        }
        return json.dumps(info, indent=2)

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    def _get_observation_dict(self) -> Dict[str, Any]:
        """Build observation data from current state."""
        if self._df is None:
            return {
                "columns": [],
                "row_count": 0,
                "duplicate_count": 0,
                "task_description": "",
                "last_action_result": self._last_result,
                "progress": 0.0,
            }
        columns_info = []
        for col in self._df.columns:
            columns_info.append({
                "name": col,
                "dtype": str(self._df[col].dtype),
                "null_count": int(self._df[col].isnull().sum()),
                "unique_count": int(self._df[col].nunique()),
                "sample_values": [str(v) for v in self._df[col].dropna().head(3).tolist()],
            })

        progress = 0.0
        if self._task and self._target is not None:
            progress = self._task.grade(self._df, self._target)

        return {
            "columns": columns_info,
            "row_count": len(self._df),
            "duplicate_count": int(self._df.duplicated().sum()),
            "task_description": self._task.description if self._task else "",
            "last_action_result": self._last_result,
            "progress": round(min(max(progress, 0.0), 1.0), 4),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset environment with a messy dataset for the configured task."""
        task_name = kwargs.get("task", os.getenv("CSV_CLEANER_TASK", "fix_column_types"))
        actual_seed = seed if seed is not None else 42

        if task_name not in TASKS:
            available = list(TASKS.keys())
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": f"Unknown task '{task_name}'. Available: {available}"},
            )

        self._task = TASKS[task_name]
        self._df = self._task.generate_messy(actual_seed)
        self._target = self._task.generate_target(actual_seed)
        self._done = False
        self._last_result = "Environment ready. Use get_dataset_info to see the current state."
        self._prev_score = self._task.grade(self._df, self._target)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        obs_data = self._get_observation_dict()
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task": task_name,
                "difficulty": self._task.difficulty,
                "max_steps": self._task.max_steps,
                "checklist": self._task.checklist,
                **obs_data,
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (returns error — use MCP tools instead)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use MCP tools (get_dataset_info, cast_column, fill_missing, etc.)",
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step. Increments step count, computes reward."""
        self._state.step_count += 1

        # Let MCPEnvironment handle tool dispatch
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Compute reward based on progress delta
        reward = 0.0
        done = False
        if self._task and self._target is not None and self._df is not None:
            current_score = self._task.grade(self._df, self._target)
            reward = max(0.0, current_score - self._prev_score)
            self._prev_score = current_score

            # Check if done (target reached or max steps exceeded)
            if current_score >= 0.95:
                done = True
                reward += 0.1  # bonus for completing
            elif self._state.step_count >= self._task.max_steps:
                done = True

            self._done = done

        # Inject our reward/done into the observation
        obs.reward = round(reward, 4)
        obs.done = done
        if obs.metadata is None:
            obs.metadata = {}
        obs.metadata.update(self._get_observation_dict())

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step used by the WebSocket handler."""
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        reward = 0.0
        done = False
        if self._task and self._target is not None and self._df is not None:
            current_score = self._task.grade(self._df, self._target)
            reward = max(0.0, current_score - self._prev_score)
            self._prev_score = current_score
            if current_score >= 0.95:
                done = True
                reward += 0.1
            elif self._state.step_count >= self._task.max_steps:
                done = True
            self._done = done

        obs.reward = round(reward, 4)
        obs.done = done
        if obs.metadata is None:
            obs.metadata = {}
        obs.metadata.update(self._get_observation_dict())
        return obs

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state
