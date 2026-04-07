"""
Inference Script — CSV Cleaner Environment
=============================================
Baseline agent using OpenAI client to clean CSV datasets across 3 tasks.

MANDATORY ENV VARS:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
  IMAGE_NAME     Docker image name (if using from_docker_image)

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from csv_cleaner_env import CsvCleanerEnv

IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")  # default allowed
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")           # default allowed
HF_TOKEN     = os.getenv("HF_TOKEN")                                           # NO default — injected by validator

BENCHMARK   = os.getenv("CSV_CLEANER_BENCHMARK", "csv_cleaner_env")
TEMPERATURE = 0.3
MAX_TOKENS  = 300

# Debug print to confirm env vars are loaded
print(f"[CONFIG] API_BASE_URL={API_BASE_URL} MODEL={MODEL_NAME} HF_TOKEN={'SET' if HF_TOKEN else 'NOT SET'}", flush=True)

# Task configurations
TASKS = [
    {"name": "fix_column_types",          "max_steps": 10},
    {"name": "clean_missing_duplicates",  "max_steps": 15},
    {"name": "full_pipeline",             "max_steps": 20},
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data cleaning agent. You interact with a CSV dataset through structured tool calls.

    Available tools:
    - get_dataset_info(): See current columns, types, null counts, samples
    - rename_column(old_name, new_name): Rename a column
    - cast_column(column, dtype): Cast column to int/float/str/datetime
    - fill_missing(column, strategy, value): Fill nulls. strategy: mean/median/mode/constant/zero
    - drop_missing(column): Drop rows with nulls (empty string for all columns)
    - drop_duplicates(columns): Remove duplicates (empty string for all columns)
    - filter_rows(column, operator, value): Filter rows. operator: ==/!=/>/</contains
    - strip_whitespace(column): Strip whitespace from string column
    - replace_values(column, old_value, new_value): Replace values in column

    You must respond with EXACTLY ONE tool call per turn as a JSON object:
    {"tool": "<tool_name>", "args": {"param1": "value1", ...}}

    Read the task description carefully and execute the cleaning steps one at a time.
    Start by calling get_dataset_info to understand the current state, then fix issues.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON tool call from model response."""
    text = text.strip()
    for start_char in ["{"]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue
    return None


def get_model_response(
    client: OpenAI,
    task_desc: str,
    dataset_info: str,
    last_result: str,
    step: int,
    history: List[str],
) -> Optional[Dict[str, Any]]:
    """Get next tool call from the model."""
    history_block = "\n".join(history[-6:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
        Task: {task_desc}

        Current Step: {step}
        Last Action Result: {last_result}

        Current Dataset State:
        {dataset_info}

        Previous Actions:
        {history_block}

        Respond with your next tool call as JSON: {{"tool": "tool_name", "args": {{...}}}}
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_tool_call(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None


async def run_task(client: OpenAI, env: CsvCleanerEnv, task_config: Dict) -> None:
    """Run a single task and log stdout."""
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    try:
        result     = await env.reset(task=task_name)
        metadata   = result.metadata or {}
        task_desc  = metadata.get("task_description", task_name)
        dataset_info = json.dumps(metadata.get("columns", []), indent=2)
        last_result  = metadata.get("last_action_result", "Ready")
        history: List[str] = []

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # First step: always get dataset info
            if step == 1:
                tool_call = {"tool": "get_dataset_info", "args": {}}
            else:
                tool_call = get_model_response(
                    client, task_desc, dataset_info, last_result, step, history
                )
                if tool_call is None:
                    tool_call = {"tool": "get_dataset_info", "args": {}}

            tool_name = tool_call.get("tool", "get_dataset_info")
            tool_args = tool_call.get("args", {})

            try:
                call_result = await env.call_tool(tool_name, **tool_args)
                result_str  = str(call_result) if call_result else ""
            except Exception as e:
                result_str = f"Error: {e}"

            reward = result.reward if hasattr(result, "reward") and result.reward else 0.0
            done   = result.done   if hasattr(result, "done")   else False

            if result.metadata:
                progress     = result.metadata.get("progress", 0.0)
                score        = progress
                dataset_info = json.dumps(result.metadata.get("columns", []), indent=2)
                last_result  = result.metadata.get("last_action_result", result_str)
            else:
                last_result = result_str

            rewards.append(reward)
            steps_taken = step
            action_str  = f"{tool_name}({json.dumps(tool_args)})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_str} -> {last_result[:100]}")

            if done:
                break

        score   = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    # Use HF_TOKEN as the API key — injected by the hackathon validator
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = await CsvCleanerEnv.from_docker_image(IMAGE_NAME)
    try:
        for task_config in TASKS:
            await run_task(client, env, task_config)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())