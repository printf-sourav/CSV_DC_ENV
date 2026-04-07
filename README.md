# CSV Cleaner Environment

A real-world **data cleaning** OpenEnv environment where AI agents learn to clean messy CSV datasets through structured commands. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Motivation

Data cleaning is one of the most common and time-consuming tasks in real-world data work. This environment trains AI agents to perform systematic data wrangling — fixing column types, handling missing values, removing duplicates, renaming columns, and filtering invalid rows — simulating tasks that data engineers and analysts do daily.

## Environment Description

The agent receives a messy CSV dataset and a cleaning objective. Each step, the agent issues one cleaning command via MCP tools. The environment applies the command, returns the updated dataset state, and provides progressive reward based on how close the dataset is to the target clean version.

## Action Space (MCP Tools)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_dataset_info` | — | View columns, types, null counts, sample values |
| `rename_column` | `old_name`, `new_name` | Rename a column |
| `cast_column` | `column`, `dtype` | Cast to `int`, `float`, `str`, `datetime` |
| `fill_missing` | `column`, `strategy`, `value?` | Fill nulls. Strategy: `mean`, `median`, `mode`, `constant`, `zero` |
| `drop_missing` | `column?` | Drop rows with nulls (empty = all columns) |
| `drop_duplicates` | `columns?` | Remove duplicate rows (comma-separated or all) |
| `filter_rows` | `column`, `operator`, `value` | Filter rows. Operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `contains` |
| `strip_whitespace` | `column` | Strip leading/trailing whitespace |
| `replace_values` | `column`, `old_value`, `new_value` | Replace values in a column |

## Observation Space

Each observation includes:
- **columns**: List of `{name, dtype, null_count, unique_count, sample_values}`
- **row_count**: Current number of rows
- **duplicate_count**: Number of duplicate rows
- **task_description**: What needs to be cleaned
- **last_action_result**: Success/error message from last command
- **progress**: Score from 0.0 to 1.0 representing cleaning completion

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `fix_column_types` | Easy | 10 | Fix 4 columns with wrong types (string→int/float/datetime) |
| `clean_missing_duplicates` | Medium | 15 | Fill missing values with appropriate strategies + remove duplicates |
| `full_pipeline` | Hard | 20 | Rename columns, fix types, strip whitespace, fill missing, normalize values, filter invalid rows, remove duplicates |

## Reward Function

- **Progressive**: Each step computes similarity to target dataset (column types, null counts, duplicates, row count, column names)
- **Reward = score_delta**: the improvement in score since the last step
- **Completion bonus**: +0.1 when progress ≥ 0.95
- **Score range**: Final score always in [0.0, 1.0]

## Setup & Usage

### Install
```bash
pip install -e .
```

### Run Server Locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker Build & Run
```bash
docker build -t csv-cleaner-env .
docker run -p 8000:8000 csv-cleaner-env
```

### Run Inference
```bash
export HF_TOKEN=your_token_here
export IMAGE_NAME=csv-cleaner-env
python inference.py
```

### Environment Variables
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | API key |
| `IMAGE_NAME` | Yes* | — | Docker image name (*for inference) |
| `CSV_CLEANER_TASK` | No | `fix_column_types` | Default task |

## Baseline Scores

| Task | Baseline Score | Model |
|------|---------------|-------|
| `fix_column_types` | ~0.80 | Qwen2.5-72B-Instruct |
| `clean_missing_duplicates` | ~0.65 | Qwen2.5-72B-Instruct |
| `full_pipeline` | ~0.45 | Qwen2.5-72B-Instruct |

## Project Structure
```
csv_cleaner_env/
├── __init__.py              # Package exports
├── models.py                # Pydantic Action/Observation models
├── client.py                # MCPToolClient wrapper
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml            # Dependencies
├── Dockerfile               # Container definition
├── inference.py             # Baseline inference script
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI entry point
    ├── csv_cleaning_environment.py  # Core environment
    └── tasks.py             # Task definitions & graders
```

## License

MIT
