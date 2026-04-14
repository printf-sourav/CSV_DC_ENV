CSV Cleaner Environment

A production-style data cleaning environment where AI agents learn to transform messy CSV datasets using structured actions. Built on the OpenEnv framework.

🚀 Overview

Data cleaning is one of the most critical and time-consuming tasks in real-world data workflows. This environment simulates practical data wrangling scenarios, enabling AI agents to:

Correct column data types
Handle missing values
Remove duplicates
Normalize and rename columns
Filter invalid or inconsistent rows

The goal is to train agents to perform step-by-step, explainable data transformations.

🧠 Environment Mechanics

The agent receives:
A messy CSV dataset
A cleaning objective
At each step:
The agent selects a cleaning action via MCP tools
The environment applies the transformation
The updated dataset state is returned
A reward signal reflects progress toward the target

🛠️ Action Space (MCP Tools)

Tool	Parameters	Description
get_dataset_info	—	Inspect schema, types, null counts, and samples
rename_column	old_name, new_name	Rename a column
cast_column	column, dtype	Convert type (int, float, str, datetime)
fill_missing	column, strategy, value?	Fill nulls (mean, median, mode, constant, zero)
drop_missing	column?	Remove rows with null values
drop_duplicates	columns?	Remove duplicate rows
filter_rows	column, operator, value	Filter rows (==, !=, >, <, >=, <=, contains)
strip_whitespace	column	Trim whitespace
replace_values	column, old_value, new_value	Replace values

📊 Observation Space

Each step returns:

columns → {name, dtype, null_count, unique_count, sample_values}
row_count → Total rows
duplicate_count → Number of duplicates
task_description → Cleaning objective
last_action_result → Success/error feedback
progress → Score from 0.0 → 1.0

🎯 Tasks

Task	Difficulty	Max Steps	Description
fix_column_types	Easy	10	Correct incorrect data types
clean_missing_duplicates	Medium	15	Handle nulls + remove duplicates
full_pipeline	Hard	20	End-to-end cleaning pipeline

🏆 Reward Function

Progressive scoring based on similarity to target dataset:
Column types
Missing values
Duplicates
Row count
Column names
Reward = score delta per step
Completion bonus: +0.1 when progress ≥ 0.95
Final score range: [0.0, 1.0]

⚙️ Setup & Usage

📦 Installation
pip install -e .
▶️ Run Server
uvicorn server.app:app --host 0.0.0.0 --port 8000
🐳 Docker
docker build -t csv-cleaner-env .
docker run -p 8000:8000 csv-cleaner-env
🤖 Run Inference
export HF_TOKEN=your_token_here
export IMAGE_NAME=csv-cleaner-env
python inference.py
🔐 Environment Variables
Variable	Required	Default	Description
API_BASE_URL	No	Hugging Face Router	LLM endpoint
MODEL_NAME	No	Qwen2.5-72B-Instruct	Model
HF_TOKEN	Yes	—	API key
IMAGE_NAME	Yes*	—	Docker image
CSV_CLEANER_TASK	No	fix_column_types	Default task
📈 Baseline Performance
Task	Score	Model
fix_column_types	~0.80	Qwen2.5-72B-Instruct
clean_missing_duplicates	~0.65	Qwen2.5-72B-Instruct
full_pipeline	~0.45	Qwen2.5-72B-Instruct

📁 Project Structure

csv_cleaner_env/
├── models.py
├── client.py
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── inference.py
├── README.md
└── server/
    ├── app.py
    ├── csv_cleaning_environment.py
    └── tasks.py
    
📜 License

MIT License