"""
Task definitions for the CSV Cleaner Environment.

Each task generates a deterministic messy dataset (given a seed) and defines
a target clean dataset plus a grading function that returns a score in [0, 1].
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd


@dataclass
class TaskDefinition:
    """Definition of a single cleaning task."""

    name: str
    description: str
    difficulty: str  # easy, medium, hard
    max_steps: int
    generate_messy: Callable[[int], pd.DataFrame]
    generate_target: Callable[[int], pd.DataFrame]
    grade: Callable[[pd.DataFrame, pd.DataFrame], float]
    checklist: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_column_types(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Score how many column types match the target."""
    if current.empty or target.empty:
        return 0.0
    matching = 0
    total = 0
    for col in target.columns:
        if col in current.columns:
            total += 1
            # Compare dtype kind (i=int, f=float, O=object, M=datetime)
            if current[col].dtype.kind == target[col].dtype.kind:
                matching += 1
        else:
            total += 1
    return matching / total if total > 0 else 0.0


def _score_null_counts(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Score how close null counts are to target."""
    if current.empty or target.empty:
        return 0.0
    scores = []
    for col in target.columns:
        if col in current.columns:
            target_nulls = target[col].isnull().sum()
            current_nulls = current[col].isnull().sum()
            if target_nulls == 0:
                scores.append(1.0 if current_nulls == 0 else max(0.0, 1.0 - current_nulls / max(len(current), 1)))
            else:
                scores.append(1.0 - min(1.0, abs(current_nulls - target_nulls) / max(len(current), 1)))
    return sum(scores) / len(scores) if scores else 0.0


def _score_duplicates(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Score duplicate removal progress."""
    target_dups = target.duplicated().sum()
    current_dups = current.duplicated().sum()
    if target_dups == 0:
        if current_dups == 0:
            return 1.0
        return max(0.0, 1.0 - current_dups / max(len(current), 1))
    return 1.0 - min(1.0, abs(current_dups - target_dups) / max(len(current), 1))


def _score_row_count(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Score how close row count is to target."""
    if len(target) == 0:
        return 1.0 if len(current) == 0 else 0.0
    diff = abs(len(current) - len(target))
    return max(0.0, 1.0 - diff / max(len(target), 1))


def _score_column_names(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Score how many column names match the target."""
    target_cols = set(target.columns)
    current_cols = set(current.columns)
    if not target_cols:
        return 1.0
    return len(target_cols & current_cols) / len(target_cols)


# ---------------------------------------------------------------------------
# Task 1: Easy — Fix Column Types
# ---------------------------------------------------------------------------

def _easy_generate_messy(seed: int) -> pd.DataFrame:
    """Generate a dataset with wrong column types."""
    rng = random.Random(seed)
    n = 20
    data = {
        "employee_id": [str(rng.randint(1000, 9999)) for _ in range(n)],
        "name": [rng.choice(["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]) for _ in range(n)],
        "age": [str(rng.randint(22, 65)) for _ in range(n)],
        "salary": [f"{rng.uniform(30000, 120000):.2f}" for _ in range(n)],
        "join_date": [f"2{rng.randint(0, 0)}2{rng.randint(0, 4)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" for _ in range(n)],
        "department": [rng.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"]) for _ in range(n)],
    }
    return pd.DataFrame(data)


def _easy_generate_target(seed: int) -> pd.DataFrame:
    """Generate the target clean dataset for task 1."""
    df = _easy_generate_messy(seed)
    df["employee_id"] = df["employee_id"].astype(int)
    df["age"] = df["age"].astype(int)
    df["salary"] = df["salary"].astype(float)
    df["join_date"] = pd.to_datetime(df["join_date"])
    return df


def _easy_grade(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Grade task 1: type matching is the primary objective."""
    type_score = _score_column_types(current, target)
    row_score = _score_row_count(current, target)
    return 0.8 * type_score + 0.2 * row_score


# ---------------------------------------------------------------------------
# Task 2: Medium — Clean Missing Values + Remove Duplicates
# ---------------------------------------------------------------------------

def _medium_generate_messy(seed: int) -> pd.DataFrame:
    """Generate a dataset with missing values and duplicates."""
    rng = random.Random(seed)
    n = 30
    base_data = []
    for i in range(n):
        row = {
            "product_id": i + 1,
            "product_name": rng.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y", "Tool M", "Tool N"]),
            "category": rng.choice(["Electronics", "Hardware", "Software", "Accessories"]),
            "price": round(rng.uniform(5.0, 500.0), 2),
            "stock": rng.randint(0, 1000),
        }
        # Inject nulls
        if rng.random() < 0.2:
            row["price"] = None
        if rng.random() < 0.15:
            row["category"] = None
        if rng.random() < 0.1:
            row["stock"] = None
        base_data.append(row)

    # Inject duplicates (copy ~5 random rows)
    for _ in range(5):
        idx = rng.randint(0, len(base_data) - 1)
        base_data.append(base_data[idx].copy())

    rng.shuffle(base_data)
    return pd.DataFrame(base_data)


def _medium_generate_target(seed: int) -> pd.DataFrame:
    """Generate the target clean dataset for task 2."""
    df = _medium_generate_messy(seed)
    # Fill missing price with median
    median_price = df["price"].median()
    df["price"] = df["price"].fillna(median_price)
    # Fill missing category with mode
    mode_cat = df["category"].mode()[0] if not df["category"].mode().empty else "Unknown"
    df["category"] = df["category"].fillna(mode_cat)
    # Fill missing stock with 0
    df["stock"] = df["stock"].fillna(0).astype(int)
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _medium_grade(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Grade task 2: null handling + duplicate removal."""
    null_score = _score_null_counts(current, target)
    dup_score = _score_duplicates(current, target)
    row_score = _score_row_count(current, target)
    return 0.4 * null_score + 0.35 * dup_score + 0.25 * row_score


# ---------------------------------------------------------------------------
# Task 3: Hard — Full Pipeline
# ---------------------------------------------------------------------------

def _hard_generate_messy(seed: int) -> pd.DataFrame:
    """Generate a dataset needing the full cleaning pipeline."""
    rng = random.Random(seed)
    n = 40
    base_data = []
    for i in range(n):
        row = {
            "cust_id": str(rng.randint(10000, 99999)),
            "  Full Name ": rng.choice([
                "  John Smith ", "Alice Johnson", " Bob Williams  ",
                "Charlie Brown", "  Diana Ross", "Eve Davis  ",
                "Frank Miller", " Grace Lee ",
            ]),
            "email_addr": rng.choice([
                "john@example.com", "alice@test.com", "bob@demo.com",
                "charlie@sample.org", "diana@mail.com", "INVALID",
                "eve@test.com", "frank@example.com",
            ]),
            "purchase_amt": f"${rng.uniform(10, 5000):.2f}" if rng.random() > 0.15 else str(round(rng.uniform(10, 5000), 2)),
            "rating": str(rng.randint(1, 5)) if rng.random() > 0.1 else None,
            "signup_date": f"2{rng.randint(0, 0)}2{rng.randint(0, 4)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" if rng.random() > 0.1 else None,
            "status": rng.choice(["active", "Active", "ACTIVE", "inactive", "Inactive", "INACTIVE"]),
        }
        # Inject some nulls
        if rng.random() < 0.12:
            row["email_addr"] = None
        base_data.append(row)

    # Inject duplicates
    for _ in range(6):
        idx = rng.randint(0, len(base_data) - 1)
        base_data.append(base_data[idx].copy())

    rng.shuffle(base_data)
    return pd.DataFrame(base_data)


def _hard_generate_target(seed: int) -> pd.DataFrame:
    """Generate the target clean dataset for task 3."""
    df = _hard_generate_messy(seed)
    # Rename columns
    df = df.rename(columns={
        "  Full Name ": "full_name",
        "email_addr": "email",
        "purchase_amt": "purchase_amount",
        "signup_date": "signup_date",
        "cust_id": "customer_id",
    })
    # Strip whitespace from full_name
    df["full_name"] = df["full_name"].str.strip()
    # Cast customer_id to int
    df["customer_id"] = df["customer_id"].astype(int)
    # Clean purchase_amount: remove $ and cast to float
    df["purchase_amount"] = df["purchase_amount"].astype(str).str.replace("$", "", regex=False).astype(float)
    # Cast rating to int/float, fill missing with median
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    median_rating = df["rating"].median()
    df["rating"] = df["rating"].fillna(median_rating).astype(int)
    # Normalize status to lowercase
    df["status"] = df["status"].str.lower()
    # Fill missing signup_date with a sentinel
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    # Fill missing email
    df["email"] = df["email"].fillna("unknown@example.com")
    # Filter out INVALID emails
    df = df[df["email"] != "INVALID"].reset_index(drop=True)
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _hard_grade(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Grade task 3: full pipeline."""
    name_score = _score_column_names(current, target)
    type_score = _score_column_types(current, target)
    null_score = _score_null_counts(current, target)
    dup_score = _score_duplicates(current, target)
    row_score = _score_row_count(current, target)
    return (0.15 * name_score + 0.25 * type_score + 0.25 * null_score +
            0.15 * dup_score + 0.20 * row_score)


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, TaskDefinition] = {
    "fix_column_types": TaskDefinition(
        name="fix_column_types",
        description=(
            "Fix column types in an employee dataset. Columns employee_id, age, "
            "salary, and join_date are stored as strings but should be int, int, "
            "float, and datetime respectively. Cast them to the correct types."
        ),
        difficulty="easy",
        max_steps=10,
        generate_messy=_easy_generate_messy,
        generate_target=_easy_generate_target,
        grade=_easy_grade,
        checklist=[
            "Cast employee_id from string to int",
            "Cast age from string to int",
            "Cast salary from string to float",
            "Cast join_date from string to datetime",
        ],
    ),
    "clean_missing_duplicates": TaskDefinition(
        name="clean_missing_duplicates",
        description=(
            "Clean a product inventory dataset. Fill missing price values with the "
            "median, fill missing category with the mode, fill missing stock with 0, "
            "then remove all duplicate rows."
        ),
        difficulty="medium",
        max_steps=15,
        generate_messy=_medium_generate_messy,
        generate_target=_medium_generate_target,
        grade=_medium_grade,
        checklist=[
            "Fill missing price with median",
            "Fill missing category with mode",
            "Fill missing stock with 0",
            "Remove duplicate rows",
        ],
    ),
    "full_pipeline": TaskDefinition(
        name="full_pipeline",
        description=(
            "Perform a full cleaning pipeline on a customer dataset: "
            "(1) Rename '  Full Name ' to 'full_name' and 'email_addr' to 'email' "
            "and 'purchase_amt' to 'purchase_amount' and 'cust_id' to 'customer_id'. "
            "(2) Strip whitespace from full_name. "
            "(3) Cast customer_id to int. "
            "(4) Remove '$' from purchase_amount and cast to float. "
            "(5) Cast rating to int, fill missing with median. "
            "(6) Normalize status to lowercase. "
            "(7) Fill missing email with 'unknown@example.com'. "
            "(8) Filter out rows where email is 'INVALID'. "
            "(9) Remove duplicate rows."
        ),
        difficulty="hard",
        max_steps=20,
        generate_messy=_hard_generate_messy,
        generate_target=_hard_generate_target,
        grade=_hard_grade,
        checklist=[
            "Rename columns to clean names",
            "Strip whitespace from full_name",
            "Cast customer_id to int",
            "Clean and cast purchase_amount to float",
            "Cast rating to int, fill missing with median",
            "Normalize status to lowercase",
            "Fill missing email",
            "Filter out INVALID emails",
            "Remove duplicate rows",
        ],
    ),
}
