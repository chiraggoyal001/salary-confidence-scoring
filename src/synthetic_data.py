from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "yearsOfExperience",
    "yearsAtCompany",
    "baseSalary",
    "avgAnnualBonusValue",
    "avgAnnualStockGrantValue",
    "totalCompensation",
]


def expand_benchmark_data(
    benchmark_df: pd.DataFrame,
    target_rows: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    if benchmark_df is None or benchmark_df.empty:
        return benchmark_df.copy() if benchmark_df is not None else pd.DataFrame()

    source = benchmark_df.copy().reset_index(drop=True)
    if target_rows <= len(source):
        return source.copy()

    rng = np.random.default_rng(random_seed)
    synthetic_rows = []
    needed = target_rows - len(source)

    for idx in range(needed):
        row = source.iloc[int(rng.integers(0, len(source)))].copy()
        for column in NUMERIC_COLUMNS:
            if column not in row.index:
                continue
            value = _safe_float(row[column])
            if column in {"yearsOfExperience", "yearsAtCompany"}:
                jitter = int(rng.integers(-1, 2))
                row[column] = max(0, int(round(value)) + jitter)
            else:
                spread = 0.08 if column == "totalCompensation" else 0.12
                factor = float(rng.normal(1.0, spread))
                row[column] = max(0.0, round(value * max(0.55, factor), 2))

        base = _safe_float(row.get("baseSalary"))
        bonus = _safe_float(row.get("avgAnnualBonusValue"))
        stock = _safe_float(row.get("avgAnnualStockGrantValue"))
        observed_total = _safe_float(row.get("totalCompensation"))
        row["totalCompensation"] = max(observed_total, base + bonus + stock)
        row["submission_id"] = f"synthetic-{idx + 1}"
        row["status"] = "PASS"
        row["synthetic"] = True
        synthetic_rows.append(row)

    expanded = pd.concat([source, pd.DataFrame(synthetic_rows)], ignore_index=True)
    if "synthetic" not in expanded.columns:
        expanded["synthetic"] = False
    expanded["synthetic"] = expanded["synthetic"].fillna(False).astype(bool)
    return expanded.head(target_rows).reset_index(drop=True)


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
