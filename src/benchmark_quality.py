from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterable

import pandas as pd

from src.data_io import normalize_benchmark_dataframe


REQUIRED_TEMPLATE_COLUMNS = [
    "submission_id",
    "company",
    "role",
    "level",
    "location",
    "jobFamilySlug",
    "locationSlug",
    "years_of_experience",
    "years_at_company",
    "base_salary",
    "bonus",
    "stock",
    "total_compensation",
    "currency",
    "offer_date",
    "workArrangement",
    "focusTag",
    "issues",
    "status",
]

DATASET_QUALITY_TOOLTIP = (
    "Upload CSV/XLSX benchmark data with required columns for company, role, "
    "level, location, experience, numeric compensation components, currency, "
    "offer date, and status. Numeric compensation fields should be parseable and non-negative. "
    "Use PASS, FLAG, or REJECT status labels. Include enough PASS rows for "
    "Isolation Forest training, ideally 40 or more after any in-app synthetic "
    "expansion. Cohort fields such as jobFamilySlug and locationSlug improve "
    "rolling IQR quality."
)


@dataclass(frozen=True)
class ChecklistItem:
    key: str
    moscow: str
    title: str
    passed: bool
    detail: str


def build_csv_template() -> str:
    rows = [
        {
            "submission_id": 1,
            "company": "ServiceNow",
            "role": "Software Engineer",
            "level": "IC1",
            "location": "Pune, India",
            "jobFamilySlug": "software-engineer",
            "locationSlug": "pune-ind",
            "years_of_experience": 1,
            "years_at_company": 0,
            "base_salary": 19000,
            "bonus": 3500,
            "stock": 5000,
            "total_compensation": 27600,
            "currency": "USD",
            "offer_date": "2026-05-20",
            "workArrangement": "hybrid",
            "focusTag": "Software Engineer",
            "issues": "None",
            "status": "PASS",
        },
        {
            "submission_id": 2,
            "company": "Google",
            "role": "Software Engineer",
            "level": "L3",
            "location": "Mountain View, USA",
            "jobFamilySlug": "software-engineer",
            "locationSlug": "mountain-view-usa",
            "years_of_experience": 2,
            "years_at_company": 1,
            "base_salary": 180000,
            "bonus": 20000,
            "stock": 40000,
            "total_compensation": 150000,
            "currency": "USD",
            "offer_date": "2026-05-19",
            "workArrangement": "onsite",
            "focusTag": "Backend",
            "issues": "Compensation mismatch",
            "status": "REJECT",
        },
    ]
    frame = pd.DataFrame(rows, columns=REQUIRED_TEMPLATE_COLUMNS)
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def evaluate_benchmark_replacement_readiness(
    raw_df: pd.DataFrame,
    min_ml_pass_rows: int = 40,
) -> list[ChecklistItem]:
    raw_df = raw_df.copy() if raw_df is not None else pd.DataFrame()
    normalized = normalize_benchmark_dataframe(raw_df)
    columns = set(raw_df.columns)

    return [
        _required_columns_item(columns),
        _numeric_compensation_item(raw_df, normalized),
        _status_labels_item(normalized),
        _ml_volume_item(normalized, min_ml_pass_rows),
        _cohort_fields_item(columns, normalized),
        _offer_dates_item(normalized),
        _optional_ml_context_item(columns),
        ChecklistItem(
            key="original_file_safety",
            moscow="WONT",
            title="Overwrite original validated_submissions.xlsx",
            passed=True,
            detail="The replacement is used only in the active app session; the original file is not modified.",
        ),
    ]


def checklist_summary(checklist: Iterable[ChecklistItem]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in checklist:
        if item.passed:
            summary[item.moscow] = summary.get(item.moscow, 0) + 1
    return summary


def _required_columns_item(columns: set[str]) -> ChecklistItem:
    groups = {
        "company": {"company"},
        "role/title": {"role", "title", "jobFamily"},
        "level": {"level"},
        "location": {"location"},
        "years_of_experience": {"years_of_experience", "yearsOfExperience"},
        "years_at_company": {"years_at_company", "yearsAtCompany"},
        "base_salary": {"base_salary", "baseSalary"},
        "bonus": {"bonus", "avgAnnualBonusValue"},
        "stock": {"stock", "avgAnnualStockGrantValue"},
        "total_compensation": {"total_compensation", "totalCompensation"},
        "currency": {"currency", "userCurrency", "baseSalaryCurrency"},
        "status": {"status"},
    }
    missing = [
        label
        for label, aliases in groups.items()
        if not columns.intersection(aliases)
    ]
    return ChecklistItem(
        key="required_columns",
        moscow="MUST",
        title="Required replacement columns",
        passed=not missing,
        detail="All required column groups are present." if not missing else "Missing: " + ", ".join(missing),
    )


def _numeric_compensation_item(raw_df: pd.DataFrame, normalized: pd.DataFrame) -> ChecklistItem:
    numeric_groups = {
        "yearsOfExperience": ["yearsOfExperience", "years_of_experience"],
        "yearsAtCompany": ["yearsAtCompany", "years_at_company"],
        "baseSalary": ["baseSalary", "base_salary"],
        "avgAnnualBonusValue": ["avgAnnualBonusValue", "bonus"],
        "avgAnnualStockGrantValue": ["avgAnnualStockGrantValue", "stock"],
        "totalCompensation": ["totalCompensation", "total_compensation"],
    }
    if normalized.empty:
        return ChecklistItem(
            "numeric_compensation",
            "MUST",
            "Numeric compensation and experience fields",
            False,
            "No rows were found.",
        )

    bad_columns = []
    for canonical, aliases in numeric_groups.items():
        source = _first_existing_series(raw_df, aliases)
        values = pd.to_numeric(source, errors="coerce") if source is not None else normalized[canonical]
        if values.isna().any() or (values < 0).any():
            bad_columns.append(canonical)

    return ChecklistItem(
        key="numeric_compensation",
        moscow="MUST",
        title="Numeric compensation and experience fields",
        passed=not bad_columns,
        detail="Numeric fields are parseable and non-negative."
        if not bad_columns
        else "Invalid values in: " + ", ".join(bad_columns),
    )


def _first_existing_series(raw_df: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    for column in aliases:
        if column in raw_df.columns:
            return raw_df[column]
    return None


def _status_labels_item(normalized: pd.DataFrame) -> ChecklistItem:
    allowed = {"PASS", "FLAG", "REJECT"}
    statuses = set(normalized["status"].astype(str).str.upper()) if not normalized.empty else set()
    invalid = sorted(statuses - allowed)
    has_pass = "PASS" in statuses
    return ChecklistItem(
        key="status_labels",
        moscow="MUST",
        title="Status labels usable for validation",
        passed=not invalid and has_pass,
        detail="Statuses use PASS, FLAG, or REJECT and include at least one PASS row."
        if not invalid and has_pass
        else _status_failure_detail(invalid, has_pass),
    )


def _ml_volume_item(normalized: pd.DataFrame, min_ml_pass_rows: int) -> ChecklistItem:
    pass_rows = int((normalized["status"].astype(str).str.upper() == "PASS").sum()) if not normalized.empty else 0
    return ChecklistItem(
        key="ml_pass_volume",
        moscow="SHOULD",
        title="Enough PASS rows for Isolation Forest",
        passed=pass_rows >= min_ml_pass_rows,
        detail=f"{pass_rows} PASS rows found; target is at least {min_ml_pass_rows} for stable ML.",
    )


def _cohort_fields_item(columns: set[str], normalized: pd.DataFrame) -> ChecklistItem:
    has_explicit = {"jobFamilySlug", "locationSlug"}.issubset(columns)
    has_derivable = not normalized.empty and normalized["jobFamilySlug"].ne("").all() and normalized["locationSlug"].ne("").all()
    return ChecklistItem(
        key="cohort_fields",
        moscow="SHOULD",
        title="Cohort fields for rolling IQR",
        passed=has_explicit or has_derivable,
        detail="jobFamilySlug and locationSlug are present or derivable from role/location.",
    )


def _offer_dates_item(normalized: pd.DataFrame) -> ChecklistItem:
    if normalized.empty:
        passed = False
        detail = "No rows were found."
    else:
        parsed = pd.to_datetime(normalized["offerDate"], errors="coerce", utc=True)
        passed = not parsed.isna().any()
        detail = "Offer dates are parseable for rolling windows." if passed else "Some offer dates could not be parsed."
    return ChecklistItem(
        key="offer_dates",
        moscow="SHOULD",
        title="Offer dates for time-aware validation",
        passed=passed,
        detail=detail,
    )


def _optional_ml_context_item(columns: set[str]) -> ChecklistItem:
    optional = {"workArrangement", "focusTag"}
    present = sorted(columns.intersection(optional))
    return ChecklistItem(
        key="optional_ml_context",
        moscow="COULD",
        title="Optional ML context columns",
        passed=bool(present),
        detail="Present optional columns: " + ", ".join(present)
        if present
        else "Add workArrangement and focusTag to improve multivariate anomaly explanations.",
    )


def _status_failure_detail(invalid: list[str], has_pass: bool) -> str:
    parts = []
    if invalid:
        parts.append("Invalid labels: " + ", ".join(invalid))
    if not has_pass:
        parts.append("At least one PASS row is required.")
    return " ".join(parts)
