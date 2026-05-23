from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd

from src.data_io import slugify


CANDIDATE_CSV_TEMPLATE_COLUMNS = [
    "uuid",
    "company",
    "companySlug",
    "title",
    "jobFamily",
    "jobFamilySlug",
    "level",
    "focusTag",
    "yearsOfExperience",
    "yearsAtCompany",
    "offerDate",
    "location",
    "locationSlug",
    "workArrangement",
    "exchangeRate",
    "baseSalary",
    "baseSalaryCurrency",
    "avgAnnualBonusValue",
    "bonusCurrency",
    "avgAnnualStockGrantValue",
    "stockGrantCurrency",
    "totalCompensation",
    "userCurrency",
    "vestingPercent1",
    "vestingPercent2",
    "vestingPercent3",
    "vestingPercent4",
]

CANDIDATE_CSV_HELP = (
    "Upload a candidate CSV with one row per submission. Include company, title, "
    "role/job family, level, location, yearsOfExperience, yearsAtCompany, "
    "baseSalary, avgAnnualBonusValue, avgAnnualStockGrantValue, totalCompensation, "
    "currency fields, and vesting percentages. Compensation fields must be numeric; "
    "totalCompensation should be at least baseSalary + bonus + stock unless you "
    "are intentionally testing a reject case."
)


@dataclass(frozen=True)
class CandidateChecklistItem:
    label: str
    moscow: str
    passed: bool
    detail: str


def build_candidate_csv_template() -> str:
    row = {
        "uuid": "candidate-1",
        "company": "ServiceNow",
        "companySlug": "servicenow",
        "title": "Software Engineer",
        "jobFamily": "Software Engineer",
        "jobFamilySlug": "software-engineer",
        "level": "IC1",
        "focusTag": "Software Engineer",
        "yearsOfExperience": 1,
        "yearsAtCompany": 0,
        "offerDate": "2026-05-20",
        "location": "Pune, MH, India",
        "locationSlug": "pune-ind",
        "workArrangement": "hybrid",
        "exchangeRate": 83.94,
        "baseSalary": 19000,
        "baseSalaryCurrency": "INR",
        "avgAnnualBonusValue": 3500,
        "bonusCurrency": "INR",
        "avgAnnualStockGrantValue": 5000,
        "stockGrantCurrency": "USD",
        "totalCompensation": 27600,
        "userCurrency": "USD",
        "vestingPercent1": 25,
        "vestingPercent2": 25,
        "vestingPercent3": 25,
        "vestingPercent4": 25,
    }
    buffer = StringIO()
    pd.DataFrame([row], columns=CANDIDATE_CSV_TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def load_candidate_csv_text(text: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(StringIO(text))
    submissions = []
    for _, row in frame.iterrows():
        submissions.append(_row_to_submission(row))
    return submissions


def evaluate_candidate_csv_checklist(df: pd.DataFrame) -> list[CandidateChecklistItem]:
    return [
        _candidate_required_columns(df),
        _candidate_numeric_fields(df),
        _candidate_compensation_math(df),
        _candidate_context_fields(df),
        CandidateChecklistItem(
            label="Behavioral and trust signals",
            moscow="Won't",
            passed=True,
            detail="CSV candidate rows do not carry IP/session/trust signals; use sidebar controls for those parameters.",
        ),
    ]


def _row_to_submission(row: pd.Series) -> dict[str, Any]:
    company = _text(row.get("company"))
    company_slug = _text(row.get("companySlug")) or slugify(company)
    job_family = _text(row.get("jobFamily")) or _text(row.get("title"))
    location = _text(row.get("location"))

    return {
        "uuid": _text(row.get("uuid")),
        "company": company,
        "title": _text(row.get("title")),
        "jobFamily": job_family,
        "jobFamilySlug": _text(row.get("jobFamilySlug")) or slugify(job_family),
        "level": _text(row.get("level")),
        "focusTag": _text(row.get("focusTag")) or job_family,
        "yearsOfExperience": _number(row.get("yearsOfExperience")),
        "yearsAtCompany": _number(row.get("yearsAtCompany")),
        "offerDate": _text(row.get("offerDate")),
        "location": location,
        "locationSlug": _text(row.get("locationSlug")) or slugify(location),
        "workArrangement": _text(row.get("workArrangement")),
        "exchangeRate": _number(row.get("exchangeRate")),
        "baseSalary": _number(row.get("baseSalary")),
        "baseSalaryCurrency": _text(row.get("baseSalaryCurrency")),
        "avgAnnualBonusValue": _number(row.get("avgAnnualBonusValue")),
        "bonusCurrency": _text(row.get("bonusCurrency")),
        "avgAnnualStockGrantValue": _number(row.get("avgAnnualStockGrantValue")),
        "stockGrantCurrency": _text(row.get("stockGrantCurrency")),
        "totalCompensation": _number(row.get("totalCompensation")),
        "userCurrency": _text(row.get("userCurrency")),
        "companyInfo": {
            "name": company,
            "slug": company_slug,
        },
        "vestingSchedule": [
            {"percent": _number(row.get("vestingPercent1"))},
            {"percent": _number(row.get("vestingPercent2"))},
            {"percent": _number(row.get("vestingPercent3"))},
            {"percent": _number(row.get("vestingPercent4"))},
        ],
    }


def _candidate_required_columns(df: pd.DataFrame) -> CandidateChecklistItem:
    required = {
        "company",
        "title",
        "jobFamily",
        "level",
        "yearsOfExperience",
        "yearsAtCompany",
        "location",
        "baseSalary",
        "avgAnnualBonusValue",
        "avgAnnualStockGrantValue",
        "totalCompensation",
    }
    missing = sorted(required - set(df.columns))
    return CandidateChecklistItem(
        label="Required columns",
        moscow="Must",
        passed=not missing,
        detail="All required candidate columns are present."
        if not missing
        else "Missing: " + ", ".join(missing),
    )


def _candidate_numeric_fields(df: pd.DataFrame) -> CandidateChecklistItem:
    numeric = [
        "yearsOfExperience",
        "yearsAtCompany",
        "baseSalary",
        "avgAnnualBonusValue",
        "avgAnnualStockGrantValue",
        "totalCompensation",
    ]
    bad = []
    for column in numeric:
        if column not in df.columns:
            bad.append(column)
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            bad.append(column)
    return CandidateChecklistItem(
        label="Numeric compensation fields",
        moscow="Must",
        passed=not bad,
        detail="Numeric fields are parseable and non-negative."
        if not bad
        else "Invalid values in: " + ", ".join(bad),
    )


def _candidate_compensation_math(df: pd.DataFrame) -> CandidateChecklistItem:
    required = {
        "baseSalary",
        "avgAnnualBonusValue",
        "avgAnnualStockGrantValue",
        "totalCompensation",
    }
    if not required.issubset(df.columns):
        return CandidateChecklistItem(
            label="Compensation math",
            moscow="Must",
            passed=False,
            detail="Compensation math requires base, bonus, stock, and total columns.",
        )
    base = pd.to_numeric(df["baseSalary"], errors="coerce")
    bonus = pd.to_numeric(df["avgAnnualBonusValue"], errors="coerce")
    stock = pd.to_numeric(df["avgAnnualStockGrantValue"], errors="coerce")
    total = pd.to_numeric(df["totalCompensation"], errors="coerce")
    invalid = total < (base + bonus + stock)
    return CandidateChecklistItem(
        label="Compensation math",
        moscow="Must",
        passed=not bool(invalid.fillna(True).any()),
        detail="totalCompensation is at least baseSalary + bonus + stock."
        if not bool(invalid.fillna(True).any())
        else "One or more rows will trigger compensation mismatch rejection.",
    )


def _candidate_context_fields(df: pd.DataFrame) -> CandidateChecklistItem:
    helpful = {
        "companySlug",
        "jobFamilySlug",
        "locationSlug",
        "offerDate",
        "workArrangement",
        "focusTag",
    }
    present = sorted(helpful.intersection(df.columns))
    return CandidateChecklistItem(
        label="Helpful context fields",
        moscow="Should",
        passed=len(present) >= 4,
        detail="Present: " + ", ".join(present)
        if present
        else "Add slugs, offerDate, workArrangement, and focusTag for better statistics and ML explanations.",
    )


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _number(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return 0.0
