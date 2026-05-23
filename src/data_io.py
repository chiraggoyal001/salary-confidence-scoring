import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def slugify(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def load_candidate_json_text(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("Candidate JSON must be an object or an array of objects.")


def load_json_path(path: str | Path) -> list[dict[str, Any]]:
    return load_candidate_json_text(Path(path).read_text(encoding="utf-8"))


def load_benchmark_file(file_or_path: Any, name: str | None = None) -> pd.DataFrame:
    file_name = name or getattr(file_or_path, "name", str(file_or_path))
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_or_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_or_path)
    raise ValueError("Benchmark data must be a CSV or Excel file.")


def normalize_benchmark_dataframe(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return _empty_normalized_frame()

    df = raw_df.copy()
    normalized = pd.DataFrame(index=df.index)
    normalized["submission_id"] = _first_present(df, ["submission_id", "uuid"], default=None)
    normalized["company"] = _first_present(df, ["company", "companyName"], default="")
    normalized["title"] = _first_present(df, ["title", "role", "jobFamily"], default="")
    normalized["jobFamily"] = _first_present(df, ["jobFamily", "role", "title"], default="")
    normalized["level"] = _first_present(df, ["level"], default="")
    normalized["location"] = _first_present(df, ["location"], default="")
    normalized["yearsOfExperience"] = _number_series(
        _first_present(df, ["yearsOfExperience", "years_of_experience"], default=0)
    )
    normalized["yearsAtCompany"] = _number_series(
        _first_present(df, ["yearsAtCompany", "years_at_company"], default=0)
    )
    normalized["baseSalary"] = _number_series(
        _first_present(df, ["baseSalary", "base_salary"], default=0)
    )
    normalized["avgAnnualBonusValue"] = _number_series(
        _first_present(df, ["avgAnnualBonusValue", "bonus"], default=0)
    )
    normalized["avgAnnualStockGrantValue"] = _number_series(
        _first_present(df, ["avgAnnualStockGrantValue", "stock"], default=0)
    )
    normalized["totalCompensation"] = _number_series(
        _first_present(df, ["totalCompensation", "total_compensation"], default=0)
    )
    normalized["userCurrency"] = _first_present(
        df, ["userCurrency", "currency", "baseSalaryCurrency"], default="USD"
    )
    normalized["status"] = (
        _first_present(df, ["status"], default="PASS")
        .fillna("PASS")
        .astype(str)
        .str.upper()
    )
    normalized["issues"] = _first_present(df, ["issues"], default="None")
    normalized["workArrangement"] = _first_present(df, ["workArrangement"], default="")
    normalized["focusTag"] = _first_present(
        df, ["focusTag", "jobFamily", "role", "title"], default=""
    )

    job_slug = _first_present(df, ["jobFamilySlug"], default=None)
    normalized["jobFamilySlug"] = [
        slugify(found) if pd.notna(found) and str(found).strip() else slugify(fallback)
        for found, fallback in zip(job_slug, normalized["jobFamily"])
    ]

    location_slug = _first_present(df, ["locationSlug"], default=None)
    normalized["locationSlug"] = [
        slugify(found) if pd.notna(found) and str(found).strip() else slugify(fallback)
        for found, fallback in zip(location_slug, normalized["location"])
    ]

    offer_dates = _first_present(df, ["offerDate", "offer_date"], default=None)
    normalized["offerDate"] = _normalize_offer_dates(offer_dates)

    return normalized.reset_index(drop=True)


def _empty_normalized_frame() -> pd.DataFrame:
    columns = [
        "submission_id",
        "company",
        "title",
        "jobFamily",
        "level",
        "location",
        "yearsOfExperience",
        "yearsAtCompany",
        "baseSalary",
        "avgAnnualBonusValue",
        "avgAnnualStockGrantValue",
        "totalCompensation",
        "userCurrency",
        "status",
        "issues",
        "workArrangement",
        "focusTag",
        "jobFamilySlug",
        "locationSlug",
        "offerDate",
    ]
    return pd.DataFrame(columns=columns)


def _first_present(df: pd.DataFrame, candidates: Iterable[str], default: Any) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _number_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _normalize_offer_dates(series: pd.Series) -> pd.Series:
    if series.isna().all():
        today = date.today()
        return pd.Series(
            [(today - timedelta(days=i % 45)).isoformat() for i in range(len(series))],
            index=series.index,
        )

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    today = pd.Timestamp(date.today(), tz="UTC")
    fallback = pd.Series(
        [today - pd.Timedelta(days=i % 45) for i in range(len(series))],
        index=series.index,
        dtype="datetime64[ns, UTC]",
    )
    parsed = parsed.fillna(fallback)
    return parsed.dt.date.astype(str)
