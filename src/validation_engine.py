from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import exp
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data_io import normalize_benchmark_dataframe, slugify
from src.synthetic_data import expand_benchmark_data

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - tested through monkeypatch.
    shap = None  # type: ignore


@dataclass
class ValidationControls:
    ip_velocity_submissions: int = 0
    form_completion_seconds: float = 10.0
    duplicate_count: int = 0
    sequential_padding_detected: bool = False
    distinct_session_locations: int = 1
    distinct_session_titles: int = 1
    honeypot_filled: bool = False
    proof_uploaded: bool = False
    domain_match: bool = False
    social_sso: bool = False
    approved_submission_count: int = 0
    account_age_days: int = 0
    reject_ratio: float = 0.0
    vpn_ip_risk: bool = False
    rolling_window_days: int = 90
    iqr_multiplier: float = 1.618
    synthetic_rows: int = 0
    ml_contamination: float = 0.08
    random_seed: int = 42
    strict_currency_feasibility: bool = False
    min_ml_rows: int = 40
    require_shap: bool = True
    current_date: str | None = None


@dataclass
class TrustResult:
    multiplier: float
    components: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    final_score: float
    route: str
    confidence: str
    layer_multipliers: dict[str, float]
    issues: list[str]
    warnings: list[str]
    recommendations: list[str]
    cohort_stats: dict[str, Any]
    ml_status: str
    ml_feature_reasons: list[str]
    trust_components: list[str]


@dataclass
class LayerResult:
    multiplier: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    reasons: list[str] = field(default_factory=list)


def validate_submission(
    submission: dict[str, Any],
    benchmark_df: pd.DataFrame,
    controls: ValidationControls | None = None,
) -> ValidationResult:
    controls = controls or ValidationControls()
    benchmark = normalize_benchmark_dataframe(benchmark_df)
    if controls.synthetic_rows and controls.synthetic_rows > len(benchmark):
        benchmark = expand_benchmark_data(
            benchmark,
            target_rows=controls.synthetic_rows,
            random_seed=controls.random_seed,
        )

    layer1 = _layer1_deterministic_gate(submission, controls)
    if layer1.multiplier == 0.0:
        layer2 = LayerResult(1.0, status="Skipped after Layer 1 hard reject")
        layer3 = LayerResult(1.0, status="Skipped after Layer 1 hard reject")
    else:
        layer2 = _layer2_statistical(submission, benchmark, controls)
        layer3 = _layer3_ml(submission, benchmark, controls)

    trust = calculate_user_multiplier(controls)
    multipliers = {
        "layer1": layer1.multiplier,
        "layer2": layer2.multiplier,
        "layer3": layer3.multiplier,
        "user": trust.multiplier,
    }
    score = round(
        100.0
        * multipliers["layer1"]
        * multipliers["layer2"]
        * multipliers["layer3"]
        * multipliers["user"],
        2,
    )
    route = _route_for_score(score)
    issues = _dedupe(layer1.issues + layer2.issues + layer3.issues)
    warnings = _dedupe(layer1.warnings + layer2.warnings + layer3.warnings)

    return ValidationResult(
        final_score=score,
        route=route,
        confidence=_confidence_for_score(score),
        layer_multipliers=multipliers,
        issues=issues,
        warnings=warnings,
        recommendations=_recommendations(route, issues, warnings),
        cohort_stats=layer2.stats,
        ml_status=layer3.status,
        ml_feature_reasons=layer3.reasons,
        trust_components=trust.components,
    )


def calculate_user_multiplier(controls: ValidationControls) -> TrustResult:
    value = 1.0
    components: list[str] = ["Baseline anonymous submission: +0.00"]

    if controls.proof_uploaded:
        value += 0.30
        components.append("Cryptographic proof uploaded: +0.30")
    if controls.domain_match:
        value += 0.15
        components.append("Work email domain matches company: +0.15")
    if controls.social_sso:
        value += 0.05
        components.append("LinkedIn/GitHub SSO present: +0.05")
    if controls.approved_submission_count >= 3:
        value += 0.10
        components.append("Proven contributor history: +0.10")
    if controls.account_age_days > 365:
        value += 0.05
        components.append("Account older than one year: +0.05")
    if controls.reject_ratio > 0.30:
        value -= 0.40
        components.append("High historical reject ratio: -0.40")
    if controls.vpn_ip_risk:
        value -= 0.20
        components.append("VPN/data-center IP risk: -0.20")

    capped = min(1.30, max(0.50, value))
    if capped != value:
        components.append(f"Trust multiplier capped from {value:.2f} to {capped:.2f}")
    return TrustResult(round(capped, 2), components)


def _layer1_deterministic_gate(
    submission: dict[str, Any], controls: ValidationControls
) -> LayerResult:
    hard_rejects: list[str] = []
    review_flags: list[str] = []

    base = _number(submission, "baseSalary")
    bonus = _number(submission, "avgAnnualBonusValue")
    stock = _number(submission, "avgAnnualStockGrantValue")
    total = _number(submission, "totalCompensation")
    yoe = _number(submission, "yearsOfExperience")
    years_at_company = _number(submission, "yearsAtCompany")

    if total < base + bonus + stock:
        hard_rejects.append("Compensation mismatch")
    if base <= 0 or stock < 0 or bonus < 0 or total <= 0:
        hard_rejects.append("Negative compensation")
    if years_at_company > yoe:
        hard_rejects.append("Experience mismatch")
    if yoe > 50:
        review_flags.append("Total experience exceeds 50 years")
    if base > 0 and stock >= 10 * base and not _is_c_suite(submission):
        review_flags.append("Unusually high equity ratio")
    if yoe > 10 and _is_entry_level(submission):
        review_flags.append("Entry-level title/level with high experience")
    if _vesting_total(submission.get("vestingSchedule")) not in (0.0, 100.0):
        hard_rejects.append("Invalid vesting schedule")
    if controls.strict_currency_feasibility and not _currency_values_feasible(submission):
        hard_rejects.append("Currency feasibility violation")

    if controls.ip_velocity_submissions > 3:
        hard_rejects.append("IP velocity abuse")
    if controls.form_completion_seconds < 3:
        hard_rejects.append("Bot-fast submission")
    if controls.duplicate_count > 0:
        hard_rejects.append("Exact duplicate submission")
    if controls.sequential_padding_detected:
        hard_rejects.append("Sequential padding")
    if controls.distinct_session_locations > 3 or controls.distinct_session_titles > 3:
        hard_rejects.append("Session spoofing")
    if controls.honeypot_filled:
        hard_rejects.append("Honeypot trigger")

    if hard_rejects:
        return LayerResult(0.0, issues=_dedupe(hard_rejects + review_flags))
    if review_flags:
        return LayerResult(0.85, issues=_dedupe(review_flags))
    return LayerResult(1.0)


def _layer2_statistical(
    submission: dict[str, Any],
    benchmark_df: pd.DataFrame,
    controls: ValidationControls,
) -> LayerResult:
    benchmark = _pass_rows(benchmark_df)
    if benchmark.empty:
        return LayerResult(
            1.0,
            warnings=["Layer 2 unavailable: no benchmark rows available."],
            status="No benchmark rows",
        )

    windowed = _within_window(benchmark, controls)
    if windowed.empty:
        windowed = benchmark

    cohort, cohort_label = _select_cohort(submission, windowed)
    values = pd.to_numeric(cohort["totalCompensation"], errors="coerce").dropna()
    if len(values) < 4:
        return LayerResult(
            1.0,
            warnings=["Layer 2 unavailable: cohort is too sparse."],
            stats={"cohort": cohort_label, "rows": int(len(values))},
        )

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    median = float(values.median())
    if iqr <= 0:
        iqr = max(abs(median) * 0.10, 1.0)
    lower = q1 - controls.iqr_multiplier * iqr
    upper = q3 + controls.iqr_multiplier * iqr
    ewma = float(values.ewm(span=min(10, len(values)), adjust=False).mean().iloc[-1])
    total = _number(submission, "totalCompensation")

    stats = {
        "cohort": cohort_label,
        "rows": int(len(values)),
        "median": round(median, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "ewma": round(ewma, 2),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
    }

    if lower <= total <= upper:
        return LayerResult(1.0, stats=stats, status="Inside rolling IQR bounds")

    excess = lower - total if total < lower else total - upper
    outside_iqr = max(0.0, excess / iqr)
    multiplier = 0.20 + 0.80 / (1.0 + exp(2.0 * (outside_iqr - 1.0)))
    return LayerResult(
        round(max(0.20, min(0.99, multiplier)), 2),
        issues=["Statistical compensation outlier"],
        stats=stats,
        status=f"Outside rolling IQR bounds by {outside_iqr:.2f} IQR",
    )


def _layer3_ml(
    submission: dict[str, Any],
    benchmark_df: pd.DataFrame,
    controls: ValidationControls,
) -> LayerResult:
    if controls.require_shap and shap is None:
        return LayerResult(
            1.0,
            warnings=["SHAP is not installed; Layer 3 cannot provide real explanations."],
            status="Layer 3 unavailable: SHAP is not installed",
        )

    training = _pass_rows(benchmark_df)
    if len(training) < controls.min_ml_rows:
        return LayerResult(
            1.0,
            warnings=[
                f"Layer 3 unavailable: {len(training)} PASS benchmark rows, "
                f"{controls.min_ml_rows} required."
            ],
            status="Layer 3 unavailable: not enough benchmark rows",
        )

    train_features, candidate_features = _build_feature_matrix(training, submission)
    model = IsolationForest(
        n_estimators=120,
        contamination=max(0.01, min(0.49, controls.ml_contamination)),
        random_state=controls.random_seed,
    )
    model.fit(train_features)
    train_scores = model.decision_function(train_features)
    candidate_score = float(model.decision_function(candidate_features)[0])
    prediction = int(model.predict(candidate_features)[0])
    percentile = float(np.mean(train_scores <= candidate_score))
    if prediction == 1:
        multiplier = 1.0
    else:
        multiplier = max(0.50, min(1.0, 0.50 + percentile))

    issues = ["ML anomaly"] if prediction == -1 else []

    if controls.require_shap:
        try:
            reasons = _shap_reasons(model, candidate_features)
            return LayerResult(
                round(multiplier, 2),
                issues=issues,
                status="SHAP explanations available",
                reasons=reasons,
            )
        except Exception as exc:
            return LayerResult(
                1.0,
                warnings=[f"SHAP failed at runtime: {exc}"],
                status="Layer 3 unavailable: SHAP runtime failure",
            )

    return LayerResult(
        round(multiplier, 2),
        issues=issues,
        status="Isolation Forest available without SHAP",
        reasons=_feature_distance_reasons(training, submission),
    )


def _build_feature_matrix(
    training: pd.DataFrame, submission: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_row = pd.DataFrame([_submission_to_row(submission)])
    train_rows = training.copy()
    for column in _feature_columns():
        if column not in train_rows.columns:
            train_rows[column] = ""
    all_rows = pd.concat([train_rows[_feature_columns()], candidate_row], ignore_index=True)
    for column in _numeric_features():
        all_rows[column] = pd.to_numeric(all_rows[column], errors="coerce").fillna(0.0)
    encoded = pd.get_dummies(all_rows, columns=_categorical_features(), dummy_na=False)
    encoded = encoded.astype(float)
    return encoded.iloc[:-1].reset_index(drop=True), encoded.iloc[[-1]].reset_index(drop=True)


def _submission_to_row(submission: dict[str, Any]) -> dict[str, Any]:
    company_info = submission.get("companyInfo") or {}
    return {
        "company": submission.get("company") or company_info.get("name") or "",
        "title": submission.get("title") or submission.get("jobFamily") or "",
        "jobFamilySlug": submission.get("jobFamilySlug") or slugify(submission.get("jobFamily")),
        "level": submission.get("level") or "",
        "locationSlug": submission.get("locationSlug") or slugify(submission.get("location")),
        "workArrangement": submission.get("workArrangement") or "",
        "focusTag": submission.get("focusTag") or submission.get("jobFamily") or "",
        "yearsOfExperience": _number(submission, "yearsOfExperience"),
        "yearsAtCompany": _number(submission, "yearsAtCompany"),
        "baseSalary": _number(submission, "baseSalary"),
        "avgAnnualBonusValue": _number(submission, "avgAnnualBonusValue"),
        "avgAnnualStockGrantValue": _number(submission, "avgAnnualStockGrantValue"),
        "totalCompensation": _number(submission, "totalCompensation"),
    }


def _feature_columns() -> list[str]:
    return _numeric_features() + _categorical_features()


def _numeric_features() -> list[str]:
    return [
        "yearsOfExperience",
        "yearsAtCompany",
        "baseSalary",
        "avgAnnualBonusValue",
        "avgAnnualStockGrantValue",
        "totalCompensation",
    ]


def _categorical_features() -> list[str]:
    return [
        "company",
        "title",
        "jobFamilySlug",
        "level",
        "locationSlug",
        "workArrangement",
        "focusTag",
    ]


def _shap_reasons(model: IsolationForest, candidate_features: pd.DataFrame) -> list[str]:
    explainer = shap.TreeExplainer(model)  # type: ignore[union-attr]
    values = explainer.shap_values(candidate_features)
    array = np.asarray(values)
    if array.ndim == 2:
        row = array[0]
    else:
        row = array.reshape(-1)
    impacts = sorted(
        zip(candidate_features.columns, np.abs(row), row),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    reasons = []
    for feature, _, signed_value in impacts[:5]:
        direction = "increased" if float(signed_value) > 0 else "decreased"
        reasons.append(f"{feature} {direction} the anomaly score")
    return reasons


def _feature_distance_reasons(
    training: pd.DataFrame, submission: dict[str, Any]
) -> list[str]:
    row = _submission_to_row(submission)
    reasons = []
    for column in _numeric_features():
        values = pd.to_numeric(training[column], errors="coerce").dropna()
        if len(values) < 2:
            continue
        std = float(values.std()) or 1.0
        z_score = abs((_safe_float(row[column]) - float(values.mean())) / std)
        reasons.append((column, z_score))
    reasons.sort(key=lambda item: item[1], reverse=True)
    return [f"{name} is {score:.1f} standard deviations from benchmark mean" for name, score in reasons[:5]]


def _pass_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized = normalize_benchmark_dataframe(df)
    pass_rows = normalized[normalized["status"].astype(str).str.upper().eq("PASS")]
    return pass_rows if not pass_rows.empty else normalized


def _within_window(df: pd.DataFrame, controls: ValidationControls) -> pd.DataFrame:
    current = pd.Timestamp(controls.current_date or date.today().isoformat(), tz="UTC")
    parsed = pd.to_datetime(df["offerDate"], errors="coerce", utc=True)
    cutoff = current - pd.Timedelta(days=controls.rolling_window_days)
    return df[parsed >= cutoff]


def _select_cohort(
    submission: dict[str, Any], df: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    location = submission.get("locationSlug") or slugify(submission.get("location"))
    job = submission.get("jobFamilySlug") or slugify(submission.get("jobFamily"))
    level = str(submission.get("level") or "")

    exact = df[
        (df["locationSlug"].astype(str) == str(location))
        & (df["jobFamilySlug"].astype(str) == str(job))
        & (df["level"].astype(str) == level)
    ]
    if len(exact) >= 4:
        return exact, "locationSlug + jobFamilySlug + level"

    job_level = df[
        (df["jobFamilySlug"].astype(str) == str(job))
        & (df["level"].astype(str) == level)
    ]
    if len(job_level) >= 4:
        return job_level, "jobFamilySlug + level"

    job_only = df[df["jobFamilySlug"].astype(str) == str(job)]
    if len(job_only) >= 4:
        return job_only, "jobFamilySlug"
    return df, "all benchmark rows"


def _vesting_total(schedule: Any) -> float:
    if not schedule:
        return 0.0
    total = 0.0
    for item in schedule:
        if isinstance(item, dict):
            total += _safe_float(item.get("percent"))
    return round(total, 6)


def _currency_values_feasible(submission: dict[str, Any]) -> bool:
    checks = [
        ("baseSalary", "baseSalaryCurrency"),
        ("avgAnnualBonusValue", "bonusCurrency"),
        ("avgAnnualStockGrantValue", "stockGrantCurrency"),
        ("totalCompensation", "userCurrency"),
    ]
    for value_key, currency_key in checks:
        value = _number(submission, value_key)
        currency = str(submission.get(currency_key) or "USD").upper()
        lower, upper = _currency_bounds(currency)
        if value < lower or value > upper:
            return False
    return True


def _currency_bounds(currency: str) -> tuple[float, float]:
    bounds = {
        "USD": (1_000, 3_000_000),
        "GBP": (1_000, 2_500_000),
        "EUR": (1_000, 2_800_000),
        "INR": (50_000, 250_000_000),
        "CAD": (1_000, 3_000_000),
    }
    return bounds.get(currency, (1, 300_000_000))


def _is_c_suite(submission: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(submission.get("title") or ""),
            str(submission.get("level") or ""),
        ]
    ).lower()
    return any(token in text for token in ["chief", "ceo", "cto", "cfo", "coo", "ciso"])


def _is_entry_level(submission: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(submission.get("title") or ""),
            str(submission.get("level") or ""),
        ]
    ).lower()
    markers = ["intern", "junior", "new grad", "entry", "associate", "ic1", "l1", "l2", "l3", "mts1"]
    return any(marker in text for marker in markers)


def _route_for_score(score: float) -> str:
    if score >= 90:
        return "PASS / Public"
    if score >= 50:
        return "FLAG / Manual Review"
    return "REJECT / Quarantine"


def _confidence_for_score(score: float) -> str:
    if score >= 90:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def _recommendations(route: str, issues: list[str], warnings: list[str]) -> list[str]:
    if route.startswith("PASS"):
        return ["Publish automatically and keep submission in benchmark history."]
    if route.startswith("FLAG"):
        return ["Route to Sarah's manual review dashboard with layer explanations."]
    if any("IP" in issue or "Bot" in issue or "Honeypot" in issue for issue in issues):
        return ["Quarantine and inspect related IP/session activity before accepting new rows."]
    if warnings:
        return ["Hold for analyst review and request proof if the submitter is trusted."]
    return ["Quarantine submission and exclude from salary aggregates."]


def _number(submission: dict[str, Any], key: str) -> float:
    return _safe_float(submission.get(key))


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for item in items:
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output
