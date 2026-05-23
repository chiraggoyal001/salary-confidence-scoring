import json
from datetime import date, timedelta

import pandas as pd
import pytest

from src.data_io import load_candidate_json_text, normalize_benchmark_dataframe
from src.synthetic_data import expand_benchmark_data
from src.validation_engine import (
    ValidationControls,
    calculate_user_multiplier,
    validate_submission,
)


def candidate(**overrides):
    base = {
        "uuid": "candidate-1",
        "company": "ServiceNow",
        "title": "Software Engineer",
        "jobFamily": "Software Engineer",
        "jobFamilySlug": "software-engineer",
        "level": "IC1",
        "yearsOfExperience": 1,
        "yearsAtCompany": 0,
        "offerDate": "2026-05-20",
        "location": "Pune, MH, India",
        "locationSlug": "pune-ind",
        "workArrangement": "hybrid",
        "focusTag": "Software Engineer",
        "exchangeRate": 83.94,
        "baseSalary": 19000,
        "baseSalaryCurrency": "INR",
        "totalCompensation": 27600,
        "avgAnnualStockGrantValue": 5000,
        "stockGrantCurrency": "USD",
        "avgAnnualBonusValue": 3500,
        "bonusCurrency": "INR",
        "userCurrency": "USD",
        "companyInfo": {"name": "ServiceNow", "slug": "servicenow"},
        "vestingSchedule": [
            {"percent": 25, "occurrences": 1},
            {"percent": 25, "occurrences": 4},
            {"percent": 25, "occurrences": 4},
            {"percent": 25, "occurrences": 4},
        ],
    }
    base.update(overrides)
    return base


def benchmark_rows(count=20, total_start=26000, status="PASS"):
    rows = []
    today = date(2026, 5, 23)
    for idx in range(count):
        total = total_start + (idx % 7) * 700
        rows.append(
            {
                "submission_id": idx + 1,
                "company": "ServiceNow",
                "role": "Software Engineer",
                "level": "IC1",
                "location": "Pune, MH, India",
                "locationSlug": "pune-ind",
                "jobFamilySlug": "software-engineer",
                "years_of_experience": 1 + (idx % 3),
                "years_at_company": idx % 2,
                "base_salary": total - 8500,
                "bonus": 3500,
                "stock": 5000,
                "total_compensation": total,
                "currency": "USD",
                "offer_date": (today - timedelta(days=idx % 45)).isoformat(),
                "issues": "None",
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def controls(**overrides):
    data = {
        "current_date": "2026-05-23",
        "synthetic_rows": 0,
        "min_ml_rows": 12,
        "ml_contamination": 0.08,
        "random_seed": 7,
    }
    data.update(overrides)
    return ValidationControls(**data)


def test_json_loader_accepts_single_object_and_array():
    single = load_candidate_json_text(json.dumps(candidate()))
    many = load_candidate_json_text(json.dumps([candidate(uuid="a"), candidate(uuid="b")]))

    assert len(single) == 1
    assert single[0]["uuid"] == "candidate-1"
    assert [item["uuid"] for item in many] == ["a", "b"]


def test_benchmark_normalization_maps_excel_columns_to_solution_schema():
    raw = pd.DataFrame(
        [
            {
                "company": "ServiceNow",
                "role": "Software Engineer",
                "level": "IC1",
                "location": "Pune, India",
                "years_of_experience": 1,
                "years_at_company": 0,
                "base_salary": 19000,
                "bonus": 3500,
                "stock": 5000,
                "total_compensation": 27600,
                "currency": "USD",
                "status": "PASS",
            }
        ]
    )

    normalized = normalize_benchmark_dataframe(raw)

    assert normalized.loc[0, "jobFamilySlug"] == "software-engineer"
    assert normalized.loc[0, "locationSlug"] == "pune-india"
    assert normalized.loc[0, "totalCompensation"] == 27600
    assert normalized.loc[0, "status"] == "PASS"


def test_layer1_auto_rejects_compensation_mismatch_and_sets_zero_multiplier():
    result = validate_submission(
        candidate(totalCompensation=20000),
        benchmark_rows(),
        controls(),
    )

    assert result.route == "REJECT / Quarantine"
    assert result.layer_multipliers["layer1"] == 0.0
    assert "Compensation mismatch" in result.issues


@pytest.mark.parametrize(
    "bad_submission, expected_issue",
    [
        (candidate(baseSalary=-1), "Negative compensation"),
        (candidate(yearsOfExperience=2, yearsAtCompany=5), "Experience mismatch"),
        (candidate(vestingSchedule=[{"percent": 60}, {"percent": 20}]), "Invalid vesting schedule"),
    ],
)
def test_layer1_auto_rejects_structural_garbage(bad_submission, expected_issue):
    result = validate_submission(bad_submission, benchmark_rows(), controls())

    assert result.route == "REJECT / Quarantine"
    assert result.layer_multipliers["layer1"] == 0.0
    assert expected_issue in result.issues


def test_layer1_auto_rejects_behavioral_spam_controls():
    result = validate_submission(
        candidate(),
        benchmark_rows(),
        controls(
            ip_velocity_submissions=4,
            form_completion_seconds=2.0,
            duplicate_count=1,
            sequential_padding_detected=True,
            distinct_session_locations=4,
            honeypot_filled=True,
        ),
    )

    assert result.route == "REJECT / Quarantine"
    assert result.layer_multipliers["layer1"] == 0.0
    assert "IP velocity abuse" in result.issues
    assert "Bot-fast submission" in result.issues
    assert "Exact duplicate submission" in result.issues
    assert "Sequential padding" in result.issues
    assert "Session spoofing" in result.issues
    assert "Honeypot trigger" in result.issues


def test_layer1_review_flags_reduce_score_without_hard_rejecting():
    result = validate_submission(
        candidate(
            yearsOfExperience=51,
            avgAnnualStockGrantValue=250000,
            totalCompensation=272500,
        ),
        pd.DataFrame(),
        controls(),
    )

    assert result.layer_multipliers["layer1"] == pytest.approx(0.85)
    assert "Total experience exceeds 50 years" in result.issues
    assert "Unusually high equity ratio" in result.issues
    assert result.route != "REJECT / Quarantine"


def test_layer2_uses_rolling_iqr_bounds_and_penalizes_extreme_outliers():
    normal = validate_submission(
        candidate(totalCompensation=27600),
        benchmark_rows(count=30),
        controls(),
    )
    outlier = validate_submission(
        candidate(
            totalCompensation=180000,
            baseSalary=150000,
            avgAnnualBonusValue=10000,
            avgAnnualStockGrantValue=20000,
        ),
        benchmark_rows(count=30),
        controls(),
    )

    assert normal.layer_multipliers["layer2"] == 1.0
    assert outlier.layer_multipliers["layer2"] < 0.5
    assert outlier.cohort_stats["upper_bound"] < 180000
    assert "Statistical compensation outlier" in outlier.issues


def test_isolation_forest_and_shap_return_feature_reasons_when_dependency_available():
    pytest.importorskip("shap")
    expanded = expand_benchmark_data(
        normalize_benchmark_dataframe(benchmark_rows(count=12)),
        target_rows=120,
        random_seed=11,
    )

    result = validate_submission(
        candidate(
            totalCompensation=150000,
            baseSalary=100000,
            avgAnnualBonusValue=5000,
            avgAnnualStockGrantValue=45000,
        ),
        expanded,
        controls(synthetic_rows=0, min_ml_rows=40),
    )

    assert result.layer_multipliers["layer3"] >= 0.5
    assert result.ml_status == "SHAP explanations available"
    assert result.ml_feature_reasons


def test_layer3_reports_unavailable_when_shap_is_required_but_missing(monkeypatch):
    import src.validation_engine as engine

    monkeypatch.setattr(engine, "shap", None)
    expanded = expand_benchmark_data(
        normalize_benchmark_dataframe(benchmark_rows(count=12)),
        target_rows=80,
        random_seed=11,
    )

    result = validate_submission(candidate(), expanded, controls(require_shap=True))

    assert result.layer_multipliers["layer3"] == 1.0
    assert result.ml_status.startswith("Layer 3 unavailable")
    assert "SHAP is not installed" in result.warnings[0]


def test_user_multiplier_caps_high_trust_and_low_trust_profiles():
    high = calculate_user_multiplier(
        controls(
            proof_uploaded=True,
            domain_match=True,
            social_sso=True,
            approved_submission_count=3,
            account_age_days=400,
        )
    )
    low = calculate_user_multiplier(
        controls(reject_ratio=0.8, vpn_ip_risk=True)
    )

    assert high.multiplier == 1.3
    assert low.multiplier == 0.5


def test_synthetic_expansion_is_in_memory_and_reaches_requested_size():
    source = normalize_benchmark_dataframe(benchmark_rows(count=8))
    expanded = expand_benchmark_data(source, target_rows=75, random_seed=3)

    assert len(source) == 8
    assert len(expanded) == 75
    assert set(source.columns).issubset(set(expanded.columns))


def test_routing_thresholds_follow_final_score():
    pass_result = validate_submission(candidate(), benchmark_rows(count=30), controls())
    flag_result = validate_submission(
        candidate(yearsOfExperience=11),
        benchmark_rows(count=30),
        controls(),
    )
    reject_result = validate_submission(
        candidate(baseSalary=-1),
        benchmark_rows(count=30),
        controls(),
    )

    assert pass_result.route == "PASS / Public"
    assert flag_result.route == "FLAG / Manual Review"
    assert reject_result.route == "REJECT / Quarantine"
