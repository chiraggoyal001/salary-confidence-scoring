import pandas as pd

from src.benchmark_quality import (
    DATASET_QUALITY_TOOLTIP,
    REQUIRED_TEMPLATE_COLUMNS,
    build_csv_template,
    evaluate_benchmark_replacement_readiness,
)


def valid_replacement_rows(count=45):
    rows = []
    for idx in range(count):
        rows.append(
            {
                "submission_id": idx + 1,
                "company": "ServiceNow",
                "role": "Software Engineer",
                "level": "IC1",
                "location": "Pune, India",
                "jobFamilySlug": "software-engineer",
                "locationSlug": "pune-ind",
                "years_of_experience": 1 + idx % 4,
                "years_at_company": idx % 3,
                "base_salary": 19000 + idx * 100,
                "bonus": 3500,
                "stock": 5000,
                "total_compensation": 27600 + idx * 100,
                "currency": "USD",
                "offer_date": "2026-05-20",
                "workArrangement": "hybrid",
                "focusTag": "Software Engineer",
                "issues": "None",
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)


def test_csv_template_contains_required_columns_and_example_rows():
    template = build_csv_template()
    header = template.splitlines()[0].split(",")

    assert REQUIRED_TEMPLATE_COLUMNS == header
    assert "ServiceNow" in template
    assert "REJECT" in template


def test_dataset_quality_tooltip_names_expected_data_properties():
    assert "required columns" in DATASET_QUALITY_TOOLTIP
    assert "numeric compensation" in DATASET_QUALITY_TOOLTIP
    assert "PASS" in DATASET_QUALITY_TOOLTIP
    assert "Isolation Forest" in DATASET_QUALITY_TOOLTIP


def test_replacement_checklist_auto_checks_valid_uploaded_data():
    checklist = evaluate_benchmark_replacement_readiness(valid_replacement_rows())
    by_key = {item.key: item for item in checklist}

    assert by_key["required_columns"].passed is True
    assert by_key["numeric_compensation"].passed is True
    assert by_key["status_labels"].passed is True
    assert by_key["ml_pass_volume"].passed is True
    assert by_key["cohort_fields"].passed is True
    assert by_key["original_file_safety"].moscow == "WONT"


def test_replacement_checklist_flags_missing_columns_bad_numbers_and_small_ml_data():
    uploaded = pd.DataFrame(
        [
            {
                "company": "ServiceNow",
                "role": "Software Engineer",
                "level": "IC1",
                "location": "Pune, India",
                "base_salary": "not-a-number",
                "bonus": 3500,
                "stock": 5000,
                "total_compensation": 27600,
                "status": "APPROVED",
            }
        ]
    )

    checklist = evaluate_benchmark_replacement_readiness(uploaded)
    by_key = {item.key: item for item in checklist}

    assert by_key["required_columns"].passed is False
    assert by_key["numeric_compensation"].passed is False
    assert by_key["status_labels"].passed is False
    assert by_key["ml_pass_volume"].passed is False
