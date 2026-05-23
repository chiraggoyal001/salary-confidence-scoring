import io

import pandas as pd

from src.csv_contract import (
    CANDIDATE_CSV_HELP,
    CANDIDATE_CSV_TEMPLATE_COLUMNS,
    build_candidate_csv_template,
    evaluate_candidate_csv_checklist,
    load_candidate_csv_text,
)


def test_candidate_csv_template_has_required_columns_and_example_row():
    template = build_candidate_csv_template()
    df = pd.read_csv(io.StringIO(template))

    assert list(df.columns) == CANDIDATE_CSV_TEMPLATE_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "company"] == "ServiceNow"
    assert "baseSalary" in CANDIDATE_CSV_HELP
    assert "totalCompensation" in CANDIDATE_CSV_HELP


def test_candidate_csv_loader_maps_flat_rows_to_submission_schema():
    csv_text = build_candidate_csv_template()
    submissions = load_candidate_csv_text(csv_text)

    assert len(submissions) == 1
    submission = submissions[0]
    assert submission["company"] == "ServiceNow"
    assert submission["companyInfo"]["slug"] == "servicenow"
    assert submission["jobFamilySlug"] == "software-engineer"
    assert submission["locationSlug"] == "pune-ind"
    assert submission["baseSalary"] == 19000
    assert submission["avgAnnualBonusValue"] == 3500
    assert submission["avgAnnualStockGrantValue"] == 5000
    assert submission["totalCompensation"] == 27600
    assert submission["vestingSchedule"] == [
        {"percent": 25.0},
        {"percent": 25.0},
        {"percent": 25.0},
        {"percent": 25.0},
    ]


def test_candidate_csv_moscow_checklist_auto_checks_quality():
    df = pd.read_csv(io.StringIO(build_candidate_csv_template()))
    checklist = evaluate_candidate_csv_checklist(df)

    by_label = {item.label: item for item in checklist}
    assert by_label["Required columns"].moscow == "Must"
    assert by_label["Required columns"].passed is True
    assert by_label["Numeric compensation fields"].passed is True
    assert by_label["Compensation math"].passed is True
    assert by_label["Helpful context fields"].moscow == "Should"
    assert by_label["Behavioral and trust signals"].moscow == "Won't"
    assert by_label["Behavioral and trust signals"].passed is True


def test_candidate_csv_moscow_checklist_flags_bad_upload():
    df = pd.DataFrame(
        [
            {
                "company": "BadCo",
                "baseSalary": "not-a-number",
                "totalCompensation": 1000,
            }
        ]
    )
    checklist = evaluate_candidate_csv_checklist(df)
    failed = {item.label for item in checklist if not item.passed}

    assert "Required columns" in failed
    assert "Numeric compensation fields" in failed
