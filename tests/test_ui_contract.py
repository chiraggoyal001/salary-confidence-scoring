from src.ui_config import ADVANCED_CONTROLS, SIDEBAR_CONTROLS


def test_sidebar_contract_covers_all_missing_sample_fields():
    control_ids = {control["id"] for control in SIDEBAR_CONTROLS}

    assert {
        "ip_velocity_submissions",
        "form_completion_seconds",
        "duplicate_count",
        "sequential_padding_detected",
        "distinct_session_locations",
        "distinct_session_titles",
        "honeypot_filled",
        "proof_uploaded",
        "domain_match",
        "social_sso",
        "approved_submission_count",
        "account_age_days",
        "reject_ratio",
        "vpn_ip_risk",
    }.issubset(control_ids)


def test_advanced_controls_match_solution_defaults():
    controls = {control["id"]: control for control in ADVANCED_CONTROLS}

    assert controls["rolling_window_days"]["default"] == 90
    assert controls["iqr_multiplier"]["default"] == 1.618
    assert "synthetic_rows" in controls
    assert "ml_contamination" in controls
    assert "random_seed" in controls
    assert "strict_currency_feasibility" in controls
