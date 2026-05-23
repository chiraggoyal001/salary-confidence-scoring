from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_io import (
    load_benchmark_file,
    load_candidate_json_text,
    load_json_path,
    normalize_benchmark_dataframe,
)
from src.synthetic_data import expand_benchmark_data
from src.ui_config import ADVANCED_CONTROLS, SIDEBAR_CONTROLS
from src.validation_engine import ValidationControls, ValidationResult, validate_submission


DEFAULT_DOWNLOADS = Path.home() / "Downloads"
DEFAULT_SAMPLE_PATH = DEFAULT_DOWNLOADS / "sample.json"
DEFAULT_BENCHMARK_PATH = DEFAULT_DOWNLOADS / "validated_submissions.xlsx"


def main() -> None:
    st.set_page_config(
        page_title="Levels.fyi Validation MVP",
        page_icon="",
        layout="wide",
    )
    st.title("Levels.fyi Crowdsourced Submission Validation")

    controls = render_sidebar_controls()
    candidate_text = render_candidate_editor()
    benchmark_df = render_benchmark_editor()

    candidates = parse_candidates(candidate_text)
    if not candidates:
        st.stop()

    selected_index = 0
    if len(candidates) > 1:
        selected_index = st.selectbox(
            "Candidate submission",
            options=list(range(len(candidates))),
            format_func=lambda idx: candidate_label(candidates[idx], idx),
        )
    selected_candidate = candidates[selected_index]

    result = validate_submission(selected_candidate, benchmark_df, controls)
    all_results = [
        summarize_result(idx, candidate, validate_submission(candidate, benchmark_df, controls))
        for idx, candidate in enumerate(candidates)
    ]

    render_report(selected_candidate, benchmark_df, controls, result, pd.DataFrame(all_results))


def render_sidebar_controls() -> ValidationControls:
    st.sidebar.header("Validation Controls")
    values: dict[str, Any] = {}
    st.sidebar.subheader("Behavior and Trust")
    for control in SIDEBAR_CONTROLS:
        values[control["id"]] = render_control(st.sidebar, control)

    with st.sidebar.expander("Advanced Model Settings", expanded=True):
        for control in ADVANCED_CONTROLS:
            values[control["id"]] = render_control(st, control)

    return ValidationControls(**values)


def render_control(container: Any, control: dict[str, Any]) -> Any:
    common = {
        "label": control["label"],
        "value": control["default"],
        "help": control.get("help"),
        "key": f"control_{control['id']}",
    }
    if control["kind"] == "toggle":
        return container.toggle(**common)
    if control["kind"] == "number":
        return container.number_input(
            min_value=control["min"],
            max_value=control["max"],
            step=control["step"],
            **common,
        )
    return container.slider(
        min_value=control["min"],
        max_value=control["max"],
        step=control["step"],
        **common,
    )


def render_candidate_editor() -> str:
    st.subheader("Candidate Submission")
    uploaded = st.file_uploader("Upload sample JSON", type=["json"])
    if uploaded is not None:
        source_text = uploaded.getvalue().decode("utf-8")
        source_name = uploaded.name
    elif DEFAULT_SAMPLE_PATH.exists():
        source_text = DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8")
        source_name = str(DEFAULT_SAMPLE_PATH)
    else:
        source_text = json.dumps(default_candidate(), indent=2)
        source_name = "built-in fallback"

    st.caption(f"Loaded candidate source: {source_name}")
    return st.text_area(
        "Edit candidate JSON",
        value=format_json_text(source_text),
        height=360,
        key=f"candidate_json_{source_name}",
    )


def render_benchmark_editor() -> pd.DataFrame:
    st.subheader("Validated Benchmark Data")
    uploaded = st.file_uploader("Upload validated submissions Excel or CSV", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        raw = load_benchmark_file(uploaded, uploaded.name)
        source_name = uploaded.name
    elif DEFAULT_BENCHMARK_PATH.exists():
        raw = load_benchmark_file(DEFAULT_BENCHMARK_PATH)
        source_name = str(DEFAULT_BENCHMARK_PATH)
    else:
        raw = default_benchmark()
        source_name = "built-in fallback"

    normalized = normalize_benchmark_dataframe(raw)
    st.caption(f"Loaded benchmark source: {source_name}")
    edited = st.data_editor(
        normalized,
        num_rows="dynamic",
        width="stretch",
        height=300,
        key=f"benchmark_editor_{source_name}",
    )
    return normalize_benchmark_dataframe(edited)


def parse_candidates(candidate_text: str) -> list[dict[str, Any]]:
    try:
        candidates = load_candidate_json_text(candidate_text)
    except Exception as exc:
        st.error(f"Candidate JSON could not be parsed: {exc}")
        return []
    if not candidates:
        st.error("Candidate JSON did not contain any submissions.")
    return candidates


def render_report(
    candidate: dict[str, Any],
    benchmark_df: pd.DataFrame,
    controls: ValidationControls,
    result: ValidationResult,
    batch_results: pd.DataFrame,
) -> None:
    tab_report, tab_layers, tab_charts, tab_batch = st.tabs(
        ["Report", "Layer Details", "Benchmark Charts", "Batch Results"]
    )

    with tab_report:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final score", f"{result.final_score:.2f}")
        col2.metric("Route", result.route)
        col3.metric("Confidence", result.confidence)
        col4.metric("Candidate total", f"{number(candidate.get('totalCompensation')):,.0f}")

        st.plotly_chart(score_gauge(result.final_score), width="stretch")

        if result.issues:
            st.error("Issues detected: " + ", ".join(result.issues))
        else:
            st.success("No hard issues detected by the active controls.")
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

        st.write("Recommendations")
        for item in result.recommendations:
            st.write(f"- {item}")

    with tab_layers:
        multipliers = pd.DataFrame(
            [
                {"layer": key, "multiplier": value}
                for key, value in result.layer_multipliers.items()
            ]
        )
        st.plotly_chart(
            px.bar(
                multipliers,
                x="layer",
                y="multiplier",
                range_y=[0, 1.35],
                text="multiplier",
                title="Multiplicative Layer Scores",
            ),
            width="stretch",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Layer 2 Cohort Statistics")
            if result.cohort_stats:
                st.dataframe(pd.DataFrame([result.cohort_stats]), width="stretch")
            else:
                st.info("No Layer 2 cohort stats available.")
        with col_b:
            st.write("Layer 3 ML / SHAP")
            st.info(result.ml_status or "Layer 3 did not run.")
            if result.ml_feature_reasons:
                for reason in result.ml_feature_reasons:
                    st.write(f"- {reason}")

        st.write("Trust Components")
        for component in result.trust_components:
            st.write(f"- {component}")

    with tab_charts:
        active_benchmark = normalize_benchmark_dataframe(benchmark_df)
        chart_df = active_benchmark.copy()
        if controls.synthetic_rows and controls.synthetic_rows > len(chart_df):
            chart_df = expand_benchmark_data(
                chart_df,
                target_rows=controls.synthetic_rows,
                random_seed=controls.random_seed,
            )

        if chart_df.empty:
            st.info("Upload or add benchmark rows to view distribution charts.")
        else:
            fig = px.histogram(
                chart_df,
                x="totalCompensation",
                color="status",
                nbins=30,
                marginal="box",
                title="Benchmark Total Compensation Distribution",
            )
            fig.add_vline(
                x=number(candidate.get("totalCompensation")),
                line_width=3,
                line_dash="dash",
                line_color="red",
                annotation_text="candidate",
            )
            st.plotly_chart(fig, width="stretch")

            scatter = px.scatter(
                chart_df,
                x="yearsOfExperience",
                y="totalCompensation",
                color="level",
                hover_data=["company", "location", "status"],
                title="Experience vs Compensation",
            )
            scatter.add_trace(
                go.Scatter(
                    x=[number(candidate.get("yearsOfExperience"))],
                    y=[number(candidate.get("totalCompensation"))],
                    mode="markers",
                    marker={"size": 16, "color": "red", "symbol": "x"},
                    name="candidate",
                )
            )
            st.plotly_chart(scatter, width="stretch")

    with tab_batch:
        st.dataframe(batch_results, width="stretch")
        st.download_button(
            "Download validation results JSON",
            data=json.dumps(batch_results.to_dict(orient="records"), indent=2),
            file_name="validation_results.json",
            mime="application/json",
        )


def score_gauge(score: float) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 90], "color": "#fef3c7"},
                    {"range": [90, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 3}, "value": score},
            },
        )
    )


def summarize_result(
    idx: int, candidate: dict[str, Any], result: ValidationResult
) -> dict[str, Any]:
    company_info = candidate.get("companyInfo") or {}
    return {
        "index": idx,
        "uuid": candidate.get("uuid", ""),
        "company": candidate.get("company") or company_info.get("name", ""),
        "title": candidate.get("title", ""),
        "level": candidate.get("level", ""),
        "location": candidate.get("location", ""),
        "totalCompensation": number(candidate.get("totalCompensation")),
        "final_score": result.final_score,
        "route": result.route,
        "confidence": result.confidence,
        "issues": ", ".join(result.issues) if result.issues else "None",
    }


def candidate_label(candidate: dict[str, Any], idx: int) -> str:
    company_info = candidate.get("companyInfo") or {}
    company = candidate.get("company") or company_info.get("name") or "Unknown company"
    title = candidate.get("title") or "Unknown title"
    level = candidate.get("level") or "Unknown level"
    return f"{idx + 1}. {company} - {title} - {level}"


def format_json_text(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2)
    except Exception:
        return text


def number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def default_candidate() -> dict[str, Any]:
    return {
        "uuid": "demo-candidate",
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
        "vestingSchedule": [{"percent": 25}, {"percent": 25}, {"percent": 25}, {"percent": 25}],
    }


def default_benchmark() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "submission_id": idx + 1,
                "company": "ServiceNow",
                "role": "Software Engineer",
                "level": "IC1",
                "location": "Pune, India",
                "years_of_experience": 1 + idx % 3,
                "years_at_company": idx % 2,
                "base_salary": 19000 + idx * 300,
                "bonus": 3500,
                "stock": 5000 + idx * 100,
                "total_compensation": 27600 + idx * 400,
                "currency": "USD",
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
