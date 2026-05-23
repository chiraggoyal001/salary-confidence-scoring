# Levels.fyi Crowdsourced Data Validation MVP

This project is a local Streamlit MVP for testing the solution described in
`solution_levels.fyi_hackathon.docx`. It validates crowdsourced compensation
submissions with a four-layer funnel:

1. Deterministic structural and spam checks.
2. Rolling-window statistical validation with IQR bounds.
3. Isolation Forest anomaly detection with SHAP explanations.
4. Multiplicative trust scoring and routing.

The app is built for fast local experimentation. It is not a production
Kafka/Redis/Timescale deployment.

## Current Status

The MVP is implemented and verified locally.

- App URL: `http://localhost:8501`
- Test command: `.\.venv\Scripts\python.exe -m pytest -q`
- Current test result: `16 passed`
- Streamlit health endpoint: `http://localhost:8501/_stcore/health`

## Is Isolation Forest Working?

Yes. Layer 3 is working in this app when there are enough benchmark rows for
training.

The provided `validated_submissions.xlsx` has only 8 rows, with 4 PASS rows.
That is too small for a useful Isolation Forest by itself. To make the MVP
usable, the app expands benchmark data in memory using the "Synthetic benchmark
target rows" control. The original Excel file is not modified.

With the default app settings:

- Synthetic benchmark target rows: `250`
- Minimum rows for ML: `40`
- ML contamination: `0.08`
- SHAP required: `True`

Layer 3 trains `sklearn.ensemble.IsolationForest` and uses `shap.TreeExplainer`
for feature-level explanations.

Verified against the real provided files:

```text
benchmark_rows 8
pass_rows 4
final_score 100.0
route PASS / Public
layer_multipliers {'layer1': 1.0, 'layer2': 1.0, 'layer3': 1.0, 'user': 1.0}
ml_status SHAP explanations available
```

Installed ML versions in the project virtual environment:

```text
shap 0.51.0
sklearn 1.8.0
```

If you set synthetic rows below the minimum ML row threshold, the app will mark
Layer 3 as unavailable instead of pretending the ML model ran.

## Files

```text
app.py
src/
  data_io.py
  synthetic_data.py
  ui_config.py
  validation_engine.py
tests/
  test_ui_contract.py
  test_validation_engine.py
requirements.txt
README.md
```

### Main Responsibilities

- `app.py`: Streamlit dashboard, uploads, editors, tabs, charts, and report UI.
- `src/data_io.py`: JSON, Excel, CSV loading and benchmark normalization.
- `src/synthetic_data.py`: In-memory synthetic expansion for sparse benchmark data.
- `src/ui_config.py`: Sidebar and advanced control definitions.
- `src/validation_engine.py`: Four-layer validation and scoring engine.
- `tests/`: Regression tests for validation rules, routing, controls, and ML behavior.

## Data Sources

By default, the app preloads these files when they exist:

```text
C:\Users\Acer\Downloads\sample.json
C:\Users\Acer\Downloads\validated_submissions.xlsx
```

You can replace both from the UI:

- Upload a `.json` file for candidate submissions.
- Upload `.xlsx`, `.xls`, or `.csv` for validated benchmark data.
- Edit the JSON and benchmark table directly inside the app before scoring.

The app accepts candidate JSON in either form:

```json
{
  "uuid": "single-submission"
}
```

or:

```json
[
  { "uuid": "submission-1" },
  { "uuid": "submission-2" }
]
```

## Setup

The project uses a local virtual environment at `.venv`.

If `.venv` already exists, install or refresh dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you need to recreate it from scratch:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

The trusted-host flags were used because this machine had certificate-chain
issues when contacting PyPI.

## Run The App

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The app currently runs on port `8501`.

## Using The Dashboard

### Candidate Submission

Use the first upload control to provide a candidate JSON or CSV file. If no
file is uploaded, the app loads `C:\Users\Acer\Downloads\sample.json`.

Use **Download candidate CSV template** to get a one-row template with the
expected flat candidate columns. When a candidate CSV is uploaded, the app shows
an auto-checked MoSCoW checklist and converts the CSV rows into editable JSON
before scoring.

The JSON editor lets you change any candidate field before validation. This is
useful for testing examples such as:

- Negative base salary.
- Total compensation less than base + bonus + stock.
- Years at company greater than total experience.
- Very high stock grant.
- Entry-level role with high experience.

### Candidate CSV Template Columns

```text
uuid
company
companySlug
title
jobFamily
jobFamilySlug
level
focusTag
yearsOfExperience
yearsAtCompany
offerDate
location
locationSlug
workArrangement
exchangeRate
baseSalary
baseSalaryCurrency
avgAnnualBonusValue
bonusCurrency
avgAnnualStockGrantValue
stockGrantCurrency
totalCompensation
userCurrency
vestingPercent1
vestingPercent2
vestingPercent3
vestingPercent4
```

Candidate CSV checklist:

- `Must`: Required columns, numeric compensation fields, and compensation math.
- `Should`: Helpful context fields such as slugs, offer date, work arrangement, and focus tag.
- `Won't`: Behavioral and trust signals are not expected in candidate CSV; use sidebar controls for those.

### Benchmark Data

Use the second upload control to provide benchmark data as Excel or CSV. If no
file is uploaded, the app loads
`C:\Users\Acer\Downloads\validated_submissions.xlsx`.

Use **Download CSV template** to get a ready-to-fill benchmark file with the
expected columns and example PASS/REJECT rows. The upload control includes a
tooltip describing the required data qualities: required columns, numeric
compensation fields, PASS/FLAG/REJECT labels, enough PASS rows for Isolation
Forest, and cohort fields for rolling IQR.

The benchmark table is editable. Edits affect validation immediately, but only
inside the running app. The original file is not changed.

When you upload a replacement CSV/XLSX, the app shows an auto-checked MoSCoW
readiness checklist for replacing `validated_submissions.xlsx` in the active
session.

### Benchmark CSV Template Columns

```text
submission_id
company
role
level
location
jobFamilySlug
locationSlug
years_of_experience
years_at_company
base_salary
bonus
stock
total_compensation
currency
offer_date
workArrangement
focusTag
issues
status
```

### Replacement Readiness Checklist

The checklist uses MoSCoW labels:

- `MUST`: Required for the upload to be trusted as a replacement dataset.
- `SHOULD`: Strongly recommended for good statistical and ML quality.
- `COULD`: Optional fields that improve model explanations.
- `WONT`: Explicitly out of scope, such as overwriting the original file.

Auto-checked requirements:

- Required replacement columns are present.
- Compensation and experience fields are numeric and non-negative.
- Status labels use `PASS`, `FLAG`, or `REJECT` and include at least one `PASS`.
- PASS row count is high enough for Isolation Forest training.
- Cohort fields are present or derivable for rolling IQR.
- Offer dates are parseable for time-aware validation.
- Optional ML context columns such as `workArrangement` and `focusTag` are present.
- Original `validated_submissions.xlsx` is not overwritten.

### Candidate Selector

If the JSON contains an array of submissions, a selector appears so you can
inspect one candidate at a time. The Batch Results tab still scores all
submissions.

## Sidebar Controls

These fields are missing from `sample.json`, so the app exposes them as
interactive controls.

### Behavioral Spam Controls

- IP submissions in 5 minutes.
- Form completion seconds.
- Exact duplicates within 1 hour.
- Sequential salary padding.
- Distinct locations per session.
- Distinct titles per session.
- Honeypot field filled.

These feed Layer 1. Hard spam signals can auto-reject the submission.

### User Trust Controls

- Offer letter / W-2 / paystub uploaded.
- Work email domain matches company.
- LinkedIn or GitHub SSO.
- Previously approved submissions.
- Account age in days.
- Historical reject ratio.
- VPN / data-center IP risk.

These feed `M(user)` in Layer 4.

### Advanced Model Settings

- Rolling window days, default `90`.
- IQR multiplier, default `1.618`.
- Synthetic benchmark target rows, default `250`.
- ML contamination, default `0.08`.
- Random seed, default `42`.
- Strict currency feasibility.
- Minimum rows for ML, default `40`.
- Require real SHAP explanations, default `True`.

## Validation Layers

### Layer 1: Deterministic Gate

Layer 1 checks structural garbage and malicious activity before statistical or
ML scoring.

Auto-reject examples:

- `totalCompensation < baseSalary + avgAnnualBonusValue + avgAnnualStockGrantValue`
- `yearsAtCompany > yearsOfExperience`
- Negative base, stock, bonus, or total compensation.
- Invalid vesting schedule where percentages do not sum to 100.
- IP velocity greater than 3 submissions in 5 minutes.
- Form completion under 3 seconds.
- Exact duplicate count greater than 0.
- Sequential salary padding.
- Session spoofing across more than 3 locations or titles.
- Honeypot field filled.

Review-flag examples:

- Total experience greater than 50 years.
- Stock grant at least 10 times base salary unless C-suite.
- Entry-level title or level with more than 10 years of experience.

Hard rejects set:

```text
M(layer1) = 0.0
```

Review flags set:

```text
M(layer1) = 0.85
```

### Layer 2: Rolling Statistical Validation

Layer 2 evaluates the candidate against recent benchmark compensation data.

Default behavior:

- Uses a trailing 90-day window.
- Prefers cohort match by `locationSlug + jobFamilySlug + level`.
- Falls back to broader cohorts if exact matches are too sparse.
- Computes median, Q1, Q3, IQR, EWMA, lower bound, and upper bound.

Default bounds:

```text
lower = Q1 - 1.618 * IQR
upper = Q3 + 1.618 * IQR
```

Inside bounds:

```text
M(layer2) = 1.0
```

Outside bounds:

```text
M(layer2) decays with distance from the nearest bound
```

### Layer 3: Isolation Forest And SHAP

Layer 3 handles multivariate anomalies. It uses:

- `sklearn.ensemble.IsolationForest`
- One-hot encoded categorical fields.
- Numeric compensation and experience fields.
- `shap.TreeExplainer` for feature explanations.

Feature inputs include:

- Years of experience.
- Years at company.
- Base salary.
- Bonus.
- Stock.
- Total compensation.
- Company.
- Title.
- Job family slug.
- Level.
- Location slug.
- Work arrangement.
- Focus tag.

If SHAP is required and unavailable, Layer 3 reports a visible warning and uses
a neutral multiplier instead of faking explanations.

### Layer 4: Multiplicative Trust Scoring

Final score:

```text
100 * M(layer1) * M(layer2) * M(layer3) * M(user)
```

User trust starts at `1.0` and applies:

```text
+0.30 proof uploaded
+0.15 domain match
+0.05 social SSO
+0.10 at least 3 approved prior submissions
+0.05 account older than one year
-0.40 reject ratio greater than 30%
-0.20 VPN / data-center IP risk
```

The final user multiplier is capped:

```text
0.5 <= M(user) <= 1.3
```

## Routing

```text
Score >= 90    PASS / Public
Score 50-89    FLAG / Manual Review
Score < 50     REJECT / Quarantine
```

## App Tabs

### Report

Shows:

- Final score.
- Route.
- Confidence.
- Candidate total compensation.
- Score gauge.
- Issues and recommendations.

### Layer Details

Shows:

- Layer multiplier bar chart.
- Layer 2 cohort statistics.
- Layer 3 ML / SHAP status and explanations.
- Trust multiplier components.

### Benchmark Charts

Shows:

- Histogram and box plot of benchmark compensation.
- Candidate marker on the distribution.
- Experience vs compensation scatter chart.

### Batch Results

Scores every candidate when the uploaded JSON contains an array. Includes a
download button for validation results as JSON.

## Verification

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected:

```text
16 passed
```

Compile-check Python files:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py src\data_io.py src\synthetic_data.py src\ui_config.py src\validation_engine.py
```

Streamlit smoke test:

```powershell
@'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py")
at.run(timeout=90)
print("exceptions", len(at.exception))
print("tabs", len(at.tabs))
print("metrics", len(at.metric))
'@ | .\.venv\Scripts\python.exe -
```

Expected:

```text
exceptions 0
tabs 4
metrics 4
```

Server health check:

```powershell
Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing
```

Expected content:

```text
ok
```

## Troubleshooting

### Layer 3 Says Not Enough Benchmark Rows

Increase "Synthetic benchmark target rows" in the sidebar, or lower "Minimum
rows for ML" for experimentation.

Recommended demo values:

```text
Synthetic benchmark target rows = 250
Minimum rows for ML = 40
```

### Layer 3 Says SHAP Is Unavailable

Install dependencies in the project venv:

```powershell
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

Then restart Streamlit.

### App Does Not Open On Port 8501

Check whether the port is already in use:

```powershell
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
```

Run on another port:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port=8502
```

### Global Anaconda Environment Fails Imports

This machine had a broken global Anaconda `numpy`/`pandas` pairing. Use the
project venv instead:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notes And Limitations

- Synthetic benchmark data is generated in memory only.
- Uploaded/edited benchmark rows are not written back to the original Excel file.
- The MVP uses local Streamlit state, not production queues or databases.
- Layer 2 approximates a time-series database with the active benchmark table.
- Layer 3 explanations depend on SHAP runtime support.
- The original workspace is not a git repository.


## Links to Our Approach
- https://docs.google.com/document/d/1vV-a-LDN07vghi3p6PLRdNuMMi9vdupSwMY_ThO7EXQ/edit?usp=drive_link

## Demo Link
- https://drive.google.com/file/d/1YLst3ryzfmfOvnj88Oext1GCSqi2ZDZh/view?usp=sharing
