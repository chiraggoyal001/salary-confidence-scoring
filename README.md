# Salary Confidence Scoring

*A product exploration into improving trust in crowdsourced compensation data.*

**Live demo:** https://salary-confidence-scoring.streamlit.app/  
**Source code:** https://github.com/chiraggoyal001/salary-confidence-scoring

## Overview

Crowdsourced salary data is useful only when people can make reasonable comparisons across roles, levels, companies, locations, and compensation structures. Incorrect figures, duplicates, spam, and misleading submissions can weaken that trust. But an unusual salary is not automatically a false one: genuine compensation can vary substantially, especially for senior roles, equity-heavy packages, and less common job cohorts.

Salary Confidence Scoring is a Streamlit MVP that explores how a salary-sharing platform could assess incoming compensation submissions, explain the signals behind an assessment, and route uncertain cases for review rather than treating every outlier as fraud.

I built this as an independent product and technical prototype around a Levels.fyi-style use case. **It is not affiliated with or endorsed by Levels.fyi, and it does not connect to, publish to, or remove submissions from Levels.fyi.** The live demo is a standalone application for exploring the validation workflow.

## The product question

**How might a platform prioritise compensation submissions for acceptance or human review while reducing obvious data-quality problems without filtering out legitimate outliers?**

The product challenge is a trade-off between two types of error:

* **False acceptance:** unreliable data is allowed to influence compensation benchmarks.
* **False rejection:** a legitimate but unusual salary is held back or rejected.

A single salary threshold is unlikely to handle this well. The prototype therefore combines several kinds of evidence and makes the suggested route inspectable.

## What the prototype does

* Accepts a candidate submission through JSON or CSV input.
* Accepts benchmark data in CSV or Excel format, with editable benchmark rows.
* Checks structural consistency and configurable suspicious-submission signals.
* Compares a submission with relevant compensation cohorts where the available data permits it.
* Uses Isolation Forest for multivariate anomaly detection when sufficient usable data and runtime support are available.
* Shows a final score, suggested route, issues, recommendations, layer-level details, cohort context, and available feature explanations.
* Supports batch scoring and downloadable results.
* Lets users change validation settings and inspect how inputs affect the outcome.

The app is designed to make the reasoning inspectable, not to present a score as unquestionable truth.

## How the validation workflow is organised

```text
Candidate submission + benchmark data
                  |
                  v
     1. Structural and spam checks
                  |
                  v
      2. Cohort-based statistics
                  |
                  v
     3. Multivariate anomaly detection
                  |
                  v
        4. Trust-signal scoring
                  |
                  v
      Score, explanation, suggested route
          /             |             \\\\
   Pass / continue   Manual review   Quarantine
```

### 1\. Structural and behavioural checks

The first layer looks for problems such as negative or inconsistent compensation values, tenure inconsistencies, invalid vesting splits, duplicates, unusually rapid submissions, and configurable bot or spam signals.

A hard logical inconsistency is handled differently from a suspicious pattern that may have a legitimate explanation.

### 2\. Cohort-based statistical checks

The prototype compares compensation against available benchmarks, prioritising relevant dimensions such as location, job family, role level, and other supported fields. It uses interquartile-range (IQR) bounds and broader cohort fallbacks when a narrowly defined group has too little data.

An outlier is a reason to look closer—not proof that a salary is fake. Results depend on the coverage, recency, and quality of the benchmark data.

### 3\. Multivariate anomaly detection

Isolation Forest looks for unusual combinations of compensation and job-context features. SHAP-based explanations are available when the runtime supports them; the app should not imply that an explanation was produced when that capability is unavailable.

The prototype can expand sparse benchmark data synthetically in memory for experimentation. Synthetic rows help exercise the workflow but are not verified salary records and cannot demonstrate real-world detection accuracy.

### 4\. Trust signals and routing

The final assessment combines the preceding signals with configurable contributor-related inputs, such as supporting-document status, work-email domain match, prior approved submissions, account history, and other available risk indicators.

These inputs are experimental. They can be missing, spoofed, or unfairly correlated with a contributor's circumstances, so they should not independently determine whether a salary is genuine.

The current prototype uses configurable routing bands:

|Score|Suggested route|Intended handling|
|-|-|-|
|90 or higher|`PASS / Public`|Eligible to proceed under the prototype's rules|
|50 to below 90|`FLAG / Manual Review`|Hold for further review|
|Below 50|`REJECT / Quarantine`|Isolate for further investigation|

These are prototype thresholds, not validated production policy. The score is a configurable heuristic—not a calibrated probability that a submission is genuine.

## Product decisions and trade-offs

|Decision|Reasoning|Trade-off to validate|
|-|-|-|
|Use multiple validation layers|Different checks catch different failure modes; no single signal is sufficient.|More layers can add complexity and make calibration harder.|
|Compare within cohorts|Salary expectations differ by role, level, company, and location.|Narrow cohorts can be sparse or unrepresentative.|
|Treat statistical outliers as signals|Unusual compensation can be legitimate.|More manual review may be needed to avoid false rejection.|
|Include explanations and layer details|Reviewers need context to understand a suggested route.|Explanations can be incomplete or misleading if the underlying data is weak.|
|Keep the MVP local-first and interactive|Makes it possible to inspect inputs, benchmark data, and scoring behaviour without production infrastructure.|It does not provide production ingestion, moderation operations, or platform integration.|
|Use synthetic benchmark expansion for experimentation|Allows the workflow to be exercised when the available sample is small.|Synthetic data cannot replace independently labelled real-world examples.|

## Product evaluation plan

The next step is to test whether the workflow improves review decisions—not simply to add more model complexity.

I would evaluate it against a simple rules-only baseline using a privacy-safe dataset with independently reviewed labels.

|Product question|Measure|
|-|-|
|Are flagged submissions actually problematic?|Precision among flagged submissions|
|How many known problematic submissions are missed?|Recall on a labelled holdout set|
|Are legitimate salaries being held or rejected too often?|False-positive rate, including breakdowns by relevant cohort|
|Does the workflow reduce review effort?|Manual-review rate and median review time|
|Are decisions consistent across cohorts?|Error rates by role, level, location, and other appropriate groups|
|Can contributors and reviewers challenge incorrect decisions?|Reviewer agreement and appeal-overturn rate|

Evaluation should use a leakage-resistant holdout, document how labels were assigned, and compare results with the rules-only baseline. I have not established production detection accuracy, fraud-detection lift, or an improvement in published salary data; those outcomes require further evaluation.

## Current scope and limitations

This is a **working MVP for exploration**, not a production moderation service.

* It is not integrated with Levels.fyi and cannot publish, hide, or remove records from that platform.
* The score is not a calibrated probability or proof of fraud.
* Statistical checks depend on benchmark coverage, recency, and quality.
* Synthetic benchmark expansion is for workflow testing, not evidence of model performance.
* Isolation Forest and SHAP outputs depend on adequate data and runtime support.
* Trust signals require careful privacy, security, fairness, and abuse-resistance review before production use.
* The current project has not demonstrated production outcomes or measured improvement to a live compensation dataset.

## Run locally

### Requirements

The repository's documented setup uses Python 3.12. Install the dependencies listed in `requirements.txt`.

**Windows (PowerShell)**

```powershell
git clone https://github.com/chiraggoyal001/salary-confidence-scoring.git
cd salary-confidence-scoring
py -3.12 -m venv .venv
.\\\\.venv\\\\Scripts\\\\python.exe -m pip install -r requirements.txt
.\\\\.venv\\\\Scripts\\\\python.exe -m streamlit run app.py
```

**macOS / Linux**

```bash
git clone https://github.com/chiraggoyal001/salary-confidence-scoring.git
cd salary-confidence-scoring
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open `http://localhost:8501`.

### Inputs

The app supports candidate JSON or CSV uploads and benchmark CSV or Excel uploads. It also supports built-in fallback data and optional local defaults, as described in the app. Use the downloadable templates to check the expected input format.

Do not upload private compensation records or personally identifiable information unless you have permission to use them. Review data handling carefully before using sensitive real-world information.

### Tests

Run the test suite from the repository root:

```bash
python -m pytest -q
```

Passing tests help check expected behaviour, but do not establish real-world detection accuracy.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── validated\\\_submissions\\\_300.csv
├── src/
│   ├── benchmark\\\_quality.py
│   ├── csv\\\_contract.py
│   ├── data\\\_io.py
│   ├── synthetic\\\_data.py
│   ├── ui\\\_config.py
│   └── validation\\\_engine.py
└── tests/
```

## Links

* [Live demo](https://salary-confidence-scoring.streamlit.app/)
* [GitHub repository](https://github.com/chiraggoyal001/salary-confidence-scoring)
* [Report an issue or suggest an improvement](https://github.com/chiraggoyal001/salary-confidence-scoring/issues)

\---

*Independent prototype exploring crowdsourced compensation data quality. Levels.fyi is referenced only to describe the use case; this project is not affiliated with or endorsed by Levels.fyi.*

