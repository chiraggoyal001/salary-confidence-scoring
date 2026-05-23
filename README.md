# Levels.fyi Validation MVP

Local Streamlit MVP for testing the four-layer crowdsourced compensation validation solution.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app preloads these files when they exist:

- `C:\Users\Acer\Downloads\sample.json`
- `C:\Users\Acer\Downloads\validated_submissions.xlsx`

You can replace both through the upload controls, edit JSON/table data in the app, and use sidebar sliders/toggles for the behavioral and trust fields missing from `sample.json`.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
