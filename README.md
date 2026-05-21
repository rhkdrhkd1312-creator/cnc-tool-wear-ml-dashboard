# CNC Mill Tool Wear ML

Kaggle dataset `shasun/tool-wear-detection-in-cnc-mill` — tool wear (worn/unworn) classification with Stratified 5-fold OOF experiments.

## Streamlit Cloud deploy

| Setting | Value |
|---------|-------|
| Main file | `streamlit_app.py` |
| Python | 3.12 |
| Requirements | `requirements.txt` |

1. Push this repo to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo & branch
3. Main file path: **`streamlit_app.py`**

Bundled artifacts for the dashboard: `data/`, `models/best_model.pkl`, `reports/`.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
# Set KAGGLE_USERNAME and KAGGLE_API_TOKEN in .env, then:
.\.venv\Scripts\kaggle datasets download -d shasun/tool-wear-detection-in-cnc-mill -p data/raw --unzip
```

## Run pipeline (local)

```powershell
.\.venv\Scripts\python.exe -m src.generate_eda
.\.venv\Scripts\python.exe -m src.run_experiments
.\.venv\Scripts\python.exe -m src.run_stat_tests
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.generate_final_report
```

## Run dashboard (local)

```powershell
.\run_dashboard.ps1
# or
.\.venv\Scripts\streamlit run streamlit_app.py
```

## Structure

```
streamlit_app.py          # Streamlit Cloud entry point
app/
  dashboard.py            # 3-tab dashboard UI
  dashboard_utils.py      # analysis helpers
src/                      # data loading, preprocessing, experiments
configs/                  # experiment grid & best config
data/                     # raw + processed features (bundled for Cloud)
models/                   # best_model.pkl
reports/                  # experiment log, OOF preds, stats
```

## Dashboard tabs

1. **실험 결과** — OOF table, statistical validation, methodology interpretation
2. **입력 변수 분석** — Feature importance, Mann-Whitney, correlation EDA
3. **오분류 데이터 심화 분석** — misclassification drill-down, label audit
