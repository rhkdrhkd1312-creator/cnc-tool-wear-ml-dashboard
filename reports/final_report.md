# CNC Mill Tool Wear — Final Report

## Executive summary
- Baseline (L0 Dummy) OOF Recall: **0.10**
- Final best run `run_0006` (L2) OOF Recall: **1.00**, F1: **1.00**
- Improvement ΔRecall: **+0.90**

## Adopted configuration
```yaml
extra: null
feature_set: sensor_global
hpo: default
level: L2
missing: median
model: xgboost
outlier: doc_rule
sampling: none
scaler: none
threshold: '0.5'
threshold_value: 0.5
```

## Level progression (best OOF per level)
- **L0**: run `run_0001` Recall=0.10 F1=0.13 model=dummy
- **L1**: run `run_0002` Recall=0.60 F1=0.57 model=logistic
- **L2**: run `run_0006` Recall=1.00 F1=1.00 model=xgboost
- **L3**: run `run_0009` Recall=1.00 F1=1.00 model=xgboost
- **L4**: run `run_0014` Recall=1.00 F1=1.00 model=xgboost
- **L5**: run `run_0015` Recall=1.00 F1=1.00 model=xgboost
- **L6**: run `run_0016` Recall=1.00 F1=1.00 model=xgboost
- **L7**: run `run_0017` Recall=1.00 F1=1.00 model=xgboost
- **L8**: run `run_0018` Recall=1.00 F1=1.00 model=xgboost
- **L9**: run `run_0021` Recall=1.00 F1=1.00 model=ensemble

## Statistical validation
See `statistical_comparison.md` for Wilcoxon / Holm-adjusted level transitions.

## Limitations
- n=18 experiments → high metric variance; interpret CI alongside p-values.
- Sensor doc_rule removes known bad CNC readings.

## Glossary (quick)
- **OOF**: Out-of-Fold predictions on all 18 experiments
- **Recall(worn)**: primary metric — minimize missed worn tools (FN)
- **Stratified 5-fold CV**: fair experiment-level cross-validation