"""Generate EDA artifacts and eda_decisions.md."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.load_data import ensure_dirs, load_all_experiments, load_train
from src.preprocess import apply_doc_rule

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIG = REPORTS / "figures"


def main():
    ensure_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    train = load_train()
    experiments = load_all_experiments()

    cross = train.groupby(["tool_condition", "feed_rate", "clamp_pressure"]).size().reset_index(name="count")
    cross.to_csv(REPORTS / "cross_tab.csv", index=False)

    doc_rates = []
    for exp_id, df in experiments.items():
        n0 = len(df)
        n1 = len(apply_doc_rule(df))
        doc_rates.append({"experiment_id": exp_id, "doc_rule_drop_rate": 1 - n1 / max(n0, 1)})
    doc_df = pd.DataFrame(doc_rates)
    doc_df = doc_df.merge(train[["experiment_id", "tool_condition"]], on="experiment_id")
    doc_df.to_csv(REPORTS / "doc_rule_rates.csv", index=False)

    feat_rows = []
    for exp_id, df in experiments.items():
        clean = apply_doc_rule(df)
        for col in ["S1_CurrentFeedback", "X1_CurrentFeedback"]:
            if col in clean.columns:
                feat_rows.append(
                    {
                        "experiment_id": exp_id,
                        "feature": col,
                        "mean": clean[col].mean(),
                        "tool_condition": train.set_index("experiment_id").loc[exp_id, "tool_condition"],
                    }
                )
    feat_df = pd.DataFrame(feat_rows)
    mw_p = []
    for feat in feat_df["feature"].unique():
        sub = feat_df[feat_df.feature == feat]
        w = sub[sub.tool_condition == "worn"]["mean"]
        u = sub[sub.tool_condition == "unworn"]["mean"]
        if len(w) > 0 and len(u) > 0:
            _, p = mannwhitneyu(w, u, alternative="two-sided")
            mw_p.append({"feature": feat, "p_value": p})
    pd.DataFrame(mw_p).to_csv(REPORTS / "mannwhitney_features.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.countplot(data=train, x="tool_condition", hue="feed_rate")
    plt.title("tool_condition x feed_rate")
    plt.tight_layout()
    plt.savefig(FIG / "tool_feed_cross.png", dpi=120)
    plt.close()

    avg_drop = doc_df["doc_rule_drop_rate"].mean()
    md = f"""# EDA Design Decisions

## Summary
- Experiments: 18 (unworn 8, worn 10)
- Primary task: binary `tool_condition` classification
- Evaluation unit: 18 experiments (not time-series rows)

## Findings → Decisions

| EDA finding | Design decision | Validation experiment |
|-------------|-----------------|----------------------|
| feed_rate / clamp_pressure vary with label | Run sensor-only vs full feature sets | `feature_set`: sensor_global vs full |
| Machining_Process phases show spindle current patterns | Add phase-wise aggregated features | `feature_set`: sensor_global vs sensor_phase |
| doc_rule removes ~{avg_drop:.1%} rows on average | Compare outlier filter on/off | `outlier`: none vs doc_rule |
| Class imbalance 10:8 | sampling + class_weight experiments | `sampling`: * |
| FN (miss worn) cost > FP | primary metric = Recall(worn) | `threshold`: fixed vs recall_target |

## Confounding check
See `cross_tab.csv` — feed_rate and clamp_pressure are associated with tool_condition; sensor-only ablation is required.

## Feature ranking (Mann-Whitney)
See `mannwhitney_features.csv` for top sensor candidates.

## Figures
- `figures/tool_feed_cross.png`
"""
    (REPORTS / "eda_decisions.md").write_text(md, encoding="utf-8")
    print("EDA reports written to", REPORTS)


if __name__ == "__main__":
    main()
