"""Generate final_report.md from experiment results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.config import CONFIGS_DIR, REPORTS_DIR
from src.stat_tests import bootstrap_ci, f1_at_threshold, recall_at_threshold


def main():
    log = pd.read_csv(REPORTS_DIR / "experiment_log.csv")
    l0 = log[(log.level == "L0") & (log.fold == "oof")].sort_values("recall", ascending=False).iloc[0]
    best = log[log.fold == "oof"].sort_values(["recall", "f1"], ascending=False).iloc[0]
    cfg = yaml.safe_load((CONFIGS_DIR / "best_config.yaml").read_text(encoding="utf-8")) if (CONFIGS_DIR / "best_config.yaml").exists() else {}

    lines = [
        "# CNC Mill Tool Wear — Final Report",
        "",
        "## Executive summary",
        f"- Baseline (L0 Dummy) OOF Recall: **{l0.recall:.2f}**",
        f"- Final best run `{best.run_id}` ({best.level}) OOF Recall: **{best.recall:.2f}**, F1: **{best.f1:.2f}**",
        f"- Improvement ΔRecall: **{best.recall - l0.recall:+.2f}**",
        "",
        "## Adopted configuration",
        "```yaml",
        yaml.safe_dump(cfg, allow_unicode=True).strip(),
        "```",
        "",
        "## Level progression (best OOF per level)",
    ]
    for level in sorted(log.level.unique()):
        sub = log[(log.level == level) & (log.fold == "oof")]
        if sub.empty:
            continue
        b = sub.sort_values(["recall", "f1"], ascending=False).iloc[0]
        lines.append(f"- **{level}**: run `{b.run_id}` Recall={b.recall:.2f} F1={b.f1:.2f} model={b.model}")

    lines.extend(
        [
            "",
            "## Statistical validation",
            "See `statistical_comparison.md` for Wilcoxon / Holm-adjusted level transitions.",
            "",
            "## Limitations",
            "- n=18 experiments → high metric variance; interpret CI alongside p-values.",
            "- Sensor doc_rule removes known bad CNC readings.",
            "",
            "## Glossary (quick)",
            "- **OOF**: Out-of-Fold predictions on all 18 experiments",
            "- **Recall(worn)**: primary metric — minimize missed worn tools (FN)",
            "- **Stratified 5-fold CV**: fair experiment-level cross-validation",
        ]
    )
    (REPORTS / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote final_report.md")


if __name__ == "__main__":
    REPORTS = REPORTS_DIR
    main()
