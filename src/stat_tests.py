from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    note: str = ""


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> TestResult:
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
    return TestResult("Wilcoxon signed-rank", float(stat), float(p))


def friedman_test(*groups: np.ndarray) -> TestResult:
    stat, p = stats.friedmanchisquare(*groups)
    return TestResult("Friedman test", float(stat), float(p))


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> TestResult:
    a = (pred_a == y_true).astype(int)
    b = (pred_b == y_true).astype(int)
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    result = stats.binomtest(min(n01, n10), n01 + n10, 0.5)
    return TestResult("McNemar (exact binomial)", float(min(n01, n10)), float(result.pvalue), note=f"n01={n01}, n10={n10}")


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(scores)), float(lo), float(hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_values[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def recall_at_threshold(y_true, y_prob, threshold=0.5):
    from src.evaluate import compute_metrics

    return compute_metrics(y_true, y_prob, threshold)["recall"]


def f1_at_threshold(y_true, y_prob, threshold=0.5):
    from src.evaluate import compute_metrics

    return compute_metrics(y_true, y_prob, threshold)["f1"]


def compare_level_runs(log_df: pd.DataFrame, run_a: str, run_b: str, metric: str = "f1") -> dict:
    fa = log_df[(log_df.run_id == run_a) & (log_df.fold.astype(str).str.isdigit())].sort_values("fold")
    fb = log_df[(log_df.run_id == run_b) & (log_df.fold.astype(str).str.isdigit())].sort_values("fold")
    w = wilcoxon_paired(fa[metric].values, fb[metric].values)
    return {"wilcoxon": w, "delta_mean": float(fb[metric].mean() - fa[metric].mean())}


def generate_statistical_report(log_path, oof_dir, out_path) -> None:
    log_df = pd.read_csv(log_path)
    lines = ["# Statistical Comparison Report", ""]
    levels = sorted(log_df["level"].unique())
    comparisons = []
    for i in range(1, len(levels)):
        prev, cur = levels[i - 1], levels[i]
        prev_runs = log_df[(log_df.level == prev) & (log_df.fold == "oof")]
        cur_runs = log_df[(log_df.level == cur) & (log_df.fold == "oof")]
        if prev_runs.empty or cur_runs.empty:
            continue
        best_prev = prev_runs.sort_values(["recall", "f1"], ascending=False).iloc[0]
        best_cur = cur_runs.sort_values(["recall", "f1"], ascending=False).iloc[0]
        cmp = compare_level_runs(log_df, best_prev.run_id, best_cur.run_id, "recall")
        w = cmp["wilcoxon"]
        lines.append(f"## {prev} → {cur}")
        lines.append(f"- Best {prev}: `{best_prev.run_id}` OOF Recall={best_prev.recall:.3f}")
        lines.append(f"- Best {cur}: `{best_cur.run_id}` OOF Recall={best_cur.recall:.3f}")
        lines.append(f"- Δ mean fold Recall: {cmp['delta_mean']:+.3f}")
        lines.append(f"- Wilcoxon p={w.p_value:.4f}")
        comparisons.append(w.p_value)
        lines.append("")
    if comparisons:
        adj = holm_adjust(comparisons)
        lines.append("## Holm-adjusted p-values (level transitions)")
        for i, p in enumerate(adj):
            lines.append(f"- Transition {i+1}: adj p={p:.4f}")
    lines.append("")
    lines.append("## n=18 limitation")
    lines.append("- Fold-level Wilcoxon is auxiliary; OOF + Bootstrap CI are primary evidence.")
    lines.append("- Recall +1 case ≈ +5.6%p on 18 experiments.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
