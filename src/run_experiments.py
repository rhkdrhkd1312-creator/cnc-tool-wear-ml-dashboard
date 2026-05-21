"""Run Level L0-L10 experiments and write logs + OOF predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import CONFIGS_DIR, OOF_DIR, RANDOM_STATE, REPORTS_DIR
from src.evaluate import compute_metrics, pick_threshold
from src.experiment_runner import RunConfig, _base_model, _fit_transform_fold, _scaler, cartesian_grid, run_single_config
from src.features import build_feature_matrix
from src.load_data import ensure_dirs, load_all_experiments, load_train
from src.preprocess import fit_outlier_bounds


def _best_from_level(log_df: pd.DataFrame, level: str) -> RunConfig | None:
    sub = log_df[log_df["level"] == level]
    if sub.empty:
        return None
    agg = sub.groupby("run_id").agg({"recall": "mean", "f1": "mean"}).reset_index()
    best_id = agg.sort_values(["recall", "f1"], ascending=False).iloc[0]["run_id"]
    row = sub[sub["run_id"] == best_id].iloc[0]
    return RunConfig(
        level=level,
        missing=row["missing"],
        outlier=row["outlier"],
        feature_set=row["feature_set"],
        scaler=row["scaler"],
        sampling=row["sampling"],
        model=row["model"],
        hpo=row["hpo"],
        threshold=row["threshold"],
    )


def _inject_best(grid: dict, best: RunConfig | None, keys: list[str]) -> dict:
    if best is None:
        return grid
    out = {k: list(v) for k, v in grid.items()}
    for k in keys:
        if k in out and len(out[k]) > 1:
            val = getattr(best, k)
            out[k] = [val]
    return out


def run_loeo(cfg: RunConfig, experiment_frames, train_meta) -> dict:
    from sklearn.impute import SimpleImputer

    exp_ids = sorted(train_meta["experiment_id"].tolist())
    probs = []
    for holdout in exp_ids:
        train_ids = [i for i in exp_ids if i != holdout]
        val_ids = [holdout]
        bounds = fit_outlier_bounds([experiment_frames[i] for i in train_ids], cfg.outlier)
        train_feat = build_feature_matrix(experiment_frames, train_ids, train_meta, cfg.feature_set, cfg.outlier, bounds)
        val_feat = build_feature_matrix(experiment_frames, val_ids, train_meta, cfg.feature_set, cfg.outlier, bounds)
        train_feat["label"] = train_meta.set_index("experiment_id").loc[train_ids, "label"].values
        val_feat["label"] = train_meta.set_index("experiment_id").loc[val_ids, "label"].values
        prep = _fit_transform_fold(train_feat, val_feat, cfg)
        if prep is None:
            continue
        X_tr, y_tr, X_va, y_va, _ = prep
        model = _base_model(cfg.model, cfg.sampling)
        model.fit(X_tr, y_tr)
        probs.append(model.predict_proba(X_va)[0, 1])
    y = train_meta.set_index("experiment_id").loc[exp_ids, "label"].values
    prob = np.array(probs)
    thr = pick_threshold(y, prob, cfg.threshold)
    return {"oof_metrics": compute_metrics(y, prob, thr), "oof_prob": prob, "experiment_ids": exp_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10")
    args = parser.parse_args()

    ensure_dirs()
    grid_yaml = yaml.safe_load((CONFIGS_DIR / "experiment_grid.yaml").read_text(encoding="utf-8"))
    experiment_frames = load_all_experiments()
    train_meta = load_train()

    log_rows = []
    run_counter = 0
    levels = args.levels.split(",")
    log_path = REPORTS_DIR / "experiment_log.csv"
    if log_path.exists():
        existing = pd.read_csv(log_path)
        log_rows.extend(existing.to_dict("records"))
        run_counter = existing["run_id"].str.extract(r"run_(\d+)").astype(float).max().fillna(0).iloc[0]

    best_by_level: dict[str, RunConfig] = {}

    for level in levels:
        spec = grid_yaml["levels"][level]
        if spec.get("mode") == "loeo":
            base = RunConfig(level=level, **spec["base_config"])
            result = run_loeo(base, experiment_frames, train_meta)
            run_counter += 1
            rid = f"run_{int(run_counter):04d}"
            m = result["oof_metrics"]
            log_rows.append(
                {
                    "run_id": rid,
                    "level": level,
                    **base.__dict__,
                    "fold": "loeo",
                    **{k: m[k] for k in ["f1", "recall", "precision", "roc_auc", "fn_count", "fp_count"]},
                    "brier": m.get("brier"),
                }
            )
            np.save(OOF_DIR / f"{rid}_oof.npy", result["oof_prob"])
            with open(OOF_DIR / f"{rid}_meta.json", "w", encoding="utf-8") as f:
                json.dump({"run_id": rid, "config": base.__dict__, "experiment_ids": result["experiment_ids"]}, f)
            continue

        grid = spec["grid"]
        if level == "L3" and "L2" in best_by_level:
            grid = _inject_best(grid, best_by_level.get("L2"), ["outlier"])
        if level in {"L4", "L5", "L6", "L7", "L8", "L9"}:
            prev = best_by_level.get(f"L{int(level[1:]) - 1}")
            if prev:
                for k in ["missing", "outlier", "feature_set", "scaler", "sampling", "model"]:
                    if k in grid and len(grid[k]) > 1:
                        grid[k] = [getattr(prev, k)]

        configs = cartesian_grid(level, grid)
        level_results = []
        for cfg in configs:
            try:
                result = run_single_config(cfg, experiment_frames, train_meta)
            except Exception as exc:
                print(f"Skip {cfg.level} {cfg.model}/{cfg.outlier}/{cfg.missing}: {exc}")
                continue
            if result is None:
                continue
            run_counter += 1
            rid = f"run_{int(run_counter):04d}"
            for fm in result["fold_metrics"]:
                log_rows.append(
                    {
                        "run_id": rid,
                        "level": level,
                        **cfg.__dict__,
                        "fold": fm["fold"],
                        "f1": fm["f1"],
                        "recall": fm["recall"],
                        "precision": fm["precision"],
                        "roc_auc": fm.get("roc_auc"),
                        "fn_count": fm["fn_count"],
                        "fp_count": fm["fp_count"],
                        "brier": fm.get("brier"),
                        "threshold": fm["threshold"],
                    }
                )
            om = result["oof_metrics"]
            log_rows.append(
                {
                    "run_id": rid,
                    "level": level,
                    **cfg.__dict__,
                    "fold": "oof",
                    "f1": om["f1"],
                    "recall": om["recall"],
                    "precision": om["precision"],
                    "roc_auc": om.get("roc_auc"),
                    "fn_count": om["fn_count"],
                    "fp_count": om["fp_count"],
                    "brier": om.get("brier"),
                    "threshold": result["threshold"],
                }
            )
            np.save(OOF_DIR / f"{rid}_oof.npy", result["oof_prob"])
            with open(OOF_DIR / f"{rid}_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"run_id": rid, "config": cfg.__dict__, "experiment_ids": result["experiment_ids"]},
                    f,
                )
            level_results.append((rid, om["recall"], om["f1"], cfg))

        if level_results:
            best = max(level_results, key=lambda x: (x[1], x[2]))
            best_by_level[level] = best[3]

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Wrote {log_path} ({len(log_rows)} rows)")


if __name__ == "__main__":
    main()
