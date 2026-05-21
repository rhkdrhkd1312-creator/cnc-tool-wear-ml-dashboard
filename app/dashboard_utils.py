"""Analysis helpers for the CNC Tool Wear Streamlit dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from src.config import DATA_PROCESSED, MODELS_DIR, OOF_DIR, REPORTS_DIR
from src.evaluate import compute_metrics
from src.load_data import load_train

NEAR_THRESHOLD_BAND = 0.15


def load_experiment_log() -> pd.DataFrame:
    path = REPORTS_DIR / "experiment_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_oof_runs(log: pd.DataFrame) -> pd.DataFrame:
    return log[log["fold"].astype(str) == "oof"].copy()


def load_best_bundle() -> dict | None:
    path = MODELS_DIR / "best_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def load_oof_bundle(run_id: str) -> dict | None:
    meta_path = OOF_DIR / f"{run_id}_meta.json"
    prob_path = OOF_DIR / f"{run_id}_oof.npy"
    if not meta_path.exists() or not prob_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    probs = np.load(prob_path)
    return {"meta": meta, "probs": probs}


def feature_importance_df(top_n: int | None = None) -> pd.DataFrame:
    bundle = load_best_bundle()
    if bundle is None:
        return pd.DataFrame(columns=["feature", "importance"])

    model = bundle["model"]
    names = list(bundle["feature_names"])
    if hasattr(model, "named_estimators_"):
        imp = np.zeros(len(names))
        count = 0
        for est in model.named_estimators_.values():
            if hasattr(est, "feature_importances_"):
                imp += est.feature_importances_
                count += 1
        imp = imp / max(count, 1)
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_.ravel())
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": names, "importance": imp}).sort_values(
        "importance", ascending=False
    )
    if top_n:
        return df.head(top_n).reset_index(drop=True)
    return df.reset_index(drop=True)


def load_feature_matrix() -> pd.DataFrame:
    path = DATA_PROCESSED / "experiment_features.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "label" in df.columns:
        df["label_num"] = (df["label"] == "worn").astype(int)
    return df


def mannwhitney_top_bottom(
    features: pd.DataFrame, importance: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    if features.empty or importance.empty:
        return pd.DataFrame()

    top = importance.head(n)["feature"].tolist()
    bottom = importance.tail(n)["feature"].tolist()
    worn = features[features["label_num"] == 1]
    unworn = features[features["label_num"] == 0]
    rows = []

    for group, feats in [("important", top), ("unimportant", bottom)]:
        for feat in feats:
            if feat not in features.columns:
                continue
            w_vals = worn[feat].dropna().values
            u_vals = unworn[feat].dropna().values
            if len(w_vals) < 2 or len(u_vals) < 2:
                p_val = np.nan
            else:
                try:
                    _, p_val = mannwhitneyu(w_vals, u_vals, alternative="two-sided")
                except ValueError:
                    p_val = np.nan
            rows.append(
                {
                    "group": group,
                    "feature": feat,
                    "worn_mean": float(np.mean(w_vals)) if len(w_vals) else np.nan,
                    "unworn_mean": float(np.mean(u_vals)) if len(u_vals) else np.nan,
                    "delta": float(np.mean(w_vals) - np.mean(u_vals)) if len(w_vals) and len(u_vals) else np.nan,
                    "p_value": float(p_val) if p_val == p_val else np.nan,
                }
            )
    return pd.DataFrame(rows)


def top_feature_correlations(features: pd.DataFrame, top_feats: list[str]) -> pd.DataFrame:
    cols = [c for c in top_feats if c in features.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return features[cols].corr(method="spearman")


def spearman_with_label(features: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    if "label_num" not in features.columns:
        return pd.DataFrame()
    rows = []
    for feat in feat_cols:
        if feat not in features.columns:
            continue
        vals = features[[feat, "label_num"]].dropna()
        if len(vals) < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(vals[feat], vals["label_num"])
        rows.append({"feature": feat, "spearman_r": rho, "p_value": p})
    return pd.DataFrame(rows)


def build_prediction_frame(run_id: str, log: pd.DataFrame) -> pd.DataFrame | None:
    bundle = load_oof_bundle(run_id)
    if bundle is None:
        return None

    row = get_oof_runs(log)
    row = row[row.run_id == run_id]
    if row.empty:
        return None
    threshold = float(row.iloc[0].get("threshold") or 0.5)

    train = load_train()
    exp_ids = bundle["meta"]["experiment_ids"]
    probs = bundle["probs"]
    meta = train.set_index("experiment_id").loc[exp_ids]
    y = meta["label"].values
    preds = (probs >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "experiment_id": exp_ids,
            "true_label": np.where(y == 1, "worn", "unworn"),
            "prob_worn": probs,
            "pred_label": np.where(preds == 1, "worn", "unworn"),
            "correct": preds == y,
            "threshold": threshold,
            "margin": np.where(y == 1, probs - threshold, threshold - probs),
            "error_type": np.where(
                preds == y,
                "correct",
                np.where(y == 1, "FN", "FP"),
            ),
        }
    )
    df["feed_rate"] = meta["feed_rate"].values
    df["clamp_pressure"] = meta["clamp_pressure"].values
    df["passed_visual_inspection"] = meta.get(
        "passed_visual_inspection", pd.Series(index=meta.index)
    ).values
    df["machining_finalized"] = meta.get(
        "machining_finalized", pd.Series(index=meta.index)
    ).values
    df["material"] = meta["material"].values
    return df


def split_errors(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    errors = pred_df[~pred_df["correct"]].copy()
    if errors.empty:
        return errors, errors, errors
    errors["near_threshold"] = errors["margin"].abs() <= NEAR_THRESHOLD_BAND
    near = errors[errors["near_threshold"]].sort_values("margin", key=np.abs)
    far = errors[~errors["near_threshold"]].sort_values("margin", key=np.abs, ascending=False)
    return errors, near, far


def label_audit(train: pd.DataFrame | None = None) -> pd.DataFrame:
    train = train if train is not None else load_train()
    rows = []
    for row in train.itertuples():
        flags = []
        vis = getattr(row, "passed_visual_inspection", None)
        worn = row.tool_condition == "worn"
        if pd.notna(vis):
            vis_pass = str(vis).lower() in {"yes", "true", "1"}
            if worn and vis_pass:
                flags.append("worn_but_visual_pass")
            if (not worn) and (not vis_pass):
                flags.append("unworn_but_visual_fail")
        fin = getattr(row, "machining_finalized", None)
        if pd.notna(fin) and str(fin).lower() == "no" and worn:
            flags.append("worn_machining_not_finalized")
        if not flags:
            continue
        rows.append(
            {
                "experiment_id": int(row.experiment_id),
                "tool_condition": row.tool_condition,
                "feed_rate": row.feed_rate,
                "clamp_pressure": row.clamp_pressure,
                "passed_visual_inspection": vis,
                "machining_finalized": fin,
                "flags": "; ".join(flags),
            }
        )
    return pd.DataFrame(rows)


def runs_with_errors(log: pd.DataFrame) -> pd.DataFrame:
    oof = get_oof_runs(log)
    oof = oof.copy()
    oof["total_errors"] = oof["fn_count"] + oof["fp_count"]
    return oof[oof["total_errors"] > 0].sort_values(["recall", "total_errors"], ascending=[False, True])


def level_progression(log: pd.DataFrame) -> pd.DataFrame:
    oof = get_oof_runs(log)
    return (
        oof.sort_values(["recall", "f1"], ascending=False)
        .groupby("level", as_index=False)
        .first()[["level", "run_id", "model", "outlier", "recall", "f1", "fn_count", "fp_count"]]
        .sort_values("level")
    )


def ablation_summary(log: pd.DataFrame) -> dict[str, pd.DataFrame]:
    oof = get_oof_runs(log)
    axes = ["outlier", "missing", "feature_set", "sampling", "model"]
    return {
        axis: oof.groupby(axis, as_index=False)["recall"].mean().sort_values("recall", ascending=False)
        for axis in axes
    }


def metrics_at_threshold(run_id: str, log: pd.DataFrame, threshold: float) -> dict:
    bundle = load_oof_bundle(run_id)
    if bundle is None:
        return {}
    train = load_train()
    y = train.set_index("experiment_id").loc[bundle["meta"]["experiment_ids"], "label"].values
    return compute_metrics(y, bundle["probs"], threshold)
