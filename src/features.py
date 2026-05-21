from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import META_COLS, PHASES
from src.preprocess import apply_outlier_strategy, sensor_columns


def _stats(series: pd.Series, prefix: str) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_max": float(s.max()),
        f"{prefix}_p95": float(np.percentile(s, 95)),
    }


def extract_experiment_features(
    df: pd.DataFrame,
    feature_set: str,
    meta: dict | None = None,
) -> pd.Series:
    meta = meta or {}
    feats: dict[str, float] = {}
    sensor_cols = sensor_columns(df)

    if feature_set == "minimal":
        col = "S1_CurrentFeedback"
        feats.update(_stats(df[col], "S1_CurrentFeedback"))
        return pd.Series(feats)

    if feature_set in ("sensor_global", "sensor_phase", "full"):
        for col in sensor_cols:
            feats.update(_stats(df[col], col))
        if feature_set in ("sensor_phase", "full") and "Machining_Process" in df.columns:
            for phase in PHASES:
                sub = df.loc[df["Machining_Process"] == phase]
                for col in ["S1_CurrentFeedback", "X1_CurrentFeedback", "Y1_CurrentFeedback", "Z1_CurrentFeedback"]:
                    if col in sub.columns:
                        feats.update(_stats(sub[col], f"{phase.replace(' ', '_')}_{col}"))
        if feature_set == "full":
            feats["feed_rate"] = float(meta.get("feed_rate", np.nan))
            feats["clamp_pressure"] = float(meta.get("clamp_pressure", np.nan))
        return pd.Series(feats)

    raise ValueError(f"Unknown feature_set: {feature_set}")


def build_feature_matrix(
    experiment_frames: dict[int, pd.DataFrame],
    experiment_ids: list[int],
    train_meta: pd.DataFrame,
    feature_set: str,
    outlier_strategy: str,
    bounds: dict | None,
) -> pd.DataFrame:
    rows = []
    meta_lookup = train_meta.set_index("experiment_id")
    for exp_id in experiment_ids:
        df = experiment_frames[exp_id]
        df = apply_outlier_strategy(df, outlier_strategy, bounds)
        meta = {}
        if exp_id in meta_lookup.index:
            row = meta_lookup.loc[exp_id]
            meta = {
                "feed_rate": row.get("feed_rate", row.get("feedrate", np.nan)),
                "clamp_pressure": row["clamp_pressure"],
            }
        feats = extract_experiment_features(df, feature_set, meta)
        feats["experiment_id"] = exp_id
        rows.append(feats)
    return pd.DataFrame(rows).set_index("experiment_id")
