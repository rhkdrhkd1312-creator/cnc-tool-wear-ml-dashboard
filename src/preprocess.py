from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DOC_RULE, META_COLS


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def apply_doc_rule(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feed_col, feed_val = DOC_RULE["feedrate_bad"]
    x_col, x_val = DOC_RULE["x_position_bad"]
    mask = (out[feed_col] == feed_val) | (out[x_col] == x_val)
    return out.loc[~mask].copy()


def _clip_with_stats(
    df: pd.DataFrame,
    cols: list[str],
    lower: pd.Series,
    upper: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].clip(lower=lower[col], upper=upper[col])
    return out


def fit_iqr_bounds(train_df: pd.DataFrame, cols: list[str], factor: float = 1.5) -> tuple[pd.Series, pd.Series]:
    q1 = train_df[cols].quantile(0.25)
    q3 = train_df[cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return lower, upper


def fit_zscore_bounds(train_df: pd.DataFrame, cols: list[str], z: float = 3.0) -> tuple[pd.Series, pd.Series]:
    mean = train_df[cols].mean()
    std = train_df[cols].std().replace(0, np.nan).fillna(1.0)
    lower = mean - z * std
    upper = mean + z * std
    return lower, upper


def apply_outlier_strategy(
    df: pd.DataFrame,
    strategy: str,
    bounds: dict | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if strategy == "none":
        return out
    if strategy.startswith("doc_rule"):
        out = apply_doc_rule(out)
    cols = sensor_columns(out)
    if strategy == "doc_rule":
        return out
    if bounds is None:
        return out
    lower, upper = bounds["lower"], bounds["upper"]
    if "iqr" in strategy:
        return _clip_with_stats(out, cols, lower, upper)
    if "zscore" in strategy:
        return _clip_with_stats(out, cols, lower, upper)
    return out


def fit_outlier_bounds(train_frames: list[pd.DataFrame], strategy: str) -> dict | None:
    if strategy in ("none", "doc_rule"):
        return None
    combined = pd.concat([apply_outlier_strategy(f, "doc_rule" if strategy.startswith("doc_rule") else "none") for f in train_frames], ignore_index=True)
    cols = sensor_columns(combined)
    if "iqr" in strategy:
        lower, upper = fit_iqr_bounds(combined, cols)
    elif "zscore" in strategy:
        lower, upper = fit_zscore_bounds(combined, cols)
    else:
        return None
    return {"lower": lower, "upper": upper}
