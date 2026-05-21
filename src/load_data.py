from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, PROJECT_ROOT


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "train.csv")
    if "feedrate" in df.columns:
        df = df.rename(columns={"feedrate": "feed_rate"})
    df["experiment_id"] = df["No"].astype(int)
    df["label"] = (df["tool_condition"] == "worn").astype(int)
    return df


def experiment_path(experiment_id: int) -> Path:
    return DATA_RAW / f"experiment_{experiment_id:02d}.csv"


def load_experiment_timeseries(experiment_id: int) -> pd.DataFrame:
    return pd.read_csv(experiment_path(experiment_id))


def load_all_experiments() -> dict[int, pd.DataFrame]:
    train = load_train()
    return {int(row.experiment_id): load_experiment_timeseries(int(row.experiment_id)) for row in train.itertuples()}


def ensure_dirs() -> None:
    for d in [
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "reports" / "oof_predictions",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "app",
    ]:
        d.mkdir(parents=True, exist_ok=True)
