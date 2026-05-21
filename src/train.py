"""Train and persist best model from experiment log."""
from __future__ import annotations

import yaml

import joblib
import pandas as pd

from src.config import CONFIGS_DIR, MODELS_DIR, REPORTS_DIR
from src.evaluate import pick_threshold
from src.experiment_runner import RunConfig, _apply_sampling, _base_model, _imputer, _prepare_matrix, _scaler
from src.features import build_feature_matrix
from src.load_data import ensure_dirs, load_all_experiments, load_train
from src.preprocess import fit_outlier_bounds


def get_best_config(log_path=None) -> tuple[RunConfig, str]:
    log_path = log_path or REPORTS_DIR / "experiment_log.csv"
    log = pd.read_csv(log_path)
    oof = log[log["fold"].astype(str) == "oof"]
    best = oof.sort_values(["recall", "f1", "level"], ascending=[False, False, False]).iloc[0]
    cfg = RunConfig(
        level=str(best["level"]),
        missing=str(best["missing"]),
        outlier=str(best["outlier"]),
        feature_set=str(best["feature_set"]),
        scaler=str(best["scaler"]),
        sampling=str(best["sampling"]),
        model=str(best["model"]),
        hpo=str(best["hpo"]),
        threshold=str(best["threshold"]),
    )
    return cfg, str(best["run_id"])


def train_final_model(cfg: RunConfig):
    ensure_dirs()
    train_meta = load_train()
    experiment_frames = load_all_experiments()
    exp_ids = sorted(train_meta["experiment_id"].tolist())
    bounds = fit_outlier_bounds([experiment_frames[i] for i in exp_ids], cfg.outlier)
    feat = build_feature_matrix(experiment_frames, exp_ids, train_meta, cfg.feature_set, cfg.outlier, bounds)
    feat["label"] = train_meta.set_index("experiment_id").loc[exp_ids, "label"].values

    y = feat["label"].values
    train_x = feat.drop(columns=["label"])
    drop_cols = []
    if cfg.missing == "drop_high_missing":
        miss_rate = train_x.isna().mean()
        drop_cols = miss_rate[miss_rate > 0.3].index.tolist()
    X, names = _prepare_matrix(train_x, cfg, drop_cols)
    if cfg.missing in {"median", "mean", "knn_5", "drop_high_missing"}:
        imp = _imputer("median" if cfg.missing == "drop_high_missing" else cfg.missing)
        X = imp.fit_transform(X)
    scaler = _scaler(cfg.scaler)
    if scaler is not None:
        X = scaler.fit_transform(X)
    X, y = _apply_sampling(X, y, cfg.sampling)
    if X is None:
        raise RuntimeError("Sampling failed for final model")
    if cfg.model == "ensemble":
        from sklearn.ensemble import VotingClassifier

        m1 = _base_model("xgboost", cfg.sampling)
        m2 = _base_model("lightgbm", cfg.sampling)
        model = VotingClassifier(
            estimators=[("xgb", m1), ("lgbm", m2)],
            voting="soft",
        )
    else:
        model = _base_model(cfg.model, cfg.sampling)
    model.fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    thr = pick_threshold(y, prob, cfg.threshold)
    bundle = {"model": model, "feature_names": names, "config": cfg.__dict__, "threshold": thr}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "best_model.pkl")
    with (CONFIGS_DIR / "best_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg.__dict__, "threshold_value": float(thr)}, f, allow_unicode=True)
    return bundle


def main():
    cfg, run_id = get_best_config()
    bundle = train_final_model(cfg)
    print(f"Saved best model from {run_id}, threshold={bundle['threshold']:.3f}")


if __name__ == "__main__":
    main()
