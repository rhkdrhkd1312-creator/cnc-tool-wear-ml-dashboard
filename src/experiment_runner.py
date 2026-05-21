from __future__ import annotations

import itertools
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import N_SPLITS, RANDOM_STATE
from src.evaluate import compute_metrics, pick_threshold
from src.features import build_feature_matrix
from src.load_data import load_all_experiments, load_train
from src.preprocess import fit_outlier_bounds

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class RunConfig:
    level: str
    missing: str = "median"
    outlier: str = "doc_rule"
    feature_set: str = "sensor_global"
    scaler: str = "none"
    sampling: str = "none"
    model: str = "xgboost"
    hpo: str = "default"
    threshold: str = "fixed"
    extra: dict | None = None

    def key(self) -> str:
        d = asdict(self)
        d.pop("extra", None)
        return json.dumps(d, sort_keys=True)


def _needs_impute(model_name: str, missing: str) -> bool:
    if missing == "none" and model_name in {"xgboost", "lightgbm", "random_forest"}:
        return False
    return missing != "none"


def _imputer(missing: str):
    if missing == "none":
        return "passthrough"
    if missing == "median":
        return SimpleImputer(strategy="median")
    if missing == "mean":
        return SimpleImputer(strategy="mean")
    if missing == "knn_5":
        return KNNImputer(n_neighbors=5)
    raise ValueError(missing)


def _scaler(name: str):
    if name in ("none", "passthrough"):
        return None
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(name)


def _apply_sampling(X: np.ndarray, y: np.ndarray, sampling: str):
    if sampling == "none":
        return X, y
    n_minority = min(np.bincount(y))
    k = max(1, min(3, n_minority - 1))
    try:
        if sampling == "smote":
            return SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X, y)
        if sampling == "borderline_smote":
            return BorderlineSMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X, y)
        if sampling == "adasyn":
            return ADASYN(n_neighbors=k, random_state=RANDOM_STATE).fit_resample(X, y)
        if sampling == "rus_smote":
            X_u, y_u = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X, y)
            return SMOTE(k_neighbors=max(1, min(k, min(np.bincount(y_u)) - 1)), random_state=RANDOM_STATE).fit_resample(X_u, y_u)
        if sampling == "tomek_smote":
            return SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=RANDOM_STATE), random_state=RANDOM_STATE).fit_resample(X, y)
    except Exception:
        return None, None
    return X, y


def _base_model(name: str, sampling: str, params: dict | None = None) -> Any:
    params = params or {}
    cw = "balanced" if sampling == "class_weight" else None
    spw = params.get("scale_pos_weight", 1.0)
    if name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    if name == "logistic":
        return LogisticRegression(C=params.get("C", 1.0), class_weight=cw, max_iter=2000, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 3),
            class_weight=cw,
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 200),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            scale_pos_weight=spw if sampling == "scale_pos_weight" else 1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            num_leaves=params.get("num_leaves", 8),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 200),
            min_child_samples=params.get("min_child_samples", 1),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 0.0),
            class_weight=cw,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    if name == "svc":
        return SVC(C=params.get("C", 1.0), kernel="rbf", probability=True, class_weight=cw, random_state=RANDOM_STATE)
    raise ValueError(name)


def _prepare_matrix(
    X_df: pd.DataFrame,
    cfg: RunConfig,
    drop_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    X = X_df.copy()
    if cfg.missing == "drop_high_missing" and drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No features remaining after drop_high_missing")
    feature_names = list(X.columns)
    return X.values.astype(float), feature_names


def _fit_transform_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: RunConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    train_x = train_df.drop(columns=["label"])
    val_x = val_df.drop(columns=["label"])

    drop_cols = []
    if cfg.missing == "drop_high_missing":
        miss_rate = train_x.isna().mean()
        drop_cols = miss_rate[miss_rate > 0.3].index.tolist()

    X_train, feat_names = _prepare_matrix(train_x, cfg, drop_cols)
    X_val, _ = _prepare_matrix(val_x, cfg, drop_cols)
    if X_train.shape[1] == 0:
        return None

    if cfg.missing in {"median", "mean", "knn_5", "drop_high_missing"}:
        imp = _imputer("median" if cfg.missing == "drop_high_missing" else cfg.missing)
        X_train = imp.fit_transform(X_train)
        X_val = imp.transform(X_val)

    scaler = _scaler(cfg.scaler)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    X_res, y_res = _apply_sampling(X_train, y_train, cfg.sampling)
    if X_res is None:
        return None
    return X_res, y_res, X_val, y_val, feat_names


def run_single_config(cfg: RunConfig, experiment_frames: dict, train_meta: pd.DataFrame) -> dict | None:
    exp_ids = sorted(train_meta["experiment_id"].tolist())
    y_all = train_meta.set_index("experiment_id").loc[exp_ids, "label"].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_prob = np.zeros(len(exp_ids))
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(exp_ids, y_all)):
        train_ids = [exp_ids[i] for i in tr_idx]
        val_ids = [exp_ids[i] for i in va_idx]
        train_frames = [experiment_frames[i] for i in train_ids]
        bounds = fit_outlier_bounds(train_frames, cfg.outlier)

        train_feat = build_feature_matrix(experiment_frames, train_ids, train_meta, cfg.feature_set, cfg.outlier, bounds)
        val_feat = build_feature_matrix(experiment_frames, val_ids, train_meta, cfg.feature_set, cfg.outlier, bounds)
        train_feat["label"] = train_meta.set_index("experiment_id").loc[train_ids, "label"].values
        val_feat["label"] = train_meta.set_index("experiment_id").loc[val_ids, "label"].values

        prep = _fit_transform_fold(train_feat, val_feat, cfg)
        if prep is None:
            return None
        X_tr, y_tr, X_va, y_va, _ = prep
        if X_tr.shape[1] == 0 or X_va.shape[1] == 0:
            return None

        if cfg.model == "ensemble":
            m1 = _base_model("xgboost", cfg.sampling)
            m2 = _base_model("lightgbm", cfg.sampling)
            m1.fit(X_tr, y_tr)
            m2.fit(X_tr, y_tr)
            prob_tr = (m1.predict_proba(X_tr)[:, 1] + m2.predict_proba(X_tr)[:, 1]) / 2
            prob_va = (m1.predict_proba(X_va)[:, 1] + m2.predict_proba(X_va)[:, 1]) / 2
        else:
            model = _base_model(cfg.model, cfg.sampling)
            model.fit(X_tr, y_tr)
            prob_tr = model.predict_proba(X_tr)[:, 1]
            prob_va = model.predict_proba(X_va)[:, 1]
        thr = pick_threshold(y_tr, prob_tr, cfg.threshold)
        m = compute_metrics(y_va, prob_va, thr)
        m.update({"fold": fold, "threshold": thr})
        fold_rows.append(m)
        for i, eid in enumerate(val_ids):
            oof_prob[exp_ids.index(eid)] = prob_va[i]

    thr_oof = pick_threshold(y_all, oof_prob, cfg.threshold)
    oof_metrics = compute_metrics(y_all, oof_prob, thr_oof)
    return {
        "config": cfg,
        "fold_metrics": fold_rows,
        "oof_metrics": oof_metrics,
        "oof_prob": oof_prob,
        "experiment_ids": exp_ids,
        "threshold": thr_oof,
    }


def cartesian_grid(level: str, grid: dict) -> list[RunConfig]:
    keys = ["missing", "outlier", "feature_set", "scaler", "sampling", "model", "hpo", "threshold"]
    values = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        d = dict(zip(keys, combo))
        configs.append(RunConfig(level=level, **d))
    return configs
