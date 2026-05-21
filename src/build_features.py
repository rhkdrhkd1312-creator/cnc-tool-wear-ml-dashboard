"""Build experiment-level feature table."""
from src.config import DATA_PROCESSED
from src.features import build_feature_matrix
from src.load_data import ensure_dirs, load_all_experiments, load_train
from src.preprocess import fit_outlier_bounds

if __name__ == "__main__":
    ensure_dirs()
    meta = load_train()
    frames = load_all_experiments()
    ids = sorted(meta.experiment_id.tolist())
    bounds = fit_outlier_bounds([frames[i] for i in ids], "doc_rule")
    feat = build_feature_matrix(frames, ids, meta, "sensor_global", "doc_rule", bounds)
    feat["label"] = meta.set_index("experiment_id").loc[ids, "tool_condition"].values
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "experiment_features.csv"
    feat.reset_index().to_csv(out, index=False)
    print(f"Wrote {out} ({len(feat)} rows)")
