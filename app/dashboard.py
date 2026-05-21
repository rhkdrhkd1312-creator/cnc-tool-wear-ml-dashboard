"""CNC Tool Wear ML dashboard — 3-tab analysis UI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from app.dashboard_utils import (
    NEAR_THRESHOLD_BAND,
    ablation_summary,
    build_prediction_frame,
    feature_importance_df,
    get_oof_runs,
    label_audit,
    level_progression,
    load_experiment_log,
    load_feature_matrix,
    load_best_bundle,
    mannwhitney_top_bottom,
    metrics_at_threshold,
    runs_with_errors,
    spearman_with_label,
    top_feature_correlations,
)
from src.config import CONFIGS_DIR, REPORTS_DIR
from src.load_data import load_experiment_timeseries, load_train

DASHBOARD_VERSION = "v2.0 (3-tab)"

st.set_page_config(page_title="CNC Tool Wear ML Dashboard", layout="wide", page_icon="🔧")
st.title("CNC Mill Tool Wear — 분석 대시보드")
st.caption(f"실험 결과 · 입력 변수 · 오분류 심화 분석 (Stratified 5-fold OOF, n=18) · **{DASHBOARD_VERSION}**")

GLOSSARY = {
    "Recall": "실제 worn 중 모델이 worn으로 맞춘 비율 (FN 최소화 목표)",
    "OOF": "Out-of-Fold: 18실험 각각 공정한 예측",
    "doc_rule": "문서 기준 CNC 센서 오류 행 제거 (feedrate=50 또는 X position=198)",
}


@st.cache_data
def cached_log():
    return load_experiment_log()


@st.cache_data
def cached_features():
    return load_feature_matrix()


@st.cache_data
def cached_importance():
    return feature_importance_df()


log = cached_log()
if log.empty:
    st.warning("실험 로그가 없습니다. 먼저 `python -m src.run_experiments` 를 실행하세요.")
    st.stop()

oof = get_oof_runs(log)
best_row = oof.sort_values(["recall", "f1", "level"], ascending=[False, False, False]).iloc[0]
bundle = load_best_bundle()

tab1, tab2, tab3 = st.tabs(["실험 결과", "입력 변수 분석", "오분류 데이터 심화 분석"])

# ── Tab 1: 실험 결과 ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("전체 실험 테이블 (OOF)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best run", best_row["run_id"])
    c2.metric("Best OOF Recall", f"{best_row['recall']:.2f}", help=GLOSSARY["Recall"])
    c3.metric("Best config", f"{best_row['model']} / {best_row['outlier']}")

    levels = st.multiselect(
        "Level 필터",
        sorted(oof["level"].unique()),
        default=sorted(oof["level"].unique()),
    )
    view = oof[oof.level.isin(levels)].sort_values(["recall", "f1"], ascending=False)
    st.dataframe(
        view[
            [
                "run_id",
                "level",
                "model",
                "missing",
                "outlier",
                "feature_set",
                "sampling",
                "threshold",
                "recall",
                "f1",
                "precision",
                "roc_auc",
                "fn_count",
                "fp_count",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Level별 최고 성능 추이")
    prog = level_progression(log)
    fig_prog = px.line(
        prog,
        x="level",
        y="recall",
        markers=True,
        title="Level progression — best OOF Recall",
        labels={"recall": "OOF Recall", "level": "Level"},
    )
    fig_prog.add_hline(y=1.0, line_dash="dot", annotation_text="ceiling")
    st.plotly_chart(fig_prog, use_container_width=True)

    st.markdown("### 통계 검증")
    stat_path = REPORTS_DIR / "statistical_comparison.md"
    if stat_path.exists():
        st.markdown(stat_path.read_text(encoding="utf-8"))
    else:
        st.info("statistical_comparison.md 가 없습니다. `python -m src.run_stat_tests` 를 실행하세요.")

    st.markdown("### 방법론 해석")
    st.markdown(
        """
| 단계 | 핵심 변화 | OOF Recall | 해석 |
|------|-----------|------------|------|
| **L0 → L1** | dummy → logistic + sensor_global | 0.10 → 0.60 | 센서 집계 feature만으로도 worn/unworn 분리 가능성 확인 |
| **L1 → L2** | xgboost + (아직 doc_rule 없음) | 0.60 → 0.40* | tree는 비선형 포착하나 outlier 미제거 시 Recall 하락 |
| **L2+ (doc_rule)** | doc_rule outlier filter | → **1.00** | 문서 기준 센서 오류 행 제거가 Recall을 결정적으로 개선 |
| **L3–L9** | scaler, class_weight, ensemble | 1.00 유지 | ceiling effect — Wilcoxon p≈1 (통계적 유의 개선 아님) |

*L2 without doc_rule (`run_0005`)는 Recall=0.40 — doc_rule 효과를 분리 확인하는 ablation 포인트.

**핵심 인사이트**
- **doc_rule**이 가장 큰 효과 (mean Recall: outlier=none ~0.46 → doc_rule ~1.00)
- **class_weight** + tree/ensemble은 doc_rule 이후 marginal (이미 완벽 분류)
- n=18 한계: fold-level Wilcoxon은 보조 지표, OOF + hold-out이 primary evidence
        """
    )

    st.markdown("### Preprocessing ablation")
    abl = ablation_summary(log)
    abl_axis = st.selectbox("Ablation axis", list(abl.keys()), key="abl_axis")
    fig_abl = px.bar(
        abl[abl_axis],
        x=abl_axis,
        y="recall",
        title=f"Mean OOF Recall by {abl_axis}",
        labels={"recall": "Mean OOF Recall"},
    )
    st.plotly_chart(fig_abl, use_container_width=True)

    st.markdown("### 향후 시도 방안")
    st.markdown(
        """
1. **Hold-out 검증** — 6+ 실험을 CV에서 완전히 분리 (현재 OOF Recall=1.0은 generalization 미검증)
2. **sensor-only ablation** — doc_rule 없이 sensor feature만으로 hold-out Recall 측정
3. **Phase-wise feature** — Layer 2/3 spindle 구간별 통계 (sensor_phase 확장)
4. **Optuna HPO** — inner 3-fold로 XGB/LGBM 하이퍼파라미터 탐색
5. **1D CNN + Grad-CAM** — raw time-series parallel track (n=18 주의, Tab2 참고)
6. **운영 threshold calibration** — Recall≥0.95 목표 operating point
        """
    )

    cfg_path = CONFIGS_DIR / "best_config.yaml"
    if cfg_path.exists():
        with st.expander("Best config (YAML)"):
            st.code(yaml.safe_load(cfg_path.read_text(encoding="utf-8")), language="yaml")

# ── Tab 2: 입력 변수 분석 ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Feature Importance & EDA")

    if bundle is None:
        st.warning("best_model.pkl 이 없습니다. `python -m src.train` 을 실행하세요.")
    else:
        st.info(
            "본 프로젝트는 tabular ML (XGB + LGBM ensemble)입니다. "
            "딥러닝/Grad-CAM 대신 tree **Feature Importance** + 통계 검定 + EDA로 대체합니다."
        )

        imp = cached_importance()
        if imp.empty:
            st.warning("Feature importance를 계산할 수 없습니다.")
        else:
            top_n = st.slider("Top-N features", 5, 25, 12, key="top_n")
            top_imp = imp.head(top_n)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                fig_imp = px.bar(
                    top_imp.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top {top_n} features (ensemble average importance)",
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            features = cached_features()
            mw = mannwhitney_top_bottom(features, imp, n=min(10, top_n))
            with col_b:
                if not mw.empty:
                    st.markdown("**Important vs Unimportant — Mann-Whitney U (worn vs unworn)**")
                    st.dataframe(
                        mw.style.format(
                            {
                                "worn_mean": "{:.3f}",
                                "unworn_mean": "{:.3f}",
                                "delta": "{:.3f}",
                                "p_value": "{:.4f}",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    sig = mw[(mw["p_value"] < 0.05) & mw["group"].eq("important")]
                    if not sig.empty:
                        st.success(
                            f"유의한 important feature {len(sig)}개: "
                            + ", ".join(sig["feature"].head(5).tolist())
                        )

            st.markdown("### Top feature — worn/unworn 분포")
            feat_pick = st.selectbox("Feature 선택", top_imp["feature"].tolist(), key="feat_box")
            if not features.empty and feat_pick in features.columns:
                plot_df = features[["experiment_id", feat_pick, "label"]].copy()
                fig_box = px.box(
                    plot_df,
                    x="label",
                    y=feat_pick,
                    points="all",
                    hover_data=["experiment_id"],
                    title=f"{feat_pick} by tool condition",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("### Top feature 간 Spearman 상관 + label 상관")
            top_feats = top_imp["feature"].tolist()
            corr = top_feature_correlations(features, top_feats)
            if not corr.empty:
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    title="Spearman correlation among top features",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            label_corr = spearman_with_label(features, top_feats)
            if not label_corr.empty:
                st.dataframe(
                    label_corr.sort_values("spearman_r", key=abs, ascending=False).style.format(
                        {"spearman_r": "{:.3f}", "p_value": "{:.4f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### EDA 인사이트")
            top3 = top_imp.head(3)["feature"].tolist()
            st.markdown(
                f"""
- **Position/Velocity 계열 지배**: 상위 feature는 주로 `X1/Z1 ActualPosition`, `ActualVelocity`, `ActualAcceleration` 통계
- **Spindle(S1) current는 상대적으로 낮은 importance** — 위치/속도 축이 wear signal의 주 경로
- **Z1 position p95** 등 peak statistic이 mean보다 분리력 높은 경우 다수 (outlier 구간 반영)
- feed_rate/clamp_pressure는 feature matrix에 직접 포함되지 않음 → **공변량(confounding)** 가능 (Tab3 label audit)
- Top feature: `{", ".join(top3)}`
                """
            )

# ── Tab 3: 오분류 데이터 심화 분석 ───────────────────────────────────────────
with tab3:
    st.subheader("오분류 케이스 심화 분석")

    err_runs = runs_with_errors(log)
    st.markdown(
        f"""
**Best model (`{best_row['run_id']}`) OOF 오류: FN={int(best_row['fn_count'])}, FP={int(best_row['fp_count'])}**  
→ 완벽 분류 run은 오분류 분석이 불가능합니다. 아래에서 **오류가 있는 run**을 선택해 패턴을 분석합니다.
        """
    )

    default_run = "run_0002" if "run_0002" in err_runs["run_id"].tolist() else err_runs.iloc[0]["run_id"]
    run_labels = [
        f"{r.run_id} ({r.level}, {r.model}, Recall={r.recall:.2f}, err={int(r.fn_count + r.fp_count)})"
        for r in err_runs.itertuples()
    ]
    run_map = {lbl: lbl.split()[0] for lbl in run_labels}
    default_lbl = next(k for k, v in run_map.items() if v == default_run)
    picked_lbl = st.selectbox("분석 run 선택", run_labels, index=run_labels.index(default_lbl))
    run_id = run_map[picked_lbl]

    row = oof[oof.run_id == run_id].iloc[0]
    thr = st.slider(
        "threshold",
        0.05,
        0.95,
        float(row.get("threshold") or 0.5),
        0.01,
        key="err_thr",
        help="worn 확률 cutoff (prob ≥ threshold → worn)",
    )
    m = metrics_at_threshold(run_id, log, thr)
    if m:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Recall", f"{m['recall']:.2f}")
        mc2.metric("F1", f"{m['f1']:.2f}")
        mc3.metric("FN", m["fn_count"])
        mc4.metric("FP", m["fp_count"])

    pred_df = build_prediction_frame(run_id, log)
    if pred_df is None:
        st.error("OOF prediction 파일을 불러올 수 없습니다.")
    else:
        pred_df = pred_df.copy()
        pred_df["pred_at_thr"] = np.where(pred_df["prob_worn"] >= thr, "worn", "unworn")
        pred_df["correct_at_thr"] = pred_df["pred_at_thr"] == pred_df["true_label"]
        pred_df["margin_at_thr"] = np.where(
            pred_df["true_label"] == "worn",
            pred_df["prob_worn"] - thr,
            thr - pred_df["prob_worn"],
        )

        st.markdown("### 전체 예측 결과")
        st.dataframe(
            pred_df[
                [
                    "experiment_id",
                    "true_label",
                    "prob_worn",
                    "pred_at_thr",
                    "correct_at_thr",
                    "margin_at_thr",
                    "feed_rate",
                    "clamp_pressure",
                    "passed_visual_inspection",
                ]
            ].sort_values("prob_worn", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        errors = pred_df[~pred_df["correct_at_thr"]].copy()
        if errors.empty:
            st.success("선택한 threshold에서 오분류가 없습니다.")
        else:
            errors["near_threshold"] = errors["margin_at_thr"].abs() <= NEAR_THRESHOLD_BAND
            near = errors[errors["near_threshold"]].sort_values("margin_at_thr", key=np.abs)
            far = errors[~errors["near_threshold"]].sort_values(
                "margin_at_thr", key=np.abs, ascending=False
            )

            st.markdown(f"### 오분류 비교 (near-threshold ≤ {NEAR_THRESHOLD_BAND})")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("**아쉽게 틀린 케이스 (임계치 근처)**")
                st.dataframe(near, use_container_width=True, hide_index=True)
            with ec2:
                st.markdown("**마진 크게 틀린 케이스**")
                st.dataframe(far, use_container_width=True, hide_index=True)

            if not near.empty and not far.empty:
                compare_cols = ["prob_worn", "feed_rate", "clamp_pressure"]
                compare = pd.DataFrame(
                    {
                        "metric": compare_cols,
                        "near_mean": [near[c].mean() for c in compare_cols],
                        "far_mean": [far[c].mean() for c in compare_cols],
                    }
                )
                st.dataframe(compare, use_container_width=True, hide_index=True)
                st.markdown(
                    """
**패턴 힌트**
- **Near-threshold**: threshold 조정 또는 calibration으로 구제 가능 — FN/FP trade-off 재설정
- **High-margin**: feature 공간에서 클래스 overlap — doc_rule, phase feature, 추가 데이터 필요
- **FN (worn→unworn)**: Recall 목표상 최우선 — feed_rate/clamp 조합과 label confounding 점검
                    """
                )

            st.markdown("### 오분류 실험 — 시계열 드릴다운")
            err_ids = errors["experiment_id"].tolist()
            pick_exp = st.selectbox("오분류 experiment", err_ids)
            err_row = errors[errors.experiment_id == pick_exp].iloc[0]
            st.write(
                f"**exp {pick_exp}** | true={err_row['true_label']} | "
                f"prob={err_row['prob_worn']:.3f} | margin={err_row['margin_at_thr']:.3f} | "
                f"type={'near' if err_row['near_threshold'] else 'far'}"
            )

            ts = load_experiment_timeseries(int(pick_exp))
            sensor = st.selectbox(
                "Sensor",
                [c for c in ts.columns if "CurrentFeedback" in c or "ActualPosition" in c][:8]
                or ts.columns[:5].tolist(),
                key="ts_sensor",
            )
            fig_ts = px.line(ts, y=sensor, title=f"experiment_{pick_exp:02d} — {sensor}")
            st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("### Label audit — 라벨링 의심 케이스")
        audit = label_audit()
        if audit.empty:
            st.success("tool_condition vs visual_inspection / machining_finalized 간 명백한 충돌 없음.")
        else:
            st.warning("아래 실험은 라벨 재검토가 필요할 수 있습니다.")
            st.dataframe(audit, use_container_width=True, hide_index=True)

        st.markdown("### 모델 고도화 아이디어 (오분류·라벨 audit 기반)")
        st.markdown(
            """
1. **doc_rule 파이프라인 고정** — L1/L2(no doc_rule) 오분류의 상당수는 센서 artifact; production에 doc_rule 필수
2. **Threshold 운영점** — near-threshold FN은 recall_target threshold (예: 0.92)로 완화 검토
3. **Confounding 분리** — feed_rate=3 구간은 worn/unworn 모두 존재; clamp/feed interaction feature 추가
4. **Label 재검토** — `worn_but_visual_pass`, `unworn_but_visual_fail` flag 실험 우선 human review
5. **Hold-out + sensor-only** — doc_rule 효과와 sensor generalization을 독립 검증
6. **Phase feature** — high-margin 오분류 실험의 Layer 2/3 spindle 구간 통계 추가
            """
        )

        plan_path = REPORTS_DIR / "model_improvement_plan.md"
        if plan_path.exists():
            with st.expander("model_improvement_plan.md (전체)"):
                st.markdown(plan_path.read_text(encoding="utf-8"))

st.sidebar.markdown("### 용어")
for k, v in GLOSSARY.items():
    st.sidebar.markdown(f"**{k}**: {v}")
st.sidebar.markdown("---")
st.sidebar.code("streamlit run streamlit_app.py", language="bash")
