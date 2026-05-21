# Model Improvement Plan v2

> Streamlit 대시보드 Tab1–3 분석 인사이트 기반 재계획 (2026-05-21)

---

## 1. 현재 상태 진단

### 1.1 성능 스냅샷

| 항목 | 값 | 의미 |
|------|-----|------|
| Best run | `run_0021` (L9, XGB+LGBM ensemble) | OOF Recall=1.00, FN=0, FP=0 |
| 단순 best | `run_0006` (L2, xgboost + doc_rule) | OOF Recall=1.00 — ensemble 추가 이득 없음 |
| 데이터 규모 | 18 experiments (unworn 8 / worn 10) | Recall 1건 = ±5.6%p |
| 평가 방식 | Stratified 5-fold OOF | 실험 단위 공정 CV |

### 1.2 대시보드에서 확인된 4가지 핵심 인사이트

**인사이트 ① — doc_rule이 유일한 게임 체인저**

| outlier 설정 | Mean OOF Recall | run 수 |
|--------------|-----------------|--------|
| `none` | **0.46** (0.10–0.60) | 5 |
| `doc_rule` | **1.00** | 14 |

→ L2 이후 scaler / class_weight / ensemble / HPO는 OOF Recall 변화 없음 (Wilcoxon p≈1, ceiling effect).

**인사이트 ② — wear signal은 spindle current가 아닌 축 kinematics**

Top feature (ensemble importance):
1. `Z1_ActualPosition_p95` (84.5)
2. `X1_ActualPosition_mean` (46.0)
3. `X1_ActualPosition_std` (16.0)
4. `X1_ActualVelocity_mean` (2.5)

→ S1_CurrentFeedback 계열은 importance 하위. Mann-Whitney로 important feature 다수가 worn/unworn 간 유의 (p<0.05).

**인사이트 ③ — OOF perfect ≠ 실전 generalization**

- Best model OOF 오류 0건 → Tab3 오분류 패턴 분석 불가
- 오분류는 전부 doc_rule **미적용** run에서만 발생 (`run_0001`~`run_0005`)
- near-threshold 오류: threshold 조정으로 구제 가능
- high-margin 오류: feature overlap / label confounding — 구조적 문제

**인사이트 ④ — 라벨·공변량 리스크**

| 리스크 | 내용 |
|--------|------|
| Label audit | 6 experiments 재검토 필요 (13,14,15,18: worn+visual pass / 7,16: worn+가공 미완료) |
| Confounding | feed_rate·clamp_pressure가 tool_condition과 연관 (`cross_tab.csv`) — sensor-only ablation 필수 |
| doc_rule | 실험별 필터율 ~100% (EDA) — production 재현·문서화 필요 |

### 1.3 전략적 결론

> **OOF 튜닝은 종료.** 다음 단계는 (1) hold-out으로 진짜 성능 검증, (2) 라벨 정합성 확보, (3) confounding 분리, (4) doc_rule 없이도 버티는 sensor signal 강화.

---

## 2. 의사결정 게이트 (Go / No-Go)

각 Phase 완료 후 아래 기준으로 다음 단계 진입 여부 결정.

```
Phase 0 (라벨 audit)
    └─ 6건 review 완료? ──No──→ 라벨 수정 후 재학습
           │
          Yes
           ▼
Phase 1 (hold-out 검증)
    └─ doc_rule+sensor Recall ≥ 0.95? ──No──→ Phase 2 (feature) + Phase 3 (모델)
           │
          Yes
           ▼
Phase 4 (배포 준비)
```

| Gate | 조건 | 미달 시 |
|------|------|---------|
| G0 | Label audit 6건 해소 | 라벨 수정 → 전 실험 재학습 |
| G1 | Hold-out Recall ≥ 0.95 (doc_rule+sensor) | Phase 2 feature engineering 우선 |
| G2 | Hold-out Recall ≥ 0.85 (sensor-only, no doc_rule) | doc_rule을 hard dependency로 명시, 추가 데이터 수집 |
| G3 | Hold-out FN ≤ 1 | threshold 재조정 → 그래도 실패 시 Phase 3 |

---

## 3. 실행 로드맵

### Phase 0 — Label & Data Integrity (P0, 1주)

**목표:** 모델이 학습하는 "정답" 자체를 신뢰할 수 있게 만든다.

| # | 작업 | 상세 | 산출물 |
|---|------|------|--------|
| 0-1 | Label audit 6건 review | exp 7,13,14,15,16,18 — domain expert 육안·공정 기록 대조 | `reports/label_audit_resolved.csv` |
| 0-2 | Confounding 매트릭스 정리 | feed_rate × clamp × label 교차표 + sensor-only 분리 가능 실험 식별 | `reports/confounding_matrix.md` |
| 0-3 | doc_rule 필터율 문서화 | 실험별 제거 행 비율, 제거 조건 (feedrate=50, X position=198) | `reports/doc_rule_rates.csv` (기존 확장) |

**완료 기준:** audit flag 0건, 수정 라벨 반영 후 `experiment_features.csv` 재생성.

---

### Phase 1 — Hold-out Validation (P0, 1–2주)

**목표:** OOF 100%가 hold-out에서도 성립하는지 검증. **모든 후속 modeling의 전제.**

| # | 작업 | 상세 | 산출물 |
|---|------|------|--------|
| 1-1 | Hold-out split 설계 | 18실험 중 6건 hold-out (stratified: unworn 3 / worn 3), CV 12건 | `configs/holdout_split.yaml` |
| 1-2 | 3-track ablation | (A) sensor-only + no doc_rule, (B) sensor-only + doc_rule, (C) full + doc_rule | `reports/holdout_results.csv` |
| 1-3 | Bootstrap 95% CI | OOF Recall point estimate + CI (n=18 한계 명시) | `reports/recall_bootstrap_ci.md` |
| 1-4 | Threshold sweep | recall_target 0.85–0.99 on hold-out, FN/FP trade-off curve | `figures/threshold_sweep.png` |

**핵심 질문 (hold-out으로 답해야 할 것):**
1. doc_rule 없이 sensor-only Recall은? → production doc_rule 의존도 판단
2. ensemble vs xgboost 단독 차이는? → 현재 OOF상 차이 없음, hold-out에서 재확인
3. class_weight 효과는? → doc_rule 이후 marginal 가능성 높음

**완료 기준:** G1, G2, G3 통과 + CI 리포트 작성.

---

### Phase 2 — Feature Engineering (P1, hold-out 결과 후)

**목표:** doc_rule 없이도 분리력을 높이고, kinematics signal을 더 정교하게 포착.

**근거 (Tab2):** Z1/X1 position·velocity가 dominant → phase별·구간별 통계로 신호 강화.

| # | 작업 | 우선순위 | 기대 효과 |
|---|------|----------|-----------|
| 2-1 | **Phase-wise features** | ★★★ | Layer 2/3 Up·Down 구간별 X1/Z1 position·velocity p95/mean/std (`sensor_phase`) |
| 2-2 | **Kinematics delta features** | ★★☆ | Command vs Actual position/velocity gap — 제어 오차가 wear와 연관 가능 |
| 2-3 | **Meta features** | ★★☆ | feed_rate, clamp_pressure, feed×clamp interaction — confounding 분리용 (ablation 필수) |
| 2-4 | **Feature selection** | ★☆☆ | importance 하위 50% pruning → hold-out Recall 유지 확인 |

**검증 방법:** 각 feature set 추가 시 hold-out Track (A)(B) Recall 변화만 측정. OOF는 참고용.

**완료 기준:** sensor-only (no doc_rule) hold-out Recall ≥ 0.85 **또는** doc_rule+sensor ≥ 0.95 유지하며 FP 감소.

---

### Phase 3 — Modeling (P1, Phase 1–2 후 필요 시만)

**전제:** Phase 1 hold-out Recall < 0.95일 때만 착수. OOF 추가 튜닝은 기대 ROI 낮음.

| # | 작업 | 조건 | 비고 |
|---|------|------|------|
| 3-1 | XGB/LGBM Optuna HPO | hold-out Recall < 0.95 | inner 3-fold on CV 12건, hold-out은 touch 금지 |
| 3-2 | Calibrated threshold | FN ≤ 1 목표 | Platt scaling or isotonic on CV, threshold on hold-out |
| 3-3 | Ensemble 유지 vs 단순화 | hold-out에서 ensemble > xgboost | OOF상 동일 → hold-out에서 xgboost 단독 채택 가능 (운영 단순화) |
| 3-4 | 1D CNN + Grad-CAM | parallel track | n=18 한계, tabular primary 유지. Grad-CAM으로 time-step attribution 탐색 |

**완료 기준:** hold-out Recall ≥ 0.95, FN ≤ 1.

---

### Phase 4 — Deployment Readiness (P2)

**목표:** production pipeline으로 이식 가능한 형태로 고정.

| # | 작업 | 상세 |
|---|------|------|
| 4-1 | Pipeline freeze | doc_rule → feature → impute → model → threshold 순서 코드화 |
| 4-2 | Streamlit monitoring | prob drift 알림, FN 발생 시 alert, experiment-level dashboard |
| 4-3 | Ensemble fallback | XGB vs LGBM prob 차이 > 0.3 → human review flag |
| 4-4 | Operating point 문서 | Recall ≥ 0.95 보장 threshold + expected FP rate |
| 4-5 | Model card | 학습 데이터, hold-out 성능, known limitations, label audit 결과 |

---

## 4. 우선순위 요약

```
[지금 당장]  Phase 0 — 라벨 6건 review
[다음]       Phase 1 — hold-out 3-track ablation + Bootstrap CI
[그 다음]    Phase 2 — phase-wise + kinematics delta features (hold-out 기준)
[필요 시]    Phase 3 — HPO / calibration (hold-out 미달일 때만)
[마지막]     Phase 4 — pipeline freeze + monitoring
```

**하지 않을 것 (OOF 인사이트 기반):**
- L3–L9 추가 OOF 실험 (ceiling effect, ROI ≈ 0)
- ensemble 복잡도 증가 (OOF상 xgboost 단독과 동일)
- S1 current 중심 feature engineering (importance 하위)

---

## 5. 성공 기준 (최종)

| Metric | Target | 측정 방법 |
|--------|--------|-----------|
| Hold-out Recall (doc_rule + sensor) | **≥ 0.95** | 6건 hold-out, stratified |
| Hold-out Recall (sensor-only, no doc_rule) | **≥ 0.85** | confounding·generalization 하한 |
| Hold-out FN | **≤ 1** | worn miss 최소화 (primary metric) |
| Hold-out FP | 최소화 (2순위) | threshold sweep으로 trade-off 명시 |
| Label audit | **0 unresolved flags** | Phase 0 완료 |
| Bootstrap 95% CI | 리포트 필수 | OOF + hold-out 각각 |
| doc_rule | production 문서화 + 재현 | 실험별 filter rate 기록 |

---

## 6. 리스크 & 대응

| 리스크 | 가능성 | 영향 | 대응 |
|--------|--------|------|------|
| Hold-out Recall << OOF | 중 | 높음 | doc_rule 의존 명시, 추가 실험 수집 (feed/clamp 다양화) |
| Label 6건 중 다수 수정 | 중 | 중 | 수정 후 전체 재학습, hold-out 재설계 |
| n=18 CI 넓음 | 높음 | 중 | Bootstrap CI 병기, "point estimate만으로 판단 금지" 원칙 |
| feed_rate confounding | 높음 | 중 | sensor-only track + meta feature ablation으로 분리 |
| Phase feature 효과 미미 | 중 | 낮 | kinematics delta(2-2) 시도, 1D CNN parallel track |

---

## 7. 다음 액션 (Immediate Next Steps)

1. **exp 7, 13, 14, 15, 16, 18** 라벨 review 일정 잡기
2. **hold-out split** 설계 (`configs/holdout_split.yaml` 작성)
3. **3-track hold-out 실험** 스크립트 추가 (`src/run_holdout.py`)
4. hold-out 결과 나오면 이 문서 **v3** 업데이트

---

## Appendix — Dashboard ↔ Plan 매핑

| Dashboard Tab | 발견 | Plan Phase |
|---------------|------|------------|
| Tab1 실험 결과 | doc_rule → Recall 0.46→1.00, L2+ ceiling | Phase 1 ablation, Phase 3 조건부 |
| Tab2 입력 변수 | Z1/X1 kinematics dominant, confounding | Phase 2 feature engineering |
| Tab3 오분류 | near-threshold vs high-margin, label audit 6건 | Phase 0 label, Phase 1 threshold sweep |
