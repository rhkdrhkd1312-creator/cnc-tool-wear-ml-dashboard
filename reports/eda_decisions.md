# EDA Design Decisions

## Summary
- Experiments: 18 (unworn 8, worn 10)
- Primary task: binary `tool_condition` classification
- Evaluation unit: 18 experiments (not time-series rows)

## Findings → Decisions

| EDA finding | Design decision | Validation experiment |
|-------------|-----------------|----------------------|
| feed_rate / clamp_pressure vary with label | Run sensor-only vs full feature sets | `feature_set`: sensor_global vs full |
| Machining_Process phases show spindle current patterns | Add phase-wise aggregated features | `feature_set`: sensor_global vs sensor_phase |
| doc_rule removes ~100.0% rows on average | Compare outlier filter on/off | `outlier`: none vs doc_rule |
| Class imbalance 10:8 | sampling + class_weight experiments | `sampling`: * |
| FN (miss worn) cost > FP | primary metric = Recall(worn) | `threshold`: fixed vs recall_target |

## Confounding check
See `cross_tab.csv` — feed_rate and clamp_pressure are associated with tool_condition; sensor-only ablation is required.

## Feature ranking (Mann-Whitney)
See `mannwhitney_features.csv` for top sensor candidates.

## Figures
- `figures/tool_feed_cross.png`
