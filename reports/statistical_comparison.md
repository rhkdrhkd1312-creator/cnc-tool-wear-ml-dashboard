# Statistical Comparison Report

## L0 → L1
- Best L0: `run_0001` OOF Recall=0.100
- Best L1: `run_0002` OOF Recall=0.600
- Δ mean fold Recall: +0.500
- Wilcoxon p=0.2500

## L2 → L3
- Best L2: `run_0006` OOF Recall=1.000
- Best L3: `run_0009` OOF Recall=1.000
- Δ mean fold Recall: +0.000
- Wilcoxon p=1.0000

## L3 → L4
- Best L3: `run_0009` OOF Recall=1.000
- Best L4: `run_0014` OOF Recall=1.000
- Δ mean fold Recall: +0.000
- Wilcoxon p=1.0000

## L4 → L5
- Best L4: `run_0014` OOF Recall=1.000
- Best L5: `run_0015` OOF Recall=1.000
- Δ mean fold Recall: +0.000
- Wilcoxon p=1.0000

## L5 → L6
- Best L5: `run_0015` OOF Recall=1.000
- Best L6: `run_0016` OOF Recall=1.000
- Δ mean fold Recall: +0.000
- Wilcoxon p=1.0000

## L6 → L7
- Best L6: `run_0016` OOF Recall=1.000
- Best L7: `run_0017` OOF Recall=1.000
- Δ mean fold Recall: +0.000
- Wilcoxon p=1.0000

## L7 → L8
- Best L7: `run_0017` OOF Recall=1.000
- Best L8: `run_0018` OOF Recall=1.000
- Δ mean fold Recall: +0.000
- Wilcoxon p=1.0000

## L8 → L9
- Best L8: `run_0018` OOF Recall=1.000
- Best L9: `run_0021` OOF Recall=1.000
- Δ mean fold Recall: +0.000
- Wilcoxon p=1.0000

## Holm-adjusted p-values (level transitions)
- Transition 1: adj p=1.0000
- Transition 2: adj p=1.0000
- Transition 3: adj p=1.0000
- Transition 4: adj p=1.0000
- Transition 5: adj p=1.0000
- Transition 6: adj p=1.0000
- Transition 7: adj p=1.0000
- Transition 8: adj p=1.0000

## n=18 limitation
- Fold-level Wilcoxon is auxiliary; OOF + Bootstrap CI are primary evidence.
- Recall +1 case ≈ +5.6%p on 18 experiments.