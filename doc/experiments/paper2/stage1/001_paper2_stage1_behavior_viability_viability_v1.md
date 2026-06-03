# Paper2 Stage1 Behavior Viability

Generated: 2026-06-03 11:08:41

## Result

This is a behavior-level viability screen, not a mechanism claim. It tests whether replacing the image with a wrong image changes target rank/logit/margin/decoded answer under the same question and answer target.

## Model Summary

| model | paired samples | rank worsened | mean target logit drop | mean margin drop | decoded changed | clean exact | wrong-image exact | strong cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma | 40 | 0.750 | 9.078 | 7.678 | 0.875 | 0.100 | 0.025 | 8 |
| qwen | 40 | 0.900 | 4.828 | 4.979 | 0.900 | 0.275 | 0.000 | 8 |

## Interpretation

- Both models show image-dependent behavior: wrong-image substitution usually worsens target rank/logit and changes decoded answers.
- Qwen has higher clean exact-answer rate in this screen, but both models lose answer support under wrong images.
- The selected strong cases are the recommended input for Paper2 Stage2 mechanism smoke; they should be treated as case-selection diagnostics, not final evidence.

## Outputs

- `paper2_stage1_behavior_viability_v1_summary.csv`
- `paper2_stage1_behavior_viability_v1_paired.csv`
- `paper2_stage1_behavior_viability_v1_strong_cases.csv`
- `paper2_stage1_behavior_viability_v1_decision.json`
