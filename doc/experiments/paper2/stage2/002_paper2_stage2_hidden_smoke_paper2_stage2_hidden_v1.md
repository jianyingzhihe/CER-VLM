# Paper2 Stage2 Hidden-Lens Smoke

Generated: 2026-06-03 11:21:49

## Result

This smoke uses two common Paper2 behavior-strong cases and applies the same hidden restore lens to Gemma and Qwen. It is a mechanism viability check, not a final Paper2 claim.

| model | target rows | samples | target effect mean | positive frac | target-minus-wrong mean | real-minus-control mean |
|---|---:|---|---:|---:|---:|---:|
| gemma | 32 | okvqa_val_00893, okvqa_val_258605 | 5.096 | 0.781 | 7.284 | -2.052 |
| qwen | 96 | okvqa_val_00893, okvqa_val_258605 | 1.090 | 0.750 | -1.562 | 0.855 |

## Notes

- The input samples are selected by Paper2 Stage1 clean-vs-wrong-image behavior differences.
- The masks for this smoke use Stage2 hidden-lens assets: `answer/union` are rendered from evidence-region union, and `shifted/shuffled` are spatial controls.
- Because Gemma clean target ranks were weaker than the Stage6 mainline gate, this smoke uses `max_clean_rank=1000`; it should be reported as exploratory.
