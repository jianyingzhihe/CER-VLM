# Paper2 Stage2 Hidden-Lens Smoke

Generated: 2026-06-03 11:28:01

## Result

This smoke uses two common Paper2 behavior-strong cases and applies the same hidden restore lens to Gemma and Qwen. It is a mechanism viability check, not a final Paper2 claim.

| model | target rows | samples | target effect mean | positive frac | target-minus-wrong mean | real-minus-control mean |
|---|---:|---|---:|---:|---:|---:|
| gemma | 64 | okvqa_val_00893, okvqa_val_02108, okvqa_val_1295955, okvqa_val_258605 | 2.879 | 0.609 | 4.844 | 0.210 |
| qwen | 240 | okvqa_val_00310, okvqa_val_00893, okvqa_val_02108, okvqa_val_1295955, okvqa_val_258605 | 0.646 | 0.646 | -0.480 | 0.421 |

## Notes

- The input samples are selected by Paper2 Stage1 clean-vs-wrong-image behavior differences.
- The masks for this smoke use Stage2 hidden-lens assets: `answer/union` are rendered from evidence-region union, and `shifted/shuffled` are spatial controls.
- Because Gemma clean target ranks were weaker than the Stage6 mainline gate, this smoke uses `max_clean_rank=1000`; it should be reported as exploratory.
