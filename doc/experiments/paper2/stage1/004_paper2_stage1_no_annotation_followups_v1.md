# Paper2 Stage1 No-Annotation Follow-ups

Generated: 2026-06-03 13:57:19

## What Was Done

This analysis reuses existing Stage1 behavior and A/B mask-composition outputs. It does not require new annotation or GPU runs.

It checks threshold robustness of the multi-evidence label, identifies samples whose label is stable across stricter thresholds, and extracts the next annotation candidates from behavior-strong samples that do not yet have A/B masks.

## Main Readout

- Robust multi-evidence rows across threshold settings: `5`.
- Robust multi-evidence by model: `{'gemma': 3, 'qwen': 2}`.
- High-priority unannotated behavior cases: `3`.
- Medium-priority unannotated behavior cases: `13`.

## Interpretation

The existing Stage1 data can support case selection and sample triage, but it still does not provide enough robust multi-evidence samples for a broad Paper2 mechanism claim. The next bottleneck is annotation expansion, especially for cases with clean exact answers and strong wrong-image sensitivity.

## Outputs

- `paper2_stage1_no_annotation_followups_v1_threshold_sweep.csv`
- `paper2_stage1_no_annotation_followups_v1_sample_stability.csv`
- `paper2_stage1_no_annotation_followups_v1_annotation_candidates.csv`
- `paper2_stage1_no_annotation_followups_v1_decision.json`
