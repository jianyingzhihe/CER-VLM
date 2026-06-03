# Paper2 Current Status

Updated: 2026-06-03 12:31 CST

## Stage1 Behavior Viability

Paper2 Stage1 `viability_v1` completed on gpu1/AutoDL for both models:

| model | rows | paired samples | rank worsened | mean target logit drop | decoded changed | strong cases |
|---|---:|---:|---:|---:|---:|---:|
| Gemma | 80 | 40 | 0.750 | 9.078 | 0.875 | 8 |
| Qwen | 80 | 40 | 0.900 | 4.828 | 0.900 | 8 |

Interpretation: wrong-image substitution reliably changes behavior in both models. This supports Paper2 as a viable follow-up direction, but Stage1 is behavior-level evidence only, not a mechanism claim.

## Stage1 Mask Composition

Paper2 `mask_v1` first ran on five behavior-strong samples with valid rendered A/B/union/random masks. We then expanded to `mask_full_v1`: all 15 Stage1 candidates for which A/B evidence masks could be rendered from available LabelMe annotations.

Full `mask_full_v1` summary:

| model | n | multi-evidence supported | one-region dominated | shortcut/prior | mixed | blocked |
|---|---:|---:|---:|---:|---:|---:|
| Gemma | 15 | 1 | 7 | 3 | 3 | 1 |
| Qwen | 15 | 3 | 9 | 2 | 1 | 0 |

Interpretation: this smoke is valuable precisely because it separates two Paper2 populations. Wrong-image sensitivity alone does not guarantee a compositional two-region mechanism. Some behavior-strong cases are genuinely multi-evidence, while others are one-region dominated. Paper2 should therefore use mask composition as a sample filter before claiming multi-region evidence routing.

## Stage2 Hidden-Lens Mechanism Smoke

The first common-case smoke used two samples shared by Gemma/Qwen strong-case lists: `okvqa_val_258605` and `okvqa_val_00893`.

The expanded smoke used five unique strong cases selected from both models, with Stage2 hidden-lens masks rendered from available LabelMe evidence regions when Paper2 mask PNGs were absent.

| model | target rows | usable samples | target effect mean | positive frac | target-minus-wrong mean | real-minus-control mean |
|---|---:|---:|---:|---:|---:|---:|
| Gemma | 64 | 4 | 2.879 | 0.609 | 4.844 | 0.210 |
| Qwen | 240 | 5 | 0.646 | 0.646 | -0.480 | 0.421 |

Interpretation: both models show a positive hidden-restore target effect on Paper2 behavior-strong cases. Gemma is cleaner under target-minus-wrong; Qwen shows positive target movement but also competition-token movement, so its bridge should be described as mixed rather than cleanly target-specific.

## Stage2 By Composition Label

The expanded hidden smoke was joined back to the full `mask_full_v1` composition labels.

| model | Stage1 label | samples | hidden target effect mean | positive frac | target-minus-wrong mean |
|---|---|---:|---:|---:|---:|
| Gemma | multi_evidence_supported | 1 | 2.898 | 0.750 | 3.133 |
| Gemma | one_region_dominated | 3 | 2.873 | 0.563 | 5.414 |
| Qwen | multi_evidence_supported | 3 | 1.239 | 0.785 | -0.661 |
| Qwen | one_region_dominated | 2 | -0.243 | 0.438 | -0.210 |

Interpretation: Qwen's hidden target effect is concentrated in the samples that pass the multi-evidence mask-composition filter, while one-region-dominated samples are weak or negative. Gemma still shows positive hidden movement in both strata, so for Gemma this smoke supports image-sensitive answer flow but does not by itself isolate a two-region compositional route. This is useful for Paper2 framing: Qwen may be the cleaner model for a compositional-A/B story, while Gemma may require PLT/source-route evidence rather than hidden-only evidence to separate compositional from single-region support.

## Stage3 Multi-Evidence Mechanism Follow-Up

We then filtered Stage1 to the actual `multi_evidence_supported` cases and built a Paper2-specific mechanism pack rather than reusing the Stage3/Stage4 paperpack route pool.

Filtered cases:

| model | samples |
|---|---|
| Gemma | `okvqa_val_1295955` |
| Qwen | `okvqa_val_00310`, `okvqa_val_00893`, `okvqa_val_258605` |

Qwen PLT mechanism used `paper2_mechanism_rank50_v1`: 3 samples, layers 13/15/17, rank gate relaxed to 50 because `okvqa_val_00310` has clean target rank 18. This is still a small Paper2 case-level run, not a broad full-dataset run.

| model | lens | label | rows | samples | key result |
|---|---|---|---:|---:|---|
| Gemma | hidden restore | multi-evidence supported | 16 | 1 | target effect mean `2.898`, positive frac `0.750` |
| Qwen | hidden restore | multi-evidence supported | 144 | 3 | target effect mean `1.239`, positive frac `0.910` |
| Qwen | PLT candidate discovery | multi-evidence supported | 72 | 3 | all 3 cases produced Paper2-specific PLT candidates |
| Qwen | PLT feature zeroing/restore | multi-evidence supported | 36 | 3 | real source effect mean `0.019`, real-minus-shuffled `0.026` |
| Qwen | PLT grouped restore | multi-evidence supported | 216 | 3 | target effect near zero but positive direction fraction `0.773` |

Interpretation: Qwen now has a filtered Paper2 mechanism bridge under both hidden and PLT lenses. The PLT signal is small in absolute magnitude but present across all three supported cases after removing the smoke cap. This fits the broader Qwen story: hidden-level evidence flow is clearer; individual/local PLT feature signals exist; grouped PLT route effects remain weak/mixed.

For Gemma, the current Paper2 evidence is still sample-limited: the one supported case has a positive hidden bridge, but Gemma PLT/source-route should be treated as pending case-level follow-up rather than inferred from hidden alone.

## Engineering Notes

- The original Paper2 Stage1 pack had empty local mask PNGs for several samples; Stage2 therefore uses a dedicated hidden-lens asset directory and renders masks from LabelMe JSON where needed.
- The Paper2 mask behavior smoke uses a separate `stage1_mask_behavior_assets_v1` directory and renders `region_A/region_B/union/random` masks from available LabelMe evidence annotations.
- The hidden-lens Stage2 smoke uses relaxed `max_clean_rank` gates because Paper2 is exploratory and some behavior-strong cases are not high-confidence first-token cases under the hidden-lattice runner.
- The Stage3 Qwen PLT mechanism follow-up uses a relaxed clean-rank gate (`max_clean_rank=50`) only for Paper2 case coverage; the original rank-10 smoke is retained as a stricter diagnostic.
- These results should not be merged into Stage4/Stage6 main-claim evidence; they are Paper2 feasibility and case-selection diagnostics.

## Outputs

- `stage1/001_paper2_stage1_behavior_viability_viability_v1.md`
- `stage1/composition_effect_by_sample_mask_v1.csv`
- `stage1/composition_effect_by_sample_mask_full_v1.csv`
- `stage1/model_comparison_summary_mask_v1.csv`
- `stage1/model_comparison_summary_mask_full_v1.csv`
- `stage1/stage1_mask_v1_decision.json`
- `stage1/stage1_mask_full_v1_decision.json`
- `stage2/002_paper2_stage2_hidden_smoke_paper2_stage2_hidden_v1.md`
- `stage2/002_paper2_stage2_hidden_smoke_paper2_stage2_hidden_expanded_v1.md`
- `stage2/paper2_stage2_hidden_expanded_v1_decision.json`
- `stage2/paper2_stage2_hidden_by_composition_label.csv`
- `stage3_mechanism/paper2_stage3_mechanism_pack_paper2_mechanism_rank50_v1_decision.json`
- `stage3_mechanism/paper2_stage3_mechanism_summary_paper2_mechanism_rank50_v1.csv`
- `stage3_mechanism/paper2_stage3_mechanism_paper2_mechanism_rank50_v1_decision.json`
- `stage3_mechanism/003_paper2_stage3_mechanism_viability_paper2_mechanism_rank50_v1.md`
