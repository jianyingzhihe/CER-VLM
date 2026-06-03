# Paper2 Stage3 Mechanism Viability

Generated: 2026-06-03 15:05:14

## Current Verdict

Status: `paper2_stage3_mechanism_full_ready`.

This stage filters Paper2 to behavior-supported multi-evidence cases before asking whether hidden/PLT mechanisms show matching evidence-to-answer movement.

## Summary Table

| model | lens | label | rows | samples | key mean |
|---|---|---|---:|---:|---:|
| gemma | hidden_restore | multi_evidence_supported | 16 | 1 | 2.8984375 |
| gemma | hidden_restore | one_region_dominated | 48 | 3 | 2.8728129069010415 |
| gemma | compact_plt_source_route | multi_evidence_supported | 4 | 1 | mixed_not_union_specific |
| qwen | hidden_restore | multi_evidence_supported | 144 | 3 | 1.2391493055555556 |
| qwen | hidden_restore | one_region_dominated | 96 | 2 | -0.24348958333333334 |
| qwen | plt_candidate_discovery | multi_evidence_supported | 72 | 3 | 0.7703522406271647 |
| qwen | plt_feature_zeroing_restore | multi_evidence_supported | 36 | 3 | 0.019097222222222224 |
| qwen | plt_group_restore | multi_evidence_supported | 216 | 3 | -0.005208333333333333 |

## Interpretation

- Hidden-lens rows are inherited from the symmetric Gemma/Qwen Paper2 Stage2 run.
- Qwen PLT rows come from Paper2-specific discovery/validation, not from the old Stage4 paperpack route pool.
- Gemma compact PLT/source-route has now been run on the one Gemma `multi_evidence_supported` case, `okvqa_val_1295955 / protest`, using `max_feature_nodes=8` and four visual conditions: `mask_A`, `mask_B`, `mask_A_union_B`, and `random_union_size`.
- The Gemma route result is mixed rather than positive: the union condition does not weaken traced path mass more than the single regions or random control. This should be reported as sample-limited case-level diagnostic evidence, not as a broad Gemma model-level negative result.
