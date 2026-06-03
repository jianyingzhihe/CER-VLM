# Paper2 Stage3 Gemma Compact PLT Source-Route Probe

Generated: 2026-06-03 15:05 CST

## Goal

This run fills a symmetry gap in the Paper2 mechanism table. Qwen already had hidden plus PLT-feature/group probes on the filtered multi-evidence subset. Gemma had a hidden bridge but no Paper2-specific PLT/source-route diagnostic.

The probe is intentionally case-level:

- model: Gemma
- sample: `okvqa_val_1295955`
- answer: `protest`
- prompt: `Why would this person be holding this sign?`
- route lens: compact PLT/source tracing
- max feature nodes: `8`
- conditions: `mask_A`, `mask_B`, `mask_A_union_B`, `random_union_size`

## Result

Status: `gemma_case_route_mixed_or_not_union_specific`.

| condition | traced path mass delta | target total-in-abs delta | node overlap | edge overlap | interpretation |
|---|---:|---:|---:|---:|---|
| `mask_A` | `+0.0319` | `-9.051` | `0.636` | `0.375` | target magnitude drops, but compact traced path mass increases |
| `mask_B` | `-0.0360` | `-5.040` | `0.289` | `0.050` | weak single-region route weakening |
| `mask_A_union_B` | `+0.0653` | `-4.890` | `0.395` | `0.286` | union does not produce stronger route weakening |
| `random_union_size` | `-0.0111` | `+4.507` | `0.233` | `0.022` | random control also changes the compact route |

The union condition is not stronger than the best single-region condition or the same-area random control under the compact traced-path-mass metric. Therefore this run does not support a clean Gemma union-specific compositional source route for this case.

## Interpretation

This is not a failure of Paper1's Gemma route claim. The Paper1 claim concerns evidence-to-answer routing under the original paperpack setup and includes stronger Gemma source-tracing assets. This Paper2 run asks a narrower question: whether the one currently annotated Gemma A/B-supported Paper2 case shows a compact, union-specific PLT/source route under `max_feature_nodes=8`.

The answer is mixed:

- The hidden bridge for this case is positive.
- The compact PLT/source-route lens does not isolate an A/B-union route.
- The case remains useful as a diagnostic, but it is not enough for a Gemma compositional mechanism claim.

## Paper2 Consequence

Paper2 should remain Qwen-led for the current compositional mechanism story. Qwen has three supported cases with hidden and local PLT-feature evidence. Gemma has one supported case with a positive hidden bridge and a mixed compact PLT/source-route diagnostic.

The next highest-value action is annotation expansion, not more Gemma mechanism probing on the same single case.

## Outputs

- `paper2_stage3_gemma_source_route_full_paper2_mechanism_rank50_v1_summary.csv`
- `paper2_stage3_gemma_source_route_full_paper2_mechanism_rank50_v1_decision.json`
- per-condition compare/intervention CSVs under `doc/experiments/paper2/stage3_mechanism/`
