# Paper2 Viability Verdict

Updated: 2026-06-03 15:05 CST

## Verdict

Paper2 is viable, but not as a broad claim that every wrong-image-sensitive VQA sample uses two-region compositional evidence routing.

The current evidence supports a narrower and cleaner direction:

> Wrong-image sensitivity is common, but true A/B compositional evidence behavior is a filtered subset. Mask composition filtering is needed before mechanism-level route analysis.

After Stage3, the mechanism picture is asymmetric:

> Qwen currently provides the cleaner Paper2 mechanism signal. In the filtered multi-evidence subset, Qwen has hidden-lens evidence flow plus small but positive local PLT-feature support. Grouped PLT restore remains mixed, so the safe wording is "mechanism signal exists under hidden and local feature lenses," not "closed sparse route."

For Gemma:

> Gemma has a positive hidden bridge on its one `multi_evidence_supported` case, but the compact PLT/source-route case probe is mixed and does not show union-specific route weakening. This is a sample-limited Paper2 diagnostic, not a model-level negative result and not a contradiction of Paper1.

## Evidence

Stage1 behavior viability:

- Gemma: 40 paired samples, rank worsened in 75%, mean target-logit drop 9.078, decoded answer changed in 87.5%.
- Qwen: 40 paired samples, rank worsened in 90%, mean target-logit drop 4.828, decoded answer changed in 90%.

Stage1 mask composition full:

- 15 mask-renderable samples, evaluated for Gemma and Qwen.
- Gemma: 1 multi-evidence, 7 one-region dominated, 3 shortcut/prior, 3 mixed, 1 blocked.
- Qwen: 3 multi-evidence, 9 one-region dominated, 2 shortcut/prior, 1 mixed, 0 blocked.

Stage1 no-annotation follow-up:

- Reanalyzed existing Stage1 outputs under multiple margin/random-control thresholds.
- Robust multi-evidence model-sample rows across threshold settings: 5 total, with 3 Gemma rows and 2 Qwen rows.
- The supported set shrinks quickly under stricter thresholds, so the current Stage1 data is best treated as sample triage rather than final sample coverage.
- High-priority unannotated cases are Gemma `okvqa_val_2847255 / china`, Gemma `okvqa_val_01162 / tony hawk`, and Qwen `okvqa_val_03127 / pelican`.
- The follow-up annotation pack is ready at `annotation/stage1_followup_ab_evidence_pack_v2`.

Stage2 hidden-lens bridge:

- Gemma: hidden restore target effect is positive for both multi-evidence and one-region-dominated cases.
- Qwen: hidden restore target effect is stronger and more positive in multi-evidence cases; one-region-dominated cases are weak/negative.

Stage3 filtered mechanism bridge:

- Qwen Paper2 PLT discovery covered all three Qwen `multi_evidence_supported` cases after relaxing the clean-rank gate to 50.
- Qwen PLT feature zeroing/restore: 36 rows across 3 samples, mean real source effect `0.019`, mean real-minus-shuffled `0.026`.
- Qwen grouped PLT restore: 216 rows across 3 samples, mean target effect close to zero but positive-direction fraction `0.773`.
- Gemma has one `multi_evidence_supported` case with positive hidden restore: `2.898` mean target effect and `0.750` positive fraction.
- Gemma compact PLT/source-route on `okvqa_val_1295955 / protest` completed with 4 condition rows. It is mixed: `mask_B` slightly weakens traced path mass (`-0.036` condition-minus-clean), `random_union_size` is also slightly negative (`-0.011`), while `mask_A` and `mask_A_union_B` increase traced path mass (`+0.032` and `+0.065`).

## Paper-Ready Interpretation

For Paper2, the most defensible framing is:

> Multi-region VQA questions contain heterogeneous evidence structures. Some require both annotated regions, but many are dominated by one region or by shortcut/prior behavior. A behavior-only wrong-image test is therefore not sufficient to identify compositional evidence routing.

For cross-model comparison:

> Qwen currently gives the cleaner compositional mechanism story because hidden restore and local PLT-feature probes both align with the filtered multi-evidence subset. Gemma remains sample-limited: its hidden bridge is positive, but its compact PLT/source-route probe does not yet show a clean union-specific route.

## Next Step

- Do not expand Gemma mechanism claims until more A/B-supported Gemma cases are annotated.
- Use the high-priority annotation pack first: Gemma `okvqa_val_2847255`, Gemma `okvqa_val_01162`, and Qwen `okvqa_val_03127`.
- For Qwen, optionally expand layers beyond 13/15/17 only if a stronger appendix table is needed.

## Boundary

This does not affect Stage4/Stage6 main claims. Paper2 is a separate, filtered follow-up about compositional evidence routing, not a replacement for the established evidence-to-answer route claim.
