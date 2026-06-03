# Paper2 Viability Verdict

Updated: 2026-06-03 12:31 CST

## Verdict

Paper2 is viable, but not as a broad claim that every wrong-image-sensitive VQA sample uses two-region compositional evidence routing.

The current evidence supports a narrower and cleaner direction:

> Wrong-image sensitivity is common, but true A/B compositional evidence behavior is a filtered subset. Qwen currently gives the cleaner hidden-lens signal for that subset; Gemma shows image-sensitive hidden flow but does not separate compositional from one-region-dominated cases under the current hidden lens.

After the Stage3 mechanism follow-up, this becomes slightly stronger for Qwen:

> For Qwen, the filtered multi-evidence subset now has both hidden-lens evidence flow and Paper2-specific PLT candidate/feature-level support. The PLT effects are small and grouped restore remains mixed, so the cleanest claim is not “closed sparse route,” but “mechanism signal exists under hidden and local feature lenses.”

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
- High-priority unannotated cases are `okvqa_val_2847255` and `okvqa_val_01162` for Gemma, plus `okvqa_val_03127` for Qwen.

Stage2 hidden-lens bridge:

- Gemma: hidden restore target effect is positive for both multi-evidence and one-region-dominated cases.
- Qwen: hidden restore target effect is stronger and more positive in multi-evidence cases; one-region-dominated cases are weak/negative.

Stage3 filtered mechanism bridge:

- Qwen Paper2 PLT discovery covered all three Qwen `multi_evidence_supported` cases after relaxing the clean-rank gate to 50.
- Qwen PLT feature zeroing/restore: 36 rows across 3 samples, mean real source effect `0.019`, mean real-minus-shuffled `0.026`.
- Qwen grouped PLT restore: 216 rows across 3 samples, mean target effect close to zero but positive-direction fraction `0.773`.
- Gemma currently has one `multi_evidence_supported` case with positive hidden restore (`2.898` mean target effect, `0.750` positive fraction), but Gemma PLT/source-route remains pending and should not be inferred from hidden alone.

## Paper-Ready Interpretation

For Paper2, the most defensible framing is:

> Multi-region VQA questions contain heterogeneous evidence structures. Some require both annotated regions, but many are dominated by one region or by shortcut/prior behavior. A behavior-only wrong-image test is therefore not sufficient to identify compositional evidence routing. Mask composition filtering is needed before mechanism-level route analysis.

For cross-model comparison:

> Qwen currently provides a cleaner hidden-lens signature for the filtered multi-evidence subset. Gemma's hidden flow remains image-sensitive, but the current hidden lens does not distinguish multi-region composition from single-region support; Gemma may require PLT/source-tracing route-level analysis for a sharper Paper2 mechanism.

For Qwen mechanism wording:

> In the filtered multi-evidence subset, Qwen shows a two-level bridge: hidden residual restore is strongest, and Paper2-specific PLT feature probes find local supporting nodes. However, grouped PLT restore remains weak/mixed, consistent with the broader Qwen pattern that evidence flow is distributed and not cleanly closed as a sparse grouped route.

## Next Step

Recommended next experiment:

- For Gemma, run compact PLT/source-route or fixed-node route-level probe on `okvqa_val_1295955`; keep it case-level unless more Gemma A/B-supported samples are identified.
- For Qwen, optionally expand from layers 13/15/17 to a few additional layers only if the Paper2 mechanism section needs a stronger appendix table.
- Annotation expansion is still the highest-value next step: current Gemma has only one supported case under the original label, and only a small robust multi-evidence subset survives stricter threshold sweeps. Start with `okvqa_val_2847255`, `okvqa_val_01162`, and `okvqa_val_03127`.

## Boundary

This does not affect Stage4/Stage6 main claims. Paper2 is a separate, filtered follow-up about compositional evidence routing, not a replacement for the established evidence-to-answer route claim.
