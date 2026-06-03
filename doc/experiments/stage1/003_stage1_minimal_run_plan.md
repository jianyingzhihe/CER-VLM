# Stage1-003 Minimal Run Plan

Updated: 2026-06-02

## Goal

Run a minimal Stage1 screen to identify samples where joint A/B evidence masking is meaningfully stronger than single-region masking.

This is not yet a full mechanism runbook. It is the first experimental pass needed before investing in Gemma source-tracing or Qwen hidden/feature analysis.

## Inputs

Required:

```text
40 candidate multi-evidence samples
A_mask
B_mask
A_union_B_mask
random_A_size
random_B_size
random_union_size
gold answer
one short competitor answer if available
```

Models:

```text
Gemma
Qwen
```

No third model in Stage1.

## Conditions

Run each sample under:

```text
clean
mask_A
mask_B
mask_A_union_B
random_A_size
random_B_size
random_union_size
wrong_image
```

For each condition, collect:

```text
target logit
target rank
target-vs-competitor margin
decoded answer
format_ok
empty_answer
```

Use the same final-answer format across conditions:

```text
The answer is <short answer>.
```

## Damage Metrics

For each metric:

```text
damage_logit(mask_X) = clean_target_logit - masked_target_logit
damage_margin(mask_X) = clean_margin - masked_margin
damage_rank(mask_X) = masked_target_rank - clean_target_rank
```

Primary composition score:

```text
composition_effect_margin =
  damage_margin(mask_A_union_B) - max(damage_margin(mask_A), damage_margin(mask_B))
```

Secondary scores:

```text
composition_effect_logit
composition_effect_rank
answer_flip_union_minus_single
```

Optional stricter score:

```text
non_additivity_margin =
  damage_margin(mask_A_union_B) - damage_margin(mask_A) - damage_margin(mask_B)
```

Do not make non-additivity the main gate.

## Stage1 Behavioral Gates

### Multi-Evidence Supported

Use this label when:

```text
mask_A_union_B causes larger margin/rank damage than both mask_A and mask_B
and random_union_size is weaker than mask_A_union_B
and clean answer is format-valid
```

### One-Region Dominated

Use this label when:

```text
one of mask_A or mask_B is nearly as damaging as mask_A_union_B
```

Interpretation:

```text
This sample may still be valid VQA, but it is not a main Paper2 composition sample.
```

### Redundant Evidence

Use this label when:

```text
mask_A and mask_B each hurt the answer,
but mask_A_union_B is not substantially stronger.
```

Interpretation:

```text
The model may have redundant evidence routes or the masks may overlap in function.
```

### Shortcut / Prior Suspected

Use this label when:

```text
mask_A_union_B does not hurt,
wrong_image does not hurt,
or decoded answer remains stable despite evidence removal.
```

### Blocked

Use this label when:

```text
format failure
empty answer
target token mismatch
unstable clean answer
mask file missing
```

## Mechanism Analysis Entry Gate

Only samples with `multi_evidence_supported` should enter the first mechanism run.

Recommended minimum:

```text
at least 16 supported samples
balanced across at least 2 sample types
both Gemma and Qwen have valid clean answers
```

If fewer than 16 samples pass:

```text
expand candidate pool before running expensive internal analysis
```

## First Mechanism Pass

Gemma:

```text
source-tracing route under clean
source-tracing route under mask_A
source-tracing route under mask_B
source-tracing route under mask_A_union_B
route overlap and route weakening
route composition / path-mass diagnostic
```

Qwen:

```text
hidden restore under mask_A
hidden restore under mask_B
hidden restore under mask_A_union_B
route-first feature-node screen
grouped feature-route diagnostic if feature nodes pass
```

Shared labels:

```text
A-sensitive route
B-sensitive route
binding-sensitive route
distributed multi-route
```

## Prompt/CoT Subset

Prompt variation is not the main Stage1 line.

If included, use a small subset only:

```text
direct
visual evidence
step only
step + visual
```

Accepted conclusion:

```text
Prompt/CoT modulates composition routes but does not define the main mechanism.
```

Do not write:

```text
CoT improves visual reasoning.
```

unless directly supported by composition and mechanism metrics.

## Output Tables

Stage1 should produce:

```text
sample_screen_summary.csv
composition_effect_by_sample.csv
composition_effect_by_type.csv
model_comparison_summary.csv
blocked_or_shortcut_samples.csv
stage1_decision.json
```

The decision JSON should include:

```text
candidate_count
supported_count
one_region_dominated_count
redundant_count
shortcut_count
blocked_count
recommended_next_action
```

## Decision Outcomes

Proceed to mechanism analysis:

```text
supported_count >= 16
and both models have enough valid rows
```

Expand annotation:

```text
supported_count < 16
but screening pipeline works
```

Revise sample definition:

```text
most samples are one-region dominated or shortcut/prior suspected
```

Pause Paper2 direction:

```text
composition_effect is absent across all sample types and both models,
after masks and scoring are verified.
```

## Stage1 Exit Criteria

Stage1 exits with one of:

```text
ready_for_stage2_mechanism
needs_more_samples
needs_annotation_revision
blocked_engineering
direction_not_supported
```

Preferred Stage1 success:

```text
ready_for_stage2_mechanism
with 16-24 strong multi-evidence samples.
```

