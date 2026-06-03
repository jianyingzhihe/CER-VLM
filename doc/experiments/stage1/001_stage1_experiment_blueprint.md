# Stage1-001 Experiment Blueprint

Updated: 2026-06-02

## Summary

Paper2 studies compositional evidence routing in VLMs.

Paper1 established that localized visual evidence can causally flow into answer production. Paper2 asks what happens when a question requires more than one visual evidence region:

```text
Does the model internally combine region A and region B into a joint reasoning route,
or does it answer from one dominant region, text priors, or shortcuts?
```

This is a separate project from Paper1. It reuses Paper1 tools but does not continue the Paper1 stage numbering.

## Paper Boundary

Paper1:

```text
single localized evidence region -> answer route
```

Paper2:

```text
multiple evidence regions -> compositional reasoning route
```

This means Paper2 should not merely repeat evidence masking on harder examples. The required new object is a composition relation between evidence regions:

```text
region A
region B
A_union_B
single-region damage
joint-region damage
internal A-sensitive / B-sensitive / binding-sensitive route behavior
```

## Why Not Mainline Failure Arbitration

Failure arbitration and hallucination remain useful related directions, but they should not be the main Paper2 story.

Reasons:

```text
1. Hallucination circuit work is becoming crowded and already has cross-model pathway analyses.
2. Paper1 already makes a strong positive mechanism claim; a failure-only Paper2 risks becoming a subset of Paper1.
3. Multi-evidence composition is more directly about multimodal reasoning.
4. Multi-evidence composition reuses the strongest Paper1 tools while asking a new question.
```

Paper2 can still use failures diagnostically:

```text
If a sample fails composition tests, ask whether the model used one-region shortcuts, language priors, or distributed hidden routes.
```

But the main claim should be about how evidence is combined, not only why answers are wrong.

## Core Claim Draft

Long version:

```text
Multimodal reasoning in VLMs is not simply a uniform flow of visual information into the answer. For VQA questions that require multiple visual evidence regions, models exhibit different internal evidence-routing regimes: some answers are dominated by a single region, while others require joint or non-additive evidence effects. These composition structures are more naturally visible as graph-closed source-tracing routes in Gemma and as hidden-flow / local feature-node patterns in Qwen.
```

Short version:

```text
VLM reasoning can be decomposed into internal composition over visual evidence regions.
```

Paper-ready conservative version:

```text
For multi-evidence VQA questions, masking region A and region B jointly can produce effects that are not explained by either region alone. These behavioral composition effects are mirrored by internal route changes, with different visibility across Gemma and Qwen.
```

## Stage1 Research Questions

Stage1 should answer four questions:

```text
RQ1: Can we define a clean set of VQA samples that genuinely require two visual evidence regions?
RQ2: Does mask(A_union_B) damage the answer more than mask(A) or mask(B) alone?
RQ3: Can internal routes be classified as A-sensitive, B-sensitive, or binding/composition-sensitive?
RQ4: Do Gemma and Qwen expose this composition through different mechanism lenses?
```

## Sample Set

Target size:

```text
v1 = 40 samples
expanded = 80 samples
```

Priority sample types:

```text
relation reasoning:
  object A + relation / object B

comparison reasoning:
  object A compared with object B

text + object binding:
  text region + carrier object / scene region

local object + context inference:
  object region + context region
```

Avoid at Stage1:

```text
diffuse mood / atmosphere questions
questions where evidence regions cannot be localized
questions answerable by one obvious text token
questions where language priors make the answer too guessable
```

## Mask Conditions

Each accepted sample needs:

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

Optional defensive variants:

```text
dilate_A
dilate_B
erode_A
erode_B
second_annotator_A
second_annotator_B
```

## Behavioral Metrics

Primary:

```text
target logit
target rank
target-vs-competitor margin
decoded answer
answer flip rate
```

Damage should be defined consistently for each metric. For example:

```text
damage_logit(mask_X) = clean_target_logit - masked_target_logit
damage_margin(mask_X) = clean_margin - masked_margin
```

Core composition effect:

```text
composition_effect = damage(mask_A_union_B) - max(damage(mask_A), damage(mask_B))
```

Interpretation:

```text
composition_effect > 0:
  masking both regions hurts more than masking either single region.

composition_effect near 0:
  the answer may be dominated by one region or the model may use redundant evidence.

composition_effect < 0:
  masking both regions does not add damage; inspect for shortcut, format failure, or noisy masks.
```

Optional stricter metric:

```text
non_additivity = damage(mask_A_union_B) - damage(mask_A) - damage(mask_B)
```

This is diagnostic only. It should not be the primary gate because masking effects need not be linear.

## Internal Mechanism Metrics

Gemma lens:

```text
source-tracing route graph
route weakening under mask_A / mask_B / mask_A_union_B
route identity overlap
route composition / topK path mass / signed mix
```

Qwen lens:

```text
hidden restore route
route-first feature nodes
grouped route closure or non-closure
layer-band composition
```

Shared node/route labels:

```text
A-sensitive:
  changed mainly by mask_A.

B-sensitive:
  changed mainly by mask_B.

binding-sensitive:
  weak under mask_A and mask_B alone, but strong under mask_A_union_B.

distributed multi-route:
  no single binding node, but multiple A/B routes jointly predict rank or margin loss.
```

## Expected Results

Result 1:

```text
In selected multi-evidence samples, mask_A_union_B hurts target rank/margin more than mask_A or mask_B alone.
```

Result 2:

```text
Internal routes can be separated into A-sensitive, B-sensitive, and composition-sensitive patterns.
```

Result 3:

```text
Composition effect predicts answer flips or rank/margin degradation.
```

Result 4:

```text
Gemma and Qwen show different visibility of composition: Gemma may expose graph-closed routes, while Qwen may expose hidden/feature-node distributed routes.
```

Result 5:

```text
Prompt/CoT may modulate composition routes, but should not be treated as the main mechanism.
```

## Failure Modes And Fallbacks

Single-region shortcut:

```text
If mask_A_union_B is not stronger than the strongest single-region mask, classify the sample as one-region dominated. Do not use it as main multi-evidence evidence.
```

Subjective masks:

```text
Use compact regions, same-area random controls, and optional morphology / second annotator checks.
```

No binding node:

```text
Do not treat this as failure. Report distributed multi-route composition if A/B routes jointly predict behavior.
```

Model asymmetry:

```text
Do not force Gemma and Qwen to expose identical objects. Require symmetric questions and explicit lens labels.
```

## Stage1 Exit Criteria

Stage1 is complete when we have:

```text
1. A documented sample taxonomy.
2. A/B annotation rules.
3. A first candidate pool target of 40 samples.
4. Behavioral composition-effect gates.
5. Gemma/Qwen mechanism lenses defined.
6. Clear fallbacks for non-compositional samples.
```

