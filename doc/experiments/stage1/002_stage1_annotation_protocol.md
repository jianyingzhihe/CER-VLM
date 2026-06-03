# Stage1-002 Annotation Protocol

Updated: 2026-06-02

## Purpose

Paper2 needs samples where the answer plausibly depends on at least two localized visual evidence regions.

The annotation goal is not to mark every visually relevant pixel. The goal is to mark the minimum evidence regions needed to test compositional dependence:

```text
region_A
region_B
A_union_B
same-area random controls
```

## Region Definitions

### Region A

Region A is the first necessary visual evidence component.

Examples:

```text
the queried object
one object in a comparison
the text-bearing region
the actor in an action relation
```

### Region B

Region B is the second necessary visual evidence component.

Examples:

```text
the related object
the second object in a comparison
the carrier object / context for a text region
the scene context needed to infer action or use
```

### A Union B

`A_union_B` is the union of A and B masks.

It should not include extra background unless the background is explicitly part of region B.

## Accepted Sample Types

### Relation Reasoning

Question structure:

```text
object A + relation + object B
```

Examples:

```text
What is the person holding?
Which object is to the left of the cup?
What is the animal standing on?
```

Annotation:

```text
A = subject / anchor object
B = related object or relation target
```

### Comparison Reasoning

Question structure:

```text
compare object A with object B
```

Examples:

```text
Which object is larger?
Which sign has more text?
Which side has more people?
```

Annotation:

```text
A = first compared region
B = second compared region
```

### Text + Object Binding

Question structure:

```text
read text and bind it to an object or scene.
```

Examples:

```text
What language is on the sign?
What brand is on the bottle?
What number is on the jersey?
```

Annotation:

```text
A = text region
B = carrier object / sign / jersey / scene region
```

### Local Object + Context Inference

Question structure:

```text
infer use, action, state, or category from object plus context.
```

Examples:

```text
What sport is being played?
What tool is being used?
What is the person likely doing?
```

Annotation:

```text
A = local object / person / tool
B = context region that disambiguates the answer
```

## Rejection Rules

Reject a sample if any of the following holds:

```text
answer can be read from a single compact text region only
only one region is visually necessary
evidence is too diffuse to annotate
region A and B cannot be separated
question requires external knowledge more than image evidence
answer is unstable or format-brittle under clean generation
model gives wrong clean answer for both Gemma and Qwen
```

Do not force a sample into multi-evidence form if it is naturally single-evidence. Single-evidence samples belong to Paper1-style controls, not Paper2 main evidence.

## Mask Construction

Required masks:

```text
A_mask
B_mask
A_union_B_mask
random_A_size
random_B_size
random_union_size
```

Recommended mask properties:

```text
compact
tightly aligned to evidence
not pixel-perfect obsessive
same resolution and format as Paper1 masks
preserve original image outside masked area
```

Masking should use the same visual corruption style as Paper1 where possible, so Paper2 results are comparable.

## Random Controls

Random masks must match area as closely as practical:

```text
random_A_size:
  same area as A_mask

random_B_size:
  same area as B_mask

random_union_size:
  same area as A_union_B_mask
```

Random controls should avoid overlapping the annotated A/B evidence regions unless the image is too small to place them otherwise.

## Morphology Defense

For a subset of samples, prepare:

```text
dilate_A
dilate_B
erode_A
erode_B
dilate_union
erode_union
```

Use morphology only as a robustness diagnostic. The main claim should not require exact invariance to mask shape.

Preferred wording:

```text
The effect is coverage-sensitive but not a pixel-perfect boundary artifact.
```

## Second Annotator Option

If time allows, run a second annotator on 10-20 percent of samples.

Second annotator should receive:

```text
question
image
answer
short explanation of A/B evidence roles
```

They should not see the first annotator's masks.

Acceptance:

```text
If second masks preserve the main composition direction, use as defensive evidence.
If second masks differ, inspect whether the sample definition was ambiguous.
```

## Metadata To Record

Each sample should record:

```text
sample_id
image_path
question
gold_answer
sample_type
region_A_role
region_B_role
A_mask_path
B_mask_path
union_mask_path
random_A_paths
random_B_paths
random_union_paths
clean_answer_gemma
clean_answer_qwen
clean_format_ok
notes
```

Sample type values:

```text
relation_reasoning
comparison_reasoning
text_object_binding
object_context_inference
```

## Annotation Acceptance Checklist

A sample is Stage1-ready if:

```text
1. A and B are visually localizable.
2. A and B correspond to distinct evidence roles.
3. A_union_B is not just the whole image.
4. Same-area random masks can be generated.
5. Clean model answer is stable enough to score target rank/margin.
6. The sample is not obviously answerable by one region alone.
```

The final condition is provisional. Behavioral screening in Stage1-003 determines whether the sample is truly multi-evidence for the tested model.

