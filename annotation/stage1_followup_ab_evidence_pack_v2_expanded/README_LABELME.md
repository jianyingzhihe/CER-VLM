# Paper2 Follow-up LabelMe Annotation

## Start

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch_labelme.ps1
```

LabelMe opens the `images/` directory.

## Labels

Use exactly two labels:

- `region_A`: first necessary evidence region.
- `region_B`: second necessary evidence region.

The exporter also accepts old aliases:

- `answer` -> `region_A`
- `relate` -> `region_B`

## What To Draw

Draw only the visual evidence needed to answer the question.

Good A/B examples:

- Text/object binding: `region_A` = the text/sign/object being read; `region_B` = the contextual object/person/scene that tells what the text refers to.
- Relation reasoning: `region_A` = entity 1; `region_B` = entity 2 or the relation-defining context.
- Comparison reasoning: `region_A` = first compared object; `region_B` = second compared object.

If a sample genuinely has only one necessary visual region, draw the key region as `region_A` and leave a note in the CSV later. Do not invent a fake `region_B`.

## Export Masks

After saving LabelMe JSON files, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\export_labelme_masks.ps1
```

This writes:

- `exported_masks/<image_stem>/region_A.png`
- `exported_masks/<image_stem>/region_B.png`
- `exported_masks/<image_stem>/region_A_union_B.png`

It also writes:

- `labelme_mask_export_summary.csv`
- `labelme_mask_export_decision.json`
