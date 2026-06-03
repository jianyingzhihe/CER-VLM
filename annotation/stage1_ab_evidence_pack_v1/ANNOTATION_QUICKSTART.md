# Paper2 Stage1 A/B Evidence Annotation Quickstart

1. Open `stage1_ab_annotation_ui.html`.
2. For each image, draw `region_A.png`, `region_B.png`, and `region_A_union_B.png`.
3. Save masks under `exported_masks/<image_stem>/`.
4. Run `preflight_stage1_ab_masks.py` before starting GPU behavior screening.

Do not use Paper1 `answer/relate/union` labels as a semantic substitute. Paper2 needs two explicit evidence roles.
