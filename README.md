# Compositional Evidence Routing in Vision-Language Models

Updated: 2026-06-02

## Project

This is a separate project from the Paper1 evidence-to-answer route work.

Paper1 asks:

```text
Can localized visual evidence causally enter the answer path?
```

Paper2 asks:

```text
When an answer requires multiple visual evidence regions, how does a VLM combine those regions internally?
```

Working title:

```text
Compositional Evidence Routing in Vision-Language Models
```

## Core Direction

The project moves away from making failure arbitration or hallucination the main Paper2 story. Those can remain as related work or later side analyses.

The main mechanism question is positive and reasoning-oriented:

```text
Do VLMs internally compose multiple visual evidence regions into a joint answer-supporting route,
or do they answer by relying on one dominant region, text priors, or shortcuts?
```

## Stage1 Documents

```text
doc/experiments/stage1/001_stage1_experiment_blueprint.md
doc/experiments/stage1/002_stage1_annotation_protocol.md
doc/experiments/stage1/003_stage1_minimal_run_plan.md
```

Stage1 is an experiment blueprint, not a command-level runbook. It defines what to prove, what data to collect, how to annotate A/B evidence regions, and how to decide which samples advance into mechanism analysis.

## Relationship To Paper1

Paper2 can reuse Paper1 assets and methods:

```text
localized mask pipeline
random / shifted / shuffled controls
Gemma source-tracing route graph
Qwen hidden route and route-first feature nodes
prompt/text/CoT modulation diagnostics
```

But Paper2 should not be placed inside Paper1 `doc/experiments/stage*`. It has a separate question, separate sample design, and separate stage numbering.
