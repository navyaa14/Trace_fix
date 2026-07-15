# TraceFix — Failure Attribution & Repair for Agentic AI Workflows

TraceFix is a research-oriented prototype that explores graph-level failure attribution and repair in multi-stage AI agent workflows.

Recent advances such as AutoChunker, AutoKB, ASK, SMART, and AutoEval-ToD have significantly improved individual stages of agentic AI systems. On the workflow-level failure attribution side, A2P Scaffolding provides complementary ideas for counterfactual attribution and repair validation. TraceFix investigates the next step: **how an evaluated failure can be systematically attributed, repaired, validated, and incorporated into future workflow optimization as a closed feedback loop.**

Instead of treating evaluation as the final stage of a pipeline, TraceFix models the entire workflow as a graph where failures can be localized, competing repair strategies can be executed, and their impact on groundedness, latency, API cost, and human escalation can be measured before adoption.

---

## Key Features

- Graph-based workflow failure attribution across multi-stage AI pipelines.
- Multiple attribution strategies evaluated under a common experimental framework.
- Counterfactual repair replay for validating candidate fixes before adoption.
- Before/after measurement of groundedness, latency, API cost, and human intervention.
- Workflow optimization recommendations driven by observed execution statistics.
- Interactive dashboard for visualizing workflow behavior, attribution accuracy, and repair outcomes.
- Multi-seed evaluation to assess robustness across independent executions.
- Comprehensive automated test suite validating the complete framework.

---

### Dashboard

> **Synthetic 8,000-trace evaluation using the heuristic judge (no LLM call).**

<p align="center">
<img src="YOUR_DASHBOARD_SCREENSHOT.png" width="100%">
</p>

---

## Workflow

```text
Chunker
    ↓
Knowledge Builder
    ↓
Retriever
    ↓
Clarifier
    ↓
Generator
    ↓
VAD/e
    ↓
Evaluator
    ↓
Failure Attribution
    ↓
Candidate Repair Generation
    ↓
Counterfactual Validation
    ↓
Repair Selection
    ↓
Continuous Improvement
```

---

## Attribution Strategies

- All-at-Once
- Step-by-Step
- Binary Search
- A2P Scaffolding

The framework evaluates multiple attribution strategies within the same execution environment, enabling direct comparison of attribution quality, computational efficiency, and downstream repair effectiveness.

---

## Counterfactual Repair

For every attributed failure, TraceFix:

1. Generates candidate repairs.
2. Executes each repair on the workflow.
3. Recomputes downstream behavior.
4. Measures groundedness, latency, API cost, and human escalation.
5. Accepts only repairs that improve measured outcomes.

---

## Motivation

Modern agentic AI systems are increasingly composed of multiple interacting components rather than single models. As these workflows grow more complex, identifying the originating stage of a failure becomes as important as detecting the failure itself.

TraceFix explores a unified framework that connects

**Evaluation → Attribution → Repair → Validation → Optimization**

into a single closed-loop workflow, providing a foundation for studying more reliable and maintainable AI systems.

---

## Experimental Highlights

- Four attribution strategies
- Counterfactual repair replay
- Continuous improvement loop
- Interactive workflow dashboard
- Multi-seed evaluation
- 248 automated tests

---

## Current Scope

TraceFix is a synthetic research prototype designed to study workflow-level failure attribution and repair.

It is **not** trained or evaluated on proprietary production traces, nor is it intended as a reproduction of any published benchmark. Instead, it provides a reusable experimental framework for exploring attribution, repair validation, and workflow optimization in agentic AI systems.

---

## Running

```bash
python demo.py

python evaluate_multiseed.py

python run_continuous_improvement.py --reset-memory



```

<img width="1445" height="698" alt="dashbord" src="https://github.com/user-attachments/assets/a9e09e6b-a733-4a60-8798-e00a2c8c4d1d" />
