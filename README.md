# TraceFix — Failure Attribution & Repair Prototype

A synthetic, closed-loop research prototype: represent an agentic RAG/support
workflow as a DAG, generate traces with realistic (non-leaking) symptom
evidence, attribute failures to a likely responsible node using several
attribution strategies, and execute an attribute → repair → replay cycle
with a measured before/after cost comparison.

## What this is, plainly

A runnable scaffold for representing an agentic RAG/support workflow as a
DAG, simulating traces with non-leaking symptom evidence, comparing
attribution strategies against real ground truth, and generating concrete,
replayed repairs. It is **not**: a reproduction of Who&When's published
results on their actual dataset, a causal-inference implementation, a
general graph-edit search engine, or a publication-ready benchmark. See
"What you can honestly claim about this today" below.

## The pipeline

Eight nodes: `chunker → kb_builder → retriever → clarifier → generator →
vade → evaluator → human`. `vade` ("Validate After Draft / evaluate") is a
hallucination/content validator that sits between generation and the final
evaluator; `human` is a terminal escalation sink, not a node that can itself
be a root cause.

## Outcome semantics

`Trace` tracks two independent things and one derived split:

- **`content_failed`**: the generated response is actually unacceptable
  (low groundedness, or an unresolved entity mismatch).
- **`evaluation_failed`**: the evaluator's verdict disagreed with the actual
  content state — either a false escalation of good content
  (`evaluator_false_escalation`) or a false acceptance of bad content
  (`evaluator_false_acceptance`).
- **`final_outcome_failed`** (`workflow_failed`) = `content_failed OR
  evaluation_failed`. This is what every accuracy/cost number in this repo
  is measured against.
- **`contained` / `user_visible_failure`**: a further split of the outcome
  above. `contained` is true whenever bad content does not reach the user
  unfiltered — either there was no defect, or the evaluator correctly
  flagged it. `user_visible_failure` is true only when a real content
  defect reaches the user unflagged (equivalently, an evaluator false
  acceptance). Because `vade` and the evaluator cannot rewrite what the
  generator produced, their repairs are judged honestly on whether they
  resolve `user_visible_failure`, not on whether they force
  `content_failed` to `False`.

`causal_model.py` is the single shared module for `retrieval_score_mean`,
`groundedness_mean`, `outcome_failed`, `evaluation_failed`, `workflow_failed`,
`contained`, and `user_visible_failure`, imported directly by both
`simulate.py` (trace generation) and `repair_engine.py` (repair replay) —
`tests/test_shared_causal_invariants.py` asserts by identity that both
modules reference the exact same functions, never independently
reimplemented copies that could drift.

## Node-level failure mechanisms

Every pipeline node has a distinct, causally-wired failure mechanism,
observable evidence, and repair:

- **`chunker`**: an `entity_span_split` mechanism degrades retrieval quality
  through its own term in `causal_model.retrieval_score_mean`, observable via
  `context_coverage_score` / `entity_span_split` / `chunk_coherence_score`.
  Root cause: `chunk_boundary_split_entity`. Repair: `RECHUNK` at `chunker`
  (distinct from `RECHUNK` at `kb_builder` — same action name, different
  node, different effect).
- **`kb_builder`**: staleness (`kb_age_days`) degrades retrieval quality.
  Root cause: `stale_knowledge_correct_entity`. Repair: `RECHUNK` at
  `kb_builder`.
- **`retriever`**: an entity/variant mismatch. Root cause:
  `wrong_entity_variant`. Repair: `ADD_FILTER` at `retriever`.
- **`generator`**: transient or sticky hallucination risk. Root causes:
  `unsupported_generation_transient` (transient — genuinely fixable by a
  retry) and `repeated_hallucination` (sticky — structural, not fixable by
  retrying). Repair: `RETRY` at `generator`, which only has a defined
  causal effect on transient risk; it is not appliable to sticky
  hallucination or to any other node's failure.
- **`vade`**: a `vade_issue_present` mechanism, mutually exclusive with the
  generator's own `hallucination_risk`, models a content defect only the
  validator is positioned to catch. Missing it (`vade_flagged=False`) is a
  node-local root cause: `vade_missed_hallucination`. Repair:
  `RETRY_VALIDATION` / `LOWER_DETECTION_THRESHOLD` at `vade`, which changes
  only detection/routing — groundedness is asserted byte-identical
  before/after in `tests/test_vade_failure.py`, and the repair's honest
  success criterion is resolving `user_visible_failure`, not forcing
  `content_failed` to `False`.
- **`evaluator`**: `evaluator_false_escalation` (good content, evaluator
  rejects it) and `evaluator_false_acceptance` (bad content, evaluator
  accepts it) are both reachable, node-local failures. A false acceptance
  is causally two things at once — the upstream node that produced the bad
  content, and the evaluator that let it through — so `Trace.ground_truth_nodes`
  carries both rather than crediting only one. Repair: `SECOND_JUDGE` /
  `RECALIBRATE_THRESHOLD` at `evaluator`, which flips the verdict in either
  direction and never touches content — `tests/test_evaluator_failure.py`
  asserts the generator's and retriever's symptoms are byte-identical
  before/after.

`human` is excluded from upstream root-cause scoring by design —
`tests/test_shared_causal_invariants.py::TestHumanStaysATerminalEscalationSink`
checks this explicitly.

## Attribution methods

Four methods, all evaluated against real (never leaked) ground truth:

| method | idea | avg judge-calls/trace |
|---|---|---|
| `all_at_once` | one prompt, one guess | 1.00 |
| `step_by_step` | walk the trace turn-by-turn, ask at each step | ~8 |
| `binary_search` | bisect the trace to find the earliest anomalous step | ~3 |
| `a2p_scaffold` | Abduce candidate causes from symptom evidence, then Act+Predict: apply each candidate's repair and keep only causes a real counterfactual replay confirms | 1.00 |

`all_at_once` and `step_by_step` and `binary_search` are the three method
families from the Who&When paper (see "Grounded in" table below).
`a2p_scaffold` implements A2P (Abduct-Act-Predict) Scaffolding — it reuses
`repair_engine.generate_and_evaluate_candidates` directly as its
counterfactual-confirmation step rather than re-implementing counterfactual
simulation separately.

Current measured accuracy (n=300, seed=11, default config):
**57.8%** all-at-once, **68.9%** binary-search, **73.3%** a2p_scaffold
(n=90 evaluable failed traces). The blended number hides real per-failure-type
variation — `demo.py` and the dashboard both print/show the breakdown, and
`multiple_simultaneous_failures` traces have two independently-true root
causes by construction, so single-label and multi-label accuracy are
reported separately (`attribution.matches_ground_truth`,
`tests/test_multi_label_attribution.py`). Two of the rarer categories
(`clarification_failed_annoyed_user`, `evaluator_false_escalation`) have
n=1 in this batch — a 0% there is one wrong guess, not evidence the method
fails on that category; `demo.py`'s printed breakdown and the dashboard both
flag any row below `attribution.MIN_RELIABLE_N` (5) instead of stating it
with the same confidence as `wrong_entity_variant` (n=29) or
`multiple_simultaneous_failures` (n=16).

**Live driver.** `run_continuous_improvement.py`'s repair loop uses
`a2p_scaffold`, chosen by comparing both methods as the actual repair-loop
driver — not attribution accuracy alone — across 5 seeds x 300 traces
(`tests/test_live_driver_choice.py`):

| metric | a2p_scaffold | binary_search |
|---|---|---|
| repair accept rate | 0.506 | 0.429 |
| unresolved rate | 0.130 | 0.161 |
| total cost after repair | $228 | $288 |
| residual user-visible failures (of 227) | 3 | 5 |

a2p_scaffold wins on every one of these full-loop metrics, not only on
attribution accuracy in isolation. `demo.py`'s per-node cost/failure-rate
recommendations use `all_at_once` instead, since that panel is about node
failure rates and cost, not attribution accuracy.

## Repair engine

`repair_engine.py` runs any candidate action against any failing trace and
measures it independently — it never assumes a single predetermined repair
is correct. `generate_and_evaluate_candidates` tries every candidate;
`select_best_candidate` picks the accepted one with the lowest measured
api+human cost; `NOT_EXECUTABLE` is a real, distinct outcome from "executed
but rejected."

A repaired trace is always recomputed from its (repaired) symptoms through
the same shared `causal_model` functions trace generation uses — never
hardcoded to `after_failed=False` (`tests/test_shared_causal_invariants.py`).
Repairs that touch no causal input relevant to groundedness (retrieval
score, entity match, hallucination/vade penalty) reuse the original
groundedness value exactly, rather than recomputing it through the
noise-free formula and shifting the number by denoising alone.

`learning_memory.py` persists every repair outcome to a JSON file keyed by
`(node_id, failure_type, action)`, with epsilon-greedy candidate selection
(`choose_candidates`) that changes what gets tried in future runs based on
stored accept-rate history. `continuous_improvement.py` runs the full
failed-traces → attribute → repair-competition → record-to-memory →
aggregate-report loop over a batch; `run_continuous_improvement.py` is the
CLI entry point (`--reset-memory` for a cold start).

## What's actually implemented, and what it's based on

| Module | Does | Grounded in |
|---|---|---|
| `graph.py` | DAG model of the pipeline (nodes = pipeline stages, edges = calls/depends_on/escalates_to) | AFlow's code-represented workflow graphs (Zhang et al. 2024, arXiv:2410.10762); GPTSwarm's graph formulation (Zhuge et al. 2024) |
| `attribution.py` | 3 failure-attribution methods: all-at-once, step-by-step, binary-search | **Zhang, Yin, Zhang et al., "Which Agent Causes Task Failures and When?"**, ICML 2025 Spotlight, arXiv:2505.00212 (the *Who&When* paper — introduces exactly these 3 method families and their cost/accuracy trade-offs) |
| `attribution.py::attribute_a2p_scaffold` | 4th attribution method: Abduction (broader symptom-evidence scan) + Action/Prediction (reuses `repair_engine.generate_and_evaluate_candidates` as a real counterfactual check) | **West, Weng, Zhu, Lin, Zhang, "Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems,"** arXiv:2509.10401, code: github.com/ResearAI/A2P |
| `attribution.py::attribute_with_topology_heuristic` | small confidence boost based on node out-degree | explicitly labeled as a graph-degree heuristic, not causal inference — arXiv:2509.08682's causal-attribution method does interventions/counterfactual removal, which this does not attempt |
| `attribution.py::matches_ground_truth` | single-label and multi-label scoring for traces with more than one genuinely true root cause | this repo's own evaluation harness, motivated by `simulate.py`'s multi-cause trace construction |
| `cost.py` | separates real `api_cost_usd` from a unitless `latency_penalty_score`, `friction_score`, `human_cost_usd` | operationalizes the "expected total conversation cost" trade-off without mislabeling an SLA-pressure proxy as a dollar figure |
| `optimizer.py` | turns aggregated stats (by per-node execution count) into KEEP/RETRY/ADD_FILTER/ASK_CLARIFICATION/RECHUNK/CACHE/HUMAN_REVIEW recommendations | action vocabulary generalizes AFlow's "Operators" (Ensemble, Review, Revise) into a graph-editing verb set |
| `simulate.py` | synthetic trace generator with a causal (not independent) failure chain: chunker/KB staleness → retrieval quality → generation groundedness → evaluator verdict; symptom evidence only, ground truth stored separately | stand-in for real production logs |
| `repair.py` | one concrete counterfactual replay: apply `ADD_FILTER` to the "wrong product variant retrieved" scenario, recompute downstream outcome/cost from the repaired symptoms | hand-written for one scenario, superseded by `repair_engine.py` for the general case |
| `repair_engine.py` | generalizes `repair.py` to any trace + any candidate action, with honest containment-aware accept/reject | see "Repair engine" above |
| `report.py` | static HTML dashboard: graph, per-node stats, attribution accuracy, before/after replay | — |
| `whowhen_adapter.py` | `TraceSource` interface + `SyntheticTraceSource` (wraps `simulate.py`) and `WhoWhenTraceSource` (loads the real Who&When benchmark from a local `.parquet`/`.jsonl`/`.json` file) | loads the same Who&When dataset `attribution.py`'s three base methods implement |
| `learning_memory.py` | persists every repair outcome, epsilon-greedy candidate selection based on stored accept-rate history | the cross-run learning loop |
| `continuous_improvement.py` | runs the full loop over a batch; `run_continuous_improvement.py` is the CLI entry point | closes the attribute → repair → learn loop end-to-end |
| `evaluate_multiseed.py` | runs the full loop across multiple seeds and reports mean/std/min/max for attribution accuracy, repair acceptance rate, human-review rate, and cost/latency, plus an independently-parameterized "adversarial" simulator profile | multi-seed robustness, not a single lucky draw |

**Important honesty check baked into the design:** the Who&When paper's own
headline result is that automated failure attribution is *hard* — the best
method gets 53.5% agent-level accuracy and only 14.2% step-level accuracy,
and SOTA reasoning models (o1, R1) don't solve it either. This prototype's
`optimizer.recommend()` reflects that: when a node is blamed frequently but
attribution *confidence* is low, it recommends `HUMAN_REVIEW` rather than an
automated edit — it does not pretend the attribution signal is more reliable
than the literature says it is.

Current values:
**57.8%** all-at-once / **68.9%** binary-search agent-level accuracy,
**30.0%** failure rate, **n=90** evaluable failed traces, **248 tests**.

## Run it

```bash
python3 demo.py
```

No API keys or external services required — it runs on a heuristic judge
stand-in (`attribution.heuristic_judge`) so you can see the full pipeline
work end-to-end immediately.

```bash
python3 evaluate_multiseed.py            # multi-seed robustness check
python3 run_continuous_improvement.py --reset-memory   # full attribute->repair->learn loop
```

## Tests

```bash
python3 -m unittest discover -s tests -t .
# or, if you have pytest installed:
pip install -r requirements-dev.txt
pytest
```

**248 tests** across every module. `tests/test_readme_freshness.py` fails
this build if the headline numbers above drift out of sync with what
`demo.py`'s `accuracy()` and a live `unittest.TestLoader` count actually
produce, so they can't silently go stale.

A few tests worth knowing about specifically:

- `tests/test_attribution.py::TestNoLeakage` — asserts ground truth (or a
  "this-step-failed" flag) never ends up in what the judge sees.
- `tests/test_whowhen_adapter.py::TestJsonlLoading::test_ground_truth_never_leaks_into_rendered_symptoms`
  enforces the same invariant for real-benchmark traces.
- `tests/fixtures/whowhen_real_sample.jsonl` + `tests/test_whowhen_real_sample.py`
  — 8 rows hand-transcribed from the real Who&When HF dataset viewer, used
  to confirm column names/types and the leakage invariant against the real
  schema. See "Wiring in the real Who&When benchmark" below for exactly
  what this does and doesn't prove about the full dataset.
- `tests/test_demo_regression.py` pins the headline accuracy numbers this
  README quotes, plus the optimizer stress-test batch.
- `tests/test_live_driver_choice.py` locks in the a2p_scaffold-vs-
  binary_search comparison directionally across 5 seeds.

One thing worth flagging honestly: in the shipped showcase scenario
(`simulate.wrong_variant_scenario`), the all-at-once attributor
**misattributes** the failure (guesses `generator`, ground truth is
`retriever`) — the entity-mismatch symptom at the retriever and the
low-groundedness symptom at the generator score close enough under the
heuristic's rules that this particular trace tips the wrong way. That's a
faithful illustration of exactly the ambiguity the Who&When paper reports
(early errors that only become visible downstream are hard to localize),
not something patched over. The repair replay still runs correctly because
it's keyed to the *scenario*, not to the attributor's possibly-wrong guess —
in a real system you'd want the repair step to only fire on high-confidence,
correctly localized attributions, which is what `optimizer.py`'s
`HUMAN_REVIEW` branch is for.

## Wiring in the real Who&When benchmark

`whowhen_adapter.py` loads the actual dataset (Zhang et al., ICML 2025
Spotlight, arXiv:2505.00212 — 184 rows across an "Algorithm-Generated" and
a "Hand-Crafted" split), rather than `simulate.py`'s synthetic stand-in.
`FailureAttributor` runs against its output completely unmodified — that's
the point of the shared `TraceSource` interface both `SyntheticTraceSource`
and `WhoWhenTraceSource` implement.

```bash
# download the dataset first, e.g.:
#   from datasets import load_dataset
#   load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")["train"] \
#       .to_json("algorithm_generated.jsonl")
python3 run_with_whowhen.py algorithm_generated.jsonl
python3 run_with_whowhen.py algorithm_generated.jsonl --judge claude   # needs ANTHROPIC_API_KEY
```

Two things worth knowing before you run it:

- **It defaults to `heuristic_judge`**, same as `demo.py`, so you can
  confirm the loader and attribution plumbing work end-to-end for free.
  `heuristic_judge`'s rules key off synthetic numeric fields
  (`retrieval_top1_score`, `groundedness`, `kb_age_days`, ...) that don't
  exist in Who&When's free-text conversation turns, so expect it to mostly
  return `RESPONSIBLE=NONE` — that's the honest, documented, expected
  result, not a bug. Pass `--judge claude` for a meaningful number.
- **Which dict key names the speaking agent in each `history` turn is not
  independently confirmed** — see `whowhen_adapter.py`'s module docstring
  for exactly what's verified from the HF dataset card versus inferred.
  `run_with_whowhen.py` prints a warning with the exact fallback count if
  `_extract_speaker`'s guesses don't match your actual file, so this fails
  visibly instead of silently mislabeling every step.

## Wiring in a real LLM judge

`llm_judge.py` is a real implementation, not a placeholder snippet. It
supports:

- **Structured output** via a forced tool call (`submit_verdict`), not
  regex-parsed free text
- **Anthropic API and Bedrock Claude**, as separate `TransportFn`s so
  neither `anthropic` nor `boto3` is a hard dependency unless you use it
- **Retry with backoff** for transient failures (rate limits, timeouts,
  5xx), plus one corrective retry for schema-invalid responses, before
  falling back to `RESPONSIBLE=NONE` rather than crashing a batch run
- **Response caching**, in-memory always and optionally persisted to a JSON
  file (`cache_path`) so replaying the same trace doesn't re-pay for or
  re-query the same verdict
- **Real cost logging** (`judge.ledger.total_cost_usd()`) against
  Anthropic's published per-token pricing, separate from `cost.py`'s
  `USD_PER_1K_TOKENS`, which is a rough proxy for "some LLM call happened
  here," not a priced Claude call
- **Model/prompt versioning** — every cache key and cost-log entry carries
  the model id and prompt version, so changing the prompt template later
  can't silently mix with old cached verdicts

Usage — no changes to `attribution.py` required, because
`make_claude_judge()` returns the same `Callable[[str], str]` signature
`heuristic_judge` already has:

```python
from llm_judge import make_claude_judge, ClaudeJudgeConfig

judge = make_claude_judge(ClaudeJudgeConfig(
    model="claude-haiku-4-5-20251001",       # cheapest current model; this is a
                                              # narrow classification task, not
                                              # reasoning-heavy -- upgrade to
                                              # claude-sonnet-5 if accuracy needs it
    known_node_ids=frozenset(graph.nodes.keys()),
    cache_path="llm_judge_cache.json",
))
attributor = FailureAttributor(judge=judge)
```

Or see `run_with_real_judge.py` for a complete, runnable example against
the same 300-trace batch `demo.py` reports on (requires
`pip install -r requirements.txt` and `ANTHROPIC_API_KEY` set).

**Honesty check on this module itself:** the Anthropic/Bedrock transports
are verified by unit tests against an injected fake transport
(`tests/test_llm_judge.py`, 24 tests covering caching, retry, schema
validation, and cost accounting). The tool-call *shape* sent to each API
matches the current documented request/response format for both
Anthropic's Messages API and Bedrock's Converse API, but neither transport
has been exercised against a live key in this environment (no network
access). Treat first real use as the actual integration test.

The all-at-once / step-by-step / binary-search / a2p_scaffold methods are
judge-agnostic — swapping this one function upgrades the whole system to a
real LLM-as-judge, matching the Who&When paper's actual experimental setup.

## What's a real prototype vs. what's still a stub

- **Real and runnable**: graph model, all 4 attribution methods, cost
  accounting, rule-based optimizer, synthetic evaluation harness,
  `whowhen_adapter.py`'s loader (unit-tested against synthetic
  Who&When-shaped records — the real dataset file itself was never
  downloaded in this environment), and `llm_judge.py`'s caching/retry/
  validation/cost-logging logic (unit-tested against a fake transport).
- **Deliberately a stub, clearly marked**: `simulate.py` (swap for real
  trace logs, or use `whowhen_adapter.WhoWhenTraceSource` for the real
  Who&When benchmark instead). `heuristic_judge` remains available and is
  still what `demo.py` uses by default, specifically so the
  zero-dependency, no-API-key demo keeps working.
- **Not implemented (next phase)**: full AFlow-style MCTS search over
  candidate graph edits with rollout-based re-scoring. The current
  `optimizer.recommend()` is an interpretable rule-based policy, which is
  the right MVP — MCTS search requires an executable environment to roll
  out and score candidate graphs, which only makes sense once you have real
  traces to evaluate against.

## What you can honestly claim about this today

> "Built a synthetic closed-loop prototype that represents an agentic
> RAG/support workflow as a DAG, generates synthetic traces with realistic
> non-leaking symptom evidence, attributes failures to a likely responsible
> node using four attribution strategies (three published Who&When method
> families plus a published A2P scaffold implementation — measured against
> real ground truth, not leaked labels), and demonstrates executable
> attribute → repair → replay cycles with a before/after cost comparison.
> The attribution tooling is compatible with the public Who&When benchmark
> format."

Do **not** claim: that this solves any specific company's internal
pipeline (this repo has not been run against one), a reproduction of
Who&When's published results (different dataset, different — much
simpler — judge), a causal-inference implementation, a general
graph-optimization/MCTS search, or a publication-ready evaluation. Those
are the clearly-scoped next phases, not things this code currently does.

## Illustrative integration note: a guardrail-graph-optimizer action space

This is a sketch of how the action vocabulary here could map onto a
similarly-shaped guardrail-optimization system, not a claim that this repo
integrates with, has been run against, or solves any specific deployed
system. An action space like KEEP / REUSE / SKIP / MOVE / HUMAN_REVIEW maps
directly onto `graph.ActionType` here — the same policy interface could
drive either a guardrail-redundancy graph or a full pipeline-stage graph.
The natural integration point: treat guardrail nodes as just another node
`kind` in `WorkflowGraph`, and the same `FailureAttributor` +
`GraphOptimizer` pair runs over both.
