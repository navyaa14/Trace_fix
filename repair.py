
from __future__ import annotations
from dataclasses import dataclass

from graph import WorkflowGraph, ActionType
from attribution import Trace
from cost import CostBreakdown
from repair_engine import evaluate_repair, apply_repair


@dataclass
class ReplayResult:
    action: ActionType
    before_failed: bool
    after_failed: bool
    before_cost: CostBreakdown
    after_cost: CostBreakdown
    before_groundedness: float
    after_groundedness: float
    explanation: str


def _groundedness_of(trace: Trace) -> float:
    step = next((s for s in trace.steps if s.node_id == "generator"), None)
    return float(step.symptoms.get("groundedness", 0.0)) if step else 0.0


def replay_with_add_filter(graph: WorkflowGraph, trace: Trace) -> ReplayResult:
    result = evaluate_repair(graph, trace, ActionType.ADD_FILTER, "retriever")
    before_groundedness = _groundedness_of(trace)

    if not result.applied:
        return ReplayResult(
            action=ActionType.ADD_FILTER, before_failed=trace.final_outcome_failed,
            after_failed=trace.final_outcome_failed, before_cost=result.before_cost,
            after_cost=result.before_cost, before_groundedness=before_groundedness,
            after_groundedness=before_groundedness,
            explanation=f"ADD_FILTER not executed: {result.reason}",
        )

    after_trace = apply_repair(graph, trace, ActionType.ADD_FILTER, "retriever")
    after_groundedness = _groundedness_of(after_trace)

    outcome_word = "removing the human-escalation cost" if result.accepted else \
        "but the measured outcome did not clear the accept bar (see reason)"
    explanation = (
        f"Retriever returned a mismatched entity/variant for this query (entity_match=False) "
        f"-> generator produced groundedness={before_groundedness:.2f} -> "
        f"{'evaluator flagged it, escalated to a human ($2.50)' if result.before_failed else 'trace was already passing'}. "
        f"Adding an entity/variant filter at the retriever recomputes groundedness at "
        f"{after_groundedness:.2f} under the same causal model used everywhere else in this "
        f"codebase (causal_model.expected_groundedness, shared by simulate.py and "
        f"repair_engine.py) -- {outcome_word}. "
        f"Reason: {result.reason}"
    )

    return ReplayResult(
        action=ActionType.ADD_FILTER,
        before_failed=result.before_failed,
        after_failed=result.after_failed,
        before_cost=result.before_cost,
        after_cost=result.after_cost,
        before_groundedness=before_groundedness,
        after_groundedness=after_groundedness,
        explanation=explanation,
    )
