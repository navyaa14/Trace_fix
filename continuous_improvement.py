
from __future__ import annotations
from dataclasses import dataclass, field

from graph import WorkflowGraph, ActionType
from attribution import FailureAttributor, Trace, AttributionResult, infer_failure_type
from optimizer import GraphOptimizer
from repair_engine import (
    evaluate_repair, generate_and_evaluate_candidates, select_best_candidate,
    CANDIDATE_ACTIONS, ValidatedRepair,
)
from learning_memory import LearningMemory, ActionDecision
from cost import CostBreakdown

LOW_CONFIDENCE_THRESHOLD = 0.45

COMPETITION_WIDTH = 2


@dataclass
class RepairDecision:
    trace_id: str
    node_id: str
    failure_type: str
    inferred_failure_type: str
    attribution_confidence: float
    policy_decision: ActionDecision | None
    candidates_tried: list[ValidatedRepair]
    winner: ValidatedRepair | None
    outcome: str


@dataclass
class ImprovementReport:
    attempted: int
    accepted: int
    rejected: int
    not_appliable: int
    human_review_selected: int
    unresolved: int
    total_before_cost_usd: float
    total_after_cost_usd: float
    human_escalations_before: int
    human_escalations_after: int
    repairs: list[ValidatedRepair]
    decisions: list[RepairDecision] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else 0.0


def _auto_candidates(node_id: str) -> list[str]:
    return [a.value for a in CANDIDATE_ACTIONS.get(node_id, []) if a != ActionType.HUMAN_REVIEW]


def run_improvement_cycle(graph: WorkflowGraph, traces: list[Trace],
                           attributions: list[AttributionResult],
                           memory: LearningMemory,
                           action_type_cls,
                           epsilon: float = 0.15,
                           rng=None) -> ImprovementReport:
    repairs: list[ValidatedRepair] = []
    decisions: list[RepairDecision] = []
    before_total = CostBreakdown()
    after_total = CostBreakdown()
    human_before = 0
    human_after = 0
    human_review_selected = 0
    unresolved = 0

    for trace, attr in zip(traces, attributions):
        if not trace.final_outcome_failed or not attr.responsible_node:
            continue
        node_id = attr.responsible_node
        inferred_failure_type = infer_failure_type(trace)
        candidate_names = _auto_candidates(node_id)

        before_cost = None
        winner: ValidatedRepair | None = None
        tried: list[ValidatedRepair] = []
        policy_decision: ActionDecision | None = None
        human_before += 1

        if not candidate_names:
            outcome = "human_review"
            human_review_selected += 1
        elif attr.confidence < LOW_CONFIDENCE_THRESHOLD:
            outcome = "human_review"
            human_review_selected += 1
        else:
            policy_decision = memory.select_action(node_id, inferred_failure_type, candidate_names,
                                                     epsilon=epsilon, rng=rng)
            to_try = [policy_decision.action]
            for name in candidate_names:
                if len(to_try) >= COMPETITION_WIDTH:
                    break
                if name not in to_try:
                    to_try.append(name)

            actions = [action_type_cls(name) for name in to_try]
            tried = generate_and_evaluate_candidates(graph, trace, node_id, actions,
                                                      failure_type_override=inferred_failure_type)
            for r in tried:
                memory.record(r)
                repairs.append(r)
            winner = select_best_candidate(tried)
            if winner is not None:
                outcome = "accepted"
            else:
                outcome = "unresolved"
                unresolved += 1

        before_cost = tried[0].before_cost if tried else _trace_cost_fallback(trace)

        before_total.api_cost_usd += before_cost.api_cost_usd
        before_total.human_cost_usd += before_cost.human_cost_usd

        if winner is not None:
            after_total.api_cost_usd += winner.after_cost.api_cost_usd
            after_total.human_cost_usd += winner.after_cost.human_cost_usd
            # An accepted repair is either fully resolved (after_failed=False)
            # or, for vade/evaluator containment repairs, still content_failed
            # but with the user-visible failure it targets resolved -- both
            # are legitimate accept states now that the two are tracked
            # separately (see evaluate_repair / causal_model.contained).
            assert not winner.after_failed or (
                winner.after_trace is not None and winner.after_trace.user_visible_failure is False
            ), "accepted repair must not be marked failed unless it is a validated containment repair"
        else:
            after_total.api_cost_usd += before_cost.api_cost_usd
            after_total.human_cost_usd += before_cost.human_cost_usd
            human_after += 1

        decisions.append(RepairDecision(
            trace_id=trace.trace_id, node_id=node_id, failure_type=trace.failure_type or "unknown",
            inferred_failure_type=inferred_failure_type,
            attribution_confidence=attr.confidence, policy_decision=policy_decision,
            candidates_tried=tried, winner=winner, outcome=outcome,
        ))

    memory.save()

    accepted = sum(1 for r in repairs if r.applied and r.accepted)
    rejected = sum(1 for r in repairs if r.applied and not r.accepted)
    not_appliable = sum(1 for r in repairs if not r.applied)

    return ImprovementReport(
        attempted=len(repairs),
        accepted=accepted,
        rejected=rejected,
        not_appliable=not_appliable,
        human_review_selected=human_review_selected,
        unresolved=unresolved,
        total_before_cost_usd=round(before_total.api_cost_usd + before_total.human_cost_usd, 4),
        total_after_cost_usd=round(after_total.api_cost_usd + after_total.human_cost_usd, 4),
        human_escalations_before=human_before,
        human_escalations_after=human_after,
        repairs=repairs,
        decisions=decisions,
    )


def _trace_cost_fallback(trace: Trace) -> CostBreakdown:
    from cost import trace_cost
    return trace_cost(trace)
