
from __future__ import annotations
from dataclasses import dataclass

from graph import WorkflowGraph
from attribution import Trace

USD_PER_1K_TOKENS = 0.003
HUMAN_ESCALATION_USD = 2.50
CLARIFICATION_TURN_USD = 0.02


@dataclass
class CostBreakdown:
    api_cost_usd: float = 0.0
    latency_penalty_score: float = 0.0
    friction_score: float = 0.0
    human_cost_usd: float = 0.0

    def weighted_objective(self, w_api=1.0, w_latency=1.0, w_friction=1.0, w_human=1.0) -> float:
        return (w_api * self.api_cost_usd
                + w_latency * self.latency_penalty_score
                + w_friction * self.friction_score
                + w_human * self.human_cost_usd)


def _latency_penalty(latency_ms: float, sla_ms: float = 300.0) -> float:
    if latency_ms <= sla_ms:
        return 0.0
    overage = (latency_ms - sla_ms) / sla_ms
    return round(overage ** 1.5, 4)


def step_cost(node_id: str, latency_ms: float, tokens: int, clarification_asked: bool = True) -> CostBreakdown:
    api = (tokens / 1000.0) * USD_PER_1K_TOKENS
    lat_pen = _latency_penalty(latency_ms)
    friction = 1.0 if (node_id == "clarifier" and clarification_asked) else 0.0
    human = HUMAN_ESCALATION_USD if node_id == "human" else 0.0
    if node_id == "clarifier" and clarification_asked:
        api += CLARIFICATION_TURN_USD
    return CostBreakdown(api_cost_usd=api, latency_penalty_score=lat_pen,
                          friction_score=friction, human_cost_usd=human)


def trace_cost(trace: Trace) -> CostBreakdown:
    total = CostBreakdown()
    for step in trace.steps:
        if step.node_id == "clarifier":
            asked = step.symptoms.get("clarification_asked") == "True"
            c = step_cost(step.node_id, step.latency_ms, step.tokens, clarification_asked=asked)
            turns = float(step.symptoms.get("clarification_turns", 1)) if asked else 0.0
            if asked and turns != 1.0:
                c.friction_score = turns
                c.api_cost_usd += CLARIFICATION_TURN_USD * (turns - 1.0)
        else:
            c = step_cost(step.node_id, step.latency_ms, step.tokens)
        total.api_cost_usd += c.api_cost_usd
        total.latency_penalty_score += c.latency_penalty_score
        total.friction_score += c.friction_score
        total.human_cost_usd += c.human_cost_usd
    return total


def graph_node_cost(graph: WorkflowGraph, node_id: str) -> CostBreakdown:
    spec = graph.nodes[node_id]
    return step_cost(node_id, spec.avg_latency_ms, int(spec.avg_tokens))
