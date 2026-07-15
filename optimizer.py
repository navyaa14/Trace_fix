
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

from graph import ActionType, WorkflowGraph
from attribution import AttributionResult, Trace
from cost import graph_node_cost, CostBreakdown, step_cost


@dataclass
class NodeStats:
    node_id: str
    failure_count: int = 0
    executions: int = 0
    avg_confidence: float = 0.0
    configured_cost_estimate: CostBreakdown = None
    observed_total_cost: CostBreakdown = None
    observed_average_cost: CostBreakdown = None

    @property
    def cost(self) -> CostBreakdown:
        return self.configured_cost_estimate

    @property
    def failure_rate(self) -> float:
        return self.failure_count / self.executions if self.executions else 0.0


@dataclass
class Recommendation:
    node_id: str
    action: ActionType
    reason: str
    failure_rate: float
    cost: CostBreakdown


class GraphOptimizer:
    def __init__(self, graph: WorkflowGraph):
        self.graph = graph

    def aggregate(self, traces: list[Trace], attributions: list[AttributionResult]) -> dict[str, NodeStats]:
        executions = defaultdict(int)
        observed_totals: dict[str, CostBreakdown] = defaultdict(CostBreakdown)
        for t in traces:
            for s in t.steps:
                executions[s.node_id] += 1
                if s.node_id == "clarifier":
                    asked = s.symptoms.get("clarification_asked") == "True"
                    c = step_cost(s.node_id, s.latency_ms, s.tokens, clarification_asked=asked)
                else:
                    c = step_cost(s.node_id, s.latency_ms, s.tokens)
                observed_totals[s.node_id].api_cost_usd += c.api_cost_usd
                observed_totals[s.node_id].latency_penalty_score += c.latency_penalty_score
                observed_totals[s.node_id].friction_score += c.friction_score
                observed_totals[s.node_id].human_cost_usd += c.human_cost_usd

        stats = {}
        for n in self.graph.nodes:
            execs = executions.get(n, 0)
            total = observed_totals.get(n, CostBreakdown())
            avg = CostBreakdown(
                api_cost_usd=total.api_cost_usd / execs if execs else 0.0,
                latency_penalty_score=total.latency_penalty_score / execs if execs else 0.0,
                friction_score=total.friction_score / execs if execs else 0.0,
                human_cost_usd=total.human_cost_usd / execs if execs else 0.0,
            ) if execs else CostBreakdown()
            stats[n] = NodeStats(n, executions=execs,
                                  configured_cost_estimate=graph_node_cost(self.graph, n),
                                  observed_total_cost=total, observed_average_cost=avg)
        conf_sums = defaultdict(float)
        for a in attributions:
            if a.responsible_node and a.responsible_node in stats:
                stats[a.responsible_node].failure_count += 1
                conf_sums[a.responsible_node] += a.confidence
        for node_id, s in stats.items():
            if s.failure_count:
                s.avg_confidence = conf_sums[node_id] / s.failure_count
        return stats

    def recommend(self, stats: dict[str, NodeStats],
                   high_failure_threshold: float = 0.20,
                   low_confidence_threshold: float = 0.45,
                   high_latency_penalty: float = 0.15) -> list[Recommendation]:
        recs = []
        for node_id, s in stats.items():
            cost = s.observed_average_cost

            if s.executions == 0:
                recs.append(Recommendation(node_id, ActionType.KEEP,
                                            "node did not execute in this window", 0.0, cost))
                continue

            if s.failure_rate >= high_failure_threshold and s.avg_confidence < low_confidence_threshold:
                recs.append(Recommendation(node_id, ActionType.HUMAN_REVIEW,
                                            f"high failure rate ({s.failure_rate:.0%}) but low attribution "
                                            f"confidence ({s.avg_confidence:.2f}) -- needs human-labeled "
                                            f"traces before an automated edit is safe (cf. Who&When's own "
                                            f"14.2% step-level accuracy ceiling)",
                                            s.failure_rate, cost))
                continue

            if s.failure_rate >= high_failure_threshold and s.avg_confidence >= low_confidence_threshold:
                if node_id == "kb_builder":
                    recs.append(Recommendation(node_id, ActionType.RECHUNK,
                                                f"confidently implicated ({s.failure_rate:.0%}, "
                                                f"conf {s.avg_confidence:.2f}); stale/poorly-structured KB "
                                                f"content -> candidate for re-chunk + re-index, not a "
                                                f"runtime patch", s.failure_rate, cost))
                elif node_id == "retriever":
                    recs.append(Recommendation(node_id, ActionType.ADD_FILTER,
                                                f"confidently implicated ({s.failure_rate:.0%}, "
                                                f"conf {s.avg_confidence:.2f}); add an entity/metadata "
                                                f"filter before generation", s.failure_rate, cost))
                elif node_id == "clarifier":
                    recs.append(Recommendation(node_id, ActionType.ASK_CLARIFICATION,
                                                f"ambiguous queries slipping through ({s.failure_rate:.0%}); "
                                                f"lower the clarification trigger threshold",
                                                s.failure_rate, cost))
                else:
                    recs.append(Recommendation(node_id, ActionType.RETRY,
                                                f"confidently implicated ({s.failure_rate:.0%}, "
                                                f"conf {s.avg_confidence:.2f}); add a bounded retry/self-check",
                                                s.failure_rate, cost))
                continue

            if cost.latency_penalty_score >= high_latency_penalty:
                recs.append(Recommendation(node_id, ActionType.CACHE,
                                            f"observed SLA-pressure score {cost.latency_penalty_score:.2f} "
                                            f"(measured this batch, not the static estimate) with low "
                                            f"failure rate ({s.failure_rate:.0%}) -- cache/reuse candidate",
                                            s.failure_rate, cost))
                continue

            if s.failure_rate == 0:
                recs.append(Recommendation(node_id, ActionType.KEEP,
                                            "no attributed failures in this window",
                                            s.failure_rate, cost))
                continue

            recs.append(Recommendation(node_id, ActionType.KEEP,
                                        "within acceptable failure-rate and latency-penalty bounds",
                                        s.failure_rate, cost))
        return recs
