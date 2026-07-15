
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EdgeType(str, Enum):
    CALLS = "calls"
    DEPENDS_ON = "depends_on"
    RETRIED_BY = "retried_by"
    ESCALATES_TO = "escalates_to"


class ActionType(str, Enum):
    KEEP = "KEEP"
    RETRY = "RETRY"
    MERGE = "MERGE"
    REMOVE = "REMOVE"
    PARALLELIZE = "PARALLELIZE"
    CACHE = "CACHE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    RETRAIN = "RETRAIN"
    ADD_FILTER = "ADD_FILTER"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    RECHUNK = "RECHUNK"
    ESCALATE = "ESCALATE"
    RETRY_VALIDATION = "RETRY_VALIDATION"
    LOWER_DETECTION_THRESHOLD = "LOWER_DETECTION_THRESHOLD"
    SECOND_JUDGE = "SECOND_JUDGE"
    RECALIBRATE_THRESHOLD = "RECALIBRATE_THRESHOLD"


@dataclass
class NodeSpec:
    node_id: str
    kind: str
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    is_llm: bool = True
    description: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    edge_type: EdgeType = EdgeType.CALLS


@dataclass
class WorkflowGraph:
    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, spec: NodeSpec) -> None:
        self.nodes[spec.node_id] = spec

    def add_edge(self, src: str, dst: str, edge_type: EdgeType = EdgeType.CALLS) -> None:
        assert src in self.nodes and dst in self.nodes, "both endpoints must exist"
        self.edges.append(Edge(src, dst, edge_type))

    def topological_order(self) -> list[str]:
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            indeg[e.dst] += 1
        queue = [n for n, d in indeg.items() if d == 0]
        order = []
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for e in self.edges:
            adj[e.src].append(e.dst)
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        return order

    def successors(self, node_id: str) -> list[str]:
        return [e.dst for e in self.edges if e.src == node_id]

    def predecessors(self, node_id: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == node_id]


def build_support_pipeline() -> WorkflowGraph:
    g = WorkflowGraph()
    g.add_node(NodeSpec("chunker", "chunker", avg_latency_ms=180, avg_tokens=600,
                         description="AutoChunker-inspired structure-aware chunking (not a reimplementation)"))
    g.add_node(NodeSpec("kb_builder", "knowledge_base", avg_latency_ms=220, avg_tokens=900,
                         description="AutoKB-inspired structured KB construction (not a reimplementation)"))
    g.add_node(NodeSpec("retriever", "retriever", avg_latency_ms=90, avg_tokens=300, is_llm=False,
                         description="dense retrieval over KB"))
    g.add_node(NodeSpec("clarifier", "clarifier", avg_latency_ms=250, avg_tokens=400,
                         description="ASK-inspired clarification-question generation (not a reimplementation)"))
    g.add_node(NodeSpec("generator", "generator", avg_latency_ms=400, avg_tokens=800,
                         description="SMART-inspired response generation (not a reimplementation)"))
    g.add_node(NodeSpec("vade", "hallucination_check", avg_latency_ms=150, avg_tokens=250,
                         description="VADE-inspired hallucination check (not a reimplementation)"))
    g.add_node(NodeSpec("evaluator", "evaluator", avg_latency_ms=300, avg_tokens=500,
                         description="AutoEval-ToD-style automated scoring"))
    g.add_node(NodeSpec("human", "human_review", avg_latency_ms=0, avg_tokens=0, is_llm=False,
                         description="human-in-the-loop escalation"))

    g.add_edge("chunker", "kb_builder", EdgeType.CALLS)
    g.add_edge("kb_builder", "retriever", EdgeType.DEPENDS_ON)
    g.add_edge("retriever", "clarifier", EdgeType.CALLS)
    g.add_edge("clarifier", "generator", EdgeType.CALLS)
    g.add_edge("retriever", "generator", EdgeType.DEPENDS_ON)
    g.add_edge("generator", "vade", EdgeType.CALLS)
    g.add_edge("generator", "evaluator", EdgeType.CALLS)
    g.add_edge("vade", "evaluator", EdgeType.DEPENDS_ON)
    g.add_edge("evaluator", "human", EdgeType.ESCALATES_TO)
    return g
