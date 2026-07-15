
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from graph import WorkflowGraph, ActionType

_STALE_KB_REASON_RE = re.compile(r"^stale_KB_(\d+)d$")
_LOW_RETRIEVAL_REASON_RE = re.compile(r"^low_retrieval_score_([\d.]+)$")
_LOW_GROUNDEDNESS_REASON_RE = re.compile(r"^low_groundedness_([\d.]+)_despite_ok_retrieval$")
_CHUNK_SPLIT_REASON_RE = re.compile(r"^entity_span_split_coverage_([\d.]+)$")
_VADE_MISS_REASON_RE = re.compile(r"^vade_missed_hallucination_conf_([\d.]+)$")
_EVAL_FALSE_ESCALATION_REASON_RE = re.compile(r"^evaluator_false_escalation_score_([\d.]+)$")

_STALE_KB_THRESHOLD_DAYS = 60
_LOW_RETRIEVAL_THRESHOLD = 0.55
_LOW_GROUNDEDNESS_THRESHOLD = 0.5
_LOW_COVERAGE_THRESHOLD = 0.6


def _hypothesis_symptom_actually_resolved(node_id: str, reason: str, after_trace) -> bool:
    step = next((s for s in after_trace.steps if s.node_id == node_id), None)
    if step is None:
        return True

    m = _STALE_KB_REASON_RE.match(reason)
    if m:
        try:
            return float(step.symptoms.get("kb_age_days", 999.0)) <= _STALE_KB_THRESHOLD_DAYS
        except (TypeError, ValueError):
            return True

    m = _LOW_RETRIEVAL_REASON_RE.match(reason)
    if m:
        try:
            return float(step.symptoms.get("retrieval_top1_score", 0.0)) >= _LOW_RETRIEVAL_THRESHOLD
        except (TypeError, ValueError):
            return True

    m = _LOW_GROUNDEDNESS_REASON_RE.match(reason)
    if m:
        try:
            return float(step.symptoms.get("groundedness", 0.0)) >= _LOW_GROUNDEDNESS_THRESHOLD
        except (TypeError, ValueError):
            return True

    m = _CHUNK_SPLIT_REASON_RE.match(reason)
    if m:
        # confirmed only if RECHUNK actually cleared the chunker's own evidence
        # (entity_span_split back to False), not merely a downstream flag flip.
        return step.symptoms.get("entity_span_split") == "False"

    m = _VADE_MISS_REASON_RE.match(reason)
    if m:
        return step.symptoms.get("vade_flagged") == "True"

    m = _EVAL_FALSE_ESCALATION_REASON_RE.match(reason)
    if m:
        return step.symptoms.get("evaluator_flags_low_score") == "False"

    return True


@dataclass
class TraceStep:
    node_id: str
    symptoms: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    tokens: int = 0


@dataclass
class Trace:
    trace_id: str
    steps: list[TraceStep]
    final_outcome_failed: bool
    ground_truth_node: Optional[str] = None
    ground_truth_step: Optional[int] = None
    scenario: str = "generic"
    failure_type: Optional[str] = None
    evaluator_disagreement: Optional[str] = None
    ground_truth_nodes: Optional[list[str]] = None
    content_failed: Optional[bool] = None
    evaluation_failed: bool = False
    evaluator_false_acceptance: bool = False
    contained: Optional[bool] = None
    user_visible_failure: Optional[bool] = None
    workflow_outcome: Optional[str] = None



@dataclass
class AttributionResult:
    trace_id: str
    responsible_node: Optional[str]
    confidence: float
    method: str
    judge_calls_used: int
    evidence: str = ""


JudgeFn = Callable[[str], str]

# Below this many evaluated traces, a per-failure-type accuracy number is a
# single (or handful of) coin flip(s), not a measurement -- 0% on n=1 is not
# evidence a method "fails" on that category, it's evidence there's one
# example. demo.py and report.py both use this to flag such rows instead of
# reporting them with the same implied confidence as a real sample.
MIN_RELIABLE_N = 5


def matches_ground_truth(trace: Trace, predicted_node: Optional[str]) -> bool:
    if trace.ground_truth_nodes:
        return predicted_node in trace.ground_truth_nodes
    return predicted_node == trace.ground_truth_node



def _rank_symptom_evidence(prompt: str) -> list[tuple[str, float, str]]:
    hits: list[tuple[str, float, str]] = []
    for line in (l for l in prompt.splitlines() if "node=" in l):
        node = line.split("node=")[1].split()[0]

        def _get(key: str):
            if f"{key}=" not in line:
                return None
            raw = line.split(f"{key}=")[1].split()[0]
            try:
                return float(raw)
            except ValueError:
                return raw

        retrieval_score = _get("retrieval_top1_score")
        entity_match = _get("entity_match")
        groundedness = _get("groundedness")
        clarified = _get("clarification_asked")
        kb_age_days = _get("kb_age_days")
        context_coverage_score = _get("context_coverage_score")
        entity_span_split = _get("entity_span_split")
        vade_flagged = _get("vade_flagged")
        vade_confidence = _get("vade_confidence")
        hallucination_present = _get("hallucination_present")
        evaluator_flags_low_score = _get("evaluator_flags_low_score")
        final_score = _get("final_score")

        if kb_age_days is not None and kb_age_days > 60:
            hits.append((node, min(0.75, 0.4 + kb_age_days / 400), f"stale_KB_{kb_age_days:.0f}d"))

        if retrieval_score is not None and retrieval_score < 0.55:
            hits.append((node, min(0.9, 0.6 + (0.55 - retrieval_score)),
                         f"low_retrieval_score_{retrieval_score:.2f}"))

        if entity_match == "False" and clarified in (None, "False"):
            hits.append((node, 0.55, "entity_mismatch_no_clarification"))

        if groundedness is not None and groundedness < 0.5 and (retrieval_score is None or retrieval_score >= 0.55):
            hits.append((node, min(0.85, 0.5 + (0.5 - groundedness)),
                         f"low_groundedness_{groundedness:.2f}_despite_ok_retrieval"))

        if "query_ambiguous=True" in line and "clarification_asked=False" in line:
            hits.append((node, 0.60, "query_ambiguous_never_clarified"))

        # --- chunker: direct chunk-quality evidence, not merely a low retrieval
        # score. entity_span_split is the chunker's own symptom, so it is ranked
        # here on the chunker node itself, distinguishable from a retriever- or
        # kb_builder-originated retrieval problem.
        if entity_span_split == "True" and context_coverage_score is not None:
            hits.append((node, min(0.88, 0.55 + (0.6 - context_coverage_score)),
                         f"entity_span_split_coverage_{context_coverage_score:.2f}"))

        # --- vade: the validator's own miss, visible directly on the vade node
        # (hallucination_present=True but vade_flagged=False), not inferred from
        # generator symptoms.
        if hallucination_present == "True" and vade_flagged == "False":
            conf = vade_confidence if vade_confidence is not None else 0.0
            hits.append((node, 0.62, f"vade_missed_hallucination_conf_{conf:.2f}"))

        # --- evaluator: a false escalation is visible directly as
        # evaluator_flags_low_score=True with no corroborating upstream
        # groundedness/retrieval problem visible on this same trace.
        if (evaluator_flags_low_score == "True" and (groundedness is None or groundedness >= 0.5)
                and (retrieval_score is None or retrieval_score >= 0.55)):
            score = final_score if final_score is not None else 0.0
            hits.append((node, 0.50, f"evaluator_false_escalation_score_{score:.2f}"))

    return hits


def heuristic_judge(prompt: str) -> str:
    best_node, best_score, best_reason = None, 0.0, ""
    for node, score, reason in _rank_symptom_evidence(prompt):
        if score > best_score:
            best_node, best_score, best_reason = node, score, reason

    if best_node:
        return f"RESPONSIBLE={best_node} CONFIDENCE={best_score:.2f} REASON={best_reason}"
    return "RESPONSIBLE=NONE CONFIDENCE=0.20 REASON=no_clear_symptom"


def _abduce_hypotheses(prompt: str) -> list[tuple[str, float, str]]:
    hits = list(_rank_symptom_evidence(prompt))

    best_per_node: dict[str, tuple[float, str]] = {}
    for node, score, reason in hits:
        if node not in best_per_node or score > best_per_node[node][0]:
            best_per_node[node] = (score, reason)
    return sorted(((n, s, r) for n, (s, r) in best_per_node.items()), key=lambda t: -t[1])


def infer_failure_type(trace: Trace) -> str:
    sym = {step.node_id: step.symptoms for step in trace.steps}
    gen = sym.get("generator", {})
    retr = sym.get("retriever", {})
    kb = sym.get("kb_builder", {})
    clar = sym.get("clarifier", {})

    if gen.get("safety_flag"):
        return "safety_regression"

    kb_age = float(kb.get("kb_age_days", 0.0)) if kb else 0.0
    stale = kb_age > 60.0
    variant_suspected = retr.get("variant_mismatch_suspected") == "True"
    entity_match = retr.get("entity_match") == "True"
    ambiguous = clar.get("query_ambiguous") == "True"
    asked = clar.get("clarification_asked") == "True"
    resolved = clar.get("clarification_resolved") == "True"
    halluc_risk = gen.get("hallucination_risk", "none")

    entity_side = None
    if stale and not entity_match:
        entity_side = "stale_knowledge_correct_entity"
    elif variant_suspected or not entity_match:
        entity_side = "wrong_entity_variant"
    if ambiguous and not asked:
        entity_side = "ambiguous_query_unclarified"
    elif ambiguous and asked and not resolved:
        entity_side = "clarification_failed_annoyed_user"

    halluc_side = None
    if halluc_risk == "sticky":
        halluc_side = "repeated_hallucination"
    elif halluc_risk == "transient":
        halluc_side = "unsupported_generation_transient"

    if entity_side and halluc_side:
        return "multiple_simultaneous_failures"
    return entity_side or halluc_side or "unknown"


class FailureAttributor:
    def __init__(self, judge: JudgeFn = heuristic_judge):
        self.judge = judge

    def attribute_all_at_once(self, trace: Trace) -> AttributionResult:
        verdict = self.judge(self._render_full_trace(trace))
        node, conf, reason = self._parse_verdict(verdict)
        return AttributionResult(trace.trace_id, node, conf, "all_at_once", 1, reason)

    def attribute_step_by_step(self, trace: Trace) -> AttributionResult:
        calls = 0
        best = AttributionResult(trace.trace_id, None, 0.0, "step_by_step", 0, "")
        for i in range(len(trace.steps)):
            calls += 1
            node, conf, reason = self._parse_verdict(self.judge(self._render_prefix(trace, i)))
            if node and node != "NONE" and conf > best.confidence:
                best = AttributionResult(trace.trace_id, node, conf, "step_by_step", calls, reason)
        return AttributionResult(best.trace_id, best.responsible_node, best.confidence,
                                  "step_by_step", calls, best.evidence)

    def attribute_binary_search(self, trace: Trace) -> AttributionResult:
        lo, hi = 0, len(trace.steps) - 1
        calls = 0
        candidate, conf, reason = None, 0.0, ""
        while lo <= hi:
            mid = (lo + hi) // 2
            calls += 1
            node, c, r = self._parse_verdict(self.judge(self._render_prefix(trace, mid)))
            if node and node != "NONE":
                candidate, conf, reason = node, c, r
                hi = mid - 1
            else:
                lo = mid + 1
        return AttributionResult(trace.trace_id, candidate, conf, "binary_search", calls, reason)

    def attribute_with_topology_heuristic(self, trace: Trace, graph: WorkflowGraph) -> AttributionResult:
        base = self.attribute_all_at_once(trace)
        if base.responsible_node is None:
            return base
        downstream = len(graph.successors(base.responsible_node))
        boost = min(0.15, 0.04 * downstream)
        return AttributionResult(base.trace_id, base.responsible_node,
                                  min(1.0, base.confidence + boost),
                                  "topology_heuristic", base.judge_calls_used, base.evidence)

    def attribute_a2p_scaffold(self, trace: Trace, graph: Optional[WorkflowGraph] = None,
                                top_k_hypotheses: int = 3) -> AttributionResult:
        prompt = self._render_full_trace(trace)
        judge_node, judge_conf, judge_reason = self._parse_verdict(self.judge(prompt))

        pool: dict[str, tuple[float, str]] = {}
        for node, score, reason in _abduce_hypotheses(prompt):
            pool[node] = (score, reason)
        if judge_node and judge_node != "NONE" and judge_node not in pool:
            pool[judge_node] = (judge_conf, judge_reason)

        ranked = sorted(((n, s, r) for n, (s, r) in pool.items()), key=lambda t: -t[1])
        ranked = ranked[:max(1, top_k_hypotheses)]

        if not ranked:
            return AttributionResult(trace.trace_id, None, 0.20, "a2p_scaffold", 1, "no_clear_symptom")

        counterfactual_checks = 0
        if graph is not None:
            from repair_engine import CANDIDATE_ACTIONS, generate_and_evaluate_candidates, select_best_candidate
            for node, score, reason in ranked:
                candidate_actions = [a for a in CANDIDATE_ACTIONS.get(node, []) if a != ActionType.HUMAN_REVIEW]
                if not candidate_actions:
                    continue
                results = generate_and_evaluate_candidates(graph, trace, node, candidate_actions)
                counterfactual_checks += len(results)
                confirmed = [r for r in results
                             if r.accepted and r.after_trace is not None
                             and _hypothesis_symptom_actually_resolved(node, reason, r.after_trace)]
                winner = select_best_candidate(confirmed)
                if winner is not None:
                    conf = min(0.97, score + 0.15)
                    evidence = f"{reason};causally_confirmed_via={winner.action.value};counterfactual_checks={counterfactual_checks}"
                    return AttributionResult(trace.trace_id, node, conf, "a2p_scaffold", 1, evidence)

        node, score, reason = ranked[0]
        tag = "counterfactual_unconfirmed" if graph is not None else "counterfactual_unavailable_no_graph"
        evidence = f"{reason};{tag};counterfactual_checks={counterfactual_checks}"
        return AttributionResult(trace.trace_id, node, score, "a2p_scaffold", 1, evidence)

    @staticmethod
    def _render_step_line(step: TraceStep) -> str:
        parts = [f"node={step.node_id}"] + [f"{k}={v}" for k, v in step.symptoms.items()]
        return " ".join(parts)

    @classmethod
    def _render_full_trace(cls, trace: Trace) -> str:
        return "\n".join(cls._render_step_line(s) for s in trace.steps)

    @classmethod
    def _render_prefix(cls, trace: Trace, upto_idx: int) -> str:
        return "\n".join(cls._render_step_line(s) for s in trace.steps[: upto_idx + 1])

    @staticmethod
    def _parse_verdict(verdict: str) -> tuple[Optional[str], float, str]:
        node, conf, reason = None, 0.0, ""
        for tok in verdict.split():
            if tok.startswith("RESPONSIBLE="):
                node = tok.split("=", 1)[1]
            elif tok.startswith("CONFIDENCE="):
                conf = float(tok.split("=", 1)[1])
            elif tok.startswith("REASON="):
                reason = tok.split("=", 1)[1].replace("_", " ")
        return node, conf, reason
