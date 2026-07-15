
from __future__ import annotations
import math
import random
from dataclasses import dataclass

from graph import WorkflowGraph
from attribution import Trace, TraceStep
from causal_model import (
    sigmoid as _sigmoid,
    retrieval_score_mean,
    groundedness_mean,
    outcome_failed,
    evaluation_failed,
    workflow_failed,
    vade_catches_hallucination,
    contained as _contained,
    user_visible_failure as _user_visible_failure,
    outcome_label as _outcome_label,
)

FAILURE_TYPES = {
    "stale_knowledge_correct_entity",
    "wrong_entity_variant",
    "ambiguous_query_unclarified",
    "clarification_failed_annoyed_user",
    "unsupported_generation_transient",
    "repeated_hallucination",
    "multiple_simultaneous_failures",
    "chunk_boundary_split_entity",
    "vade_missed_hallucination",
    "evaluator_false_escalation",
}


@dataclass
class ScenarioConfig:
    p_kb_stale: float = 0.15
    p_wrong_variant: float = 0.10
    p_ambiguous_query: float = 0.20
    clarifier_catch_rate: float = 0.55
    p_hallucination_transient: float = 0.07
    p_hallucination_sticky: float = 0.05
    evaluator_noise_std: float = 0.30
    # --- chunker: independent chunk-boundary/entity-splitting failure path ---
    p_chunk_entity_split: float = 0.08
    # --- vade: independent validator-blind-spot failure path ---
    p_vade_issue: float = 0.06
    p_vade_catch: float = 0.55


def _bool_str(v: bool) -> str:
    return str(v)


def generate_traces(graph: WorkflowGraph, n: int, config: ScenarioConfig = ScenarioConfig(),
                     seed: int = 7) -> list[Trace]:
    rng = random.Random(seed)
    order = [nid for nid in graph.topological_order() if nid != "human"]
    traces = []

    for i in range(n):
        kb_stale = rng.random() < config.p_kb_stale
        kb_age_days = rng.uniform(70, 200) if kb_stale else rng.uniform(1, 45)
        wrong_variant = rng.random() < config.p_wrong_variant
        ambiguous = rng.random() < config.p_ambiguous_query
        halluc_sticky = rng.random() < config.p_hallucination_sticky
        halluc_transient = (not halluc_sticky) and rng.random() < config.p_hallucination_transient

        # --- chunker: independent of KB staleness. A chunk boundary can split an
        # entity mention even on a perfectly fresh KB, and a stale KB can have
        # perfectly-formed chunks -- these are two distinct failure paths that both
        # degrade retrieval quality through different mechanisms.
        entity_span_split = rng.random() < config.p_chunk_entity_split
        if entity_span_split:
            context_coverage_score = rng.uniform(0.25, 0.55)
            chunk_coherence_score = rng.uniform(0.30, 0.60)
        else:
            context_coverage_score = rng.uniform(0.82, 0.99)
            chunk_coherence_score = rng.uniform(0.80, 0.99)

        retrieval_score = _sigmoid(rng.gauss(
            retrieval_score_mean(kb_stale, entity_span_split, context_coverage_score), 0.6))

        entity_match = True
        if wrong_variant:
            entity_match = False
        elif retrieval_score <= 0.5:
            entity_match = False
        elif ambiguous and rng.random() < 0.30:
            entity_match = False
        elif rng.random() < 0.05:
            entity_match = False

        clarified = False
        clarify_resolved = False
        if ambiguous:
            clarified = rng.random() < 0.7
            if clarified:
                clarify_resolved = rng.random() < config.clarifier_catch_rate
                if clarify_resolved:
                    entity_match = True
                    retrieval_score = min(1.0, retrieval_score + 0.25)

        # --- vade: an independent, catchable content issue distinct from the
        # generator's own hallucination_risk mechanism (which is left untouched
        # below). vade_issue_present never co-occurs with hallucination_risk, so
        # it's always separately attributable evidence, not a relabeling of an
        # existing generator failure.
        vade_issue_present = (not halluc_sticky) and (not halluc_transient) \
            and rng.random() < config.p_vade_issue
        hallucination_present = halluc_sticky or halluc_transient or vade_issue_present
        vade_confidence = _sigmoid(rng.gauss(1.5 if hallucination_present else -1.5, 0.7))
        if vade_issue_present:
            vade_flagged = rng.random() < config.p_vade_catch
        else:
            # evidence-only detection draw for the generator's own hallucination
            # mechanism -- recorded for realism/observability, but (by design,
            # see causal_model.vade_catches_hallucination) it only causally clears
            # the penalty for the *transient* case, matching the generator's own
            # RETRY-clears-transient behavior in repair_engine.
            base_catch = 0.85 if halluc_sticky else (0.5 if halluc_transient else 0.05)
            vade_flagged = rng.random() < base_catch

        halluc_penalty = 3.2 if (halluc_sticky or halluc_transient) else 0.0
        vade_penalty = 3.0 if (vade_issue_present and not vade_flagged) else 0.0
        total_penalty = halluc_penalty + vade_penalty

        groundedness = _sigmoid(rng.gauss(
            groundedness_mean(retrieval_score, entity_match, total_penalty), 0.5))

        content_failed = outcome_failed(groundedness, entity_match, clarified)

        _eps = 1e-6
        g_clamped = min(1 - _eps, max(_eps, groundedness))
        evaluator_score = _sigmoid(rng.gauss(
            math.log(g_clamped / (1 - g_clamped)), config.evaluator_noise_std))
        evaluator_flags_low_score = evaluator_score < 0.5
        evaluator_disagreement = None
        if evaluator_flags_low_score and not content_failed:
            evaluator_disagreement = "false_positive"
        elif (not evaluator_flags_low_score) and content_failed:
            evaluator_disagreement = "false_negative"

        eval_failed = evaluation_failed(evaluator_disagreement)
        failed = workflow_failed(content_failed, eval_failed)
        evaluator_false_acceptance = content_failed and evaluator_disagreement == "false_negative"
        trace_contained = _contained(content_failed, evaluator_flags_low_score)
        trace_user_visible_failure = _user_visible_failure(content_failed, evaluator_flags_low_score)
        trace_workflow_outcome = _outcome_label(content_failed, evaluator_flags_low_score)

        root_cause = None
        failure_type = None
        multi_cause_nodes = None
        if content_failed:
            entity_side_cause = None
            chunk_split_operative = (entity_span_split and not kb_stale and not wrong_variant
                                      and retrieval_score < 0.55)
            if kb_stale and retrieval_score < 0.55:
                entity_side_cause = ("kb_builder", "stale_knowledge_correct_entity")
            elif wrong_variant:
                entity_side_cause = ("retriever", "wrong_entity_variant")
            elif chunk_split_operative:
                entity_side_cause = ("chunker", "chunk_boundary_split_entity")
            elif ambiguous and not clarified:
                entity_side_cause = ("clarifier", "ambiguous_query_unclarified")
            elif ambiguous and clarified and not clarify_resolved:
                entity_side_cause = ("clarifier", "clarification_failed_annoyed_user")
            elif not entity_match and not clarified:
                entity_side_cause = ("retriever", "wrong_entity_variant")

            halluc_cause = None
            if halluc_sticky:
                halluc_cause = ("generator", "repeated_hallucination")
            elif halluc_transient:
                halluc_cause = ("generator", "unsupported_generation_transient")

            vade_cause = None
            if vade_issue_present and not vade_flagged and not entity_side_cause and not halluc_cause:
                vade_cause = ("vade", "vade_missed_hallucination")

            causes = [c for c in (entity_side_cause, halluc_cause) if c]
            if len(causes) == 2:
                root_cause, failure_type = causes[0][0], "multiple_simultaneous_failures"
                multi_cause_nodes = [c[0] for c in causes]
            elif len(causes) == 1:
                root_cause, failure_type = causes[0]
            elif vade_cause:
                root_cause, failure_type = vade_cause
            else:
                root_cause, failure_type = "generator", "unsupported_generation_transient"
        elif eval_failed:
            # content was actually fine; the evaluator itself is the root cause.
            root_cause, failure_type = "evaluator", "evaluator_false_escalation"

        # A false acceptance is causally two things at once: the upstream node
        # that actually produced bad content, AND the evaluator that wrongly
        # let it through. Represent both -- do not replace the upstream cause
        # just to credit the evaluator, and do not credit the evaluator alone
        # when it did not independently originate the defect.
        if evaluator_false_acceptance and root_cause and root_cause != "evaluator":
            if multi_cause_nodes:
                if "evaluator" not in multi_cause_nodes:
                    multi_cause_nodes = multi_cause_nodes + ["evaluator"]
            else:
                multi_cause_nodes = [root_cause, "evaluator"]

        hallucination_risk = "sticky" if halluc_sticky else ("transient" if halluc_transient else "none")

        steps = []
        root_step = None
        for idx, node_id in enumerate(order):
            spec = graph.nodes[node_id]
            latency = max(1.0, rng.gauss(spec.avg_latency_ms, spec.avg_latency_ms * 0.15))
            tokens = max(0, int(rng.gauss(spec.avg_tokens, spec.avg_tokens * 0.15)))
            symptoms = {}

            if node_id == "chunker":
                symptoms = {"context_coverage_score": round(context_coverage_score, 2),
                            "entity_span_split": _bool_str(entity_span_split),
                            "chunk_coherence_score": round(chunk_coherence_score, 2)}
            elif node_id == "kb_builder":
                symptoms = {"kb_age_days": round(kb_age_days, 1)}
            elif node_id == "retriever":
                symptoms = {"retrieval_top1_score": round(retrieval_score, 2),
                            "entity_match": str(entity_match),
                            "variant_mismatch_suspected": str(wrong_variant)}
            elif node_id == "clarifier":
                symptoms = {"clarification_asked": str(clarified),
                            "query_ambiguous": str(ambiguous),
                            "clarification_resolved": str(clarify_resolved)}
            elif node_id == "generator":
                symptoms = {"groundedness": round(groundedness, 2),
                            "entity_match": str(entity_match),
                            "hallucination_risk": hallucination_risk}
            elif node_id == "vade":
                symptoms = {"hallucination_present": _bool_str(hallucination_present),
                            "vade_flagged": _bool_str(vade_flagged),
                            "vade_confidence": round(vade_confidence, 2)}
            elif node_id == "evaluator":
                symptoms = {"final_score": round(evaluator_score, 2),
                            "evaluator_flags_low_score": str(evaluator_flags_low_score)}

            steps.append(TraceStep(node_id=node_id, symptoms=symptoms, latency_ms=latency, tokens=tokens))
            if node_id == root_cause and root_step is None:
                root_step = idx

        if failed:
            steps.append(TraceStep(node_id="human", symptoms={"escalated": "True"}, latency_ms=0, tokens=0))

        trace_failure_type = failure_type
        if not failed and (kb_stale or entity_span_split):
            trace_failure_type = "recovered_upstream_error"

        traces.append(Trace(trace_id=f"t{i}", steps=steps, final_outcome_failed=failed,
                             ground_truth_node=root_cause, ground_truth_step=root_step,
                             scenario="synthetic_causal", failure_type=trace_failure_type,
                             evaluator_disagreement=evaluator_disagreement,
                             ground_truth_nodes=multi_cause_nodes,
                             content_failed=content_failed,
                             evaluation_failed=eval_failed,
                             evaluator_false_acceptance=evaluator_false_acceptance,
                             contained=trace_contained,
                             user_visible_failure=trace_user_visible_failure,
                             workflow_outcome=trace_workflow_outcome))
    return traces


def generate_adversarial_traces(graph: WorkflowGraph, n: int, seed: int = 7) -> list[Trace]:
    adversarial_config = ScenarioConfig(
        p_kb_stale=0.22,
        p_wrong_variant=0.18,
        p_ambiguous_query=0.28,
        clarifier_catch_rate=0.35,
        p_hallucination_transient=0.14,
        p_hallucination_sticky=0.12,
        evaluator_noise_std=0.30,
        p_chunk_entity_split=0.15,
        p_vade_issue=0.12,
        p_vade_catch=0.35,
    )
    return generate_traces(graph, n, config=adversarial_config, seed=seed)


def wrong_variant_scenario(graph: WorkflowGraph) -> Trace:
    order = [nid for nid in graph.topological_order() if nid != "human"]
    steps = []
    for node_id in order:
        spec = graph.nodes[node_id]
        symptoms = {}
        if node_id == "chunker":
            symptoms = {"context_coverage_score": 0.94, "entity_span_split": "False",
                        "chunk_coherence_score": 0.92}
        elif node_id == "kb_builder":
            symptoms = {"kb_age_days": 12.0}
        elif node_id == "retriever":
            symptoms = {"retrieval_top1_score": 0.81, "entity_match": "False",
                        "requested_variant": "Echo_Dot_Gen5", "returned_variant": "Echo_Dot_Gen4",
                        "variant_mismatch_suspected": "True"}
        elif node_id == "clarifier":
            symptoms = {"clarification_asked": "False", "query_ambiguous": "False",
                        "clarification_resolved": "False"}
        elif node_id == "generator":
            symptoms = {"groundedness": 0.44, "entity_match": "False", "hallucination_risk": "none"}
        elif node_id == "vade":
            symptoms = {"hallucination_present": "False", "vade_flagged": "False", "vade_confidence": 0.12}
        elif node_id == "evaluator":
            symptoms = {"final_score": 0.31, "evaluator_flags_low_score": "True"}
        steps.append(TraceStep(node_id=node_id, symptoms=symptoms,
                                latency_ms=spec.avg_latency_ms, tokens=int(spec.avg_tokens)))
    steps.append(TraceStep(node_id="human", symptoms={"escalated": "True"}, latency_ms=0, tokens=0))
    return Trace(trace_id="showcase_wrong_variant", steps=steps, final_outcome_failed=True,
                 ground_truth_node="retriever", ground_truth_step=order.index("retriever"),
                 scenario="wrong_product_variant_retrieved", failure_type="wrong_entity_variant",
                 evaluator_disagreement=None, content_failed=True, evaluation_failed=False,
                 contained=_contained(True, True), user_visible_failure=_user_visible_failure(True, True),
                 workflow_outcome=_outcome_label(True, True))
