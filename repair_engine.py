
from __future__ import annotations
from dataclasses import dataclass, field

from graph import WorkflowGraph, ActionType
from attribution import Trace, TraceStep
from cost import trace_cost, CostBreakdown
from causal_model import (
    expected_retrieval_score,
    expected_groundedness,
    outcome_failed,
    evaluation_failed,
    workflow_failed,
    contained as _contained,
    user_visible_failure as _user_visible_failure,
    outcome_label as _outcome_label,
)

APPLIABLE_ACTIONS = {
    ActionType.ADD_FILTER,
    ActionType.RECHUNK,
    ActionType.ASK_CLARIFICATION,
    ActionType.RETRY,
    ActionType.RETRY_VALIDATION,
    ActionType.LOWER_DETECTION_THRESHOLD,
    ActionType.SECOND_JUDGE,
    ActionType.RECALIBRATE_THRESHOLD,
}

CANDIDATE_ACTIONS: dict[str, list[ActionType]] = {
    "chunker": [ActionType.RECHUNK, ActionType.HUMAN_REVIEW],
    "retriever": [ActionType.ADD_FILTER, ActionType.RETRY, ActionType.HUMAN_REVIEW],
    "kb_builder": [ActionType.RECHUNK, ActionType.RETRY, ActionType.HUMAN_REVIEW],
    "clarifier": [ActionType.ASK_CLARIFICATION, ActionType.RETRY, ActionType.HUMAN_REVIEW],
    "generator": [ActionType.RETRY, ActionType.HUMAN_REVIEW],
    "vade": [ActionType.RETRY_VALIDATION, ActionType.LOWER_DETECTION_THRESHOLD, ActionType.HUMAN_REVIEW],
    "evaluator": [ActionType.SECOND_JUDGE, ActionType.RECALIBRATE_THRESHOLD, ActionType.HUMAN_REVIEW],
}

_STALE_KB_THRESHOLD_DAYS = 60


def _get_step(trace: Trace, node_id: str) -> TraceStep | None:
    return next((s for s in trace.steps if s.node_id == node_id), None)


def _bool(v) -> bool:
    return str(v) == "True"


@dataclass
class ValidatedRepair:
    trace_id: str
    node_id: str
    action: ActionType
    applied: bool
    before_failed: bool
    after_failed: bool
    before_cost: CostBreakdown
    after_cost: CostBreakdown
    accepted: bool
    reason: str
    failure_type: str | None = None
    not_executable: bool = False
    after_trace: Trace | None = None


def apply_repair(graph: WorkflowGraph, trace: Trace, action: ActionType, node_id: str) -> Trace:
    """Apply a node-specific repair action and recompute the ENTIRE downstream
    pipeline state through the single shared causal model (causal_model.py) --
    the same formulas simulate.generate_traces uses to produce the trace in the
    first place. No outcome is ever hardcoded here; every after-state field is
    derived from the (possibly repaired) upstream symptoms.
    """
    chunker = _get_step(trace, "chunker")
    retriever = _get_step(trace, "retriever")
    generator = _get_step(trace, "generator")
    kb_step = _get_step(trace, "kb_builder")
    clarifier = _get_step(trace, "clarifier")
    vade = _get_step(trace, "vade")
    evaluator = _get_step(trace, "evaluator")
    if not (retriever and generator):
        return trace

    retrieval_score = float(retriever.symptoms.get("retrieval_top1_score", 0.7))
    entity_match = _bool(retriever.symptoms.get("entity_match", "False"))
    orig_retrieval_score = retrieval_score
    orig_entity_match = entity_match
    kb_age_days = float(kb_step.symptoms.get("kb_age_days", 10.0)) if kb_step else 10.0
    kb_stale = kb_age_days > _STALE_KB_THRESHOLD_DAYS
    clarified = _bool(clarifier.symptoms.get("clarification_asked", "False")) if clarifier else False
    hallucination_risk = generator.symptoms.get("hallucination_risk", "none")

    entity_span_split = _bool(chunker.symptoms.get("entity_span_split", "False")) if chunker else False
    context_coverage_score = float(chunker.symptoms.get("context_coverage_score", 1.0)) if chunker else 1.0
    chunk_coherence_score = float(chunker.symptoms.get("chunk_coherence_score", 1.0)) if chunker else 1.0

    vade_hallucination_present = _bool(vade.symptoms.get("hallucination_present", "False")) if vade else False
    vade_flagged = _bool(vade.symptoms.get("vade_flagged", "False")) if vade else False
    vade_flagged_original = vade_flagged

    evaluator_flagged_override: bool | None = None

    extra_latency = {"chunker": 0.0, "retriever": 0.0, "kb_builder": 0.0, "clarifier": 0.0, "vade": 0.0,
                      "evaluator": 0.0}
    extra_tokens = {"chunker": 0, "retriever": 0, "kb_builder": 0, "clarifier": 0, "vade": 0, "evaluator": 0}
    filter_tag = None
    recompute_retrieval_from_scratch = False

    clears_hallucination = (action == ActionType.RETRY and node_id == "generator"
                             and hallucination_risk == "transient")

    if action == ActionType.ADD_FILTER and node_id == "retriever":
        entity_match = True
        retrieval_score = min(1.0, retrieval_score + 0.05)
        extra_latency["retriever"] = 15.0
        extra_tokens["retriever"] = 40
        filter_tag = "entity_variant"

    elif action == ActionType.RECHUNK and node_id == "kb_builder":
        kb_age_days = 5.0
        kb_stale = False
        recompute_retrieval_from_scratch = True
        extra_latency["kb_builder"] = 400.0
        extra_tokens["kb_builder"] = 900

    elif action == ActionType.RECHUNK and node_id == "chunker":
        # Rebuild the chunk boundaries: fixes the chunker's OWN evidence
        # (entity_span_split / context_coverage_score / chunk_coherence_score),
        # not kb_age_days -- a genuinely stale KB is untouched by re-chunking.
        # If entity_span_split was already False, there is nothing for this
        # repair to change -- treat it as not appliable to this trace shape
        # rather than recomputing a noise-free retrieval score that could
        # accidentally "resolve" a failure this repair never actually touched
        # (e.g. a pure kb_builder staleness failure).
        if not entity_span_split:
            return trace
        entity_span_split = False
        context_coverage_score = max(context_coverage_score, 0.92)
        chunk_coherence_score = max(chunk_coherence_score, 0.90)
        recompute_retrieval_from_scratch = True
        extra_latency["chunker"] = 220.0
        extra_tokens["chunker"] = 500

    elif action == ActionType.ASK_CLARIFICATION and node_id == "clarifier":
        clarified = True
        entity_match = True
        retrieval_score = min(1.0, retrieval_score + 0.25)
        extra_latency["clarifier"] = 250.0
        extra_tokens["clarifier"] = 400

    elif action == ActionType.RETRY and node_id in ("generator", "retriever", "kb_builder", "clarifier"):
        if node_id != "generator":
            return trace
        if hallucination_risk != "transient":
            # nothing for a plain regeneration retry to fix here -- not
            # appliable to this trace shape. Falling through to a full
            # recompute with unchanged inputs would only ever "resolve" such
            # traces by discarding the original noisy draw for a deterministic
            # mean, which is not something this action actually did.
            return trace

    elif action in (ActionType.RETRY_VALIDATION, ActionType.LOWER_DETECTION_THRESHOLD) and node_id == "vade":
        if vade is None or not vade_hallucination_present or vade_flagged:
            # nothing for the validator to catch, or it already caught it --
            # not appliable to this trace shape.
            return trace
        if hallucination_risk != "none":
            # this hallucination is the generator's own (sticky/transient),
            # structural mechanism -- re-running the validator does not by
            # itself rewrite what the generator produced.
            return trace
        # VAD/e detects and routes; it cannot rewrite what the generator
        # already produced. So the fix here is CONTAINMENT-only: vade now
        # flags the still-defective content, which in turn makes the
        # evaluator correctly reject/escalate it below (evaluator_flagged_
        # override=True, groundedness left untouched). It is not treated as
        # having "cleared" any generator-side penalty.
        vade_flagged = True
        evaluator_flagged_override = True
        extra_latency["vade"] = 120.0 if action == ActionType.RETRY_VALIDATION else 20.0
        extra_tokens["vade"] = 250 if action == ActionType.RETRY_VALIDATION else 30

    elif action in (ActionType.SECOND_JUDGE, ActionType.RECALIBRATE_THRESHOLD) and node_id == "evaluator":
        if evaluator is None:
            return trace
        currently_flagged = _bool(evaluator.symptoms.get("evaluator_flags_low_score", "False"))
        content_is_bad = outcome_failed(
            float(generator.symptoms.get("groundedness", 0.0)),
            _bool(generator.symptoms.get("entity_match", "True")),
            clarified,
        )
        if currently_flagged == content_is_bad:
            # verdict already matches the content -- nothing for a second
            # opinion (or a recalibrated threshold) to change.
            return trace
        # Either direction is a genuine evaluator-only repair: a false
        # escalation flips True->False, a false acceptance flips False->True.
        # Both change ONLY the verdict/routing -- never the content itself.
        evaluator_flagged_override = content_is_bad
        extra_latency["evaluator"] = 300.0 if action == ActionType.SECOND_JUDGE else 30.0
        extra_tokens["evaluator"] = 500 if action == ActionType.SECOND_JUDGE else 20

    else:
        return trace

    if recompute_retrieval_from_scratch:
        retrieval_score = expected_retrieval_score(kb_stale, entity_span_split, context_coverage_score)
        entity_match = entity_match or (retrieval_score > 0.5)

    halluc_penalty = 3.2 if hallucination_risk in ("sticky", "transient") else 0.0
    if clears_hallucination:
        halluc_penalty = 0.0

    # The vade-borne penalty only ever exists on traces where the generator's
    # OWN hallucination_risk is "none" (see simulate.py: vade_issue_present and
    # hallucination_risk are mutually exclusive by construction) -- so this
    # never double-counts with the generator's own mechanism above.
    # Penalty uses the PRE-repair vade_flagged value on purpose: vade catching
    # a hallucination is containment (routing), not correction, so even the
    # vade-repair branch below (which mutates vade_flagged=True for display /
    # evaluator routing) must not silently zero its own penalty here.
    vade_penalty = 3.0 if (vade_hallucination_present and not vade_flagged_original
                            and hallucination_risk == "none") else 0.0

    evaluator_only_repair = (node_id == "evaluator"
                             and action in (ActionType.SECOND_JUDGE, ActionType.RECALIBRATE_THRESHOLD))
    vade_only_repair = (node_id == "vade"
                        and action in (ActionType.RETRY_VALIDATION, ActionType.LOWER_DETECTION_THRESHOLD))

    orig_halluc_penalty = 3.2 if hallucination_risk in ("sticky", "transient") else 0.0
    orig_total_penalty = orig_halluc_penalty + vade_penalty  # vade_penalty already uses the pre-repair flag
    inputs_unchanged = (retrieval_score == orig_retrieval_score and entity_match == orig_entity_match
                         and (halluc_penalty + vade_penalty) == orig_total_penalty)

    if evaluator_only_repair or vade_only_repair or inputs_unchanged:
        # No causal input to groundedness (retrieval_score, entity_match, the
        # hallucination/vade penalty) actually changed -- reuse the EXACT
        # original groundedness rather than recomputing through the
        # (noise-free) formula, which would otherwise shift the numeric value
        # by a "denoising" artifact alone (deterministic mean vs. the
        # original noisy draw), silently "resolving" a failure this action
        # never actually touched.
        new_groundedness = float(generator.symptoms.get("groundedness", 0.0))
    else:
        new_groundedness = expected_groundedness(retrieval_score, entity_match, halluc_penalty + vade_penalty)

    new_steps = []
    for step in trace.steps:
        if step.node_id == "chunker":
            sym = dict(step.symptoms)
            sym["entity_span_split"] = str(entity_span_split)
            sym["context_coverage_score"] = round(context_coverage_score, 2)
            sym["chunk_coherence_score"] = round(chunk_coherence_score, 2)
            new_steps.append(TraceStep(step.node_id, sym,
                                        step.latency_ms + extra_latency["chunker"],
                                        step.tokens + extra_tokens["chunker"]))
        elif step.node_id == "retriever":
            sym = dict(step.symptoms)
            sym["retrieval_top1_score"] = round(retrieval_score, 2)
            sym["entity_match"] = str(entity_match)
            if filter_tag:
                sym["filter_applied"] = filter_tag
            new_steps.append(TraceStep(step.node_id, sym,
                                        step.latency_ms + extra_latency["retriever"],
                                        step.tokens + extra_tokens["retriever"]))
        elif step.node_id == "kb_builder":
            sym = dict(step.symptoms)
            sym["kb_age_days"] = round(kb_age_days, 1)
            new_steps.append(TraceStep(step.node_id, sym,
                                        step.latency_ms + extra_latency["kb_builder"],
                                        step.tokens + extra_tokens["kb_builder"]))
        elif step.node_id == "clarifier":
            sym = dict(step.symptoms)
            sym["clarification_asked"] = str(clarified)
            new_steps.append(TraceStep(step.node_id, sym,
                                        step.latency_ms + extra_latency["clarifier"],
                                        step.tokens + extra_tokens["clarifier"]))
        elif step.node_id == "generator":
            sym = dict(step.symptoms)
            sym["entity_match"] = str(entity_match)
            sym["groundedness"] = round(new_groundedness, 2)
            if clears_hallucination:
                sym["hallucination_risk"] = "cleared_by_retry"
            new_steps.append(TraceStep(step.node_id, sym, step.latency_ms, step.tokens))
        elif step.node_id == "vade":
            sym = dict(step.symptoms)
            sym["vade_flagged"] = str(vade_flagged)
            new_steps.append(TraceStep(step.node_id, sym,
                                        step.latency_ms + extra_latency["vade"],
                                        step.tokens + extra_tokens["vade"]))
        elif step.node_id == "evaluator":
            sym = dict(step.symptoms)
            if evaluator_flagged_override is not None:
                # Evaluator-only repair: the verdict changes, the content does
                # NOT -- final_score is left untouched here on purpose so this
                # never silently improves groundedness.
                sym["evaluator_flags_low_score"] = str(evaluator_flagged_override)
            else:
                sym["final_score"] = round(new_groundedness, 2)
                sym["evaluator_flags_low_score"] = str(new_groundedness < 0.5)
            new_steps.append(TraceStep(step.node_id, sym, step.latency_ms, step.tokens))
        elif step.node_id == "human":
            pass  # recomputed below, appended only if still failing
        else:
            new_steps.append(step)

    content_failed = outcome_failed(new_groundedness, entity_match, clarified)
    if evaluator_flagged_override is not None:
        evaluator_flags_low_score = evaluator_flagged_override
    else:
        evaluator_flags_low_score = new_groundedness < 0.5
    evaluator_disagreement = None
    if evaluator_flags_low_score and not content_failed:
        evaluator_disagreement = "false_positive"
    elif (not evaluator_flags_low_score) and content_failed:
        evaluator_disagreement = "false_negative"
    eval_failed = evaluation_failed(evaluator_disagreement)
    after_failed = workflow_failed(content_failed, eval_failed)

    if after_failed:
        new_steps.append(TraceStep(node_id="human", symptoms={"escalated": "True"}, latency_ms=0, tokens=0))

    return Trace(trace_id=trace.trace_id + "_repaired", steps=new_steps,
                 final_outcome_failed=after_failed, ground_truth_node=None,
                 scenario=trace.scenario + "_repaired", failure_type=trace.failure_type,
                 content_failed=content_failed, evaluation_failed=eval_failed,
                 evaluator_disagreement=evaluator_disagreement,
                 contained=_contained(content_failed, evaluator_flags_low_score),
                 user_visible_failure=_user_visible_failure(content_failed, evaluator_flags_low_score),
                 workflow_outcome=_outcome_label(content_failed, evaluator_flags_low_score))


def evaluate_repair(graph: WorkflowGraph, trace: Trace, action: ActionType, node_id: str,
                     cost_tolerance_usd: float = 0.05,
                     failure_type_override: str | None = None) -> ValidatedRepair:
    before_cost = trace_cost(trace)
    failure_type = failure_type_override if failure_type_override is not None else trace.failure_type

    if not trace.final_outcome_failed:
        return ValidatedRepair(trace.trace_id, node_id, action, False,
                                False, False, before_cost, before_cost, False,
                                "trace was not failing; no repair needed", failure_type)

    if action not in APPLIABLE_ACTIONS:
        return ValidatedRepair(trace.trace_id, node_id, action, False,
                                True, True, before_cost, before_cost, False,
                                f"{action.value} has no auto-apply implementation in this engine "
                                f"-- NOT_EXECUTABLE", failure_type, not_executable=True)

    after_trace = apply_repair(graph, trace, action, node_id)
    if after_trace is trace:
        return ValidatedRepair(trace.trace_id, node_id, action, False,
                                True, True, before_cost, before_cost, False,
                                f"{action.value} is not appliable to node '{node_id}' for this trace shape "
                                f"-- NOT_EXECUTABLE", failure_type, not_executable=True)

    after_cost = trace_cost(after_trace)
    after_failed = after_trace.final_outcome_failed

    # vade and the evaluator cannot rewrite what the generator produced --
    # their honest success criterion is resolving USER-VISIBLE failure
    # (a real content defect reaching the user unflagged), not full content
    # resolution. A repair here can legitimately succeed while
    # final_outcome_failed stays True (content_failed persists, correctly
    # routed to human instead).
    containment_repair = node_id in ("vade", "evaluator")
    before_uvf = trace.user_visible_failure if trace.user_visible_failure is not None \
        else _user_visible_failure(bool(trace.content_failed), not bool(trace.content_failed))
    after_uvf = after_trace.user_visible_failure
    resolved = (not after_failed) or (containment_repair and before_uvf and after_uvf is False)

    if not resolved:
        return ValidatedRepair(trace.trace_id, node_id, action, True,
                                True, True, before_cost, after_cost, False,
                                "repair applied but failure persisted after recomputation", failure_type,
                                after_trace=after_trace)

    if after_failed and containment_repair:
        # content_failed persists (this repair never touches content) but the
        # user-visible failure it targets is gone -- an honest, distinct
        # accept reason from full resolution.
        cost_delta = after_cost.api_cost_usd - before_cost.api_cost_usd
        if cost_delta > cost_tolerance_usd:
            return ValidatedRepair(trace.trace_id, node_id, action, True,
                                    True, True, before_cost, after_cost, False,
                                    f"content defect persists and user-visible failure was contained, "
                                    f"but api cost rose ${cost_delta:.4f}, over tolerance "
                                    f"${cost_tolerance_usd:.2f}", failure_type,
                                    after_trace=after_trace)
        return ValidatedRepair(trace.trace_id, node_id, action, True,
                                True, True, before_cost, after_cost, True,
                                "content defect persists (not this node's responsibility) but "
                                "user-visible failure resolved: bad content now correctly "
                                "blocked/routed instead of reaching the user", failure_type,
                                after_trace=after_trace)

    cost_delta = after_cost.api_cost_usd - before_cost.api_cost_usd
    if cost_delta > cost_tolerance_usd:
        return ValidatedRepair(trace.trace_id, node_id, action, True,
                                True, False, before_cost, after_cost, False,
                                f"failure resolved but api cost rose ${cost_delta:.4f}, "
                                f"over tolerance ${cost_tolerance_usd:.2f}", failure_type,
                                after_trace=after_trace)

    return ValidatedRepair(trace.trace_id, node_id, action, True,
                            True, False, before_cost, after_cost, True,
                            "failure resolved; cost within tolerance", failure_type,
                            after_trace=after_trace)


def generate_and_evaluate_candidates(graph: WorkflowGraph, trace: Trace, node_id: str,
                                      candidate_actions: list[ActionType],
                                      cost_tolerance_usd: float = 0.05,
                                      failure_type_override: str | None = None) -> list[ValidatedRepair]:
    return [evaluate_repair(graph, trace, action, node_id, cost_tolerance_usd, failure_type_override)
            for action in candidate_actions]


def select_best_candidate(results: list[ValidatedRepair]) -> ValidatedRepair | None:
    accepted = [r for r in results if r.accepted]
    if not accepted:
        return None
    return min(accepted, key=lambda r: r.after_cost.api_cost_usd + r.after_cost.human_cost_usd)
