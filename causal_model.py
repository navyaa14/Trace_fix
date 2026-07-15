"""Single shared causal model for the support-pipeline simulation.

This module exists so that synthetic trace generation (simulate.py) and
counterfactual repair replay (repair_engine.py) can never drift apart on
what "groundedness", "content failure", "evaluation failure", or "workflow
failure" mean. Both call the *same* functions here.

Vocabulary:

    content_failed:
        the generated response is actually unacceptable (low groundedness,
        or an unresolved entity mismatch).

    evaluation_failed:
        the evaluator's verdict disagreed with the actual content state
        (either a false escalation of good content, or a false acceptance
        of bad content).

    workflow_failed:
        content_failed OR evaluation_failed. This is what
        `Trace.final_outcome_failed` measures end-to-end.

    contained / user_visible_failure:
        a further split of the outcome above. `contained` is true whenever
        bad content does not reach the user unfiltered (no defect at all,
        or the evaluator correctly flagged it). `user_visible_failure` is
        true only when a real content defect reaches the user unflagged
        (equivalently: an evaluator false acceptance). VAD/e and evaluator
        repairs cannot rewrite content, so their honest job is reducing
        user_visible_failure, not necessarily resolving content_failed.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def retrieval_score_mean(kb_stale: bool, entity_span_split: bool = False,
                          context_coverage_score: float = 1.0) -> float:
    """Mean (pre-noise, pre-sigmoid) retrieval quality.

    Two INDEPENDENT degradation paths feed into retrieval quality:
      - kb_stale: the knowledge base content itself is out of date
        (kb_builder's responsibility).
      - entity_span_split (chunker): the chunk boundary cut an entity
        mention in half, so even a fresh, correctly-variant KB can't be
        retrieved cleanly. This is a distinct causal path from staleness --
        a chunker failure can occur on a perfectly fresh KB, and a stale KB
        failure can occur with perfect chunking.
    """
    base = 2.0 - (1.8 if kb_stale else 0.0)
    coverage_penalty = (1.0 - context_coverage_score) * 1.6 if entity_span_split else 0.0
    return base - coverage_penalty


def expected_retrieval_score(kb_stale: bool, entity_span_split: bool = False,
                              context_coverage_score: float = 1.0) -> float:
    """Deterministic (noise-free) retrieval score -- used for repair replay,
    where we want the expected post-repair value, not a fresh random draw."""
    return sigmoid(retrieval_score_mean(kb_stale, entity_span_split, context_coverage_score))


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------

def groundedness_mean(retrieval_score: float, entity_match: bool,
                       hallucination_penalty: float = 0.0) -> float:
    return 2.2 * retrieval_score + (1.0 if entity_match else -0.5) - hallucination_penalty


def expected_groundedness(retrieval_score: float, entity_match: bool,
                           hallucination_penalty: float = 0.0) -> float:
    return sigmoid(groundedness_mean(retrieval_score, entity_match, hallucination_penalty))


# ---------------------------------------------------------------------------
# Content failure
# ---------------------------------------------------------------------------

def outcome_failed(groundedness: float, entity_match: bool, clarified: bool) -> bool:
    """content_failed: is the generated response itself unacceptable?"""
    return (groundedness < 0.5) or (not entity_match and not clarified)


# ---------------------------------------------------------------------------
# Evaluator / evaluation failure
# ---------------------------------------------------------------------------

def evaluation_failed(evaluator_disagreement: Optional[str]) -> bool:
    """evaluation_failed: did the evaluator's verdict disagree with the
    actual content state, in either direction?

      - "false_positive": evaluator flagged/escalated GOOD content
        -> evaluator_false_escalation
      - "false_negative": evaluator accepted BAD content
        -> evaluator_false_acceptance

    Both are legitimate evaluator-rooted failures of the evaluation layer,
    independent of whether the underlying content itself was good or bad.
    """
    return evaluator_disagreement in ("false_positive", "false_negative")


def workflow_failed(content_failed: bool, eval_failed: bool) -> bool:
    """workflow_failed = content_failed OR evaluation_failed.

    This is what Trace.final_outcome_failed measures end-to-end.
    """
    return content_failed or eval_failed


# ---------------------------------------------------------------------------
# Containment: separates "is the content itself defective" from "did the
# defect actually reach the user unfiltered". A VAD/e or evaluator repair
# can legitimately succeed by fixing the SECOND thing without ever touching
# the first -- neither node can rewrite what the generator already produced.
# ---------------------------------------------------------------------------

def contained(content_failed: bool, evaluator_flags_low_score: bool) -> bool:
    """True whenever bad content does not reach the user unfiltered: either
    there was no content defect to begin with, or the evaluator correctly
    flagged/escalated it."""
    return (not content_failed) or evaluator_flags_low_score


def user_visible_failure(content_failed: bool, evaluator_flags_low_score: bool) -> bool:
    """True only when a real content defect reaches the user unflagged --
    the one outcome VAD/e and evaluator repairs actually exist to reduce.
    Equivalent to an evaluator false acceptance."""
    return content_failed and not evaluator_flags_low_score


def outcome_label(content_failed: bool, evaluator_flags_low_score: bool) -> str:
    """Four-way categorical summary used by the dashboard/report layer."""
    if not content_failed and not evaluator_flags_low_score:
        return "clean"
    if not content_failed and evaluator_flags_low_score:
        return "false_escalation"
    if content_failed and evaluator_flags_low_score:
        return "contained"
    return "user_visible_failure"


# ---------------------------------------------------------------------------
# Validator (vade) — detection vs. generation
# ---------------------------------------------------------------------------

def vade_catches_hallucination(hallucination_risk: str, vade_flagged: bool) -> bool:
    """Only TRANSIENT hallucination is a genuinely catchable/detection-layer
    problem in this model -- vade successfully flagging it lets a corrective
    pass clear the penalty before it ever reaches the evaluator.

    STICKY hallucination is modeled as structural (the generator is
    confidently, repeatedly wrong), so vade catching it changes what the
    *evaluator* sees (vade_flagged=True gets recorded as evidence) but does
    NOT by itself clear the underlying generation defect -- this keeps
    vade's failure mode distinct from the generator's own hallucination
    mechanism.
    """
    return hallucination_risk == "transient" and vade_flagged


@dataclass
class PipelineState:
    retrieval_score: float
    entity_match: bool
    groundedness: float
    content_failed: bool
    evaluation_failed: bool
    workflow_failed: bool


def recompute_pipeline_state(
    *,
    kb_stale: bool,
    entity_match: bool,
    clarified: bool,
    hallucination_risk: str,
    entity_span_split: bool = False,
    context_coverage_score: float = 1.0,
    vade_flagged: bool = False,
    hallucination_penalty_base: float = 3.2,
    evaluator_disagreement: Optional[str] = None,
    retrieval_score_override: Optional[float] = None,
) -> PipelineState:
    """The ONE shared recomputation path.

    Both synthetic trace generation (for the deterministic/no-noise showcase
    scenarios and for repair-replay-equivalent recomputation) and the repair
    engine (repair_engine.apply_repair) route their after-state through this
    function, so the causal formulas for retrieval quality, groundedness,
    content failure, evaluation failure, and workflow failure can never
    silently drift apart between the simulator and the repair replay path.
    """
    retrieval_score = (retrieval_score_override if retrieval_score_override is not None
                        else expected_retrieval_score(kb_stale, entity_span_split, context_coverage_score))

    penalty = hallucination_penalty_base if hallucination_risk in ("sticky", "transient") else 0.0
    if vade_catches_hallucination(hallucination_risk, vade_flagged):
        penalty = 0.0

    groundedness = expected_groundedness(retrieval_score, entity_match, penalty)
    content_failed = outcome_failed(groundedness, entity_match, clarified)
    eval_failed = evaluation_failed(evaluator_disagreement)
    wf_failed = workflow_failed(content_failed, eval_failed)

    return PipelineState(
        retrieval_score=retrieval_score,
        entity_match=entity_match,
        groundedness=groundedness,
        content_failed=content_failed,
        evaluation_failed=eval_failed,
        workflow_failed=wf_failed,
    )
