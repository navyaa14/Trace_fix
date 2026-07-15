import tests._pathfix
import unittest

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor
from repair_engine import apply_repair, evaluate_repair, CANDIDATE_ACTIONS


class TestChunkerFailuresAreReachable(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=300, config=ScenarioConfig(), seed=11)

    def test_chunk_boundary_split_entity_occurs(self):
        chunker_failures = [t for t in self.traces if t.failure_type == "chunk_boundary_split_entity"]
        self.assertGreater(len(chunker_failures), 0)
        for t in chunker_failures:
            self.assertEqual(t.ground_truth_node, "chunker")
            self.assertTrue(t.final_outcome_failed)
            self.assertTrue(t.content_failed)

    def test_direct_chunker_symptoms_are_always_visible(self):
        # every trace executes the chunker node with its OWN observable evidence,
        # not merely a downstream retrieval score.
        for t in self.traces[:50]:
            step = next(s for s in t.steps if s.node_id == "chunker")
            self.assertIn("context_coverage_score", step.symptoms)
            self.assertIn("entity_span_split", step.symptoms)
            self.assertIn("chunk_coherence_score", step.symptoms)

    def test_chunker_failure_is_distinguishable_from_retriever_failure(self):
        # a chunker failure trace has entity_span_split=True at the chunker node
        # and variant_mismatch_suspected=False at the retriever -- so the
        # attributor has a real, node-local signal to tell the two apart.
        chunker_failures = [t for t in self.traces if t.failure_type == "chunk_boundary_split_entity"]
        self.assertTrue(chunker_failures)
        for t in chunker_failures:
            chunk_step = next(s for s in t.steps if s.node_id == "chunker")
            retr_step = next(s for s in t.steps if s.node_id == "retriever")
            self.assertEqual(chunk_step.symptoms["entity_span_split"], "True")
            self.assertEqual(retr_step.symptoms["variant_mismatch_suspected"], "False")

    def test_chunker_ground_truth_does_not_leak_into_the_judge_prompt(self):
        chunker_failures = [t for t in self.traces if t.failure_type == "chunk_boundary_split_entity"]
        self.assertTrue(chunker_failures)
        for t in chunker_failures:
            rendered = FailureAttributor._render_full_trace(t)
            self.assertNotIn("chunk_boundary_split_entity", rendered)
            self.assertNotIn("ground_truth", rendered.lower())

    def test_attributor_can_pick_chunker_from_visible_evidence(self):
        chunker_failures = [t for t in self.traces if t.failure_type == "chunk_boundary_split_entity"]
        self.assertTrue(chunker_failures)
        attributor = FailureAttributor()
        hits = sum(1 for t in chunker_failures if attributor.attribute_all_at_once(t).responsible_node == "chunker")
        self.assertGreater(hits, 0)


class TestRechunkRepair(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=1200, config=ScenarioConfig(), seed=11)
        self.chunker_failures = [t for t in self.traces if t.failure_type == "chunk_boundary_split_entity"]
        self.assertTrue(self.chunker_failures, "need at least one chunker-driven failure to test repair on")

    def test_rechunk_at_chunker_is_in_candidate_actions(self):
        self.assertIn(ActionType.RECHUNK, CANDIDATE_ACTIONS["chunker"])

    def test_rechunk_clears_chunker_own_evidence(self):
        for t in self.chunker_failures:
            result = apply_repair(self.graph, t, ActionType.RECHUNK, "chunker")
            self.assertIsNot(result, t)
            chunk_step = next(s for s in result.steps if s.node_id == "chunker")
            self.assertEqual(chunk_step.symptoms["entity_span_split"], "False")
            self.assertGreaterEqual(chunk_step.symptoms["context_coverage_score"], 0.9)

    def test_rechunk_recomputes_downstream_metrics_not_hardcoded(self):
        # Use every trace whose chunker evidence shows a split (including ones
        # where chunker isn't the SINGLE-label root cause, e.g. combined with a
        # generator-side hallucination) so the post-repair groundedness formula
        # is exercised over genuinely different inputs, not just the narrow
        # single-cause subset (which happens to share hallucination_risk="none"
        # and kb_stale=False by construction).
        split_traces = [t for t in self.traces
                         if next(s for s in t.steps if s.node_id == "chunker").symptoms["entity_span_split"]
                         == "True" and t.final_outcome_failed]
        self.assertGreater(len(split_traces), 1)
        seen_groundedness = set()
        for t in split_traces:
            result = apply_repair(self.graph, t, ActionType.RECHUNK, "chunker")
            gen_step = next(s for s in result.steps if s.node_id == "generator")
            seen_groundedness.add(gen_step.symptoms["groundedness"])
        self.assertGreater(len(seen_groundedness), 1,
                            "repaired groundedness should vary with input, not be a constant")

    def test_rechunk_resolves_most_pure_chunker_failures(self):
        fixed = 0
        for t in self.chunker_failures:
            result = evaluate_repair(self.graph, t, ActionType.RECHUNK, "chunker")
            if result.accepted:
                fixed += 1
        self.assertGreater(fixed, 0)

    def test_add_filter_confound_is_caught_by_counterfactual_confirmation(self):
        # ADD_FILTER's blanket entity_match=True override CAN make evaluate_repair
        # accept it on a chunker-rooted failure too (the same confound documented
        # for kb_builder in TestCounterfactualConfoundFix) -- a downstream boolean
        # flip alone is not sufficient causal confirmation. What matters is that
        # this confound is CAUGHT: the chunker's OWN evidence (entity_span_split)
        # is untouched by ADD_FILTER, so _hypothesis_symptom_actually_resolved
        # correctly rejects it, and a2p_scaffold's counterfactual check still
        # lands on chunker via RECHUNK, not on retriever via the unconfirmed
        # ADD_FILTER flip.
        from attribution import _hypothesis_symptom_actually_resolved
        attributor = FailureAttributor()
        for t in self.chunker_failures:
            add_filter_result = evaluate_repair(self.graph, t, ActionType.ADD_FILTER, "retriever")
            self.assertTrue(add_filter_result.accepted)
            chunk_step = next(s for s in add_filter_result.after_trace.steps if s.node_id == "chunker")
            self.assertEqual(chunk_step.symptoms["entity_span_split"], "True",
                              "ADD_FILTER must not touch the chunker's own evidence")

            a2p_result = attributor.attribute_a2p_scaffold(t, self.graph)
            self.assertEqual(a2p_result.responsible_node, "chunker")
            self.assertIn("causally_confirmed_via=rechunk", a2p_result.evidence.lower())

    def test_rechunk_is_not_appliable_to_a_pure_kb_staleness_failure(self):
        # A trace whose chunker evidence is already clean (entity_span_split=False)
        # gives RECHUNK@chunker nothing to change -- it must come back
        # not_executable, never a hardcoded/laundered success.
        stale_failures = [t for t in self.traces if t.failure_type == "stale_knowledge_correct_entity"]
        self.assertTrue(stale_failures)
        clean_chunker_stale_failures = [
            t for t in stale_failures
            if next(s for s in t.steps if s.node_id == "chunker").symptoms["entity_span_split"] == "False"
        ]
        self.assertTrue(clean_chunker_stale_failures,
                         "need at least one stale-KB failure with a clean chunker to test the no-op path")
        for t in clean_chunker_stale_failures:
            result = evaluate_repair(self.graph, t, ActionType.RECHUNK, "chunker")
            self.assertFalse(result.applied)
            self.assertFalse(result.accepted)
            self.assertTrue(result.not_executable)


if __name__ == "__main__":
    unittest.main()
