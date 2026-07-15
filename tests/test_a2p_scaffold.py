
import tests._pathfix
import unittest

from graph import build_support_pipeline, ActionType
from attribution import FailureAttributor, Trace, TraceStep
from simulate import generate_traces, ScenarioConfig
from demo import accuracy, accuracy_by_failure_type


class TestA2PInterfaceCompatibility(unittest.TestCase):

    def setUp(self):
        self.graph = build_support_pipeline()
        self.attributor = FailureAttributor()
        self.traces = generate_traces(self.graph, n=50, seed=7)

    def test_returns_attribution_result_with_method_name_set(self):
        failed = [t for t in self.traces if t.final_outcome_failed]
        self.assertTrue(failed)
        result = self.attributor.attribute_a2p_scaffold(failed[0], self.graph)
        self.assertEqual(result.method, "a2p_scaffold")
        self.assertEqual(result.trace_id, failed[0].trace_id)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.judge_calls_used, 1)

    def test_judge_calls_used_stays_at_one_like_all_at_once(self):
        failed = [t for t in self.traces if t.final_outcome_failed]
        for t in failed[:10]:
            result = self.attributor.attribute_a2p_scaffold(t, self.graph)
            self.assertEqual(result.judge_calls_used, 1)

    def test_works_without_a_graph_degrading_to_abduction_only(self):
        failed = [t for t in self.traces if t.final_outcome_failed]
        result = self.attributor.attribute_a2p_scaffold(failed[0], graph=None)
        self.assertEqual(result.method, "a2p_scaffold")
        self.assertIn("counterfactual_unavailable_no_graph", result.evidence)


class TestNoGroundTruthLeakage(unittest.TestCase):

    def test_a2p_never_uses_ground_truth_node_to_pick_its_answer(self):
        graph = build_support_pipeline()
        attributor = FailureAttributor()
        traces = generate_traces(graph, n=100, seed=7)
        failed = [t for t in traces if t.final_outcome_failed and t.ground_truth_node]
        for t in failed[:15]:
            rendered = FailureAttributor._render_full_trace(t)
            self.assertNotIn("ground_truth", rendered.lower())
            self.assertNotIn("root_cause", rendered.lower())
            self.assertNotIn("fail_signal", rendered.lower())
            self.assertNotIn("responsible=", rendered.lower())


class TestA2PRealBatchMeasurement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_support_pipeline()
        cls.traces = generate_traces(cls.graph, n=300, config=ScenarioConfig(), seed=11)
        cls.attributor = FailureAttributor()
        cls.a2p_fn = staticmethod(lambda t: cls.attributor.attribute_a2p_scaffold(t, cls.graph))

    def test_a2p_accuracy_matches_pinned_value(self):
        # chunker/vade/evaluator are reachable, confirmable root causes here, so
        # a2p's counterfactual confirmation (RECHUNK@chunker, RETRY_VALIDATION@vade,
        # SECOND_JUDGE@evaluator) applies across all evaluable failure types.
        acc, avg_calls, n_eval = accuracy(self.a2p_fn, self.traces)
        self.assertEqual(n_eval, 90)
        self.assertAlmostEqual(acc, 0.7333, places=3)
        self.assertAlmostEqual(avg_calls, 1.00, places=2)

    def test_a2p_beats_all_at_once_on_this_batch(self):
        acc_a2p, _, _ = accuracy(self.a2p_fn, self.traces)
        acc_all, _, _ = accuracy(self.attributor.attribute_all_at_once, self.traces)
        self.assertGreater(acc_a2p, acc_all)

    def test_a2p_now_beats_binary_search_on_this_batch(self):
        # This flips from the pre-node-mechanism revision: a2p_scaffold's
        # counterfactual confirmation is now strong on chunk_boundary_split_entity
        # (1.00 vs. binary_search's 1.00 -- tied) and especially on
        # vade_missed_hallucination (0.60 vs. binary_search's 0.00 -- binary_search
        # has no rule that ever lands on the vade node from a prefix scan, while
        # a2p's counterfactual RETRY_VALIDATION@vade check confirms it directly).
        # That is a real, measured algorithmic win from giving vade a genuine,
        # confirmable repair -- not a re-pinned cosmetic change.
        acc_a2p, _, _ = accuracy(self.a2p_fn, self.traces)
        acc_bs, _, _ = accuracy(self.attributor.attribute_binary_search, self.traces)
        self.assertGreater(acc_a2p, acc_bs)

    def test_ambiguous_query_unclarified_no_longer_needs_a2p_to_be_caught(self):
        # The "query_ambiguous=True, clarification_asked=False" rule lives in the
        # shared _rank_symptom_evidence used by all_at_once/step_by_step/
        # binary_search/a2p_scaffold alike, so a2p has no exclusive edge on this
        # failure type -- it should be tied with all_at_once here, not ahead of it.
        breakdown_a2p = accuracy_by_failure_type(self.a2p_fn, self.traces)
        breakdown_all = accuracy_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        ft = "ambiguous_query_unclarified"
        self.assertIn(ft, breakdown_a2p)
        acc_a2p, n = breakdown_a2p[ft]
        acc_all, _ = breakdown_all[ft]
        self.assertGreater(acc_all, 0.0)
        self.assertAlmostEqual(acc_a2p, acc_all, places=6)

    def test_multiple_simultaneous_failures_improves_but_stays_far_below_binary_search(self):
        # Re-measured: with the new evidence rules competing for the highest score
        # across a trace, all_at_once's single-label floor on this category is no
        # longer a hard 0.0% (it occasionally lands on the right node by accident:
        # 2/16 = 12.5%), but a2p's counterfactual confirmation still does
        # meaningfully better (25%) while binary_search, which gets to look at
        # growing prefixes of the SAME trace multiple times, remains well ahead (75%).
        breakdown_a2p = accuracy_by_failure_type(self.a2p_fn, self.traces)
        breakdown_all = accuracy_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        breakdown_bs = accuracy_by_failure_type(self.attributor.attribute_binary_search, self.traces)
        ft = "multiple_simultaneous_failures"
        acc_a2p, _ = breakdown_a2p[ft]
        acc_all, _ = breakdown_all[ft]
        acc_bs, _ = breakdown_bs[ft]
        self.assertLess(acc_all, 0.2)
        self.assertGreaterEqual(acc_a2p, acc_all)
        self.assertLess(acc_a2p, acc_bs)


class TestCounterfactualConfirmationCanBeFooled(unittest.TestCase):

    def test_add_filter_can_resolve_a_kb_staleness_failure_it_did_not_cause(self):
        from repair_engine import evaluate_repair

        graph = build_support_pipeline()
        steps = [
            TraceStep(node_id="kb_builder", symptoms={"kb_age_days": 150.0}, latency_ms=50, tokens=100),
            TraceStep(node_id="retriever", symptoms={"retrieval_top1_score": 0.22, "entity_match": "False",
                                                       "variant_mismatch_suspected": "False"},
                       latency_ms=80, tokens=150),
            TraceStep(node_id="clarifier", symptoms={"clarification_asked": "False", "query_ambiguous": "False"},
                       latency_ms=10, tokens=5),
            TraceStep(node_id="generator", symptoms={"groundedness": 0.15, "entity_match": "False",
                                                       "hallucination_risk": "none"},
                       latency_ms=200, tokens=300),
            TraceStep(node_id="evaluator", symptoms={"final_score": 0.15}, latency_ms=10, tokens=5),
            TraceStep(node_id="human", symptoms={"escalated": "True"}, latency_ms=0, tokens=0),
        ]
        trace = Trace(trace_id="kb_stale_demo", steps=steps, final_outcome_failed=True,
                       ground_truth_node="kb_builder", failure_type="stale_knowledge_correct_entity")

        add_filter_result = evaluate_repair(graph, trace, ActionType.ADD_FILTER, "retriever")
        self.assertTrue(add_filter_result.applied)
        self.assertTrue(add_filter_result.accepted)
        self.assertFalse(add_filter_result.after_failed)


class TestCounterfactualConfoundFix(unittest.TestCase):

    def test_a2p_scaffold_attributes_to_kb_builder_not_retriever(self):
        graph = build_support_pipeline()
        attributor = FailureAttributor()
        steps = [
            TraceStep(node_id="kb_builder", symptoms={"kb_age_days": 150.0}, latency_ms=50, tokens=100),
            TraceStep(node_id="retriever", symptoms={"retrieval_top1_score": 0.22, "entity_match": "False",
                                                       "variant_mismatch_suspected": "False"},
                       latency_ms=80, tokens=150),
            TraceStep(node_id="clarifier", symptoms={"clarification_asked": "False", "query_ambiguous": "False"},
                       latency_ms=10, tokens=5),
            TraceStep(node_id="generator", symptoms={"groundedness": 0.15, "entity_match": "False",
                                                       "hallucination_risk": "none"},
                       latency_ms=200, tokens=300),
            TraceStep(node_id="evaluator", symptoms={"final_score": 0.15}, latency_ms=10, tokens=5),
            TraceStep(node_id="human", symptoms={"escalated": "True"}, latency_ms=0, tokens=0),
        ]
        trace = Trace(trace_id="kb_stale_demo", steps=steps, final_outcome_failed=True,
                       ground_truth_node="kb_builder", failure_type="stale_knowledge_correct_entity")

        result = attributor.attribute_a2p_scaffold(trace, graph)

        self.assertEqual(result.responsible_node, "kb_builder")
        self.assertIn("causally_confirmed_via=rechunk", result.evidence.lower())
        self.assertGreaterEqual(int(result.evidence.split("counterfactual_checks=")[1]), 2)

    def test_pre_fix_reproduction_the_raw_symptom_check_alone_would_have_rejected_add_filter(self):
        from attribution import _hypothesis_symptom_actually_resolved
        from repair_engine import evaluate_repair

        graph = build_support_pipeline()
        steps = [
            TraceStep(node_id="kb_builder", symptoms={"kb_age_days": 150.0}, latency_ms=50, tokens=100),
            TraceStep(node_id="retriever", symptoms={"retrieval_top1_score": 0.22, "entity_match": "False"},
                       latency_ms=80, tokens=150),
            TraceStep(node_id="clarifier", symptoms={"clarification_asked": "False"}, latency_ms=10, tokens=5),
            TraceStep(node_id="generator", symptoms={"groundedness": 0.15, "entity_match": "False",
                                                       "hallucination_risk": "none"}, latency_ms=200, tokens=300),
            TraceStep(node_id="evaluator", symptoms={"final_score": 0.15}, latency_ms=10, tokens=5),
            TraceStep(node_id="human", symptoms={"escalated": "True"}, latency_ms=0, tokens=0),
        ]
        trace = Trace(trace_id="kb_stale_demo2", steps=steps, final_outcome_failed=True,
                       ground_truth_node="kb_builder", failure_type="stale_knowledge_correct_entity")

        add_filter_result = evaluate_repair(graph, trace, ActionType.ADD_FILTER, "retriever")
        self.assertTrue(add_filter_result.accepted)

        resolved = _hypothesis_symptom_actually_resolved(
            "retriever", "low_retrieval_score_0.22", add_filter_result.after_trace)
        self.assertFalse(resolved, "raw retrieval_top1_score (0.27) is still below the 0.55 "
                                    "threshold that flagged retriever -- should not count as confirmed")


if __name__ == "__main__":
    unittest.main()
