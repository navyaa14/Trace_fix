
import tests._pathfix
import unittest
import statistics

from graph import build_support_pipeline
from attribution import FailureAttributor, Trace, TraceStep, matches_ground_truth
from simulate import generate_traces, ScenarioConfig
from demo import accuracy_by_failure_type, accuracy_multi_label_by_failure_type


class TestMatchesGroundTruthIsBackwardCompatible(unittest.TestCase):

    def test_single_cause_trace_uses_plain_equality(self):
        t = Trace(trace_id="t1", steps=[], final_outcome_failed=True,
                   ground_truth_node="retriever", failure_type="wrong_entity_variant")
        self.assertTrue(matches_ground_truth(t, "retriever"))
        self.assertFalse(matches_ground_truth(t, "generator"))
        self.assertFalse(matches_ground_truth(t, None))

    def test_trace_with_no_ground_truth_does_not_match_a_real_prediction(self):
        t = Trace(trace_id="t2", steps=[], final_outcome_failed=False, ground_truth_node=None)
        self.assertFalse(matches_ground_truth(t, "retriever"))
        self.assertTrue(matches_ground_truth(t, None))

    def test_ground_truth_nodes_defaults_to_none(self):
        t = Trace(trace_id="t3", steps=[], final_outcome_failed=True, ground_truth_node="kb_builder")
        self.assertIsNone(t.ground_truth_nodes)


class TestMultiLabelGroundTruthOnlyForMultiCauseTraces(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_support_pipeline()
        cls.traces = generate_traces(cls.graph, n=300, config=ScenarioConfig(), seed=11)

    def test_multi_cause_traces_have_exactly_two_ground_truth_nodes(self):
        multi = [t for t in self.traces if t.failure_type == "multiple_simultaneous_failures"]
        self.assertGreater(len(multi), 0, "headline batch should contain multi-cause traces")
        for t in multi:
            self.assertIsNotNone(t.ground_truth_nodes)
            self.assertEqual(len(t.ground_truth_nodes), 2)
            self.assertIn(t.ground_truth_node, t.ground_truth_nodes)
            self.assertIn("generator", t.ground_truth_nodes)

    def test_non_multi_cause_traces_have_no_ground_truth_nodes(self):
        # evaluator_false_acceptance traces are a second, deliberate multi-cause
        # category (upstream content cause + evaluator) -- excluded here and
        # covered by test_evaluator_false_acceptance_is_multi_cause below.
        non_multi = [t for t in self.traces
                     if t.final_outcome_failed and t.failure_type != "multiple_simultaneous_failures"
                     and not t.evaluator_false_acceptance]
        self.assertGreater(len(non_multi), 0)
        for t in non_multi:
            self.assertIsNone(t.ground_truth_nodes)

    def test_evaluator_false_acceptance_is_multi_cause(self):
        # A false acceptance is causally two things at once: whatever upstream
        # node actually produced the bad content, AND the evaluator that wrongly
        # let it through. The original upstream cause must be preserved, not
        # replaced, and "evaluator" must be added alongside it.
        false_acceptances = [t for t in self.traces if t.evaluator_false_acceptance]
        self.assertGreater(len(false_acceptances), 0,
                            "batch should contain at least one evaluator false acceptance")
        for t in false_acceptances:
            if t.ground_truth_node == "evaluator":
                continue  # vade-miss-driven false acceptance already routes through vade
            self.assertIsNotNone(t.ground_truth_nodes)
            self.assertIn("evaluator", t.ground_truth_nodes)
            self.assertIn(t.ground_truth_node, t.ground_truth_nodes)
            self.assertTrue(matches_ground_truth(t, "evaluator"))
            self.assertTrue(matches_ground_truth(t, t.ground_truth_node))

    def test_successful_traces_have_no_ground_truth_nodes(self):
        for t in self.traces:
            if not t.final_outcome_failed:
                self.assertIsNone(t.ground_truth_nodes)


class TestSingleLabelScoringUnchangedByTheFix(unittest.TestCase):

    def test_all_at_once_single_label_breakdown_unchanged(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=300, config=ScenarioConfig(), seed=11)
        attributor = FailureAttributor()
        breakdown = accuracy_by_failure_type(attributor.attribute_all_at_once, traces)
        acc, n = breakdown["multiple_simultaneous_failures"]
        self.assertEqual(n, 16)
        self.assertAlmostEqual(acc, 0.125, places=4)


class TestMultiLabelScoringMeasuresTheRealFix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_support_pipeline()
        cls.traces = generate_traces(cls.graph, n=300, config=ScenarioConfig(), seed=11)
        cls.attributor = FailureAttributor()

    def test_all_at_once_multi_label_accuracy_on_multi_cause_category(self):
        breakdown = accuracy_multi_label_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        acc, n = breakdown["multiple_simultaneous_failures"]
        self.assertEqual(n, 16)
        self.assertGreater(acc, 0.85,
                            "multi-label accuracy should recover most of the single-label "
                            "gap if most 'wrong' predictions were actually the other true cause")

    def test_a2p_scaffold_multi_label_accuracy_on_multi_cause_category(self):
        a2p_fn = lambda t: self.attributor.attribute_a2p_scaffold(t, self.graph)
        breakdown = accuracy_multi_label_by_failure_type(a2p_fn, self.traces)
        acc, n = breakdown["multiple_simultaneous_failures"]
        self.assertEqual(n, 16)
        self.assertGreater(acc, 0.75)

    def test_multi_label_never_scores_lower_than_single_label_for_any_failure_type(self):
        single = accuracy_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        multi = accuracy_multi_label_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        for failure_type, (single_acc, n) in single.items():
            multi_acc, _ = multi[failure_type]
            self.assertGreaterEqual(multi_acc, single_acc - 1e-9,
                                     f"{failure_type}: multi-label ({multi_acc}) should never "
                                     f"score below single-label ({single_acc})")

    def test_other_failure_types_unaffected_by_multi_label_scoring(self):
        single = accuracy_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        multi = accuracy_multi_label_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        for failure_type, (single_acc, n) in single.items():
            if failure_type == "multiple_simultaneous_failures":
                continue
            multi_acc, _ = multi[failure_type]
            self.assertAlmostEqual(single_acc, multi_acc, places=4,
                                    msg=f"{failure_type} should be unaffected by the multi-label fix")


class TestMultiLabelEffectHoldsAcrossSeeds(unittest.TestCase):

    def test_gap_is_large_and_reproducible_across_five_seeds(self):
        graph = build_support_pipeline()
        attributor = FailureAttributor()
        single_accs, multi_accs = [], []
        for seed in range(100, 105):
            traces = generate_traces(graph, n=250, config=ScenarioConfig(), seed=seed)
            single = accuracy_by_failure_type(attributor.attribute_all_at_once, traces)
            multi = accuracy_multi_label_by_failure_type(attributor.attribute_all_at_once, traces)
            if "multiple_simultaneous_failures" not in single:
                continue
            s_acc, _ = single["multiple_simultaneous_failures"]
            m_acc, _ = multi["multiple_simultaneous_failures"]
            single_accs.append(s_acc)
            multi_accs.append(m_acc)

        self.assertGreaterEqual(len(single_accs), 4,
                                 "expected multi-cause traces in most of these 5 seeds")
        mean_single = statistics.mean(single_accs)
        mean_multi = statistics.mean(multi_accs)
        self.assertLess(mean_single, 0.10,
                         "single-label accuracy should stay low across seeds, confirming "
                         "the near-zero headline number isn't specific to seed=11")
        self.assertGreater(mean_multi, 0.75,
                            "multi-label accuracy should stay high across seeds, confirming "
                            "the fix is a reproducible effect, not a single lucky seed")


if __name__ == "__main__":
    unittest.main()
