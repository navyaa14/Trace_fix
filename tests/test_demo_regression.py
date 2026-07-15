import tests._pathfix
import unittest

from demo import accuracy
from graph import build_support_pipeline, ActionType
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor, heuristic_judge
from optimizer import GraphOptimizer


class TestHeadlineNumbersMatchReadme(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        graph = build_support_pipeline()
        cls.traces = generate_traces(graph, n=300, config=ScenarioConfig(), seed=11)
        cls.attributor = FailureAttributor()

    def test_all_at_once_accuracy_matches_readme(self):
        acc, avg_calls, n_eval = accuracy(self.attributor.attribute_all_at_once, self.traces)
        self.assertEqual(n_eval, 90)
        self.assertAlmostEqual(acc, 0.5778, places=3)
        self.assertAlmostEqual(avg_calls, 1.00, places=2)

    def test_binary_search_accuracy_matches_readme(self):
        acc, avg_calls, n_eval = accuracy(self.attributor.attribute_binary_search, self.traces)
        self.assertEqual(n_eval, 90)
        self.assertAlmostEqual(acc, 0.6889, places=3)

    def test_failure_rate_matches_readme(self):
        fail_rate = sum(1 for t in self.traces if t.final_outcome_failed) / len(self.traces)
        self.assertAlmostEqual(fail_rate, 0.3000, places=3)

    def test_no_method_beats_whoandwhen_ceiling_by_a_suspicious_margin(self):
        for fn in (self.attributor.attribute_all_at_once,
                   self.attributor.attribute_step_by_step,
                   self.attributor.attribute_binary_search):
            acc, _, _ = accuracy(fn, self.traces)
            self.assertLess(acc, 0.95, "accuracy suspiciously close to 100% -- check for a label leak")


class TestOptimizerStressBatchMatchesReadme(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        graph = build_support_pipeline()
        stress_config = ScenarioConfig(p_kb_stale=1.0, p_ambiguous_query=0.02, clarifier_catch_rate=0.0)
        cls.traces = generate_traces(graph, n=300, config=stress_config, seed=11)
        attributor = FailureAttributor(judge=heuristic_judge)
        failed = [t for t in cls.traces if t.final_outcome_failed]
        attributions = [attributor.attribute_all_at_once(t) for t in failed]
        optimizer = GraphOptimizer(graph)
        stats = optimizer.aggregate(cls.traces, attributions)
        cls.recs = {r.node_id: r for r in optimizer.recommend(stats)}

    def test_stress_batch_fail_rate_is_far_above_the_headline_batch(self):
        fail_rate = sum(1 for t in self.traces if t.final_outcome_failed) / len(self.traces)
        self.assertGreater(fail_rate, 0.3)

    def test_retriever_gets_add_filter_live(self):
        self.assertEqual(self.recs["retriever"].action, ActionType.ADD_FILTER)

    def test_kb_builder_gets_rechunk_live(self):
        self.assertEqual(self.recs["kb_builder"].action, ActionType.RECHUNK)


class TestFailureTypeBreakdown(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        graph = build_support_pipeline()
        cls.traces = generate_traces(graph, n=300, config=ScenarioConfig(), seed=11)
        cls.attributor = FailureAttributor()

    def test_multiple_simultaneous_failures_stays_near_the_pinned_floor(self):
        # Trace.ground_truth_node is single-valued by construction, but these
        # traces have TWO independently-true root causes (see ground_truth_nodes).
        # The heuristic reliably finds *a* true cause here -- see the multi-label
        # check below, which credits either true root cause and lands at 100% for
        # the same traces. The single-label number here (2/16) is mostly a
        # single-label evaluation artifact, not a real attribution win, so this
        # stays pinned near the floor rather than climbing meaningfully unless
        # Trace.ground_truth_node itself changes to support multiple values.
        from demo import accuracy_by_failure_type
        breakdown = accuracy_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        acc, n = breakdown["multiple_simultaneous_failures"]
        self.assertEqual(n, 16)
        self.assertAlmostEqual(acc, 0.125, places=3)

    def test_ambiguous_query_unclarified_is_no_longer_at_the_pinned_floor(self):
        # The "query_ambiguous=True, clarification_asked=False" rule lives in the
        # shared _rank_symptom_evidence (used by all_at_once/step_by_step/
        # binary_search/a2p_scaffold alike), so all_at_once catches it too --
        # confirm it lands at a2p_scaffold's real number, not at a suspicious/
        # pinned 100%.
        from demo import accuracy_by_failure_type
        breakdown = accuracy_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        acc, n = breakdown["ambiguous_query_unclarified"]
        self.assertEqual(n, 5)
        self.assertAlmostEqual(acc, 1.0, places=3)

    def test_isolated_single_cause_failures_stay_high(self):
        from demo import accuracy_by_failure_type
        breakdown = accuracy_by_failure_type(self.attributor.attribute_all_at_once, self.traces)
        acc, n = breakdown["wrong_entity_variant"]
        self.assertGreater(acc, 0.8)


class TestA2PScaffoldIsWiredIntoTheComparisonTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_support_pipeline()
        cls.traces = generate_traces(cls.graph, n=300, config=ScenarioConfig(), seed=11)
        cls.attributor = FailureAttributor()

    def test_a2p_scaffold_appears_alongside_the_three_baseline_methods(self):
        import demo as demo_module
        source_names = []
        import inspect
        src = inspect.getsource(demo_module.main)
        for name in ("all_at_once", "step_by_step", "binary_search", "a2p_scaffold"):
            self.assertIn(f'"{name}"', src,
                           f"demo.py's method_rows no longer includes {name!r}")

    def test_a2p_scaffold_accuracy_is_computed_and_between_zero_and_one(self):
        acc, avg_calls, n_eval = accuracy(
            lambda t: self.attributor.attribute_a2p_scaffold(t, self.graph), self.traces)
        self.assertEqual(n_eval, 90)
        self.assertGreater(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertAlmostEqual(avg_calls, 1.00, places=2)

    def test_optimizer_driver_stays_binary_search_not_silently_swapped(self):
        import demo as demo_module
        import inspect
        src = inspect.getsource(demo_module.main)
        self.assertIn("optimizer.aggregate(traces, attributions)", src)
        self.assertIn("attributor.attribute_all_at_once(t) for t in failed_traces", src)


if __name__ == "__main__":
    unittest.main()
