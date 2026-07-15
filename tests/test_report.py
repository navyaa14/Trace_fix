import tests._pathfix
import unittest

from graph import build_support_pipeline
from simulate import generate_traces, wrong_variant_scenario
from attribution import FailureAttributor, Trace, TraceStep
from optimizer import GraphOptimizer
from repair import replay_with_add_filter
from report import build_dashboard, _node_status, STATUS_COLOR


class TestNodeStatus(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.optimizer = GraphOptimizer(self.graph)

    def test_idle_when_zero_executions(self):
        trace = Trace(trace_id="t0",
                       steps=[TraceStep(node_id="retriever", symptoms={})],
                       final_outcome_failed=False)
        stats = self.optimizer.aggregate([trace], [])
        self.assertEqual(_node_status("vade", stats), "idle")

    def test_status_missing_from_stats_dict_is_idle(self):
        self.assertEqual(_node_status("nonexistent_node", {}), "idle")


class TestBuildDashboard(unittest.TestCase):
    def test_full_pipeline_produces_valid_looking_html_without_crashing(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=60, seed=4)
        attributor = FailureAttributor()
        optimizer = GraphOptimizer(graph)

        failed = [t for t in traces if t.final_outcome_failed]
        attributions = [attributor.attribute_all_at_once(t) for t in failed]
        stats = optimizer.aggregate(traces, attributions)
        recs = optimizer.recommend(stats)

        showcase = wrong_variant_scenario(graph)
        replay = replay_with_add_filter(graph, showcase)

        html = build_dashboard(graph, stats, recs, agent_accuracy=0.5, n_evaluated=10,
                                method_rows=[("all_at_once", 0.5, 1.0)], replay=replay)

        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", html)
        self.assertIn("TraceFix", html)
        self.assertIn("<svg", html)
        for node_id in graph.nodes:
            self.assertIn(node_id, html)

    def test_all_status_colors_are_valid_hex(self):
        for name, color in STATUS_COLOR.items():
            self.assertTrue(color.startswith("#"), f"{name} color {color!r} isn't hex")
            self.assertEqual(len(color), 7)


class TestDashboardNeverHardcodesReplayOutcome(unittest.TestCase):

    def setUp(self):
        self.graph = build_support_pipeline()
        self.optimizer = GraphOptimizer(self.graph)
        traces = generate_traces(self.graph, n=40, seed=4)
        failed = [t for t in traces if t.final_outcome_failed]
        self.stats = self.optimizer.aggregate(traces, [])
        self.recs = self.optimizer.recommend(self.stats)

    def _dashboard_for(self, replay):
        return build_dashboard(self.graph, self.stats, self.recs, agent_accuracy=0.5,
                                n_evaluated=10, method_rows=[("all_at_once", 0.5, 1.0)], replay=replay)

    def test_successful_repair_renders_resolved(self):
        showcase = wrong_variant_scenario(self.graph)
        replay = replay_with_add_filter(self.graph, showcase)
        self.assertFalse(replay.after_failed)
        html = self._dashboard_for(replay)
        self.assertIn("RESOLVED, no escalation", html)
        self.assertNotIn("STILL FAILING", html)

    def test_failed_repair_renders_failed_not_resolved(self):
        steps = [
            TraceStep(node_id="kb_builder", symptoms={"kb_age_days": 5.0}, latency_ms=50, tokens=100),
            TraceStep(node_id="retriever", symptoms={"retrieval_top1_score": 0.9, "entity_match": "True",
                                                       "variant_mismatch_suspected": "False"},
                       latency_ms=80, tokens=150),
            TraceStep(node_id="clarifier", symptoms={"clarification_asked": "False"}, latency_ms=10, tokens=5),
            TraceStep(node_id="generator", symptoms={"groundedness": 0.2, "entity_match": "True",
                                                       "hallucination_risk": "sticky"},
                       latency_ms=200, tokens=300),
            TraceStep(node_id="evaluator", symptoms={"final_score": 0.2}, latency_ms=10, tokens=5),
            TraceStep(node_id="human", symptoms={"escalated": "True"}, latency_ms=0, tokens=0),
        ]
        stuck_trace = Trace(trace_id="stuck", steps=steps, final_outcome_failed=True,
                             ground_truth_node="generator", failure_type="repeated_hallucination")
        replay = replay_with_add_filter(self.graph, stuck_trace)
        self.assertTrue(replay.after_failed)
        html = self._dashboard_for(replay)
        self.assertIn("STILL FAILING", html)
        self.assertNotIn("RESOLVED, no escalation", html)


class TestFailureTypeTableFlagsLowN(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.optimizer = GraphOptimizer(self.graph)
        traces = generate_traces(self.graph, n=40, seed=4)
        self.stats = self.optimizer.aggregate(traces, [])
        self.recs = self.optimizer.recommend(self.stats)
        showcase = wrong_variant_scenario(self.graph)
        self.replay = replay_with_add_filter(self.graph, showcase)

    def _dashboard_with_breakdown(self, breakdown):
        return build_dashboard(self.graph, self.stats, self.recs, agent_accuracy=0.5,
                                n_evaluated=10, method_rows=[("all_at_once", 0.5, 1.0)],
                                replay=self.replay, failure_type_breakdown=breakdown)

    def test_n_below_threshold_is_flagged_low_n_not_worse_than_random(self):
        html = self._dashboard_with_breakdown({"rare_failure_type": (0.0, 1)})
        self.assertIn("low n (1)", html)
        self.assertNotIn("worse than uniform-random", html)
        self.assertIn(STATUS_COLOR["idle"], html)

    def test_n_at_or_above_threshold_gets_the_normal_treatment(self):
        from attribution import MIN_RELIABLE_N
        html = self._dashboard_with_breakdown({"common_failure_type": (0.0, MIN_RELIABLE_N)})
        self.assertIn("worse than uniform-random", html)
        self.assertNotIn(f"low n ({MIN_RELIABLE_N})", html)


if __name__ == "__main__":
    unittest.main()
