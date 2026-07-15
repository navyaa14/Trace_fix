import tests._pathfix
import unittest
import tempfile
import os

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, wrong_variant_scenario
from attribution import FailureAttributor
from optimizer import GraphOptimizer
from repair import replay_with_add_filter
from learning_memory import LearningMemory
from continuous_improvement import run_improvement_cycle
from report import build_dashboard


class TestDashboardBackwardCompatible(unittest.TestCase):

    def test_original_call_signature_still_works(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=100, seed=7)
        attributor = FailureAttributor()
        attributions = [attributor.attribute_binary_search(t) for t in traces]
        optimizer = GraphOptimizer(graph)
        stats = optimizer.aggregate(traces, attributions)
        recs = optimizer.recommend(stats)
        replay = replay_with_add_filter(graph, wrong_variant_scenario(graph))

        html = build_dashboard(graph, stats, recs, 0.56, 25,
                                [("all_at_once", 0.56, 1.0)], replay)
        self.assertIn("<html", html)
        self.assertNotIn("Continuous improvement", html)


class TestDashboardWithContinuousImprovement(unittest.TestCase):
    def test_learning_and_repair_data_actually_renders(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=300, seed=7)
        attributor = FailureAttributor()
        attributions = [attributor.attribute_binary_search(t) for t in traces]
        optimizer = GraphOptimizer(graph)
        stats = optimizer.aggregate(traces, attributions)
        recs = optimizer.recommend(stats)
        replay = replay_with_add_filter(graph, wrong_variant_scenario(graph))

        tmpdir = tempfile.mkdtemp()
        mem = LearningMemory(os.path.join(tmpdir, "mem.json"))
        report = run_improvement_cycle(graph, traces, attributions, mem, ActionType)

        html = build_dashboard(
            graph, stats, recs, 0.56, 25, [("all_at_once", 0.56, 1.0)], replay,
            learning_entries=mem.all_entries(),
            improvement_repairs=report.repairs,
            improvement_summary={
                "attempted": report.attempted, "accepted": report.accepted,
                "rejected": report.rejected,
                "human_before": report.human_escalations_before,
                "human_after": report.human_escalations_after,
                "cost_before": report.total_before_cost_usd,
                "cost_after": report.total_after_cost_usd,
            })

        self.assertIn("ACCEPTED", html)
        self.assertIn("ADD_FILTER", html)
        self.assertIn("RECHUNK", html)
        self.assertIn(str(report.attempted), html)
        self.assertNotIn("no batch improvement cycle was run", html)


if __name__ == "__main__":
    unittest.main()
