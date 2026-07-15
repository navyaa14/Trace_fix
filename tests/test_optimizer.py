import tests._pathfix
import unittest

from graph import build_support_pipeline, ActionType
from attribution import Trace, TraceStep, AttributionResult
from optimizer import GraphOptimizer


def _trace_touching(node_ids, trace_id="t"):
    return Trace(trace_id=trace_id,
                 steps=[TraceStep(node_id=n, symptoms={}) for n in node_ids],
                 final_outcome_failed=False)


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.optimizer = GraphOptimizer(self.graph)

    def test_failure_rate_uses_execution_count_not_total_trace_count(self):
        traces = [
            _trace_touching(["retriever", "generator"], "t0"),
            _trace_touching(["retriever", "generator"], "t1"),
            _trace_touching(["retriever", "clarifier", "generator"], "t2"),
            _trace_touching(["retriever", "generator"], "t3"),
            _trace_touching(["retriever", "generator"], "t4"),
        ]
        attributions = [
            AttributionResult("t2", "clarifier", 0.8, "all_at_once", 1, "test"),
        ]
        stats = self.optimizer.aggregate(traces, attributions)
        self.assertEqual(stats["clarifier"].executions, 1)
        self.assertEqual(stats["clarifier"].failure_rate, 1.0)

    def test_node_with_zero_executions_has_zero_failure_rate_not_a_crash(self):
        traces = [_trace_touching(["retriever"], "t0")]
        stats = self.optimizer.aggregate(traces, [])
        self.assertEqual(stats["vade"].executions, 0)
        self.assertEqual(stats["vade"].failure_rate, 0.0)

    def test_avg_confidence_only_averages_over_attributed_failures(self):
        traces = [_trace_touching(["generator"], "t0"),
                  _trace_touching(["generator"], "t1")]
        attributions = [
            AttributionResult("t0", "generator", 0.4, "all_at_once", 1, ""),
            AttributionResult("t1", "generator", 0.8, "all_at_once", 1, ""),
        ]
        stats = self.optimizer.aggregate(traces, attributions)
        self.assertAlmostEqual(stats["generator"].avg_confidence, 0.6)

    def test_observed_cost_differs_across_batches_with_different_latency(self):
        fast_batch = [Trace("t0", [TraceStep("generator", {}, latency_ms=50, tokens=100)], False)]
        slow_batch = [Trace("t0", [TraceStep("generator", {}, latency_ms=900, tokens=100)], False)]

        fast_stats = self.optimizer.aggregate(fast_batch, [])
        slow_stats = self.optimizer.aggregate(slow_batch, [])

        self.assertNotAlmostEqual(
            fast_stats["generator"].observed_average_cost.latency_penalty_score,
            slow_stats["generator"].observed_average_cost.latency_penalty_score,
        )
        self.assertEqual(fast_stats["generator"].observed_average_cost.latency_penalty_score, 0.0)
        self.assertGreater(slow_stats["generator"].observed_average_cost.latency_penalty_score, 0.0)
        self.assertEqual(
            fast_stats["generator"].configured_cost_estimate.latency_penalty_score,
            slow_stats["generator"].configured_cost_estimate.latency_penalty_score,
        )


class TestRecommend(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.optimizer = GraphOptimizer(self.graph)

    def test_never_executed_node_gets_keep(self):
        traces = [_trace_touching(["retriever"], "t0")]
        stats = self.optimizer.aggregate(traces, [])
        recs = {r.node_id: r for r in self.optimizer.recommend(stats)}
        self.assertEqual(recs["vade"].action, ActionType.KEEP)

    def test_high_failure_low_confidence_triggers_human_review(self):
        traces = [_trace_touching(["chunker"] * 1, f"t{i}") for i in range(10)]
        attributions = [
            AttributionResult(f"t{i}", "chunker", 0.3, "all_at_once", 1, "")
            for i in range(3)
        ]
        stats = self.optimizer.aggregate(traces, attributions)
        recs = {r.node_id: r for r in self.optimizer.recommend(stats)}
        self.assertEqual(recs["chunker"].action, ActionType.HUMAN_REVIEW)

    def test_kb_builder_high_confidence_high_failure_gets_rechunk(self):
        traces = [_trace_touching(["kb_builder"], f"t{i}") for i in range(10)]
        attributions = [
            AttributionResult(f"t{i}", "kb_builder", 0.9, "all_at_once", 1, "")
            for i in range(5)
        ]
        stats = self.optimizer.aggregate(traces, attributions)
        recs = {r.node_id: r for r in self.optimizer.recommend(stats)}
        self.assertEqual(recs["kb_builder"].action, ActionType.RECHUNK)

    def test_retriever_high_confidence_high_failure_gets_add_filter(self):
        traces = [_trace_touching(["retriever"], f"t{i}") for i in range(10)]
        attributions = [
            AttributionResult(f"t{i}", "retriever", 0.9, "all_at_once", 1, "")
            for i in range(5)
        ]
        stats = self.optimizer.aggregate(traces, attributions)
        recs = {r.node_id: r for r in self.optimizer.recommend(stats)}
        self.assertEqual(recs["retriever"].action, ActionType.ADD_FILTER)

    def test_no_attributed_failures_gets_keep(self):
        traces = [_trace_touching(["evaluator"], f"t{i}") for i in range(5)]
        stats = self.optimizer.aggregate(traces, [])
        recs = {r.node_id: r for r in self.optimizer.recommend(stats)}
        self.assertEqual(recs["evaluator"].action, ActionType.KEEP)
        self.assertIn("no attributed failures", recs["evaluator"].reason)

    def test_cache_is_reachable_from_observed_latency(self):
        traces = [Trace(f"t{i}", [TraceStep("generator", {}, latency_ms=1200, tokens=100)], False)
                  for i in range(10)]
        stats = self.optimizer.aggregate(traces, [])
        recs = {r.node_id: r for r in self.optimizer.recommend(stats)}
        self.assertEqual(recs["generator"].action, ActionType.CACHE)
        self.assertEqual(recs["generator"].failure_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
