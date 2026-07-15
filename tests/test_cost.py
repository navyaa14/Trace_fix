import tests._pathfix
import unittest

from cost import step_cost, trace_cost, CostBreakdown, _latency_penalty, HUMAN_ESCALATION_USD
from attribution import Trace, TraceStep


class TestLatencyPenalty(unittest.TestCase):
    def test_zero_penalty_within_sla(self):
        self.assertEqual(_latency_penalty(250, sla_ms=300), 0.0)
        self.assertEqual(_latency_penalty(300, sla_ms=300), 0.0)

    def test_positive_penalty_past_sla(self):
        self.assertGreater(_latency_penalty(600, sla_ms=300), 0.0)

    def test_penalty_grows_superlinearly(self):
        p1 = _latency_penalty(450, sla_ms=300)
        p2 = _latency_penalty(600, sla_ms=300)
        ratio = p2 / p1
        self.assertGreater(ratio, 2.0)


class TestStepCost(unittest.TestCase):
    def test_clarifier_incurs_friction_and_extra_turn_cost(self):
        c = step_cost("clarifier", latency_ms=100, tokens=400)
        self.assertEqual(c.friction_score, 1.0)
        self.assertGreater(c.api_cost_usd, (400 / 1000.0) * 0.003)

    def test_non_clarifier_has_zero_friction(self):
        c = step_cost("retriever", latency_ms=100, tokens=300)
        self.assertEqual(c.friction_score, 0.0)

    def test_human_node_only_incurs_human_cost(self):
        c = step_cost("human", latency_ms=0, tokens=0)
        self.assertEqual(c.human_cost_usd, HUMAN_ESCALATION_USD)
        self.assertEqual(c.api_cost_usd, 0.0)

    def test_non_human_node_has_zero_human_cost(self):
        c = step_cost("generator", latency_ms=100, tokens=500)
        self.assertEqual(c.human_cost_usd, 0.0)

    def test_api_cost_scales_with_tokens(self):
        cheap = step_cost("generator", latency_ms=100, tokens=100)
        expensive = step_cost("generator", latency_ms=100, tokens=1000)
        self.assertLess(cheap.api_cost_usd, expensive.api_cost_usd)


class TestTraceCost(unittest.TestCase):
    def test_trace_cost_sums_across_steps(self):
        steps = [
            TraceStep(node_id="retriever", symptoms={}, latency_ms=100, tokens=200),
            TraceStep(node_id="generator", symptoms={}, latency_ms=100, tokens=300),
            TraceStep(node_id="human", symptoms={}, latency_ms=0, tokens=0),
        ]
        trace = Trace(trace_id="t", steps=steps, final_outcome_failed=True)
        total = trace_cost(trace)
        expected_api = (200 / 1000.0) * 0.003 + (300 / 1000.0) * 0.003
        self.assertAlmostEqual(total.api_cost_usd, expected_api, places=6)
        self.assertEqual(total.human_cost_usd, HUMAN_ESCALATION_USD)

    def test_empty_trace_has_zero_cost(self):
        trace = Trace(trace_id="empty", steps=[], final_outcome_failed=False)
        total = trace_cost(trace)
        self.assertEqual(total.api_cost_usd, 0.0)
        self.assertEqual(total.human_cost_usd, 0.0)


class TestWeightedObjective(unittest.TestCase):
    def test_default_weights_sum_all_terms(self):
        c = CostBreakdown(api_cost_usd=1.0, latency_penalty_score=2.0,
                           friction_score=3.0, human_cost_usd=4.0)
        self.assertEqual(c.weighted_objective(), 10.0)

    def test_zero_weight_excludes_a_term(self):
        c = CostBreakdown(api_cost_usd=1.0, latency_penalty_score=2.0,
                           friction_score=3.0, human_cost_usd=4.0)
        self.assertEqual(c.weighted_objective(w_human=0.0), 6.0)

    def test_objective_mixes_units_by_design_not_accident(self):
        c = CostBreakdown(api_cost_usd=0.01, latency_penalty_score=5.0,
                           friction_score=0.0, human_cost_usd=0.0)
        self.assertGreater(c.weighted_objective(), c.api_cost_usd + c.human_cost_usd)


if __name__ == "__main__":
    unittest.main()

class TestClarifierFrictionOnlyWhenAsked(unittest.TestCase):

    def test_clarifier_executed_but_no_question_asked_has_zero_friction(self):
        steps = [TraceStep(node_id="clarifier",
                            symptoms={"clarification_asked": "False", "query_ambiguous": "False"},
                            latency_ms=20, tokens=10)]
        trace = Trace(trace_id="t", steps=steps, final_outcome_failed=False)
        total = trace_cost(trace)
        self.assertEqual(total.friction_score, 0.0)

    def test_clarification_actually_asked_incurs_friction(self):
        steps = [TraceStep(node_id="clarifier",
                            symptoms={"clarification_asked": "True", "query_ambiguous": "True"},
                            latency_ms=250, tokens=400)]
        trace = Trace(trace_id="t", steps=steps, final_outcome_failed=False)
        total = trace_cost(trace)
        self.assertEqual(total.friction_score, 1.0)

    def test_repeated_clarification_turns_scale_friction(self):
        steps = [TraceStep(node_id="clarifier",
                            symptoms={"clarification_asked": "True", "clarification_turns": "3"},
                            latency_ms=250, tokens=400)]
        trace = Trace(trace_id="t", steps=steps, final_outcome_failed=False)
        total = trace_cost(trace)
        self.assertEqual(total.friction_score, 3.0)


class TestObservedVsConfiguredCost(unittest.TestCase):

    def test_two_batches_with_different_latency_produce_different_observed_cost(self):
        from graph import build_support_pipeline
        from optimizer import GraphOptimizer

        graph = build_support_pipeline()
        optimizer = GraphOptimizer(graph)

        cheap_traces = [Trace(trace_id=f"c{i}",
                               steps=[TraceStep(node_id="retriever", symptoms={}, latency_ms=50, tokens=100)],
                               final_outcome_failed=False) for i in range(5)]
        expensive_traces = [Trace(trace_id=f"e{i}",
                                   steps=[TraceStep(node_id="retriever", symptoms={}, latency_ms=900, tokens=2000)],
                                   final_outcome_failed=False) for i in range(5)]

        cheap_stats = optimizer.aggregate(cheap_traces, [])
        expensive_stats = optimizer.aggregate(expensive_traces, [])

        self.assertNotAlmostEqual(
            cheap_stats["retriever"].observed_average_cost.api_cost_usd,
            expensive_stats["retriever"].observed_average_cost.api_cost_usd, places=6)
        self.assertLess(cheap_stats["retriever"].observed_average_cost.api_cost_usd,
                         expensive_stats["retriever"].observed_average_cost.api_cost_usd)
        self.assertEqual(cheap_stats["retriever"].configured_cost_estimate.api_cost_usd,
                          expensive_stats["retriever"].configured_cost_estimate.api_cost_usd)


if __name__ == "__main__":
    unittest.main()
