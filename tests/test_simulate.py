import tests._pathfix
import unittest

from graph import build_support_pipeline
from simulate import generate_traces, wrong_variant_scenario, ScenarioConfig, outcome_failed
from causal_model import evaluation_failed as _evaluation_failed, workflow_failed as _workflow_failed


class TestGenerateTraces(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()

    def test_same_seed_produces_identical_traces(self):
        t1 = generate_traces(self.graph, n=30, seed=42)
        t2 = generate_traces(self.graph, n=30, seed=42)
        outcomes1 = [(t.final_outcome_failed, t.ground_truth_node) for t in t1]
        outcomes2 = [(t.final_outcome_failed, t.ground_truth_node) for t in t2]
        self.assertEqual(outcomes1, outcomes2)

    def test_different_seed_can_produce_different_traces(self):
        t1 = generate_traces(self.graph, n=30, seed=1)
        t2 = generate_traces(self.graph, n=30, seed=2)
        outcomes1 = [(t.final_outcome_failed, t.ground_truth_node) for t in t1]
        outcomes2 = [(t.final_outcome_failed, t.ground_truth_node) for t in t2]
        self.assertNotEqual(outcomes1, outcomes2)

    def test_every_failed_trace_has_a_ground_truth_node(self):
        traces = generate_traces(self.graph, n=200, seed=5)
        for t in traces:
            if t.final_outcome_failed:
                self.assertIsNotNone(t.ground_truth_node,
                                      f"{t.trace_id} failed but has no ground_truth_node")

    def test_every_successful_trace_has_no_ground_truth_node(self):
        traces = generate_traces(self.graph, n=200, seed=5)
        for t in traces:
            if not t.final_outcome_failed:
                self.assertIsNone(t.ground_truth_node,
                                   f"{t.trace_id} succeeded but has a ground_truth_node")

    def test_failure_rate_is_in_a_plausible_range(self):
        traces = generate_traces(self.graph, n=300, seed=11)
        fail_rate = sum(1 for t in traces if t.final_outcome_failed) / len(traces)
        self.assertGreater(fail_rate, 0.01)
        self.assertLess(fail_rate, 0.5)

    def test_extreme_kb_staleness_config_increases_failure_rate(self):
        low_stale = generate_traces(self.graph, n=200,
                                     config=ScenarioConfig(p_kb_stale=0.0), seed=9)
        high_stale = generate_traces(self.graph, n=200,
                                      config=ScenarioConfig(p_kb_stale=0.9), seed=9)
        rate_low = sum(1 for t in low_stale if t.final_outcome_failed) / len(low_stale)
        rate_high = sum(1 for t in high_stale if t.final_outcome_failed) / len(high_stale)
        self.assertGreater(rate_high, rate_low)


class TestOutcomeFailed(unittest.TestCase):

    def test_low_groundedness_alone_fails(self):
        self.assertTrue(outcome_failed(groundedness=0.3, entity_match=True, clarified=True))

    def test_entity_mismatch_without_clarification_fails(self):
        self.assertTrue(outcome_failed(groundedness=0.9, entity_match=False, clarified=False))

    def test_entity_mismatch_with_clarification_does_not_fail_on_that_branch(self):
        self.assertFalse(outcome_failed(groundedness=0.9, entity_match=False, clarified=True))

    def test_high_groundedness_and_entity_match_succeeds(self):
        self.assertFalse(outcome_failed(groundedness=0.9, entity_match=True, clarified=False))

    def test_matches_generate_traces_failure_flag(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=150, seed=21)
        for t in traces:
            generator_step = next(s for s in t.steps if s.node_id == "generator")
            clarifier_step = next((s for s in t.steps if s.node_id == "clarifier"), None)
            groundedness = float(generator_step.symptoms["groundedness"])
            entity_match = generator_step.symptoms["entity_match"] == "True"
            clarified = (clarifier_step.symptoms.get("clarification_asked") == "True"
                         if clarifier_step else False)
            # content_failed (outcome_failed) is a pure function of generator/clarifier
            # symptoms -- this invariant is UNCHANGED from the prior revision.
            expected_content_failed = outcome_failed(groundedness, entity_match, clarified)
            self.assertEqual(expected_content_failed, t.content_failed, t.trace_id)
            # final_outcome_failed is now workflow_failed = content_failed OR
            # evaluation_failed (REQUIRED: evaluator can independently cause a
            # workflow failure -- see causal_model.workflow_failed).
            expected_eval_failed = _evaluation_failed(t.evaluator_disagreement)
            expected_workflow_failed = _workflow_failed(expected_content_failed, expected_eval_failed)
            self.assertEqual(expected_workflow_failed, t.final_outcome_failed, t.trace_id)


class TestWrongVariantScenario(unittest.TestCase):
    def test_ground_truth_is_retriever(self):
        graph = build_support_pipeline()
        trace = wrong_variant_scenario(graph)
        self.assertEqual(trace.ground_truth_node, "retriever")

    def test_trace_is_marked_failed(self):
        graph = build_support_pipeline()
        trace = wrong_variant_scenario(graph)
        self.assertTrue(trace.final_outcome_failed)

    def test_includes_human_escalation_step(self):
        graph = build_support_pipeline()
        trace = wrong_variant_scenario(graph)
        node_ids = [s.node_id for s in trace.steps]
        self.assertIn("human", node_ids)

    def test_is_deterministic_no_randomness(self):
        graph = build_support_pipeline()
        t1 = wrong_variant_scenario(graph)
        t2 = wrong_variant_scenario(graph)
        self.assertEqual(
            [(s.node_id, s.symptoms) for s in t1.steps],
            [(s.node_id, s.symptoms) for s in t2.steps],
        )


if __name__ == "__main__":
    unittest.main()
