import tests._pathfix
import unittest

from graph import build_support_pipeline, ActionType
from simulate import wrong_variant_scenario, outcome_failed
from repair import replay_with_add_filter


class TestReplayWithAddFilter(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.trace = wrong_variant_scenario(self.graph)
        self.result = replay_with_add_filter(self.graph, self.trace)

    def test_before_reflects_the_original_failed_trace(self):
        self.assertTrue(self.result.before_failed)
        self.assertEqual(self.result.before_groundedness, 0.44)

    def test_after_is_a_success(self):
        self.assertFalse(self.result.after_failed)

    def test_after_groundedness_improves(self):
        self.assertGreater(self.result.after_groundedness, self.result.before_groundedness)

    def test_human_escalation_cost_is_removed_after_repair(self):
        self.assertGreater(self.result.before_cost.human_cost_usd, 0.0)
        self.assertEqual(self.result.after_cost.human_cost_usd, 0.0)

    def test_repair_costs_slightly_more_in_api_tokens(self):
        self.assertGreater(self.result.after_cost.api_cost_usd, self.result.before_cost.api_cost_usd)

    def test_action_recorded_is_add_filter(self):
        self.assertEqual(self.result.action, ActionType.ADD_FILTER)

    def test_original_trace_object_is_not_mutated(self):
        retriever_step = next(s for s in self.trace.steps if s.node_id == "retriever")
        self.assertEqual(retriever_step.symptoms.get("entity_match"), "False")

    def test_after_failed_is_recomputed_not_hardcoded(self):
        after_entity_match = True
        after_clarified = False
        expected = outcome_failed(self.result.after_groundedness, after_entity_match, after_clarified)
        self.assertEqual(self.result.after_failed, expected)
        self.assertFalse(expected)


if __name__ == "__main__":
    unittest.main()
