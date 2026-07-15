import unittest
import tempfile
import os
import tests._pathfix

from graph import build_support_pipeline, ActionType
from simulate import wrong_variant_scenario
from repair_engine import evaluate_repair
from learning_memory import LearningMemory


class TestLearningMemoryPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "mem.json")
        self.graph = build_support_pipeline()

    def test_records_and_persists_across_instances(self):
        mem = LearningMemory(self.path)
        trace = wrong_variant_scenario(self.graph)
        result = evaluate_repair(self.graph, trace, ActionType.ADD_FILTER, "retriever")
        mem.record(result)
        mem.save()

        self.assertTrue(os.path.exists(self.path))

        mem2 = LearningMemory(self.path)
        stats = mem2.stats("retriever", "ADD_FILTER")
        self.assertEqual(stats.attempts, 1)
        self.assertEqual(stats.accepted_count, 1)
        self.assertEqual(stats.accept_rate, 1.0)

    def test_accept_rate_math_over_multiple_records(self):
        mem = LearningMemory(self.path)

        class FakeCost:
            def __init__(self, api=0.0, human=0.0):
                self.api_cost_usd = api
                self.human_cost_usd = human

        from repair_engine import ValidatedRepair
        mem.record(ValidatedRepair("t1", "retriever", ActionType.ADD_FILTER, True,
                                    True, False, FakeCost(0.0), FakeCost(0.001), True, "ok"))
        mem.record(ValidatedRepair("t2", "retriever", ActionType.ADD_FILTER, True,
                                    True, True, FakeCost(0.0), FakeCost(0.001), False, "still failed"))
        stats = mem.stats("retriever", "ADD_FILTER")
        self.assertEqual(stats.attempts, 2)
        self.assertEqual(stats.accept_rate, 0.5)

    def test_skips_unapplied_repairs(self):
        mem = LearningMemory(self.path)
        traces_pass_result = evaluate_repair(
            self.graph,
            wrong_variant_scenario(self.graph), ActionType.CACHE, "retriever")
        mem.record(traces_pass_result)
        self.assertEqual(len(mem.all_entries()), 0)

    def test_best_action_for_prefers_higher_accept_rate(self):
        mem = LearningMemory(self.path)
        from repair_engine import ValidatedRepair

        class FakeCost:
            def __init__(self):
                self.api_cost_usd = 0.0
                self.human_cost_usd = 0.0

        for _ in range(3):
            mem.record(ValidatedRepair("t", "retriever", ActionType.ADD_FILTER, True,
                                        True, False, FakeCost(), FakeCost(), True, "ok"))
        for _ in range(3):
            mem.record(ValidatedRepair("t", "retriever", ActionType.RETRY, True,
                                        True, True, FakeCost(), FakeCost(), False, "failed"))

        best = mem.best_action_for("retriever", ["ADD_FILTER", "RETRY"])
        self.assertEqual(best, "ADD_FILTER")

    def test_returns_none_with_no_history(self):
        mem = LearningMemory(self.path)
        self.assertIsNone(mem.best_action_for("retriever", ["ADD_FILTER"]))


if __name__ == "__main__":
    unittest.main()
