import unittest
import tempfile
import os
import tests._pathfix

from graph import build_support_pipeline, ActionType
from simulate import generate_traces
from attribution import FailureAttributor
from learning_memory import LearningMemory
from continuous_improvement import run_improvement_cycle


class TestContinuousImprovementLoop(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "mem.json")
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=300, seed=7)
        self.attributor = FailureAttributor()

    def test_full_loop_over_real_batch(self):
        attributions = [self.attributor.attribute_binary_search(t) for t in self.traces]
        mem = LearningMemory(self.path)
        report = run_improvement_cycle(self.graph, self.traces, attributions, mem, ActionType)

        self.assertGreater(report.attempted, 0)
        self.assertGreater(report.accepted, 0)
        self.assertGreaterEqual(report.human_escalations_before, report.human_escalations_after)
        self.assertLessEqual(report.total_after_cost_usd, report.total_before_cost_usd + 1e-6)

    def test_memory_persists_across_two_independent_batches(self):
        attributions = [self.attributor.attribute_binary_search(t) for t in self.traces]
        mem1 = LearningMemory(self.path)
        run_improvement_cycle(self.graph, self.traces, attributions, mem1, ActionType)

        second_batch = generate_traces(self.graph, n=300, seed=99)
        second_attrs = [self.attributor.attribute_binary_search(t) for t in second_batch]
        mem2 = LearningMemory(self.path)
        self.assertGreater(len(mem2.all_entries()), 0,
                            "second run must see history the first run persisted")
        run_improvement_cycle(self.graph, second_batch, second_attrs, mem2, ActionType)

        combined_stats = mem2.stats("retriever", "ADD_FILTER")
        self.assertGreaterEqual(combined_stats.attempts, 2,
                                 "attempts should accumulate across independent runs, not reset")

    def test_human_review_selected_when_attribution_confidence_is_low(self):
        from attribution import AttributionResult
        from continuous_improvement import LOW_CONFIDENCE_THRESHOLD

        trace = self.traces[0]
        trace = [t for t in self.traces if t.final_outcome_failed][0]
        low_conf_attr = AttributionResult(trace.trace_id, "retriever",
                                           LOW_CONFIDENCE_THRESHOLD - 0.1, "all_at_once", 1, "uncertain")
        memory = LearningMemory(self.path)
        report = run_improvement_cycle(self.graph, [trace], [low_conf_attr], memory, ActionType)
        self.assertEqual(report.human_review_selected, 1)
        self.assertEqual(report.attempted, 0, "no automated candidate should even be tried")
        self.assertEqual(report.decisions[0].outcome, "human_review")


    def test_run2_selects_differently_because_of_run1_learning(self):
        import random

        traces1 = generate_traces(self.graph, n=400, seed=7)
        attrs1 = [self.attributor.attribute_binary_search(t) for t in traces1]
        mem_run1 = LearningMemory(self.path)
        run_improvement_cycle(self.graph, traces1, attrs1, mem_run1, ActionType,
                               epsilon=0.0, rng=random.Random(1))

        traces2 = generate_traces(self.graph, n=200, seed=99)
        attrs2 = [self.attributor.attribute_binary_search(t) for t in traces2]

        cold_mem = LearningMemory(os.path.join(self.tmpdir, "cold.json"))
        cold_report = run_improvement_cycle(self.graph, traces2, attrs2, cold_mem, ActionType,
                                             epsilon=0.0, rng=random.Random(1))

        learned_mem = LearningMemory(self.path)
        self.assertGreater(len(learned_mem.all_entries()), 0,
                            "run 2 must be able to load run 1's persisted memory")
        learned_report = run_improvement_cycle(self.graph, traces2, attrs2, learned_mem, ActionType,
                                                epsilon=0.0, rng=random.Random(1))

        cold_actions = {d.trace_id: (d.policy_decision.action if d.policy_decision else None)
                         for d in cold_report.decisions}
        learned_actions = {d.trace_id: (d.policy_decision.action if d.policy_decision else None)
                            for d in learned_report.decisions}
        differing = [tid for tid in cold_actions
                     if cold_actions[tid] is not None and cold_actions[tid] != learned_actions.get(tid)]
        exploit_in_learned = sum(1 for d in learned_report.decisions
                                  if d.policy_decision and d.policy_decision.exploration_or_exploitation == "exploit")
        exploit_in_cold = sum(1 for d in cold_report.decisions
                               if d.policy_decision and d.policy_decision.exploration_or_exploitation == "exploit")

        self.assertGreater(exploit_in_learned, 0,
                            "the learned run should have UCB1 'exploit' decisions backed by run-1 history")
        self.assertGreater(exploit_in_learned, exploit_in_cold,
                            "the learned run should exploit prior history far more than the cold run, "
                            "which has nothing to exploit")
        self.assertTrue(differing or exploit_in_learned > 0)


if __name__ == "__main__":
    unittest.main()
