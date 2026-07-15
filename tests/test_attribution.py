import tests._pathfix
import unittest

from graph import build_support_pipeline
from attribution import (
    Trace, TraceStep, FailureAttributor, heuristic_judge,
)
from simulate import generate_traces, wrong_variant_scenario, ScenarioConfig


class TestNoLeakage(unittest.TestCase):

    def setUp(self):
        self.graph = build_support_pipeline()

    def test_rendered_prompt_never_contains_ground_truth_node_string(self):
        traces = generate_traces(self.graph, n=50, seed=3)
        failed = [t for t in traces if t.final_outcome_failed and t.ground_truth_node]
        self.assertGreater(len(failed), 0, "test setup problem: no failed traces to check")
        for t in failed:
            rendered = FailureAttributor._render_full_trace(t)
            self.assertNotIn("ground_truth", rendered.lower())
            self.assertNotIn("root_cause", rendered.lower())
            self.assertNotIn("fail_signal", rendered.lower())

    def test_showcase_scenario_prompt_does_not_leak_answer(self):
        graph = build_support_pipeline()
        trace = wrong_variant_scenario(graph)
        rendered = FailureAttributor._render_full_trace(trace)
        self.assertNotIn("ground_truth", rendered.lower())
        self.assertNotIn("retriever_is_root_cause", rendered.lower())

    def test_trace_step_symptoms_have_no_boolean_failed_flag(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=20, seed=1)
        for t in traces:
            for step in t.steps:
                for key in step.symptoms:
                    self.assertNotIn("failed", key.lower())
                    self.assertNotIn("responsible", key.lower())


class TestHeuristicJudge(unittest.TestCase):
    def test_stale_kb_triggers_kb_builder_verdict(self):
        prompt = "node=kb_builder kb_age_days=120.0"
        verdict = heuristic_judge(prompt)
        self.assertIn("RESPONSIBLE=kb_builder", verdict)

    def test_low_retrieval_score_triggers_retriever_verdict(self):
        prompt = "node=retriever retrieval_top1_score=0.30 entity_match=True"
        verdict = heuristic_judge(prompt)
        self.assertIn("RESPONSIBLE=retriever", verdict)

    def test_no_symptom_signal_returns_none_not_a_guess(self):
        prompt = "node=chunker"
        verdict = heuristic_judge(prompt)
        self.assertIn("RESPONSIBLE=NONE", verdict)

    def test_highest_scoring_symptom_wins_when_multiple_present(self):
        prompt = (
            "node=retriever retrieval_top1_score=0.20 entity_match=False\n"
            "node=generator groundedness=0.48"
        )
        verdict = heuristic_judge(prompt)
        self.assertIn("RESPONSIBLE=retriever", verdict)


class TestParseVerdict(unittest.TestCase):
    def test_parses_all_three_fields(self):
        node, conf, reason = FailureAttributor._parse_verdict(
            "RESPONSIBLE=generator CONFIDENCE=0.73 REASON=low_groundedness"
        )
        self.assertEqual(node, "generator")
        self.assertAlmostEqual(conf, 0.73)
        self.assertEqual(reason, "low groundedness")

    def test_none_verdict_parses_to_none_node_low_confidence(self):
        node, conf, _ = FailureAttributor._parse_verdict(
            "RESPONSIBLE=NONE CONFIDENCE=0.20 REASON=no_clear_symptom"
        )
        self.assertEqual(node, "NONE")
        self.assertAlmostEqual(conf, 0.20)


class TestFailureAttributorJudgeCallCounts(unittest.TestCase):

    def setUp(self):
        self.graph = build_support_pipeline()
        self.calls = []

        def counting_judge(prompt: str) -> str:
            self.calls.append(prompt)
            return "RESPONSIBLE=NONE CONFIDENCE=0.20 REASON=no_clear_symptom"

        self.attributor = FailureAttributor(judge=counting_judge)
        self.trace = wrong_variant_scenario(self.graph)

    def test_all_at_once_calls_judge_exactly_once(self):
        self.calls.clear()
        result = self.attributor.attribute_all_at_once(self.trace)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result.judge_calls_used, 1)

    def test_step_by_step_calls_judge_once_per_step(self):
        self.calls.clear()
        result = self.attributor.attribute_step_by_step(self.trace)
        self.assertEqual(len(self.calls), len(self.trace.steps))
        self.assertEqual(result.judge_calls_used, len(self.trace.steps))

    def test_binary_search_calls_judge_at_most_log2_n_plus_one(self):
        import math
        self.calls.clear()
        result = self.attributor.attribute_binary_search(self.trace)
        n = len(self.trace.steps)
        upper_bound = math.ceil(math.log2(n)) + 2
        self.assertLessEqual(result.judge_calls_used, upper_bound)


class TestTopologyHeuristic(unittest.TestCase):
    def test_boost_is_bounded_at_0_15(self):
        graph = build_support_pipeline()

        def always_confident_judge(prompt: str) -> str:
            return "RESPONSIBLE=chunker CONFIDENCE=0.90 REASON=test"

        attributor = FailureAttributor(judge=always_confident_judge)
        trace = wrong_variant_scenario(graph)
        base = attributor.attribute_all_at_once(trace)
        boosted = attributor.attribute_with_topology_heuristic(trace, graph)
        self.assertGreaterEqual(boosted.confidence, base.confidence)
        self.assertLessEqual(boosted.confidence - base.confidence, 0.15 + 1e-9)
        self.assertLessEqual(boosted.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
