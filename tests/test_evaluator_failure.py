import tests._pathfix
import unittest

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor
from causal_model import outcome_failed, evaluation_failed, workflow_failed
from repair_engine import apply_repair, evaluate_repair, CANDIDATE_ACTIONS


class TestContentVsEvaluationVsWorkflow(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        # a larger batch to reliably surface the rare false-escalation case
        self.traces = generate_traces(self.graph, n=1500, config=ScenarioConfig(), seed=11)

    def test_content_failed_evaluation_failed_and_workflow_failed_are_tracked_separately(self):
        for t in self.traces[:100]:
            self.assertIsNotNone(t.content_failed)
            self.assertIn(t.evaluation_failed, (True, False))
            self.assertEqual(t.final_outcome_failed, workflow_failed(t.content_failed, t.evaluation_failed))

    def test_good_content_plus_evaluator_rejects_it_is_reachable(self):
        # good content + evaluator rejects it -> evaluator_false_escalation
        false_escalations = [t for t in self.traces if t.failure_type == "evaluator_false_escalation"]
        self.assertGreater(len(false_escalations), 0)
        for t in false_escalations:
            self.assertFalse(t.content_failed)
            self.assertTrue(t.evaluation_failed)
            self.assertTrue(t.final_outcome_failed)
            self.assertEqual(t.ground_truth_node, "evaluator")
            eval_step = next(s for s in t.steps if s.node_id == "evaluator")
            self.assertEqual(eval_step.symptoms["evaluator_flags_low_score"], "True")

    def test_bad_content_plus_evaluator_accepts_it_is_reachable(self):
        # bad content + evaluator accepts it -> evaluator_false_acceptance
        false_acceptances = [t for t in self.traces if t.evaluator_false_acceptance]
        self.assertGreater(len(false_acceptances), 0)
        for t in false_acceptances:
            self.assertTrue(t.content_failed)
            self.assertTrue(t.evaluation_failed)
            self.assertTrue(t.final_outcome_failed)  # already failing on content alone
            eval_step = next(s for s in t.steps if s.node_id == "evaluator")
            self.assertEqual(eval_step.symptoms["evaluator_flags_low_score"], "False")

    def test_evaluator_can_independently_cause_a_workflow_failure(self):
        # A trace exists where content_failed is False (the generated response
        # was actually fine) but workflow_failed is True SOLELY because of the
        # evaluator's own verdict -- this is the crux of "evaluator can
        # independently cause workflow failure without changing content quality".
        independently_evaluator_caused = [
            t for t in self.traces
            if t.final_outcome_failed and not t.content_failed
        ]
        self.assertGreater(len(independently_evaluator_caused), 0)
        for t in independently_evaluator_caused:
            self.assertEqual(t.ground_truth_node, "evaluator")

    def test_workflow_failed_formula_matches_causal_model(self):
        for t in self.traces[:200]:
            self.assertEqual(t.final_outcome_failed,
                              workflow_failed(t.content_failed, t.evaluation_failed))

    def test_evaluator_ground_truth_does_not_leak_into_the_judge_prompt(self):
        false_escalations = [t for t in self.traces if t.failure_type == "evaluator_false_escalation"]
        self.assertTrue(false_escalations)
        for t in false_escalations:
            rendered = FailureAttributor._render_full_trace(t)
            self.assertNotIn("evaluator_false_escalation", rendered)
            self.assertNotIn("ground_truth", rendered.lower())


class TestEvaluatorRepair(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=1500, config=ScenarioConfig(), seed=11)
        self.false_escalations = [t for t in self.traces if t.failure_type == "evaluator_false_escalation"]
        self.assertTrue(self.false_escalations, "need at least one evaluator-driven failure to test repair on")

    def test_evaluator_candidate_actions_are_registered(self):
        self.assertIn(ActionType.SECOND_JUDGE, CANDIDATE_ACTIONS["evaluator"])
        self.assertIn(ActionType.RECALIBRATE_THRESHOLD, CANDIDATE_ACTIONS["evaluator"])

    def test_second_judge_changes_the_verdict_only(self):
        for t in self.false_escalations:
            before_gen = dict(next(s for s in t.steps if s.node_id == "generator").symptoms)
            before_retr = dict(next(s for s in t.steps if s.node_id == "retriever").symptoms)
            result = apply_repair(self.graph, t, ActionType.SECOND_JUDGE, "evaluator")
            self.assertIsNot(result, t)
            after_gen = next(s for s in result.steps if s.node_id == "generator").symptoms
            after_retr = next(s for s in result.steps if s.node_id == "retriever").symptoms
            after_eval = next(s for s in result.steps if s.node_id == "evaluator").symptoms
            # content is untouched, byte for byte
            self.assertEqual(before_gen, after_gen)
            self.assertEqual(before_retr, after_retr)
            # only the verdict flips
            self.assertEqual(after_eval["evaluator_flags_low_score"], "False")

    def test_second_judge_resolves_the_false_escalation(self):
        fixed = 0
        for t in self.false_escalations:
            result = evaluate_repair(self.graph, t, ActionType.SECOND_JUDGE, "evaluator")
            if result.accepted:
                fixed += 1
        self.assertEqual(fixed, len(self.false_escalations))

    def test_second_judge_is_not_appliable_when_evaluator_already_agrees(self):
        passing = next(t for t in self.traces if not t.final_outcome_failed)
        # not failing at all -> evaluate_repair short-circuits before apply_repair
        result = evaluate_repair(self.graph, passing, ActionType.SECOND_JUDGE, "evaluator")
        self.assertFalse(result.applied)

    def test_second_judge_does_not_resolve_a_genuine_content_failure(self):
        # bad content + evaluator correctly rejects it (an ordinary content
        # failure, not a false escalation) -- a second opinion on an evaluator
        # that already agrees with reality changes nothing.
        genuine_failures = [t for t in self.traces
                             if t.content_failed and not t.evaluator_false_acceptance
                             and t.failure_type != "evaluator_false_escalation"]
        self.assertTrue(genuine_failures)
        fixed_any = False
        for t in genuine_failures[:40]:
            result = evaluate_repair(self.graph, t, ActionType.SECOND_JUDGE, "evaluator")
            if result.accepted:
                fixed_any = True
        self.assertFalse(fixed_any, "SECOND_JUDGE must not resolve a genuine content failure "
                                     "the evaluator already correctly flagged")


if __name__ == "__main__":
    unittest.main()
