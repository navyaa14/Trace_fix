import tests._pathfix
import unittest

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor
from repair_engine import apply_repair, evaluate_repair, CANDIDATE_ACTIONS


class TestVadeFailuresAreReachable(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=300, config=ScenarioConfig(), seed=11)
        self.vade_failures = [t for t in self.traces if t.failure_type == "vade_missed_hallucination"]

    def test_vade_missed_hallucination_occurs(self):
        self.assertGreater(len(self.vade_failures), 0)
        for t in self.vade_failures:
            self.assertEqual(t.ground_truth_node, "vade")
            self.assertTrue(t.final_outcome_failed)
            self.assertTrue(t.content_failed)

    def test_vade_symptoms_are_always_visible(self):
        for t in self.traces[:50]:
            step = next(s for s in t.steps if s.node_id == "vade")
            self.assertIn("hallucination_present", step.symptoms)
            self.assertIn("vade_flagged", step.symptoms)
            self.assertIn("vade_confidence", step.symptoms)

    def test_validator_miss_is_distinct_from_generator_own_hallucination(self):
        # A vade_missed_hallucination trace has the generator's OWN mechanism
        # clean (hallucination_risk="none") -- the defect is only visible on the
        # vade node, not duplicated as a generator symptom. This is what makes it
        # a genuinely different failure MECHANISM, not a relabeling of
        # repeated_hallucination/unsupported_generation_transient.
        self.assertTrue(self.vade_failures)
        for t in self.vade_failures:
            gen_step = next(s for s in t.steps if s.node_id == "generator")
            vade_step = next(s for s in t.steps if s.node_id == "vade")
            self.assertEqual(gen_step.symptoms["hallucination_risk"], "none")
            self.assertEqual(vade_step.symptoms["hallucination_present"], "True")
            self.assertEqual(vade_step.symptoms["vade_flagged"], "False")

    def test_repeated_hallucination_and_transient_hallucination_still_exist(self):
        # vade's own mechanism must be ADDITIVE, not a replacement for the
        # generator's existing hallucination_risk-driven failure types.
        types = {t.failure_type for t in self.traces}
        self.assertIn("repeated_hallucination", types)
        self.assertIn("unsupported_generation_transient", types)

    def test_vade_ground_truth_does_not_leak_into_the_judge_prompt(self):
        self.assertTrue(self.vade_failures)
        for t in self.vade_failures:
            rendered = FailureAttributor._render_full_trace(t)
            self.assertNotIn("vade_missed_hallucination", rendered)
            self.assertNotIn("ground_truth", rendered.lower())

    def test_attributor_can_pick_vade_from_visible_evidence(self):
        self.assertTrue(self.vade_failures)
        attributor = FailureAttributor()
        hits = sum(1 for t in self.vade_failures
                   if attributor.attribute_all_at_once(t).responsible_node == "vade")
        self.assertGreater(hits, 0)


class TestVadeRepair(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()
        # n=1500 (not the headline n=300) so both containment shapes -- an
        # evaluator that independently already caught the same defect, and
        # one that didn't -- reliably both appear; at n=300 this particular
        # seed's small sample can land entirely on one side of that split.
        self.traces = generate_traces(self.graph, n=1500, config=ScenarioConfig(), seed=11)
        self.vade_failures = [t for t in self.traces if t.failure_type == "vade_missed_hallucination"]
        self.assertTrue(self.vade_failures, "need at least one vade-driven failure to test repair on")

    def test_vade_candidate_actions_are_registered(self):
        self.assertIn(ActionType.RETRY_VALIDATION, CANDIDATE_ACTIONS["vade"])
        self.assertIn(ActionType.LOWER_DETECTION_THRESHOLD, CANDIDATE_ACTIONS["vade"])

    def test_retry_validation_changes_detection_not_retrieval_or_entity_match(self):
        # The repair must change vade's OWN detection/routing state, not
        # magically improve generator groundedness via retrieval/entity_match --
        # those fields must be byte-identical before/after.
        for t in self.vade_failures:
            before_retr = next(s for s in t.steps if s.node_id == "retriever").symptoms
            result = apply_repair(self.graph, t, ActionType.RETRY_VALIDATION, "vade")
            self.assertIsNot(result, t)
            after_retr = next(s for s in result.steps if s.node_id == "retriever").symptoms
            self.assertEqual(before_retr["retrieval_top1_score"], after_retr["retrieval_top1_score"])
            self.assertEqual(before_retr["entity_match"], after_retr["entity_match"])
            vade_step = next(s for s in result.steps if s.node_id == "vade")
            self.assertEqual(vade_step.symptoms["vade_flagged"], "True")

    def test_retry_validation_resolves_the_vade_specific_failure(self):
        # vade catching a hallucination is containment (routing), not
        # correction -- it can never flip content_failed to False, since it
        # never rewrites what the generator produced. Its honest job is
        # resolving USER-VISIBLE failure (bad content reaching the user
        # unflagged). Traces where the evaluator had already independently
        # caught the same defect have nothing left for vade's repair to
        # observably change and are correctly left unresolved.
        already_contained = [t for t in self.vade_failures if not t.user_visible_failure]
        genuinely_user_visible = [t for t in self.vade_failures if t.user_visible_failure]
        self.assertTrue(genuinely_user_visible, "need at least one uncontained vade miss to test repair on")

        fixed = 0
        for t in genuinely_user_visible:
            result = evaluate_repair(self.graph, t, ActionType.RETRY_VALIDATION, "vade")
            self.assertTrue(result.accepted)
            self.assertTrue(result.after_failed, "content defect persists -- vade cannot rewrite content")
            self.assertFalse(result.after_trace.user_visible_failure,
                              "the user-visible component must be resolved")
            fixed += 1
        self.assertEqual(fixed, len(genuinely_user_visible))

        for t in already_contained:
            result = evaluate_repair(self.graph, t, ActionType.RETRY_VALIDATION, "vade")
            self.assertFalse(result.accepted, "nothing observable left for vade's repair to fix here")

    def test_retry_validation_is_not_appliable_once_already_flagged(self):
        t = self.vade_failures[0]
        once = apply_repair(self.graph, t, ActionType.RETRY_VALIDATION, "vade")
        twice = apply_repair(self.graph, once, ActionType.RETRY_VALIDATION, "vade")
        self.assertIs(twice, once)

    def test_retry_validation_does_not_fix_sticky_generator_hallucination(self):
        # sticky hallucination is the generator's own structural failure -- vade
        # catching it (evidence-only) must not by itself clear the penalty.
        sticky = [t for t in self.traces if t.failure_type == "repeated_hallucination"]
        self.assertTrue(sticky)
        fixed_any = False
        for t in sticky:
            result = evaluate_repair(self.graph, t, ActionType.RETRY_VALIDATION, "vade")
            if result.accepted:
                fixed_any = True
        self.assertFalse(fixed_any, "RETRY_VALIDATION must not resolve a structural, "
                                     "generator-rooted sticky hallucination")

    def test_a2p_scaffold_attributes_most_vade_failures_to_vade(self):
        attributor = FailureAttributor()

        # Where vade's containment repair has an observable effect (the
        # evaluator had NOT already independently caught the same defect),
        # a2p's counterfactual step has real signal and should reliably
        # confirm vade.
        observable = [t for t in self.vade_failures if t.user_visible_failure]
        self.assertTrue(observable, "need at least one uncontained vade miss with an observable outcome")
        correct_observable = sum(1 for t in observable
                                  if attributor.attribute_a2p_scaffold(t, self.graph).responsible_node == "vade")
        self.assertEqual(correct_observable, len(observable),
                          "a2p should reliably confirm vade whenever its containment repair has an "
                          "observable outcome to confirm against")

        # Where the evaluator already independently caught the same defect
        # (contained), vade's repair changes nothing observable -- a2p falls
        # back to static evidence alone there, which recovers some but not
        # all of these. The overall rate should still clear a modest floor.
        overall_correct = sum(1 for t in self.vade_failures
                               if attributor.attribute_a2p_scaffold(t, self.graph).responsible_node == "vade")
        self.assertGreaterEqual(overall_correct / len(self.vade_failures), 0.3)


if __name__ == "__main__":
    unittest.main()
