import unittest
import tests._pathfix

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, wrong_variant_scenario, ScenarioConfig, outcome_failed
from repair_engine import apply_repair, evaluate_repair, APPLIABLE_ACTIONS


class TestApplyRepairGeneralizes(unittest.TestCase):

    def setUp(self):
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=200, seed=11)
        self.failed = [t for t in self.traces if t.final_outcome_failed]
        self.assertGreater(len(self.failed), 5, "need failing traces to test repair on")

    def test_add_filter_applies_to_arbitrary_failed_traces(self):
        applied_any = False
        for t in self.failed:
            result = apply_repair(self.graph, t, ActionType.ADD_FILTER, "retriever")
            if result is not t:
                applied_any = True
                gen_step = next(s for s in result.steps if s.node_id == "generator")
                retr_step = next(s for s in result.steps if s.node_id == "retriever")
                self.assertEqual(retr_step.symptoms["entity_match"], "True")
        self.assertTrue(applied_any)

    def test_groundedness_is_recomputed_not_hardcoded(self):
        seen_values = set()
        for t in self.failed[:15]:
            result = apply_repair(self.graph, t, ActionType.ADD_FILTER, "retriever")
            if result is t:
                continue
            gen_step = next(s for s in result.steps if s.node_id == "generator")
            seen_values.add(gen_step.symptoms["groundedness"])
        self.assertGreater(len(seen_values), 1,
                            "repaired groundedness should vary with input, not be a constant")

    def test_showcase_scenario_still_works_via_generalized_engine(self):
        trace = wrong_variant_scenario(self.graph)
        result = apply_repair(self.graph, trace, ActionType.ADD_FILTER, "retriever")
        self.assertFalse(result.final_outcome_failed)

    def test_unappliable_action_is_a_noop(self):
        trace = wrong_variant_scenario(self.graph)
        result = apply_repair(self.graph, trace, ActionType.HUMAN_REVIEW, "retriever")
        self.assertIs(result, trace)

    def test_rechunk_freshens_kb_and_can_resolve_kb_driven_failures(self):
        stale_failures = [t for t in self.failed
                           if float(next(s for s in t.steps if s.node_id == "kb_builder")
                                     .symptoms["kb_age_days"]) > 60]
        self.assertTrue(stale_failures, "need at least one KB-staleness-driven failure")
        fixed_any = False
        for t in stale_failures:
            result = apply_repair(self.graph, t, ActionType.RECHUNK, "kb_builder")
            if not result.final_outcome_failed:
                fixed_any = True
        self.assertTrue(fixed_any)


class TestEvaluateRepairAcceptReject(unittest.TestCase):
    def setUp(self):
        self.graph = build_support_pipeline()

    def test_accepts_when_failure_resolved_within_cost_tolerance(self):
        trace = wrong_variant_scenario(self.graph)
        result = evaluate_repair(self.graph, trace, ActionType.ADD_FILTER, "retriever")
        self.assertTrue(result.applied)
        self.assertTrue(result.accepted)
        self.assertFalse(result.after_failed)
        self.assertLess(result.after_cost.human_cost_usd, result.before_cost.human_cost_usd)

    def test_rejects_non_appliable_action(self):
        trace = wrong_variant_scenario(self.graph)
        result = evaluate_repair(self.graph, trace, ActionType.CACHE, "retriever")
        self.assertFalse(result.applied)
        self.assertFalse(result.accepted)

    def test_rejects_when_trace_not_failing(self):
        traces = generate_traces(self.graph, n=50, seed=3)
        passing = next(t for t in traces if not t.final_outcome_failed)
        result = evaluate_repair(self.graph, passing, ActionType.ADD_FILTER, "retriever")
        self.assertFalse(result.applied)
        self.assertFalse(result.accepted)
        self.assertIn("not failing", result.reason)

    def test_rejects_when_action_wrong_for_node(self):
        trace = wrong_variant_scenario(self.graph)
        result = evaluate_repair(self.graph, trace, ActionType.RECHUNK, "retriever")
        self.assertFalse(result.applied)
        self.assertFalse(result.accepted)

    def test_accept_rate_varies_by_action_not_uniformly_accepting(self):
        traces = generate_traces(self.graph, n=300, seed=7)
        failed = [t for t in traces if t.final_outcome_failed]
        self.assertTrue(failed)

        def evaluate_for(action, node):
            return [evaluate_repair(self.graph, t, action, node) for t in failed]

        add_filter_results = evaluate_for(ActionType.ADD_FILTER, "retriever")
        retry_results = evaluate_for(ActionType.RETRY, "generator")

        add_filter_applied = [r for r in add_filter_results if r.applied]
        retry_applied = [r for r in retry_results if r.applied]
        self.assertTrue(add_filter_applied)
        self.assertTrue(retry_applied)

        # Behavior-based, not an arbitrary numeric floor/ceiling: RETRY's only
        # defined causal effect is clearing a transient generator hallucination
        # -- confirm it is appliable if and only if that's actually present,
        # regardless of what the trace's failure_type/root-cause label is.
        for t, r in zip(failed, retry_results):
            gen_step = next(s for s in t.steps if s.node_id == "generator")
            if gen_step.symptoms.get("hallucination_risk") == "transient":
                continue
            self.assertFalse(r.applied,
                              f"RETRY must not be appliable without a transient hallucination_risk "
                              f"(failure_type={t.failure_type}, node={t.ground_truth_node})")

        # ADD_FILTER is broadly appliable across many failure types (it always
        # attempts an entity/variant filter), so it covers a much larger share
        # of the total failing population than RETRY, which is narrowly scoped
        # to transient hallucination only.
        self.assertGreater(len(add_filter_applied), len(retry_applied) * 2,
                            "ADD_FILTER should be appliable to a much larger share of failures "
                            "than the narrowly-scoped RETRY")

        # Within the cases RETRY IS appliable to (transient hallucination), it
        # should be highly reliable -- that is its one real job.
        retry_accept_rate = sum(r.accepted for r in retry_applied) / len(retry_applied)
        self.assertGreater(retry_accept_rate, 0.6)


class TestNoSingleActionSolvesEverything(unittest.TestCase):

    def setUp(self):
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=500, seed=7)
        self.failed = [t for t in self.traces if t.final_outcome_failed]
        self.assertTrue(self.failed)

    def _accept_rate(self, action, node, failure_type):
        subset = [t for t in self.failed if t.failure_type == failure_type]
        if not subset:
            return None
        results = [evaluate_repair(self.graph, t, action, node) for t in subset]
        applied = [r for r in results if r.applied]
        return sum(r.accepted for r in applied) / len(applied) if applied else None

    def test_add_filter_does_not_solve_hallucination_failures(self):
        rate = self._accept_rate(ActionType.ADD_FILTER, "retriever", "repeated_hallucination")
        if rate is not None:
            self.assertLess(rate, 0.3, "ADD_FILTER shouldn't rescue generator-side hallucination")

    def test_retry_does_not_reliably_fix_sticky_hallucination(self):
        # sticky hallucination is structural, not noise -- RETRY has no
        # defined causal effect on it at all, so it is correctly reported as
        # NOT_APPLIABLE (rate=None) rather than "applied but rarely accepted".
        subset = [t for t in self.failed if t.failure_type == "repeated_hallucination"]
        self.assertTrue(subset)
        results = [evaluate_repair(self.graph, t, ActionType.RETRY, "generator") for t in subset]
        self.assertTrue(all(not r.applied and r.not_executable for r in results),
                         "RETRY must be not-appliable to structural sticky hallucination")
        rate = self._accept_rate(ActionType.RETRY, "generator", "repeated_hallucination")
        self.assertIsNone(rate)

    def test_retry_does_fix_transient_hallucination(self):
        rate = self._accept_rate(ActionType.RETRY, "generator", "unsupported_generation_transient")
        self.assertIsNotNone(rate)
        self.assertGreater(rate, 0.6, "transient/noise-driven hallucination should mostly clear on RETRY")

    def test_different_failure_types_prefer_different_actions(self):
        # RETRY is appliable (and reliable) for transient hallucination but
        # not appliable at all for sticky hallucination -- a stronger,
        # behavior-based distinction than comparing two success rates.
        transient_rate = self._accept_rate(ActionType.RETRY, "generator", "unsupported_generation_transient")
        sticky_rate = self._accept_rate(ActionType.RETRY, "generator", "repeated_hallucination")
        self.assertIsNotNone(transient_rate)
        self.assertGreater(transient_rate, 0.6)
        self.assertIsNone(sticky_rate, "RETRY should not even be appliable to sticky hallucination")


class TestRepairCompetition(unittest.TestCase):

    def setUp(self):
        self.graph = build_support_pipeline()

    def test_multiple_candidates_are_independently_measured(self):
        from repair_engine import generate_and_evaluate_candidates, CANDIDATE_ACTIONS, select_best_candidate

        traces = generate_traces(self.graph, n=100, seed=7)
        failed_retriever_traces = [t for t in traces if t.final_outcome_failed
                                    and t.failure_type == "wrong_entity_variant"]
        self.assertTrue(failed_retriever_traces)
        trace = failed_retriever_traces[0]

        results = generate_and_evaluate_candidates(self.graph, trace, "retriever",
                                                     CANDIDATE_ACTIONS["retriever"])
        self.assertEqual(len(results), len(CANDIDATE_ACTIONS["retriever"]))
        actions_tried = {r.action for r in results}
        self.assertEqual(actions_tried, set(CANDIDATE_ACTIONS["retriever"]))

        human_review_result = next(r for r in results if r.action == ActionType.HUMAN_REVIEW)
        self.assertTrue(human_review_result.not_executable)
        self.assertFalse(human_review_result.applied)

        winner = select_best_candidate(results)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.action, ActionType.ADD_FILTER)

    def test_costly_repair_can_be_rejected_despite_resolving_failure(self):
        traces = generate_traces(self.graph, n=200, seed=7)
        failed = [t for t in traces if t.final_outcome_failed and t.failure_type == "stale_knowledge_correct_entity"]
        self.assertTrue(failed)
        trace = failed[0]
        result = evaluate_repair(self.graph, trace, ActionType.RECHUNK, "kb_builder",
                                  cost_tolerance_usd=0.0000001)
        self.assertTrue(result.applied)
        self.assertFalse(result.after_failed, "RECHUNK should still resolve the underlying failure")
        self.assertFalse(result.accepted, "but must be rejected for exceeding the cost tolerance")
        self.assertIn("cost", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
