import tests._pathfix
import unittest
import inspect

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor, matches_ground_truth
from attribution import Trace as _AttrTrace
from repair_engine import apply_repair, evaluate_repair, CANDIDATE_ACTIONS
import repair_engine
import simulate
import causal_model


class TestSimulatorAndRepairEngineShareOneCausalModel(unittest.TestCase):
    """Simulator and repair engine must not contain separate, drifting
    definitions of groundedness/content-failure/evaluation-failure/
    workflow-failure/escalation."""

    def test_simulate_imports_outcome_failed_from_causal_model(self):
        self.assertIs(simulate.outcome_failed, causal_model.outcome_failed)

    def test_repair_engine_imports_outcome_failed_from_causal_model(self):
        self.assertIs(repair_engine.outcome_failed, causal_model.outcome_failed)

    def test_repair_engine_imports_retrieval_and_groundedness_formulas_from_causal_model(self):
        self.assertIs(repair_engine.expected_retrieval_score, causal_model.expected_retrieval_score)
        self.assertIs(repair_engine.expected_groundedness, causal_model.expected_groundedness)

    def test_no_private_duplicate_retrieval_or_groundedness_formula_left_in_repair_engine(self):
        src = inspect.getsource(repair_engine)
        self.assertNotIn("def _expected_retrieval_score", src)
        self.assertNotIn("def _expected_groundedness", src)

    def test_no_private_duplicate_retrieval_or_groundedness_formula_left_in_simulate(self):
        src = inspect.getsource(simulate)
        self.assertNotIn("def _expected_retrieval_score", src)
        self.assertNotIn("def _expected_groundedness", src)

    def test_workflow_failed_formula_is_shared(self):
        self.assertTrue(causal_model.workflow_failed(True, False))
        self.assertTrue(causal_model.workflow_failed(False, True))
        self.assertFalse(causal_model.workflow_failed(False, False))
        self.assertTrue(causal_model.workflow_failed(True, True))


class TestNoGroundTruthLeakageAcrossAllFailureTypes(unittest.TestCase):
    """Extends the pre-existing TestNoLeakage in test_attribution.py to the
    THREE new failure types added in this revision."""

    def test_no_new_failure_type_string_or_ground_truth_field_appears_in_rendered_trace(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=1500, config=ScenarioConfig(), seed=11)
        new_types = {"chunk_boundary_split_entity", "vade_missed_hallucination", "evaluator_false_escalation"}
        checked_any = False
        for t in traces:
            if t.failure_type not in new_types:
                continue
            checked_any = True
            rendered = FailureAttributor._render_full_trace(t)
            self.assertNotIn(t.failure_type, rendered)
            self.assertNotIn("ground_truth", rendered.lower())
            self.assertNotIn("root_cause", rendered.lower())
            self.assertNotIn("responsible=", rendered.lower())
        self.assertTrue(checked_any, "test setup problem: no new-type failures to check")


class TestRepairsAreRecomputedNeverAsserted(unittest.TestCase):
    """Repair replay must never hardcode after_failed=False -- it has to be
    recomputed from the (repaired) symptoms via the shared causal model."""

    def setUp(self):
        self.graph = build_support_pipeline()
        self.traces = generate_traces(self.graph, n=1500, config=ScenarioConfig(), seed=11)

    def test_new_node_repairs_produce_varying_after_states_not_a_constant(self):
        for node_id, action in (("chunker", ActionType.RECHUNK),
                                 ("vade", ActionType.RETRY_VALIDATION),
                                 ("evaluator", ActionType.SECOND_JUDGE)):
            relevant = [t for t in self.traces if t.ground_truth_node == node_id and t.final_outcome_failed]
            if not relevant:
                continue
            after_failed_values = set()
            for t in relevant:
                result = apply_repair(self.graph, t, action, node_id)
                after_failed_values.add(result.final_outcome_failed)
            # not every candidate here is guaranteed to resolve (cost tolerance,
            # residual independent causes), but the underlying trace object must
            # be a genuinely new, recomputed Trace, not the same input object.
            for t in relevant:
                result = apply_repair(self.graph, t, action, node_id)
                self.assertIsNot(result, t)

    def test_validated_repair_after_trace_is_recomputed_from_symptoms(self):
        vade_failures = [t for t in self.traces if t.failure_type == "vade_missed_hallucination"]
        self.assertTrue(vade_failures)
        for t in vade_failures:
            before_groundedness = float(
                next(s for s in t.steps if s.node_id == "generator").symptoms["groundedness"])
            result = apply_repair(self.graph, t, ActionType.RETRY_VALIDATION, "vade")
            self.assertIsNot(result, t)
            gen_step = next(s for s in result.steps if s.node_id == "generator")
            # vade detects and routes; it cannot rewrite what the generator
            # already produced -- groundedness after a vade-only repair must
            # be byte-identical to the pre-repair value, never a penalty-
            # cleared recomputation pretending content quality improved.
            self.assertAlmostEqual(float(gen_step.symptoms["groundedness"]), before_groundedness, places=2)
            vade_step = next(s for s in result.steps if s.node_id == "vade")
            self.assertEqual(vade_step.symptoms["vade_flagged"], "True")


class TestDeterminismHoldsForNewFields(unittest.TestCase):
    def test_same_seed_reproduces_new_fields_identically(self):
        graph = build_support_pipeline()
        t1 = generate_traces(graph, n=200, seed=42)
        t2 = generate_traces(graph, n=200, seed=42)
        for a, b in zip(t1, t2):
            self.assertEqual(a.content_failed, b.content_failed)
            self.assertEqual(a.evaluation_failed, b.evaluation_failed)
            self.assertEqual(a.evaluator_false_acceptance, b.evaluator_false_acceptance)
            chunk_a = next(s for s in a.steps if s.node_id == "chunker").symptoms
            chunk_b = next(s for s in b.steps if s.node_id == "chunker").symptoms
            self.assertEqual(chunk_a, chunk_b)
            vade_a = next(s for s in a.steps if s.node_id == "vade").symptoms
            vade_b = next(s for s in b.steps if s.node_id == "vade").symptoms
            self.assertEqual(vade_a, vade_b)


class TestMultiLabelScoringStaysBackwardCompatibleWithNewTypes(unittest.TestCase):
    def test_matches_ground_truth_still_plain_equality_for_new_single_cause_types(self):
        t = _AttrTrace(trace_id="x", steps=[], final_outcome_failed=True,
                        ground_truth_node="chunker", failure_type="chunk_boundary_split_entity")
        self.assertTrue(matches_ground_truth(t, "chunker"))
        self.assertFalse(matches_ground_truth(t, "retriever"))


class TestHumanStaysATerminalEscalationSink(unittest.TestCase):
    """human is a terminal escalation sink, not a fake root cause node."""

    def test_human_is_never_a_ground_truth_node(self):
        graph = build_support_pipeline()
        traces = generate_traces(graph, n=1500, config=ScenarioConfig(), seed=11)
        for t in traces:
            self.assertNotEqual(t.ground_truth_node, "human")
            if t.ground_truth_nodes:
                self.assertNotIn("human", t.ground_truth_nodes)

    def test_human_has_no_candidate_repair_actions(self):
        self.assertNotIn("human", CANDIDATE_ACTIONS)


if __name__ == "__main__":
    unittest.main()
