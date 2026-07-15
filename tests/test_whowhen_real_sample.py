import os
import tests._pathfix
import unittest

from whowhen_adapter import WhoWhenTraceSource, WhoWhenConfig
from attribution import FailureAttributor

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "whowhen_real_sample.jsonl")


class TestWhoWhenRealSample(unittest.TestCase):
    def setUp(self):
        self.source = WhoWhenTraceSource(_FIXTURE, WhoWhenConfig(split="Algorithm-Generated"))
        self.traces = self.source.load()

    def test_loads_all_real_rows(self):
        self.assertEqual(len(self.traces), 8)

    def test_real_ground_truth_fields_land_on_trace_not_symptoms(self):
        t = next(t for t in self.traces if t.trace_id ==
                 "4dbedc5e1a0205e14b7ff3ba89bce3060dab15d0ada3b7e1351a6f2aa8287aec")
        self.assertTrue(t.final_outcome_failed)
        self.assertEqual(t.ground_truth_node, "Verification_Expert")
        self.assertEqual(t.ground_truth_step, 1)
        for step in t.steps:
            self.assertNotIn("mistake", str(step.symptoms))
            self.assertNotIn("Verification_Expert", str(step.symptoms.get("content", "")))

    def test_question_id_formats_both_survive_unmodified(self):
        ids = {t.trace_id for t in self.traces}
        self.assertIn("4dbedc5e1a0205e14b7ff3ba89bce3060dab15d0ada3b7e1351a6f2aa8287aec", ids)
        self.assertIn("f3917a3d-1d17-4ee2-90c5-683b072218fe", ids)

    def test_speaker_extraction_falls_back_honestly_on_real_truncated_turns(self):
        t = self.traces[0]
        self.assertEqual(t.steps[0].node_id, "agent_step_0")

    def test_attributor_runs_against_real_data_without_crashing(self):
        attributor = FailureAttributor()
        for t in self.traces:
            result = attributor.attribute_all_at_once(t)
            self.assertEqual(result.method, "all_at_once")
            self.assertIn(result.responsible_node, (None, "NONE"))


if __name__ == "__main__":
    unittest.main()
