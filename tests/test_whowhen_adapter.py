import tests._pathfix
import json
import os
import tempfile
import unittest

from attribution import FailureAttributor, heuristic_judge
from whowhen_adapter import WhoWhenTraceSource, WhoWhenConfig, SyntheticTraceSource, TraceSource
from graph import build_support_pipeline


def _write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


FAILED_RECORD = {
    "question_ID": "q1", "question": "What is 2+2?", "ground_truth": "4",
    "is_correct": False, "mistake_agent": "Coder", "mistake_step": "1",
    "mistake_reason": "arithmetic slip",
    "history": [{"name": "Coder", "content": "Let's compute 2+2"},
                {"name": "Coder", "content": "2+2=5"}],
}

SUCCESS_RECORD = {
    "question_ID": "q2", "question": "Capital of France?", "ground_truth": "Paris",
    "is_correct": True, "mistake_agent": None, "mistake_step": None,
    "history": [{"name": "Researcher", "content": "Paris"}],
}


class TestWhoWhenTraceSourceIsATraceSource(unittest.TestCase):
    def test_implements_the_shared_interface(self):
        self.assertTrue(issubclass(WhoWhenTraceSource, TraceSource))
        self.assertTrue(issubclass(SyntheticTraceSource, TraceSource))


class TestJsonlLoading(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "sample.jsonl")
        _write_jsonl([FAILED_RECORD, SUCCESS_RECORD], self.path)
        self.traces = WhoWhenTraceSource(self.path).load()

    def test_loads_one_trace_per_record(self):
        self.assertEqual(len(self.traces), 2)

    def test_failed_record_maps_to_failed_trace_with_ground_truth(self):
        t = next(t for t in self.traces if t.trace_id == "q1")
        self.assertTrue(t.final_outcome_failed)
        self.assertEqual(t.ground_truth_node, "Coder")
        self.assertEqual(t.ground_truth_step, 1)

    def test_successful_record_has_no_ground_truth(self):
        t = next(t for t in self.traces if t.trace_id == "q2")
        self.assertFalse(t.final_outcome_failed)
        self.assertIsNone(t.ground_truth_node)
        self.assertIsNone(t.ground_truth_step)

    def test_speaker_extracted_from_name_key(self):
        t = next(t for t in self.traces if t.trace_id == "q1")
        self.assertEqual([s.node_id for s in t.steps], ["Coder", "Coder"])

    def test_ground_truth_never_leaks_into_rendered_symptoms(self):
        t = next(t for t in self.traces if t.trace_id == "q1")
        rendered = FailureAttributor._render_full_trace(t)
        self.assertNotIn("ground_truth", rendered.lower())
        self.assertNotIn("mistake_agent", rendered.lower())
        self.assertNotIn("root_cause", rendered.lower())


class TestJsonListAndNestedDictLoading(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_json_list_of_records(self):
        path = os.path.join(self.tmpdir, "list.json")
        with open(path, "w") as f:
            json.dump([FAILED_RECORD, SUCCESS_RECORD], f)
        traces = WhoWhenTraceSource(path).load()
        self.assertEqual(len(traces), 2)

    def test_json_dict_with_records_nested_under_a_split_key(self):
        path = os.path.join(self.tmpdir, "nested.json")
        with open(path, "w") as f:
            json.dump({"train": [FAILED_RECORD, SUCCESS_RECORD]}, f)
        traces = WhoWhenTraceSource(path).load()
        self.assertEqual(len(traces), 2)

    def test_json_dict_with_no_list_inside_raises(self):
        path = os.path.join(self.tmpdir, "bad.json")
        with open(path, "w") as f:
            json.dump({"not_a_list": "oops"}, f)
        with self.assertRaises(ValueError):
            WhoWhenTraceSource(path).load()

    def test_unrecognized_extension_raises(self):
        path = os.path.join(self.tmpdir, "data.csv")
        open(path, "w").close()
        with self.assertRaises(ValueError):
            WhoWhenTraceSource(path).load()


class TestSpeakerFallback(unittest.TestCase):
    def test_missing_speaker_key_falls_back_loudly(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "no_speaker.jsonl")
        record = dict(FAILED_RECORD)
        record["question_ID"] = "q_no_speaker"
        record["history"] = [{"content": "no speaker key in this turn"}]
        _write_jsonl([record], path)
        traces = WhoWhenTraceSource(path).load()
        t = traces[0]
        self.assertEqual(t.steps[0].node_id, "agent_step_0")

    def test_malformed_history_entries_are_skipped_not_crashed_on(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "malformed.jsonl")
        record = dict(FAILED_RECORD)
        record["question_ID"] = "q_malformed"
        record["history"] = ["not a dict", {"name": "Coder", "content": "ok"}, 42]
        _write_jsonl([record], path)
        traces = WhoWhenTraceSource(path).load()
        self.assertEqual(len(traces[0].steps), 1)


class TestNonNumericMistakeStep(unittest.TestCase):
    def test_non_numeric_mistake_step_does_not_crash_the_load(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "weird_step.jsonl")
        record = dict(FAILED_RECORD)
        record["question_ID"] = "q_weird_step"
        record["mistake_step"] = "not-a-number"
        _write_jsonl([record], path)
        traces = WhoWhenTraceSource(path).load()
        self.assertIsNone(traces[0].ground_truth_step)


class TestFailureAttributorWorksUnmodifiedAgainstWhoWhenOutput(unittest.TestCase):

    def test_attributor_runs_without_error_on_real_shaped_traces(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "sample.jsonl")
        _write_jsonl([FAILED_RECORD, SUCCESS_RECORD], path)
        traces = WhoWhenTraceSource(path).load()
        attributor = FailureAttributor(judge=heuristic_judge)
        for t in traces:
            result = attributor.attribute_all_at_once(t)
            self.assertEqual(result.method, "all_at_once")

    def test_heuristic_judge_mostly_returns_none_on_whowhen_shaped_data(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "sample.jsonl")
        _write_jsonl([FAILED_RECORD, SUCCESS_RECORD], path)
        traces = WhoWhenTraceSource(path).load()
        attributor = FailureAttributor(judge=heuristic_judge)
        results = [attributor.attribute_all_at_once(t) for t in traces]
        self.assertTrue(all(r.responsible_node in (None, "NONE") for r in results))


class TestSyntheticTraceSourceStillWorksUnchanged(unittest.TestCase):
    def test_wraps_generate_traces_directly(self):
        graph = build_support_pipeline()
        source = SyntheticTraceSource(graph, n=20, seed=3)
        traces = source.load()
        self.assertEqual(len(traces), 20)


if __name__ == "__main__":
    unittest.main()
