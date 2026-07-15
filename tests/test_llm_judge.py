import tests._pathfix
import sys
import tempfile
import os
import types
import unittest
from unittest.mock import MagicMock

from llm_judge import (
    make_claude_judge, ClaudeJudgeConfig, TransportResult, TransientError,
    CostLedger, PRICING_USD_PER_MTOK, DEFAULT_MODEL, bedrock_transport,
)
from attribution import FailureAttributor, Trace, TraceStep


def _fake_transport_factory(*, node="retriever", confidence=0.8, reason="low score",
                             input_tokens=100, output_tokens=20,
                             fail_times=0, invalid_times=0):
    state = {"calls": 0}

    def transport(system, prompt, config):
        state["calls"] += 1
        call_num = state["calls"]
        if call_num <= fail_times:
            raise TransientError("simulated rate limit")
        if call_num <= fail_times + invalid_times:
            return TransportResult("not_a_real_node", confidence, reason, input_tokens, output_tokens)
        return TransportResult(node, confidence, reason, input_tokens, output_tokens)

    transport.state = state
    return transport


class TestBasicJudgeCall(unittest.TestCase):
    def test_returns_correctly_formatted_verdict_string(self):
        transport = _fake_transport_factory(node="kb_builder", confidence=0.73, reason="stale kb")
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=False), transport=transport)
        verdict = judge("node=kb_builder kb_age_days=90")
        self.assertEqual(verdict, "RESPONSIBLE=kb_builder CONFIDENCE=0.73 REASON=stale_kb")

    def test_is_a_plain_callable_compatible_with_judgefn(self):
        transport = _fake_transport_factory()
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=False), transport=transport)
        self.assertTrue(callable(judge))
        result = judge("some prompt")
        self.assertIsInstance(result, str)

    def test_confidence_is_clamped_to_0_1(self):
        transport = _fake_transport_factory(confidence=1.7)
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=False), transport=transport)
        verdict = judge("node=x")
        self.assertIn("CONFIDENCE=1.00", verdict)


class TestCostLogging(unittest.TestCase):
    def test_cost_matches_pricing_table(self):
        ledger = CostLedger()
        transport = _fake_transport_factory(input_tokens=1_000_000, output_tokens=1_000_000)
        judge = make_claude_judge(ClaudeJudgeConfig(model=DEFAULT_MODEL, use_cache=False),
                                   transport=transport, ledger=ledger)
        judge("node=x")
        in_rate, out_rate = PRICING_USD_PER_MTOK[DEFAULT_MODEL]
        self.assertAlmostEqual(ledger.total_cost_usd(), in_rate + out_rate, places=6)

    def test_ledger_accumulates_across_multiple_distinct_calls(self):
        ledger = CostLedger()
        transport = _fake_transport_factory(input_tokens=100, output_tokens=50)
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=False), transport=transport, ledger=ledger)
        judge("node=a")
        judge("node=b")
        self.assertEqual(ledger.total_calls(), 2)
        self.assertGreater(ledger.total_cost_usd(), 0.0)

    def test_unknown_model_raises_rather_than_silently_reporting_zero_cost(self):
        transport = _fake_transport_factory()
        judge = make_claude_judge(ClaudeJudgeConfig(model="claude-made-up-model", use_cache=False),
                                   transport=transport)
        verdict = judge("node=x")
        self.assertIn("REASON=judge_error", verdict)


class TestCaching(unittest.TestCase):
    def test_second_identical_call_is_a_cache_hit_and_costs_nothing(self):
        ledger = CostLedger()
        transport = _fake_transport_factory()
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=True), transport=transport, ledger=ledger)
        judge("node=retriever retrieval_top1_score=0.2")
        judge("node=retriever retrieval_top1_score=0.2")
        self.assertEqual(transport.state["calls"], 1)
        self.assertEqual(ledger.total_calls(), 2)
        self.assertTrue(ledger.calls[1].cache_hit)
        self.assertEqual(ledger.calls[1].cost_usd, 0.0)

    def test_different_prompts_are_not_cached_together(self):
        transport = _fake_transport_factory()
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=True), transport=transport)
        judge("node=a x=1")
        judge("node=b x=2")
        self.assertEqual(transport.state["calls"], 2)

    def test_cache_disabled_calls_transport_every_time(self):
        transport = _fake_transport_factory()
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=False), transport=transport)
        judge("node=a")
        judge("node=a")
        self.assertEqual(transport.state["calls"], 2)

    def test_cache_persists_to_disk_and_survives_a_new_judge_instance(self):
        transport = _fake_transport_factory(node="generator", confidence=0.6, reason="test")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "judge_cache.json")
            config = ClaudeJudgeConfig(use_cache=True, cache_path=path)
            judge1 = make_claude_judge(config, transport=transport)
            judge1("node=generator groundedness=0.4")
            self.assertEqual(transport.state["calls"], 1)

            judge2 = make_claude_judge(config, transport=transport)
            verdict = judge2("node=generator groundedness=0.4")
            self.assertEqual(transport.state["calls"], 1, "should have hit the persisted cache, not called transport again")
            self.assertIn("RESPONSIBLE=generator", verdict)

    def test_different_prompt_versions_do_not_share_cache_entries(self):
        transport = _fake_transport_factory()
        c1 = ClaudeJudgeConfig(use_cache=True, prompt_version="v1")
        c2 = ClaudeJudgeConfig(use_cache=True, prompt_version="v2")
        judge1 = make_claude_judge(c1, transport=transport)
        judge2 = make_claude_judge(c2, transport=transport)
        judge1("node=a")
        judge2("node=a")
        self.assertEqual(transport.state["calls"], 2)


class TestRetry(unittest.TestCase):
    def test_recovers_after_transient_failures_within_retry_budget(self):
        transport = _fake_transport_factory(fail_times=2, node="retriever")
        judge = make_claude_judge(ClaudeJudgeConfig(max_retries=3, backoff_base_s=0.001, use_cache=False),
                                   transport=transport)
        verdict = judge("node=x")
        self.assertIn("RESPONSIBLE=retriever", verdict)
        self.assertEqual(transport.state["calls"], 3)

    def test_falls_back_to_none_when_retries_exhausted_does_not_raise(self):
        transport = _fake_transport_factory(fail_times=10)
        judge = make_claude_judge(ClaudeJudgeConfig(max_retries=2, backoff_base_s=0.001, use_cache=False),
                                   transport=transport)
        verdict = judge("node=x")
        self.assertIn("RESPONSIBLE=NONE", verdict)
        self.assertIn("REASON=judge_error", verdict)

    def test_logs_the_failed_call_with_an_error_message(self):
        ledger = CostLedger()
        transport = _fake_transport_factory(fail_times=10)
        judge = make_claude_judge(ClaudeJudgeConfig(max_retries=2, backoff_base_s=0.001, use_cache=False),
                                   transport=transport, ledger=ledger)
        judge("node=x")
        self.assertIsNotNone(ledger.calls[-1].error)


class TestSchemaValidation(unittest.TestCase):
    def test_unknown_node_id_triggers_one_corrective_retry_then_succeeds(self):
        transport = _fake_transport_factory(invalid_times=1, node="retriever")
        config = ClaudeJudgeConfig(use_cache=False, known_node_ids=frozenset({"retriever", "generator"}))
        judge = make_claude_judge(config, transport=transport)
        verdict = judge("node=retriever retrieval_top1_score=0.2")
        self.assertIn("RESPONSIBLE=retriever", verdict)
        self.assertEqual(transport.state["calls"], 2)

    def test_unknown_node_id_twice_falls_back_to_none(self):
        transport = _fake_transport_factory(invalid_times=5)
        config = ClaudeJudgeConfig(use_cache=False, known_node_ids=frozenset({"retriever"}))
        judge = make_claude_judge(config, transport=transport)
        verdict = judge("node=retriever")
        self.assertIn("RESPONSIBLE=NONE", verdict)

    def test_none_node_id_is_always_valid_even_with_known_node_ids_set(self):
        transport = _fake_transport_factory(node="NONE")
        config = ClaudeJudgeConfig(use_cache=False, known_node_ids=frozenset({"retriever"}))
        judge = make_claude_judge(config, transport=transport)
        verdict = judge("node=retriever")
        self.assertIn("RESPONSIBLE=NONE", verdict)

    def test_no_validation_when_known_node_ids_is_none(self):
        transport = _fake_transport_factory(node="anything_goes")
        config = ClaudeJudgeConfig(use_cache=False, known_node_ids=None)
        judge = make_claude_judge(config, transport=transport)
        verdict = judge("node=x")
        self.assertIn("RESPONSIBLE=anything_goes", verdict)


class TestIntegrationWithFailureAttributor(unittest.TestCase):

    def test_all_at_once_works_with_a_claude_judge(self):
        transport = _fake_transport_factory(node="generator", confidence=0.65, reason="low groundedness")
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=False), transport=transport)
        attributor = FailureAttributor(judge=judge)
        trace = Trace(trace_id="t1",
                       steps=[TraceStep(node_id="generator", symptoms={"groundedness": 0.3})],
                       final_outcome_failed=True)
        result = attributor.attribute_all_at_once(trace)
        self.assertEqual(result.responsible_node, "generator")
        self.assertAlmostEqual(result.confidence, 0.65, places=2)

    def test_step_by_step_calls_the_claude_judge_once_per_step(self):
        transport = _fake_transport_factory(node="NONE", confidence=0.1, reason="no signal")
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=False), transport=transport)
        attributor = FailureAttributor(judge=judge)
        trace = Trace(trace_id="t1",
                       steps=[TraceStep(node_id="a", symptoms={}),
                              TraceStep(node_id="b", symptoms={})],
                       final_outcome_failed=True)
        attributor.attribute_step_by_step(trace)
        self.assertEqual(transport.state["calls"], 2)


class TestBedrockTransportFactory(unittest.TestCase):

    def setUp(self):
        self.fake_client = MagicMock()
        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = MagicMock(return_value=self.fake_client)

        class FakeClientError(Exception):
            def __init__(self, error_response, operation_name=""):
                self.response = error_response
                super().__init__(str(error_response))

        fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
        fake_botocore_exceptions.ClientError = FakeClientError
        fake_botocore = types.ModuleType("botocore")
        fake_botocore.exceptions = fake_botocore_exceptions

        self._patched = {
            "boto3": sys.modules.get("boto3"),
            "botocore": sys.modules.get("botocore"),
            "botocore.exceptions": sys.modules.get("botocore.exceptions"),
        }
        sys.modules["boto3"] = fake_boto3
        sys.modules["botocore"] = fake_botocore
        sys.modules["botocore.exceptions"] = fake_botocore_exceptions
        self.FakeClientError = FakeClientError

    def tearDown(self):
        for name, mod in self._patched.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def test_successful_call_parses_converse_response(self):
        self.fake_client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"input": {"responsible_node": "retriever",
                                        "confidence": 0.9, "reason": "test"}}}
            ]}},
            "usage": {"inputTokens": 42, "outputTokens": 7},
        }
        transport = bedrock_transport("anthropic.claude-haiku-fake-v1:0")
        result = transport("sys", "node=retriever", ClaudeJudgeConfig(use_cache=False))
        self.assertEqual(result.responsible_node, "retriever")
        self.assertEqual(result.input_tokens, 42)
        self.assertEqual(result.output_tokens, 7)

    def test_throttling_error_maps_to_transient_error(self):
        self.fake_client.converse.side_effect = self.FakeClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}
        )
        transport = bedrock_transport("anthropic.claude-haiku-fake-v1:0")
        with self.assertRaises(TransientError):
            transport("sys", "node=x", ClaudeJudgeConfig(use_cache=False))

    def test_validation_error_maps_to_judge_error_not_transient(self):
        from llm_judge import JudgeError
        self.fake_client.converse.side_effect = self.FakeClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad request"}}
        )
        transport = bedrock_transport("anthropic.claude-haiku-fake-v1:0")
        with self.assertRaises(JudgeError):
            transport("sys", "node=x", ClaudeJudgeConfig(use_cache=False))


if __name__ == "__main__":
    unittest.main()

class TestDashboardGenerationDoesNotDuplicatePaidCalls(unittest.TestCase):

    def test_repeated_attribution_over_same_traces_only_pays_once_per_unique_prompt(self):
        transport = _fake_transport_factory(node="retriever", confidence=0.8)
        ledger = CostLedger()
        judge = make_claude_judge(ClaudeJudgeConfig(use_cache=True), transport=transport, ledger=ledger)
        attributor = FailureAttributor(judge=judge)

        traces = [
            Trace(trace_id=f"t{i}",
                  steps=[TraceStep(node_id="retriever", symptoms={"retrieval_top1_score": 0.3}),
                         TraceStep(node_id="generator", symptoms={"groundedness": 0.3})],
                  final_outcome_failed=True, ground_truth_node="retriever")
            for i in range(5)
        ]

        first_pass = [attributor.attribute_all_at_once(t) for t in traces]
        second_pass = [attributor.attribute_all_at_once(t) for t in traces]

        self.assertEqual(len(first_pass), len(second_pass))
        real_calls = sum(1 for c in ledger.calls if not c.cache_hit)
        cache_hits = sum(1 for c in ledger.calls if c.cache_hit)
        self.assertLessEqual(real_calls, len(traces))
        self.assertGreaterEqual(cache_hits, len(traces),
                                 "the second full pass over identical traces must be all cache hits")
        self.assertLessEqual(transport.state["calls"], len(traces),
                              "the underlying transport (the thing that actually costs money) "
                              "must not be invoked more than once per unique trace")
        self.assertLess(transport.state["calls"], len(traces) * 2,
                         "re-running attribution over the SAME batch a second time must not "
                         "roughly double the number of real transport calls")


if __name__ == "__main__":
    unittest.main()
