import tests._pathfix
import os
import random
import statistics
import tempfile
import unittest

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor
from continuous_improvement import run_improvement_cycle
from learning_memory import LearningMemory


def _run_full_loop(graph, attributor_fn, seed, label, n=300):
    traces = generate_traces(graph, n=n, config=ScenarioConfig(), seed=seed)
    attributions = [attributor_fn(t) for t in traces]
    path = f"{tempfile.gettempdir()}/test_live_driver_{label}_{seed}.json"
    if os.path.exists(path):
        os.remove(path)
    memory = LearningMemory(path)
    report = run_improvement_cycle(graph, traces, attributions, memory, ActionType, rng=random.Random(seed))

    failed = [t for t in traces if t.final_outcome_failed]
    id_to_trace = {t.trace_id: t for t in failed}
    uvf_before = sum(1 for t in failed if t.user_visible_failure)
    uvf_after = 0
    for d in report.decisions:
        t = id_to_trace[d.trace_id]
        if d.winner is not None and d.winner.after_trace is not None:
            uvf_after += 1 if d.winner.after_trace.user_visible_failure else 0
        else:
            uvf_after += 1 if t.user_visible_failure else 0

    return {
        "accept_rate": report.accept_rate,
        "unresolved_rate": report.unresolved / report.attempted if report.attempted else 0.0,
        "cost_after": report.total_after_cost_usd,
        "uvf_before": uvf_before,
        "uvf_after": uvf_after,
    }


class TestLiveDriverChoiceIsEvidenceBased(unittest.TestCase):
    """Compare a2p_scaffold vs binary_search as the live repair-loop driver
    using repair acceptance, unresolved rate, cost, and user-visible failure
    reduction -- not attribution accuracy alone. See run_continuous_improvement.py's
    comment for the single measured comparison this locks in directionally
    across 5 seeds."""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_support_pipeline()
        cls.attributor = FailureAttributor()
        seeds = range(200, 205)
        a2p_fn = lambda t: cls.attributor.attribute_a2p_scaffold(t, cls.graph)
        bs_fn = cls.attributor.attribute_binary_search
        cls.a2p_runs = [_run_full_loop(cls.graph, a2p_fn, s, "a2p") for s in seeds]
        cls.bs_runs = [_run_full_loop(cls.graph, bs_fn, s, "bs") for s in seeds]

    def test_a2p_scaffold_has_higher_mean_accept_rate(self):
        a2p_mean = statistics.mean(r["accept_rate"] for r in self.a2p_runs)
        bs_mean = statistics.mean(r["accept_rate"] for r in self.bs_runs)
        self.assertGreater(a2p_mean, bs_mean)

    def test_a2p_scaffold_has_lower_mean_unresolved_rate(self):
        a2p_mean = statistics.mean(r["unresolved_rate"] for r in self.a2p_runs)
        bs_mean = statistics.mean(r["unresolved_rate"] for r in self.bs_runs)
        self.assertLess(a2p_mean, bs_mean)

    def test_a2p_scaffold_leaves_fewer_residual_user_visible_failures(self):
        a2p_uvf_after = sum(r["uvf_after"] for r in self.a2p_runs)
        bs_uvf_after = sum(r["uvf_after"] for r in self.bs_runs)
        # both should substantially reduce user-visible failures from baseline
        a2p_uvf_before = sum(r["uvf_before"] for r in self.a2p_runs)
        self.assertLess(a2p_uvf_after, a2p_uvf_before)
        self.assertLessEqual(a2p_uvf_after, bs_uvf_after)

    def test_a2p_scaffold_has_lower_total_post_repair_cost(self):
        a2p_cost = sum(r["cost_after"] for r in self.a2p_runs)
        bs_cost = sum(r["cost_after"] for r in self.bs_runs)
        self.assertLess(a2p_cost, bs_cost)

    def test_run_continuous_improvement_uses_the_chosen_driver(self):
        import run_continuous_improvement as rci
        import inspect
        src = inspect.getsource(rci.main)
        self.assertIn("attribute_a2p_scaffold", src)


if __name__ == "__main__":
    unittest.main()
