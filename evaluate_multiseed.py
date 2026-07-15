"""Multi-seed evaluation harness for TraceFix: measures attribution accuracy,
repair acceptance, and cost/latency deltas across default and adversarial
scenario configs, over several seeds, to distinguish stable results from
single-seed noise."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import tempfile
from dataclasses import dataclass, asdict

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, generate_adversarial_traces, ScenarioConfig
from attribution import FailureAttributor
from repair_engine import CANDIDATE_ACTIONS
from optimizer import GraphOptimizer
from learning_memory import LearningMemory
from continuous_improvement import run_improvement_cycle
from demo import accuracy_by_failure_type, accuracy_multi_label_by_failure_type


@dataclass
class SeedResult:
    seed: int
    n_traces: int
    n_failed: int
    n_evaluable: int
    attribution_accuracy: float
    repair_acceptance_rate: float
    human_review_rate: float
    avg_api_cost_usd: float
    avg_latency_penalty: float
    accuracy_by_failure_type: dict = None
    attribution_accuracy_a2p: float = 0.0
    attribution_accuracy_binary_search: float = 0.0
    accuracy_by_failure_type_a2p: dict = None
    accuracy_by_failure_type_multi_label: dict = None
    accuracy_by_failure_type_multi_label_a2p: dict = None


def _summary(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def run_one_seed(graph, seed: int, n: int, adversarial: bool = False) -> SeedResult:
    traces = (generate_adversarial_traces(graph, n=n, seed=seed) if adversarial
              else generate_traces(graph, n=n, config=ScenarioConfig(), seed=seed))
    attributor = FailureAttributor()
    attributions = [attributor.attribute_all_at_once(t) for t in traces]

    evaluable = [(t, a) for t, a in zip(traces, attributions) if t.final_outcome_failed and t.ground_truth_node]
    accuracy = (sum(1 for t, a in evaluable if a.responsible_node == t.ground_truth_node) / len(evaluable)
                if evaluable else 0.0)

    memory_path = f"{tempfile.gettempdir()}/multiseed_memory_seed{seed}_{'adv' if adversarial else 'std'}.json"
    if os.path.exists(memory_path):
        os.remove(memory_path)
    memory = LearningMemory(memory_path)
    improvement = run_improvement_cycle(graph, traces, attributions, memory, ActionType,
                                         rng=random.Random(seed))

    optimizer = GraphOptimizer(graph)
    stats = optimizer.aggregate(traces, attributions)
    avg_api = statistics.mean(s.observed_average_cost.api_cost_usd for s in stats.values()
                               if s.executions) if any(s.executions for s in stats.values()) else 0.0
    avg_lat = statistics.mean(s.observed_average_cost.latency_penalty_score for s in stats.values()
                               if s.executions) if any(s.executions for s in stats.values()) else 0.0

    failed_n = sum(1 for t in traces if t.final_outcome_failed)
    human_review_rate = improvement.human_review_selected / failed_n if failed_n else 0.0

    by_type_raw = accuracy_by_failure_type(attributor.attribute_all_at_once, traces)
    by_type = {ft: {"accuracy": round(acc, 4), "n": n} for ft, (acc, n) in by_type_raw.items()}

    a2p_fn = lambda t: attributor.attribute_a2p_scaffold(t, graph)
    a2p_correct = sum(1 for t, a in evaluable if a2p_fn(t).responsible_node == t.ground_truth_node)
    a2p_accuracy = a2p_correct / len(evaluable) if evaluable else 0.0
    by_type_a2p_raw = accuracy_by_failure_type(a2p_fn, traces)
    by_type_a2p = {ft: {"accuracy": round(acc, 4), "n": n} for ft, (acc, n) in by_type_a2p_raw.items()}

    bs_correct = sum(1 for t, a in evaluable
                      if attributor.attribute_binary_search(t).responsible_node == t.ground_truth_node)
    bs_accuracy = bs_correct / len(evaluable) if evaluable else 0.0

    by_type_ml_raw = accuracy_multi_label_by_failure_type(attributor.attribute_all_at_once, traces)
    by_type_ml = {ft: {"accuracy": round(acc, 4), "n": n} for ft, (acc, n) in by_type_ml_raw.items()}
    by_type_ml_a2p_raw = accuracy_multi_label_by_failure_type(a2p_fn, traces)
    by_type_ml_a2p = {ft: {"accuracy": round(acc, 4), "n": n} for ft, (acc, n) in by_type_ml_a2p_raw.items()}

    return SeedResult(
        seed=seed, n_traces=n, n_failed=failed_n, n_evaluable=len(evaluable),
        attribution_accuracy=round(accuracy, 4),
        repair_acceptance_rate=round(improvement.accept_rate, 4),
        human_review_rate=round(human_review_rate, 4),
        avg_api_cost_usd=round(avg_api, 6),
        avg_latency_penalty=round(avg_lat, 4),
        accuracy_by_failure_type=by_type,
        attribution_accuracy_a2p=round(a2p_accuracy, 4),
        attribution_accuracy_binary_search=round(bs_accuracy, 4),
        accuracy_by_failure_type_a2p=by_type_a2p,
        accuracy_by_failure_type_multi_label=by_type_ml,
        accuracy_by_failure_type_multi_label_a2p=by_type_ml_a2p,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=8, help="number of seeds to evaluate")
    parser.add_argument("--n", type=int, default=300, help="traces per seed")
    parser.add_argument("--first-seed", type=int, default=100,
                         help="starting seed (kept distinct from the deterministic demo's seed=7)")
    args = parser.parse_args()

    graph = build_support_pipeline()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))

    print(f"Running {len(seeds)} seeds x {args.n} traces (default ScenarioConfig)...")
    default_results = [run_one_seed(graph, s, args.n, adversarial=False) for s in seeds]

    print(f"Running {len(seeds)} seeds x {args.n} traces (ADVERSARIAL simulator profile, "
          f"independent evaluation pressure)...")
    adversarial_results = [run_one_seed(graph, s, args.n, adversarial=True) for s in seeds]

    def _aggregate_by_failure_type(results: list, key: str = "accuracy_by_failure_type") -> dict:
        by_type: dict[str, list] = {}
        for r in results:
            for ft, d in (getattr(r, key) or {}).items():
                by_type.setdefault(ft, []).append(d["accuracy"])
        return {ft: _summary(vals) for ft, vals in by_type.items()}

    report = {
        "seeds_evaluated": seeds,
        "n_traces_per_seed": args.n,
        "default_profile": {
            "attribution_accuracy": _summary([r.attribution_accuracy for r in default_results]),
            "attribution_accuracy_a2p_scaffold": _summary([r.attribution_accuracy_a2p for r in default_results]),
            "attribution_accuracy_binary_search": _summary(
                [r.attribution_accuracy_binary_search for r in default_results]),
            "repair_acceptance_rate": _summary([r.repair_acceptance_rate for r in default_results]),
            "human_review_rate": _summary([r.human_review_rate for r in default_results]),
            "avg_api_cost_usd": _summary([r.avg_api_cost_usd for r in default_results]),
            "avg_latency_penalty": _summary([r.avg_latency_penalty for r in default_results]),
            "n_evaluated_failed_traces": _summary([r.n_evaluable for r in default_results]),
            "accuracy_by_failure_type_across_seeds": _aggregate_by_failure_type(default_results),
            "accuracy_by_failure_type_across_seeds_a2p_scaffold": _aggregate_by_failure_type(
                default_results, key="accuracy_by_failure_type_a2p"),
            "accuracy_by_failure_type_across_seeds_multi_label": _aggregate_by_failure_type(
                default_results, key="accuracy_by_failure_type_multi_label"),
            "accuracy_by_failure_type_across_seeds_multi_label_a2p_scaffold": _aggregate_by_failure_type(
                default_results, key="accuracy_by_failure_type_multi_label_a2p"),
            "per_seed": [asdict(r) for r in default_results],
        },
        "adversarial_profile": {
            "note": "second, independently-parameterized simulator profile (simulate.generate_adversarial_traces) "
                    "-- attribution accuracy here is the honest out-of-distribution check for the heuristic judge, "
                    "not expected to match the default profile's number.",
            "attribution_accuracy": _summary([r.attribution_accuracy for r in adversarial_results]),
            "repair_acceptance_rate": _summary([r.repair_acceptance_rate for r in adversarial_results]),
            "human_review_rate": _summary([r.human_review_rate for r in adversarial_results]),
            "per_seed": [asdict(r) for r in adversarial_results],
        },
    }

    with open("multiseed_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    d = report["default_profile"]
    a = report["adversarial_profile"]
    print("\n" + "=" * 78)
    print("MULTI-SEED SUMMARY (default synthetic profile)")
    print("=" * 78)
    print(f"attribution_accuracy   : mean={d['attribution_accuracy']['mean']:.1%}  "
          f"std={d['attribution_accuracy']['std']:.3f}  "
          f"min={d['attribution_accuracy']['min']:.1%}  max={d['attribution_accuracy']['max']:.1%}")
    print(f"attribution_accuracy (a2p_scaffold, arXiv:2509.10401): "
          f"mean={d['attribution_accuracy_a2p_scaffold']['mean']:.1%}  "
          f"std={d['attribution_accuracy_a2p_scaffold']['std']:.3f}")
    bs_mean = d['attribution_accuracy_binary_search']['mean']
    a2p_mean = d['attribution_accuracy_a2p_scaffold']['mean']
    comparison = "beats" if a2p_mean > bs_mean else ("ties" if a2p_mean == bs_mean else "does NOT beat")
    print(f"attribution_accuracy (binary_search): mean={bs_mean:.1%}  "
          f"std={d['attribution_accuracy_binary_search']['std']:.3f}")
    print(f"-> a2p_scaffold {comparison} binary_search on this mean, multi-seed number.")
    print(f"repair_acceptance_rate : mean={d['repair_acceptance_rate']['mean']:.1%}  "
          f"std={d['repair_acceptance_rate']['std']:.3f}")
    print(f"human_review_rate      : mean={d['human_review_rate']['mean']:.1%}")
    print(f"n_evaluated_failed_traces (per seed): mean={d['n_evaluated_failed_traces']['mean']:.0f}")

    print("\n" + "=" * 78)
    print("ROOT-CAUSE CHECK: single-label vs. multi-label scoring on")
    print("'multiple_simultaneous_failures', mean across seeds")
    print("=" * 78)
    single = d["accuracy_by_failure_type_across_seeds"].get("multiple_simultaneous_failures")
    multi = d["accuracy_by_failure_type_across_seeds_multi_label"].get("multiple_simultaneous_failures")
    single_a2p = d["accuracy_by_failure_type_across_seeds_a2p_scaffold"].get("multiple_simultaneous_failures")
    multi_a2p = d["accuracy_by_failure_type_across_seeds_multi_label_a2p_scaffold"].get("multiple_simultaneous_failures")
    if single and multi:
        print(f"all_at_once   single-label: mean={single['mean']:.1%}  std={single['std']:.3f}")
        print(f"all_at_once   multi-label : mean={multi['mean']:.1%}  std={multi['std']:.3f}")
    if single_a2p and multi_a2p:
        print(f"a2p_scaffold  single-label: mean={single_a2p['mean']:.1%}  std={single_a2p['std']:.3f}")
        print(f"a2p_scaffold  multi-label : mean={multi_a2p['mean']:.1%}  std={multi_a2p['std']:.3f}")
    print("A large single-label-to-multi-label gap, reproducible across seeds (not just one "
          "seed's draw), means most of this category's low headline score is the evaluation "
          "crediting only one of two genuinely true root causes -- not the attributor guessing "
          "wrong.")
    print("\n" + "=" * 78)
    print("ADVERSARIAL PROFILE (independent evaluation pressure)")
    print("=" * 78)
    print(f"attribution_accuracy   : mean={a['attribution_accuracy']['mean']:.1%}  "
          f"std={a['attribution_accuracy']['std']:.3f}")
    delta = d['attribution_accuracy']['mean'] - a['attribution_accuracy']['mean']
    print(f"accuracy delta (default - adversarial): {delta:+.1%} "
          f"({'heuristic degrades out-of-distribution, as expected' if delta > 0.02 else 'roughly stable across profiles'})")
    print("\nWritten: multiseed_metrics.json")


if __name__ == "__main__":
    main()
