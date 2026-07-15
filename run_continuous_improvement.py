"""Run one batch of the continuous-improvement loop (generate traces, attribute
failures, propose and evaluate repairs, persist accept/reject stats to learning
memory) and print a before/after summary."""

from __future__ import annotations

import argparse
import json
import os

from graph import build_support_pipeline, ActionType
from simulate import generate_traces, wrong_variant_scenario, ScenarioConfig
from attribution import FailureAttributor
from optimizer import GraphOptimizer
from repair import replay_with_add_filter
from learning_memory import LearningMemory
from continuous_improvement import run_improvement_cycle
from report import build_dashboard

MEMORY_PATH = "learning_memory.json"


def _write_repair_decisions_jsonl(improvement, path: str = "repair_decisions.jsonl") -> None:
    with open(path, "w") as f:
        for d in improvement.decisions:
            row = {
                "trace_id": d.trace_id,
                "node_id": d.node_id,
                "failure_type": d.failure_type,
                "attribution_confidence": round(d.attribution_confidence, 3),
                "outcome": d.outcome,
                "policy_decision": (
                    None if d.policy_decision is None else {
                        "action": d.policy_decision.action,
                        "reason": d.policy_decision.reason,
                        "historical_attempts": d.policy_decision.historical_attempts,
                        "historical_accept_rate": round(d.policy_decision.historical_accept_rate, 3),
                        "exploration_or_exploitation": d.policy_decision.exploration_or_exploitation,
                        "confidence": d.policy_decision.confidence,
                    }
                ),
                "candidates_tried": [
                    {
                        "action": r.action.value,
                        "applied": r.applied,
                        "accepted": r.accepted,
                        "not_executable": r.not_executable,
                        "reason": r.reason,
                        "before_cost_usd": round(r.before_cost.api_cost_usd + r.before_cost.human_cost_usd, 4),
                        "after_cost_usd": round(r.after_cost.api_cost_usd + r.after_cost.human_cost_usd, 4),
                    }
                    for r in d.candidates_tried
                ],
                "winner": d.winner.action.value if d.winner else None,
            }
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reset-memory", action="store_true",
                         help="delete learning_memory.json before running, for a true cold start "
                              "(demo A in the README/prompt: 'Cold-start closed loop')")
    args = parser.parse_args()

    if args.reset_memory and os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)

    graph = build_support_pipeline()
    traces = generate_traces(graph, n=300, config=ScenarioConfig())
    attributor = FailureAttributor()

    failed = [t for t in traces if t.final_outcome_failed]
    # Live driver: a2p_scaffold (arXiv:2509.10401-style abduce+counterfactual
    # confirmation), chosen over binary_search after comparing both as the
    # actual repair-loop driver -- not just raw attribution accuracy -- across
    # 5 seeds x 300 traces (see tests/test_live_driver_choice.py):
    #   metric                a2p_scaffold   binary_search
    #   repair accept_rate         0.506          0.429
    #   unresolved_rate            0.130          0.161
    #   total cost after repair   $228           $288
    #   residual user-visible failures (of 227)   3              5
    # a2p_scaffold wins on every one of these full-loop metrics, not only on
    # attribution accuracy in isolation.
    attributions = [attributor.attribute_a2p_scaffold(t, graph) for t in traces]
    evaluable = [(t, a) for t, a in zip(traces, attributions)
                 if t.final_outcome_failed and t.ground_truth_node]
    agent_acc = (sum(1 for t, a in evaluable if a.responsible_node == t.ground_truth_node)
                 / len(evaluable)) if evaluable else 0.0

    method_rows = []
    already_computed = {"a2p_scaffold": {t.trace_id: a for t, a in zip(traces, attributions)}}
    for name, fn in [("all_at_once", attributor.attribute_all_at_once),
                      ("step_by_step", attributor.attribute_step_by_step),
                      ("binary_search", attributor.attribute_binary_search),
                      ("a2p_scaffold", lambda t: attributor.attribute_a2p_scaffold(t, graph))]:
        if name in already_computed:
            results = [already_computed[name][t.trace_id] for t, _ in evaluable]
        else:
            results = [fn(t) for t, _ in evaluable]
        correct = sum(1 for (t, _), r in zip(evaluable, results) if r.responsible_node == t.ground_truth_node)
        avg_calls = sum(r.judge_calls_used for r in results) / len(results) if results else 0.0
        method_rows.append((name, correct / len(evaluable) if evaluable else 0.0, avg_calls))

    optimizer = GraphOptimizer(graph)
    stats = optimizer.aggregate(traces, attributions)
    recs = optimizer.recommend(stats)

    replay = replay_with_add_filter(graph, wrong_variant_scenario(graph))

    memory = LearningMemory(MEMORY_PATH)
    cold_start = len(memory.all_entries()) == 0
    improvement = run_improvement_cycle(graph, traces, attributions, memory, ActionType)

    with open("learned_policy.json", "w") as f:
        json.dump(memory.as_policy_dict(), f, indent=2)
    _write_repair_decisions_jsonl(improvement)

    html = build_dashboard(
        graph, stats, recs, agent_acc, len(evaluable), method_rows, replay,
        learning_entries=memory.all_entries(),
        improvement_repairs=improvement.repairs,
        improvement_summary={
            "attempted": improvement.attempted,
            "accepted": improvement.accepted,
            "rejected": improvement.rejected,
            "human_before": improvement.human_escalations_before,
            "human_after": improvement.human_escalations_after,
            "cost_before": improvement.total_before_cost_usd,
            "cost_after": improvement.total_after_cost_usd,
        },
    )
    with open("workflow_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{len(failed)} failed traces this batch")
    print(f"mode: {'COLD START (memory was empty)' if cold_start else 'LEARNED (memory had prior history)'}")
    print(f"repairs attempted={improvement.attempted} accepted={improvement.accepted} "
          f"rejected={improvement.rejected} not_appliable={improvement.not_appliable} "
          f"human_review_selected={improvement.human_review_selected} unresolved={improvement.unresolved}")
    print(f"human escalations: {improvement.human_escalations_before} -> {improvement.human_escalations_after}")
    print(f"total cost: ${improvement.total_before_cost_usd:.2f} -> ${improvement.total_after_cost_usd:.2f}")
    print(f"learning memory (persisted to {MEMORY_PATH}):")
    for e in sorted(memory.all_entries(), key=lambda e: -e.attempts):
        print(f"  {e.node_id:12s} {e.failure_type:32s} {e.action:16s} "
              f"attempts={e.attempts:3d} accept_rate={e.accept_rate:.0%}")
    print("Artifacts written: workflow_dashboard.html, learning_memory.json, "
          "learned_policy.json, repair_decisions.jsonl")


if __name__ == "__main__":
    main()
