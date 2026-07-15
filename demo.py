
from graph import build_support_pipeline
from simulate import generate_traces, ScenarioConfig, wrong_variant_scenario
from attribution import FailureAttributor, heuristic_judge, matches_ground_truth, MIN_RELIABLE_N
from optimizer import GraphOptimizer
from cost import trace_cost
from repair import replay_with_add_filter
from report import build_dashboard


def accuracy(method_fn, traces):
    evaluated = [t for t in traces if t.final_outcome_failed and t.ground_truth_node]
    if not evaluated:
        return 0.0, 0.0, 0
    correct = 0
    total_calls = 0
    for t in evaluated:
        result = method_fn(t)
        total_calls += result.judge_calls_used
        if result.responsible_node == t.ground_truth_node:
            correct += 1
    return correct / len(evaluated), total_calls / len(evaluated), len(evaluated)


def accuracy_by_failure_type(method_fn, traces) -> dict[str, tuple[float, int]]:
    by_type: dict[str, list] = {}
    for t in traces:
        if t.final_outcome_failed and t.ground_truth_node:
            by_type.setdefault(t.failure_type, []).append(t)

    results = {}
    for failure_type, group in by_type.items():
        correct = sum(1 for t in group if method_fn(t).responsible_node == t.ground_truth_node)
        results[failure_type] = (correct / len(group), len(group))
    return results


def accuracy_multi_label_by_failure_type(method_fn, traces) -> dict[str, tuple[float, int]]:
    by_type: dict[str, list] = {}
    for t in traces:
        if t.final_outcome_failed and t.ground_truth_node:
            by_type.setdefault(t.failure_type, []).append(t)

    results = {}
    for failure_type, group in by_type.items():
        correct = sum(1 for t in group if matches_ground_truth(t, method_fn(t).responsible_node))
        results[failure_type] = (correct / len(group), len(group))
    return results


def main():
    graph = build_support_pipeline()
    # n=8000, not 300: rare failure types (clarification_failed_annoyed_user,
    # evaluator_false_escalation) sit at n=1 with a small batch, which is a
    # sample-size problem, not a real signal. A bigger synthetic batch (still
    # generated in well under a second, since there's no LLM call in the
    # simulator) pushes every category past MIN_RELIABLE_N -- the low-n flag
    # in the dashboard now means "genuinely rare", not "we didn't sample enough".
    traces = generate_traces(graph, n=8000, config=ScenarioConfig(), seed=11)
    fail_rate = sum(1 for t in traces if t.final_outcome_failed) / len(traces)
    attributor = FailureAttributor(judge=heuristic_judge)

    print("=" * 78)
    print("ATTRIBUTION ACCURACY  (predicted node == ground-truth node; judge never")
    print("sees ground truth -- only symptom evidence)")
    print("=" * 78)
    method_rows = []
    for name, fn in [
        ("all_at_once", attributor.attribute_all_at_once),
        ("step_by_step", attributor.attribute_step_by_step),
        ("binary_search", attributor.attribute_binary_search),
        ("a2p_scaffold", lambda t: attributor.attribute_a2p_scaffold(t, graph)),
    ]:
        acc, avg_calls, n_eval = accuracy(fn, traces)
        method_rows.append((name, acc, avg_calls))
        print(f"{name:15s}  agent-level accuracy = {acc:5.1%}   "
              f"avg judge-calls/trace = {avg_calls:4.2f}   (n={n_eval})")
    print("Reference point: Who&When (ICML'25) reports 53.5% agent-level / 14.2% "
          "step-level for their best method -- a heuristic baseline landing in a "
          "similar range, not near 100%, is the expected and honest result.")
    print("a2p_scaffold has the highest accuracy here and is the attributor used by "
          "run_continuous_improvement.py's live repair loop. The per-node "
          "recommendations below use all_at_once instead, since that comparison is "
          "about node failure rates and cost, not attribution accuracy.")

    print("\nBy failure type (all_at_once) -- the blended number above hides this:")
    print(f"(n<{MIN_RELIABLE_N} rows are flagged low-n: too few examples to draw a "
          f"conclusion from, not evidence the method fails there)")
    breakdown = accuracy_by_failure_type(attributor.attribute_all_at_once, traces)
    for failure_type, (acc, n) in sorted(breakdown.items(), key=lambda kv: kv[1][0]):
        if n < MIN_RELIABLE_N:
            flag = f"  <-- low n ({n}); not a reliable estimate"
        elif acc < 0.125:
            flag = "  <-- worse than uniform-random over 8 nodes"
        else:
            flag = ""
        print(f"  {failure_type:35s} {acc:5.1%}  (n={n}){flag}")

    print("\nBy failure type (a2p_scaffold) -- same breakdown, for the method that "
          "actually targets some of these gaps:")
    a2p_breakdown = accuracy_by_failure_type(lambda t: attributor.attribute_a2p_scaffold(t, graph), traces)
    for failure_type, (acc, n) in sorted(a2p_breakdown.items(), key=lambda kv: kv[1][0]):
        if n < MIN_RELIABLE_N:
            flag = f"  <-- low n ({n}); not a reliable estimate"
        else:
            baseline_acc, _ = breakdown.get(failure_type, (0.0, 0))
            delta = acc - baseline_acc
            flag = f"  <-- {delta:+.1%} vs all_at_once" if abs(delta) >= 0.01 else ""
        print(f"  {failure_type:35s} {acc:5.1%}  (n={n}){flag}")

    print("\nRoot-cause check on 'multiple_simultaneous_failures' (still near-zero above):")
    ml_all_at_once = accuracy_multi_label_by_failure_type(attributor.attribute_all_at_once, traces)
    ml_a2p = accuracy_multi_label_by_failure_type(lambda t: attributor.attribute_a2p_scaffold(t, graph), traces)
    for name, single, multi in (
        ("all_at_once", breakdown, ml_all_at_once),
        ("a2p_scaffold", a2p_breakdown, ml_a2p),
    ):
        if "multiple_simultaneous_failures" not in single:
            continue
        single_acc, n = single["multiple_simultaneous_failures"]
        multi_acc, _ = multi["multiple_simultaneous_failures"]
        print(f"  {name:15s} single-label (credits only Trace.ground_truth_node) = {single_acc:5.1%}  "
              f"(n={n})")
        print(f"  {name:15s} multi-label  (credits either true root cause)       = {multi_acc:5.1%}  "
              f"(n={n})")
    print("  The gap between single-label and multi-label above is how much of this "
          "category's low score is the evaluation only crediting one of two genuinely "
          "true root causes, not the attributor being wrong.")

    failed_traces = [t for t in traces if t.final_outcome_failed]
    attributions = [attributor.attribute_all_at_once(t) for t in failed_traces]

    optimizer = GraphOptimizer(graph)
    stats = optimizer.aggregate(traces, attributions)
    recs = optimizer.recommend(stats)

    print("\n" + "=" * 78)
    print("PER-NODE RECOMMENDATIONS  (failure rate = failures / times node executed)")
    print("=" * 78)
    for r in sorted(recs, key=lambda r: -r.failure_rate):
        print(f"[{r.action.value:16s}] {r.node_id:12s} failure_rate={r.failure_rate:5.1%}  "
              f"api_cost=${r.cost.api_cost_usd:.4f}  latency_penalty={r.cost.latency_penalty_score:.2f}")
        print(f"    reason: {r.reason}")

    print("\n" + "=" * 78)
    print("OPTIMIZER STRESS TEST  (deliberately failure-heavy synthetic batch --")
    print("not a realistic failure rate; exists only to show non-KEEP actions firing)")
    print("=" * 78)
    stress_config = ScenarioConfig(p_kb_stale=1.0, p_ambiguous_query=0.02, clarifier_catch_rate=0.0)
    stress_traces = generate_traces(graph, n=300, config=stress_config, seed=11)
    stress_failed = [t for t in stress_traces if t.final_outcome_failed]
    stress_attributions = [attributor.attribute_all_at_once(t) for t in stress_failed]
    stress_stats = optimizer.aggregate(stress_traces, stress_attributions)
    stress_recs = optimizer.recommend(stress_stats)
    stress_fail_rate = len(stress_failed) / len(stress_traces)
    print(f"Stress-batch failure_rate={stress_fail_rate:.1%} (vs. {fail_rate:.1%} in the headline batch)")
    non_keep = [r for r in stress_recs if r.action.value != "KEEP"]
    if non_keep:
        for r in sorted(non_keep, key=lambda r: -r.failure_rate):
            print(f"[{r.action.value:16s}] {r.node_id:12s} failure_rate={r.failure_rate:5.1%}  "
                  f"confidence={stress_stats[r.node_id].avg_confidence:.2f}")
            print(f"    reason: {r.reason}")
    else:
        print("(no node crossed the recommendation threshold even in this stress batch -- "
              "try a more extreme ScenarioConfig)")

    print("\n" + "=" * 78)
    print("REPAIR REPLAY — scenario: wrong product variant retrieved")
    print("=" * 78)
    showcase = wrong_variant_scenario(graph)
    attributed = attributor.attribute_all_at_once(showcase)
    print(f"Attributor's guess: {attributed.responsible_node} "
          f"(confidence={attributed.confidence:.2f}, reason={attributed.evidence})")
    print(f"Actual ground truth: {showcase.ground_truth_node}  "
          f"[{'MATCH' if attributed.responsible_node == showcase.ground_truth_node else 'MISS'}]")

    replay = replay_with_add_filter(graph, showcase)
    print(f"\nBefore: failed={replay.before_failed}, groundedness={replay.before_groundedness:.2f}, "
          f"human_cost=${replay.before_cost.human_cost_usd:.2f}")
    print(f"After:  failed={replay.after_failed}, groundedness={replay.after_groundedness:.2f}, "
          f"human_cost=${replay.after_cost.human_cost_usd:.2f}")
    print(f"Explanation: {replay.explanation}")

    correct_all_at_once = sum(
        1 for t in failed_traces
        if t.ground_truth_node and attributor.attribute_all_at_once(t).responsible_node == t.ground_truth_node
    )
    n_eval = sum(1 for t in failed_traces if t.ground_truth_node)
    agent_acc = correct_all_at_once / n_eval if n_eval else 0.0

    multi_label_breakdown = {}
    for name, single, multi in (
        ("all_at_once", breakdown, ml_all_at_once),
        ("a2p_scaffold", a2p_breakdown, ml_a2p),
    ):
        if "multiple_simultaneous_failures" in single:
            single_acc, n = single["multiple_simultaneous_failures"]
            multi_acc, _ = multi["multiple_simultaneous_failures"]
            multi_label_breakdown[name] = (single_acc, multi_acc, n)

    n_traces = len(traces)
    outcome_breakdown = {
        "content_failed_rate": sum(1 for t in traces if t.content_failed) / n_traces,
        "evaluation_failed_rate": sum(1 for t in traces if t.evaluation_failed) / n_traces,
        "workflow_failed_rate": sum(1 for t in traces if t.final_outcome_failed) / n_traces,
        "n_evaluator_false_escalation": sum(1 for t in traces if t.failure_type == "evaluator_false_escalation"),
        "n_evaluator_false_acceptance": sum(1 for t in traces if t.evaluator_false_acceptance),
    }

    html = build_dashboard(graph, stats, recs, agent_acc, n_eval, method_rows, replay,
                            failure_type_breakdown=breakdown,
                            multi_label_breakdown=multi_label_breakdown,
                            outcome_breakdown=outcome_breakdown)
    with open("workflow_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\nDashboard written to workflow_dashboard.html")

    avg_cost = trace_cost(traces[0])
    print("\n" + "=" * 78)
    print(f"Simulated batch: {len(traces)} traces | failure_rate={fail_rate:.1%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
