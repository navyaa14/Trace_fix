
import os
import sys

from graph import build_support_pipeline
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor
from llm_judge import make_claude_judge, ClaudeJudgeConfig

from demo import accuracy


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. This script makes real, billed API "
              "calls, so it refuses to guess at a key. Set it and re-run:\n"
              "  export ANTHROPIC_API_KEY=sk-ant-...\n\n"
              "To see the offline, zero-dependency version instead, run demo.py.")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package isn't installed. Run:\n"
              "  pip install -r requirements.txt --break-system-packages")
        sys.exit(1)

    graph = build_support_pipeline()
    traces = generate_traces(graph, n=300, config=ScenarioConfig(), seed=11)

    config = ClaudeJudgeConfig(
        model="claude-haiku-4-5-20251001",
        known_node_ids=frozenset(graph.nodes.keys()),
        use_cache=True,
        cache_path="llm_judge_cache.json",
    )
    judge = make_claude_judge(config)
    attributor = FailureAttributor(judge=judge)

    print("=" * 78)
    print(f"REAL LLM JUDGE ({config.model}) -- attribution accuracy vs. ground truth")
    print("=" * 78)
    acc, avg_calls, n_eval = accuracy(attributor.attribute_all_at_once, traces)
    print(f"all_at_once  agent-level accuracy = {acc:5.1%}   n={n_eval}")
    print("\nReference point: Who&When (ICML'25) reports 53.5% agent-level accuracy "
          "for their best method on their own dataset -- these are different traces "
          "and a much smaller/cheaper judge model, so treat this as a sanity check "
          "that a real Claude call beats the heuristic, not a reproduction of their number.")

    ledger = judge.ledger
    print("\n" + "=" * 78)
    print("COST")
    print("=" * 78)
    print(f"Total calls: {ledger.total_calls()}   Cache hit rate: {ledger.cache_hit_rate():.0%}")
    print(f"Total cost:  ${ledger.total_cost_usd():.4f}")


if __name__ == "__main__":
    main()
