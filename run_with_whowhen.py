
from __future__ import annotations

import argparse
import os
import sys

from attribution import FailureAttributor, heuristic_judge
from whowhen_adapter import WhoWhenTraceSource, WhoWhenConfig, _SPEAKER_KEYS


def _speaker_fallback_count(traces) -> tuple[int, int]:
    total_steps = sum(len(t.steps) for t in traces)
    fallback_steps = sum(
        1 for t in traces for s in t.steps if s.node_id.startswith("agent_step_")
    )
    return fallback_steps, total_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset_path", help="local .parquet, .jsonl, or .json Who&When file")
    parser.add_argument("--split-label", default="Algorithm-Generated",
                         choices=["Algorithm-Generated", "Hand-Crafted"],
                         help="label only -- does not change parsing, just Trace.scenario")
    parser.add_argument("--judge", choices=["heuristic", "claude"], default="heuristic",
                         help="heuristic (default, zero-dependency, expected to mostly return "
                              "NONE on this data) or claude (real LLM judge, needs "
                              "ANTHROPIC_API_KEY and `pip install -r requirements.txt`)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"'{args.dataset_path}' not found. Download the dataset first -- see "
              f"whowhen_adapter.py's module docstring for accepted formats and a "
              f"datasets.load_dataset(...).to_json(...) export path if you only have "
              f"the .parquet file.")
        sys.exit(1)

    source = WhoWhenTraceSource(args.dataset_path, WhoWhenConfig(split=args.split_label))
    traces = source.load()
    dataset_status = "full benchmark validation" if len(traces) >= 150 else "partial dataset (not the full 184-row benchmark)"
    print(f"Loaded {len(traces)} traces from '{args.dataset_path}' (split label: {args.split_label})")
    print(f"dataset_status = {dataset_status!r}  "
          f"(distinct from 'fixture-only validation', which is what tests/test_whowhen_real_sample.py "
          f"runs against 8 hand-transcribed rows -- this run is against your real local file, not that fixture)")

    fallback_steps, total_steps = _speaker_fallback_count(traces)
    if fallback_steps:
        pct = fallback_steps / total_steps if total_steps else 0.0
        print(f"WARNING: {fallback_steps}/{total_steps} steps ({pct:.0%}) used the "
              f"'agent_step_<i>' fallback because none of {_SPEAKER_KEYS} matched a key "
              f"in that turn's dict. Attribution against those steps is attributing to a "
              f"placeholder, not a real agent id -- inspect one raw record "
              f"(e.g. `traces[0].steps[0]`) and adjust whowhen_adapter._SPEAKER_KEYS if "
              f"the real schema uses a different key.")

    if args.judge == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set. --judge claude makes real, billed API "
                  "calls, so it refuses to guess at a key. Set it and re-run, or drop "
                  "--judge claude to use the offline heuristic judge instead.")
            sys.exit(1)
        try:
            from llm_judge import make_claude_judge, ClaudeJudgeConfig
        except ImportError:
            print("The 'anthropic' package isn't installed. Run:\n"
                  "  pip install -r requirements.txt --break-system-packages")
            sys.exit(1)
        known_ids = {s.node_id for t in traces for s in t.steps}
        judge_fn = make_claude_judge(ClaudeJudgeConfig(
            known_node_ids=frozenset(known_ids),
            cache_path="whowhen_llm_judge_cache.json",
        ))
        judge_label = "live Claude evaluation (judge='claude', real billed API calls)"
    else:
        judge_fn = heuristic_judge
        judge_label = "heuristic (judge='heuristic', expected to mostly return NONE on this data -- see above)"

    attributor = FailureAttributor(judge=judge_fn)

    evaluated = [t for t in traces if t.final_outcome_failed and t.ground_truth_node]
    print(f"\n{len(evaluated)}/{len(traces)} traces are failed with a known ground-truth agent "
          f"(is_correct=False and mistake_agent present) -- only these are evaluable.")

    print("=" * 78)
    print(f"ATTRIBUTION ACCURACY vs. real Who&When ground truth  (judge = {judge_label})")
    print("=" * 78)
    correct = 0
    for t in evaluated:
        result = attributor.attribute_all_at_once(t)
        match = result.responsible_node == t.ground_truth_node
        correct += match
    acc = correct / len(evaluated) if evaluated else 0.0
    print(f"all_at_once agent-level accuracy = {acc:.1%}   (n={len(evaluated)})")
    print("Reference point: Who&When's own best published method gets 53.5% agent-level / "
          "14.2% step-level accuracy on this same dataset. This run uses a much simpler "
          "judge (heuristic rules built for synthetic numeric fields, or a single "
          "un-tuned Claude call with no few-shot examples) -- treat any number here as a "
          "sanity check that the loader/attribution plumbing works end-to-end against the "
          "real benchmark, not as a comparable result to their paper.")

    if args.judge == "claude":
        ledger = judge_fn.ledger
        print(f"\nTotal calls: {ledger.total_calls()}   Cache hit rate: {ledger.cache_hit_rate():.0%}")
        print(f"Total cost:  ${ledger.total_cost_usd():.4f}")


if __name__ == "__main__":
    main()
