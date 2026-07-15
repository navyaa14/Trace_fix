
from __future__ import annotations
from typing import Optional
from graph import WorkflowGraph
from optimizer import NodeStats, Recommendation
from repair import ReplayResult
from attribution import MIN_RELIABLE_N

NODE_POS = {
    "chunker": (60, 60), "kb_builder": (220, 60), "retriever": (380, 60),
    "clarifier": (380, 180), "generator": (540, 60), "vade": (700, 60),
    "evaluator": (700, 180), "human": (700, 300),
}

STATUS_COLOR = {"ok": "#33D17A", "warn": "#F5A623", "bad": "#FF5C5C", "idle": "#3A4152"}


def _node_status(node_id: str, stats: dict[str, NodeStats]) -> str:
    s = stats.get(node_id)
    if not s or s.executions == 0:
        return "idle"
    if s.failure_rate >= 0.20:
        return "bad"
    if s.failure_rate > 0:
        return "warn"
    return "ok"


def _svg_graph(graph: WorkflowGraph, stats: dict[str, NodeStats]) -> str:
    lines = []
    for e in graph.edges:
        if e.src not in NODE_POS or e.dst not in NODE_POS:
            continue
        x1, y1 = NODE_POS[e.src]
        x2, y2 = NODE_POS[e.dst]
        lines.append(f'<line x1="{x1+45}" y1="{y1+18}" x2="{x2+5}" y2="{y2+18}" '
                     f'stroke="#3A4152" stroke-width="1.5" marker-end="url(#arrow)"/>')

    nodes = []
    for node_id, (x, y) in NODE_POS.items():
        if node_id not in graph.nodes:
            continue
        status = _node_status(node_id, stats)
        color = STATUS_COLOR[status]
        fr = stats[node_id].failure_rate if node_id in stats else 0.0
        nodes.append(f'''
        <g>
          <rect x="{x}" y="{y}" width="90" height="36" rx="4"
                fill="#131722" stroke="{color}" stroke-width="{2.5 if status=='bad' else 1.5}"/>
          <text x="{x+45}" y="{y+16}" text-anchor="middle" fill="#E6E8EE"
                font-family="ui-monospace,Menlo,monospace" font-size="11">{node_id}</text>
          <text x="{x+45}" y="{y+29}" text-anchor="middle" fill="{color}"
                font-family="ui-monospace,Menlo,monospace" font-size="10">{fr:.0%}</text>
        </g>''')

    return f'''<svg viewBox="0 0 820 360" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#3A4152"/>
        </marker>
      </defs>
      {''.join(lines)}
      {''.join(nodes)}
    </svg>'''


def _recs_table(recs: list[Recommendation]) -> str:
    rows = []
    for r in sorted(recs, key=lambda r: -r.failure_rate):
        color = STATUS_COLOR["bad"] if r.failure_rate >= 0.2 else (
            STATUS_COLOR["warn"] if r.failure_rate > 0 else STATUS_COLOR["ok"])
        rows.append(f'''
        <tr>
          <td class="mono">{r.node_id}</td>
          <td><span class="pill" style="border-color:{color};color:{color}">{r.action.value}</span></td>
          <td class="mono" style="color:{color}">{r.failure_rate:.1%}</td>
          <td class="mono">${r.cost.api_cost_usd:.4f}</td>
          <td class="mono">{r.cost.latency_penalty_score:.2f}</td>
          <td class="reason">{r.reason}</td>
        </tr>''')
    return "\n".join(rows)


def _learning_memory_table(entries: list) -> str:
    if not entries:
        return '<tr><td colspan="4" class="reason">no persisted history yet -- run continuous_improvement.run_improvement_cycle to populate it</td></tr>'
    rows = []
    for e in sorted(entries, key=lambda e: -e.attempts):
        color = STATUS_COLOR["ok"] if e.accept_rate >= 0.5 else STATUS_COLOR["bad"]
        rows.append(f'''
        <tr>
          <td class="mono">{e.node_id}</td>
          <td><span class="pill" style="border-color:{color};color:{color}">{e.action}</span></td>
          <td class="mono">{e.attempts}</td>
          <td class="mono" style="color:{color}">{e.accept_rate:.0%}</td>
          <td class="mono">${e.avg_cost_delta_usd:+.4f}</td>
        </tr>''')
    return "\n".join(rows)


def _repairs_table(repairs: list) -> str:
    if not repairs:
        return '<tr><td colspan="5" class="reason">no repairs attempted this run</td></tr>'
    rows = []
    for r in repairs:
        if not r.applied:
            status, color = "SKIPPED", STATUS_COLOR["idle"]
        elif r.accepted:
            status, color = "ACCEPTED", STATUS_COLOR["ok"]
        else:
            status, color = "REJECTED", STATUS_COLOR["bad"]
        rows.append(f'''
        <tr>
          <td class="mono">{r.trace_id}</td>
          <td class="mono">{r.node_id}</td>
          <td><span class="pill" style="border-color:{color};color:{color}">{r.action.value}</span></td>
          <td><span class="pill" style="border-color:{color};color:{color}">{status}</span></td>
          <td class="reason">{r.reason}</td>
        </tr>''')
    return "\n".join(rows)


def build_dashboard(graph: WorkflowGraph, stats: dict[str, NodeStats],
                     recs: list[Recommendation],
                     agent_accuracy: float, n_evaluated: int,
                     method_rows: list[tuple[str, float, float]],
                     replay: ReplayResult,
                     learning_entries: Optional[list] = None,
                     improvement_repairs: Optional[list] = None,
                     improvement_summary: Optional[dict] = None,
                     failure_type_breakdown: Optional[dict] = None,
                     multi_label_breakdown: Optional[dict] = None,
                     outcome_breakdown: Optional[dict] = None) -> str:
    svg = _svg_graph(graph, stats)
    table = _recs_table(recs)

    method_table = "\n".join(
        f'<tr><td class="mono">{name}</td><td class="mono">{acc:.1%}</td>'
        f'<td class="mono">{calls:.2f}</td></tr>'
        for name, acc, calls in method_rows
    )

    failure_type_rows = ""
    if failure_type_breakdown:
        for failure_type, (acc, n) in sorted(failure_type_breakdown.items(), key=lambda kv: kv[1][0]):
            if n < MIN_RELIABLE_N:
                color = STATUS_COLOR["idle"]
                flag = f"low n ({n}) -- not a reliable estimate, not evidence the method fails here"
            else:
                color = STATUS_COLOR["ok"] if acc >= 0.5 else STATUS_COLOR["bad"]
                flag = "worse than uniform-random (1/8 nodes)" if acc < 0.125 else ""
            failure_type_rows += (
                f'<tr><td class="mono">{failure_type}</td>'
                f'<td class="mono" style="color:{color}">{acc:.1%}</td>'
                f'<td class="mono">{n}</td><td class="reason">{flag}</td></tr>\n'
            )

    multi_label_panel = ""
    if multi_label_breakdown:
        rows = ""
        for method_name, (single_acc, multi_acc, n) in multi_label_breakdown.items():
            gap = multi_acc - single_acc
            rows += (
                f'<tr><td class="mono">{method_name}</td>'
                f'<td class="mono">{single_acc:.1%}</td>'
                f'<td class="mono">{multi_acc:.1%}</td>'
                f'<td class="mono" style="color:{STATUS_COLOR["ok"] if gap > 0.1 else STATUS_COLOR["bad"]}">'
                f'+{gap:.1%}</td><td class="mono">{n}</td></tr>\n'
            )
        multi_label_panel = f'''
  <div class="panel" style="margin-bottom:20px">
    <h2>Root cause: "multiple_simultaneous_failures" — single-label vs. multi-label scoring</h2>
    <table>
      <tr><th>method</th><th>single-label acc.</th><th>multi-label acc.</th><th>gap</th><th>n</th></tr>
      {rows}
    </table>
  </div>'''

    replay_delta_latency = replay.after_cost.latency_penalty_score - replay.before_cost.latency_penalty_score
    replay_delta_api = replay.after_cost.api_cost_usd - replay.before_cost.api_cost_usd
    replay_delta_human = replay.after_cost.human_cost_usd - replay.before_cost.human_cost_usd
    before_outcome_text = "FAILED \u2192 escalated" if replay.before_failed else "PASSED"
    before_outcome_css = "bad" if replay.before_failed else "good"
    after_outcome_text = "RESOLVED, no escalation" if not replay.after_failed else "STILL FAILING \u2192 escalated"
    after_outcome_css = "good" if not replay.after_failed else "bad"

    learning_panel = ""
    if learning_entries is not None or improvement_repairs is not None:
        learning_rows = _learning_memory_table(learning_entries or [])
        repairs_rows = _repairs_table(improvement_repairs or [])
        summary_stats = ""
        if improvement_summary:
            s = improvement_summary
            summary_stats = f'''
            <div class="stat"><span class="k">Repairs attempted</span><span class="v">{s.get('attempted', 0)}</span></div>
            <div class="stat"><span class="k">Accepted</span><span class="v good">{s.get('accepted', 0)}</span></div>
            <div class="stat"><span class="k">Rejected</span><span class="v bad">{s.get('rejected', 0)}</span></div>
            <div class="stat"><span class="k">Human escalations before → after</span>
              <span class="v">{s.get('human_before', 0)} → {s.get('human_after', 0)}</span></div>
            <div class="stat"><span class="k">Total cost before → after</span>
              <span class="v">${s.get('cost_before', 0):.2f} → ${s.get('cost_after', 0):.2f}</span></div>'''
        learning_panel = f'''
  <div class="grid" style="grid-template-columns:1fr 1.3fr;">
    <div class="panel">
      <h2>Continuous improvement — this run</h2>
      {summary_stats or '<p class="explanation">No batch improvement cycle was run -- see continuous_improvement.run_improvement_cycle.</p>'}
    </div>
    <div class="panel">
      <h2>Persisted learning memory — accept rate by (node, action) across all runs</h2>
      <table>
        <tr><th>node</th><th>action</th><th>attempts</th><th>accept rate</th><th>avg cost delta</th></tr>
        {learning_rows}
      </table>
    </div>
  </div>
  <div class="panel" style="margin-bottom:20px">
    <h2>Repair attempts this run — accepted / rejected / skipped</h2>
    <table>
      <tr><th>trace</th><th>node</th><th>action</th><th>status</th><th>reason</th></tr>
      {repairs_rows}
    </table>
  </div>'''

    outcome_panel = ""
    if outcome_breakdown:
        ob = outcome_breakdown
        outcome_panel = f'''
  <div class="panel" style="margin-bottom:20px">
    <h2>Content failure vs. evaluation failure vs. workflow failure</h2>
    <div class="stat"><span class="k">content_failed rate</span>
      <span class="v {'bad' if ob.get('content_failed_rate', 0) > 0 else 'good'}">{ob.get('content_failed_rate', 0):.1%}</span></div>
    <div class="stat"><span class="k">evaluation_failed rate</span>
      <span class="v {'bad' if ob.get('evaluation_failed_rate', 0) > 0 else 'good'}">{ob.get('evaluation_failed_rate', 0):.1%}</span></div>
    <div class="stat"><span class="k">workflow_failed rate (content OR evaluation)</span>
      <span class="v {'bad' if ob.get('workflow_failed_rate', 0) > 0 else 'good'}">{ob.get('workflow_failed_rate', 0):.1%}</span></div>
    <div class="stat"><span class="k">evaluator_false_escalation (good content, evaluator rejected)</span>
      <span class="v">{ob.get('n_evaluator_false_escalation', 0)}</span></div>
    <div class="stat"><span class="k">evaluator_false_acceptance (bad content, evaluator accepted)</span>
      <span class="v">{ob.get('n_evaluator_false_acceptance', 0)}</span></div>
  </div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>TraceFix — Failure Attribution &amp; Repair Dashboard</title>
<style>
  :root {{
    --bg:#0B0E14; --panel:#131722; --line:#232838; --text:#E6E8EE; --dim:#7C8299;
    --ok:#33D17A; --warn:#F5A623; --bad:#FF5C5C; --accent:#5B8CFF;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    background:var(--bg); color:var(--text); margin:0; padding:32px;
    font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;
    font-size:13px;
  }}
  h1 {{ font-size:20px; font-weight:600; margin:0 0 4px; letter-spacing:-0.02em; }}
  .sub {{ color:var(--dim); margin:0 0 28px; font-size:12px; }}
  .grid {{ display:grid; grid-template-columns: 1.3fr 1fr; gap:20px; margin-bottom:20px; }}
  .panel {{
    background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:18px;
  }}
  .panel h2 {{
    font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--dim);
    margin:0 0 14px; font-weight:600;
  }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{
    text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:0.05em;
    color:var(--dim); padding:6px 8px; border-bottom:1px solid var(--line);
  }}
  td {{ padding:7px 8px; border-bottom:1px solid var(--line); }}
  td.mono {{ font-variant-numeric: tabular-nums; }}
  td.reason {{ color:var(--dim); font-size:11.5px; max-width:340px; }}
  .pill {{
    border:1px solid; border-radius:3px; padding:2px 7px; font-size:11px;
  }}
  .stat {{ display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--line); }}
  .stat:last-child {{ border-bottom:none; }}
  .stat .k {{ color:var(--dim); }}
  .stat .v {{ font-weight:600; }}
  .v.good {{ color:var(--ok); }} .v.bad {{ color:var(--bad); }}
  .flag {{ display:inline-block; font-size:10px; padding:2px 6px; border-radius:3px; margin-right:6px; }}
  .flag.synthetic {{ background:#232838; color:var(--warn); border:1px solid var(--warn); }}
  .flag.heuristic {{ background:#232838; color:var(--accent); border:1px solid var(--accent); }}
  .explanation {{ color:var(--dim); line-height:1.6; margin-top:10px; font-size:12px; }}
  .replay-cols {{ display:grid; grid-template-columns:1fr auto 1fr; gap:14px; align-items:center; margin-top:12px; }}
  .replay-box {{ background:#0F1420; border:1px solid var(--line); border-radius:6px; padding:14px; }}
  .replay-box .label {{ color:var(--dim); font-size:10px; text-transform:uppercase; margin-bottom:8px; }}
  .arrow {{ text-align:center; color:var(--dim); font-size:18px; }}
</style>
</head>
<body>
  <h1>TraceFix — Failure Attribution &amp; Repair Dashboard</h1>
  <p class="sub">
    <span class="flag synthetic">SYNTHETIC TRACES</span>
    <span class="flag heuristic">HEURISTIC JUDGE (no LLM call)</span>
  </p>

  <div class="grid">
    <div class="panel">
      <h2>Workflow graph — node color = failure rate this window</h2>
      {svg}
    </div>
    <div class="panel">
      <h2>Attribution accuracy (predicted node == ground-truth node)</h2>
      <div class="stat"><span class="k">Traces evaluated</span><span class="v">{n_evaluated}</span></div>
      <div class="stat"><span class="k">all_at_once agent-level accuracy</span>
        <span class="v {'good' if agent_accuracy>=0.5 else 'bad'}">{agent_accuracy:.1%}</span></div>
      <table style="margin-top:12px">
        <tr><th>method</th><th>accuracy</th><th>avg judge-calls/trace</th></tr>
        {method_table}
      </table>
    </div>
  </div>
  {outcome_panel}

  <div class="panel" style="margin-bottom:20px">
    <h2>Accuracy by failure type — the blended number above hides this</h2>
    <table>
      <tr><th>failure type</th><th>accuracy</th><th>n evaluated</th><th>note</th></tr>
      {failure_type_rows or '<tr><td colspan="4" class="reason">no failure_type breakdown passed to build_dashboard</td></tr>'}
    </table>
  </div>
  {multi_label_panel}

  <div class="panel" style="margin-bottom:20px">
    <h2>Per-node recommendations</h2>
    <table>
      <tr><th>node</th><th>action</th><th>failure rate</th><th>api cost</th><th>latency penalty</th><th>reason</th></tr>
      {table}
    </table>
  </div>

  <div class="panel">
    <h2>Repair replay — scenario: wrong product variant retrieved</h2>
    <div class="replay-cols">
      <div class="replay-box">
        <div class="label">Before ({replay.action.value} not applied)</div>
        <div class="stat"><span class="k">Outcome</span><span class="v {before_outcome_css}">{before_outcome_text}</span></div>
        <div class="stat"><span class="k">Groundedness</span><span class="v {'good' if replay.before_groundedness>=0.5 else 'bad'}">{replay.before_groundedness:.2f}</span></div>
        <div class="stat"><span class="k">API cost</span><span class="v">${replay.before_cost.api_cost_usd:.4f}</span></div>
        <div class="stat"><span class="k">Latency penalty</span><span class="v">{replay.before_cost.latency_penalty_score:.2f}</span></div>
        <div class="stat"><span class="k">Human cost</span><span class="v bad">${replay.before_cost.human_cost_usd:.2f}</span></div>
      </div>
      <div class="arrow">&#8594;</div>
      <div class="replay-box">
        <div class="label">After ({replay.action.value} applied)</div>
        <div class="stat"><span class="k">Outcome</span><span class="v {after_outcome_css}">{after_outcome_text}</span></div>
        <div class="stat"><span class="k">Groundedness</span><span class="v {'good' if replay.after_groundedness>=0.5 else 'bad'}">{replay.after_groundedness:.2f}</span></div>
        <div class="stat"><span class="k">API cost</span><span class="v">${replay.after_cost.api_cost_usd:.4f} ({replay_delta_api:+.4f})</span></div>
        <div class="stat"><span class="k">Latency penalty</span><span class="v">{replay.after_cost.latency_penalty_score:.2f} ({replay_delta_latency:+.2f})</span></div>
        <div class="stat"><span class="k">Human cost</span><span class="v good">${replay.after_cost.human_cost_usd:.2f} ({replay_delta_human:+.2f})</span></div>
      </div>
    </div>
  </div>
{learning_panel}
</body>
</html>'''
