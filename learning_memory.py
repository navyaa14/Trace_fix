
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional

from repair_engine import ValidatedRepair


@dataclass
class MemoryEntry:
    node_id: str
    action: str
    accepted_count: int = 0
    rejected_count: int = 0
    not_executable_count: int = 0
    total_cost_delta_usd: float = 0.0
    failure_type: str = "unknown"

    @property
    def attempts(self) -> int:
        return self.accepted_count + self.rejected_count

    @property
    def accept_rate(self) -> float:
        return self.accepted_count / self.attempts if self.attempts else 0.0

    @property
    def avg_cost_delta_usd(self) -> float:
        return self.total_cost_delta_usd / self.attempts if self.attempts else 0.0


@dataclass
class ActionDecision:
    action: str
    reason: str
    historical_attempts: int
    historical_accept_rate: float
    exploration_or_exploitation: str
    confidence: float


class LearningMemory:
    def __init__(self, path: str = "learning_memory.json"):
        self.path = path
        self._entries: dict[tuple[str, str, str], MemoryEntry] = {}
        self._history: list[dict] = []
        self._total_attempts_at_node: dict[str, int] = {}
        if os.path.exists(path):
            self._load()

    def _key(self, node_id: str, failure_type: str, action: str) -> tuple[str, str, str]:
        return (node_id, failure_type or "unknown", action)

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        for row in data.get("entries", []):
            row.setdefault("failure_type", "unknown")
            row.setdefault("not_executable_count", 0)
            e = MemoryEntry(**row)
            self._entries[self._key(e.node_id, e.failure_type, e.action)] = e
        self._history = data.get("history", [])
        self._recompute_node_totals()

    def _recompute_node_totals(self) -> None:
        self._total_attempts_at_node = {}
        for e in self._entries.values():
            self._total_attempts_at_node[e.node_id] = (
                self._total_attempts_at_node.get(e.node_id, 0) + e.attempts)

    def save(self) -> None:
        data = {
            "entries": [asdict(e) for e in self._entries.values()],
            "history": self._history,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def record(self, repair: ValidatedRepair) -> None:
        if not repair.applied:
            return

        failure_type = repair.failure_type or "unknown"
        key = self._key(repair.node_id, failure_type, repair.action.value)
        entry = self._entries.setdefault(
            key, MemoryEntry(repair.node_id, repair.action.value, failure_type=failure_type))

        cost_delta = repair.after_cost.api_cost_usd - repair.before_cost.api_cost_usd
        if repair.accepted:
            entry.accepted_count += 1
        else:
            entry.rejected_count += 1
        entry.total_cost_delta_usd += cost_delta
        self._total_attempts_at_node[repair.node_id] = self._total_attempts_at_node.get(repair.node_id, 0) + 1
        self._history.append({
            "trace_id": repair.trace_id,
            "node_id": repair.node_id,
            "failure_type": failure_type,
            "action": repair.action.value,
            "accepted": repair.accepted,
            "reason": repair.reason,
            "cost_delta_usd": round(cost_delta, 6),
        })

    def stats(self, node_id: str, action: str, failure_type: Optional[str] = None) -> MemoryEntry:
        if failure_type is not None:
            return self._entries.get(self._key(node_id, failure_type, action),
                                       MemoryEntry(node_id, action, failure_type=failure_type))
        matches = [e for (n, _, a), e in self._entries.items() if n == node_id and a == action]
        if not matches:
            return MemoryEntry(node_id, action)
        agg = MemoryEntry(node_id, action, failure_type="__aggregate__")
        for e in matches:
            agg.accepted_count += e.accepted_count
            agg.rejected_count += e.rejected_count
            agg.not_executable_count += e.not_executable_count
            agg.total_cost_delta_usd += e.total_cost_delta_usd
        return agg

    def best_action_for(self, node_id: str, candidate_actions: list[str],
                         failure_type: Optional[str] = None) -> str | None:
        scored = [(a, self.stats(node_id, a, failure_type)) for a in candidate_actions]
        scored = [(a, e) for a, e in scored if e.attempts > 0]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[1].accept_rate)[0]


    def select_action(self, node_id: str, failure_type: str, candidate_actions: list[str],
                       epsilon: float = 0.15, rng=None) -> ActionDecision:
        import random as _random
        rng = rng or _random

        stats_by_action = {a: self.stats(node_id, a, failure_type) for a in candidate_actions}
        tried = {a: e for a, e in stats_by_action.items() if e.attempts > 0}

        if not tried:
            chosen = candidate_actions[0]
            return ActionDecision(
                action=chosen,
                reason=f"no prior history for ({node_id}, {failure_type}) -- cold start, "
                       f"trying {chosen} first",
                historical_attempts=0, historical_accept_rate=0.0,
                exploration_or_exploitation="cold_start", confidence=0.0)

        if rng.random() < epsilon:
            chosen = rng.choice(candidate_actions)
            e = stats_by_action[chosen]
            return ActionDecision(
                action=chosen, reason=f"epsilon-greedy exploration (p={epsilon}) of {chosen}",
                historical_attempts=e.attempts, historical_accept_rate=e.accept_rate,
                exploration_or_exploitation="explore",
                confidence=self._confidence(e.attempts))

        total_attempts_at_node = max(1, self._total_attempts_at_node.get(node_id, sum(
            e.attempts for e in stats_by_action.values())))

        def ucb(action: str) -> float:
            e = stats_by_action[action]
            if e.attempts == 0:
                return float("inf")
            return e.accept_rate + math.sqrt(2 * math.log(total_attempts_at_node + 1) / e.attempts)

        chosen = max(candidate_actions, key=ucb)
        e = stats_by_action[chosen]
        if e.attempts == 0:
            return ActionDecision(
                action=chosen,
                reason=f"UCB1: {chosen} has no attempts yet for ({node_id}, {failure_type}) among "
                       f"otherwise-tried candidates -- trying it before trusting the measured leader",
                historical_attempts=0, historical_accept_rate=0.0,
                exploration_or_exploitation="explore", confidence=0.0)
        return ActionDecision(
            action=chosen,
            reason=(f"UCB1 exploit: {chosen} has accept_rate={e.accept_rate:.2f} over "
                    f"{e.attempts} prior attempts for ({node_id}, {failure_type})"),
            historical_attempts=e.attempts, historical_accept_rate=e.accept_rate,
            exploration_or_exploitation="exploit",
            confidence=self._confidence(e.attempts))

    @staticmethod
    def _confidence(attempts: int) -> float:
        return round(1 - math.exp(-attempts / 5.0), 3)

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def as_policy_dict(self) -> dict:
        by_node_ft: dict[str, dict[str, list[MemoryEntry]]] = {}
        for e in self._entries.values():
            if e.attempts == 0:
                continue
            by_node_ft.setdefault(e.node_id, {}).setdefault(e.failure_type, []).append(e)

        policy = {}
        for node_id, by_ft in by_node_ft.items():
            policy[node_id] = {}
            for ft, entries in by_ft.items():
                best = max(entries, key=lambda e: e.accept_rate)
                policy[node_id][ft] = {
                    "preferred_action": best.action,
                    "attempts": best.attempts,
                    "accept_rate": round(best.accept_rate, 4),
                    "mean_cost_delta_usd": round(best.avg_cost_delta_usd, 6),
                }
        return policy
