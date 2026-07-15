
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from attribution import Trace, TraceStep
from graph import WorkflowGraph

_SPEAKER_KEYS = ("name", "role", "agent", "sender", "speaker")
_PAREN_SUFFIX = __import__("re").compile(r"\s*\(")


def _extract_speaker(turn: dict, index: int) -> str:
    for key in _SPEAKER_KEYS:
        value = turn.get(key)
        if value:
            value = str(value)
            if key == "role":
                value = _PAREN_SUFFIX.split(value, maxsplit=1)[0].strip()
            return value
    return f"agent_step_{index}"


class TraceSource(ABC):

    @abstractmethod
    def load(self) -> list[Trace]:
        raise NotImplementedError


class SyntheticTraceSource(TraceSource):

    def __init__(self, graph: WorkflowGraph, n: int = 300, config=None, seed: int = 11):
        self.graph = graph
        self.n = n
        self.config = config
        self.seed = seed

    def load(self) -> list[Trace]:
        from simulate import generate_traces, ScenarioConfig
        config = self.config if self.config is not None else ScenarioConfig()
        return generate_traces(self.graph, n=self.n, config=config, seed=self.seed)


@dataclass
class WhoWhenConfig:
    split: str = "Algorithm-Generated"
    max_content_chars: int = 500


class WhoWhenTraceSource(TraceSource):

    def __init__(self, path: str, config: Optional[WhoWhenConfig] = None):
        self.path = path
        self.config = config or WhoWhenConfig()

    def load(self) -> list[Trace]:
        records = self._read_records(self.path)
        return [self._record_to_trace(r, i) for i, r in enumerate(records)]

    @staticmethod
    def _read_records(path: str) -> list[dict]:
        if path.endswith(".parquet"):
            try:
                import pandas as pd
                df = pd.read_parquet(path)
            except ImportError as e:
                raise ImportError(
                    "reading .parquet requires pandas + a parquet engine "
                    "(pip install pandas pyarrow --break-system-packages). "
                    "If you'd rather avoid that dependency, export to JSONL "
                    "instead: datasets.load_dataset(...).to_json('out.jsonl')"
                ) from e
            return df.to_dict(orient="records")

        if path.endswith(".jsonl"):
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]

        if path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        return value
                raise ValueError(f"{path}: found a JSON object but no record list inside it")
            raise ValueError(f"{path}: expected a JSON list or dict, got {type(data).__name__}")

        raise ValueError(f"unrecognized file extension for {path!r}; expected .parquet, .json, or .jsonl")

    def _record_to_trace(self, record: dict, idx: int) -> Trace:
        history = record.get("history") or []
        is_correct = bool(record.get("is_correct", False))
        question_id = str(record.get("question_ID", f"whowhen_{idx}"))
        mistake_agent = record.get("mistake_agent")

        mistake_step_raw = record.get("mistake_step")
        try:
            mistake_step = int(mistake_step_raw) if mistake_step_raw is not None else None
        except (TypeError, ValueError):
            mistake_step = None

        steps: list[TraceStep] = []
        for i, turn in enumerate(history):
            if not isinstance(turn, dict):
                continue
            content = str(turn.get("content", ""))[: self.config.max_content_chars]
            speaker = _extract_speaker(turn, i)
            steps.append(TraceStep(node_id=speaker, symptoms={"content": content, "turn_index": i}))

        ground_truth_node = mistake_agent if not is_correct else None
        ground_truth_step = mistake_step if not is_correct else None

        return Trace(
            trace_id=question_id,
            steps=steps,
            final_outcome_failed=not is_correct,
            ground_truth_node=ground_truth_node,
            ground_truth_step=ground_truth_step,
            scenario=f"whowhen_{self.config.split.lower().replace('-', '_')}",
        )
