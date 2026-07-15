
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-4-8": (5.0, 25.0),
}
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

PROMPT_VERSION = "judge_prompt_v1"

JUDGE_SYSTEM_PROMPT = """You are a failure-attribution judge for a customer-support pipeline trace.
You will see one line per pipeline step, in execution order, containing only
observable evidence (retrieval scores, entity-match flags, groundedness,
KB age, etc.) -- never a "this step failed" label or the true answer.

Your job: decide which single node is most likely responsible for the
trace's failure, and how confident you are. If no step's evidence looks
anomalous, say so honestly -- do not guess a node just to give an answer.

You MUST call the submit_verdict tool exactly once with your answer."""

SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Report which node is responsible for the trace's failure.",
    "input_schema": {
        "type": "object",
        "properties": {
            "responsible_node": {
                "type": "string",
                "description": "The node id most likely responsible, or 'NONE' if no step shows anomalous evidence.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0 to 1.0. Reflects genuine uncertainty -- do not default to a high number.",
            },
            "reason": {
                "type": "string",
                "description": "Short (<12 words), snake_case-friendly justification citing the specific evidence.",
            },
        },
        "required": ["responsible_node", "confidence", "reason"],
    },
}


class JudgeError(Exception):
    pass


class TransientError(Exception):
    pass


@dataclass
class TransportResult:
    responsible_node: str
    confidence: float
    reason: str
    input_tokens: int
    output_tokens: int


TransportFn = Callable[[str, str, "ClaudeJudgeConfig"], TransportResult]


@dataclass
class ClaudeJudgeConfig:
    model: str = DEFAULT_MODEL
    max_tokens: int = 300
    max_retries: int = 3
    backoff_base_s: float = 0.5
    known_node_ids: Optional[frozenset[str]] = None
    use_cache: bool = True
    cache_path: Optional[str] = None
    prompt_version: str = PROMPT_VERSION


@dataclass
class CallLog:
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_hit: bool
    latency_ms: float
    timestamp: float
    responsible_node: str
    error: Optional[str] = None


@dataclass
class CostLedger:
    calls: list[CallLog] = field(default_factory=list)

    def log(self, entry: CallLog) -> None:
        self.calls.append(entry)

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def total_calls(self) -> int:
        return len(self.calls)

    def cache_hit_rate(self) -> float:
        if not self.calls:
            return 0.0
        return sum(1 for c in self.calls if c.cache_hit) / len(self.calls)

    def by_model(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.model] = out.get(c.model, 0.0) + c.cost_usd
        return out

    def to_json(self) -> str:
        return json.dumps([c.__dict__ for c in self.calls], indent=2)


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING_USD_PER_MTOK:
        raise JudgeError(
            f"no pricing entry for model {model!r} -- add it to PRICING_USD_PER_MTOK "
            f"(check https://docs.claude.com/en/docs/about-claude/pricing first, "
            f"pricing changes over time)"
        )
    in_rate, out_rate = PRICING_USD_PER_MTOK[model]
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def _cache_key(model: str, prompt_version: str, prompt: str) -> str:
    h = hashlib.sha256(f"{model}|{prompt_version}|{prompt}".encode()).hexdigest()
    return h


class _ReplayCache:

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._data: dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        self._data[key] = value
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)


def _validate(result: TransportResult, known_node_ids: Optional[frozenset[str]]) -> TransportResult:
    node = result.responsible_node
    conf = result.confidence
    if not isinstance(conf, (int, float)):
        raise JudgeError(f"confidence is not numeric: {conf!r}")
    conf = max(0.0, min(1.0, float(conf)))
    if known_node_ids is not None and node != "NONE" and node not in known_node_ids:
        raise JudgeError(f"responsible_node {node!r} is not a known node id")
    return TransportResult(node, conf, result.reason, result.input_tokens, result.output_tokens)


def _with_retry(fn: Callable[[], TransportResult], config: ClaudeJudgeConfig) -> TransportResult:
    last_exc: Optional[Exception] = None
    for attempt in range(config.max_retries):
        try:
            return fn()
        except TransientError as e:
            last_exc = e
            sleep_s = config.backoff_base_s * (2 ** attempt) + random.uniform(0, 0.1)
            time.sleep(sleep_s)
    raise JudgeError(f"exhausted {config.max_retries} retries; last error: {last_exc}")


def make_claude_judge(config: ClaudeJudgeConfig = ClaudeJudgeConfig(),
                       transport: Optional[TransportFn] = None,
                       ledger: Optional[CostLedger] = None) -> Callable[[str], str]:
    transport = transport or _default_anthropic_transport
    cache = _ReplayCache(config.cache_path if config.use_cache else None)
    ledger = ledger if ledger is not None else CostLedger()
    in_memory_cache: dict[str, dict] = {}

    def judge(prompt: str) -> str:
        key = _cache_key(config.model, config.prompt_version, prompt)
        cached = None
        if config.use_cache:
            cached = in_memory_cache.get(key) or cache.get(key)
        t0 = time.monotonic()

        if cached:
            ledger.log(CallLog(config.model, config.prompt_version, 0, 0, 0.0,
                                True, 0.0, time.time(), cached["responsible_node"]))
            return _format_verdict(cached["responsible_node"], cached["confidence"], cached["reason"])

        def _call() -> TransportResult:
            return transport(JUDGE_SYSTEM_PROMPT, prompt, config)

        try:
            result = _with_retry(_call, config)
            result = _validate(result, config.known_node_ids)
            cost = _cost_usd(config.model, result.input_tokens, result.output_tokens)
        except JudgeError as e:
            try:
                corrective_prompt = prompt + f"\n\n[Your previous response was invalid: {e}. Try again.]"
                result = transport(JUDGE_SYSTEM_PROMPT, corrective_prompt, config)
                result = _validate(result, config.known_node_ids)
                cost = _cost_usd(config.model, result.input_tokens, result.output_tokens)
            except Exception as e2:
                latency_ms = (time.monotonic() - t0) * 1000
                ledger.log(CallLog(config.model, config.prompt_version, 0, 0, 0.0,
                                    False, latency_ms, time.time(), "NONE", error=str(e2)))
                return "RESPONSIBLE=NONE CONFIDENCE=0.00 REASON=judge_error"

        latency_ms = (time.monotonic() - t0) * 1000
        ledger.log(CallLog(config.model, config.prompt_version, result.input_tokens,
                            result.output_tokens, cost, False, latency_ms, time.time(),
                            result.responsible_node))

        entry = {"responsible_node": result.responsible_node, "confidence": result.confidence,
                 "reason": result.reason}
        if config.use_cache:
            in_memory_cache[key] = entry
            cache.set(key, entry)

        return _format_verdict(result.responsible_node, result.confidence, result.reason)

    judge.ledger = ledger
    return judge


def _format_verdict(node: str, confidence: float, reason: str) -> str:
    safe_reason = reason.strip().replace(" ", "_") or "no_reason_given"
    return f"RESPONSIBLE={node} CONFIDENCE={confidence:.2f} REASON={safe_reason}"



def _default_anthropic_transport(system: str, prompt: str, config: ClaudeJudgeConfig) -> TransportResult:
    try:
        import anthropic
    except ImportError as e:
        raise JudgeError(
            "the 'anthropic' package is required for the default transport: "
            "pip install anthropic --break-system-packages"
        ) from e

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[SUBMIT_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "submit_verdict"},
        )
    except anthropic.APIConnectionError as e:
        raise TransientError(str(e)) from e
    except anthropic.RateLimitError as e:
        raise TransientError(str(e)) from e
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            raise TransientError(str(e)) from e
        raise JudgeError(str(e)) from e

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        raise JudgeError("model did not call submit_verdict")

    inp = tool_use.input
    return TransportResult(
        responsible_node=str(inp.get("responsible_node", "NONE")),
        confidence=inp.get("confidence", 0.0),
        reason=str(inp.get("reason", "")),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def bedrock_transport(bedrock_model_id: str, region_name: str = "us-east-1") -> TransportFn:
    def _transport(system: str, prompt: str, config: ClaudeJudgeConfig) -> TransportResult:
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as e:
            raise JudgeError("the 'boto3' package is required for the Bedrock transport: "
                              "pip install boto3 --break-system-packages") from e

        client = boto3.client("bedrock-runtime", region_name=region_name)
        tool_config = {
            "tools": [{
                "toolSpec": {
                    "name": SUBMIT_VERDICT_TOOL["name"],
                    "description": SUBMIT_VERDICT_TOOL["description"],
                    "inputSchema": {"json": SUBMIT_VERDICT_TOOL["input_schema"]},
                }
            }],
            "toolChoice": {"tool": {"name": SUBMIT_VERDICT_TOOL["name"]}},
        }
        try:
            response = client.converse(
                modelId=bedrock_model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": config.max_tokens},
                toolConfig=tool_config,
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException"):
                raise TransientError(str(e)) from e
            raise JudgeError(str(e)) from e

        content = response["output"]["message"]["content"]
        tool_use = next((b["toolUse"] for b in content if "toolUse" in b), None)
        if tool_use is None:
            raise JudgeError("model did not call submit_verdict")

        inp = tool_use["input"]
        usage = response["usage"]
        return TransportResult(
            responsible_node=str(inp.get("responsible_node", "NONE")),
            confidence=inp.get("confidence", 0.0),
            reason=str(inp.get("reason", "")),
            input_tokens=usage["inputTokens"],
            output_tokens=usage["outputTokens"],
        )

    return _transport
