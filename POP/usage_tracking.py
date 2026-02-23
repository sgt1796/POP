"""Token and usage accounting helpers for POP.

This module normalizes provider-reported usage into a single schema and
adds a provider-agnostic local token estimate fallback.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

ANOMALY_THRESHOLD = 0.5


def init_usage_totals() -> Dict[str, Any]:
    """Return a fresh per-instance usage totals dictionary."""
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "provider_calls": 0,
        "estimated_calls": 0,
        "hybrid_calls": 0,
        "anomaly_calls": 0,
        "provider_cost_total": 0.0,
    }


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except Exception:
            return {}
    return {}


def _first(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    as_dict = _to_dict(value)
    if as_dict:
        return _jsonable(as_dict)
    return str(value)


def _count_tokens(text: str, model: str) -> Tuple[int, str]:
    clean_text = text or ""
    try:
        import tiktoken  # type: ignore

        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(clean_text)), "tiktoken"
    except Exception:
        if not clean_text:
            return 0, "utf8_char4"
        return int(math.ceil(len(clean_text.encode("utf-8")) / 4.0)), "utf8_char4"


def extract_provider_usage(response: Any) -> Dict[str, Any]:
    """Extract usage fields from heterogeneous provider response objects."""
    payload = _to_dict(response)
    usage_obj: Any = None
    if isinstance(response, dict):
        usage_obj = response.get("usage")
    else:
        usage_obj = getattr(response, "usage", None)
    usage = _to_dict(usage_obj)

    # Fallback for providers that return usage-like counters at top-level.
    if not usage:
        if isinstance(response, dict):
            usage = response
        elif isinstance(payload, dict):
            usage = payload.get("usage", payload) if isinstance(payload.get("usage"), dict) else payload

    prompt_details = _to_dict(
        _first(usage, "prompt_tokens_details", "input_tokens_details", "prompt_token_details")
    )
    completion_details = _to_dict(
        _first(usage, "completion_tokens_details", "output_tokens_details", "completion_token_details")
    )

    input_tokens = _safe_int(
        _first(usage, "prompt_tokens", "input_tokens", "prompt_eval_count", "input_token_count")
    )
    output_tokens = _safe_int(
        _first(usage, "completion_tokens", "output_tokens", "eval_count", "output_token_count", "candidates_token_count")
    )
    total_tokens = _safe_int(_first(usage, "total_tokens", "total_token_count"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cached_tokens = _safe_int(_first(usage, "cached_tokens"))
    if cached_tokens is None:
        cached_tokens = _safe_int(_first(prompt_details, "cached_tokens", "cache_hit_tokens", "cache_tokens"))

    reasoning_tokens = _safe_int(_first(usage, "reasoning_tokens"))
    if reasoning_tokens is None:
        reasoning_tokens = _safe_int(
            _first(completion_details, "reasoning_tokens", "reasoning_output_tokens")
        )

    provider_cost = _safe_float(_first(usage, "cost", "total_cost", "usd_cost"))
    provider_cost_currency = _first(usage, "cost_currency", "currency")

    cost_details = _to_dict(_first(usage, "costs", "cost_details", "cost_detail"))
    if provider_cost is None and cost_details:
        provider_cost = _safe_float(_first(cost_details, "total", "amount", "value"))
    if provider_cost_currency is None and cost_details:
        provider_cost_currency = _first(cost_details, "currency", "unit")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "provider_cost": provider_cost,
        "provider_cost_currency": provider_cost_currency,
    }


def estimate_usage(
    messages: List[Dict[str, Any]],
    reply_text: str,
    provider: str,
    model: str,
    tools: Any = None,
    response_format: Any = None,
) -> Dict[str, Any]:
    """Estimate request/response token usage with best-effort tokenizer logic."""
    del provider  # reserved for future provider-specific estimation strategies

    request_payload = {
        "messages": _jsonable(messages),
        "tools": _jsonable(tools),
        "response_format": _jsonable(response_format),
    }
    request_text = json.dumps(request_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    estimate_input_tokens, method = _count_tokens(request_text, model)
    estimate_output_tokens, _ = _count_tokens(reply_text or "", model)

    return {
        "estimate_input_tokens": estimate_input_tokens,
        "estimate_output_tokens": estimate_output_tokens,
        "estimate_total_tokens": estimate_input_tokens + estimate_output_tokens,
        "estimate_method": method,
    }


def build_usage_record(
    response: Any,
    messages: List[Dict[str, Any]],
    reply_text: str,
    provider: str,
    model: str,
    tools: Any = None,
    response_format: Any = None,
    latency_ms: int = 0,
    timestamp: Optional[float] = None,
    anomaly_threshold: float = ANOMALY_THRESHOLD,
) -> Dict[str, Any]:
    """Build one normalized usage record for a request."""
    provider_usage = extract_provider_usage(response)
    estimate = estimate_usage(
        messages=messages,
        reply_text=reply_text,
        provider=provider,
        model=model,
        tools=tools,
        response_format=response_format,
    )

    p_input = provider_usage.get("input_tokens")
    p_output = provider_usage.get("output_tokens")
    p_total = provider_usage.get("total_tokens")
    e_input = estimate.get("estimate_input_tokens")
    e_output = estimate.get("estimate_output_tokens")
    e_total = estimate.get("estimate_total_tokens")

    provider_has_any = any(
        value is not None
        for value in (
            p_input,
            p_output,
            p_total,
            provider_usage.get("cached_tokens"),
            provider_usage.get("reasoning_tokens"),
            provider_usage.get("provider_cost"),
        )
    )
    estimate_has_any = any(value is not None for value in (e_input, e_output, e_total))
    provider_complete = p_input is not None and p_output is not None and p_total is not None

    if provider_complete:
        source = "provider"
    elif provider_has_any and estimate_has_any:
        source = "hybrid"
    elif provider_has_any:
        source = "provider"
    elif estimate_has_any:
        source = "estimate"
    else:
        source = "none"

    input_tokens = p_input if p_input is not None else (e_input if source in ("hybrid", "estimate") else None)
    output_tokens = p_output if p_output is not None else (e_output if source in ("hybrid", "estimate") else None)

    total_tokens = p_total
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if total_tokens is None and source in ("hybrid", "estimate"):
        total_tokens = e_total

    provider_compare_total = p_total
    if provider_compare_total is None and p_input is not None and p_output is not None:
        provider_compare_total = p_input + p_output

    anomaly_ratio: Optional[float] = None
    anomaly_flag = False
    if provider_compare_total is not None and e_total is not None:
        denominator = max(provider_compare_total, e_total, 1)
        anomaly_ratio = abs(provider_compare_total - e_total) / float(denominator)
        anomaly_flag = anomaly_ratio > anomaly_threshold

    if timestamp is None:
        timestamp = time.time()

    return {
        "provider": provider,
        "model": model,
        "source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": provider_usage.get("cached_tokens"),
        "reasoning_tokens": provider_usage.get("reasoning_tokens"),
        "provider_cost": provider_usage.get("provider_cost"),
        "provider_cost_currency": provider_usage.get("provider_cost_currency"),
        "estimate_input_tokens": e_input,
        "estimate_output_tokens": e_output,
        "estimate_total_tokens": e_total,
        "estimate_method": estimate.get("estimate_method"),
        "anomaly_flag": anomaly_flag,
        "anomaly_ratio": anomaly_ratio,
        "anomaly_threshold": anomaly_threshold,
        "latency_ms": int(latency_ms),
        "timestamp": float(timestamp),
    }


def accumulate_totals(totals: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """Accumulate one usage record into a totals dictionary."""
    target = totals
    target["calls"] = int(target.get("calls", 0)) + 1

    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = record.get(key)
        if isinstance(value, int):
            target[key] = int(target.get(key, 0)) + value

    source = record.get("source")
    if source == "provider":
        target["provider_calls"] = int(target.get("provider_calls", 0)) + 1
    elif source == "estimate":
        target["estimated_calls"] = int(target.get("estimated_calls", 0)) + 1
    elif source == "hybrid":
        target["hybrid_calls"] = int(target.get("hybrid_calls", 0)) + 1

    if bool(record.get("anomaly_flag")):
        target["anomaly_calls"] = int(target.get("anomaly_calls", 0)) + 1

    provider_cost = record.get("provider_cost")
    if isinstance(provider_cost, (int, float)):
        target["provider_cost_total"] = float(target.get("provider_cost_total", 0.0)) + float(provider_cost)

    return target


__all__ = [
    "ANOMALY_THRESHOLD",
    "init_usage_totals",
    "extract_provider_usage",
    "estimate_usage",
    "build_usage_record",
    "accumulate_totals",
]
