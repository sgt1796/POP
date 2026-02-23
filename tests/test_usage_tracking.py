from __future__ import annotations

from types import SimpleNamespace

from POP.usage_tracking import (
    accumulate_totals,
    build_usage_record,
    extract_provider_usage,
    init_usage_totals,
)


def test_extract_provider_usage_openai_like():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
            total_cost=0.42,
            currency="USD",
        )
    )

    usage = extract_provider_usage(response)

    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 18
    assert usage["cached_tokens"] == 3
    assert usage["reasoning_tokens"] == 2
    assert usage["provider_cost"] == 0.42
    assert usage["provider_cost_currency"] == "USD"


def test_extract_provider_usage_ollama_like_top_level():
    usage = extract_provider_usage({"prompt_eval_count": 9, "eval_count": 4})
    assert usage["input_tokens"] == 9
    assert usage["output_tokens"] == 4
    assert usage["total_tokens"] == 13


def test_build_usage_record_estimate_fallback():
    record = build_usage_record(
        response=SimpleNamespace(),
        messages=[{"role": "user", "content": "hello"}],
        reply_text="world",
        provider="openai",
        model="gpt-4o",
    )

    assert record["source"] == "estimate"
    assert record["total_tokens"] == record["estimate_total_tokens"]
    assert record["total_tokens"] is not None


def test_build_usage_record_hybrid_when_provider_usage_partial():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10))
    record = build_usage_record(
        response=response,
        messages=[{"role": "user", "content": "hello"}],
        reply_text="This is a longer completion for estimation.",
        provider="openai",
        model="gpt-4o",
    )

    assert record["source"] == "hybrid"
    assert record["input_tokens"] == 10
    assert record["output_tokens"] is not None
    assert record["total_tokens"] is not None


def test_build_usage_record_sets_anomaly_flag_above_threshold():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=400, completion_tokens=600, total_tokens=1000)
    )
    record = build_usage_record(
        response=response,
        messages=[{"role": "user", "content": "tiny"}],
        reply_text="short",
        provider="openai",
        model="gpt-4o",
    )

    assert record["anomaly_flag"] is True
    assert record["anomaly_ratio"] is not None
    assert record["anomaly_ratio"] > 0.5


def test_cost_only_present_when_provider_returns_it():
    no_cost = build_usage_record(
        response=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
        messages=[{"role": "user", "content": "x"}],
        reply_text="y",
        provider="openai",
        model="gpt-4o",
    )
    with_cost = build_usage_record(
        response=SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
                total_cost=1.25,
                cost_currency="USD",
            )
        ),
        messages=[{"role": "user", "content": "x"}],
        reply_text="y",
        provider="openai",
        model="gpt-4o",
    )

    assert no_cost["provider_cost"] is None
    assert no_cost["provider_cost_currency"] is None
    assert with_cost["provider_cost"] == 1.25
    assert with_cost["provider_cost_currency"] == "USD"


def test_accumulate_totals_tracks_counters():
    totals = init_usage_totals()
    record = {
        "source": "estimate",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "anomaly_flag": True,
        "provider_cost": None,
    }
    accumulate_totals(totals, record)

    assert totals["calls"] == 1
    assert totals["estimated_calls"] == 1
    assert totals["input_tokens"] == 10
    assert totals["output_tokens"] == 5
    assert totals["total_tokens"] == 15
    assert totals["anomaly_calls"] == 1
