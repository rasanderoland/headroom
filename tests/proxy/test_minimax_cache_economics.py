"""MiniMax cache-economics profile: detection + multipliers."""

from __future__ import annotations

import pytest

from headroom.proxy.cost import (
    CostTracker,
    _CACHE_ECONOMICS,
    build_prefix_cache_stats,
    resolve_cache_economics_provider,
)
from headroom.proxy.prometheus_metrics import PrometheusMetrics


def test_cache_economics_has_minimax_profile() -> None:
    econ = _CACHE_ECONOMICS["minimax"]
    assert econ["read_multiplier"] == 0.1
    assert econ["write_multiplier"] == 1.0


@pytest.mark.parametrize(
    "model",
    [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "minimax-m3",
        "minimax/MiniMax-M3",
    ],
)
def test_minimax_model_name_resolves_to_minimax_key(model: str) -> None:
    assert resolve_cache_economics_provider(model=model, provider="anthropic") == "minimax"


@pytest.mark.parametrize(
    "url",
    [
        "https://api.minimax.io/anthropic",
        "https://api.minimax.io/v1",
        "http://API.MINIMAX.IO/anthropic",
    ],
)
def test_minimax_url_resolves_to_minimax_key(url: str) -> None:
    assert (
        resolve_cache_economics_provider(
            upstream_url=url,
            provider="anthropic",
            model="claude-opus-4-6",
        )
        == "minimax"
    )


def test_non_minimax_stays_on_explicit_provider() -> None:
    assert (
        resolve_cache_economics_provider(
            model="claude-opus-4-6",
            upstream_url="https://api.anthropic.com",
            provider="anthropic",
        )
        == "anthropic"
    )


def test_minimax_cache_read_uses_openai_style_multipliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cache_read=11000, cache_creation=0 → 0.1 read / 1.0 write (no write premium)."""
    metrics = PrometheusMetrics()
    metrics.cache_by_provider["minimax"].update(
        {
            "requests": 1,
            "hit_requests": 1,
            "cache_read_tokens": 11_000,
            "cache_write_tokens": 0,
            "cache_write_5m_tokens": 0,
            "cache_write_1h_tokens": 0,
            "cache_write_5m_requests": 0,
            "cache_write_1h_requests": 0,
            "uncached_input_tokens": 0,
        }
    )

    tracker = CostTracker()
    tracker._tokens_sent_by_model.update({"MiniMax-M3": 1})
    # $100 / 1M tokens → $0.0001 per token
    monkeypatch.setattr(CostTracker, "_get_list_price", lambda _self, _model: 100.0)

    stats = build_prefix_cache_stats(metrics, tracker)
    minimax = stats["by_provider"]["minimax"]

    # savings = 11000 * 0.0001 * (1.0 - 0.1) = 0.99
    assert minimax["savings_usd"] == 0.99
    assert minimax["write_premium_usd"] == 0.0
    assert minimax["net_savings_usd"] == 0.99
    assert minimax["write_premium"] == "none"
    assert _CACHE_ECONOMICS["minimax"]["write_multiplier"] == 1.0


def test_anthropic_still_uses_write_premium(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: provider_name=anthropic keeps the 1.25x write multiplier."""
    metrics = PrometheusMetrics()
    metrics.cache_by_provider["anthropic"].update(
        {
            "requests": 1,
            "hit_requests": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 1_000,
            "cache_write_5m_tokens": 1_000,
            "cache_write_1h_tokens": 0,
            "cache_write_5m_requests": 1,
            "cache_write_1h_requests": 0,
            "uncached_input_tokens": 0,
        }
    )

    tracker = CostTracker()
    tracker._tokens_sent_by_model.update({"claude-opus-4-6": 1})
    monkeypatch.setattr(CostTracker, "_get_list_price", lambda _self, _model: 100.0)

    stats = build_prefix_cache_stats(metrics, tracker)
    anthropic = stats["by_provider"]["anthropic"]

    # write_premium = 1000 * 0.0001 * (1.25 - 1.0) = 0.025
    assert _CACHE_ECONOMICS["anthropic"]["write_multiplier"] == 1.25
    assert anthropic["write_premium_usd"] == 0.025
    assert anthropic["write_premium"] == "25%"
