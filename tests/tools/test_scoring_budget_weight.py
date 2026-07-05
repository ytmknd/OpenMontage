"""Tests for budget-pressure-adaptive cost weighting in lib/scoring.py (item 6).

lib.scoring.score_provider() only ever calls three methods on a candidate:
get_info(), get_status(), and estimate_cost(). _FakeTool below doubles just
those, mirroring how rank_providers() actually consumes candidates.
"""

from __future__ import annotations

from lib.scoring import rank_providers
from tools.base_tool import ToolStatus


class _FakeTool:
    def __init__(self, *, name, provider, cost, quality, stability, best_for):
        self.name = name
        self._cost = cost
        self._info = {
            "name": name,
            "provider": provider,
            "capability": "video_generation",
            "tier": "generate",
            "stability": stability,
            "best_for": best_for,
            "supports": {},
            "quality_score": quality,
            "historical_success_rate": None,
            "latency_p50_seconds": None,
            "runtime": "api",
        }

    def get_info(self):
        return dict(self._info)

    def get_status(self):
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs):
        return self._cost


def _build_tools():
    # Expensive, high-quality, production-stable provider.
    expensive = _FakeTool(
        name="premium_video",
        provider="premium",
        cost=2.00,
        quality=0.95,
        stability="production",
        best_for=["cinematic trailer footage"],
    )
    # Cheap, lower-quality, beta-stable provider (mirrors grok_video_fal-style
    # $0.05/s economics vs. a premium per-clip provider).
    cheap = _FakeTool(
        name="cheap_video",
        provider="cheap",
        cost=0.05,
        quality=0.55,
        stability="beta",
        best_for=["cinematic trailer footage"],
    )
    return expensive, cheap


def _task_context(budget_remaining_usd=None):
    context = {
        "intent": "cinematic trailer footage",
        "style_keywords": ["cinematic", "trailer"],
        "asset_type": "video",
    }
    if budget_remaining_usd is not None:
        context["budget_remaining_usd"] = budget_remaining_usd
    return context


def test_ranking_without_budget_context_matches_fixed_weight_baseline():
    expensive, cheap = _build_tools()

    ranking = rank_providers([expensive, cheap], _task_context())
    by_name = {r.tool_name: r for r in ranking}

    # No budget signal -> cost_weight stays at the historical default and the
    # quality/reliability-heavy premium provider wins, same as before D-4/item6.
    assert by_name["premium_video"].cost_weight == 0.10
    assert by_name["cheap_video"].cost_weight == 0.10
    assert ranking[0].tool_name == "premium_video"


def test_low_budget_remaining_uses_0_40_cost_weight_and_favors_cheap_provider():
    expensive, cheap = _build_tools()

    baseline = rank_providers([expensive, cheap], _task_context())
    baseline_by_name = {r.tool_name: r for r in baseline}
    baseline_gap = (
        baseline_by_name["premium_video"].weighted_score
        - baseline_by_name["cheap_video"].weighted_score
    )

    pressured = rank_providers([expensive, cheap], _task_context(budget_remaining_usd=0.30))
    pressured_by_name = {r.tool_name: r for r in pressured}
    pressured_gap = (
        pressured_by_name["premium_video"].weighted_score
        - pressured_by_name["cheap_video"].weighted_score
    )

    cheap_score = pressured_by_name["cheap_video"]
    assert cheap_score.cost_weight == 0.40
    assert cheap_score.to_dict()["effective_weights"]["cost_efficiency"] == 0.40

    # The premium/cheap gap must narrow under budget pressure, and with these
    # realistic inputs the cheap provider actually overtakes the premium one.
    assert pressured_gap < baseline_gap
    assert pressured[0].tool_name == "cheap_video"


def test_mid_budget_remaining_uses_0_25_cost_weight():
    expensive, cheap = _build_tools()

    ranking = rank_providers([expensive, cheap], _task_context(budget_remaining_usd=1.0))
    by_name = {r.tool_name: r for r in ranking}

    assert by_name["cheap_video"].cost_weight == 0.25
    assert by_name["premium_video"].cost_weight == 0.25


def test_ample_budget_remaining_keeps_default_0_10_cost_weight():
    expensive, cheap = _build_tools()

    ranking = rank_providers([expensive, cheap], _task_context(budget_remaining_usd=5.0))
    by_name = {r.tool_name: r for r in ranking}

    assert by_name["cheap_video"].cost_weight == 0.10
    assert by_name["premium_video"].cost_weight == 0.10
