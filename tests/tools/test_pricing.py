"""Tests for fal.ai pricing lookups (lib/pricing.py).

Covers: reading real pricing.yaml for a known tool, None for an unknown
tool, staleness_warning for a fresh vs. stale as_of, and the estimate_cost()
fallback contract -- deleting/breaking pricing.yaml must never change a
tool's cost-estimation behavior.
"""
from __future__ import annotations

from datetime import date

import pytest
import yaml

import lib.pricing as pricing
from lib.pricing import (
    get_tool_pricing,
    pricing_age_days,
    reset_pricing_cache,
    staleness_warning,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_pricing_cache()
    yield
    reset_pricing_cache()


# ----------------------------------------------------------------------
# get_tool_pricing against the real repo pricing.yaml
# ----------------------------------------------------------------------


class TestGetToolPricingRealFile:
    def test_returns_data_for_kling_video(self):
        result = get_tool_pricing("kling_video")
        assert result is not None
        assert result["rates"]["standard"] == 0.10
        assert result["rates"]["pro"] == 0.20
        assert result["rates"]["master"] == 0.30

    def test_returns_none_for_unknown_tool(self):
        assert get_tool_pricing("not_a_real_tool") is None


# ----------------------------------------------------------------------
# staleness_warning / pricing_age_days against a controlled tmp yaml
# ----------------------------------------------------------------------


class TestStalenessWarning:
    def test_none_for_fresh_as_of(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "pricing.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"as_of": date.today().isoformat(), "tools": {}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(pricing, "_PRICING_PATH", yaml_path)
        reset_pricing_cache()

        assert staleness_warning() is None

    def test_string_for_stale_as_of(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "pricing.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"as_of": "2020-01-01", "tools": {}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(pricing, "_PRICING_PATH", yaml_path)
        reset_pricing_cache()

        warning = staleness_warning()
        assert warning is not None
        assert "pricing.yaml" in warning

    def test_pricing_age_days_none_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pricing, "_PRICING_PATH", tmp_path / "does_not_exist.yaml")
        reset_pricing_cache()

        assert pricing_age_days() is None
        assert get_tool_pricing("kling_video") is None
        assert staleness_warning() is None


# ----------------------------------------------------------------------
# estimate_cost() fallback contract
# ----------------------------------------------------------------------


class TestEstimateCostFallback:
    def test_kling_falls_back_to_legacy_value_when_pricing_missing(self, monkeypatch):
        from tools.video.kling_video import KlingVideo

        # estimate_cost() does `from lib.pricing import get_tool_pricing`
        # lazily inside the method body, so patching the source module's
        # attribute is what the fresh import binds to.
        monkeypatch.setattr(pricing, "get_tool_pricing", lambda name: None)
        reset_pricing_cache()

        tool = KlingVideo()
        cost = tool.estimate_cost({"model_variant": "v3/standard", "duration": "5"})
        assert cost == 0.10
