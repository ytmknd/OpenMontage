"""Tests for BaseTool.execute_tracked (tools/base_tool.py) and the CostTracker
governance path it wires up (tools/cost_tracker.py).

Covers: estimate -> reserve -> execute (with retry) -> reconcile, approval
gating (single-action threshold + first-paid-use-of-tool), budget cap mode,
and persistence of approved tools across CostTracker instances.
"""
from __future__ import annotations

from typing import Any, Optional

import pytest

from lib.config_model import BudgetMode
from tools.base_tool import BaseTool, RetryPolicy, ToolResult
from tools.cost_tracker import CostTracker, EntryStatus


class _FakeTool(BaseTool):
    """A controllable BaseTool subclass: pops ToolResults off a queue."""

    name = "fake_tool"

    def __init__(self, results: list[ToolResult], cost: float = 0.10) -> None:
        self._results = list(results)
        self._cost = cost
        self.retry_policy = RetryPolicy(
            max_retries=2, backoff_seconds=0.01, retryable_errors=["rate_limit", "timeout"]
        )
        self.call_count = 0

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return self._cost

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        self.call_count += 1
        return self._results.pop(0)


def _make_tracker(tmp_path, **kwargs) -> CostTracker:
    kwargs.setdefault("budget_total_usd", 10.0)
    kwargs.setdefault("mode", BudgetMode.WARN)
    kwargs.setdefault("cost_log_path", tmp_path / "cost_log.json")
    return CostTracker(**kwargs)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("tools.base_tool.time.sleep", lambda _seconds: None)


def _entry(tracker: CostTracker, entry_id: str) -> dict[str, Any]:
    for entry in tracker.entries:
        if entry["id"] == entry_id:
            return entry
    raise AssertionError(f"entry {entry_id!r} not found")


# ----------------------------------------------------------------------
# 1. Success on first attempt
# ----------------------------------------------------------------------


def test_success_reconciles_completed_with_actual_cost(tmp_path):
    tracker = _make_tracker(
        tmp_path,
        single_action_approval_usd=10.0,
        require_approval_for_new_paid_tool=False,
    )
    tool = _FakeTool([ToolResult(success=True, cost_usd=0.12)])

    result = tool.execute_tracked({}, tracker)

    assert result.success is True
    entry = _entry(tracker, result.data["cost_entry_id"])
    assert entry["status"] == EntryStatus.COMPLETED.value
    assert entry["actual_usd"] == pytest.approx(0.12)
    assert result.data["attempts"] == 1
    assert result.data["cost_tracked"] is True
    assert "cost_entry_id" in result.data
    assert tool.call_count == 1


# ----------------------------------------------------------------------
# 2. Non-retryable failure
# ----------------------------------------------------------------------


def test_non_retryable_failure_stops_after_one_attempt(tmp_path):
    tracker = _make_tracker(
        tmp_path,
        single_action_approval_usd=10.0,
        require_approval_for_new_paid_tool=False,
    )
    tool = _FakeTool([ToolResult(success=False, error="invalid prompt")])

    result = tool.execute_tracked({}, tracker)

    assert result.success is False
    assert tool.call_count == 1
    entry = _entry(tracker, result.data["cost_entry_id"])
    assert entry["status"] == EntryStatus.FAILED.value
    assert result.data["attempts"] == 1


# ----------------------------------------------------------------------
# 3. Retryable failures then success
# ----------------------------------------------------------------------


def test_retryable_rate_limit_failures_then_success(tmp_path):
    tracker = _make_tracker(
        tmp_path,
        single_action_approval_usd=10.0,
        require_approval_for_new_paid_tool=False,
    )
    tool = _FakeTool(
        [
            ToolResult(success=False, error="HTTP 429 Too Many Requests"),
            ToolResult(success=False, error="HTTP 429 Too Many Requests"),
            ToolResult(success=True, cost_usd=0.15),
        ]
    )

    result = tool.execute_tracked({}, tracker)

    assert result.success is True
    assert result.data["attempts"] == 3
    assert tool.call_count == 3
    entry = _entry(tracker, result.data["cost_entry_id"])
    assert entry["status"] == EntryStatus.COMPLETED.value
    assert entry["actual_usd"] == pytest.approx(0.15)


# ----------------------------------------------------------------------
# 4. Blocked by single-action approval threshold
# ----------------------------------------------------------------------


def test_blocked_by_single_action_approval_threshold(tmp_path):
    tracker = _make_tracker(
        tmp_path,
        single_action_approval_usd=0.50,
        require_approval_for_new_paid_tool=False,
    )
    tool = _FakeTool([ToolResult(success=True, cost_usd=2.00)], cost=2.00)

    result = tool.execute_tracked({}, tracker)

    assert result.success is False
    assert result.data["blocked_by"] == "approval_required"
    assert tool.call_count == 0

    entry = _entry(tracker, result.data["cost_entry_id"])
    assert entry["status"] == EntryStatus.ESTIMATED.value


# ----------------------------------------------------------------------
# 5. approved=True bypasses the single-action threshold
# ----------------------------------------------------------------------


def test_approved_true_bypasses_single_action_threshold(tmp_path):
    tracker = _make_tracker(
        tmp_path,
        single_action_approval_usd=0.50,
        require_approval_for_new_paid_tool=False,
    )
    tool = _FakeTool([ToolResult(success=True, cost_usd=2.00)], cost=2.00)

    result = tool.execute_tracked({}, tracker, approved=True)

    assert result.success is True
    assert tool.call_count == 1
    entry = _entry(tracker, result.data["cost_entry_id"])
    assert entry["status"] == EntryStatus.COMPLETED.value


# ----------------------------------------------------------------------
# 6. First paid use of an unapproved tool, then approve and retry
# ----------------------------------------------------------------------


def test_first_paid_use_blocked_then_allowed_after_approve_tool(tmp_path):
    tracker = _make_tracker(
        tmp_path,
        single_action_approval_usd=0.50,
        require_approval_for_new_paid_tool=True,
    )
    tool = _FakeTool([ToolResult(success=True, cost_usd=0.10)], cost=0.10)

    blocked = tool.execute_tracked({}, tracker)
    assert blocked.success is False
    assert blocked.data["blocked_by"] == "approval_required"
    assert tool.call_count == 0

    tracker.approve_tool(tool.name)

    tool2 = _FakeTool([ToolResult(success=True, cost_usd=0.10)], cost=0.10)
    allowed = tool2.execute_tracked({}, tracker)
    assert allowed.success is True
    assert tool2.call_count == 1


# ----------------------------------------------------------------------
# 7. Approval persistence across CostTracker instances
# ----------------------------------------------------------------------


def test_approved_tools_persist_across_tracker_instances(tmp_path):
    cost_log_path = tmp_path / "cost_log.json"
    tracker1 = _make_tracker(
        tmp_path,
        cost_log_path=cost_log_path,
        single_action_approval_usd=0.50,
        require_approval_for_new_paid_tool=True,
    )
    tracker1.approve_tool("fake_tool")

    tracker2 = CostTracker(
        budget_total_usd=10.0,
        mode=BudgetMode.WARN,
        cost_log_path=cost_log_path,
        single_action_approval_usd=10.0,
        require_approval_for_new_paid_tool=True,
    )

    entry_id = tracker2.estimate("fake_tool", "execute", 0.10)
    # Should not raise ApprovalRequiredError: the tool was approved and
    # persisted by tracker1, and single_action_approval_usd is high enough
    # here that the threshold check itself is not the blocker.
    tracker2.reserve(entry_id)
    entry = _entry(tracker2, entry_id)
    assert entry["status"] == EntryStatus.RESERVED.value


# ----------------------------------------------------------------------
# 8. CAP mode budget exceeded
# ----------------------------------------------------------------------


def test_cap_mode_blocks_when_estimate_exceeds_budget(tmp_path):
    tracker = _make_tracker(
        tmp_path,
        budget_total_usd=10.0,
        mode=BudgetMode.CAP,
        single_action_approval_usd=100.0,
        require_approval_for_new_paid_tool=False,
    )
    tool = _FakeTool([ToolResult(success=True, cost_usd=50.0)], cost=50.0)

    result = tool.execute_tracked({}, tracker)

    assert result.success is False
    assert result.data["blocked_by"] == "budget_exceeded"
    assert tool.call_count == 0
