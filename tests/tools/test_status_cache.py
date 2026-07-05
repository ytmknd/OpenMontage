"""Tests for BaseTool.get_status_cached() (D-4).

get_status_cached() must call the polymorphic get_status() at most once per
TTL window, so registry hot paths (get_by_status, provider_menu) stop
re-running shutil.which/__import__/env checks for every query.
"""

from __future__ import annotations

from tools.base_tool import BaseTool, ToolResult, ToolStatus


class _CountingTool(BaseTool):
    """Minimal concrete BaseTool whose get_status() counts invocations."""

    name = "counting_tool"

    def __init__(self) -> None:
        self.status_calls = 0

    def get_status(self) -> ToolStatus:
        self.status_calls += 1
        return ToolStatus.AVAILABLE

    def execute(self, inputs):
        return ToolResult(success=True)


def test_get_status_cached_calls_get_status_once_within_ttl():
    tool = _CountingTool()

    first = tool.get_status_cached(ttl_seconds=60.0)
    second = tool.get_status_cached(ttl_seconds=60.0)

    assert first == ToolStatus.AVAILABLE
    assert second == ToolStatus.AVAILABLE
    assert tool.status_calls == 1


def test_get_status_cached_with_zero_ttl_recomputes_every_call():
    tool = _CountingTool()

    tool.get_status_cached(ttl_seconds=0.0)
    tool.get_status_cached(ttl_seconds=0.0)
    tool.get_status_cached(ttl_seconds=0.0)

    assert tool.status_calls == 3


def test_get_status_cached_default_ttl_is_used_when_unspecified():
    tool = _CountingTool()

    tool.get_status_cached()
    tool.get_status_cached()

    assert tool.status_calls == 1


def test_get_status_cached_expires_after_ttl(monkeypatch):
    tool = _CountingTool()

    times = iter([100.0, 100.0, 200.0])
    monkeypatch.setattr("tools.base_tool.time.monotonic", lambda: next(times))

    tool.get_status_cached(ttl_seconds=60.0)  # cached at t=100
    tool.get_status_cached(ttl_seconds=60.0)  # served from cache, checked at t=100
    tool.get_status_cached(ttl_seconds=60.0)  # t=200, well past ttl -> recompute

    assert tool.status_calls == 2
