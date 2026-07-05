"""Recovery tool for fal.ai queue jobs (tools/video/_shared.py's request log).

A fal queue submit is billed the moment it is accepted, but the clip is only
collectable once the job reaches COMPLETED. If the process polling it dies
(crash, timeout, killed session) the billed job's request_id would otherwise
be lost. The 5 fal video tools (kling_video, minimax_video, veo_video,
seedance_video, grok_video_fal) write a `fal_requests.jsonl` ledger next to
their output as they submit/complete/fail/timeout via
`submit_and_poll_fal_queue(..., request_log_path=...)`. This tool reads that
ledger (or explicit URLs) to check status, collect a finished clip without
re-submitting and paying twice, or audit which requests are still
unresolved.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


# Events that mean a request no longer needs attention.
_TERMINAL_EVENTS = {"completed", "failed", "cancelled", "collected"}


class FalQueue(BaseTool):
    name = "fal_queue"
    version = "0.1.0"
    tier = ToolTier.CORE
    # NOTE: capability is deliberately NOT "video_generation" — this tool
    # recovers/audits jobs after the fact, it does not generate video. Using
    # "video_generation" would pollute video_selector's provider discovery.
    capability = "job_recovery"
    provider = "fal"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set FAL_KEY to your fal.ai API key.\n"
        "  Get one at https://fal.ai/dashboard/keys"
    )

    capabilities = ["job_status", "job_collect", "job_audit"]
    supports = {
        "status": True,
        "collect": True,
        "list": True,
    }
    best_for = [
        "recovering billed fal jobs after a timeout or crash",
        "checking fal queue job status with logs",
        "auditing unresolved fal requests from fal_requests.jsonl",
    ]
    not_good_for = ["generating new video (use video_selector or a provider tool)"]
    fallback_tools = []

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "collect", "list"],
            },
            "status_url": {"type": "string"},
            "response_url": {"type": "string"},
            "request_id": {"type": "string"},
            "request_log_path": {"type": "string"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["operation", "request_id", "status_url"]
    side_effects = ["reads fal request logs", "downloads video files on collect"]
    user_visible_verification = [
        "Confirm the collected clip plays correctly before reusing it downstream",
    ]

    def _get_api_key(self) -> str | None:
        return os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY")

    def get_status(self) -> ToolStatus:
        if self._get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    # ---- log helpers ----

    @staticmethod
    def _read_log_records(path: "str | Path") -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    def _resolve_urls(self, inputs: dict[str, Any]) -> tuple[str, str]:
        """Resolve (status_url, response_url) from explicit inputs or the request log."""
        status_url = inputs.get("status_url")
        response_url = inputs.get("response_url")
        if status_url and response_url:
            return status_url, response_url

        request_id = inputs.get("request_id")
        request_log_path = inputs.get("request_log_path")
        if not request_id or not request_log_path:
            raise ValueError(
                "Cannot resolve fal job URLs: provide either both status_url and "
                "response_url, or request_id + request_log_path to look them up."
            )

        log_path = Path(request_log_path)
        if not log_path.exists():
            raise ValueError(f"request_log_path does not exist: {log_path}")

        records = self._read_log_records(log_path)
        match = None
        for record in records:
            if record.get("event") == "submitted" and record.get("request_id") == request_id:
                match = record

        if match is None:
            raise ValueError(
                f"No 'submitted' record for request_id={request_id!r} found in {log_path}"
            )

        resolved_status = match.get("status_url")
        resolved_response = match.get("response_url")
        if not resolved_status or not resolved_response:
            raise ValueError(
                f"'submitted' record for request_id={request_id!r} is missing status_url/response_url"
            )
        return resolved_status, resolved_response

    @staticmethod
    def _extract_log_messages(raw_logs: Any, limit: int = 10) -> list[str]:
        messages: list[str] = []
        for entry in raw_logs or []:
            if isinstance(entry, dict):
                message = entry.get("message")
                if message:
                    messages.append(str(message))
            elif entry:
                messages.append(str(entry))
        return messages[-limit:]

    # ---- execution ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation")

        try:
            if operation == "status":
                return self._status(inputs)
            if operation == "collect":
                return self._collect(inputs)
            if operation == "list":
                return self._list(inputs)
            return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _status(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(success=False, error="FAL_KEY not set. " + self.install_instructions)

        status_url, response_url = self._resolve_urls(inputs)
        headers = {"Authorization": f"Key {api_key}"}

        resp = requests.get(f"{status_url}?logs=1", headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        return ToolResult(
            success=True,
            data={
                "request_id": inputs.get("request_id") or data.get("request_id"),
                "status": data.get("status"),
                "logs": self._extract_log_messages(data.get("logs")),
                "status_url": status_url,
                "response_url": response_url,
            },
        )

    def _collect(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(success=False, error="FAL_KEY not set. " + self.install_instructions)

        output_path_raw = inputs.get("output_path")
        if not output_path_raw:
            return ToolResult(success=False, error="collect requires output_path")

        status_url, response_url = self._resolve_urls(inputs)
        headers = {"Authorization": f"Key {api_key}"}

        status_resp = requests.get(f"{status_url}?logs=1", headers=headers, timeout=15)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        current_status = status_data.get("status", "UNKNOWN")

        if current_status != "COMPLETED":
            return ToolResult(
                success=False,
                error=(
                    f"fal job is not ready to collect (status={current_status}). "
                    "Retry later once the job reaches COMPLETED."
                ),
                data={"status": current_status},
            )

        result_resp = requests.get(response_url, headers=headers, timeout=30)
        result_resp.raise_for_status()
        result_data = result_resp.json()

        video_url = None
        if isinstance(result_data.get("video"), dict):
            video_url = result_data["video"].get("url")
        if not video_url:
            video_url = result_data.get("video_url")
        if not video_url:
            video_url = result_data.get("url")
        if not video_url:
            return ToolResult(
                success=False,
                error=(
                    "Could not find a video URL in the fal response. "
                    f"Available top-level keys: {sorted(result_data.keys())}"
                ),
            )

        download = requests.get(video_url, timeout=180)
        download.raise_for_status()

        output_path = Path(output_path_raw)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(download.content)

        request_log_path = inputs.get("request_log_path")
        if request_log_path:
            from tools.video._shared import _append_request_log

            _append_request_log(
                request_log_path,
                {
                    "request_id": inputs.get("request_id"),
                    "event": "collected",
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )

        from tools.video._shared import probe_output

        return ToolResult(
            success=True,
            data={
                "output": str(output_path),
                "output_path": str(output_path),
                "status": "COMPLETED",
                **probe_output(output_path),
            },
            artifacts=[str(output_path)],
        )

    def _list(self, inputs: dict[str, Any]) -> ToolResult:
        request_log_path = inputs.get("request_log_path")
        if not request_log_path:
            return ToolResult(success=False, error="list requires request_log_path")

        log_path = Path(request_log_path)
        if not log_path.exists():
            return ToolResult(success=False, error=f"request_log_path does not exist: {log_path}")

        records = self._read_log_records(log_path)

        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            request_id = record.get("request_id")
            if not request_id:
                continue
            groups.setdefault(request_id, []).append(record)

        unresolved: list[dict[str, Any]] = []
        resolved_count = 0
        for request_id, group_records in groups.items():
            latest_event = group_records[-1].get("event")
            if latest_event in _TERMINAL_EVENTS:
                resolved_count += 1
                continue

            submitted_record = next(
                (r for r in group_records if r.get("event") == "submitted"), None
            )
            unresolved.append(
                {
                    "request_id": request_id,
                    "status_url": (submitted_record or {}).get("status_url"),
                    "prompt": (submitted_record or {}).get("prompt"),
                    "submitted_at": (submitted_record or {}).get("at"),
                }
            )

        return ToolResult(
            success=True,
            data={
                "total_requests": len(groups),
                "unresolved": unresolved,
                "resolved_count": resolved_count,
            },
        )
