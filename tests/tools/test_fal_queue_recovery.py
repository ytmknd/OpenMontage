"""Tests for the fal.ai request log (recovery ledger) and its consumer,
the `fal_queue` tool (tools/video/fal_queue.py).

Bug this guards against: a fal queue submit is billed immediately, but if
the polling process dies mid-poll (crash, timeout, killed session) the
request_id is lost and the already-billed clip becomes unrecoverable.
`submit_and_poll_fal_queue(..., request_log_path=...)` writes a JSONL ledger
of every state transition (submitted/completed/failed/cancelled/timeout);
`fal_queue` reads that ledger back to recover or audit those jobs.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.video._shared import FalJobError, submit_and_poll_fal_queue
from tools.video.fal_queue import FalQueue


def _mock_response(json_data=None, status_code=200, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- submit_and_poll_fal_queue: request log persistence ----


class TestRequestLogHappyPath:
    def test_writes_submitted_then_completed(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        submit_resp = _mock_response(
            {
                "request_id": "req-log-1",
                "status_url": "https://queue.fal.run/req-log-1/status",
                "response_url": "https://queue.fal.run/req-log-1",
            }
        )
        completed = _mock_response({"status": "COMPLETED"})
        result_resp = _mock_response({"video": {"url": "https://cdn.fal.ai/out.mp4"}})

        with patch("requests.post", return_value=submit_resp), \
             patch("requests.get", side_effect=[completed, result_resp]), \
             patch("time.sleep"):
            result = submit_and_poll_fal_queue(
                "https://queue.fal.run/fal-ai/some-model",
                {"prompt": "a cat"},
                "test-key",
                request_log_path=log_path,
            )

        assert result == {"video": {"url": "https://cdn.fal.ai/out.mp4"}}
        records = _read_jsonl(log_path)
        assert len(records) == 2
        assert records[0]["event"] == "submitted"
        assert records[0]["request_id"] == "req-log-1"
        assert records[1]["event"] == "completed"
        assert records[1]["request_id"] == "req-log-1"


class TestRequestLogTimeout:
    def test_writes_timeout_record_and_still_raises(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        submit_resp = _mock_response(
            {
                "request_id": "req-log-2",
                "status_url": "https://queue.fal.run/req-log-2/status",
                "response_url": "https://queue.fal.run/req-log-2",
            }
        )
        in_queue_resp = _mock_response({"status": "IN_QUEUE"})

        with patch("requests.post", return_value=submit_resp), \
             patch("requests.get", return_value=in_queue_resp), \
             patch("time.sleep"):
            with pytest.raises(TimeoutError):
                submit_and_poll_fal_queue(
                    "https://queue.fal.run/fal-ai/some-model",
                    {"prompt": "a cat"},
                    "test-key",
                    timeout=0.01,
                    poll_interval=0.01,
                    request_log_path=log_path,
                )

        records = _read_jsonl(log_path)
        assert len(records) == 2
        assert records[0]["event"] == "submitted"
        assert records[1]["event"] == "timeout"
        assert records[1]["request_id"] == "req-log-2"


class TestRequestLogFailure:
    def test_writes_failed_record_before_raising(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        submit_resp = _mock_response(
            {
                "request_id": "req-log-3",
                "status_url": "https://queue.fal.run/req-log-3/status",
                "response_url": "https://queue.fal.run/req-log-3",
            }
        )
        failed_resp = _mock_response(
            {"status": "FAILED", "logs": [{"message": "moderation blocked"}]}
        )

        with patch("requests.post", return_value=submit_resp), \
             patch("requests.get", return_value=failed_resp), \
             patch("time.sleep"):
            with pytest.raises(FalJobError):
                submit_and_poll_fal_queue(
                    "https://queue.fal.run/fal-ai/some-model",
                    {"prompt": "a cat"},
                    "test-key",
                    request_log_path=log_path,
                )

        records = _read_jsonl(log_path)
        assert len(records) == 2
        assert records[0]["event"] == "submitted"
        assert records[1]["event"] == "failed"
        assert records[1]["request_id"] == "req-log-3"


class TestRequestLogNeverRaisesOnLoggingFailure:
    def test_unwritable_log_path_does_not_break_job(self, tmp_path):
        # tmp_path/"blocker" is a FILE, not a directory: mkdir(parents=True)
        # for "blocker/log.jsonl"'s parent will fail. The job must still
        # succeed — a logging failure must never fail a billed job.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        log_path = blocker / "log.jsonl"

        submit_resp = _mock_response(
            {
                "request_id": "req-log-4",
                "status_url": "https://queue.fal.run/req-log-4/status",
                "response_url": "https://queue.fal.run/req-log-4",
            }
        )
        completed = _mock_response({"status": "COMPLETED"})
        result_resp = _mock_response({"video": {"url": "https://cdn.fal.ai/out.mp4"}})

        with patch("requests.post", return_value=submit_resp), \
             patch("requests.get", side_effect=[completed, result_resp]), \
             patch("time.sleep"):
            result = submit_and_poll_fal_queue(
                "https://queue.fal.run/fal-ai/some-model",
                {"prompt": "a cat"},
                "test-key",
                request_log_path=log_path,
            )

        assert result == {"video": {"url": "https://cdn.fal.ai/out.mp4"}}
        assert not log_path.exists()


# ---- FalQueue tool: collect ----


class TestFalQueueCollect:
    def test_collect_downloads_and_marks_collected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        log_path = tmp_path / "log.jsonl"
        output_path = tmp_path / "out.mp4"

        status_resp = _mock_response({"status": "COMPLETED"})
        result_resp = _mock_response({"video": {"url": "https://cdn.fal.ai/out.mp4"}})
        download_resp = MagicMock()
        download_resp.status_code = 200
        download_resp.content = b"fake video bytes"
        download_resp.raise_for_status.return_value = None

        tool = FalQueue()
        with patch("requests.get", side_effect=[status_resp, result_resp, download_resp]):
            result = tool.execute(
                {
                    "operation": "collect",
                    "status_url": "https://queue.fal.run/req-1/status",
                    "response_url": "https://queue.fal.run/req-1",
                    "request_id": "req-1",
                    "output_path": str(output_path),
                    "request_log_path": str(log_path),
                }
            )

        assert result.success is True
        assert output_path.read_bytes() == b"fake video bytes"
        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["event"] == "collected"
        assert records[0]["request_id"] == "req-1"

    def test_collect_on_in_progress_reports_status_and_fails(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        status_resp = _mock_response({"status": "IN_PROGRESS"})

        tool = FalQueue()
        with patch("requests.get", return_value=status_resp):
            result = tool.execute(
                {
                    "operation": "collect",
                    "status_url": "https://queue.fal.run/req-2/status",
                    "response_url": "https://queue.fal.run/req-2",
                    "output_path": "/tmp/whatever_not_written.mp4",
                }
            )

        assert result.success is False
        assert "IN_PROGRESS" in result.error


# ---- FalQueue tool: list ----


class TestFalQueueList:
    def test_list_reports_only_the_submitted_only_request_as_unresolved(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        lines = [
            {
                "request_id": "req-a",
                "event": "submitted",
                "status_url": "https://queue.fal.run/req-a/status",
                "prompt": "cat",
                "at": "2026-01-01T00:00:00+00:00",
            },
            {
                "request_id": "req-b",
                "event": "submitted",
                "status_url": "https://queue.fal.run/req-b/status",
                "prompt": "dog",
                "at": "2026-01-01T00:01:00+00:00",
            },
            {
                "request_id": "req-b",
                "event": "completed",
                "at": "2026-01-01T00:02:00+00:00",
            },
        ]
        log_path.write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
        )

        tool = FalQueue()
        result = tool.execute({"operation": "list", "request_log_path": str(log_path)})

        assert result.success is True
        assert result.data["total_requests"] == 2
        assert result.data["resolved_count"] == 1
        assert len(result.data["unresolved"]) == 1
        assert result.data["unresolved"][0]["request_id"] == "req-a"
        assert result.data["unresolved"][0]["prompt"] == "cat"


# ---- FalQueue tool: resolving URLs from request_log_path + request_id ----


class TestFalQueueUrlResolutionFromLog:
    def test_status_resolves_urls_from_log(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        log_path = tmp_path / "log.jsonl"
        record = {
            "request_id": "req-resolve",
            "status_url": "https://queue.fal.run/req-resolve/status",
            "response_url": "https://queue.fal.run/req-resolve",
            "event": "submitted",
            "prompt": "a cat",
            "at": "2026-01-01T00:00:00+00:00",
        }
        log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        status_resp = _mock_response({"status": "IN_PROGRESS", "logs": []})
        tool = FalQueue()
        with patch("requests.get", return_value=status_resp) as mock_get:
            result = tool.execute(
                {
                    "operation": "status",
                    "request_id": "req-resolve",
                    "request_log_path": str(log_path),
                }
            )

        assert result.success is True
        assert result.data["status_url"] == "https://queue.fal.run/req-resolve/status"
        assert result.data["response_url"] == "https://queue.fal.run/req-resolve"
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert called_url.startswith("https://queue.fal.run/req-resolve/status")

    def test_collect_resolves_urls_from_log(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        log_path = tmp_path / "log.jsonl"
        record = {
            "request_id": "req-resolve-2",
            "status_url": "https://queue.fal.run/req-resolve-2/status",
            "response_url": "https://queue.fal.run/req-resolve-2",
            "event": "submitted",
            "prompt": "a dog",
            "at": "2026-01-01T00:00:00+00:00",
        }
        log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        output_path = tmp_path / "out.mp4"

        status_resp = _mock_response({"status": "COMPLETED"})
        result_resp = _mock_response({"video": {"url": "https://cdn.fal.ai/out.mp4"}})
        download_resp = MagicMock()
        download_resp.status_code = 200
        download_resp.content = b"bytes"
        download_resp.raise_for_status.return_value = None

        tool = FalQueue()
        with patch("requests.get", side_effect=[status_resp, result_resp, download_resp]):
            result = tool.execute(
                {
                    "operation": "collect",
                    "request_id": "req-resolve-2",
                    "request_log_path": str(log_path),
                    "output_path": str(output_path),
                }
            )

        assert result.success is True
        assert output_path.read_bytes() == b"bytes"
