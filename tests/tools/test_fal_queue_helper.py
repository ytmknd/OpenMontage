"""Tests for the shared fal.ai queue submit/poll helper (tools/video/_shared.py).

Bug this guards against: the 5 fal video tools (kling, minimax, veo,
seedance, grok) each hand-rolled an unbounded `while True` poll loop with a
fixed 5s sleep and no deadline, and on FAILED/CANCELLED surfaced only the
bare status string — dropping the moderation/error detail that fal returns
via the `?logs=1` status endpoint. `submit_and_poll_fal_queue` centralizes
deadline + exponential backoff + rich error extraction so all 5 tools get
the fix at once.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.video._shared import FalJobError, submit_and_poll_fal_queue


def _mock_response(json_data=None, status_code=200, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


class TestSubmitAndPollFalQueueHappyPath:
    def test_returns_final_result_after_completion(self):
        submit_resp = _mock_response(
            {
                "request_id": "req-123",
                "status_url": "https://queue.fal.run/req-123/status",
                "response_url": "https://queue.fal.run/req-123",
            }
        )
        in_progress = _mock_response({"status": "IN_PROGRESS"})
        completed = _mock_response({"status": "COMPLETED"})
        result_resp = _mock_response({"video": {"url": "https://cdn.fal.ai/out.mp4"}})

        with patch("requests.post", return_value=submit_resp) as mock_post, \
             patch("requests.get", side_effect=[in_progress, completed, result_resp]) as mock_get, \
             patch("time.sleep") as mock_sleep:
            result = submit_and_poll_fal_queue(
                "https://queue.fal.run/fal-ai/some-model",
                {"prompt": "a cat"},
                "test-key",
            )

        assert result == {"video": {"url": "https://cdn.fal.ai/out.mp4"}}
        mock_post.assert_called_once()
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2


class TestSubmitAndPollFalQueueFailure:
    def test_failed_status_raises_with_log_tail_and_request_id(self):
        submit_resp = _mock_response(
            {
                "request_id": "req-456",
                "status_url": "https://queue.fal.run/req-456/status",
                "response_url": "https://queue.fal.run/req-456",
            }
        )
        failed_resp = _mock_response(
            {
                "status": "FAILED",
                "logs": [{"message": "moderation blocked"}],
            }
        )

        with patch("requests.post", return_value=submit_resp), \
             patch("requests.get", return_value=failed_resp), \
             patch("time.sleep"):
            with pytest.raises(FalJobError) as exc_info:
                submit_and_poll_fal_queue(
                    "https://queue.fal.run/fal-ai/some-model",
                    {"prompt": "a cat"},
                    "test-key",
                )

        message = str(exc_info.value)
        assert "moderation blocked" in message
        assert "req-456" in message


class TestSubmitAndPollFalQueueDeadline:
    def test_timeout_error_includes_status_url(self):
        submit_resp = _mock_response(
            {
                "request_id": "req-789",
                "status_url": "https://queue.fal.run/req-789/status",
                "response_url": "https://queue.fal.run/req-789",
            }
        )
        in_queue_resp = _mock_response({"status": "IN_QUEUE"})

        with patch("requests.post", return_value=submit_resp), \
             patch("requests.get", return_value=in_queue_resp), \
             patch("time.sleep"):
            with pytest.raises(TimeoutError) as exc_info:
                submit_and_poll_fal_queue(
                    "https://queue.fal.run/fal-ai/some-model",
                    {"prompt": "a cat"},
                    "test-key",
                    timeout=0.01,
                    poll_interval=0.01,
                )

        assert "https://queue.fal.run/req-789/status" in str(exc_info.value)


class TestSubmitAndPollFalQueueSubmitRejected:
    def test_submit_http_error_includes_status_and_body(self):
        error_resp = MagicMock()
        error_resp.status_code = 422
        error_resp.text = '{"detail":"bad param"}'

        http_error = requests.HTTPError("422 Client Error")
        http_error.response = error_resp

        submit_resp = MagicMock()
        submit_resp.raise_for_status.side_effect = http_error

        with patch("requests.post", return_value=submit_resp):
            with pytest.raises(FalJobError) as exc_info:
                submit_and_poll_fal_queue(
                    "https://queue.fal.run/fal-ai/some-model",
                    {"prompt": "a cat"},
                    "test-key",
                )

        message = str(exc_info.value)
        assert "422" in message
        assert "bad param" in message
