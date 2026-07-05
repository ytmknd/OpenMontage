"""Tests for the fal generation clip cache (tools/video/_shared.py wiring).

Bug this guards against: same-input re-runs of the 5 fal video tools
(kling/minimax/veo/seedance/grok) re-submitted to fal.ai and re-billed even
though the identical clip was already on disk. ``fal_cache_lookup`` /
``fal_cache_store`` route those re-runs through the local clip cache keyed
by ``{tool.name}_{idempotency_key}`` — and the idempotency key now covers
ALL generation-relevant inputs (e.g. aspect_ratio), so a 9:16 render can
never cache-hit a 16:9 clip.

No network: requests.post/get and time.sleep are patched, mirroring
tests/tools/test_fal_queue_helper.py.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.video import clip_cache
from tools.video.kling_video import KlingVideo

# Must be >= clip_cache._MIN_USABLE_BYTES (1024) or ingest rejects the clip.
_FAKE_VIDEO_BYTES = b"\x00fake-mp4-bytes\x00" * 128


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the default clip cache at a per-test tmp dir and reset around it."""
    monkeypatch.setenv("OPENMONTAGE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.delenv("OPENMONTAGE_CLIP_CACHE", raising=False)
    clip_cache.reset_default_cache()
    try:
        yield
    finally:
        clip_cache.reset_default_cache()


def _mock_response(json_data=None, content: bytes = b""):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.raise_for_status.return_value = None
    return resp


def _run_kling(inputs: dict):
    """Execute KlingVideo with a fully mocked fal queue round-trip.

    Returns (result, mock_post). requests.get serves: status poll
    (COMPLETED), result fetch (video url), then the video download.
    """
    submit_resp = _mock_response(
        {
            "request_id": "req-1",
            "status_url": "https://queue.fal.run/req-1/status",
            "response_url": "https://queue.fal.run/req-1",
        }
    )
    completed = _mock_response({"status": "COMPLETED"})
    result_resp = _mock_response({"video": {"url": "https://cdn.fal.ai/out.mp4"}})
    download_resp = _mock_response(content=_FAKE_VIDEO_BYTES)

    with patch("requests.post", return_value=submit_resp) as mock_post, \
         patch("requests.get", side_effect=[completed, result_resp, download_resp]), \
         patch("time.sleep"):
        result = KlingVideo().execute(inputs)

    return result, mock_post


def _base_inputs(tmp_path: Path, name: str = "first.mp4") -> dict:
    return {
        "prompt": "a red fox running through snow",
        "model_variant": "v3/standard",
        "duration": "5",
        "aspect_ratio": "16:9",
        "output_path": str(tmp_path / "out" / name),
    }


class TestWarmCacheHit:
    def test_second_run_serves_from_cache_without_billing(self, tmp_path):
        first_inputs = _base_inputs(tmp_path, "first.mp4")
        result1, mock_post1 = _run_kling(first_inputs)
        assert result1.success, result1.error
        assert mock_post1.called
        assert Path(first_inputs["output_path"]).read_bytes() == _FAKE_VIDEO_BYTES

        # Same inputs, DIFFERENT output_path (excluded from the key).
        second_inputs = _base_inputs(tmp_path, "second.mp4")
        with patch("requests.post") as mock_post2, patch("time.sleep"):
            result2 = KlingVideo().execute(second_inputs)

        assert result2.success, result2.error
        assert result2.data["cache_hit"] is True
        assert result2.cost_usd == 0.0
        mock_post2.assert_not_called()
        assert Path(second_inputs["output_path"]).read_bytes() == _FAKE_VIDEO_BYTES
        assert result2.artifacts == [second_inputs["output_path"]]


class TestAspectRatioInKey:
    def test_different_aspect_ratio_misses_cache(self, tmp_path):
        result1, _ = _run_kling(_base_inputs(tmp_path, "first.mp4"))
        assert result1.success, result1.error

        # A 9:16 request must NOT be served the cached 16:9 clip.
        vertical_inputs = _base_inputs(tmp_path, "vertical.mp4")
        vertical_inputs["aspect_ratio"] = "9:16"
        result2, mock_post2 = _run_kling(vertical_inputs)

        assert result2.success, result2.error
        assert mock_post2.called
        assert "cache_hit" not in result2.data


class TestForceRegenerate:
    def test_force_regenerate_bypasses_warm_cache(self, tmp_path):
        result1, _ = _run_kling(_base_inputs(tmp_path, "first.mp4"))
        assert result1.success, result1.error

        forced_inputs = _base_inputs(tmp_path, "forced.mp4")
        forced_inputs["force_regenerate"] = True
        result2, mock_post2 = _run_kling(forced_inputs)

        assert result2.success, result2.error
        assert mock_post2.called
        assert "cache_hit" not in result2.data


class TestCacheDisabled:
    def test_env_kill_switch_disables_ingest_and_lookup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENMONTAGE_CLIP_CACHE", "0")

        result1, _ = _run_kling(_base_inputs(tmp_path, "first.mp4"))
        assert result1.success, result1.error

        # Nothing was ingested into the cache.
        assert clip_cache.get_default_cache().stats()["entry_count"] == 0

        # Second identical call misses and re-submits.
        result2, mock_post2 = _run_kling(_base_inputs(tmp_path, "second.mp4"))
        assert result2.success, result2.error
        assert mock_post2.called
        assert "cache_hit" not in result2.data
