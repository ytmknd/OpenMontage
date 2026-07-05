"""Grok Imagine Video generation via fal.ai gateway.

xAI's grok-imagine-video hosted on fal.ai — cost-effective clips
($0.05/s at 480p, $0.07/s at 720p) with native audio and wide
aspect-ratio support including vertical 9:16. Max resolution is 720p;
plan an upscale step when the deliverable is 1080p.
"""

from __future__ import annotations

import os
import time
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


class GrokVideoFal(BaseTool):
    name = "grok_video_fal"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "grok"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set FAL_KEY to your fal.ai API key.\n"
        "  Get one at https://fal.ai/dashboard/keys"
    )
    agent_skills = ["ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "native_audio": True,
        "vertical_9_16": True,
    }
    best_for = [
        "cost-effective clips via FAL ($0.05/s at 480p, $0.07/s at 720p)",
        "vertical 9:16 and unusual aspect ratios (7 ratios supported)",
        "image-to-video with composition locked by a reference frame",
    ]
    not_good_for = [
        "1080p-native deliverables (720p max — needs upscale)",
        "offline generation",
    ]
    fallback_tools = ["veo_video", "minimax_video", "kling_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video", "reference_to_video"],
                "default": "text_to_video",
            },
            "duration": {
                "type": "integer",
                "default": 6,
                "description": "Video duration in seconds (fal default 6)",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"],
                "default": "16:9",
                "description": "Ignored for image_to_video (follows the image)",
            },
            "resolution": {
                "type": "string",
                "enum": ["480p", "720p"],
                "default": "720p",
            },
            "image_url": {"type": "string", "description": "Reference image URL for image_to_video"},
            "image_path": {"type": "string", "description": "Local start-frame path for image_to_video (auto-uploaded to fal storage)"},
            "reference_image_urls": {
                "type": "array", "items": {"type": "string"},
                "description": "Reference image URLs for reference_to_video (character/style consistency)",
            },
            "reference_image_paths": {
                "type": "array", "items": {"type": "string"},
                "description": "Local reference image paths for reference_to_video (auto-uploaded to fal storage)",
            },
            "output_path": {"type": "string"},
            "force_regenerate": {
                "type": "boolean",
                "default": False,
                "description": "Bypass the local clip cache and re-generate (re-bills the provider).",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = [
        "aspect_ratio",
        "duration",
        "image_path",
        "image_url",
        "operation",
        "prompt",
        "reference_image_paths",
        "reference_image_urls",
        "resolution",
    ]
    side_effects = ["writes video file to output_path", "calls fal.ai API"]
    user_visible_verification = ["Watch generated clip for motion coherence and visual quality"]

    # Fallback per-second rates (USD) used when pricing.yaml is missing,
    # malformed, or lacks a "grok_video_fal" entry. Keep in sync with
    # pricing.yaml's tools.grok_video_fal.rates.
    _FALLBACK_RATES = {"480p": 0.05, "720p": 0.07}

    def _get_api_key(self) -> str | None:
        return os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY")

    def get_status(self) -> ToolStatus:
        if self._get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        duration = int(inputs.get("duration", 6))
        resolution = inputs.get("resolution", "720p")

        rate = None
        try:
            from lib.pricing import get_tool_pricing

            pricing = get_tool_pricing(self.name)
            if pricing:
                rate = (pricing.get("rates") or {}).get(resolution)
        except Exception:
            rate = None
        if rate is None:
            rate = self._FALLBACK_RATES.get(resolution, 0.07)

        return round(rate * duration, 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 120.0  # ~2 minutes typical for queue + generation

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="FAL_KEY not set. " + self.install_instructions,
            )

        import requests

        start = time.time()
        operation = inputs.get("operation", "text_to_video")
        operation_path = operation.replace("_", "-")
        model_path = f"xai/grok-imagine-video/{operation_path}"

        payload: dict[str, Any] = {
            "prompt": inputs["prompt"],
            "duration": int(inputs.get("duration", 6)),
            "resolution": inputs.get("resolution", "720p"),
        }
        if operation == "image_to_video":
            image_url = inputs.get("image_url")
            if not image_url and inputs.get("image_path"):
                from tools.video._shared import upload_image_fal
                image_url = upload_image_fal(inputs["image_path"])
            if not image_url:
                return ToolResult(
                    success=False,
                    error="image_to_video requires image_url or image_path",
                )
            payload["image_url"] = image_url
        elif operation == "reference_to_video":
            ref_urls = list(inputs.get("reference_image_urls") or [])
            if inputs.get("reference_image_paths"):
                from tools.video._shared import upload_image_fal
                ref_urls.extend(upload_image_fal(p) for p in inputs["reference_image_paths"])
            if not ref_urls:
                return ToolResult(
                    success=False,
                    error="reference_to_video requires reference_image_urls or reference_image_paths",
                )
            payload["reference_image_urls"] = ref_urls
            if inputs.get("aspect_ratio"):
                payload["aspect_ratio"] = inputs["aspect_ratio"]
        elif inputs.get("aspect_ratio"):
            payload["aspect_ratio"] = inputs["aspect_ratio"]

        from tools.video._shared import submit_and_poll_fal_queue

        output_path = Path(inputs.get("output_path", "grok_fal_output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from tools.video._shared import fal_cache_lookup, fal_cache_store

        if fal_cache_lookup(self, inputs, output_path):
            from tools.video._shared import probe_output

            return ToolResult(
                success=True,
                data={
                    "provider": "grok",
                    "model": model_path,
                    "prompt": inputs["prompt"],
                    "operation": operation,
                    "aspect_ratio": inputs.get("aspect_ratio", "16:9"),
                    "resolution": inputs.get("resolution", "720p"),
                    "output": str(output_path),
                    "output_path": str(output_path),
                    "format": "mp4",
                    "cache_hit": True,
                    **probe_output(output_path),
                },
                artifacts=[str(output_path)],
                cost_usd=0.0,
                model=model_path,
            )

        try:
            data = submit_and_poll_fal_queue(
                f"https://queue.fal.run/{model_path}",
                payload,
                api_key,
                request_log_path=output_path.parent / "fal_requests.jsonl",
            )

            video_url = data["video"]["url"]
            video_response = requests.get(video_url, timeout=180)
            video_response.raise_for_status()

            output_path.write_bytes(video_response.content)
            fal_cache_store(self, inputs, output_path, video_url=video_url)

        except Exception as e:
            return ToolResult(success=False, error=f"Grok video generation failed: {e}")

        from tools.video._shared import probe_output

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "grok",
                "model": model_path,
                "prompt": inputs["prompt"],
                "operation": operation,
                "aspect_ratio": inputs.get("aspect_ratio", "16:9"),
                "resolution": inputs.get("resolution", "720p"),
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=model_path,
        )
