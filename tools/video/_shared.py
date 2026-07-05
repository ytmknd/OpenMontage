"""Shared helpers for provider-specific video generation tools."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.base_tool import ToolResult, ToolStatus


HEYGEN_PROVIDERS = {
    "veo_3_1": {"name": "Google VEO 3.1", "quality": "highest", "speed": "slow"},
    "veo_3_1_fast": {"name": "Google VEO 3.1 Fast", "quality": "high", "speed": "medium"},
    "veo3": {"name": "Google VEO 3", "quality": "high", "speed": "slow"},
    "veo3_fast": {"name": "Google VEO 3 Fast", "quality": "high", "speed": "medium"},
    "veo2": {"name": "Google VEO 2", "quality": "medium", "speed": "medium"},
    "kling_pro": {"name": "Kling Pro", "quality": "high", "speed": "medium"},
    "kling_v2": {"name": "Kling v2", "quality": "medium", "speed": "fast"},
    "sora_v2": {"name": "Sora v2", "quality": "high", "speed": "slow"},
    "sora_v2_pro": {"name": "Sora v2 Pro", "quality": "highest", "speed": "slow"},
    "runway_gen4": {"name": "Runway Gen-4", "quality": "high", "speed": "medium"},
    # NOTE: HeyGen's `seedance_lite` / `seedance_pro` provider strings map to
    # Seedance 1.x. Seedance 2.0 on HeyGen is exposed through Video Agent and
    # Avatar Shots endpoints, NOT via the workflow provider parameter. For 2.0
    # access today, use `seedance_video` (fal.ai) or `seedance_replicate`.
    "seedance_lite": {"name": "Seedance Lite (1.x)", "quality": "medium", "speed": "fast"},
    "seedance_pro": {"name": "Seedance Pro (1.x)", "quality": "high", "speed": "medium"},
    "ltx_distilled": {"name": "LTX Distilled", "quality": "low", "speed": "fastest"},
}

WAN_VARIANTS = {
    "wan2.1-1.3b": {
        "name": "Wan 2.1 (1.3B)",
        "hf_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "hf_i2v_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "pipeline_class": "WanPipeline",
        "vram_mb": 8000,
        "quality": "high",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 832,
        "default_height": 480,
        "default_num_frames": 81,
        "fps": 16,
    },
    "wan2.1-14b": {
        "name": "Wan 2.1 (14B)",
        "hf_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "hf_i2v_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "pipeline_class": "WanPipeline",
        "vram_mb": 24000,
        "quality": "highest",
        "speed": "slow",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 1280,
        "default_height": 720,
        "default_num_frames": 81,
        "fps": 16,
    },
}

HUNYUAN_VARIANTS = {
    "hunyuan-1.5": {
        "name": "HunyuanVideo 1.5",
        "hf_id": "tencent/HunyuanVideo-1.5",
        "pipeline_class": "HunyuanVideoPipeline",
        "vram_mb": 14000,
        "quality": "high",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 848,
        "default_height": 480,
        "default_num_frames": 121,
        "fps": 24,
    },
}

LTX_LOCAL_VARIANTS = {
    "ltx2-local": {
        "name": "LTX-2 (Local)",
        "hf_id": "Lightricks/LTX-2",
        "pipeline_class": "LTXPipeline",
        "vram_mb": 12000,
        "quality": "high",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "LTX-2-Community",
        "default_width": 768,
        "default_height": 512,
        "default_num_frames": 121,
        "fps": 30,
    },
}

COGVIDEO_VARIANTS = {
    "cogvideo-5b": {
        "name": "CogVideoX 1.5 (5B)",
        "hf_id": "THUDM/CogVideoX-5b",
        "pipeline_class": "CogVideoXPipeline",
        "vram_mb": 12000,
        "quality": "medium",
        "speed": "medium",
        "t2v": True,
        "i2v": True,
        "license": "Apache-2.0",
        "default_width": 720,
        "default_height": 480,
        "default_num_frames": 49,
        "fps": 8,
    },
    "cogvideo-2b": {
        "name": "CogVideoX (2B)",
        "hf_id": "THUDM/CogVideoX-2b",
        "pipeline_class": "CogVideoXPipeline",
        "vram_mb": 6000,
        "quality": "medium",
        "speed": "fast",
        "t2v": True,
        "i2v": False,
        "license": "Apache-2.0",
        "default_width": 720,
        "default_height": 480,
        "default_num_frames": 49,
        "fps": 8,
    },
}

LTX2_FRAME_COUNTS = {
    "1s": 25,
    "2s": 49,
    "3s": 73,
    "4s": 97,
    "5s": 121,
    "6.7s": 161,
    "8s": 193,
}


def get_torch_device() -> str:
    """Return best available torch device: cuda > mps (Apple Silicon Metal) > cpu.

    Priority order:
      1. cuda  — NVIDIA GPU (fastest for most diffusion models)
      2. mps   — Apple Silicon Metal (M1/M2/M3/M4/M5, macOS >= 12.3)
      3. cpu   — fallback, always available but slow

    MPS detection is guarded for torch builds that lack ``torch.backends.mps``
    (e.g. older pip wheels or Linux builds).  We check both build-time support
    (``is_built()``) and runtime availability (``is_available()``).
    """
    try:
        import torch as _torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if _torch.cuda.is_available():
        return "cuda"
    # Guard: torch.backends.mps may not exist on older/non-macOS builds
    try:
        mps_backend = getattr(_torch, "backends", None)
        mps_backend = getattr(mps_backend, "mps", None) if mps_backend else None
        if mps_backend is not None:
            # Check build-time support first, then runtime availability
            is_built = getattr(mps_backend, "is_built", lambda: True)()
            is_available = getattr(mps_backend, "is_available", lambda: False)()
            if is_built and is_available:
                return "mps"
    except Exception:
        pass
    return "cpu"


def local_generation_enabled() -> bool:
    return os.environ.get("VIDEO_GEN_LOCAL_ENABLED", "").lower() in {"true", "1", "yes"}


def local_generation_status() -> ToolStatus:
    if not local_generation_enabled():
        return ToolStatus.UNAVAILABLE
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return ToolStatus.UNAVAILABLE
    return ToolStatus.AVAILABLE


def local_install_instructions() -> str:
    return (
        "Enable local video generation and install the diffusers stack:\n"
        "  export VIDEO_GEN_LOCAL_ENABLED=true\n"
        "  uv pip install diffusers transformers accelerate torch pillow requests\n"
        "\n"
        "GPU support — pick what matches your hardware:\n"
        "  NVIDIA CUDA    — works out of the box with the above\n"
        "  Apple Silicon (MPS, macOS >= 12.3) — works out of the box; no extra build\n"
        "  CPU fallback   — slow but functional on any machine\n"
        "\n"
        "VRAM profile: see the selected tool's resource_profile for minimum VRAM."
    )


def estimate_quality_cost(quality: str) -> float:
    if quality == "highest":
        return 0.50
    if quality == "high":
        return 0.35
    if quality == "low":
        return 0.15
    return 0.20


def estimate_speed_runtime(speed: str) -> float:
    return {"fastest": 30.0, "fast": 60.0, "medium": 120.0, "slow": 300.0}.get(speed, 120.0)


def estimate_local_runtime(speed: str) -> float:
    return {"fast": 120.0, "medium": 240.0, "slow": 600.0}.get(speed, 240.0)


def load_diffusers_pipeline(pipeline_class: str, model_id: str, enable_offload: bool):
    import diffusers
    import torch

    pipeline_map = {
        "WanPipeline": "WanPipeline",
        "HunyuanVideoPipeline": "HunyuanVideoPipeline",
        "LTXPipeline": "LTXPipeline",
        "CogVideoXPipeline": "CogVideoXPipeline",
    }
    pipeline_name = pipeline_map.get(pipeline_class, pipeline_class)
    pipeline_class_obj = getattr(diffusers, pipeline_name)

    device = get_torch_device()
    # bfloat16 is only reliable on CUDA; MPS uses float16 for inference,
    # CPU must use float32 (float16 is emulated and unreliable on CPU)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16

    pipeline = pipeline_class_obj.from_pretrained(model_id, torch_dtype=dtype)

    if enable_offload:
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            # enable_model_cpu_offload() is CUDA-only; fall back to direct device placement
            pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to(device)

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
    return pipeline


def load_reference_image(inputs: dict[str, Any], width: int, height: int):
    from io import BytesIO

    import requests
    from PIL import Image

    ref_path = inputs.get("reference_image_path")
    ref_url = inputs.get("reference_image_url")

    if ref_path:
        image = Image.open(ref_path).convert("RGB")
    elif ref_url:
        response = requests.get(ref_url, timeout=60)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return ToolResult(
            success=False,
            error="image_to_video requires reference_image_url or reference_image_path",
        )

    return image.resize((width, height), Image.LANCZOS)


def generate_local_video(
    *,
    tool_name: str,
    variants: dict[str, dict[str, Any]],
    default_variant: str,
    inputs: dict[str, Any],
) -> ToolResult:
    import torch
    from diffusers.utils import export_to_video

    variant = inputs.get("model_variant", default_variant)
    if variant not in variants:
        return ToolResult(
            success=False,
            error=f"Unknown model_variant: {variant}. Available: {', '.join(sorted(variants))}",
        )

    meta = variants[variant]
    prompt = inputs["prompt"]
    operation = inputs.get("operation", "text_to_video")
    seed = inputs.get("seed")
    enable_offload = inputs.get("enable_model_offload", True)

    if operation == "image_to_video" and not meta.get("i2v"):
        return ToolResult(
            success=False,
            error=f"{meta['name']} does not support image_to_video.",
        )

    width = inputs.get("width", meta["default_width"])
    height = inputs.get("height", meta["default_height"])
    num_frames = inputs.get("num_frames", meta["default_num_frames"])
    fps = meta["fps"]
    model_id = meta.get("hf_i2v_id") if operation == "image_to_video" and meta.get("hf_i2v_id") else meta["hf_id"]
    pipeline = load_diffusers_pipeline(meta["pipeline_class"], model_id, enable_offload)

    generation_args: dict[str, Any] = {
        "prompt": prompt,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "num_inference_steps": inputs.get("num_inference_steps", 30),
    }
    if seed is not None:
        generation_args["generator"] = torch.Generator(device="cpu").manual_seed(seed)
    if operation == "image_to_video":
        image = load_reference_image(inputs, width, height)
        if isinstance(image, ToolResult):
            return image
        generation_args["image"] = image
    if meta["pipeline_class"] == "CogVideoXPipeline":
        generation_args["negative_prompt"] = "worst quality, low quality, blurry, distorted, watermark"

    output = pipeline(**generation_args)
    frames = output.frames[0] if hasattr(output, "frames") else output.images

    output_path = Path(inputs.get("output_path", f"{tool_name}_{variant}.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=fps)

    return ToolResult(
        success=True,
        data={
            "provider": tool_name,
            "model_variant": variant,
            "provider_name": meta["name"],
            "mode": "local",
            "prompt": prompt,
            "model_id": model_id,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": round(num_frames / fps, 2),
            "operation": operation,
            "output": str(output_path),
            "format": "mp4",
            "license": meta["license"],
            **probe_output(output_path),
        },
        artifacts=[str(output_path)],
        seed=seed,
        model=model_id,
    )


def poll_heygen(execution_id: str, api_key: str, timeout: int = 600) -> str:
    import requests

    headers = {"X-Api-Key": api_key}
    url = f"https://api.heygen.com/v1/workflows/executions/{execution_id}"
    deadline = time.time() + timeout
    interval = 5.0

    while time.time() < deadline:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", {})
        status = data.get("status", "")

        if status == "completed":
            video_url = (
                data.get("output", {}).get("video", {}).get("video_url")
                or data.get("output", {}).get("video_url")
            )
            if video_url:
                return video_url
            raise RuntimeError(f"Completed but no video_url in output: {data}")

        if status in {"failed", "error"}:
            raise RuntimeError(f"HeyGen generation failed: {data.get('error', 'Unknown')}")

        time.sleep(min(interval, max(0.0, deadline - time.time())))
        interval = min(interval * 1.2, 30.0)

    raise TimeoutError(f"HeyGen execution {execution_id} timed out after {timeout}s")


def upload_image_fal(image_path: str) -> str:
    """Upload a local image to fal.ai storage and return a public URL."""
    import requests

    api_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY")
    if not api_key:
        raise RuntimeError("FAL_KEY or FAL_AI_API_KEY required for image upload")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    content_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        suffix.lstrip("."), "image/png"
    )

    # Initiate upload
    init_resp = requests.post(
        "https://rest.alpha.fal.ai/storage/upload/initiate",
        headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
        json={"content_type": content_type, "file_name": path.name},
        timeout=30,
    )
    init_resp.raise_for_status()
    data = init_resp.json()

    # Upload file content
    put_resp = requests.put(
        data["upload_url"],
        headers={"Content-Type": content_type},
        data=path.read_bytes(),
        timeout=60,
    )
    put_resp.raise_for_status()

    return data["file_url"]


def upload_image_heygen(image_path: str, api_key: str) -> str:
    """Upload a local image to HeyGen and return a public URL.

    Tries the v2 presigned-upload endpoint first, falls back to fal.ai storage.
    """
    import requests

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Try HeyGen v2 presigned upload
    try:
        resp = requests.post(
            "https://api.heygen.com/v2/assets/upload",
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            json={"content_type": "image/png", "file_name": path.name},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            upload_url = data.get("upload_url")
            file_url = data.get("url") or data.get("file_url")
            if upload_url and file_url:
                put_resp = requests.put(
                    upload_url,
                    headers={"Content-Type": "image/png"},
                    data=path.read_bytes(),
                    timeout=60,
                )
                put_resp.raise_for_status()
                return file_url
    except Exception:
        pass

    # Fallback to fal.ai storage upload
    return upload_image_fal(image_path)


def generate_heygen_video(inputs: dict[str, Any]) -> ToolResult:
    import requests

    api_key = os.environ.get("HEYGEN_API_KEY")
    if not api_key:
        return ToolResult(success=False, error="HEYGEN_API_KEY not set.")

    provider = inputs.get("provider_variant", "veo_3_1")
    if provider not in HEYGEN_PROVIDERS:
        return ToolResult(
            success=False,
            error=f"Unknown provider_variant: {provider}. Available: {', '.join(sorted(HEYGEN_PROVIDERS))}",
        )

    prompt = inputs["prompt"]
    aspect_ratio = inputs.get("aspect_ratio", "16:9")
    operation = inputs.get("operation", "text_to_video")
    workflow_input: dict[str, Any] = {
        "prompt": prompt,
        "provider": provider,
        "aspect_ratio": aspect_ratio,
    }
    if operation == "image_to_video":
        ref_url = inputs.get("reference_image_url")
        ref_path = inputs.get("reference_image_path")
        if ref_path and not ref_url:
            ref_url = upload_image_heygen(ref_path, api_key)
        if not ref_url:
            return ToolResult(
                success=False,
                error="image_to_video requires reference_image_url or reference_image_path",
            )
        workflow_input["reference_image_url"] = ref_url

    response = requests.post(
        "https://api.heygen.com/v1/workflows/executions",
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        json={"workflow_type": "GenerateVideoNode", "input": workflow_input},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    execution_id = payload.get("data", {}).get("execution_id")
    if not execution_id:
        return ToolResult(success=False, error=f"No execution_id in response: {payload}")

    video_url = poll_heygen(execution_id, api_key, timeout=600)
    output_path = Path(inputs.get("output_path", f"heygen_video_{execution_id}.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download = requests.get(video_url, timeout=120)
    download.raise_for_status()
    output_path.write_bytes(download.content)

    meta = HEYGEN_PROVIDERS[provider]
    return ToolResult(
        success=True,
        data={
            "provider": "heygen",
            "provider_variant": provider,
            "provider_name": meta["name"],
            "mode": "api",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "operation": operation,
            "execution_id": execution_id,
            "output": str(output_path),
            "format": "mp4",
        },
        artifacts=[str(output_path)],
        model=provider,
    )


def generate_ltx_modal_video(inputs: dict[str, Any]) -> ToolResult:
    import base64

    import requests

    endpoint_url = os.environ.get("MODAL_LTX2_ENDPOINT_URL")
    if not endpoint_url:
        return ToolResult(success=False, error="MODAL_LTX2_ENDPOINT_URL not set.")

    prompt = inputs["prompt"]
    operation = inputs.get("operation", "text_to_video")
    aspect = inputs.get("aspect_ratio", "16:9")
    width = inputs.get("width")
    height = inputs.get("height")
    if width is None or height is None:
        if aspect == "16:9":
            width, height = 1024, 576
        elif aspect == "9:16":
            width, height = 576, 1024
        else:
            width, height = 512, 512

    num_frames = inputs.get("num_frames", LTX2_FRAME_COUNTS.get(inputs.get("duration_hint", "5s"), 121))
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames - 1) // 8) * 8 + 1

    payload: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": 24,
        "steps": inputs.get("num_inference_steps", 30),
        "negative_prompt": "worst quality, low quality, blurry, distorted, watermark, text, logo",
    }
    if inputs.get("seed") is not None:
        payload["seed"] = inputs["seed"]

    if operation == "image_to_video":
        ref_path = inputs.get("reference_image_path")
        ref_url = inputs.get("reference_image_url")
        if ref_path:
            payload["input_image"] = base64.b64encode(Path(ref_path).read_bytes()).decode()
        elif ref_url:
            payload["input_image_url"] = ref_url
        else:
            return ToolResult(
                success=False,
                error="image_to_video requires reference_image_url or reference_image_path",
            )

    response = requests.post(endpoint_url, json=payload, timeout=300)
    response.raise_for_status()
    output_path = Path(inputs.get("output_path", "ltx_video_modal.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_type = response.headers.get("content-type", "")
    if "video" in content_type or "octet-stream" in content_type:
        output_path.write_bytes(response.content)
    else:
        response_payload = response.json()
        video_url = response_payload.get("video_url") or response_payload.get("url")
        if not video_url:
            return ToolResult(success=False, error=f"No video data in response: {response_payload}")
        download = requests.get(video_url, timeout=120)
        download.raise_for_status()
        output_path.write_bytes(download.content)

    return ToolResult(
        success=True,
        data={
            "provider": "ltx-modal",
            "provider_name": "LTX-2 (Modal)",
            "mode": "modal",
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": 24,
            "duration_seconds": round(num_frames / 24, 2),
            "operation": operation,
            "output": str(output_path),
            "format": "mp4",
        },
        artifacts=[str(output_path)],
        seed=inputs.get("seed"),
        model="ltx-2",
    )


class FalJobError(RuntimeError):
    """A fal.ai queue job failed or was rejected. Message carries HTTP body / job logs."""


def _http_error_detail(exc) -> str:
    """Extract 'HTTP <code>: <body[:500]>' from a requests.HTTPError, best-effort."""
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    status_code = getattr(response, "status_code", "?")
    try:
        body = response.text
    except Exception:
        body = ""
    body = (body or "")[:500]
    return f"HTTP {status_code}: {body}"


def _append_request_log(path: "str | Path", record: dict[str, Any]) -> None:
    """Append one JSON line to the fal request log. Best-effort — never raises."""
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def submit_and_poll_fal_queue(
    submit_url: str,
    payload: dict[str, Any],
    api_key: str,
    *,
    timeout: float = 900.0,
    poll_interval: float = 5.0,
    max_interval: float = 30.0,
    request_log_path: "str | Path | None" = None,
) -> dict[str, Any]:
    """Submit a job to a fal.ai queue endpoint and poll it to completion.

    Uses a hard deadline (``timeout`` seconds) combined with exponential
    backoff (interval grows by 1.5x each poll, capped at ``max_interval``,
    and each sleep is further capped by the remaining time to the deadline).

    FAILED/CANCELLED reasons are pulled from the ``?logs=1`` status endpoint
    so the raised error carries the actual failure detail (e.g. moderation
    rejection) instead of a bare status string.

    If the deadline expires, the raised ``TimeoutError`` deliberately
    includes ``status_url`` — the job may still complete on fal's side (and
    has already been billed), so the caller/agent can poll that URL directly
    to recover the result without re-submitting and paying twice.

    If ``request_log_path`` is given, every meaningful transition (submitted,
    completed, failed/cancelled, timeout) is appended as a JSON line to that
    path. This log is the recovery ledger for billed-but-uncollected jobs —
    a submit is billed immediately, but if this process dies mid-poll the
    ``request_id`` would otherwise be lost. The ``fal_queue`` tool's
    ``collect``/``list`` operations consume this log to recover or audit
    those jobs after the fact.
    """
    import requests

    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    try:
        submit_resp = requests.post(submit_url, headers=headers, json=payload, timeout=30)
        submit_resp.raise_for_status()
    except requests.HTTPError as e:
        raise FalJobError(f"fal queue submit rejected: {_http_error_detail(e)}") from e

    queue_data = submit_resp.json()
    try:
        request_id = queue_data["request_id"]
        status_url = queue_data["status_url"]
        response_url = queue_data["response_url"]
    except KeyError as e:
        raise FalJobError(
            f"fal queue submit response missing {e}: {repr(queue_data)[:300]}"
        ) from e

    if request_log_path is not None:
        _append_request_log(
            request_log_path,
            {
                "request_id": request_id,
                "status_url": status_url,
                "response_url": response_url,
                "submit_url": submit_url,
                "prompt": str(payload.get("prompt", ""))[:200],
                "event": "submitted",
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )

    deadline = time.time() + timeout
    interval = poll_interval
    consecutive_failures = 0
    last_status = "UNKNOWN"

    while True:
        now = time.time()
        if now >= deadline:
            if request_log_path is not None:
                _append_request_log(
                    request_log_path,
                    {
                        "request_id": request_id,
                        "event": "timeout",
                        "status": last_status,
                        "at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            raise TimeoutError(
                f"fal job {request_id} still {last_status} after {int(timeout)}s. "
                "The job may still complete and has already been billed — poll "
                f"{status_url} to recover the result without re-submitting."
            )

        time.sleep(min(interval, max(0.0, deadline - now)))
        interval = min(interval * 1.5, max_interval)

        try:
            status_resp = requests.get(f"{status_url}?logs=1", headers=headers, timeout=15)
            status_resp.raise_for_status()
            status_json = status_resp.json()
        except requests.RequestException as e:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                raise FalJobError(
                    f"fal job {request_id} polling failed 5 times in a row: {e}"
                ) from e
            continue

        consecutive_failures = 0
        last_status = status_json.get("status", "UNKNOWN")

        if last_status == "COMPLETED":
            break

        if last_status in ("FAILED", "CANCELLED"):
            raw_logs = status_json.get("logs") or []
            messages = []
            for entry in raw_logs:
                if isinstance(entry, dict):
                    message = entry.get("message")
                    if message:
                        messages.append(str(message))
                elif entry:
                    messages.append(str(entry))
            log_tail = "; ".join(messages[-5:]) if messages else "no logs available"
            if request_log_path is not None:
                _append_request_log(
                    request_log_path,
                    {
                        "request_id": request_id,
                        "event": last_status.lower(),
                        "at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            raise FalJobError(
                f"fal job {request_id} {last_status.lower()}: {log_tail}"
            )

    try:
        result_resp = requests.get(response_url, headers=headers, timeout=30)
        result_resp.raise_for_status()
    except requests.HTTPError as e:
        raise FalJobError(f"fal queue result fetch failed: {_http_error_detail(e)}") from e

    if request_log_path is not None:
        _append_request_log(
            request_log_path,
            {
                "request_id": request_id,
                "event": "completed",
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )

    return result_resp.json()


def clip_cache_enabled() -> bool:
    """Clip caching for generation tools. Disable with OPENMONTAGE_CLIP_CACHE=0/false/off."""
    return os.environ.get("OPENMONTAGE_CLIP_CACHE", "1").lower() not in {"0", "false", "off"}


def fal_cache_lookup(tool, inputs: dict[str, Any], output_path: Path) -> bool:
    """Return True and materialize the cached clip at output_path on a hit.

    ``force_regenerate`` bypasses the lookup only — a fresh generation still
    gets stored via ``fal_cache_store`` afterward. Cache trouble (missing
    dependency, filesystem drift, lock timeout) must never block generation,
    so any exception here is swallowed and treated as a miss.
    """
    if not clip_cache_enabled() or inputs.get("force_regenerate"):
        return False

    clip_id = f"{tool.name}_{tool.idempotency_key(inputs)}"
    try:
        from tools.video.clip_cache import get_default_cache

        return get_default_cache().try_link(clip_id, output_path)
    except Exception:
        return False


def fal_cache_store(tool, inputs: dict[str, Any], output_path: Path, video_url: str = "") -> None:
    """Best-effort ingest of a freshly generated clip. Never raises."""
    if not clip_cache_enabled():
        return

    clip_id = f"{tool.name}_{tool.idempotency_key(inputs)}"
    try:
        from tools.video.clip_cache import get_default_cache

        get_default_cache().ingest(
            clip_id,
            output_path,
            metadata={
                "source": tool.name,
                "source_id": tool.idempotency_key(inputs),
                "source_url": video_url,
            },
        )
    except Exception:
        pass


def probe_output(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"file_size_bytes": path.stat().st_size}
    if not shutil.which("ffprobe"):
        return info

    import json

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            probe = json.loads(proc.stdout)
            fmt = probe.get("format", {})
            info["duration_seconds"] = float(fmt.get("duration", 0))
            info["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["video_width"] = int(stream.get("width", 0))
                    info["video_height"] = int(stream.get("height", 0))
                    info["video_codec"] = stream.get("codec_name", "")
                    break
    except Exception:
        pass
    return info
