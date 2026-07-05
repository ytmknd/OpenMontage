"""Mixin engines composed into `tools.video.video_compose.VideoCompose`.

Each module implements one composition backend or cross-cutting concern as a
plain mixin (no BaseTool inheritance) — they reference `self.*` attributes
provided by the final `VideoCompose` class. Keeping the split at the mixin
level (rather than moving `VideoCompose` itself) preserves the registry's
`cls.__module__ == module.__name__` auto-discovery and existing
`from tools.video.video_compose import VideoCompose` imports.
"""

from tools.video.compose_engines.atelier import AtelierMixin
from tools.video.compose_engines.ffmpeg_engine import FFmpegEngineMixin
from tools.video.compose_engines.final_review import FinalReviewMixin
from tools.video.compose_engines.hyperframes_bridge import HyperFramesBridgeMixin
from tools.video.compose_engines.remotion_engine import RemotionEngineMixin

__all__ = [
    "AtelierMixin",
    "FFmpegEngineMixin",
    "FinalReviewMixin",
    "HyperFramesBridgeMixin",
    "RemotionEngineMixin",
]
