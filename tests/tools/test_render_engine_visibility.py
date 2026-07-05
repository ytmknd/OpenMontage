"""D-2 governance: `render_result.data["engine_used"]` must always be visible.

`_render` routes to one of several composition engines based on
`edit_decisions.render_runtime` and `_needs_remotion`. The specific hazard
this covers: when `render_runtime == "remotion"` is locked at proposal but
`_needs_remotion(resolved_cuts)` is False (pure video cuts), `_render`
silently routes to the FFmpeg `_compose` path. That routing itself is
intentional and NOT changed here — but the engine that actually ran must be
visible on the result, and the remotion-locked-but-downgraded case must also
carry an explicit `engine_downgrade_reason`.

These tests monkeypatch the underlying engine calls (`_remotion_render`,
`_compose`, `_render_via_ffmpeg`) with canned ToolResults and bypass
`_pre_compose_validation` (irrelevant to this contract) so they exercise only
the routing/tagging logic in `_render`, not real ffmpeg/Remotion execution.
"""

from __future__ import annotations

from tools.base_tool import ToolResult
from tools.video.video_compose import VideoCompose


def _edit_decisions(render_runtime: str) -> dict:
    return {
        "version": "1.0",
        "renderer_family": "explainer-data",
        "render_runtime": render_runtime,
        "cuts": [
            {"id": "c1", "source": "clip_a", "in_seconds": 0, "out_seconds": 2},
        ],
    }


def _asset_manifest() -> dict:
    return {"assets": [{"id": "clip_a", "path": "clip_a.mp4"}]}


def _base_inputs(tmp_path, render_runtime: str) -> dict:
    return {
        "operation": "render",
        "edit_decisions": _edit_decisions(render_runtime),
        "asset_manifest": _asset_manifest(),
        "output_path": str(tmp_path / "out.mp4"),
    }


def test_remotion_path_sets_engine_used_remotion(tmp_path, monkeypatch):
    """render_runtime='remotion' + _needs_remotion True -> engine_used='remotion'."""
    monkeypatch.setattr(
        VideoCompose, "_pre_compose_validation", lambda self, *a, **k: None, raising=True
    )
    monkeypatch.setattr(VideoCompose, "_needs_remotion", lambda self, cuts: True, raising=True)
    monkeypatch.setattr(
        VideoCompose,
        "_remotion_render",
        lambda self, inputs: ToolResult(success=True, data={"operation": "remotion_render"}),
        raising=True,
    )

    result = VideoCompose().execute(_base_inputs(tmp_path, "remotion"))

    assert result.success
    assert result.data["engine_used"] == "remotion"
    assert "engine_downgrade_reason" not in result.data


def test_remotion_locked_but_ffmpeg_downgrade_sets_reason(tmp_path, monkeypatch):
    """render_runtime='remotion' locked, but _needs_remotion False (pure video
    cuts) -> silently routes to FFmpeg. Routing is unchanged; the engine used
    and the downgrade reason MUST both be visible on the result."""
    monkeypatch.setattr(
        VideoCompose, "_pre_compose_validation", lambda self, *a, **k: None, raising=True
    )
    monkeypatch.setattr(VideoCompose, "_needs_remotion", lambda self, cuts: False, raising=True)
    monkeypatch.setattr(
        VideoCompose,
        "_compose",
        lambda self, inputs: ToolResult(success=True, data={"operation": "compose"}),
        raising=True,
    )

    result = VideoCompose().execute(_base_inputs(tmp_path, "remotion"))

    assert result.success
    assert result.data["engine_used"] == "ffmpeg"
    assert "engine_downgrade_reason" in result.data
    reason = result.data["engine_downgrade_reason"]
    assert "render_runtime='remotion'" in reason
    assert "FFmpeg" in reason


def test_explicit_ffmpeg_path_sets_engine_used_ffmpeg(tmp_path, monkeypatch):
    """render_runtime='ffmpeg' (explicit, not a downgrade) -> engine_used='ffmpeg'
    with no downgrade_reason, since nothing was locked-then-swapped."""
    monkeypatch.setattr(
        VideoCompose, "_pre_compose_validation", lambda self, *a, **k: None, raising=True
    )
    monkeypatch.setattr(
        VideoCompose,
        "_render_via_ffmpeg",
        lambda self, **kwargs: ToolResult(success=True, data={"operation": "compose"}),
        raising=True,
    )

    result = VideoCompose().execute(_base_inputs(tmp_path, "ffmpeg"))

    assert result.success
    assert result.data["engine_used"] == "ffmpeg"
    assert "engine_downgrade_reason" not in result.data


def test_hyperframes_path_sets_engine_used_hyperframes(tmp_path, monkeypatch):
    """render_runtime='hyperframes' -> engine_used='hyperframes' on success."""
    monkeypatch.setattr(
        VideoCompose, "_pre_compose_validation", lambda self, *a, **k: None, raising=True
    )
    monkeypatch.setattr(
        VideoCompose,
        "_render_via_hyperframes",
        lambda self, **kwargs: ToolResult(success=True, data={"operation": "render"}),
        raising=True,
    )

    result = VideoCompose().execute(_base_inputs(tmp_path, "hyperframes"))

    assert result.success
    assert result.data["engine_used"] == "hyperframes"


def test_atelier_path_sets_engine_used_remotion_atelier(tmp_path, monkeypatch):
    """composition_mode='atelier' -> engine_used='remotion-atelier' on success."""
    monkeypatch.setattr(
        VideoCompose,
        "_render_via_atelier",
        lambda self, inputs, edit_decisions: ToolResult(
            success=True, data={"operation": "render", "composition_mode": "atelier"}
        ),
        raising=True,
    )

    edit_decisions = _edit_decisions("remotion")
    edit_decisions["composition_mode"] = "atelier"
    result = VideoCompose().execute(
        {
            "operation": "render",
            "edit_decisions": edit_decisions,
            "output_path": str(tmp_path / "out.mp4"),
        }
    )

    assert result.success
    assert result.data["engine_used"] == "remotion-atelier"
