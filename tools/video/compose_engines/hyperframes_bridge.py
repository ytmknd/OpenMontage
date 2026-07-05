"""HyperFrames composition engine bridge mixin for VideoCompose.

Delegates to `tools.video.hyperframes_compose.HyperFramesCompose` for
HTML/CSS/GSAP composition (kinetic typography, product promos,
website-to-video, registry-block-driven scenes). Governance forbids silently
routing to Remotion or FFmpeg if HyperFrames is unavailable or fails — this
mixin surfaces a structured blocker instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tools.base_tool import ToolResult


class HyperFramesBridgeMixin:
    """HyperFrames availability check and render delegation.

    Plain mixin — no BaseTool inheritance. References `self.*` attributes
    (e.g. `self._read_text_file`, `self._run_final_review`) provided by the
    final composed VideoCompose class.
    """

    def _hyperframes_available(self) -> bool:
        """Check if HyperFrames rendering is available.

        Delegates to the dedicated tool so the availability check stays in
        one place (node 22 floor, ffmpeg + npx on PATH).
        """
        try:
            from tools.video.hyperframes_compose import HyperFramesCompose
            return bool(HyperFramesCompose()._runtime_check()["runtime_available"])
        except Exception:
            return False

    def _render_via_hyperframes(
        self,
        *,
        inputs: dict[str, Any],
        edit_decisions: dict[str, Any],
        asset_manifest: dict[str, Any],
        resolved_cuts: list[dict],
        output_path: Path,
        profile: Optional[str],
    ) -> ToolResult:
        """Delegate to hyperframes_compose and run the mandatory final self-review.

        Governance: if HyperFrames is unavailable or fails, return a structured
        blocker — do NOT silently route to Remotion or FFmpeg. The agent must
        surface the blocker and get user approval before any runtime swap.
        """
        if not self._hyperframes_available():
            return ToolResult(
                success=False,
                error=(
                    "render_runtime='hyperframes' was locked at proposal, but "
                    "the HyperFrames runtime is not available on this machine. "
                    "Per governance this is a BLOCKER — surface it to the user "
                    "per AGENT_GUIDE.md > 'Escalate Blockers Explicitly' and wait "
                    "for approval before switching runtime. Requirements: "
                    "Node.js >= 22, FFmpeg, and npx on PATH. See "
                    "tools/video/hyperframes_compose.py for the specific missing piece."
                ),
            )

        try:
            from tools.video.hyperframes_compose import HyperFramesCompose
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Could not import hyperframes_compose: {e}",
            )

        workspace_path = (
            inputs.get("workspace_path")
            or str(output_path.parent.parent / "hyperframes")
        )

        # Pass the playbook through so the style bridge can emit CSS vars.
        playbook_data = inputs.get("playbook")
        if not playbook_data:
            playbook_name = (
                inputs.get("playbook_name")
                or (edit_decisions.get("metadata") or {}).get("playbook")
            )
            if playbook_name:
                try:
                    from styles.playbook_loader import load_playbook  # type: ignore
                    playbook_data = load_playbook(playbook_name)
                except Exception:
                    playbook_data = None

        hf_inputs: dict[str, Any] = {
            "operation": "render",
            "workspace_path": workspace_path,
            "output_path": str(output_path),
            "edit_decisions": dict(edit_decisions, cuts=resolved_cuts),
            "asset_manifest": asset_manifest,
        }
        if playbook_data:
            hf_inputs["playbook"] = playbook_data
        if profile:
            hf_inputs["profile"] = profile
        if "quality" in inputs:
            hf_inputs["quality"] = inputs["quality"]
        if "fps" in inputs:
            hf_inputs["fps"] = inputs["fps"]
        if "strict" in inputs:
            hf_inputs["strict"] = inputs["strict"]
        if "skip_contrast" in inputs:
            hf_inputs["skip_contrast"] = inputs["skip_contrast"]

        render_result = HyperFramesCompose().execute(hf_inputs)

        if not render_result.success:
            return ToolResult(
                success=False,
                error=(
                    f"HyperFrames render failed: {render_result.error}. "
                    "Per governance: do NOT silently fall back to Remotion or "
                    "FFmpeg. Surface the failure to the user along with the "
                    "hyperframes_compose step log before proposing a swap."
                ),
                data=render_result.data,
            )

        # Post-render: mandatory final self-review (identical contract to the Remotion path).
        if output_path.exists():
            final_review = self._run_final_review(
                output_path,
                edit_decisions,
                inputs.get("proposal_packet"),
                narration_transcript_path=inputs.get("narration_transcript_path"),
                script_text=inputs.get("script_text") or self._read_text_file(
                    inputs.get("script_path")
                ),
            )
            if render_result.data is None:
                render_result.data = {}
            render_result.data["final_review"] = final_review
            render_result.data["final_review_status"] = final_review["status"]
            if final_review["status"] == "fail":
                return ToolResult(
                    success=False,
                    error=(
                        "Post-render self-review FAILED (HyperFrames). The output is not presentable.\n"
                        + "\n".join(f"  • {i}" for i in final_review.get("issues_found", []))
                    ),
                    data=render_result.data,
                )

        return render_result
