"""Video composition tool — FFmpeg + Remotion + HyperFrames (runtime-aware).

Pipeline-facing orchestration surface for composition. Takes `edit_decisions`,
`asset_manifest`, and audio, and delegates to the technical runtime chosen
at proposal stage.

Routing is driven by `edit_decisions.render_runtime` (locked at proposal):

- `remotion`   → React-based frame-accurate render via `npx remotion render`.
                 Handles the existing scene-component stack, word-level captions,
                 TalkingHead/CinematicRenderer. Current default.
- `hyperframes` → HTML/CSS/GSAP render via `hyperframes_compose`.
                 Handles kinetic typography, product promos, website-to-video,
                 registry blocks. Added in the parallel-runtime initiative.
- `ffmpeg`     → FFmpeg concat/trim. Used only for simple video cuts without
                 composition, or when the approved path explicitly names FFmpeg.

Authoring mode is orthogonal to runtime. Setting
`edit_decisions.composition_mode = "atelier"` (or `renderer_family="bespoke"`)
routes to a hand-authored, project-local Remotion composition that BYPASSES the
cut-schema and the stock scene-type registry entirely — the "hand-stitched
every time" path for hero/bespoke pieces. See `_render_via_atelier`.

Silent runtime swaps are forbidden by governance. If the chosen runtime is
unavailable or fails, this tool surfaces a structured blocker and waits for
the agent to re-ask the user rather than substituting a different engine.

Implementation note: the render-engine bodies live in
`tools/video/compose_engines/` as plain mixins (FFmpegEngineMixin,
RemotionEngineMixin, AtelierMixin, HyperFramesBridgeMixin, FinalReviewMixin).
`VideoCompose` itself MUST stay defined in this module — the tool registry
auto-discovers tools by walking modules and registering classes where
`cls.__module__ == module.__name__`, and existing code imports
`from tools.video.video_compose import VideoCompose` directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ResumeSupport,
    ToolResult,
    ToolStability,
    ToolTier,
)
from tools.video.compose_engines import (
    AtelierMixin,
    FFmpegEngineMixin,
    FinalReviewMixin,
    HyperFramesBridgeMixin,
    RemotionEngineMixin,
)


class VideoCompose(FFmpegEngineMixin, RemotionEngineMixin, AtelierMixin, HyperFramesBridgeMixin, FinalReviewMixin, BaseTool):
    name = "video_compose"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = "Install FFmpeg: https://ffmpeg.org/download.html"
    agent_skills = ["remotion-best-practices", "remotion", "ffmpeg"]

    capabilities = [
        "compose_cuts",
        "burn_subtitles",
        "overlay_assets",
        "encode_profile",
        "remotion_render",
    ]

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["compose", "render", "remotion_render", "burn_subtitles", "overlay", "encode"],
                "description": (
                    "compose: low-level concat cuts + audio + subtitles. "
                    "render: high-level — resolves asset IDs, auto-routes to Remotion "
                    "for images/animations or FFmpeg for video-only. Preferred for compose-director. "
                    "remotion_render: render via Remotion (Node.js). "
                    "burn_subtitles: burn subtitle file into existing video. "
                    "overlay: composite overlays onto base video. "
                    "encode: re-encode to a target profile/codec."
                ),
            },
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "edit_decisions": {
                "type": "object",
                "description": "Full edit_decisions artifact (required for compose/render)",
            },
            "asset_manifest": {
                "type": "object",
                "description": (
                    "Full asset_manifest artifact (required for render). "
                    "Used to resolve asset IDs in cuts[].source to file paths."
                ),
            },
            "proposal_packet": {
                "type": "object",
                "description": (
                    "Full proposal_packet artifact. Optional but STRONGLY "
                    "recommended — when present, final_review compares "
                    "proposal_packet.production_plan.render_runtime against "
                    "edit_decisions.render_runtime and flags runtime_swap_detected. "
                    "Without it, runtime-swap detection falls back to checking "
                    "edit_decisions.metadata.proposal_render_runtime."
                ),
            },
            "narration_transcript_path": {
                "type": "string",
                "description": (
                    "Path to a word-level transcript JSON (from `transcriber` "
                    "tool output). Optional but STRONGLY recommended: when "
                    "combined with script_path/script_text, final_review "
                    "runs transcript_comparison and catches TTS failures "
                    "like 'Chirp3-HD reads ... as the word dot'. Without "
                    "it, content-level audio bugs ship silently."
                ),
            },
            "script_path": {
                "type": "string",
                "description": (
                    "Path to the source narration script (plain text). "
                    "Used by transcript_comparison to diff against the "
                    "transcribed audio. Provide this OR script_text."
                ),
            },
            "script_text": {
                "type": "string",
                "description": (
                    "Inline source narration script. Used by "
                    "transcript_comparison when a file path is unavailable."
                ),
            },
            "subtitle_path": {"type": "string"},
            "subtitle_style": {
                "type": "object",
                "description": "ASS subtitle styling. Also extracted from edit_decisions.subtitles if not provided.",
                "properties": {
                    "font": {"type": "string", "default": "Arial"},
                    "font_size": {"type": "integer", "default": 24},
                    "primary_color": {"type": "string", "default": "&HFFFFFF"},
                    "outline_color": {"type": "string", "default": "&H000000"},
                    "outline_width": {"type": "number", "default": 2},
                    "margin_v": {"type": "integer", "default": 40},
                    "alignment": {"type": "integer", "default": 2},
                },
            },
            "overlays": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "asset_path": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "start_seconds": {"type": "number"},
                        "end_seconds": {"type": "number"},
                        "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "audio_path": {"type": "string", "description": "Mixed audio to mux into output"},
            "profile": {
                "type": "string",
                "description": (
                    "Media profile name from media_profiles.py "
                    "(e.g. youtube_landscape, tiktok, instagram_reels). "
                    "Applied in render and encode operations."
                ),
            },
            "options": {
                "type": "object",
                "description": "Render options (used by the render operation)",
                "properties": {
                    "subtitle_burn": {"type": "boolean", "default": True},
                    "two_pass_encode": {"type": "boolean", "default": False},
                },
            },
            "codec": {"type": "string", "default": "libx264"},
            "crf": {"type": "integer", "default": 23},
            "preset": {"type": "string", "default": "medium"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=4, ram_mb=2048, vram_mb=0, disk_mb=5000, network_required=False
    )

    best_for = [
        "Final render for explainer and animation pipelines",
        "Image-to-video with spring animations (Remotion)",
        "Animated text cards, stat cards, charts (Remotion)",
        "Complex transitions between scenes (Remotion)",
        "Pure video concat and trim (FFmpeg)",
    ]
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["Conversion failed"])
    resume_support = ResumeSupport.FROM_START
    idempotency_key_fields = ["operation", "input_path", "edit_decisions"]
    side_effects = ["writes video file to output_path"]
    user_visible_verification = [
        "Play the composed output and verify cuts, subtitles, and overlays",
    ]

    def get_info(self) -> dict[str, Any]:
        """Extend base get_info to surface all available render runtimes.

        Preflight reports each runtime's availability separately so the agent
        can choose an appropriate `render_runtime` at proposal stage. Silent
        fallback between runtimes is forbidden.
        """
        info = super().get_info()
        remotion_ok = self._remotion_available()
        hyperframes_ok = self._hyperframes_available()
        info["render_engines"] = {
            "ffmpeg": True,
            "remotion": remotion_ok,
            "hyperframes": hyperframes_ok,
        }
        # Backwards-compat alias — some proposal skills inspect this name.
        info["render_runtimes"] = info["render_engines"]

        if remotion_ok:
            info["remotion_components"] = self._REMOTION_COMPONENTS
            info["remotion_note"] = (
                "Remotion is available for React-based rendering. Use it for "
                "image-to-video with spring animations, animated text/stat cards, "
                "charts, callouts, comparisons, and word-level caption burn. "
                "Prefer Remotion over Ken Burns pan-and-zoom for explainer "
                "and motion-graphics pipelines that already use the scene-component stack."
            )
        else:
            composer_dir = Path(__file__).resolve().parent.parent.parent / "remotion-composer"
            if composer_dir.exists() and (composer_dir / "package.json").exists() and not (composer_dir / "node_modules").exists():
                info["remotion_note"] = (
                    "Remotion project exists but node_modules are NOT installed. "
                    "Run 'cd remotion-composer && npm install' to enable Remotion rendering."
                )
            else:
                info["remotion_note"] = (
                    "Remotion is NOT available (needs Node.js/npx + remotion-composer + node_modules)."
                )

        if hyperframes_ok:
            info["hyperframes_note"] = (
                "HyperFrames is available for HTML/CSS/GSAP composition. Use it "
                "for kinetic typography, product promos, launch reels, "
                "website-to-video, and registry-block-driven scenes. Consumed via "
                "'npx hyperframes' (npm package: 'hyperframes'). "
                "Before locking render_runtime='hyperframes' at the proposal stage, "
                "verify the runtime with `hyperframes_compose` operation='doctor' "
                "or `make hyperframes-doctor`. An 'available' flag from the runtime "
                "check means node + ffmpeg + the npm package all resolve; it does "
                "not guarantee a render will succeed on the first specific "
                "composition."
            )
        else:
            info["hyperframes_note"] = (
                "HyperFrames is NOT available. Requires Node.js >= 22, FFmpeg, "
                "npx on PATH, and the 'hyperframes' npm package to be resolvable. "
                "Run `make hyperframes-doctor` to see the specific missing piece, "
                "or call `hyperframes_compose` operation='doctor' directly."
            )

        # Governance note — agents and reviewers consume this.
        info["runtime_governance"] = (
            "render_runtime is locked at proposal stage and carried unchanged "
            "through edit_decisions. Silent swaps are forbidden. If the "
            "chosen runtime fails, surface a structured blocker and wait for "
            "user approval before switching."
        )
        return info

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs["operation"]
        start = time.time()

        try:
            if operation == "compose":
                result = self._compose(inputs)
            elif operation == "render":
                result = self._render(inputs)
            elif operation == "remotion_render":
                result = self._remotion_render(inputs)
            elif operation == "burn_subtitles":
                result = self._burn_subtitles(inputs)
            elif operation == "overlay":
                result = self._overlay(inputs)
            elif operation == "encode":
                result = self._encode(inputs)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        result.duration_seconds = round(time.time() - start, 2)
        return result

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    @staticmethod
    def _is_image(path: Path) -> bool:
        """Check if a file is a still image (routes to Remotion, not FFmpeg)."""
        return path.suffix.lower() in VideoCompose._IMAGE_EXTENSIONS

    def _pre_compose_validation(
        self,
        edit_decisions: dict[str, Any],
        resolved_cuts: list[dict],
        scene_plan: list[dict] | None = None,
    ) -> ToolResult | None:
        """Pre-compose quality gate — blocks render on critical violations.

        Checks:
        1. Delivery promise violation: motion-required brief with >70% still cuts → BLOCK
        2. Slideshow risk score "fail" (average ≥ 4.0) → BLOCK
        3. Missing renderer_family → WARN (log only, don't block)

        Returns a failed ToolResult if render should be blocked, None if OK to proceed.
        """
        log = logging.getLogger("video_compose")
        warnings: list[str] = []
        blocks: list[str] = []

        # --- 1. Delivery promise check ---
        delivery_data = edit_decisions.get("metadata", {}).get("delivery_promise")
        if not delivery_data:
            # Also check top-level (proposal_packet nests it at top level)
            delivery_data = edit_decisions.get("delivery_promise")

        if delivery_data:
            try:
                from lib.delivery_promise import DeliveryPromise
                promise = DeliveryPromise.from_dict(delivery_data)
                result = promise.validate_cuts(resolved_cuts)
                if not result["valid"]:
                    for v in result["violations"]:
                        blocks.append(f"Delivery promise violation: {v}")
            except Exception as e:
                log.warning("Could not validate delivery promise: %s", e)
        else:
            warnings.append("No delivery_promise in edit_decisions — skipping promise validation")

        # --- 2. Slideshow risk check ---
        renderer_family = edit_decisions.get("renderer_family")
        scenes = scene_plan or []

        # If no scene_plan passed, try to extract scene info from cuts
        if not scenes and resolved_cuts:
            scenes = [
                {
                    "type": c.get("type", ""),
                    "description": c.get("reason", ""),
                    "shot_language": c.get("shot_language", {}),
                    "shot_intent": c.get("shot_intent"),
                    "narrative_role": c.get("narrative_role"),
                    "information_role": c.get("information_role"),
                    "hero_moment": c.get("hero_moment", False),
                }
                for c in resolved_cuts
            ]

        if scenes:
            try:
                from lib.slideshow_risk import score_slideshow_risk
                render_runtime = edit_decisions.get("render_runtime")
                risk = score_slideshow_risk(
                    scenes, edit_decisions, renderer_family, render_runtime
                )
                if risk["verdict"] == "fail":
                    blocks.append(
                        f"Slideshow risk score {risk['average']:.1f}/5.0 (verdict: fail). "
                        f"Video plan looks like a slideshow — revise scene plan before rendering."
                    )
                elif risk["verdict"] == "revise":
                    warnings.append(
                        f"Slideshow risk score {risk['average']:.1f}/5.0 (verdict: revise). "
                        f"Consider improving scene variety before final render."
                    )
            except Exception as e:
                log.warning("Could not compute slideshow risk: %s", e)

        # --- 3. Missing renderer_family (BLOCK — must be set at proposal) ---
        if not renderer_family:
            blocks.append(
                "No renderer_family in edit_decisions. "
                "renderer_family must be set at proposal stage and locked before compose. "
                "Re-run the proposal stage with a renderer_family selection."
            )

        # Log warnings
        for w in warnings:
            log.warning("[pre-compose] %s", w)

        # Block on critical violations
        if blocks:
            return ToolResult(
                success=False,
                error=(
                    "Pre-compose validation failed — render blocked.\n"
                    + "\n".join(f"  • {b}" for b in blocks)
                    + ("\n\nWarnings:\n" + "\n".join(f"  • {w}" for w in warnings) if warnings else "")
                ),
            )

        return None

    def _render(self, inputs: dict[str, Any]) -> ToolResult:
        """High-level render: assemble edit decisions + asset manifest into final video.

        This is the primary entry point for the compose-director skill.
        It resolves asset IDs and routes to the composition engine:

        - **Remotion (default):** Used for all compositions when available —
          video clips, images, animated scenes, component types, mixed content.
          Remotion embeds video via <OffthreadVideo> and handles transitions,
          overlays, and profile scaling natively.
        - **FFmpeg (fallback):** Used only when Remotion is unavailable, or
          when the agent explicitly calls operation='compose' for simple
          trim/concat operations.

        The agent should pass edit_decisions, asset_manifest, and optionally
        profile, subtitle_path, audio_path, and options.

        Governance (D-2): every successful path sets `render_result.data`
        `["engine_used"]` (one of "remotion", "ffmpeg", "hyperframes",
        "remotion-atelier") so the actual engine that rendered the output is
        always visible — including when render_runtime="remotion" was locked
        but `_needs_remotion` routed to FFmpeg anyway (pure video cuts). That
        specific case additionally sets `engine_downgrade_reason` so the
        hidden engine swap isn't silent. Routing itself is unchanged.
        """
        edit_decisions = inputs.get("edit_decisions")
        asset_manifest = inputs.get("asset_manifest")
        if not edit_decisions:
            return ToolResult(success=False, error="edit_decisions required for render")

        # --- Atelier (bespoke) mode -------------------------------------
        # Hand-authored, project-local Remotion composition. Deliberately
        # bypasses the cut-schema, the stock scene-type registry, and the
        # RENDERER_FAMILY_MAP. This is the "hand-stitched every time" path:
        # the agent writes a fresh composition (its own scenes, theme, motion)
        # under remotion-composer/projects/<slug>/ and points this renderer at
        # it. No reusable creative components; a new visual language per video.
        # Triggered by composition_mode="atelier" (or renderer_family="bespoke").
        if (edit_decisions.get("composition_mode") == "atelier"
                or edit_decisions.get("renderer_family") == "bespoke"):
            atelier_result = self._render_via_atelier(inputs, edit_decisions)
            if atelier_result.success:
                if atelier_result.data is None:
                    atelier_result.data = {}
                atelier_result.data["engine_used"] = "remotion-atelier"
            return atelier_result

        if not asset_manifest:
            return ToolResult(success=False, error="asset_manifest required for render")

        output_path = Path(inputs.get("output_path", "renders/output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build asset lookup: id -> asset info
        asset_lookup = {a["id"]: a for a in asset_manifest.get("assets", [])}

        cuts = edit_decisions.get("cuts", [])
        if not cuts:
            return ToolResult(success=False, error="No cuts in edit_decisions")

        # Resolve asset IDs in cuts to file paths
        resolved_cuts = []
        for cut in cuts:
            source_id = cut.get("source", "")
            resolved_cut = dict(cut)
            if source_id in asset_lookup:
                resolved_cut["source"] = asset_lookup[source_id]["path"]
            resolved_cuts.append(resolved_cut)

        # --- Pre-compose validation gate ---
        scene_plan = inputs.get("scene_plan")
        validation_block = self._pre_compose_validation(edit_decisions, resolved_cuts, scene_plan)
        if validation_block is not None:
            return validation_block

        # Also accept profile as "output_profile" (skill convention) or "profile"
        profile = inputs.get("profile") or inputs.get("output_profile")

        # --- Runtime routing: honor render_runtime locked at proposal ---
        # Silent swaps are forbidden by governance. If the chosen runtime
        # is unavailable, surface a structured blocker rather than quietly
        # picking a different engine. Missing render_runtime is itself a
        # governance violation — edit_decisions.schema.json requires it.
        render_runtime = (edit_decisions.get("render_runtime") or "").strip().lower()

        if not render_runtime:
            return ToolResult(
                success=False,
                error=(
                    "render_runtime is not set in edit_decisions. Per governance, "
                    "it MUST be locked at proposal stage (proposal_packet."
                    "production_plan.render_runtime) and carried forward through "
                    "edit_decisions.render_runtime. Valid values: 'remotion', "
                    "'hyperframes', 'ffmpeg'. Re-run the proposal stage with an "
                    "explicit runtime choice — do NOT default this field."
                ),
            )

        if render_runtime == "hyperframes":
            hf_result = self._render_via_hyperframes(
                inputs=inputs,
                edit_decisions=edit_decisions,
                asset_manifest=asset_manifest,
                resolved_cuts=resolved_cuts,
                output_path=output_path,
                profile=profile,
            )
            if hf_result.success:
                if hf_result.data is None:
                    hf_result.data = {}
                hf_result.data["engine_used"] = "hyperframes"
            return hf_result
        if render_runtime == "ffmpeg":
            # Caller explicitly asked for FFmpeg — don't auto-upgrade to Remotion.
            ffmpeg_result = self._render_via_ffmpeg(
                inputs=inputs,
                edit_decisions=edit_decisions,
                resolved_cuts=resolved_cuts,
                output_path=output_path,
                profile=profile,
            )
            if ffmpeg_result.success:
                if ffmpeg_result.data is None:
                    ffmpeg_result.data = {}
                ffmpeg_result.data["engine_used"] = "ffmpeg"
            return ffmpeg_result
        if render_runtime != "remotion":
            return ToolResult(
                success=False,
                error=(
                    f"Unknown render_runtime {render_runtime!r}. "
                    f"Valid values: remotion, hyperframes, ffmpeg. "
                    f"render_runtime must be set at proposal stage."
                ),
            )

        # --- Explicit Remotion path (render_runtime == 'remotion') ---
        if self._needs_remotion(resolved_cuts):
            remotion_inputs: dict[str, Any] = {
                "edit_decisions": dict(edit_decisions, cuts=resolved_cuts),
                "output_path": str(output_path),
            }
            if profile:
                remotion_inputs["profile"] = profile
            render_result = self._remotion_render(remotion_inputs)

            # Governance: NEVER silently fall back to FFmpeg when Remotion fails.
            # The agent must decide the fallback path, not the tool.
            if not render_result.success:
                renderer_family = edit_decisions.get("renderer_family", "unknown")
                return ToolResult(
                    success=False,
                    error=(
                        f"Remotion render failed for renderer_family={renderer_family!r}. "
                        f"Underlying error: {render_result.error}\n\n"
                        f"This composition requires Remotion (images, text cards, animations). "
                        f"Options:\n"
                        f"  1. Fix Remotion setup (cd remotion-composer && npm install)\n"
                        f"  2. Re-run with operation='compose' for FFmpeg-only (video cuts only)\n"
                        f"  3. Approve a degraded FFmpeg render (still images → Ken Burns)\n\n"
                        f"Per governance: renderer downgrade requires user approval."
                    ),
                )

            if render_result.data is None:
                render_result.data = {}
            render_result.data["engine_used"] = "remotion"
        else:
            # --- FFmpeg fallback: only when Remotion is unavailable ---
            options = inputs.get("options", {})
            subtitle_burn = options.get("subtitle_burn", True)

            # Resolve subtitle_path from edit_decisions if not provided
            subtitle_path = inputs.get("subtitle_path")
            if subtitle_burn and not subtitle_path:
                ed_subs = edit_decisions.get("subtitles", {})
                if ed_subs.get("enabled") and ed_subs.get("source"):
                    subtitle_path = ed_subs["source"]

            # Build compose inputs
            compose_inputs = dict(inputs)
            compose_inputs["edit_decisions"] = dict(edit_decisions, cuts=resolved_cuts)
            compose_inputs["output_path"] = str(output_path)
            if subtitle_path:
                compose_inputs["subtitle_path"] = subtitle_path
            if profile:
                compose_inputs["profile"] = profile

            render_result = self._compose(compose_inputs)

            if render_result.success:
                if render_result.data is None:
                    render_result.data = {}
                render_result.data["engine_used"] = "ffmpeg"
                if render_runtime == "remotion":
                    # render_runtime was locked to Remotion at proposal, but
                    # _needs_remotion determined no cut actually requires it
                    # (pure video cuts) — this IS a hidden engine swap from
                    # the user's perspective even though routing intentionally
                    # allows it. Surface it rather than letting it stay silent.
                    render_result.data["engine_downgrade_reason"] = (
                        "render_runtime='remotion' was locked but no cut "
                        "requires Remotion (pure video cuts); rendered via "
                        "FFmpeg concat. Transitions/overlays that only "
                        "Remotion provides were not applied."
                    )

        # --- Post-render: mandatory final self-review ---
        if render_result.success and output_path.exists():
            final_review = self._run_final_review(
                output_path,
                edit_decisions,
                inputs.get("proposal_packet"),
                narration_transcript_path=inputs.get("narration_transcript_path"),
                script_text=inputs.get("script_text") or self._read_text_file(
                    inputs.get("script_path")
                ),
            )

            # Attach final_review to the ToolResult data so the compose-director
            # skill can include it in the checkpoint alongside the render_report.
            if render_result.data is None:
                render_result.data = {}
            render_result.data["final_review"] = final_review
            render_result.data["final_review_status"] = final_review["status"]

            # If the self-review says fail, downgrade the ToolResult
            if final_review["status"] == "fail":
                return ToolResult(
                    success=False,
                    error=(
                        "Post-render self-review FAILED. The output is not presentable.\n"
                        + "\n".join(f"  • {i}" for i in final_review.get("issues_found", []))
                    ),
                    data=render_result.data,
                )

        return render_result
