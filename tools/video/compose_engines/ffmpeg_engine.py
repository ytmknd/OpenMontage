"""FFmpeg composition engine mixin for VideoCompose.

Pure-FFmpeg render path: concat/trim cuts, mux audio, burn subtitles,
composite overlays, and re-encode. Used directly when `render_runtime`
is locked to "ffmpeg", and as the fallback path when Remotion/HyperFrames
are unavailable or unnecessary for a given set of cuts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import ToolResult


class FFmpegEngineMixin:
    """FFmpeg-only composition, subtitle burn, overlay, and encode operations.

    Plain mixin — no BaseTool inheritance. References `self.*` attributes
    (e.g. `self.run_command`, `self._is_image`, `self._run_final_review`)
    provided by the final composed VideoCompose class.
    """

    def _compose(self, inputs: dict[str, Any]) -> ToolResult:
        """FFmpeg composition: concat video cuts, add audio, burn subtitles.

        Handles video sources only. Still images and animated scene types
        are routed to Remotion via the render operation — call compose
        directly only for pure video pipelines (e.g. talking-head).
        """
        edit_decisions = inputs.get("edit_decisions")
        if not edit_decisions:
            return ToolResult(success=False, error="edit_decisions required for compose")

        output_path = Path(inputs.get("output_path", "composed_output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path = inputs.get("audio_path")
        subtitle_path = inputs.get("subtitle_path")
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)
        preset = inputs.get("preset", "medium")
        profile_name = inputs.get("profile")

        # Resolve target resolution from profile or default
        resolution = "1920x1080"
        if profile_name:
            try:
                from lib.media_profiles import get_profile
                p = get_profile(profile_name)
                resolution = f"{p.width}x{p.height}"
            except (ImportError, ValueError):
                pass

        cuts = edit_decisions.get("cuts", [])
        if not cuts:
            return ToolResult(success=False, error="No cuts in edit_decisions")

        # Resolve subtitle style using the layered priority resolver
        # (explicit > edit_decisions > playbook > defaults)
        playbook_data = inputs.get("playbook")
        resolved_sub_style = self._resolve_subtitle_style(
            inputs.get("subtitle_style"),
            edit_decisions,
            playbook_data,
        )
        inputs = dict(inputs)
        inputs["subtitle_style"] = resolved_sub_style

        ed_subs = edit_decisions.get("subtitles", {})
        if ed_subs.get("source") and not subtitle_path:
            subtitle_path = ed_subs["source"]

        temp_dir = output_path.parent / ".compose_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_segments: list[Path] = []
        concat_path: Path | None = None
        concat_out: Path | None = None

        try:
            for i, cut in enumerate(cuts):
                source = Path(cut["source"])
                if not source.exists():
                    return ToolResult(success=False, error=f"Cut source not found: {source}")

                seg_path = temp_dir / f"seg_{i:04d}.mp4"
                in_s = cut["in_seconds"]
                out_s = cut["out_seconds"]
                duration = out_s - in_s
                speed = cut.get("speed", 1.0)

                if self._is_image(source):
                    return ToolResult(
                        success=False,
                        error=(
                            f"Still image '{source.name}' in cuts. "
                            "Use operation='render' (auto-routes to Remotion) "
                            "or operation='remotion_render' for compositions "
                            "with images, animations, or component scenes."
                        ),
                    )
                else:
                    # Video source: trim to segment.
                    #
                    # Semantics:
                    #   -ss BEFORE -i   → fast input-level seek to in_s
                    #   -t  AFTER  -i   → "play for `duration` seconds"
                    #                     (unambiguous regardless of seek mode)
                    #
                    # We MUST re-encode here — `-c copy` cannot do frame-accurate
                    # cuts because it snaps to keyframes. With sparse GOPs (common
                    # in Pexels / AI-generated clips), stream-copy can produce
                    # segments significantly longer than `duration`, breaking the
                    # target timeline. Re-encoding with libx264/AAC is slower but
                    # gives exact cut boundaries. Same resolution in → same
                    # resolution out, so same-res inputs concat cleanly.
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(in_s),
                        "-t", str(duration),
                        "-i", str(source),
                    ]

                    # Normalize every segment to a consistent container so the
                    # concat-copy step is always safe. The concat demuxer with
                    # `-c copy` requires identical codec / resolution / fps /
                    # pix_fmt / sar across ALL segments — otherwise it throws
                    # "Non-monotonous DTS" or silently produces corrupt output.
                    #
                    # Default target is 1920x1080 @ 30fps, yuv420p, sar=1. If the
                    # source is smaller it letterboxes; if larger it downscales.
                    # Callers can override via edit_decisions.metadata.compose_target
                    # (future extension) but the defaults match the most common
                    # delivery profile (YouTube landscape).
                    vf_parts: list[str] = [
                        "scale=1920:1080:force_original_aspect_ratio=decrease",
                        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
                        "setsar=1",
                        "fps=30",
                    ]
                    af_parts: list[str] = []
                    if speed != 1.0:
                        vf_parts.append(f"setpts={1.0/speed}*PTS")
                        af_parts.append(self._build_atempo(speed))

                    cmd.extend(["-filter:v", ",".join(vf_parts)])
                    if af_parts:
                        cmd.extend(["-filter:a", ",".join(af_parts)])

                    cmd.extend([
                        "-c:v", codec,
                        "-crf", str(crf),
                        "-preset", preset,
                        "-pix_fmt", "yuv420p",
                        "-r", "30",
                    ])

                    # Audio handling: some source clips have no audio stream
                    # (Pexels stock often ships silent). If we unconditionally
                    # ask ffmpeg to copy/encode the 0:a stream it errors out.
                    # Probe for an audio stream first — if present, transcode
                    # to AAC; if absent, synthesize a silent stereo track so
                    # concat segments have a consistent stream layout.
                    has_audio = self._has_audio_stream(source)
                    if has_audio:
                        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
                    else:
                        # Inject silent audio via lavfi before the output.
                        # We have to rebuild cmd to add the lavfi input
                        # before the output path and map streams explicitly.
                        cmd = [
                            "ffmpeg", "-y",
                            "-ss", str(in_s),
                            "-t", str(duration),
                            "-i", str(source),
                            "-f", "lavfi",
                            "-t", str(duration),
                            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                            "-filter:v", ",".join(vf_parts),
                        ]
                        if af_parts:
                            cmd.extend(["-filter:a", ",".join(af_parts)])
                        cmd.extend([
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-c:v", codec,
                            "-crf", str(crf),
                            "-preset", preset,
                            "-pix_fmt", "yuv420p",
                            "-r", "30",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-ar", "48000",
                            "-ac", "2",
                        ])

                    cmd.append(str(seg_path))
                    self.run_command(cmd)

                temp_segments.append(seg_path)

            # Step 2: Concat segments
            concat_path = temp_dir / "concat_list.txt"
            with open(concat_path, "w", encoding="utf-8") as f:
                for seg in temp_segments:
                    safe = str(seg.resolve()).replace("\\", "/")
                    f.write(f"file '{safe}'\n")

            concat_out = temp_dir / "concat.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_path),
                "-c", "copy",
                str(concat_out),
            ]
            self.run_command(cmd)

            # Step 3: Apply subtitles and/or replace audio
            final_input = concat_out
            vfilters = []

            if subtitle_path and Path(subtitle_path).exists():
                style = inputs.get("subtitle_style", {})
                ass_style = self._build_subtitle_style(style)
                sub_escaped = str(Path(subtitle_path).resolve()).replace("\\", "/").replace(":", "\\:")
                vfilters.append(f"subtitles='{sub_escaped}':force_style='{ass_style}'")

            cmd = ["ffmpeg", "-y", "-i", str(final_input)]

            if audio_path and Path(audio_path).exists():
                cmd.extend(["-i", audio_path])

            # Determine if profile requires re-encoding (resize/fps change)
            # This must be checked BEFORE choosing copy vs encode, because
            # -s and -r are incompatible with -c:v copy.
            profile_flags: list[str] = []
            if profile_name:
                try:
                    from lib.media_profiles import get_profile
                    p = get_profile(profile_name)
                    profile_flags = ["-s", f"{p.width}x{p.height}", "-r", str(p.fps)]
                except (ImportError, ValueError):
                    pass

            needs_reencode = bool(vfilters) or bool(profile_flags)

            if needs_reencode:
                if vfilters:
                    cmd.extend(["-vf", ",".join(vfilters)])
                cmd.extend(["-c:v", codec, "-crf", str(crf), "-preset", preset])
                cmd.extend(profile_flags)
            else:
                cmd.extend(["-c:v", "copy"])

            if audio_path and Path(audio_path).exists():
                # Use type-based selectors (0:v, 1:a) instead of index-based
                # (0:v:0) because source videos may have audio as stream 0
                # and video as stream 1 (e.g. Kling-generated clips).
                cmd.extend(["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"])
            else:
                cmd.extend(["-c:a", "copy"])

            cmd.append(str(output_path))
            self.run_command(cmd)

            return ToolResult(
                success=True,
                data={
                    "operation": "compose",
                    "cut_count": len(cuts),
                    "has_subtitles": subtitle_path is not None,
                    "has_mixed_audio": audio_path is not None,
                    "profile": profile_name,
                    "output": str(output_path),
                },
                artifacts=[str(output_path)],
            )
        finally:
            # Cleanup temp files
            for f in temp_segments:
                if f.exists():
                    f.unlink()
            for f in [concat_path, concat_out]:
                if f is not None and f.exists():
                    f.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass

    def _render_via_ffmpeg(
        self,
        *,
        inputs: dict[str, Any],
        edit_decisions: dict[str, Any],
        resolved_cuts: list[dict],
        output_path: Path,
        profile: Optional[str],
    ) -> ToolResult:
        """Explicit FFmpeg-only render path.

        Use when the proposal locked `render_runtime="ffmpeg"` — e.g. simple
        source-footage concat/trim jobs that don't benefit from composition.
        Still runs the mandatory final self-review.
        """
        options = inputs.get("options", {})
        subtitle_burn = options.get("subtitle_burn", True)

        subtitle_path = inputs.get("subtitle_path")
        if subtitle_burn and not subtitle_path:
            ed_subs = edit_decisions.get("subtitles", {})
            if ed_subs.get("enabled") and ed_subs.get("source"):
                subtitle_path = ed_subs["source"]

        compose_inputs = dict(inputs)
        compose_inputs["edit_decisions"] = dict(edit_decisions, cuts=resolved_cuts)
        compose_inputs["output_path"] = str(output_path)
        if subtitle_path:
            compose_inputs["subtitle_path"] = subtitle_path
        if profile:
            compose_inputs["profile"] = profile

        render_result = self._compose(compose_inputs)

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
            if render_result.data is None:
                render_result.data = {}
            render_result.data["final_review"] = final_review
            render_result.data["final_review_status"] = final_review["status"]
            if final_review["status"] == "fail":
                return ToolResult(
                    success=False,
                    error=(
                        "Post-render self-review FAILED (FFmpeg). The output is not presentable.\n"
                        + "\n".join(f"  • {i}" for i in final_review.get("issues_found", []))
                    ),
                    data=render_result.data,
                )

        return render_result

    @staticmethod
    def _has_audio_stream(path: Path) -> bool:
        """Return True iff ffprobe reports at least one audio stream.

        Many stock video clips (especially from Pexels) ship with no audio
        stream at all. If we blindly tell ffmpeg to transcode the 0:a stream
        on such a file it errors out. This helper lets the segment builder
        branch on stream presence so it can synthesize a silent track when
        needed, keeping the concat segment layout consistent.
        """
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=nw=1:nk=1",
                    str(path),
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
            return "audio" in out
        except Exception:
            return False

    @staticmethod
    def _parse_probe_fps(fps_str: str) -> float:
        """Parse ffprobe fps string like '30/1' or '24000/1001'."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return round(int(num) / max(int(den), 1), 2)
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _burn_subtitles(self, inputs: dict[str, Any]) -> ToolResult:
        """Burn subtitle file into video."""
        input_path = Path(inputs["input_path"])
        subtitle_path = Path(inputs["subtitle_path"])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_subtitled"))))

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not subtitle_path.exists():
            return ToolResult(success=False, error=f"Subtitle file not found: {subtitle_path}")

        style = inputs.get("subtitle_style", {})
        ass_style = self._build_subtitle_style(style)
        sub_escaped = str(subtitle_path.resolve()).replace("\\", "/").replace(":", "\\:")
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"subtitles='{sub_escaped}':force_style='{ass_style}'",
            "-c:v", codec, "-crf", str(crf),
            "-c:a", "copy",
            str(output_path),
        ]

        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "burn_subtitles",
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    def _overlay(self, inputs: dict[str, Any]) -> ToolResult:
        """Composite overlay images/videos on top of base video."""
        input_path = Path(inputs["input_path"])
        overlays = inputs.get("overlays", [])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_overlay"))))
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not overlays:
            return ToolResult(success=False, error="No overlays provided")

        # Build complex filter for each overlay
        input_args = ["-i", str(input_path)]
        filter_parts = []
        prev_label = "0:v"

        for i, ov in enumerate(overlays):
            asset_path = Path(ov["asset_path"])
            if not asset_path.exists():
                return ToolResult(success=False, error=f"Overlay asset not found: {asset_path}")

            input_args.extend(["-i", str(asset_path)])

            x = int(ov.get("x", 0))
            y = int(ov.get("y", 0))
            start = ov.get("start_seconds", 0)
            end = ov.get("end_seconds")
            opacity = ov.get("opacity", 1.0)

            overlay_input = f"{i + 1}:v"

            # Scale overlay if dimensions specified
            if "width" in ov and "height" in ov:
                w = int(ov["width"])
                h = int(ov["height"])
                filter_parts.append(f"[{overlay_input}]scale={w}:{h}[ov_scaled_{i}]")
                overlay_input = f"ov_scaled_{i}"

            # Build enable expression for timed overlays
            enable = f"between(t,{start},{end})" if end else f"gte(t,{start})"
            out_label = f"v{i}"

            filter_parts.append(
                f"[{prev_label}][{overlay_input}]overlay={x}:{y}:enable='{enable}'[{out_label}]"
            )
            prev_label = out_label

        filter_complex = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"]
        cmd.extend(input_args)
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", f"[{prev_label}]", "-map", "0:a?"])
        cmd.extend(["-c:v", codec, "-crf", str(crf), "-c:a", "copy"])
        cmd.append(str(output_path))

        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "overlay",
                "overlay_count": len(overlays),
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    def _encode(self, inputs: dict[str, Any]) -> ToolResult:
        """Re-encode video with a specific profile/codec settings."""
        input_path = Path(inputs["input_path"])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_encoded"))))
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)
        preset = inputs.get("preset", "medium")
        profile_name = inputs.get("profile")

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", codec, "-crf", str(crf), "-preset", preset,
            "-c:a", "aac", "-b:a", "192k",
        ]

        # Apply media profile if specified
        if profile_name:
            try:
                from lib.media_profiles import get_profile, ffmpeg_output_args
                profile = get_profile(profile_name)
                cmd.extend(["-s", f"{profile.width}x{profile.height}"])
                cmd.extend(["-r", str(profile.fps)])
            except (ImportError, ValueError):
                pass  # proceed without profile

        cmd.append(str(output_path))
        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "encode",
                "codec": codec,
                "crf": crf,
                "profile": profile_name,
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    @staticmethod
    def _resolve_subtitle_style(
        explicit_style: dict | None,
        edit_decisions: dict | None,
        playbook: dict | None,
    ) -> dict:
        """Resolve subtitle style with layered priority.

        Priority: explicit_style > edit_decisions.subtitles.style > playbook > defaults.
        This prevents every video from looking identical (Arial bold white).
        """
        # Start with minimal fallback defaults
        resolved = {
            "font": "Inter",
            "font_size": 28,
            "bold": True,
            "outline_width": 2,
            "shadow": 0,
            "margin_v": 40,
            "alignment": 2,
        }

        # Layer 1: Playbook-derived style
        if playbook:
            typo = playbook.get("typography", {})
            colors = playbook.get("visual_language", {}).get("color_palette", {})
            if typo.get("body", {}).get("family"):
                resolved["font"] = typo["body"]["family"]
            if colors.get("text"):
                resolved["primary_color"] = colors["text"]
            if colors.get("background"):
                resolved["outline_color"] = colors["background"]
                # Semi-transparent background for readability
                bg = colors["background"]
                resolved["back_color"] = bg

        # Layer 2: edit_decisions subtitle style
        if edit_decisions:
            ed_style = edit_decisions.get("subtitles", {}).get("style", {})
            for k, v in ed_style.items():
                if v is not None:
                    resolved[k] = v

        # Layer 3: Explicit override (highest priority)
        if explicit_style:
            for k, v in explicit_style.items():
                if v is not None:
                    resolved[k] = v

        return resolved

    @staticmethod
    def _build_subtitle_style(style: dict) -> str:
        """Build ASS force_style string from style dict."""
        parts = []
        parts.append(f"FontName={style.get('font', 'Inter')}")
        parts.append(f"FontSize={style.get('font_size', 28)}")
        parts.append(f"Bold={1 if style.get('bold', True) else 0}")
        if style.get("primary_color"):
            parts.append(f"PrimaryColour={style['primary_color']}")
        if style.get("outline_color"):
            parts.append(f"OutlineColour={style['outline_color']}")
        if style.get("back_color"):
            parts.append(f"BackColour={style['back_color']}")
        border_style = style.get("border_style", 1)
        parts.append(f"BorderStyle={border_style}")
        parts.append(f"Outline={style.get('outline_width', 2)}")
        parts.append(f"Shadow={style.get('shadow', 0)}")
        parts.append(f"MarginV={style.get('margin_v', 40)}")
        parts.append(f"Alignment={style.get('alignment', 2)}")
        return ",".join(parts)

    @staticmethod
    def _build_atempo(factor: float) -> str:
        """Build atempo filter chain for audio speed adjustment."""
        filters = []
        remaining = factor
        while remaining > 100.0:
            filters.append("atempo=100.0")
            remaining /= 100.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)
