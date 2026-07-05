"""Remotion composition engine mixin for VideoCompose.

React-based frame-accurate rendering via `npx remotion render`. Handles the
existing scene-component stack (text cards, stat cards, charts, callouts,
comparisons), word-level caption burn, and TalkingHead/CinematicRenderer.
Remotion is the default composition engine when available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.base_tool import ToolResult


class RemotionEngineMixin:
    """Remotion availability checks, theming, routing heuristics, and render.

    Plain mixin — no BaseTool inheritance. References `self.*` attributes
    (e.g. `self.run_command`, `self._IMAGE_EXTENSIONS`) provided by the final
    composed VideoCompose class.
    """

    # Remotion scene types that trigger React-based rendering
    _REMOTION_COMPONENTS = [
        "text_card", "stat_card", "callout", "comparison",
        "progress", "chart", "bar_chart", "line_chart", "pie_chart", "kpi_grid",
    ]

    _REMOTION_SCENE_TYPES = {
        "text_card", "stat_card", "callout", "comparison", "progress", "chart",
    }

    # Maps renderer_family (set at proposal stage) to Remotion composition ID.
    # Each family MUST map to a distinct composition — collapsing defeats visual grammar.
    # Maps renderer_family → Remotion composition ID.
    # Only compositions registered in remotion-composer/src/Root.tsx are valid.
    # Current compositions: Explainer, CinematicRenderer, TalkingHead
    RENDERER_FAMILY_MAP = {
        "explainer-data": "Explainer",
        "explainer-teacher": "Explainer",
        "cinematic-trailer": "CinematicRenderer",
        "documentary-montage": "CinematicRenderer",
        "product-reveal": "Explainer",
        "screen-demo": "Explainer",
        "presenter": "TalkingHead",
        "animation-first": "Explainer",
    }

    def _remotion_available(self) -> bool:
        """Check if Remotion rendering is available (requires npx + composer project + node_modules)."""
        import shutil as _shutil

        if not _shutil.which("npx"):
            return False
        composer_dir = Path(__file__).resolve().parent.parent.parent.parent / "remotion-composer"
        if not composer_dir.exists() or not (composer_dir / "package.json").exists():
            return False
        # Check that node_modules are actually installed — without this,
        # npx remotion render will fail even though the project exists.
        if not (composer_dir / "node_modules").exists():
            return False
        return True

    @classmethod
    def _get_composition_id(cls, renderer_family: str) -> str:
        """Resolve renderer_family to Remotion composition ID.

        Raises ValueError if renderer_family is not recognized — the caller
        must set it at proposal stage.
        """
        comp = cls.RENDERER_FAMILY_MAP.get(renderer_family)
        if comp is None:
            raise ValueError(
                f"Unknown renderer_family {renderer_family!r}. "
                f"Valid families: {sorted(cls.RENDERER_FAMILY_MAP)}. "
                f"Set renderer_family at proposal stage."
            )
        return comp

    @staticmethod
    def _build_theme_from_playbook(
        playbook_name: str | None,
        composition_data: dict | None,
    ) -> dict[str, Any] | None:
        """Derive a Remotion ThemeConfig from a playbook's actual color values.

        Instead of passing a playbook name and hoping Remotion has a matching
        preset, we read the playbook YAML and extract concrete colors/fonts.
        This means custom playbooks, overridden palettes, and per-project
        styles all flow through to Remotion automatically.

        Falls back to extracting colors from edit_decisions metadata if
        no playbook is loadable.
        """
        theme: dict[str, Any] = {}

        # Try to load the playbook YAML
        playbook: dict[str, Any] = {}
        if playbook_name:
            try:
                from styles.playbook_loader import load_playbook
                playbook = load_playbook(playbook_name)
            except Exception:
                pass

        if playbook:
            vl = playbook.get("visual_language", {})
            palette = vl.get("color_palette", {})
            typo = playbook.get("typography", {})

            # Extract primary/accent — may be a list (gradient stops) or string
            primary_raw = palette.get("primary", ["#2563EB"])
            accent_raw = palette.get("accent", ["#F59E0B"])
            primary = primary_raw[0] if isinstance(primary_raw, list) else primary_raw
            accent = accent_raw[0] if isinstance(accent_raw, list) else accent_raw

            bg = palette.get("background", "#FFFFFF")
            text = palette.get("text", "#1F2937")
            surface = palette.get("surface", bg)
            muted = palette.get("muted_text", "#6B7280")

            # Build chart colors from all palette entries
            chart_colors = []
            for key in ["primary", "accent", "secondary", "success", "warning", "info"]:
                val = palette.get(key)
                if val:
                    chart_colors.append(val[0] if isinstance(val, list) else val)
            if len(chart_colors) < 3:
                chart_colors = [primary, accent, "#10B981", "#8B5CF6", "#EC4899", "#06B6D4"]

            theme = {
                "primaryColor": primary,
                "accentColor": accent,
                "backgroundColor": bg,
                "surfaceColor": surface,
                "textColor": text,
                "mutedTextColor": muted,
                "headingFont": typo.get("heading", {}).get("font", "Inter"),
                "bodyFont": typo.get("body", {}).get("font", "Inter"),
                "monoFont": typo.get("code", {}).get("font", "JetBrains Mono"),
                "chartColors": chart_colors[:6],
                "springConfig": {"damping": 20, "stiffness": 120, "mass": 1},
                "transitionDuration": 0.4,
            }

            # Derive caption colors from the palette
            theme["captionHighlightColor"] = primary
            # Caption background: semi-transparent version of the bg color
            theme["captionBackgroundColor"] = (
                f"rgba(255, 255, 255, 0.85)" if bg.upper() in ("#FFFFFF", "#FAFAFA", "#F9FAFB")
                else f"rgba(15, 23, 42, 0.75)"
            )

            # Motion style from playbook
            motion = playbook.get("motion", {})
            pace = motion.get("pace", "moderate")
            if pace == "fast":
                theme["springConfig"] = {"damping": 12, "stiffness": 80, "mass": 1}
                theme["transitionDuration"] = 0.3
            elif pace == "slow":
                theme["springConfig"] = {"damping": 25, "stiffness": 150, "mass": 1}
                theme["transitionDuration"] = 0.6

        # Fallback: try to extract from edit_decisions metadata
        if not theme and composition_data:
            meta = composition_data.get("metadata", {})
            if meta.get("primary_color"):
                theme = {
                    "primaryColor": meta["primary_color"],
                    "accentColor": meta.get("accent_color", "#F59E0B"),
                    "backgroundColor": meta.get("background_color", "#FFFFFF"),
                    "surfaceColor": meta.get("surface_color", "#F9FAFB"),
                    "textColor": meta.get("text_color", "#1F2937"),
                    "mutedTextColor": "#6B7280",
                    "headingFont": meta.get("heading_font", "Inter"),
                    "bodyFont": meta.get("body_font", "Inter"),
                    "monoFont": "JetBrains Mono",
                    "chartColors": meta.get("chart_colors", ["#2563EB", "#F59E0B", "#10B981"]),
                    "springConfig": {"damping": 20, "stiffness": 120, "mass": 1},
                    "transitionDuration": 0.4,
                    "captionHighlightColor": meta["primary_color"],
                    "captionBackgroundColor": "rgba(255, 255, 255, 0.85)",
                }

        return theme if theme else None

    def _needs_remotion(self, cuts: list[dict]) -> bool:
        """Determine whether Remotion should handle this composition.

        Remotion is the DEFAULT composition engine when available.  It handles
        video clips (via <OffthreadVideo>), still images, animated scene types,
        component types, transitions, and mixed content — all in a single
        React-based render pass.

        Returns False (i.e. use FFmpeg) only when Remotion is not
        available. For `operation="render"` the governance default is
        Remotion-first: the renderer family was chosen earlier, and the
        tool should preserve that decision instead of silently
        downgrading to FFmpeg.

        This "Remotion-first" policy means mixed content (video clips +
        animated stills + text cards) is always composed in Remotion, which
        can embed <OffthreadVideo> alongside React components natively.
        """
        # If Remotion isn't installed, fall back to FFmpeg
        if not self._remotion_available():
            return False

        # Any rich content → Remotion (fast path, catches the obvious cases)
        for cut in cuts:
            source = cut.get("source", "")
            if source and Path(source).suffix.lower() in self._IMAGE_EXTENSIONS:
                return True
            if cut.get("type") in self._REMOTION_SCENE_TYPES:
                return True
            if cut.get("animation") or cut.get("transition_in") or cut.get("transition_out"):
                return True
            transform = cut.get("transform", {})
            if transform and transform.get("animation"):
                return True

        # Even for pure-video cuts, default to Remotion — it handles video
        # clips natively via <OffthreadVideo> and gives us transitions,
        # overlays, and profile scaling for free.
        return True

    def _remotion_render(self, inputs: dict[str, Any]) -> ToolResult:
        """Render via Remotion (requires Node.js + npx).

        Handles compositions with still images, animated scenes, component
        types, and transitions using React-based frame-accurate rendering.
        Accepts edit_decisions (with resolved file paths) or raw composition_data.
        """
        import shutil

        if not shutil.which("npx"):
            return ToolResult(
                success=False,
                error="npx not found. Install Node.js to use Remotion rendering.",
            )

        composition_data = inputs.get("edit_decisions") or inputs.get("composition_data")
        if not composition_data:
            return ToolResult(
                success=False,
                error="edit_decisions or composition_data required for remotion_render",
            )

        output_path = Path(inputs.get("output_path", "renders/remotion_output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Absolutise so the CLI can resolve the output regardless of cwd.
        output_path = output_path.resolve()

        # Deep-copy props so we don't mutate the original
        props = json.loads(json.dumps(composition_data))

        # Convert absolute file paths to file:// URIs for Remotion's
        # Img and OffthreadVideo components
        for cut in props.get("cuts", []):
            source = cut.get("source", "")
            if source and not source.startswith(("http://", "https://", "file://")):
                resolved = Path(source).resolve()
                if resolved.exists():
                    posix = resolved.as_posix()
                    cut["source"] = f"file:///{posix}" if not posix.startswith("/") else f"file://{posix}"

        # Build a custom themeConfig from the playbook's actual colors.
        # This ensures every video gets a unique visual identity derived
        # from its production decisions — not picked from a preset menu.
        if "themeConfig" not in props:
            playbook_name = (
                props.get("playbook")
                or props.get("theme")
                or props.get("metadata", {}).get("playbook")
            )
            theme_config = self._build_theme_from_playbook(playbook_name, composition_data)
            if theme_config:
                props["themeConfig"] = theme_config

        # Write props to temp file for Remotion CLI
        props_path = output_path.parent / ".remotion_props.json"
        with open(props_path, "w", encoding="utf-8") as f:
            json.dump(props, f)

        # remotion-composer lives at project root
        composer_dir = Path(__file__).resolve().parent.parent.parent.parent / "remotion-composer"
        if not composer_dir.exists():
            return ToolResult(
                success=False,
                error=f"Remotion composer project not found at {composer_dir}",
            )

        # Route to the correct Remotion composition based on renderer_family.
        # This prevents all pipelines from collapsing into the Explainer visual grammar.
        renderer_family = (composition_data or {}).get("renderer_family", "explainer-data")
        composition_id = self._get_composition_id(renderer_family)

        cmd = [
            "npx", "remotion", "render",
            str(composer_dir / "src" / "index.tsx"),
            composition_id,
            str(output_path),
            # Use the `--props=<path>` equals form rather than two separate
            # args. On Windows, passing `--props` and the path separately makes
            # Remotion mis-parse the value (quote escaping differs), failing
            # with "neither valid JSON nor a file path". The equals form is the
            # API Remotion recommends for file paths and is cross-platform safe.
            f"--props={props_path}",
        ]

        # Apply media profile dimensions
        profile_name = inputs.get("profile")
        if profile_name:
            try:
                from lib.media_profiles import get_profile
                p = get_profile(profile_name)
                cmd.extend(["--width", str(p.width), "--height", str(p.height)])
            except (ImportError, ValueError):
                pass

        try:
            # Invoke from inside the composer dir so npx can resolve the
            # local remotion binary via node_modules/.bin. Without this,
            # Windows npx cannot locate the CLI and returns "could not
            # determine executable to run".
            self.run_command(cmd, timeout=600, cwd=composer_dir)
        except Exception as e:
            return ToolResult(success=False, error=f"Remotion render failed: {e}")
        finally:
            if props_path.exists():
                props_path.unlink()

        if not output_path.exists():
            return ToolResult(
                success=False,
                error=f"Remotion render completed but output file missing: {output_path}",
            )

        return ToolResult(
            success=True,
            data={
                "operation": "remotion_render",
                "output": str(output_path),
                "profile": profile_name,
            },
            artifacts=[str(output_path)],
        )
