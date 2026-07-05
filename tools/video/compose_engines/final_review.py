"""Final self-review mixin for VideoCompose.

Mandatory post-render inspection: this is the governance contract that the
compose runtime MUST inspect the actual rendered output (ffprobe technical
probe, sampled-frame visual spotcheck, audio volume spotcheck, promise
preservation, subtitle presence, and transcript-vs-script comparison) before
any stage can claim a video is ready.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any


class FinalReviewMixin:
    """Post-render probes and the transcript-vs-script content check.

    Plain mixin — no BaseTool inheritance. References `self.*`/`cls.*`
    attributes (e.g. `self._parse_probe_fps`) provided by the final composed
    VideoCompose class.
    """

    # Punctuation/SSML-leak words that should NEVER appear in rendered audio.
    # When a TTS engine reads a literal "..." as the word "dot", or a "—" as
    # "hyphen", those leak into the transcript. Catching these in the final
    # review is the difference between catching a bad voice render in-tool
    # vs. shipping a video that says "dot dot dot" twelve times. CRITICAL.
    _TTS_PUNCTUATION_LEAK_WORDS = {
        "dot", "dots", "ellipsis", "period", "periods",
        "comma", "commas", "semicolon", "colon",
        "dash", "hyphen", "emdash", "endash",
        "parenthesis", "bracket", "brace",
        "asterisk", "slash", "backslash",
        "exclamation", "question mark",
    }

    @staticmethod
    def _read_text_file(path: str | Path | None) -> str | None:
        """Read a small text file if given a path; None-safe and exception-safe."""
        if not path:
            return None
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception:
            return None

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Split text into comparable word tokens (lowercased, punctuation
        stripped, numeric-word-aware). Empty tokens dropped."""
        import re

        # Preserve hyphenated words as single tokens ("many-worlds" -> "many-worlds").
        # Drop everything except letters, digits, hyphens, apostrophes.
        cleaned = re.sub(r"[^A-Za-z0-9\-' ]+", " ", text.lower())
        return [t for t in cleaned.split() if t and t != "-"]

    @classmethod
    def _compare_transcript_to_script(
        cls,
        transcript_path: Path,
        script_text: str,
    ) -> dict[str, Any]:
        """Compare a word-level transcript against the source script.

        Purpose: catch TTS failures that look fine on audio-volume/duration
        checks but produce garbage content. The canonical example is
        Chirp3-HD reading ellipses ("...") literally as the word "dot" — our
        volume check says "narration present, not clipped" and the video
        ships. This check diffs the actual transcribed audio against what
        was supposed to be said, and flags:

        - Spurious punctuation-leak words ("dot", "comma", "hyphen", etc.)
          that appear in audio but not script → CRITICAL
        - Overall word-accuracy ratio against script → SUGGESTION if < 0.9

        Returns the transcript_comparison section of final_review, or a
        placeholder with an issue describing why the check couldn't run
        (missing transcript, missing script) so the review never goes
        silently quiet on this contract.
        """
        result: dict[str, Any] = {
            "transcript_matches_script": False,
            "word_accuracy": None,
            "script_word_count": 0,
            "transcript_word_count": 0,
            "spurious_punctuation_words": [],
            "issues": [],
        }

        if not transcript_path or not Path(transcript_path).is_file():
            result["issues"].append(
                "transcript_comparison skipped: narration_transcript not provided"
            )
            return result
        if not script_text:
            result["issues"].append(
                "transcript_comparison skipped: script_text not provided"
            )
            return result

        try:
            transcript_data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
        except Exception as e:
            result["issues"].append(f"transcript_comparison could not parse transcript: {e}")
            return result

        transcript_words = [
            w.get("word", "").strip() for w in transcript_data.get("word_timestamps", [])
        ]
        transcript_tokens = cls._tokenize(" ".join(transcript_words))
        script_tokens = cls._tokenize(script_text)

        result["script_word_count"] = len(script_tokens)
        result["transcript_word_count"] = len(transcript_tokens)

        if not script_tokens or not transcript_tokens:
            result["issues"].append(
                f"transcript_comparison: empty token set "
                f"(script={len(script_tokens)}, transcript={len(transcript_tokens)})"
            )
            return result

        # --- Punctuation-leak detection (TTS reading literal punctuation) ---
        script_set = set(script_tokens)
        leak_occurrences: dict[str, int] = {}
        for token in transcript_tokens:
            if token in cls._TTS_PUNCTUATION_LEAK_WORDS and token not in script_set:
                leak_occurrences[token] = leak_occurrences.get(token, 0) + 1

        if leak_occurrences:
            formatted = ", ".join(
                f"{w!r}×{n}" for w, n in sorted(leak_occurrences.items(), key=lambda x: -x[1])
            )
            result["spurious_punctuation_words"] = [
                {"word": w, "count": n} for w, n in leak_occurrences.items()
            ]
            result["issues"].append(
                f"TTS punctuation leak: transcript contains {formatted} — "
                f"these words are NOT in the script, which means the voice "
                f"engine is reading literal punctuation aloud. Rewrite the "
                f"script to eliminate the corresponding characters (ellipses, "
                f"em-dashes, etc.) and regenerate narration."
            )

        # --- Word accuracy via set overlap (cheap & ordering-insensitive) ---
        # We don't penalize small word-order differences or minor TTS
        # hallucinations; we just want to know "did 90%+ of the script's
        # content make it into the audio." Using set overlap on the script
        # side is robust to transcription noise.
        matched = sum(1 for t in script_tokens if t in set(transcript_tokens))
        accuracy = matched / max(1, len(script_tokens))
        result["word_accuracy"] = round(accuracy, 3)
        result["transcript_matches_script"] = accuracy >= 0.9 and not leak_occurrences

        if accuracy < 0.9:
            result["issues"].append(
                f"Low transcript-to-script match: only {accuracy:.0%} of script "
                f"words appear in the transcribed audio ({matched}/"
                f"{len(script_tokens)}). Narration may be truncated, mispronounced, "
                f"or the wrong script was used."
            )

        return result

    def _run_final_review(
        self,
        output_path: Path,
        edit_decisions: dict[str, Any] | None = None,
        proposal_packet: dict[str, Any] | None = None,
        narration_transcript_path: str | Path | None = None,
        script_text: str | None = None,
    ) -> dict[str, Any]:
        """Run post-render self-review and produce a final_review artifact.

        This is the governance contract: the compose runtime MUST inspect
        the actual rendered output before marking the stage complete.
        Never claim a video is ready without a real probe + frame sample.

        When `proposal_packet` is provided, its
        `production_plan.render_runtime` is compared against
        `edit_decisions.render_runtime` so `runtime_swap_detected` can
        actually flip. Without it, we fall back to
        `edit_decisions.metadata.proposal_render_runtime` (which the edit
        director can set explicitly to opt into swap detection).

        Returns a dict conforming to final_review.schema.json.
        """
        log = logging.getLogger("video_compose.final_review")
        issues: list[str] = []

        # --- 1. Technical probe via ffprobe ---
        technical_probe: dict[str, Any] = {
            "valid_container": False,
            "issues": [],
        }
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(output_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                probe_data = json.loads(proc.stdout)
                fmt = probe_data.get("format", {})
                streams = probe_data.get("streams", [])
                video_stream = next(
                    (s for s in streams if s.get("codec_type") == "video"), {}
                )
                audio_stream = next(
                    (s for s in streams if s.get("codec_type") == "audio"), {}
                )

                duration = float(fmt.get("duration", 0))
                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))
                fps_str = video_stream.get("r_frame_rate", "0/1")
                fps = self._parse_probe_fps(fps_str)

                technical_probe = {
                    "valid_container": bool(video_stream),
                    "duration_seconds": round(duration, 2),
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "has_audio": bool(audio_stream),
                    "codec": video_stream.get("codec_name", "unknown"),
                    "file_size_bytes": int(fmt.get("size", 0)),
                    "issues": [],
                }

                # Sanity checks
                if duration < 1.0:
                    technical_probe["issues"].append(
                        f"Output is only {duration:.1f}s — suspiciously short"
                    )

                # Check target duration from edit_decisions
                target_dur = None
                if edit_decisions:
                    target_dur = (
                        edit_decisions.get("total_duration_seconds")
                        or edit_decisions.get("metadata", {}).get("target_duration_seconds")
                    )
                if target_dur and target_dur > 0:
                    drift_pct = abs(duration - target_dur) / target_dur
                    if drift_pct > 0.25:
                        technical_probe["issues"].append(
                            f"Duration drift: rendered {duration:.1f}s vs target {target_dur}s "
                            f"({drift_pct:.0%} off). Review pacing or trim."
                        )
                    technical_probe["target_duration"] = target_dur
                    technical_probe["duration_drift_pct"] = round(drift_pct * 100, 1)
                if width < 320 or height < 240:
                    technical_probe["issues"].append(
                        f"Resolution {width}x{height} is very low"
                    )
                if not audio_stream:
                    technical_probe["issues"].append("No audio stream in output")
            else:
                technical_probe["issues"].append(
                    f"ffprobe failed with exit code {proc.returncode}"
                )
        except FileNotFoundError:
            technical_probe["issues"].append("ffprobe not found — cannot validate output")
        except Exception as e:
            technical_probe["issues"].append(f"ffprobe error: {e}")

        issues.extend(technical_probe.get("issues", []))

        # --- 2. Visual spotcheck: sample 4 frames ---
        visual_spotcheck: dict[str, Any] = {
            "frames_sampled": 0,
            "frame_paths": [],
            "black_frames_detected": False,
            "broken_overlays": False,
            "missing_assets": False,
            "unreadable_text": False,
            "issues": [],
        }
        duration = technical_probe.get("duration_seconds", 0)
        if duration > 0 and technical_probe.get("valid_container"):
            try:
                frame_dir = output_path.parent / ".final_review_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                # Sample at 10%, 35%, 65%, 90% of duration
                sample_points = [0.10, 0.35, 0.65, 0.90]
                frame_paths = []
                for i, pct in enumerate(sample_points):
                    ts = round(duration * pct, 2)
                    frame_path = frame_dir / f"review_frame_{i}.png"
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(ts),
                        "-i", str(output_path),
                        "-frames:v", "1", "-q:v", "2",
                        str(frame_path),
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=15)
                    if frame_path.exists():
                        frame_paths.append(str(frame_path))

                        # Check for black frames (file size heuristic:
                        # a 1920x1080 PNG of pure black is ~5KB)
                        if frame_path.stat().st_size < 2000:
                            visual_spotcheck["black_frames_detected"] = True

                visual_spotcheck["frames_sampled"] = len(frame_paths)
                visual_spotcheck["frame_paths"] = frame_paths

                if len(frame_paths) < 4:
                    visual_spotcheck["issues"].append(
                        f"Only {len(frame_paths)}/4 frames extracted — some timestamps may be out of range"
                    )
                if visual_spotcheck["black_frames_detected"]:
                    visual_spotcheck["issues"].append(
                        "Black frame detected — possible missing asset or failed render segment"
                    )
            except Exception as e:
                visual_spotcheck["issues"].append(f"Frame sampling error: {e}")

        issues.extend(visual_spotcheck.get("issues", []))

        # --- 3. Audio spotcheck ---
        audio_spotcheck: dict[str, Any] = {
            "narration_present": False,
            "music_present": False,
            "unexpected_silence": False,
            "clipping_detected": False,
            "mix_intelligible": True,
            "issues": [],
        }
        if technical_probe.get("has_audio") and duration > 0:
            try:
                # Use ffmpeg volumedetect to check audio levels
                cmd = [
                    "ffmpeg", "-i", str(output_path),
                    "-af", "volumedetect", "-f", "null", "-",
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                stderr = proc.stderr or ""
                # Parse mean_volume and max_volume
                mean_vol = None
                max_vol = None
                for line in stderr.split("\n"):
                    if "mean_volume:" in line:
                        try:
                            mean_vol = float(line.split("mean_volume:")[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                    if "max_volume:" in line:
                        try:
                            max_vol = float(line.split("max_volume:")[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass

                if mean_vol is not None:
                    if mean_vol < -60:
                        audio_spotcheck["unexpected_silence"] = True
                        audio_spotcheck["issues"].append(
                            f"Mean volume {mean_vol:.1f} dB — effectively silent"
                        )
                    # Assume narration present if mean volume is reasonable
                    if mean_vol > -40:
                        audio_spotcheck["narration_present"] = True
                    # Assume music present if audio exists (conservative)
                    if mean_vol > -50:
                        audio_spotcheck["music_present"] = True

                if max_vol is not None and max_vol > -0.5:
                    audio_spotcheck["clipping_detected"] = True
                    audio_spotcheck["issues"].append(
                        f"Max volume {max_vol:.1f} dB — possible clipping"
                    )
            except Exception as e:
                audio_spotcheck["issues"].append(f"Audio analysis error: {e}")

        issues.extend(audio_spotcheck.get("issues", []))

        # --- 4. Promise preservation ---
        promise_preservation: dict[str, Any] = {
            "delivery_promise_honored": True,
            "silent_downgrade_detected": False,
            "runtime_swap_detected": False,
            "issues": [],
        }
        if edit_decisions:
            renderer_family = edit_decisions.get("renderer_family", "")
            promise_preservation["renderer_family_used"] = renderer_family

            # Runtime governance — record what actually ran and flag a swap.
            # Three sources of truth, in priority order:
            #   1. proposal_packet.production_plan.render_runtime (authoritative)
            #   2. edit_decisions.metadata.proposal_render_runtime (if edit stage
            #      explicitly copied it to opt into in-tool swap detection)
            #   3. edit_decisions.render_runtime itself (cannot detect a swap in
            #      this case — reviewer does cross-artifact comparison instead)
            render_runtime_edit = (edit_decisions.get("render_runtime") or "").strip().lower()
            if render_runtime_edit:
                promise_preservation["render_runtime_used"] = render_runtime_edit

                proposal_runtime: str | None = None
                runtime_source: str | None = None
                if proposal_packet:
                    pp_runtime = (
                        (proposal_packet.get("production_plan") or {}).get("render_runtime")
                        or ""
                    ).strip().lower()
                    if pp_runtime:
                        proposal_runtime = pp_runtime
                        runtime_source = "proposal_packet.production_plan.render_runtime"
                if proposal_runtime is None:
                    md_runtime = (
                        (edit_decisions.get("metadata") or {}).get("proposal_render_runtime")
                        or ""
                    ).strip().lower()
                    if md_runtime:
                        proposal_runtime = md_runtime
                        runtime_source = "edit_decisions.metadata.proposal_render_runtime"

                if proposal_runtime is None:
                    promise_preservation["runtime_swap_check"] = (
                        "skipped — no proposal_packet or proposal_render_runtime "
                        "metadata provided. Reviewer skill does cross-artifact "
                        "comparison separately."
                    )
                elif proposal_runtime != render_runtime_edit:
                    promise_preservation["runtime_swap_detected"] = True
                    promise_preservation["runtime_swap_check"] = (
                        f"detected — source: {runtime_source}"
                    )
                    promise_preservation["issues"].append(
                        f"render_runtime changed between proposal ({proposal_runtime}) "
                        f"and compose ({render_runtime_edit}) — this is a contract "
                        f"violation unless a render_runtime_selection decision was logged."
                    )
                else:
                    promise_preservation["runtime_swap_check"] = (
                        f"ok — proposal and edit agree ({runtime_source})"
                    )

            delivery_data = (
                edit_decisions.get("metadata", {}).get("delivery_promise")
                or edit_decisions.get("delivery_promise")
            )
            if delivery_data:
                try:
                    from lib.delivery_promise import DeliveryPromise
                    promise = DeliveryPromise.from_dict(delivery_data)
                    cuts = edit_decisions.get("cuts", [])
                    result = promise.validate_cuts(cuts)
                    motion_ratio = result.get("motion_ratio", 0)
                    promise_preservation["motion_ratio_actual"] = round(motion_ratio, 3)

                    if not result["valid"]:
                        promise_preservation["delivery_promise_honored"] = False
                        for v in result["violations"]:
                            promise_preservation["issues"].append(v)

                    # Detect silent downgrade: motion-led promise but <50% motion
                    if (delivery_data.get("type") == "motion_led"
                            and motion_ratio < 0.5):
                        promise_preservation["silent_downgrade_detected"] = True
                        promise_preservation["issues"].append(
                            f"Motion-led promise but only {motion_ratio:.0%} motion — "
                            f"silent downgrade to still-led"
                        )
                except Exception as e:
                    promise_preservation["issues"].append(
                        f"Could not validate delivery promise: {e}"
                    )

        issues.extend(promise_preservation.get("issues", []))

        # --- 5. Subtitle check ---
        subtitle_check: dict[str, Any] = {
            "subtitles_expected": False,
            "subtitles_present": False,
            "issues": [],
        }
        if edit_decisions:
            ed_subs = edit_decisions.get("subtitles", {})
            subtitle_check["subtitles_expected"] = bool(ed_subs.get("enabled"))

            # Check if output has subtitle stream
            if technical_probe.get("valid_container"):
                try:
                    cmd = [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_streams", "-select_streams", "s",
                        str(output_path),
                    ]
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=15
                    )
                    if proc.returncode == 0:
                        sub_data = json.loads(proc.stdout)
                        sub_streams = sub_data.get("streams", [])
                        subtitle_check["subtitles_present"] = len(sub_streams) > 0

                    # If subtitles were expected but not found as a stream,
                    # they may be burned in (which is fine — not a failure)
                    if (subtitle_check["subtitles_expected"]
                            and not subtitle_check["subtitles_present"]):
                        # Check if subtitle_path was used (burned in)
                        sub_source = ed_subs.get("source")
                        if sub_source and Path(sub_source).exists():
                            # Burned-in subtitles are not detectable as streams
                            subtitle_check["subtitles_present"] = True
                            subtitle_check["coverage_ratio"] = 1.0
                        else:
                            subtitle_check["issues"].append(
                                "Subtitles expected but not found in output and "
                                "no subtitle source file exists for burn-in"
                            )
                except Exception as e:
                    subtitle_check["issues"].append(f"Subtitle check error: {e}")

        issues.extend(subtitle_check.get("issues", []))

        # --- 6. Transcript-vs-script comparison ---
        # Catches content-level TTS failures (the classic "Chirp reads `...`
        # as the word 'dot'" trap) that volume-based audio checks miss.
        # Only runs when caller provides both the transcript and script; when
        # skipped, issues list records that so the silence is visible.
        transcript_comparison = self._compare_transcript_to_script(
            Path(narration_transcript_path) if narration_transcript_path else None,
            script_text,
        )
        issues.extend(transcript_comparison.get("issues", []))

        # --- 7. Determine overall status ---
        critical_issues = [
            i for i in issues
            if any(kw in i.lower() for kw in [
                "silent downgrade", "delivery promise violation",
                "effectively silent", "ffprobe failed", "suspiciously short",
                "tts punctuation leak",  # reading literal punctuation aloud
            ])
        ]

        if critical_issues:
            status = "revise"
            recommended_action = "re_render"
        elif issues:
            status = "pass"
            recommended_action = "present_to_user"
        else:
            status = "pass"
            recommended_action = "present_to_user"

        if not technical_probe.get("valid_container"):
            status = "fail"
            recommended_action = "re_render"

        final_review = {
            "version": "1.0",
            "output_path": str(output_path),
            "status": status,
            "checks": {
                "technical_probe": technical_probe,
                "visual_spotcheck": visual_spotcheck,
                "audio_spotcheck": audio_spotcheck,
                "promise_preservation": promise_preservation,
                "subtitle_check": subtitle_check,
                "transcript_comparison": transcript_comparison,
            },
            "issues_found": issues,
            "recommended_action": recommended_action,
        }

        log.info(
            "Final review: status=%s, issues=%d, action=%s",
            status, len(issues), recommended_action,
        )

        return final_review
