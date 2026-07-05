"""Tests for the canonical .env parser in lib/env_loader.py (D-3).

Covers the deliberate hand-rolled parsing semantics that tools/base_tool.py
and tools/tool_registry.py now delegate to: quoted values taken verbatim,
inline '#' comments stripped only outside quotes, existing os.environ never
overridden, and blank/comment lines skipped.
"""

from __future__ import annotations

import os

from lib.env_loader import load_env


def test_quoted_value_with_trailing_text_is_taken_verbatim(tmp_path, monkeypatch):
    monkeypatch.delenv("QUOTED_KEY", raising=False)
    (tmp_path / ".env").write_text('QUOTED_KEY="hello # not a comment" trailing garbage\n')

    load_env(project_root=tmp_path)

    assert os.environ["QUOTED_KEY"] == "hello # not a comment"


def test_unquoted_value_strips_inline_comment(tmp_path, monkeypatch):
    monkeypatch.delenv("PLAIN_KEY", raising=False)
    (tmp_path / ".env").write_text("PLAIN_KEY=val # note\n")

    load_env(project_root=tmp_path)

    assert os.environ["PLAIN_KEY"] == "val"


def test_existing_environ_value_is_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("EXISTING_KEY", "original")
    (tmp_path / ".env").write_text("EXISTING_KEY=should_not_apply\n")

    load_env(project_root=tmp_path)

    assert os.environ["EXISTING_KEY"] == "original"


def test_blank_and_comment_lines_are_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("AFTER_BLANK", raising=False)
    (tmp_path / ".env").write_text(
        "\n"
        "# full line comment\n"
        "   \n"
        "AFTER_BLANK=value\n"
    )

    load_env(project_root=tmp_path)

    assert os.environ["AFTER_BLANK"] == "value"


def test_missing_env_file_is_a_noop(tmp_path):
    # No .env written in tmp_path.
    load_env(project_root=tmp_path)  # must not raise
