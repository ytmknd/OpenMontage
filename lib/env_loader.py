"""Canonical .env loader for OpenMontage.

This is the single source of truth for .env parsing semantics. Both
tools/base_tool.py (module-level, runs at import) and
tools/tool_registry.py (ToolRegistry.discover()) delegate to load_env()
here rather than maintaining their own copies of the parser.

Deliberate parsing semantics (do not "fix" to standard dotenv behavior
without updating all three call sites' docstrings):
  - Never overrides a variable already present in os.environ.
  - Quoted values ('...' or "...") are taken verbatim between the quotes;
    no further processing (no escape handling, no comment stripping).
  - Unquoted values have inline `#` comments stripped (only when the `#`
    is at the start of the value or preceded by whitespace), then the
    result is stripped of surrounding whitespace.
  - Blank lines and lines starting with `#` are skipped.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_INLINE_COMMENT_RE = re.compile(r"(^|\s)#")


def load_env(project_root: Optional[Path] = None) -> None:
    """Load .env file from project root into os.environ.

    Only sets variables that are not already present in os.environ.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    env_path = Path(project_root) / ".env"
    if not env_path.is_file():
        return

    with open(env_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Quoted value: take the content inside the quotes verbatim.
            if value[:1] in ("'", '"'):
                quote = value[0]
                end = value.find(quote, 1)
                value = value[1:end] if end != -1 else value[1:]
            else:
                # Strip an inline comment ('#' at line start or after
                # whitespace) so "VAR=   # note" yields "" not "# note".
                match = _INLINE_COMMENT_RE.search(value)
                if match:
                    value = value[: match.start()]
                value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable with optional default."""
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    """Get a required environment variable. Raises if missing."""
    value = os.environ.get(key)
    if value is None:
        raise EnvironmentError(f"Required environment variable {key!r} is not set")
    return value
