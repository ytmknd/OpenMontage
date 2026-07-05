"""fal.ai provider pricing lookups, backed by ``pricing.yaml`` at repo root.

Tools call ``get_tool_pricing()`` from their ``estimate_cost()`` and fall
back to their own hardcoded constants whenever this returns ``None`` (file
missing, malformed, or lacking the tool's entry) or the requested rate key
is absent — so deleting ``pricing.yaml`` never changes cost-estimation
behavior, it only removes the ability to update rates without a code change.

``staleness_warning()`` is surfaced by ``tools/tool_registry.py``'s
``provider_menu_summary()`` so agents see a nudge to refresh pricing when
``as_of`` drifts too far into the past.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import yaml

_PRICING_PATH = Path(__file__).resolve().parent.parent / "pricing.yaml"

# Module-level cache. ``_loaded`` distinguishes "not yet attempted" from
# "attempted and the file is missing/malformed" (where ``_cache`` is also
# None) so we don't re-read the file on every call.
_cache: Optional[dict[str, Any]] = None
_loaded: bool = False


def _load() -> Optional[dict[str, Any]]:
    """Load and cache pricing.yaml. Returns None on any failure — never raises."""
    global _cache, _loaded
    if _loaded:
        return _cache

    _loaded = True
    try:
        if not _PRICING_PATH.exists():
            _cache = None
            return None
        with open(_PRICING_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _cache = data if isinstance(data, dict) else None
    except Exception:
        _cache = None
    return _cache


def get_tool_pricing(tool_name: str) -> Optional[dict[str, Any]]:
    """Return the pricing block for ``tool_name`` (e.g. its ``rates`` dict), or None.

    None is returned when pricing.yaml is missing/malformed, when it has no
    ``tools`` mapping, or when ``tool_name`` has no entry — callers should
    treat None as "use my fallback constants".
    """
    data = _load()
    if not data:
        return None
    tools = data.get("tools")
    if not isinstance(tools, dict):
        return None
    entry = tools.get(tool_name)
    return entry if isinstance(entry, dict) else None


def pricing_age_days() -> Optional[int]:
    """Return the age in days of pricing.yaml's ``as_of`` field, or None if unavailable."""
    data = _load()
    if not data:
        return None
    as_of = data.get("as_of")
    if not as_of:
        return None
    try:
        as_of_date = datetime.strptime(str(as_of), "%Y-%m-%d").date()
    except ValueError:
        return None
    return (date.today() - as_of_date).days


def staleness_warning(max_age_days: int = 90) -> Optional[str]:
    """Return a one-line staleness warning naming the file and age, or None if fresh/missing."""
    age = pricing_age_days()
    if age is None or age <= max_age_days:
        return None
    return (
        f"pricing.yaml is {age} days old (as_of exceeds the {max_age_days}-day "
        "freshness window) -- verify fal.ai rates are still current."
    )


def reset_pricing_cache() -> None:
    """Drop the cached pricing data so the next call re-reads pricing.yaml. For tests."""
    global _cache, _loaded
    _cache = None
    _loaded = False
