"""Regression: get_pipeline_stages must fail fast on a broken manifest.

Bug: the previous implementation caught `(FileNotFoundError, Exception)` —
redundant since Exception already covers FileNotFoundError — which meant
*any* error while loading a pipeline manifest (a YAML syntax error, a
jsonschema.ValidationError from a malformed manifest, etc.) was silently
swallowed and replaced with the canonical STAGES fallback. That hides real
authoring bugs. Only a missing manifest (FileNotFoundError, i.e. an unknown
pipeline_type) should fall back; everything else must propagate.
"""
from __future__ import annotations

import pytest

from lib.checkpoint import STAGES, get_pipeline_stages


def test_unknown_pipeline_type_falls_back_to_canonical_stages():
    assert get_pipeline_stages("definitely-not-a-pipeline") == list(STAGES)


def test_other_exceptions_from_manifest_loading_propagate(monkeypatch):
    def _boom(pipeline_type):
        raise ValueError("malformed manifest")

    monkeypatch.setattr("lib.pipeline_loader.load_pipeline", _boom)

    with pytest.raises(ValueError, match="malformed manifest"):
        get_pipeline_stages("some-pipeline")
