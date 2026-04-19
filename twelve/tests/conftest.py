"""Shared pytest fixtures for twelve/tests.

Key isolation: every test runs with an empty reigen_meta_knowledge cache and
a tmp-file redirect for _MK_PATH. Production reigen_meta_knowledge.json is
NEVER written to from test runs.

This preserves determinism: test ordering cannot leak learned self-params
between tests. Uses autouse so no test needs to opt in explicitly.
"""
import os
import sys
import pytest


# Ensure twelve package is importable from tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def _reset_reigen_meta_knowledge(tmp_path, monkeypatch):
    """Redirect _MK_PATH to a fresh tmp file and clear _MK cache for each test.

    After the test: tmp_path is cleaned up by pytest. Production JSON untouched.
    """
    # Import lazily so tests that don't use reigen don't need it on path
    try:
        import twelve.agent.reigen as reigen_mod
    except ImportError:
        return

    # Each test gets its own tmp meta_knowledge file
    tmp_mk = tmp_path / "test_reigen_meta_knowledge.json"
    monkeypatch.setattr(reigen_mod, "_MK_PATH", str(tmp_mk))
    monkeypatch.setattr(reigen_mod, "_MK", {})
    yield
    # monkeypatch teardown restores originals automatically
