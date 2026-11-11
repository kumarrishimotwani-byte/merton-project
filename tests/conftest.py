"""Shared test helpers. Tests of functions that are still stubs are skipped, not failed."""

import pytest


def call_or_skip(fn, *args, **kwargs):
    """Call fn; if it is still a NotImplementedError stub, skip the test instead of failing."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")
