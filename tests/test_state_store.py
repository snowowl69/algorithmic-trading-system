"""Tests for the State Store — write/read/is_live."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state_store import write_state, read_state, is_live, _STATE_FILE


@pytest.fixture(autouse=True)
def cleanup():
    """Remove state file before and after each test."""
    if _STATE_FILE.exists():
        _STATE_FILE.unlink()
    yield
    if _STATE_FILE.exists():
        _STATE_FILE.unlink()


class TestStateStore:
    def test_write_and_read(self):
        write_state({"equity": 100_000, "positions": []})
        state = read_state()
        assert state is not None
        assert state["equity"] == 100_000
        assert "_updated_at" in state

    def test_read_nonexistent(self):
        assert read_state() is None

    def test_is_live_after_write(self):
        write_state({"test": True})
        assert is_live() is True

    def test_overwrite(self):
        write_state({"version": 1})
        write_state({"version": 2})
        state = read_state()
        assert state["version"] == 2
