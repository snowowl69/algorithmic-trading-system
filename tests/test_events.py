"""Tests for the Event Bus."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.events import EventBus


@pytest.fixture
def fresh_bus():
    """Return a fresh bus (clear singleton state)."""
    b = EventBus()
    b.clear()
    return b


class TestEventBus:
    def test_publish_subscribe(self, fresh_bus):
        received = []
        fresh_bus.subscribe("test", lambda data: received.append(data))
        fresh_bus.publish("test", {"value": 42})
        assert len(received) == 1
        assert received[0]["value"] == 42

    def test_multiple_subscribers(self, fresh_bus):
        results = []
        fresh_bus.subscribe("ch", lambda d: results.append("a"))
        fresh_bus.subscribe("ch", lambda d: results.append("b"))
        fresh_bus.publish("ch")
        assert results == ["a", "b"]

    def test_no_subscribers(self, fresh_bus):
        # Should not raise
        fresh_bus.publish("empty_channel", "data")

    def test_subscriber_exception_doesnt_break_others(self, fresh_bus):
        results = []

        def bad_handler(data):
            raise ValueError("oops")

        fresh_bus.subscribe("ch", bad_handler)
        fresh_bus.subscribe("ch", lambda d: results.append("ok"))
        fresh_bus.publish("ch", None)
        assert results == ["ok"]

    def test_clear(self, fresh_bus):
        received = []
        fresh_bus.subscribe("ch", lambda d: received.append(d))
        fresh_bus.clear()
        fresh_bus.publish("ch", "data")
        assert received == []
