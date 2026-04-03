"""
Lightweight in-process event bus.

Modules publish / subscribe to named channels so they remain decoupled.
Thread-safe; callbacks are invoked synchronously in the publisher thread.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List

from config.logging_config import get_logger

log = get_logger(__name__)

Callback = Callable[..., None]


class EventBus:
    """Simple publish / subscribe message bus."""

    _instance: "EventBus | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "EventBus":
        """Singleton – one bus per process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._subscribers: Dict[str, List[Callback]] = defaultdict(list)
            return cls._instance

    # ── public API ────────────────────────────────────────────────────
    def subscribe(self, channel: str, callback: Callback) -> None:
        self._subscribers[channel].append(callback)
        log.debug("Subscribed %s → %s", callback.__qualname__, channel)

    def publish(self, channel: str, payload: Any = None) -> None:
        for cb in self._subscribers.get(channel, []):
            try:
                cb(payload)
            except Exception:
                log.exception("Error in subscriber %s on channel '%s'", cb, channel)

    def clear(self) -> None:
        """Remove all subscriptions (useful in tests)."""
        self._subscribers.clear()


# Convenience singleton
bus = EventBus()
