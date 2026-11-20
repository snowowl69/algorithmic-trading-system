"""
Shared State Store — JSON-file-based IPC between trading engine and dashboard.

The trading engine writes snapshots to a JSON file on disk.  The Streamlit
dashboard reads the same file on each refresh.  This avoids the need for
Redis or a database during development while remaining swappable later.

Thread-safe writes via file locking.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger

log = get_logger(__name__)

_STATE_DIR = Path(__file__).resolve().parents[1] / "data"
_STATE_FILE = _STATE_DIR / "live_state.json"
_LOCK = threading.Lock()


def _ensure_dir() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  Writer (called by the trading engine)
# ═══════════════════════════════════════════════════════════════════════

def write_state(state: Dict[str, Any]) -> None:
    """Atomically write the full state dict to disk."""
    _ensure_dir()
    state["_updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = _STATE_FILE.with_suffix(".tmp")
    with _LOCK:
        tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
        tmp.replace(_STATE_FILE)


def publish_snapshot(
    snapshot: Dict[str, Any],
    positions: List[Dict],
    bars: Dict[str, List[Dict]],
    sentiment_feed: List[Dict],
    trade_log: List[Dict],
    equity_curve: List[Dict],
    prev_closes: Dict[str, float] = None,
    dividends: List[Dict] = None,
    options_chains: Dict[str, Any] = None,
    regime_info: Dict[str, Any] = None,
    session_info: Dict[str, Any] = None,
    forming_candles: Dict[str, Any] = None,
    trade_suggestions: List[Dict] = None,
) -> None:
    """Convenience wrapper — packs all dashboard data into one write."""
    write_state({
        "snapshot": snapshot,
        "positions": positions,
        "bars": bars,
        "sentiment_feed": sentiment_feed,
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "prev_closes": prev_closes or {},
        "dividends": dividends or [],
        "options_chains": options_chains or {},
        "regime_info": regime_info or {},
        "session_info": session_info or {},
        "forming_candles": forming_candles or {},
        "trade_suggestions": trade_suggestions or [],
    })


# ═══════════════════════════════════════════════════════════════════════
#  Reader (called by the dashboard)
# ═══════════════════════════════════════════════════════════════════════

def read_state() -> Optional[Dict[str, Any]]:
    """Read the latest state from disk, or return None if no file exists."""
    if not _STATE_FILE.exists():
        return None
    for _attempt in range(2):
        try:
            raw = _STATE_FILE.read_text(encoding="utf-8")
            return json.loads(raw)
        except (json.JSONDecodeError, OSError):
            if _attempt == 0:
                time.sleep(0.05)
    log.warning("Failed to read state file after retry")
    return None


def is_live(staleness_seconds: float = 30.0) -> bool:
    """Return True if the state file was updated within *staleness_seconds*."""
    state = read_state()
    if state is None:
        return False
    try:
        updated = datetime.fromisoformat(state["_updated_at"])
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        return age < staleness_seconds
    except (KeyError, ValueError):
        return False
