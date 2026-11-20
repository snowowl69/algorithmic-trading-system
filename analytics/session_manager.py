"""
Indian Market Session Manager.

Provides session classification for Indian equity markets (NSE/BSE),
Opening Range Breakout (ORB) tracking, and session-aware trading filters.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Dict, Optional


# IST is UTC+05:30
_IST = timezone(timedelta(hours=5, minutes=30))


class IndianSession(str, Enum):
    """Named sessions of the Indian trading day."""

    PRE_MARKET = "pre_market"
    OPENING = "opening"          # 09:15 - 09:45
    MORNING = "morning"          # 09:45 - 12:00
    LUNCH = "lunch"              # 12:00 - 13:00
    AFTERNOON = "afternoon"      # 13:00 - 14:30
    POWER_HOUR = "power_hour"    # 14:30 - 15:30
    CLOSED = "closed"


# Session boundaries as (start, end) in IST local time
_SESSION_BOUNDS: Dict[IndianSession, tuple] = {
    IndianSession.PRE_MARKET: (time(9, 0), time(9, 15)),
    IndianSession.OPENING:    (time(9, 15), time(9, 45)),
    IndianSession.MORNING:    (time(9, 45), time(12, 0)),
    IndianSession.LUNCH:      (time(12, 0), time(13, 0)),
    IndianSession.AFTERNOON:  (time(13, 0), time(14, 30)),
    IndianSession.POWER_HOUR: (time(14, 30), time(15, 30)),
}

_SESSION_LABELS: Dict[IndianSession, str] = {
    IndianSession.PRE_MARKET: "Pre-Market",
    IndianSession.OPENING:    "Opening Session",
    IndianSession.MORNING:    "Morning Session",
    IndianSession.LUNCH:      "Lunch Hour",
    IndianSession.AFTERNOON:  "Afternoon Session",
    IndianSession.POWER_HOUR: "Power Hour",
    IndianSession.CLOSED:     "Market Closed",
}


# ------------------------------------------------------------------
# public helpers
# ------------------------------------------------------------------

def get_current_session(now: Optional[datetime] = None) -> IndianSession:
    """Return the current :class:`IndianSession` based on IST clock.

    Parameters
    ----------
    now : datetime, optional
        Override the current timestamp (must be timezone-aware or naive IST).
        Defaults to ``datetime.now(IST)``.
    """
    if now is None:
        now = datetime.now(_IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_IST)
    else:
        now = now.astimezone(_IST)

    t = now.time()
    for session, (start, end) in _SESSION_BOUNDS.items():
        if start <= t < end:
            return session
    return IndianSession.CLOSED


def get_session_info(now: Optional[datetime] = None) -> Dict[str, object]:
    """Return a dict with detailed information about the current session.

    Keys
    ----
    session : IndianSession
    label : str
    start_time : str  (HH:MM)
    end_time : str    (HH:MM)
    minutes_remaining : int
    should_avoid_entry : bool  -- True during LUNCH
    is_orb_window : bool       -- True during the first 15 minutes (09:15-09:30)
    """
    if now is None:
        now = datetime.now(_IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_IST)
    else:
        now = now.astimezone(_IST)

    session = get_current_session(now)
    t = now.time()

    if session in _SESSION_BOUNDS:
        start, end = _SESSION_BOUNDS[session]
        start_str = start.strftime("%H:%M")
        end_str = end.strftime("%H:%M")
        # minutes remaining in the current session
        end_dt = now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
        remaining = max(0, int((end_dt - now).total_seconds() // 60))
    else:
        start_str = "--:--"
        end_str = "--:--"
        remaining = 0

    is_orb = (time(9, 15) <= t < time(9, 30))

    return {
        "session": session,
        "label": _SESSION_LABELS.get(session, "Unknown"),
        "start_time": start_str,
        "end_time": end_str,
        "minutes_remaining": remaining,
        "should_avoid_entry": session == IndianSession.LUNCH,
        "is_orb_window": is_orb,
    }


# ------------------------------------------------------------------
# Opening Range Breakout tracker
# ------------------------------------------------------------------

class ORBTracker:
    """Track the Opening Range (09:15 - 09:30 IST) and signal breakouts.

    Usage
    -----
    1. Feed every bar with :meth:`update`.
    2. After 09:30, call :meth:`check_breakout` with the current price to
       see if the ORB range has been broken.

    Attributes
    ----------
    orb_high : float or None
        Highest price seen during the opening range.
    orb_low : float or None
        Lowest price seen during the opening range.
    is_complete : bool
        ``True`` once the 09:30 boundary has passed.
    """

    def __init__(self) -> None:
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.is_complete: bool = False
        self._date: Optional[datetime] = None

    def reset(self) -> None:
        """Reset the tracker for a new trading day."""
        self.orb_high = None
        self.orb_low = None
        self.is_complete = False
        self._date = None

    def update(self, high: float, low: float, timestamp: datetime) -> None:
        """Feed an OHLC bar's high/low and timestamp.

        Only bars between 09:15 and 09:30 IST contribute to the ORB range.
        The first bar after 09:30 marks the range as complete.

        Parameters
        ----------
        high : float
        low : float
        timestamp : datetime
            Timezone-aware timestamp of the bar.
        """
        if timestamp.tzinfo is None:
            ts = timestamp.replace(tzinfo=_IST)
        else:
            ts = timestamp.astimezone(_IST)

        t = ts.time()

        # Auto-reset on a new calendar day
        if self._date is None or self._date.date() != ts.date():
            self.reset()
            self._date = ts

        if self.is_complete:
            return

        if time(9, 15) <= t < time(9, 30):
            if self.orb_high is None or high > self.orb_high:
                self.orb_high = high
            if self.orb_low is None or low < self.orb_low:
                self.orb_low = low
        elif t >= time(9, 30) and self.orb_high is not None:
            self.is_complete = True

    def check_breakout(self, price: float) -> Optional[str]:
        """Check whether *price* breaks the opening range.

        Returns
        -------
        ``"breakout_above"`` if price exceeds the ORB high,
        ``"breakout_below"`` if price falls below the ORB low,
        or ``None`` if the range is not yet complete or no breakout occurred.
        """
        if not self.is_complete:
            return None
        if self.orb_high is not None and price > self.orb_high:
            return "breakout_above"
        if self.orb_low is not None and price < self.orb_low:
            return "breakout_below"
        return None
