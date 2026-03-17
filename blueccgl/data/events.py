"""
Mega-event window definitions and timestamp utilities.

Event windows (±3 days around each event; E3 uses full ceasefire period):
  E1  Oct  4–10  2023   Hamas attack / Israel declares war
  E2  Oct 14–20  2023   Al-Ahli hospital explosion
  E3  Nov 21–Dec 3 2023 Qatar ceasefire & hostage exchanges
  E4  Jan 23–29  2024   ICJ genocide ruling
  E5  Feb 22–28  2024   Aaron Bushnell self-immolation
  E6  Feb 26–Mar 3 2024 Flour massacre + 30k death toll milestone
"""

from __future__ import annotations

from datetime import datetime

# Full conflict analysis window
CONFLICT_START = datetime(2023, 10, 1)
CONFLICT_END   = datetime(2024, 3, 18, 23, 59)

# (start_inclusive, end_inclusive) — both at day precision (end is 23:59)
MEGA_EVENTS: dict[str, tuple[datetime, datetime]] = {
    "E1_Oct7_Attack":     (datetime(2023, 10,  4),        datetime(2023, 10, 10, 23, 59)),
    "E2_AlAhli_Hospital": (datetime(2023, 10, 14),        datetime(2023, 10, 20, 23, 59)),
    "E3_Ceasefire":       (datetime(2023, 11, 21),        datetime(2023, 12,  3, 23, 59)),
    "E4_ICJ_Ruling":      (datetime(2024,  1, 23),        datetime(2024,  1, 29, 23, 59)),
    "E5_Bushnell":        (datetime(2024,  2, 22),        datetime(2024,  2, 28, 23, 59)),
    "E6_Flour_Massacre":  (datetime(2024,  2, 26),        datetime(2024,  3,  3, 23, 59)),
}

# E5 and E6 windows overlap (Feb 26–28). A post in the overlap is assigned to E5
# first (ordered priority below matches key order above).
EVENT_ORDER: list[str] = list(MEGA_EVENTS.keys())


def parse_date(raw: int | str | None) -> datetime | None:
    """
    Parse the dataset's integer date format: 202310071230 → datetime(2023,10,7,12,30).
    Returns None on any parse failure.
    """
    if raw is None:
        return None
    s = str(int(raw))
    # Pad to 12 digits if needed (e.g. midnight may be stored as 10-digit YYYYMMDDMM)
    if len(s) == 10:
        s = s + "00"  # assume HH=s[-4:-2], MM=s[-2:] → already OK; just length 10 = YYYYMMDDHH
    if len(s) == 12:
        try:
            return datetime.strptime(s, "%Y%m%d%H%M")
        except ValueError:
            pass
    # Try 8-digit YYYYMMDD (no time component)
    if len(s) == 8:
        try:
            return datetime.strptime(s, "%Y%m%d")
        except ValueError:
            pass
    return None


def in_conflict_window(dt: datetime) -> bool:
    return CONFLICT_START <= dt <= CONFLICT_END


def assign_event(dt: datetime | None) -> str | None:
    """
    Returns the event label for a datetime, or "baseline" if in conflict window
    but outside all event windows, or None if outside the conflict window entirely.
    """
    if dt is None or not in_conflict_window(dt):
        return None
    for name in EVENT_ORDER:
        start, end = MEGA_EVENTS[name]
        if start <= dt <= end:
            return name
    return "baseline"
