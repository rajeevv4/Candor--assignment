"""Small, general date/time parsing helpers (no dependencies).

Used by retrieval (dates mentioned in questions and records) and by the action planner
(due times, meeting times). Times are America/Los_Angeles.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Los_Angeles")

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
            "thursday": 3, "thu": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
            "sunday": 6, "sun": 6}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
            "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
            "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
            "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twenty-second": 22, "twenty-third": 23,
            "twenty-fourth": 24, "twenty-fifth": 25, "twenty-sixth": 26, "twenty-seventh": 27,
            "twenty-eighth": 28, "twenty-ninth": 29, "thirtieth": 30, "thirty-first": 31}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
                "thirty": 30, "forty": 40, "forty-five": 45, "sixty": 60, "ninety": 90, "a": 1, "an": 1, "half": 0.5}

_MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DAY_RE = r"(\d{1,2})(?:st|nd|rd|th)?"


def to_local(dt: datetime) -> datetime:
    return dt.astimezone(LOCAL_TZ) if dt.tzinfo else dt.replace(tzinfo=LOCAL_TZ)


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def explicit_dates(text: str, ref: datetime) -> Set[date]:
    """Calendar dates written out in text: 'Sep 23', 'September 23rd', '9/23', '2026-09-23', 'the 23rd'."""
    t = text.lower()
    year = ref.year
    found: Set[date] = set()
    for m in re.finditer(r"\b(20\d\d)-(\d{2})-(\d{2})", t):
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.add(d)
    for m in re.finditer(rf"\b{_MONTH_RE}\.?\s+{_DAY_RE}\b", t):
        d = _safe_date(year, MONTHS[m.group(1)[:3]], int(m.group(2)))
        if d:
            found.add(d)
    for m in re.finditer(rf"\b{_DAY_RE}\s+(?:of\s+)?{_MONTH_RE}\b", t):
        d = _safe_date(year, MONTHS[m.group(2)[:3]], int(m.group(1)))
        if d:
            found.add(d)
    for m in re.finditer(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/])", t):
        mo, dy = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= dy <= 31:
            d = _safe_date(year, mo, dy)
            if d:
                found.add(d)
    # "the 25th" / "the twenty-third": day of the reference month (or next month if already past)
    for m in re.finditer(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b|\bthe\s+(" + "|".join(sorted(ORDINALS, key=len, reverse=True)) + r")\b", t):
        dy = int(m.group(1)) if m.group(1) else ORDINALS[m.group(2)]
        d = _safe_date(ref.year, ref.month, dy)
        if d:
            found.add(d)
    return found


def relative_dates(text: str, ref: datetime) -> Set[date]:
    """today / tomorrow / yesterday / weekday names, resolved against `ref`."""
    t = text.lower()
    today = to_local(ref).date()
    found: Set[date] = set()
    if re.search(r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b", t):
        found.add(today)
    if re.search(r"\btomorrow\b", t):
        found.add(today + timedelta(days=1))
    if re.search(r"\byesterday\b", t):
        found.add(today - timedelta(days=1))
    for m in re.finditer(r"\b(last|next|this)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t):
        found.add(resolve_weekday(m.group(2), today, m.group(1)))
    return found


def resolve_weekday(name: str, today: date, qualifier: Optional[str] = None, future: bool = True) -> date:
    wd = WEEKDAYS[name]
    if qualifier == "last":
        delta = (today.weekday() - wd) % 7 or 7
        return today - timedelta(days=delta)
    delta = (wd - today.weekday()) % 7
    if qualifier == "next" and delta == 0:
        delta = 7
    if delta == 0 and qualifier is None and future:
        return today
    return today + timedelta(days=delta)


def parse_clock(text: str) -> Optional[time]:
    """First clock time in text: '3pm', '3:30 pm', 'at 2', 'noon', '9am'. Bare hours 1-6 mean pm."""
    t = text.lower()
    if re.search(r"\bnoon\b|\bmidday\b", t):
        return time(12, 0)
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", t)
    if m:
        h, mi, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)[0]
        if mer == "p" and h < 12:
            h += 12
        if mer == "a" and h == 12:
            h = 0
        return time(h % 24, mi)
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 1 <= h <= 6:
            h += 12
        return time(h % 24, mi)
    m = re.search(r"\b(?:at|@|by|from)\s+(\d{1,2})\b(?!\s*(?:st|nd|rd|th|min|minute|hour|day|week|/))", t)
    if m:
        h = int(m.group(1))
        if 1 <= h <= 6:
            h += 12
        if 0 <= h <= 23:
            return time(h, 0)
    if re.search(r"\bmorning\b", t):
        return time(9, 0)
    if re.search(r"\bafternoon\b", t):
        return time(14, 0)
    if re.search(r"\b(?:evening|tonight)\b", t):
        return time(18, 0)
    return None


def parse_duration(text: str) -> Optional[timedelta]:
    """'30 minutes', 'an hour', '1.5 hours', 'half an hour', '45 min'."""
    t = text.lower()
    if re.search(r"\bhalf an? hour\b", t):
        return timedelta(minutes=30)
    m = re.search(r"\b(\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r")\s*(?:-\s*)?(minutes?|mins?|hours?|hrs?|h)\b", t)
    if not m:
        return None
    raw = m.group(1)
    n = float(raw) if re.match(r"[\d.]+$", raw) else float(NUMBER_WORDS[raw])
    return timedelta(hours=n) if m.group(2).startswith("h") else timedelta(minutes=n)


def resolve_day(text: str, ref: datetime) -> Optional[date]:
    """The single day a command refers to, or None. Prefers explicit dates, then relative ones."""
    ref = to_local(ref)
    ex = sorted(explicit_dates(text, ref))
    if ex:
        d = ex[0]
        # "the 5th" said on the 20th means next month
        if d < ref.date() and re.search(r"\bthe\s+\d{1,2}(?:st|nd|rd|th)\b", text.lower()) and not re.search(_MONTH_RE, text.lower()):
            nm = ref.month % 12 + 1
            d = _safe_date(ref.year + (ref.month == 12), nm, d.day) or d
        return d
    rel = relative_dates(text, ref)
    if rel:
        return sorted(rel)[0]
    return None


def occurrences(start: datetime, recurrence: List[str], window: Tuple[date, date]) -> List[date]:
    """Dates of an event (weekly RRULEs expanded) within a window."""
    lo, hi = window
    s = to_local(start).date()
    if not recurrence:
        return [s] if lo <= s <= hi else []
    out: List[date] = []
    rule = " ".join(recurrence)
    days = re.search(r"BYDAY=([A-Z,]+)", rule)
    interval = int((re.search(r"INTERVAL=(\d+)", rule) or [None, 1])[1])
    codes = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
    wds = {codes[c] for c in days.group(1).split(",")} if days else {s.weekday()}
    exdates = {datetime.strptime(x[:8], "%Y%m%d").date() for x in re.findall(r"EXDATE[^:]*:(\d{8})", rule)}
    d = max(s, lo)
    while d <= hi:
        weeks = (d - (s - timedelta(days=s.weekday()))).days // 7
        if d.weekday() in wds and weeks % interval == 0 and d not in exdates:
            out.append(d)
        d += timedelta(days=1)
    return out


def parse_datetime_safe(s: Optional[str]) -> Optional[datetime]:
    """ISO datetime or date string -> aware local datetime (None if missing/invalid)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return to_local(dt)
