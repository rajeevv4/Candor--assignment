"""Point-in-time view of memory: what existed, what was deleted, which edits applied."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from candor.models import MemoryUnit


class TemporalEngine:
    """Answers "what could the user know at `as_of`?".

    A unit exists from its delivery time on. A deleted message is gone from its deletion time on.
    An edit replaces the text from the edit time on (the old text is kept in `history` so it can
    still be found, but it is never the current text).
    """

    def __init__(self, units: List[MemoryUnit], deleted: Dict[str, datetime], edits: Dict[str, List[Tuple[datetime, str]]]):
        self.all_units = units
        self.deleted = deleted
        self.edits = edits
        self.avail_map: Dict[str, datetime] = {}
        for u in units:
            self.avail_map[u.id] = u.timestamp
        # a container record (whole meeting, conversation) exists as soon as its first unit does
        for u in units:
            if u.record_id not in self.avail_map or u.timestamp < self.avail_map[u.record_id]:
                self.avail_map[u.record_id] = u.timestamp

    def is_forbidden(self, cid: str, as_of: datetime) -> Optional[str]:
        t = self.avail_map.get(cid)
        if t and t > as_of:
            return "not yet delivered"
        if cid in self.deleted and self.deleted[cid] <= as_of:
            return "deleted"
        return None

    def get_visible_units(self, as_of: datetime) -> List[MemoryUnit]:
        return [u.at(as_of) for u in self.all_units if u.is_visible_at(as_of)]
