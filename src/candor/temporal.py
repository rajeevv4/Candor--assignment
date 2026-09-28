"""Temporal reasoning, visibility filtering, and event timeline engine."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from candor.models import MemoryUnit


class TemporalEngine:
    """Handles temporal visibility, time-travel queries, and history reconstruction."""

    def __init__(self, units: List[MemoryUnit], deleted: Dict[str, datetime], edits: Dict[str, List[Tuple[datetime, str]]]):
        self.all_units = units
        self.deleted = deleted
        self.edits = edits
        self.avail_map: Dict[str, datetime] = {}
        self.record_of: Dict[str, str] = {}
        self._build_availability_index()

    def _build_availability_index(self) -> None:
        """Compute earliest availability for every unit ID and container record ID."""
        for u in self.all_units:
            self.avail_map[u.id] = u.timestamp
            self.record_of[u.id] = u.record_id

        # A container record (whole meeting, conversation) is available as soon as its first unit is
        for u in self.all_units:
            rec = u.record_id
            if rec not in self.avail_map or u.timestamp < self.avail_map[rec]:
                self.avail_map[rec] = u.timestamp

    def is_forbidden(self, cid: str, as_of: datetime) -> Optional[str]:
        """Check if an ID is forbidden (future delivery or deleted)."""
        t = self.avail_map.get(cid)
        if t and t > as_of:
            return "not yet delivered"
        if cid in self.deleted and self.deleted[cid] <= as_of:
            return "deleted"
        return None

    def get_visible_units(self, as_of: datetime) -> List[MemoryUnit]:
        """Returns units strictly visible at as_of with appropriate text versions."""
        visible: List[MemoryUnit] = []
        for u in self.all_units:
            # Check availability
            if u.timestamp > as_of:
                continue
            # Check deletion
            if u.id in self.deleted and self.deleted[u.id] <= as_of:
                continue

            # Determine text at as_of (applying edits if any)
            unit_text = u.text
            if u.id in self.edits:
                valid_edits = [txt for t, txt in self.edits[u.id] if t <= as_of]
                if valid_edits:
                    prefix = u.text.partition(": ")[0]
                    unit_text = f"{prefix}: {valid_edits[-1]} (edited)"

            # Create a point-in-time representation of the unit
            visible_unit = MemoryUnit(
                id=u.id,
                record_id=u.record_id,
                source_type=u.source_type,
                timestamp=u.timestamp,
                title_or_context=u.title_or_context,
                author_name=u.author_name,
                author_id=u.author_id,
                recipients=u.recipients,
                raw_text=u.raw_text,
                text=unit_text,
                entities=u.entities,
                metadata=u.metadata,
                edits=u.edits,
                deletion_time=u.deletion_time,
            )
            visible.append(visible_unit)

        return visible
