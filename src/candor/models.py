"""Canonical data models for Candor memory and action system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class MemoryUnit:
    """A single citable atom of information in the memory system."""
    id: str  # Unique citable unit id (e.g., MTG-0909-ACME#0001, EM-F-001, SL-F-0001, CAL-BOARD)
    record_id: str  # Container record id (e.g. MTG-0909-ACME, CGPT-0913-BOARD, or unit id)
    source_type: str  # meeting, dictation, slack, email, calendar, codex, chatgpt
    timestamp: datetime  # When this unit became available / delivered
    title_or_context: str  # Channel, subject, meeting title, app context, repo name
    author_name: Optional[str] = None  # Resolved human name (e.g. "Alex Rivera", "Sarah Kim")
    author_id: Optional[str] = None  # e.g. "U03SARAHK", "sarah.kim@brightline.example.com"
    recipients: List[str] = field(default_factory=list)  # Attendees, email recipients, channel members
    raw_text: str = ""  # Original raw text
    text: str = ""  # Enriched text with metadata header for retrieval
    entities: List[str] = field(default_factory=list)  # Extracted entities (people, orgs, dates)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Source-specific extra fields
    edits: List[tuple[datetime, str]] = field(default_factory=list)  # [(edit_time, new_text)]
    deletion_time: Optional[datetime] = None  # If deleted, timestamp when deleted

    def is_visible_at(self, as_of: datetime) -> bool:
        """Whether this unit existed and was not deleted at as_of."""
        if self.timestamp > as_of:
            return False
        if self.deletion_time is not None and self.deletion_time <= as_of:
            return False
        return True

    def get_text_at(self, as_of: datetime) -> str:
        """Returns the version of text that existed at as_of."""
        if not self.is_visible_at(as_of):
            return ""
        valid_edits = [new_txt for edit_time, new_txt in self.edits if edit_time <= as_of]
        if valid_edits:
            # Reconstruct with latest edited text
            prefix = self.text.partition(": ")[0]
            return f"{prefix}: {valid_edits[-1]} (edited)"
        return self.text


@dataclass
class MemoryQuery:
    """Input query for memory system."""
    id: str
    question: str
    as_of: str
    category: Optional[str] = None
    storyline: Optional[str] = None


@dataclass
class MemoryAnswer:
    """Output answer from memory system."""
    id: str
    answer: str
    sources: List[str]
    retrieved: List[str]
    abstained: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "answer": self.answer,
            "sources": self.sources,
            "retrieved": self.retrieved,
            "abstained": self.abstained,
        }


@dataclass
class ActionCommand:
    """Input command for action system."""
    id: str
    command: str
    as_of: str


@dataclass
class ActionPrediction:
    """Output structured action prediction."""
    id: str
    actions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "actions": self.actions,
        }
