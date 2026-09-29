"""Canonical data models for Candor memory and action system."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MemoryUnit:
    """A single citable atom of memory (meeting segment, Slack message, email, ...)."""
    id: str                      # citable id: a meeting segment ('<meeting>#0042'), a message, a ChatGPT turn ('<conv>#m3')
    record_id: str               # container record (meeting / conversation), or the id itself
    source_type: str             # meeting | dictation | slack | email | calendar | codex | chatgpt
    timestamp: datetime          # delivery time: the unit does not exist before this
    title_or_context: str        # channel, subject, meeting title, app context, repo
    author_name: Optional[str] = None
    author_id: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    raw_text: str = ""           # body only
    text: str = ""               # header + body (same shape as the eval harness)
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)          # related unit/record ids (thread, edit target, ...)
    edits: List[Tuple[datetime, str]] = field(default_factory=list)
    deletion_time: Optional[datetime] = None
    history: List[str] = field(default_factory=list)       # superseded versions still valid to search, not to quote

    def is_visible_at(self, as_of: datetime) -> bool:
        if self.timestamp > as_of:
            return False
        if self.deletion_time is not None and self.deletion_time <= as_of:
            return False
        return True

    def at(self, as_of: datetime) -> "MemoryUnit":
        """Point-in-time copy: edits made by `as_of` replace the text, older text moves to history."""
        applied = [txt for t, txt in self.edits if t <= as_of]
        if not applied:
            return self
        prefix = self.text.partition(": ")[0]
        return replace(
            self,
            text=f"{prefix}: {applied[-1]} (edited)",
            raw_text=applied[-1],
            history=[self.raw_text] + applied[:-1],
        )


@dataclass
class MemoryQuery:
    id: str
    question: str
    as_of: str
    category: Optional[str] = None
    storyline: Optional[str] = None


@dataclass
class MemoryAnswer:
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
    id: str
    command: str
    as_of: str


@dataclass
class ActionPrediction:
    id: str
    actions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "actions": self.actions}
