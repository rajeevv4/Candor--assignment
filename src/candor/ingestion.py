"""Load every source in data/ into citable MemoryUnits with delivery times, edits, deletions and links."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from candor.models import MemoryUnit

WEEKDAYS = {"MO": "Monday", "TU": "Tuesday", "WE": "Wednesday", "TH": "Thursday",
            "FR": "Friday", "SA": "Saturday", "SU": "Sunday"}


def parse_datetime(s: Any) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def split_address(s: str) -> Tuple[str, str]:
    """'Sarah Patel <sarah.patel@x.com>' -> ('Sarah Patel', 'sarah.patel@x.com')."""
    m = re.match(r"\s*(.*?)\s*<([^>]+)>\s*$", s or "")
    if m:
        return m.group(1).strip('" '), m.group(2).strip().lower()
    return "", (s or "").strip().lower()


def human_time(dt: datetime) -> str:
    return dt.strftime("%A %B %-d %Y %-I:%M %p").replace(":00 ", " ")


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


class DataIngestion:
    """Parses all connectors and native captures into one list of units."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.units: List[MemoryUnit] = []
        self.units_by_id: Dict[str, MemoryUnit] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.deleted: Dict[str, datetime] = {}
        self.edits: Dict[str, List[Tuple[datetime, str]]] = {}
        self.owner_name = "Me"
        self.load_all()

    # ------------------------------------------------------------------ loading
    def load_all(self) -> None:
        self._load_slack_metadata()
        self._load_meetings()
        self._load_dictations()
        self._load_slack_messages()
        self._load_gmail_messages()
        self._load_calendar_events()
        self._load_codex_sessions()
        self._load_chatgpt_conversations()

        for target_id, t in self.deleted.items():
            if target_id in self.units_by_id:
                self.units_by_id[target_id].deletion_time = t
        for target_id, edit_list in self.edits.items():
            if target_id in self.units_by_id:
                self.units_by_id[target_id].edits.extend(sorted(edit_list))

        self._link_dictations_to_outputs()
        self._link_calendar_notifications()
        self.units.sort(key=lambda u: u.timestamp)

    def _add(self, unit: MemoryUnit) -> None:
        self.units.append(unit)
        self.units_by_id[unit.id] = unit

    def _link(self, a: str, b: str) -> None:
        if a in self.units_by_id and b in self.units_by_id and a != b:
            if b not in self.units_by_id[a].links:
                self.units_by_id[a].links.append(b)
            if a not in self.units_by_id[b].links:
                self.units_by_id[b].links.append(a)

    def _jsonl(self, rel: str) -> List[Dict[str, Any]]:
        p = self.data_dir / rel
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_slack_metadata(self) -> None:
        f = self.data_dir / "connectors/slack/users.json"
        if f.exists():
            self.users = {u["id"]: u for u in json.loads(f.read_text(encoding="utf-8"))}
        f = self.data_dir / "connectors/slack/channels.json"
        if f.exists():
            self.channels = {c["id"]: c for c in json.loads(f.read_text(encoding="utf-8"))}
        # the memory's owner (who dictates, codes, asks ChatGPT) is the one member of every DM
        dms = [set(c.get("members", [])) for c in self.channels.values() if c.get("is_dm")]
        if dms:
            owner = set.intersection(*dms)
            if len(owner) == 1:
                self.owner_name = self.users.get(owner.pop(), {}).get("real_name") or self.owner_name

    def _load_meetings(self) -> None:
        d = self.data_dir / "native/meetings"
        if not d.exists():
            return
        for f in sorted(d.glob("*.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            start = parse_datetime(m["start"])
            title = m.get("title", "")
            for s in m["segments"]:
                speaker = s.get("speaker_name") or s.get("speaker_label") or "Unknown speaker"
                self._add(MemoryUnit(
                    id=s["seg_id"],
                    record_id=m["id"],
                    source_type="meeting",
                    timestamp=start + timedelta(seconds=s["end_s"]),
                    title_or_context=title,
                    author_name=speaker,
                    recipients=m.get("participants_known", []),
                    raw_text=s["text"],
                    text=f"[{title}, {m['start'][:10]}] {speaker}: {s['text']}",
                    metadata={
                        "meeting_id": m["id"],
                        "meeting_start": m["start"],
                        "speaker_identified": bool(s.get("speaker_name")),
                        "speaker_confidence": s.get("speaker_confidence"),
                        "calendar_event_id": m.get("calendar_event_id"),
                        "location": m.get("location"),
                    },
                ))

    def _load_dictations(self) -> None:
        for x in self._jsonl("native/dictation/dictations.jsonl"):
            clean = x.get("cleaned_text", "")
            raw = x.get("raw_transcript", "")
            raw_part = f"\n(raw transcript: {raw})" if raw else ""
            self._add(MemoryUnit(
                id=x["id"],
                record_id=x["id"],
                source_type="dictation",
                timestamp=parse_datetime(x["timestamp"]),
                title_or_context=f"Dictation into {x.get('target_app', '')} ({x.get('target_context', '')})",
                author_name=self.owner_name,
                raw_text=clean,
                text=(f"[Dictation {x.get('mode')} into {x.get('target_app')} – {x.get('target_context')}, "
                      f"{x.get('delivery_state')}] {clean}{raw_part}"),
                metadata=x,
            ))

    def _slack_name(self, uid: str | None, bot: str | None = None) -> str:
        return self.users.get(uid or "", {}).get("real_name") or bot or uid or "Unknown"

    def _load_slack_messages(self) -> None:
        for x in self._jsonl("connectors/slack/messages.jsonl"):
            t = parse_datetime(x["ts"])
            ch = self.channels.get(x["channel_id"], {})
            where = ch.get("name", x["channel_id"])
            sub = x.get("subtype")
            base = dict(record_id=x["id"], source_type="slack", timestamp=t,
                        title_or_context=f"Slack #{where}", metadata=x)
            if sub == "message_deleted":
                self.deleted[x["target_id"]] = t
                txt = f"[Slack {where}] (message {x['target_id']} was deleted)"
                self._add(MemoryUnit(id=x["id"], raw_text="", text=txt, **base))
                continue
            if sub == "message_changed":
                self.edits.setdefault(x["target_id"], []).append((t, x["text"]))
                target = self.units_by_id.get(x["target_id"])
                author = target.author_name if target else self._slack_name(x.get("user"))
                self._add(MemoryUnit(id=x["id"], author_name=author, raw_text=x["text"],
                                     text=f"[Slack {where}, edit of {x['target_id']}] {x['text']}",
                                     links=[x["target_id"]], **base))
                self._link(x["id"], x["target_id"])
                continue
            who = self._slack_name(x.get("user"), x.get("bot_name"))
            self._add(MemoryUnit(id=x["id"], author_name=who, author_id=x.get("user"),
                                 recipients=ch.get("members", []), raw_text=x.get("text", ""),
                                 text=f"[Slack {where}] {who}: {x.get('text', '')}", **base))
            if x.get("thread_parent_id"):
                self._link(x["id"], x["thread_parent_id"])

    def _load_gmail_messages(self) -> None:
        threads: Dict[str, List[str]] = {}
        for x in self._jsonl("connectors/gmail/messages.jsonl"):
            sender_name, sender_email = split_address(x.get("from", ""))
            to, cc = x.get("to", []), x.get("cc", [])
            cc_str = f" Cc {', '.join(cc)}" if cc else ""
            self._add(MemoryUnit(
                id=x["id"],
                record_id=x["id"],
                source_type="email",
                timestamp=parse_datetime(x["date"]),
                title_or_context=f"Email: {x.get('subject', '')}",
                author_name=sender_name or sender_email,
                author_id=sender_email,
                recipients=to + cc,
                raw_text=x.get("body", ""),
                text=f"[Email {x['date'][:16]}] From {x.get('from')} To {', '.join(to)}{cc_str} | {x.get('subject', '')}\n{x.get('body', '')}",
                metadata=x,
            ))
            threads.setdefault(x.get("thread_id") or x["id"], []).append(x["id"])
        for ids in threads.values():
            for a in ids:
                for b in ids:
                    self._link(a, b)

    def _load_calendar_events(self) -> None:
        for x in self._jsonl("connectors/google_calendar/events.jsonl"):
            st, en = x.get("start", {}), x.get("end", {})
            when = f"{st.get('dateTime') or st.get('date')} to {en.get('dateTime') or en.get('date')}"
            if st.get("dateTime"):
                readable = f"{human_time(parse_datetime(st['dateTime']))} – {parse_datetime(en['dateTime']).strftime('%-I:%M %p')}"
            else:
                readable = f"all day {datetime.fromisoformat(st['date']).strftime('%A %B %-d %Y')}"
            rep = ""
            for rule in x.get("recurrence") or []:
                days = re.search(r"BYDAY=([A-Z,]+)", rule)
                if days:
                    rep = " | repeats every " + " ".join(WEEKDAYS.get(d, d) for d in days.group(1).split(","))
            attendees = [a.get("email") for a in x.get("attendees", []) if isinstance(a, dict)]
            self._add(MemoryUnit(
                id=x["id"],
                record_id=x["id"],
                source_type="calendar",
                timestamp=parse_datetime(x["updated"]),
                title_or_context=f"Calendar: {x.get('summary', '')}",
                author_name=x.get("organizer", ""),
                author_id=x.get("organizer", ""),
                recipients=attendees,
                raw_text=f"{x.get('summary', '')}\n{x.get('description') or ''}",
                text=(f"[Calendar, {x.get('status')}] {x.get('summary')} | {when} | {readable}{rep} | "
                      f"{x.get('location') or ''} | attendees: {', '.join(attendees)} | {x.get('description') or ''}"),
                metadata=x,
            ))

    def _load_codex_sessions(self) -> None:
        d = self.data_dir / "connectors/codex/sessions"
        if not d.exists():
            return
        for f in sorted(d.glob("*.jsonl")):
            events = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not events:
                continue
            meta, body = events[0], events[1:]
            turns = []
            for e in body:
                who = e.get("role") or e.get("tool") or e.get("type", "")
                content = e.get("content") or e.get("input") or ""
                out = f"\nOutput: {e['output']}" if e.get("output") else ""
                turns.append(f"{who}: {content}{out}")
            joined = "\n".join(turns)
            self._add(MemoryUnit(
                id=meta["id"],
                record_id=meta["id"],
                source_type="codex",
                timestamp=parse_datetime(body[-1]["timestamp"] if body else meta["started_at"]),
                title_or_context=f"Codex session: {meta.get('repo', '')}",
                author_name=self.owner_name,
                raw_text=joined,
                text=f"[Codex session, repo {meta.get('repo')}]\n{joined}",
                metadata={**meta, "turns": turns},
            ))

    def _load_chatgpt_conversations(self) -> None:
        f = self.data_dir / "connectors/chatgpt/conversations.json"
        if not f.exists():
            return
        for c in json.loads(f.read_text(encoding="utf-8")):
            title = c.get("title", "")
            for m in c.get("messages", []):
                role = m.get("role", "")
                self._add(MemoryUnit(
                    id=m["id"],
                    record_id=c["id"],
                    source_type="chatgpt",
                    timestamp=parse_datetime(m["create_time"]),
                    title_or_context=f"ChatGPT: {title}",
                    author_name=self.owner_name if role == "user" else "ChatGPT",
                    raw_text=m.get("content", ""),
                    text=f"[ChatGPT '{title}'] {role}: {m.get('content', '')}",
                    metadata={"conversation_id": c["id"], "title": title, "role": role},
                ))

    # ------------------------------------------------------------------ links
    def _link_dictations_to_outputs(self) -> None:
        """A dictation that was sent into Slack/Gmail is the same content as the message it produced."""
        outputs = [u for u in self.units if u.source_type in ("slack", "email")]
        for d in self.units:
            if d.source_type != "dictation" or not d.raw_text:
                continue
            key = _norm(d.raw_text)[:60]
            for o in outputs:
                if abs((o.timestamp - d.timestamp).total_seconds()) <= 15 * 60 and key and key in _norm(o.raw_text):
                    self._link(d.id, o.id)

    def _link_calendar_notifications(self) -> None:
        """Calendar invitation / update emails describe an event's earlier and current state."""
        events = {_norm(u.metadata.get("summary", "")): u.id for u in self.units if u.source_type == "calendar"}
        for u in self.units:
            if u.source_type != "email":
                continue
            m = re.match(r"^(?:updated invitation|invitation|accepted|declined|canceled|cancelled)[^:]*:\s*(.+?)\s*@", u.metadata.get("subject", ""), re.I)
            if m and _norm(m.group(1)) in events:
                self._link(u.id, events[_norm(m.group(1))])
