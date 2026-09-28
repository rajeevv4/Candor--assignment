"""Data ingestion and normalization module for all Candor sources."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from candor.models import MemoryUnit


def parse_datetime(s: Any) -> datetime:
    """Parse ISO datetime string with timezone offset into datetime object."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


class DataIngestion:
    """Ingests, parses, normalizes, and connects all mock data sources."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.units: List[MemoryUnit] = []
        self.units_by_id: Dict[str, MemoryUnit] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.deleted: Dict[str, datetime] = {}
        self.edits: Dict[str, List[Tuple[datetime, str]]] = {}
        self.load_all()

    def load_all(self) -> None:
        """Load all data connectors and native records."""
        self._load_slack_metadata()
        self._load_meetings()
        self._load_dictations()
        self._load_slack_messages()
        self._load_gmail_messages()
        self._load_calendar_events()
        self._load_codex_sessions()
        self._load_chatgpt_conversations()

        # Connect edits and deletions to their target units
        for target_id, del_time in self.deleted.items():
            if target_id in self.units_by_id:
                self.units_by_id[target_id].deletion_time = del_time

        for target_id, edit_list in self.edits.items():
            if target_id in self.units_by_id:
                self.units_by_id[target_id].edits.extend(edit_list)

        # Sort all units chronologically by availability timestamp
        self.units.sort(key=lambda u: u.timestamp)

    def _load_slack_metadata(self) -> None:
        users_file = self.data_dir / "connectors/slack/users.json"
        if users_file.exists():
            for u in json.loads(users_file.read_text(encoding="utf-8")):
                self.users[u["id"]] = u

        channels_file = self.data_dir / "connectors/slack/channels.json"
        if channels_file.exists():
            for c in json.loads(channels_file.read_text(encoding="utf-8")):
                self.channels[c["id"]] = c

    def _add_unit(self, unit: MemoryUnit) -> None:
        self.units.append(unit)
        self.units_by_id[unit.id] = unit

    def _load_meetings(self) -> None:
        meetings_dir = self.data_dir / "native/meetings"
        if not meetings_dir.exists():
            return
        for f in sorted(meetings_dir.glob("*.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            start = parse_datetime(m["start"])
            m_id = m["id"]
            title = m.get("title", "")
            participants = m.get("participants_known", [])
            for s in m["segments"]:
                seg_id = s["seg_id"]
                end_s = s["end_s"]
                seg_time = start + timedelta(seconds=end_s)
                speaker = s.get("speaker_name") or s.get("speaker_label") or "Unknown speaker"
                raw_text = s["text"]
                formatted_text = f"[{title}, {m['start'][:10]}] {speaker}: {raw_text}"
                unit = MemoryUnit(
                    id=seg_id,
                    record_id=m_id,
                    source_type="meeting",
                    timestamp=seg_time,
                    title_or_context=title,
                    author_name=speaker,
                    recipients=participants,
                    raw_text=raw_text,
                    text=formatted_text,
                    metadata={
                        "meeting_id": m_id,
                        "channel": s.get("channel"),
                        "start_s": s.get("start_s"),
                        "end_s": end_s,
                        "location": m.get("location"),
                        "calendar_event_id": m.get("calendar_event_id"),
                        "speaker_confidence": s.get("speaker_confidence"),
                    }
                )
                self._add_unit(unit)

    def _load_dictations(self) -> None:
        dict_file = self.data_dir / "native/dictation/dictations.jsonl"
        if not dict_file.exists():
            return
        for line in dict_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            d_id = x["id"]
            d_time = parse_datetime(x["timestamp"])
            raw = x.get("raw_transcript", "")
            clean = x.get("cleaned_text", "")
            mode = x.get("mode", "dictation")
            app = x.get("target_app", "")
            context = x.get("target_context", "")
            state = x.get("delivery_state", "")

            raw_part = f"\n(raw transcript: {raw})" if raw else ""
            formatted_text = f"[Dictation {mode} into {app} – {context}, {state}] {clean}{raw_part}"

            unit = MemoryUnit(
                id=d_id,
                record_id=d_id,
                source_type="dictation",
                timestamp=d_time,
                title_or_context=f"Dictation into {app} ({context})",
                author_name="Alex Rivera",
                author_id="alex@brightline.example.com",
                raw_text=clean,
                text=formatted_text,
                metadata=x
            )
            self._add_unit(unit)

    def _load_slack_messages(self) -> None:
        msg_file = self.data_dir / "connectors/slack/messages.jsonl"
        if not msg_file.exists():
            return
        for line in msg_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            s_id = x["id"]
            t = parse_datetime(x["ts"])
            ch_id = x["channel_id"]
            ch_info = self.channels.get(ch_id, {})
            where = ch_info.get("name", ch_id)

            subtype = x.get("subtype")
            if subtype == "message_deleted":
                target_id = x["target_id"]
                self.deleted[target_id] = t
                formatted_text = f"[Slack {where}] (message {target_id} was deleted)"
                unit = MemoryUnit(
                    id=s_id,
                    record_id=s_id,
                    source_type="slack",
                    timestamp=t,
                    title_or_context=f"Slack #{where}",
                    author_name=self.users.get(x.get("user"), {}).get("real_name", x.get("user")),
                    author_id=x.get("user"),
                    raw_text=formatted_text,
                    text=formatted_text,
                    metadata=x
                )
                self._add_unit(unit)
                continue

            if subtype == "message_changed":
                target_id = x["target_id"]
                new_text = x["text"]
                self.edits.setdefault(target_id, []).append((t, new_text))
                formatted_text = f"[Slack {where}, edit of {target_id}] {new_text}"
                unit = MemoryUnit(
                    id=s_id,
                    record_id=s_id,
                    source_type="slack",
                    timestamp=t,
                    title_or_context=f"Slack #{where}",
                    author_name=self.users.get(x.get("user"), {}).get("real_name", x.get("user")),
                    author_id=x.get("user"),
                    raw_text=new_text,
                    text=formatted_text,
                    metadata=x
                )
                self._add_unit(unit)
                continue

            # Standard message
            user_id = x.get("user")
            user_info = self.users.get(user_id, {})
            who = user_info.get("real_name") or x.get("bot_name") or user_id or "Unknown"
            raw_text = x.get("text", "")
            formatted_text = f"[Slack {where}] {who}: {raw_text}"

            unit = MemoryUnit(
                id=s_id,
                record_id=s_id,
                source_type="slack",
                timestamp=t,
                title_or_context=f"Slack #{where}",
                author_name=who,
                author_id=user_id,
                recipients=ch_info.get("members", []),
                raw_text=raw_text,
                text=formatted_text,
                metadata=x
            )
            self._add_unit(unit)

    def _load_gmail_messages(self) -> None:
        gmail_file = self.data_dir / "connectors/gmail/messages.jsonl"
        if not gmail_file.exists():
            return
        for line in gmail_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            e_id = x["id"]
            t = parse_datetime(x["date"])
            sender = x.get("from", "")
            to_list = x.get("to", [])
            cc_list = x.get("cc", [])
            subject = x.get("subject", "")
            body = x.get("body", "")

            cc_str = f" Cc {', '.join(cc_list)}" if cc_list else ""
            formatted_text = f"[Email {x['date'][:16]}] From {sender} To {', '.join(to_list)}{cc_str} | {subject}\n{body}"

            unit = MemoryUnit(
                id=e_id,
                record_id=e_id,
                source_type="email",
                timestamp=t,
                title_or_context=f"Email: {subject}",
                author_name=sender,
                author_id=sender,
                recipients=to_list + cc_list,
                raw_text=body,
                text=formatted_text,
                metadata=x
            )
            self._add_unit(unit)

    def _load_calendar_events(self) -> None:
        cal_file = self.data_dir / "connectors/google_calendar/events.jsonl"
        if not cal_file.exists():
            return
        for line in cal_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            c_id = x["id"]
            t = parse_datetime(x["updated"])
            summary = x.get("summary", "")
            description = x.get("description", "")
            location = x.get("location", "")
            st = x.get("start", {})
            en = x.get("end", {})
            when = f"{st.get('dateTime') or st.get('date')} to {en.get('dateTime') or en.get('date')}"
            attendees = [a.get("email") for a in x.get("attendees", []) if isinstance(a, dict)]
            att_str = ", ".join(attendees)
            rep_str = f" | repeats {x['recurrence']}" if x.get("recurrence") else ""

            formatted_text = f"[Calendar, {x.get('status')}] {summary} | {when} | {location} | attendees: {att_str} | {description}{rep_str}"

            unit = MemoryUnit(
                id=c_id,
                record_id=c_id,
                source_type="calendar",
                timestamp=t,
                title_or_context=f"Calendar: {summary}",
                author_name=x.get("organizer", ""),
                author_id=x.get("organizer", ""),
                recipients=attendees,
                raw_text=f"{summary}\n{description}",
                text=formatted_text,
                metadata=x
            )
            self._add_unit(unit)

    def _load_codex_sessions(self) -> None:
        codex_dir = self.data_dir / "connectors/codex/sessions"
        if not codex_dir.exists():
            return
        for f in sorted(codex_dir.glob("*.jsonl")):
            lines = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                continue
            meta, body = lines[0], lines[1:]
            s_id = meta["id"]
            t = parse_datetime(body[-1]["timestamp"] if body else meta["started_at"])
            text_lines = []
            for e in body:
                role_or_tool = e.get("role") or e.get("tool") or e.get("type", "")
                content = e.get("content") or e.get("input") or ""
                out = e.get("output", "")
                out_str = f"\nOutput: {out}" if out else ""
                text_lines.append(f"{role_or_tool}: {content}{out_str}")

            joined_text = "\n".join(text_lines)
            repo = meta.get("repo", "")
            formatted_text = f"[Codex session, repo {repo}]\n{joined_text}"

            unit = MemoryUnit(
                id=s_id,
                record_id=s_id,
                source_type="codex",
                timestamp=t,
                title_or_context=f"Codex session: {repo}",
                author_name="Alex Rivera",
                raw_text=joined_text,
                text=formatted_text,
                metadata=meta
            )
            self._add_unit(unit)

    def _load_chatgpt_conversations(self) -> None:
        cgpt_file = self.data_dir / "connectors/chatgpt/conversations.json"
        if not cgpt_file.exists():
            return
        for c in json.loads(cgpt_file.read_text(encoding="utf-8")):
            c_id = c["id"]
            title = c.get("title", "")
            for m in c.get("messages", []):
                m_id = m["id"]
                t = parse_datetime(m["create_time"])
                role = m.get("role", "")
                content = m.get("content", "")
                formatted_text = f"[ChatGPT '{title}'] {role}: {content}"

                unit = MemoryUnit(
                    id=m_id,
                    record_id=c_id,
                    source_type="chatgpt",
                    timestamp=t,
                    title_or_context=f"ChatGPT: {title}",
                    author_name="Alex Rivera" if role == "user" else "ChatGPT",
                    raw_text=content,
                    text=formatted_text,
                    metadata={"conversation_id": c_id, "title": title, "role": role}
                )
                self._add_unit(unit)
