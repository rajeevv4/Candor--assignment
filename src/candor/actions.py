"""Text-command planner (dry run): natural-language command -> structured actions.

Nothing is hard-coded about particular commands, people or events. Everything the planner
resolves comes from data/:
  - people: Slack users (ids, emails), Gmail From/To/Cc headers, calendar attendees
  - channels: Slack channels.json (and DM channels, for "the owner" detection)
  - events: calendar events.jsonl (matched by title overlap, next occurrence after `as_of`)
Times are parsed relative to `as_of` in America/Los_Angeles.

Safety rules: destructive commands become `confirm`; commands that name an ambiguous person
(two Sarahs) become `clarify`; questions become `memory.ask`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from candor.ingestion import split_address
from candor.models import ActionCommand, ActionPrediction
from candor.retrieval import terms
from candor.timeparse import (LOCAL_TZ, NUMBER_WORDS, occurrences, parse_clock, parse_datetime_safe,
                              parse_duration, resolve_day, to_local)

ACTION_VERBS = (r"email|e-mail|mail|message|msg|slack|dm|text|tell|ping|post|remind|book|schedule|set up|"
                r"arrange|open|launch|move|reschedule|push|shift|cancel|delete|remove|thank|send|reply|"
                r"create|add|let|ask|invite|forward")
DESTRUCTIVE = re.compile(r"\b(?:delete|remove|erase|wipe|purge|trash|clear out|clear all|unsubscribe|cancel|archive all|mark all)\b", re.I)
QUESTION_START = re.compile(r"^(?:what|what's|whats|when|who|whom|whose|why|where|which|how|is|are|was|were|did|do|does|has|have|had|can|could|should|will|would)\b", re.I)
KNOWN_APPS = ["figma", "slack", "notion", "linear", "gmail", "google calendar", "calendar", "chrome", "safari",
              "zoom", "google meet", "github", "terminal", "vs code", "vscode", "cursor", "spotify", "finder",
              "codex", "chatgpt", "excel", "google docs", "google sheets", "1password", "apple notes", "notes"]


def _title_tokens(text: str) -> List[str]:
    """Event-title tokens without stopword removal ("go/no-go" is all stopwords otherwise)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ("the", "my", "a", "an", "to", "with", "for", "of", "on", "in", "and")]


@dataclass
class Person:
    name: str
    email: Optional[str] = None
    slack_id: Optional[str] = None
    dm_id: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

    @property
    def first(self) -> str:
        return self.name.split()[0]


class Directory:
    """People, channels and calendar events known from the data."""

    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.people: Dict[str, Person] = {}   # key: email or slack id
        self.channels: Dict[str, str] = {}    # normalised name -> channel id
        self.events: List[Dict[str, Any]] = []
        self.owner_ids: set = set()
        users = self._json(d / "connectors/slack/users.json") or []
        channels = self._json(d / "connectors/slack/channels.json") or []

        dms = [c for c in channels if c.get("is_dm")]
        if dms:
            common = set(dms[0].get("members", []))
            for c in dms[1:]:
                common &= set(c.get("members", []))
            self.owner_ids = common
        for u in users:
            if not u.get("email") and not u.get("real_name"):
                continue
            if not u.get("email") and u["id"].startswith("B"):
                continue  # bots
            p = Person(u.get("real_name") or u.get("name"), (u.get("email") or "").lower() or None, u["id"])
            self._add(p)
        for c in channels:
            if c.get("is_dm"):
                others = [m for m in c.get("members", []) if m not in self.owner_ids]
                for p in self.people.values():
                    if others and p.slack_id == others[0]:
                        p.dm_id = c["id"]
            else:
                name = c["name"].lower()
                for variant in {name, name.replace("-", " "), name.replace("-", "")}:
                    self.channels[variant] = c["id"]

        for path in [d / "connectors/gmail/messages.jsonl"]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                x = json.loads(line)
                for addr in [x.get("from", "")] + x.get("to", []) + x.get("cc", []):
                    name, email = split_address(addr)
                    if email and "@" in email and not re.match(r"(no-?reply|notif|news|digest|calendar|hello|events|partners|reminders|notify)", email):
                        self._add(Person(name or self._name_from_email(email), email))

        path = d / "connectors/google_calendar/events.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ev = json.loads(line)
                    self.events.append(ev)
                    for a in ev.get("attendees", []):
                        email = (a.get("email") or "").lower()
                        if "." in email.split("@")[0]:
                            self._add(Person(self._name_from_email(email), email))

    @staticmethod
    def _json(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @staticmethod
    def _name_from_email(email: str) -> str:
        return " ".join(w.capitalize() for w in re.split(r"[._]", email.split("@")[0]) if w)

    def _add(self, p: Person) -> None:
        for existing in self.people.values():
            if (p.email and existing.email == p.email) or (p.slack_id and existing.slack_id == p.slack_id):
                existing.email = existing.email or p.email
                existing.slack_id = existing.slack_id or p.slack_id
                if len(p.name.split()) > len(existing.name.split()):
                    existing.name = p.name
                return
        self.people[p.email or p.slack_id or p.name] = p

    @property
    def owner_first(self) -> str:
        owner = next((p for p in self.people.values() if self.is_owner(p)), None)
        return owner.first if owner else ""

    def is_owner(self, p: Person) -> bool:
        return bool(p.slack_id and p.slack_id in self.owner_ids)

    def find_people(self, text: str) -> List[List[Person]]:
        """People mentioned in text. Each mention -> list of candidates (len > 1 means ambiguous)."""
        t = text.lower()
        mentions: List[List[Person]] = []
        used: List[Tuple[int, int]] = []
        people = [p for p in self.people.values() if not self.is_owner(p)]
        # full names first, then single first names
        for p in sorted(people, key=lambda p: -len(p.name)):
            full = p.name.lower()
            if " " in full:
                for m in re.finditer(rf"\b{re.escape(full)}\b", t):
                    if not any(a <= m.start() < b for a, b in used):
                        used.append((m.start(), m.end()))
                        mentions.append([p])
        by_first: Dict[str, List[Person]] = {}
        for p in people:
            by_first.setdefault(p.first.lower(), []).append(p)
        for first, cands in by_first.items():
            if len(first) < 3:
                continue
            for m in re.finditer(rf"\b{re.escape(first)}\b", t):
                if not any(a <= m.start() < b for a, b in used):
                    used.append((m.start(), m.end()))
                    mentions.append(cands)
        return mentions

    def find_channel(self, text: str) -> Optional[str]:
        t = text.lower()
        m = re.search(r"#([\w-]+)", t)
        if m and m.group(1) in self.channels:
            return self.channels[m.group(1)]
        for name in sorted(self.channels, key=len, reverse=True):
            if re.search(rf"\b(?:the\s+)?{re.escape(name)}\s+(?:channel|room)\b|\b(?:in|to|on)\s+(?:the\s+)?#?{re.escape(name)}\b(?!\s+\w+\s+(?:is|are))", t):
                return self.channels[name]
        return None

    def find_event(self, phrase: str, as_of: datetime) -> Optional[Tuple[Dict[str, Any], datetime, datetime]]:
        """Best-matching calendar event for a phrase, with its next start/end at or after as_of."""
        want = set(terms(phrase)) or set(_title_tokens(phrase))
        if not want:
            return None
        best, best_key = None, None
        for ev in self.events:
            if ev.get("status") == "cancelled":
                continue
            upd = parse_datetime_safe(ev.get("updated"))
            if upd and upd > to_local(as_of):
                continue  # this version of the event didn't exist yet
            have = set(terms(ev.get("summary", ""))) | set(_title_tokens(ev.get("summary", "")))
            hit = want & have
            if not hit:
                continue
            start = parse_datetime_safe((ev.get("start") or {}).get("dateTime") or (ev.get("start") or {}).get("date"))
            end = parse_datetime_safe((ev.get("end") or {}).get("dateTime") or (ev.get("end") or {}).get("date"))
            if not start or not end:
                continue
            if ev.get("recurrence"):
                days = occurrences(start, ev["recurrence"], (to_local(as_of).date(), to_local(as_of).date() + timedelta(days=60)))
                nxt = next((d for d in days if datetime.combine(d, start.timetz()) >= to_local(as_of)), None)
                if nxt is None:
                    continue
                shift = datetime.combine(nxt, start.timetz()) - start
                start, end = start + shift, end + shift
            upcoming = start >= to_local(as_of)
            key = (len(hit) / len(want), upcoming, len(ev.get("attendees", [])), -len(have - want))
            if best_key is None or key > best_key:
                best, best_key = (ev, start, end), key
        return best


class ActionPlanner:
    """Plans dry-run actions for a command. `memory` (optional) is a callable question -> evidence text."""

    def __init__(self, data_dir: str | Path = "data", memory=None):
        self.dir = Directory(data_dir)
        self.memory = memory

    # ------------------------------------------------------------------ entry
    def plan_command(self, cmd: ActionCommand) -> ActionPrediction:
        as_of = to_local(datetime.fromisoformat(cmd.as_of.replace("Z", "+00:00")))
        actions: List[Dict[str, Any]] = []
        for clause in self._split(cmd.command.strip()):
            actions.extend(self._plan_clause(clause, as_of))
        return ActionPrediction(id=cmd.id, actions=actions)

    def _split(self, command: str) -> List[str]:
        """'Email John X and thank Ben on Slack' -> two clauses. 'Email Sarah and ask if ...' stays one."""
        parts = re.split(rf"\s*,?\s+and\s+(?:then\s+|also\s+)?(?=(?:{ACTION_VERBS})\b)", command, flags=re.I)
        merged: List[str] = []
        for p in parts:
            continuation = re.match(r"^(?:ask|tell|let|remind|thank)\s+(?:him|her|them|if|whether)\b", p, re.I) \
                or (re.match(r"^ask\b", p, re.I) and merged and not self.dir.find_people(p))
            if merged and continuation:
                merged[-1] = f"{merged[-1]} and {p}"
            else:
                merged.append(p)
        return merged

    def _plan_clause(self, clause: str, as_of: datetime) -> List[Dict[str, Any]]:
        c = re.sub(r"^(?:hey\s+\w+,?\s*)?(?:please|can you|could you|would you|i want you to|go ahead and)\s+", "", clause.strip(), flags=re.I).rstrip(" .!")
        low = c.lower()

        if DESTRUCTIVE.search(low) and not re.match(r"^(?:what|when|who|did|is|are)\b", low):
            return [{"type": "confirm", "args": {"summary": f"This is destructive: \"{c}\". Should I go ahead? (yes/no)"}}]
        if (QUESTION_START.match(low) or c.endswith("?") or re.match(r"^(?:find out|look up|check|remind me what|remind me when|remind me who)\b", low)) \
                and not re.match(rf"^(?:{ACTION_VERBS})\b", low.replace("remind me what", "x").replace("remind me when", "x")):
            return [{"type": "memory.ask", "args": {"question": clause.strip()}}]
        m = re.match(r"^(?:open|launch|start|bring up|pull up|show me)\s+(?:up\s+)?(?:the\s+|my\s+)?(.+?)(?:\s+app)?$", low)
        if m and not self.dir.find_event(m.group(1), as_of):
            return [{"type": "app.open", "args": {"app": self._app_name(m.group(1))}}]
        if re.search(r"\bremind me\b|\breminder\b", low):
            return [self._reminder(c, as_of)]
        m = re.match(r"^(?:move|reschedule|push|shift|change|bump)\s+(?:the\s+|my\s+)?(.+?)\s+(to|by|back by|forward by|later by|earlier by)\s+(.+)$", c, re.I)
        if m:
            planned = self._update_event(m.group(1), m.group(2).lower(), m.group(3), as_of)
            if planned:
                return [planned]
        if re.match(r"^(?:book|schedule|set up|arrange|put|create|add|find time|invite)\b", low) and \
                re.search(r"\b(?:with|meeting|call|sync|1:1|minutes?|hours?|event|calendar|time|invite)\b", low):
            return [self._create_event(c, as_of)]
        if re.match(r"^(?:email|e-mail|mail|send (?:an? )?e-?mail|write (?:an? )?(?:email|note) to|reply to)\b", low) or \
                re.search(r"\b(?:by|via|over) e-?mail\b", low):
            return [self._email(c, as_of)]
        if re.match(r"^(?:message|msg|slack|dm|text|tell|ping|post|thank|let|send|ask|share)\b", low):
            return [self._message(c, as_of)]
        return [{"type": "memory.ask", "args": {"question": clause.strip()}}]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _app_name(raw: str) -> str:
        r = raw.lower().strip()
        for app in KNOWN_APPS:
            if re.search(rf"\b{re.escape(app)}\b", r):
                return app.title() if app not in ("vs code", "vscode", "chatgpt", "github", "1password") else app
        return raw.strip().title()

    def _clarify_people(self, mentions: List[List[Person]]) -> Optional[Dict[str, Any]]:
        for cands in mentions:
            if len(cands) > 1:
                names = " or ".join(p.name for p in cands)
                return {"type": "clarify", "args": {"question": f"Which {cands[0].first} do you mean: {names}?"}}
        return None

    def _strip_time_phrases(self, text: str) -> str:
        t = re.sub(r"\b(?:\d+|an?|one|two|three|half an?|thirty|fifteen|forty-five|ten|five|twenty)\s*(?:minutes?|mins?|hours?|hrs?|days?)\s+(?:before|after|ahead of)\s+(?:the\s+|my\s+)?[\w/ -]+?(?=\s+to\b|$)", " ", text, flags=re.I)
        t = re.sub(r"\b(?:on\s+)?(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\b", " ", t, flags=re.I)
        t = re.sub(r"\b(?:at|by|around)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?", " ", t, flags=re.I)
        t = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", " ", t, flags=re.I)
        t = re.sub(r"\b(?:on\s+|next\s+|this\s+)?(?:today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", " ", t, flags=re.I)
        t = re.sub(r"\b(?:on\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b", " ", t, flags=re.I)
        t = re.sub(r"\bin\s+(?:\d+|an?|half an?)\s*(?:minutes?|mins?|hours?)\b", " ", t, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip(" ,.")
        return re.sub(r"\s+(?:on|at|by)$", "", t)

    def _when(self, text: str, as_of: datetime, default_time: time = time(9, 0)) -> Optional[datetime]:
        day = resolve_day(text, as_of)
        clock = parse_clock(text)
        if day is None and clock is None:
            return None
        if day is None:
            cand = datetime.combine(as_of.date(), clock, tzinfo=LOCAL_TZ)
            return cand if cand > as_of else cand + timedelta(days=1)
        return datetime.combine(day, clock or default_time, tzinfo=LOCAL_TZ)

    def _memory_snippet(self, phrase: str, as_of: datetime) -> str:
        if not self.memory or not phrase:
            return ""
        try:
            return self.memory(phrase, as_of) or ""
        except Exception:
            return ""

    # ------------------------------------------------------------------ action builders
    def _reminder(self, c: str, as_of: datetime) -> Dict[str, Any]:
        due: Optional[datetime] = None
        m = re.search(r"\b(\d+|an?|one|two|three|half an?|thirty|fifteen|forty-five|ten|five|twenty)\s*(minutes?|mins?|hours?|hrs?|days?)\s+(before|after|ahead of)\s+(?:the\s+|my\s+)?(.+?)(?=\s+to\b|$)", c, re.I)
        if m:
            raw = m.group(1).lower()
            n = float(raw) if raw.isdigit() else (0.5 if raw.startswith("half") else float(NUMBER_WORDS.get(raw, 1)))
            unit = m.group(2).lower()
            delta = timedelta(days=n) if unit.startswith("d") else timedelta(hours=n) if unit.startswith("h") else timedelta(minutes=n)
            ev = self.dir.find_event(m.group(4), as_of)
            if ev:
                due = ev[1] - delta if m.group(3).lower() != "after" else ev[2] + delta
        if due is None:
            rel = re.search(r"\bin\s+(\d+|an?|half an?)\s*(minutes?|mins?|hours?)\b", c, re.I)
            if rel:
                due = as_of + (parse_duration(rel.group(0)) or timedelta(hours=1))
        if due is None:
            due = self._when(c, as_of)
        if due is None:
            due = as_of + timedelta(hours=1)
        text = re.sub(r"^.*?\bremind me\s+(?:to\s+|that\s+|about\s+)?|^.*?\bset (?:a |up a )?reminder\s+(?:to\s+|for\s+|about\s+)?", "", c, flags=re.I)
        # "remind me an hour before X to print the deck" -> "print the deck"
        tail = re.search(r"\bto\s+(.+)$", text, re.I)
        if m and tail:
            text = tail.group(1)
        text = self._strip_time_phrases(text)
        text = re.sub(r"^to\s+", "", text, flags=re.I)
        return {"type": "reminder.create", "args": {"text": text[:1].upper() + text[1:] if text else c, "due": due.isoformat()}}

    def _update_event(self, phrase: str, mode: str, target: str, as_of: datetime) -> Optional[Dict[str, Any]]:
        found = self.dir.find_event(phrase, as_of)
        if not found:
            return None
        ev, start, end = found
        duration = end - start
        if mode != "to":
            delta = parse_duration(target) or timedelta(hours=1)
            sign = -1 if "earlier" in mode or "forward" in mode else 1
            new_start = start + sign * delta
        else:
            day = resolve_day(target, as_of)
            clock = parse_clock(target) or (parse_clock(f"at {target.strip()}") if re.match(r"^\d{1,2}$", target.strip()) else None)
            new_start = datetime.combine(day or start.date(), clock or start.timetz().replace(tzinfo=None), tzinfo=LOCAL_TZ)
        return {"type": "calendar.update_event", "args": {
            "event_id": ev["id"], "start": new_start.isoformat(), "end": (new_start + duration).isoformat()}}

    def _create_event(self, c: str, as_of: datetime) -> Dict[str, Any]:
        mentions = self.dir.find_people(c)
        clar = self._clarify_people(mentions)
        if clar:
            return clar
        attendees = [cands[0].email for cands in mentions if cands[0].email]
        duration = parse_duration(c) or timedelta(minutes=30)
        start = self._when(re.sub(r"\b\d+\s*(?:minutes?|mins?|hours?)\b|\ban? hour\b", " ", c, flags=re.I), as_of, default_time=time(10, 0))
        if start is None:
            return {"type": "clarify", "args": {"question": "What day and time should I book it for?"}}
        topic = None
        for pattern in (r"\b(?:about|regarding|re:?)\s+(?:the\s+)?(.+)$", r"\b(to (?:review|discuss|go over|talk about|plan|prep(?:are)?)\s+.+)$",
                        r"\b(?:for|on)\s+(?:the\s+)?(?!(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow)\b)(.+)$"):
            topic = re.search(pattern, c, re.I)
            if topic:
                break
        title = self._strip_time_phrases(topic.group(1)) if topic else ""
        title = re.sub(r"^to\s+", "", title, flags=re.I)
        if not title or re.match(r"^\d", title):
            title = "Meeting with " + ", ".join(cands[0].name for cands in mentions) if mentions else "Meeting"
        return {"type": "calendar.create_event", "args": {
            "title": title[:1].upper() + title[1:], "start": start.isoformat(),
            "end": (start + duration).isoformat(), "attendees": attendees}}

    def _body(self, c: str, person_names: List[str], as_of: datetime) -> Tuple[str, str]:
        """(subject, body) from a command like 'email X that ...' / 'email X the corrected NRR'."""
        rest = c
        for n in sorted(person_names, key=len, reverse=True):
            rest = re.sub(rf"\b{re.escape(n)}\b", " ", rest, count=1, flags=re.I)
        rest = re.sub(r"^\s*(?:email|e-mail|mail|send (?:an? )?e-?mail(?: to)?|write (?:an? )?(?:email|note) to|reply to|message|msg|slack|dm|text|tell|ping|post|send|share|let)\b", "", rest, flags=re.I)
        rest = re.sub(r"\b(?:on|via|over|in) (?:slack|e-?mail)\b|\bby e-?mail\b", " ", rest, flags=re.I)
        rest = re.sub(r"\s+", " ", rest).strip(" ,.:")
        m = re.match(r"^(?:know\s+)?(?:that|saying|to say|and say|:)\s+(.+)$", rest, re.I)
        if m:
            body = m.group(1)
            subject = " ".join(body.split()[:6])
            return subject[:1].upper() + subject[1:], body[:1].upper() + body[1:]
        m = re.match(r"^(?:and\s+)?(ask(?:ing)?|thank(?:ing)?)\s+(?:him|her|them)?\s*(.+)$", rest, re.I)
        if m:
            verb = m.group(1).lower()
            what = m.group(2).strip()
            what = re.sub(r"^(?:if|whether)\s+(?:she|he|they)(?:'s| has| have|'ve)\s+", "have you ", what, flags=re.I)
            what = re.sub(r"^(?:if|whether)\s+(?:she|he|they)(?:'ll| will)\s+", "will you ", what, flags=re.I)
            what = re.sub(r"^(?:if|whether)\s+(?:she|he|they)\s+(?:is|are)\s+", "are you ", what, flags=re.I)
            what = re.sub(r"^(?:if|whether)\s+(?:she|he|they)\s+", "do you ", what, flags=re.I)
            what = re.sub(r"^(?:for|to)\s+", "", what) if verb.startswith("ask") else what
            what = what[:1].upper() + what[1:]
            body = (f"{what}?" if verb.startswith("ask") else f"Thank you {what}!").replace("??", "?")
            topic = re.search(r"\b(?:the|your|our)\s+([\w -]+)$", what)
            return (topic.group(1).capitalize() if topic else "Quick question"), body
        m = re.match(r"^(?:about|regarding|re)\s+(.+)$", rest, re.I)
        topic = m.group(1) if m else rest
        snippet = self._memory_snippet(topic, as_of)
        body = f"Following up on {topic}." + (f" {snippet}" if snippet else "")
        return topic[:1].upper() + topic[1:], body

    def _email(self, c: str, as_of: datetime) -> Dict[str, Any]:
        mentions = self.dir.find_people(c)
        emails = [a.lower() for a in re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", c)]
        mentions = [[p for p in cands if p.email] or cands for cands in mentions]
        clar = self._clarify_people(mentions)
        if clar:
            return clar
        to = emails + [cands[0].email for cands in mentions if cands[0].email]
        if not to:
            return {"type": "clarify", "args": {"question": "Who should I send the email to?"}}
        cc_m = re.search(r"\bcc(?:'?ing)?\s+(.+?)(?:\s+(?:that|about|saying)\b|$)", c, re.I)
        cc = [cands[0].email for cands in self.dir.find_people(cc_m.group(1))] if cc_m else []
        to = [t for t in to if t not in cc]
        subject, body = self._body(re.sub(r"\bcc(?:'?ing)?\s+\w+(?:\s+\w+)?", " ", c, flags=re.I),
                                   [p.name for cands in mentions for p in cands] + [cands[0].first for cands in mentions], as_of)
        greet = mentions[0][0].first if len(mentions) == 1 else "all"
        return {"type": "gmail.send", "args": {"to": to, "cc": cc, "subject": subject, "body": f"Hi {greet},\n\n{body}\n\n{self.dir.owner_first}"}}

    def _message(self, c: str, as_of: datetime) -> Dict[str, Any]:
        low = c.lower()
        channel = self.dir.find_channel(c)
        body_src = c
        if channel:
            body_src = re.sub(r"\b(?:in|to|on)?\s*(?:the\s+)?#?[\w-]+(?:\s+[\w-]+)?\s+(?:channel|room)\b|\b(?:in|to|on)\s+#[\w-]+", " ", c, count=1, flags=re.I)
            _, text = self._slack_text(body_src, [], as_of)
            return {"type": "slack.send_message", "args": {"to": channel, "text": text}}
        mentions = self.dir.find_people(c)
        wants_slack = bool(re.search(r"\b(?:on|via|in|over) slack\b|^slack\b|^dm\b", low))
        if wants_slack:
            mentions = [[p for p in cands if p.slack_id] or cands for cands in mentions]
        clar = self._clarify_people(mentions)
        if clar:
            return clar
        if not mentions:
            return {"type": "clarify", "args": {"question": "Who (or which channel) should I send this to?"}}
        p = mentions[0][0]
        names = [x.name for cands in mentions for x in cands] + [p.first]
        if not p.slack_id and p.email:
            return self._email(re.sub(r"^\w+", "email", c, count=1), as_of)
        _, text = self._slack_text(c, names, as_of)
        return {"type": "slack.send_message", "args": {"to": p.slack_id or p.dm_id, "text": text}}

    def _slack_text(self, c: str, names: List[str], as_of: datetime) -> Tuple[str, str]:
        low = c.lower().strip()
        m = re.match(r"^thank\s+(\w+(?:\s+\w+)?)\s*(?:on slack|via slack)?\s*(?:for\s+(.+))?$", low)
        if m:
            what = re.sub(r"\b(?:on|via) slack\b", "", m.group(2) or "").strip()
            return "Thanks", f"Thanks{', ' + names[-1] if names else ''}{' for ' + what if what else ''}!"
        subject, body = self._body(c, names, as_of)
        body = re.sub(r"^Following up on (.+?)\.", r"\1", body) if not re.match(r"^(?:the|our|my)\b", subject.lower()) else body
        return subject, body[:1].upper() + body[1:]
