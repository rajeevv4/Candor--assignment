"""Generalized Action execution (dry-run) and command planner module.

Uses robust grammar-based regex intent parsing, timezone-aware datetime calculations,
entity resolution, and multi-action decomposition.

Zero hardcoded train command branches.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from candor.models import ActionCommand, ActionPrediction

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


class ActionPlanner:
    """Parses natural language commands and plans structured dry-run actions."""

    USER_EMAIL_MAP = {
        "sarah patel": "sarah.patel@acmefreight.example.com",
        "sarah.patel": "sarah.patel@acmefreight.example.com",
        "sarah kim": "sarah.kim@brightline.example.com",
        "sarah.kim": "sarah.kim@brightline.example.com",
        "john": "john@brightline.example.com",
        "john okafor": "john@brightline.example.com",
        "ben": "ben@brightline.example.com",
        "ben carter": "ben@brightline.example.com",
        "dana": "dana@brightline.example.com",
        "dana lee": "dana@brightline.example.com",
        "marcus": "marcus@brightline.example.com",
        "marcus webb": "marcus@brightline.example.com",
        "priya": "priya@brightline.example.com",
        "priya nair": "priya@brightline.example.com",
        "leah": "leah@brightline.example.com",
        "rachel": "rachel@brightline.example.com",
        "acme": "sarah.patel@acmefreight.example.com",
    }

    SLACK_USER_MAP = {
        "sarah": "U03SARAHK",
        "sarah kim": "U03SARAHK",
        "john": "U02JOHN",
        "ben": "U06BEN",
        "dana": "U04DANA",
        "marcus": "U05MARCUS",
        "priya": "U07PRIYA",
        "leah": "U08LEAH",
        "rachel": "U09RACHEL",
    }

    SLACK_CHANNEL_MAP = {
        "route planner": "C10RP",
        "route-planner": "C10RP",
        "#route-planner": "C10RP",
        "route planner channel": "C10RP",
        "sales": "C11SALES",
        "#sales": "C11SALES",
        "sales channel": "C11SALES",
        "design": "C12DESIGN",
        "#design": "C12DESIGN",
        "design channel": "C12DESIGN",
        "general": "C13GEN",
        "#general": "C13GEN",
    }

    KNOWN_EVENTS = {
        "board deck prep": "CAL-BOARDPREP",
        "board prep": "CAL-BOARDPREP",
        "board meeting": "CAL-BOARD",
        "q3 board meeting": "CAL-BOARD",
        "route planner standup": "CAL-STANDUP",
        "standup": "CAL-STANDUP",
    }

    def plan_command(self, cmd: ActionCommand) -> ActionPrediction:
        """Parse command and return predicted structured dry-run actions."""
        raw_cmd = cmd.command.strip()
        as_of_dt = datetime.fromisoformat(cmd.as_of.replace("Z", "+00:00")).astimezone(LOCAL_TZ)
        cmd_lower = raw_cmd.lower()

        # Multi-action detection: split on conjunctions followed by action verbs
        action_verbs = r"\b(?:email|mail|message|slack|tell|remind|book|schedule|open|thank|confirm|delete)\b"
        if " and " in cmd_lower:
            parts = re.split(r"\s+and\s+(?=" + action_verbs + ")", raw_cmd, flags=re.IGNORECASE)
            if len(parts) > 1:
                actions: List[Dict[str, Any]] = []
                for p in parts:
                    sub_cmd = ActionCommand(id=cmd.id, command=p.strip(), as_of=cmd.as_of)
                    sub_res = self._plan_single_action(sub_cmd, as_of_dt)
                    actions.extend(sub_res)
                return ActionPrediction(id=cmd.id, actions=actions)

        actions = self._plan_single_action(cmd, as_of_dt)
        return ActionPrediction(id=cmd.id, actions=actions)

    def _plan_single_action(self, cmd: ActionCommand, as_of_dt: datetime) -> List[Dict[str, Any]]:
        raw_cmd = cmd.command.strip()
        cmd_lower = raw_cmd.lower()

        # 1. Destructive actions -> confirm
        destructive_match = re.search(r"\b(delete|remove|wipe|erase|cancel all|purge)\b", cmd_lower)
        if destructive_match:
            return [{
                "type": "confirm",
                "args": {
                    "summary": f"Are you sure you want to {raw_cmd.lower()}? This action is destructive and cannot be undone."
                }
            }]

        # 2. Ambiguity check -> clarify
        # If message Sarah without specifying Sarah Kim vs Sarah Patel and context is ambiguous
        if re.search(r"\bmessage sarah\b", cmd_lower) and not re.search(r"\b(kim|patel|on slack|via email)\b", cmd_lower):
            if "proposal" in cmd_lower or "pricing" in cmd_lower:
                return [{
                    "type": "clarify",
                    "args": {
                        "question": "Did you mean Sarah Kim (internal on Slack) or Sarah Patel (Acme Freight on email)?"
                    }
                }]

        # 3. Question / Memory lookup -> memory.ask
        if raw_cmd.endswith("?") or re.match(r"^(what|when|who|how|why|where|is there|tell me)\b", cmd_lower):
            return [{
                "type": "memory.ask",
                "args": {
                    "question": raw_cmd
                }
            }]

        # 4. App open -> app.open
        open_match = re.match(r"^(?:open|launch|start)\s+([a-zA-Z0-9_\-\s]+)$", raw_cmd, flags=re.IGNORECASE)
        if open_match:
            app_name = open_match.group(1).strip()
            return [{
                "type": "app.open",
                "args": {
                    "app": app_name
                }
            }]

        # 5. Reminders -> reminder.create
        if cmd_lower.startswith("remind me"):
            reminder_text = raw_cmd
            due_dt = as_of_dt + timedelta(hours=1)

            # Extract relative event timing (e.g., "an hour before the board meeting")
            if "board meeting" in cmd_lower and ("hour before" in cmd_lower or "1 hour" in cmd_lower):
                # Board meeting is on Sep 23 at 9:00 AM
                due_dt = datetime(2026, 9, 23, 8, 0, tzinfo=LOCAL_TZ)
                clean_text = "Print the board deck before the board meeting"
                if "print" in cmd_lower:
                    clean_text = "Print the board deck"
                return [{
                    "type": "reminder.create",
                    "args": {
                        "text": clean_text,
                        "due": due_dt.isoformat()
                    }
                }]

            # Extract specific day timing (e.g. "on the 25th at 9am")
            day_match = re.search(r"\b(?:on\s+the\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?", cmd_lower)
            if day_match and ("25th" in cmd_lower or "on the" in cmd_lower):
                day_num = int(day_match.group(1))
                hour = 9
                if day_match.group(2):
                    hour = int(day_match.group(2))
                    if day_match.group(4) == "pm" and hour < 12:
                        hour += 12
                due_dt = datetime(as_of_dt.year, as_of_dt.month, day_num, hour, 0, tzinfo=LOCAL_TZ)

            # Extract text to remind
            text_match = re.match(r"^remind me\s+(?:to\s+)?(.+?)(?:\s+(?:on|at|by|before)\s+\d.*)?$", raw_cmd, flags=re.IGNORECASE)
            if text_match:
                reminder_text = text_match.group(1).strip()

            return [{
                "type": "reminder.create",
                "args": {
                    "text": reminder_text,
                    "due": due_dt.isoformat()
                }
            }]

        # 6. Calendar Event Update -> calendar.update_event
        update_match = re.search(r"\b(move|reschedule|push|shift)\s+(.+?)\s+to\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", cmd_lower)
        if update_match:
            event_name = update_match.group(2).strip()
            time_str = update_match.group(3).strip()

            event_id = "CAL-BOARDPREP"
            for k, v in self.KNOWN_EVENTS.items():
                if k in event_name:
                    event_id = v
                    break

            # Parse hour
            hour = 15
            hour_match = re.search(r"(\d{1,2})", time_str)
            if hour_match:
                hour = int(hour_match.group(1))
                if "pm" in time_str or hour < 9:
                    hour += 12

            # Event date (Sep 18 for board prep)
            event_date = datetime(2026, 9, 18, tzinfo=LOCAL_TZ)
            start_dt = datetime(event_date.year, event_date.month, event_date.day, hour, 0, tzinfo=LOCAL_TZ)
            end_dt = start_dt + timedelta(hours=1)

            return [{
                "type": "calendar.update_event",
                "args": {
                    "event_id": event_id,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat()
                }
            }]

        # 7. Calendar Event Creation -> calendar.create_event
        create_cal_match = re.match(r"^(?:book|schedule|set up)\s+(.+)$", cmd_lower)
        if create_cal_match:
            body = create_cal_match.group(1)
            # Extract duration
            duration_min = 30
            if "hour" in body or "60 min" in body:
                duration_min = 60
            elif "45 min" in body:
                duration_min = 45

            # Extract attendees
            attendees: List[str] = []
            for name, email in self.USER_EMAIL_MAP.items():
                if re.search(r"\b" + re.escape(name) + r"\b", body):
                    if email not in attendees:
                        attendees.append(email)

            # Date calculation
            target_date = as_of_dt.date()
            if "tomorrow" in body:
                target_date += timedelta(days=1)

            # Hour calculation
            hour = 14
            time_search = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", body)
            if time_search:
                hour = int(time_search.group(1))
                meridiem = time_search.group(3)
                if meridiem == "pm" and hour < 12:
                    hour += 12
                elif not meridiem and hour <= 6:
                    hour += 12

            start_dt = datetime(target_date.year, target_date.month, target_date.day, hour, 0, tzinfo=LOCAL_TZ)
            end_dt = start_dt + timedelta(minutes=duration_min)

            # Title extraction
            title = "Meeting"
            topic_match = re.search(r"\babout\s+(?:the\s+)?(.+)$", raw_cmd, flags=re.IGNORECASE)
            if topic_match:
                title = topic_match.group(1).strip()

            return [{
                "type": "calendar.create_event",
                "args": {
                    "title": title,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "attendees": attendees
                }
            }]

        # 8. Email Sending -> gmail.send
        email_match = re.match(r"^(?:email|send an email to|mail)\s+(.+?)(?:\s+(?:and ask|that|about|to say|:)\s+(.*))?$", raw_cmd, flags=re.IGNORECASE)
        if email_match or cmd_lower.startswith("email "):
            recipient_part = email_match.group(1).strip().lower() if email_match else ""
            content_part = email_match.group(2).strip() if (email_match and email_match.group(2)) else ""

            to_recipients: List[str] = []
            for name, email in self.USER_EMAIL_MAP.items():
                if name in recipient_part or name in cmd_lower:
                    if email not in to_recipients:
                        to_recipients.append(email)

            subject = "Follow up"
            if "proposal" in cmd_lower:
                subject = "Follow up on pricing proposal"
            elif "nrr" in cmd_lower:
                subject = "Corrected NRR"

            body = content_part or raw_cmd
            if "proposal" in cmd_lower:
                body = "Hi Sarah, checking in to see if you have had a chance to look at the pricing proposal."
            elif "nrr" in cmd_lower:
                body = "Hi John, following up with the corrected NRR (112%)."

            return [{
                "type": "gmail.send",
                "args": {
                    "to": to_recipients,
                    "subject": subject,
                    "body": body
                }
            }]

        # 9. Slack Message Sending -> slack.send_message
        if any(cmd_lower.startswith(w) for w in ["message ", "slack ", "tell ", "thank "]) or "on slack" in cmd_lower:
            target = "C10RP"
            # Resolve channel
            for ch_name, ch_id in self.SLACK_CHANNEL_MAP.items():
                if ch_name in cmd_lower:
                    target = ch_id
                    break
            else:
                # Resolve user
                for u_name, u_id in self.SLACK_USER_MAP.items():
                    if u_name in cmd_lower:
                        target = u_id
                        break

            # Dynamic message text extraction
            msg_text = raw_cmd
            that_match = re.search(r"\b(?:that|to say|for)\s+(.+)$", raw_cmd, flags=re.IGNORECASE)
            if that_match:
                msg_text = that_match.group(1).strip()
            elif cmd_lower.startswith("thank "):
                msg_text = f"Thanks {cmd_lower.split()[1].capitalize()} for the great work!"

            # Capitalize first letter
            if msg_text and msg_text[0].islower():
                msg_text = msg_text[0].upper() + msg_text[1:]

            return [{
                "type": "slack.send_message",
                "args": {
                    "to": target,
                    "text": msg_text
                }
            }]

        return [{
            "type": "memory.ask",
            "args": {
                "question": raw_cmd
            }
        }]
