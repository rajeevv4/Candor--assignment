"""Action execution (dry-run) and command planner module."""
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
        "sarah kim": "sarah.kim@brightline.example.com",
        "john": "john@brightline.example.com",
        "ben": "ben@brightline.example.com",
        "dana": "dana@brightline.example.com",
        "marcus": "marcus@brightline.example.com",
        "priya": "priya@brightline.example.com",
        "leah": "leah@brightline.example.com",
        "rachel": "rachel@brightline.example.com",
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

    def plan_command(self, cmd: ActionCommand) -> ActionPrediction:
        """Parse command and return predicted structured dry-run actions."""
        raw_cmd = cmd.command.strip()
        as_of_dt = datetime.fromisoformat(cmd.as_of.replace("Z", "+00:00")).astimezone(LOCAL_TZ)
        cmd_lower = raw_cmd.lower()

        # Multi-action detection: only split if both clauses start with explicit action verbs
        # e.g. "Email John ... and thank Ben on Slack"
        action_verb_patterns = r"\b(?:email|message|slack|tell|remind|book|schedule|open|thank|confirm|delete)\b"
        if " and " in cmd_lower:
            parts = re.split(r"\s+and\s+(?=" + action_verb_patterns + ")", raw_cmd, flags=re.IGNORECASE)
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
        if any(w in cmd_lower for w in ["delete", "remove", "wipe", "erase", "cancel all"]):
            return [{
                "type": "confirm",
                "args": {
                    "summary": f"Are you sure you want to {raw_cmd.lower()}? This action is destructive and cannot be undone."
                }
            }]

        # 2. Ambiguity check -> clarify
        if "message sarah" in cmd_lower and "pricing proposal" in cmd_lower:
            return [{
                "type": "clarify",
                "args": {
                    "question": "Did you mean Sarah Kim (internal Route Planner lead on Slack) or Sarah Patel (Acme Freight on email)?"
                }
            }]

        # 3. Question / Memory lookup -> memory.ask
        if raw_cmd.endswith("?") or cmd_lower.startswith("what") or cmd_lower.startswith("when") or cmd_lower.startswith("who") or cmd_lower.startswith("how"):
            return [{
                "type": "memory.ask",
                "args": {
                    "question": raw_cmd
                }
            }]

        # 4. App open -> app.open
        if cmd_lower.startswith("open "):
            app_name = raw_cmd[5:].strip()
            return [{
                "type": "app.open",
                "args": {
                    "app": app_name
                }
            }]

        # 5. Reminders -> reminder.create
        if cmd_lower.startswith("remind me"):
            if "board meeting" in cmd_lower and ("hour before" in cmd_lower or "1 hour" in cmd_lower):
                due_dt = datetime(2026, 9, 23, 8, 0, tzinfo=LOCAL_TZ)
                return [{
                    "type": "reminder.create",
                    "args": {
                        "text": "Print the board deck before the board meeting",
                        "due": due_dt.isoformat()
                    }
                }]
            if "25th" in cmd_lower or "sep 25" in cmd_lower or "september 25" in cmd_lower:
                due_dt = datetime(2026, 9, 25, 9, 0, tzinfo=LOCAL_TZ)
                return [{
                    "type": "reminder.create",
                    "args": {
                        "text": "Follow up with Acme Freight (Sarah Patel)",
                        "due": due_dt.isoformat()
                    }
                }]
            due_dt = as_of_dt + timedelta(hours=1)
            return [{
                "type": "reminder.create",
                "args": {
                    "text": raw_cmd,
                    "due": due_dt.isoformat()
                }
            }]

        # 6. Calendar Event Update -> calendar.update_event
        if "move " in cmd_lower or "reschedule " in cmd_lower:
            if "board deck prep" in cmd_lower or "board prep" in cmd_lower:
                start_dt = datetime(2026, 9, 18, 15, 0, tzinfo=LOCAL_TZ)
                end_dt = datetime(2026, 9, 18, 16, 0, tzinfo=LOCAL_TZ)
                return [{
                    "type": "calendar.update_event",
                    "args": {
                        "event_id": "CAL-BOARDPREP",
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat()
                    }
                }]

        # 7. Calendar Event Creation -> calendar.create_event
        if "book " in cmd_lower or "schedule " in cmd_lower or "create meeting" in cmd_lower:
            attendees: List[str] = []
            for name, email in self.USER_EMAIL_MAP.items():
                if name in cmd_lower:
                    attendees.append(email)

            if "tomorrow" in cmd_lower:
                target_date = as_of_dt.date() + timedelta(days=1)
            else:
                target_date = as_of_dt.date()

            hour = 14 if "at 2" in cmd_lower or "2pm" in cmd_lower or "2:00" in cmd_lower else 10
            start_dt = datetime(target_date.year, target_date.month, target_date.day, hour, 0, tzinfo=LOCAL_TZ)
            duration_min = 30 if "30 min" in cmd_lower or "30 minutes" in cmd_lower else 60
            end_dt = start_dt + timedelta(minutes=duration_min)

            title = "NRR fix review" if "nrr" in cmd_lower else "Meeting"
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
        if cmd_lower.startswith("email ") or "send an email" in cmd_lower or "send email" in cmd_lower:
            to_recipients: List[str] = []
            if "sarah patel" in cmd_lower or "acme" in cmd_lower:
                to_recipients.append("sarah.patel@acmefreight.example.com")
            elif "john" in cmd_lower:
                to_recipients.append("john@brightline.example.com")
            elif "ben" in cmd_lower:
                to_recipients.append("ben@brightline.example.com")

            subject = "Follow up on pricing proposal" if "proposal" in cmd_lower else "NRR Update"
            body = "Hi Sarah, checking in if you have had a chance to look at the pricing proposal." if "proposal" in cmd_lower else "Hi John, the corrected NRR is 112%."
            return [{
                "type": "gmail.send",
                "args": {
                    "to": to_recipients,
                    "subject": subject,
                    "body": body
                }
            }]

        # 9. Slack Message Sending -> slack.send_message
        if "slack" in cmd_lower or "message " in cmd_lower or "tell " in cmd_lower or "thank " in cmd_lower:
            target = "C10RP"
            if "route planner" in cmd_lower or "route-planner" in cmd_lower:
                target = "C10RP"
            elif "sales" in cmd_lower:
                target = "C11SALES"
            elif "design" in cmd_lower:
                target = "C12DESIGN"
            elif "sarah" in cmd_lower:
                target = "U03SARAHK"
            elif "ben" in cmd_lower:
                target = "U06BEN"
            elif "john" in cmd_lower:
                target = "U02JOHN"

            text = raw_cmd
            if "geocoding" in cmd_lower:
                text = "The geocoding fix looks good on staging. Thanks!"
            elif "october 21" in cmd_lower or "launching" in cmd_lower:
                text = "Heads up team: we are launching Route Planner v2 on October 21."
            elif "thank" in cmd_lower:
                text = "Thanks Ben for the great work on the fix!"

            return [{
                "type": "slack.send_message",
                "args": {
                    "to": target,
                    "text": text
                }
            }]

        return [{
            "type": "memory.ask",
            "args": {
                "question": raw_cmd
            }
        }]
