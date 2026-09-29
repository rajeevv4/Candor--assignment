"""Action planner (dry run)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candor.actions import ActionPlanner
from candor.models import ActionCommand


def plan(text, as_of="2026-09-18T09:00:00-07:00", memory=None):
    return ActionPlanner("data", memory=memory).plan_command(ActionCommand("T", text, as_of)).actions


class TestActionPlanner(unittest.TestCase):

    def test_ambiguous_person_asks_which(self):
        a = plan("Message Sarah about the pricing proposal")
        self.assertEqual([x["type"] for x in a], ["clarify"])
        self.assertIn("Sarah Kim", a[0]["args"]["question"])
        self.assertIn("Sarah Patel", a[0]["args"]["question"])

    def test_channel_constraint_resolves_ambiguity(self):
        a = plan("Message Sarah on Slack that the fix looks good")
        self.assertEqual(a[0]["type"], "slack.send_message")
        self.assertEqual(a[0]["args"]["to"], "U03SARAHK")

    def test_destructive_needs_confirmation(self):
        for cmd in ("Delete all my emails from Marcus", "Cancel all my meetings next week"):
            self.assertEqual(plan(cmd)[0]["type"], "confirm")

    def test_multi_action(self):
        a = plan("Email John the corrected NRR and thank Ben on Slack", "2026-09-16T12:00:00-07:00")
        self.assertEqual(sorted(x["type"] for x in a), ["gmail.send", "slack.send_message"])

    def test_body_filled_from_memory(self):
        a = plan("Email John the corrected NRR", memory=lambda phrase, as_of: "NRR is 112%, not 118%.")
        self.assertIn("112", a[0]["args"]["body"])

    def test_relative_to_event_reminder(self):
        a = plan("Remind me an hour before the board meeting to print the deck")
        self.assertEqual(a[0]["args"]["due"], "2026-09-23T08:00:00-07:00")
        self.assertIn("print", a[0]["args"]["text"].lower())

    def test_move_event_keeps_duration(self):
        a = plan("Move board deck prep to 3pm", "2026-09-17T12:00:00-07:00")
        self.assertEqual(a[0]["args"], {"event_id": "CAL-BOARDPREP", "start": "2026-09-18T15:00:00-07:00",
                                        "end": "2026-09-18T16:00:00-07:00"})

    def test_question_goes_to_memory(self):
        self.assertEqual(plan("What's our launch date again?")[0]["type"], "memory.ask")


if __name__ == "__main__":
    unittest.main()
