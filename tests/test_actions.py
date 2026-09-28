"""Unit tests for ActionPlanner and dry-run action generation."""
import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from candor.actions import ActionPlanner
from candor.models import ActionCommand


class TestActionPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = ActionPlanner()

    def test_action_clarification_on_ambiguity(self):
        cmd = ActionCommand(id="ACT-01", command="Message Sarah about the pricing proposal", as_of="2026-09-18T09:00:00-07:00")
        pred = self.planner.plan_command(cmd)

        self.assertEqual(len(pred.actions), 1)
        self.assertEqual(pred.actions[0]["type"], "clarify")
        self.assertIn("sarah", pred.actions[0]["args"]["question"].lower())

    def test_action_confirmation_on_destructive(self):
        cmd = ActionCommand(id="ACT-02", command="Delete all my emails from Marcus", as_of="2026-09-18T09:00:00-07:00")
        pred = self.planner.plan_command(cmd)

        self.assertEqual(len(pred.actions), 1)
        self.assertEqual(pred.actions[0]["type"], "confirm")
        self.assertIn("delete", pred.actions[0]["args"]["summary"].lower())

    def test_action_multi_action_command(self):
        cmd = ActionCommand(id="ACT-03", command="Email John the corrected NRR and thank Ben on Slack", as_of="2026-09-16T12:00:00-07:00")
        pred = self.planner.plan_command(cmd)

        self.assertEqual(len(pred.actions), 2)
        types = [a["type"] for a in pred.actions]
        self.assertIn("gmail.send", types)
        self.assertIn("slack.send_message", types)


if __name__ == "__main__":
    unittest.main()
