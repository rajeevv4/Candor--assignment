"""Unit and integration tests for TemporalEngine and visibility logic."""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from candor.ingestion import DataIngestion
from candor.temporal import TemporalEngine

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


class TestTemporalEngine(unittest.TestCase):

    def setUp(self):
        self.ing = DataIngestion("data")
        self.engine = TemporalEngine(self.ing.units, self.ing.deleted, self.ing.edits)

    def test_temporal_visibility_and_future_exclusion(self):
        # As of Sep 9, 2026: No events after Sep 9 should be visible
        as_of = datetime(2026, 9, 9, 12, 0, tzinfo=LOCAL_TZ)
        visible = self.engine.get_visible_units(as_of)

        for u in visible:
            self.assertLessEqual(u.timestamp, as_of)
            self.assertIsNone(self.engine.is_forbidden(u.id, as_of))

    def test_message_deletion_handling(self):
        # SL-DM-AB-0915-2 (Slack secret message) was deleted at 2026-09-15T16:05:02-07:00
        del_target = "SL-DM-AB-0915-2"

        # Before deletion
        before_del = datetime(2026, 9, 15, 16, 3, tzinfo=LOCAL_TZ)
        vis_before = [u.id for u in self.engine.get_visible_units(before_del)]
        self.assertIn(del_target, vis_before)
        self.assertIsNone(self.engine.is_forbidden(del_target, before_del))

        # After deletion
        after_del = datetime(2026, 9, 15, 16, 10, tzinfo=LOCAL_TZ)
        vis_after = [u.id for u in self.engine.get_visible_units(after_del)]
        self.assertNotIn(del_target, vis_after)
        self.assertEqual(self.engine.is_forbidden(del_target, after_del), "deleted")

    def test_message_edit_handling(self):
        # SL-RP-0916-1 was edited by SL-EV-0916-EDIT1 at 2026-09-16T13:20:15-07:00
        target_id = "SL-RP-0916-1"

        # Before edit: should have 60/64
        before_edit = datetime(2026, 9, 16, 13, 10, tzinfo=LOCAL_TZ)
        vis_before = {u.id: u for u in self.engine.get_visible_units(before_edit)}
        self.assertIn(target_id, vis_before)
        self.assertIn("60/64", vis_before[target_id].text)
        self.assertNotIn("61/64", vis_before[target_id].text)

        # After edit: should have 61/64
        after_edit = datetime(2026, 9, 16, 13, 30, tzinfo=LOCAL_TZ)
        vis_after = {u.id: u for u in self.engine.get_visible_units(after_edit)}
        self.assertIn(target_id, vis_after)
        self.assertIn("61/64", vis_after[target_id].text)


if __name__ == "__main__":
    unittest.main()
