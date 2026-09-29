"""Edge cases: speakers, time travel, time parsing."""
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candor.ingestion import DataIngestion
from candor.system import MemorySystem
from candor.timeparse import explicit_dates, parse_clock, parse_duration, resolve_day

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


class TestEdgeCases(unittest.TestCase):

    def test_every_meeting_segment_has_a_speaker_label(self):
        for u in DataIngestion("data").units:
            if u.source_type == "meeting":
                self.assertTrue(u.author_name)

    def test_time_travel_changes_the_answer(self):
        mem = MemorySystem("data")
        early = mem.ask("T", "When is Route Planner v2 launching?", datetime(2026, 9, 9, 9, 0, tzinfo=LOCAL_TZ))
        late = mem.ask("T", "When is Route Planner v2 launching?", datetime(2026, 9, 18, 18, 0, tzinfo=LOCAL_TZ))
        self.assertNotIn("Oct", early.answer)
        self.assertNotEqual(early.answer, late.answer)

    def test_date_and_time_parsing(self):
        ref = datetime(2026, 9, 16, 12, 0, tzinfo=LOCAL_TZ)
        self.assertEqual(explicit_dates("see you Sep 23rd", ref), {date(2026, 9, 23)})
        self.assertEqual(explicit_dates("by 9/25", ref), {date(2026, 9, 25)})
        self.assertEqual(resolve_day("tomorrow at 2", ref), date(2026, 9, 17))
        self.assertEqual(resolve_day("on Monday", ref), date(2026, 9, 21))
        self.assertEqual(parse_clock("at 2").hour, 14)
        self.assertEqual(parse_clock("9am").hour, 9)
        self.assertEqual(parse_duration("an hour").total_seconds(), 3600)


if __name__ == "__main__":
    unittest.main()
