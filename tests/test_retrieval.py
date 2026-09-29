"""Retrieval: point-in-time safety and passage-level ranking."""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candor.retrieval import mine_acronyms, stem, tokenize
from candor.system import MemorySystem

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mem = MemorySystem("data")

    def test_nothing_forbidden_is_retrieved(self):
        for as_of in (datetime(2026, 9, 9, 9, 0, tzinfo=LOCAL_TZ), datetime(2026, 9, 12, 12, 0, tzinfo=LOCAL_TZ),
                      datetime(2026, 9, 15, 17, 0, tzinfo=LOCAL_TZ), datetime(2026, 9, 18, 18, 0, tzinfo=LOCAL_TZ)):
            ids = self.mem.retriever(as_of).retrieve("When is Route Planner v2 launching?", as_of, top_k=20)
            self.assertTrue(ids)
            for rid in ids:
                self.assertIsNone(self.mem.temporal.is_forbidden(rid, as_of), rid)

    def test_passage_level_ids(self):
        as_of = datetime(2026, 9, 18, 18, 0, tzinfo=LOCAL_TZ)
        ids = self.mem.retriever(as_of).retrieve("What's the p95 routing latency?", as_of)
        self.assertTrue(any("#" in i for i in ids[:5]), "meeting segments, not whole meetings")

    def test_codex_session_found_by_topic(self):
        as_of = datetime(2026, 9, 18, 18, 0, tzinfo=LOCAL_TZ)
        ids = self.mem.retriever(as_of).retrieve("In the ETA prototype, which database did I pick and why?", as_of)
        self.assertIn("CDX-0912", ids[:5])

    def test_edit_and_original_come_together(self):
        as_of = datetime(2026, 9, 16, 14, 0, tzinfo=LOCAL_TZ)
        ids = self.mem.retriever(as_of).retrieve("How many regression cases were passing?", as_of)
        self.assertIn("SL-EV-0916-EDIT1", ids[:10])
        self.assertIn("SL-RP-0916-1", ids[:10])

    def test_stemming_and_tokenizing(self):
        self.assertEqual(stem("paged"), stem("pages"))
        self.assertEqual(stem("launching"), stem("launched"))
        self.assertIn("p95", tokenize("the 95th percentile latency"))

    def test_acronyms_are_mined_from_text(self):
        acr = mine_acronyms(["Do you support SSO? Like single sign-on with our IdP?"])
        self.assertEqual(acr["sso"], {"single", "sign", "on"})


if __name__ == "__main__":
    unittest.main()
