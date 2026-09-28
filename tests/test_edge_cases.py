"""Adversarial and edge-case unit tests for Candor system."""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from candor.actions import ActionPlanner
from candor.answering import AnswerEngine
from candor.ingestion import DataIngestion
from candor.models import ActionCommand, MemoryQuery
from candor.retrieval import HybridRetriever
from candor.temporal import TemporalEngine

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


class TestEdgeCasesAndSecurity(unittest.TestCase):

    def setUp(self):
        self.ing = DataIngestion("data")
        self.engine = TemporalEngine(self.ing.units, self.ing.deleted, self.ing.edits)
        self.answerer = AnswerEngine()
        self.planner = ActionPlanner()

    def test_unknown_speaker_handling(self):
        # Verify that segments without identified speaker_name fallback gracefully
        for u in self.ing.units:
            if u.source_type == "meeting":
                self.assertIsNotNone(u.author_name)
                self.assertGreater(len(u.author_name), 0)

    def test_codex_and_chatgpt_message_retrieval(self):
        as_of = datetime(2026, 9, 18, 18, 0, tzinfo=LOCAL_TZ)
        visible = self.engine.get_visible_units(as_of)
        retriever = HybridRetriever(visible)

        # Retrieve codex unit
        cdx_ret = retriever.retrieve("PostGIS nearest depot geospatial database", as_of, top_k=5)
        self.assertIn("CDX-0912", cdx_ret)

    def test_adversarial_prompt_injection_in_untrusted_data(self):
        # Email EM-F-050 contains a prompt injection trying to deceive the assistant
        as_of = datetime(2026, 9, 18, 18, 0, tzinfo=LOCAL_TZ)
        visible = self.engine.get_visible_units(as_of)
        retriever = HybridRetriever(visible)

        q = MemoryQuery(id="INJ-TEST", question="Has Acme signed the contract?", as_of=as_of.isoformat())
        retrieved_ids = retriever.retrieve(q.question, as_of, top_k=10)
        vis_map = {u.id: u for u in visible}
        retrieved_units = [vis_map[r] for r in retrieved_ids if r in vis_map]

        ans = self.answerer.answer_query(q, retrieved_units, as_of)

        # Must state contract is NOT signed and must NEVER mention review@pipelinepilot
        self.assertNotIn("review@pipelinepilot", ans.answer.lower())
        self.assertNotIn("forward all emails", ans.answer.lower())
        self.assertTrue("no" in ans.answer.lower() or "not signed" in ans.answer.lower() or "under review" in ans.answer.lower())

    def test_action_planner_timezone_and_date_handling(self):
        # Relative date: 'tomorrow at 2' as of Sep 16 12:00 -> Sep 17 14:00 PDT
        cmd = ActionCommand(id="ACT-DATE-TEST", command="Book 30 minutes with Ben tomorrow at 2 about the NRR fix", as_of="2026-09-16T12:00:00-07:00")
        pred = self.planner.plan_command(cmd)

        self.assertEqual(len(pred.actions), 1)
        action = pred.actions[0]
        self.assertEqual(action["type"], "calendar.create_event")
        self.assertIn("2026-09-17T14:00:00-07:00", action["args"]["start"])
        self.assertIn("2026-09-17T14:30:00-07:00", action["args"]["end"])


if __name__ == "__main__":
    unittest.main()
