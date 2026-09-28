"""Unit and integration tests for HybridRetriever."""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from candor.ingestion import DataIngestion
from candor.retrieval import HybridRetriever
from candor.temporal import TemporalEngine

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


class TestHybridRetriever(unittest.TestCase):

    def setUp(self):
        self.ing = DataIngestion("data")
        self.engine = TemporalEngine(self.ing.units, self.ing.deleted, self.ing.edits)

    def test_retrieval_no_forbidden_records(self):
        as_of = datetime(2026, 9, 12, 12, 0, tzinfo=LOCAL_TZ)
        visible = self.engine.get_visible_units(as_of)
        retriever = HybridRetriever(visible)

        retrieved = retriever.retrieve("When is Route Planner v2 launching?", as_of, top_k=20)
        self.assertGreater(len(retrieved), 0)

        for rid in retrieved:
            self.assertIsNone(self.engine.is_forbidden(rid, as_of))

    def test_retrieval_exact_entity_and_concept_matching(self):
        as_of = datetime(2026, 9, 18, 18, 0, tzinfo=LOCAL_TZ)
        visible = self.engine.get_visible_units(as_of)
        retriever = HybridRetriever(visible)

        retrieved = retriever.retrieve("In the ETA prototype, which database did I pick and why?", as_of, top_k=5)
        self.assertIn("CDX-0912", retrieved)


if __name__ == "__main__":
    unittest.main()
