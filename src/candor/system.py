"""One object that owns the memory: ingestion, point-in-time views, retrieval and answering."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from candor.answering import AnswerEngine, redact
from candor.ingestion import DataIngestion
from candor.models import MemoryAnswer, MemoryQuery
from candor.retrieval import HybridRetriever
from candor.temporal import TemporalEngine


class MemorySystem:

    def __init__(self, data_dir: str | Path = "data", model_provider: Optional[str] = None, model_name: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.ingestion = DataIngestion(self.data_dir)
        self.temporal = TemporalEngine(self.ingestion.units, self.ingestion.deleted, self.ingestion.edits)
        self.answerer = AnswerEngine(model_provider=model_provider, model_name=model_name)
        self._retrievers: Dict[datetime, HybridRetriever] = {}

    def retriever(self, as_of: datetime) -> HybridRetriever:
        """Index of what existed at `as_of` (cached per moment)."""
        if as_of not in self._retrievers:
            self._retrievers[as_of] = HybridRetriever(self.temporal.get_visible_units(as_of))
        return self._retrievers[as_of]

    def ask(self, qid: str, question: str, as_of: datetime, top_k: int = 20) -> MemoryAnswer:
        r = self.retriever(as_of)
        ids = r.retrieve(question, as_of, top_k=top_k)
        units = [r.unit(i) for i in ids]
        vocab = set(r.own.idf) | set(r.title.idf)
        return self.answerer.answer_query(MemoryQuery(qid, question, as_of.isoformat()), units, as_of,
                                          vocabulary=vocab, idf=r.own.idf, q_weights=r.query_terms(question))

    def snippet(self, phrase: str, as_of: datetime) -> str:
        """Best single evidence sentence for a phrase (used to fill message bodies like 'the corrected NRR')."""
        r = self.retriever(as_of)
        units = [r.unit(i) for i in r.retrieve(phrase, as_of, top_k=8)]
        cands = self.answerer.evidence(phrase, units, r.own.idf, r.query_terms(phrase))
        if not cands:
            return ""
        # prefer the most recent of the strong candidates: the current value of a fact
        best = cands[0][0]
        strong = sorted((c for c in cands if c[0] >= 0.7 * best), key=lambda c: c[2].timestamp, reverse=True)
        return redact(strong[0][1]).strip()
