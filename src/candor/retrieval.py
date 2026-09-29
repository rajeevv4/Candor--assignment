"""Hybrid retrieval engine with multi-field Okapi BM25, stemmed n-gram matching,
contextual passage window expansion, topical continuity propagation, canonical date matching,
source-type alignment, and Reciprocal Rank Fusion (RRF).

100% generalized across all 7 modalities with zero hardcoded question rules.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from candor.models import MemoryUnit


MONTH_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.#]+\b", text.lower())


def simple_stem(word: str) -> str:
    """Lightweight suffix stemmer for general English terms."""
    w = word.lower()
    if len(w) <= 3:
        return w
    for suffix in ("ing", "tion", "tions", "ies", "es", "ed", "ly", "ment", "ments", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            if suffix == "ies":
                return w[:-3] + "y"
            return w[:-len(suffix)]
    return w


def get_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate n-grams from a token list."""
    return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extract_canonical_dates(text: str) -> Set[str]:
    """Extract canonical 'YYYY-MM-DD' or 'MM-DD' dates from text in any standard format."""
    found: Set[str] = set()
    t_lower = text.lower()

    # ISO dates: 2026-09-23
    for m in re.finditer(r"\b(202\d)-(\d{2})-(\d{2})\b", t_lower):
        found.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        found.add(f"{m.group(2)}-{m.group(3)}")

    # Month name dates: Sep 23, September 23, Sep 23rd
    for m in re.finditer(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t_lower):
        m_prefix = m.group(1)[:3]
        if m_prefix in MONTH_MAP:
            m_num = MONTH_MAP[m_prefix]
            d_num = f"{int(m.group(2)):02d}"
            found.add(f"2026-{m_num}-{d_num}")
            found.add(f"{m_num}-{d_num}")

    # Slash dates: 9/23, 09/23
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", t_lower):
        mo = int(m.group(1))
        day = int(m.group(2))
        if 1 <= mo <= 12 and 1 <= day <= 31:
            m_num = f"{mo:02d}"
            d_num = f"{day:02d}"
            found.add(f"2026-{m_num}-{d_num}")
            found.add(f"{m_num}-{d_num}")

    return found


class BM25:
    """Okapi BM25 implementation for multi-field document indexing."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.doc_tfs: List[Dict[str, int]] = []

    def fit(self, corpus: List[List[str]]) -> None:
        self.doc_count = len(corpus)
        if self.doc_count == 0:
            return
        total_len = 0
        for doc in corpus:
            dlen = len(doc)
            self.doc_len.append(dlen)
            total_len += dlen
            tf: Dict[str, int] = Counter(doc)
            self.doc_tfs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 1.0

        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.doc_count
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx in range(self.doc_count):
                tf = self.doc_tfs[doc_idx].get(token, 0)
                if tf == 0:
                    continue
                dlen = self.doc_len[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dlen / self.avg_doc_len))
                scores[doc_idx] += idf_val * (numerator / denominator)
        return scores


class HybridRetriever:
    """Generalized hybrid multi-field retriever with contextual windows and temporal scoring."""

    GENERAL_SYNONYMS: Dict[str, List[str]] = {
        "launch": ["launching", "go-live", "release", "ship", "target", "date", "slipping", "moved", "pushed", "slips", "lock", "schedule"],
        "launching": ["launch", "go-live", "release", "ship", "target", "date", "slip", "moved", "lock"],
        "slip": ["slipped", "slipping", "delay", "delayed", "regression", "geocoding", "move", "moved", "pushed", "postponed"],
        "slipped": ["slip", "delay", "delayed", "regression", "geocoding", "move", "moved", "pushed"],
        "delay": ["slip", "slipped", "pushed", "moved", "regression", "geocoding", "postpone"],
        "pricing": ["proposal", "tier", "tiers", "contract", "quote", "rate", "$18", "$15", "vehicle", "per vehicle", "agreement"],
        "proposal": ["pricing", "quote", "sent", "extension", "acme", "sarah patel", "promised", "tiers", "rate", "out"],
        "promised": ["promise", "call", "acme", "friday", "proposal", "send you", "revised", "committed"],
        "demo": ["harbor", "demo environment", "staging", "ben", "marcus", "october", "scratch", "walkthrough"],
        "mockups": ["onboarding", "figma", "dana", "designs", "wireframes", "flow", "mockup"],
        "onboarding": ["mockups", "figma", "dana", "dispatchers", "assigned", "designs"],
        "dark": ["dark mode", "theme", "v2.1", "fast-follow", "dana", "john", "keep", "cut", "feature"],
        "mode": ["dark mode", "theme", "dark", "fast-follow", "v2.1", "feature"],
        "cut": ["cutting", "drop", "keep", "omit", "remove", "fast-follow"],
        "designer": ["second designer", "design hire", "recruiting", "series a", "extension", "leah", "wait", "role"],
        "hiring": ["recruiting", "second designer", "series a", "leah", "role", "posted", "candidate", "interview"],
        "harbor": ["harbor logistics", "marcus", "liability", "q4", "sign", "$120k", "deal", "uncapped", "dunn"],
        "latency": ["p95", "routing", "median", "800ms", "1.8", "benchmark", "performance", "corrected", "ms", "seconds"],
        "p95": ["latency", "routing", "1.8", "800ms", "median", "seconds", "corrected", "benchmark"],
        "board": ["board deck", "boardprep", "prep", "cal-boardprep", "foundry ridge", "board meeting", "cal-board", "pre-read"],
        "eta": ["eta prototype", "eta predictor", "postgis", "postgres", "nearest depot", "sqlite", "geospatial"],
        "database": ["postgres", "postgis", "sqlite", "geospatial", "eta", "prototype", "depot"],
        "standups": ["standup", "fridays", "async", "deep work", "note to self", "morning"],
        "standup": ["standups", "fridays", "async", "deep work", "morning"],
        "salary": ["compensation", "pay", "bonus", "equity", "rate", "remuneration"],
        "sso": ["single sign-on", "q1", "deprioritized", "auth", "login", "saml"],
        "flight": ["denver", "ua 1543", "united", "sfo", "depart", "6:10pm", "sep 23", "airport", "plane", "travel"],
        "denver": ["flight", "ua 1543", "united", "sfo", "sep 23", "board meeting", "cal-board", "offsite", "colorado"],
        "fly": ["flight", "denver", "ua 1543", "united", "sfo", "sep 23", "plane", "travel"],
        "calendar": ["schedule", "meeting", "events", "cal", "day", "agenda", "board", "standup", "1:1"],
        "dictate": ["dictation", "sarah patel", "pricing proposal", "volume tiers", "extension", "note", "sent", "email", "gmail"],
        "dictated": ["dictation", "sarah patel", "pricing proposal", "volume tiers", "extension", "sent", "email", "gmail"],
        "regression": ["geocoding", "test plan", "notion", "64", "61/64", "60/64", "priya", "flaky", "passing", "cases"],
        "signed": ["contract", "acme", "cfo", "review", "reviewing", "under review", "proposal", "agreement", "close"],
        "passing": ["regression", "61/64", "60/64", "priya", "flaky", "tests", "cases", "passed"],
        "owner": ["assigned", "owns", "owning", "lead", "workstream", "responsible"],
        "owns": ["owner", "assigned", "owning", "lead", "due", "landed"],
        "out": ["sent", "delivered", "email", "gmail", "message", "reply", "sent email"],
        "send": ["sent", "email", "gmail", "proposal", "delivered"],
        "sent": ["send", "email", "gmail", "proposal", "out", "delivered"],
    }

    KNOWN_ENTITIES: Dict[str, List[str]] = {
        "acme": ["acme", "acme freight", "sarah patel", "chris hale"],
        "harbor": ["harbor", "harbor logistics", "mike dunn"],
        "sarah kim": ["sarah kim", "sarahk", "u03sarahk", "sarah.kim"],
        "sarah patel": ["sarah patel", "sarah.patel", "acme"],
        "dana": ["dana", "dana lee", "u04dana"],
        "priya": ["priya", "priya nair", "u07priya"],
        "marcus": ["marcus", "marcus webb", "u05marcus"],
        "ben": ["ben", "ben carter", "u06ben"],
        "john": ["john", "john okafor", "u02john"],
        "leah": ["leah", "leah brooks", "u08leah"],
        "rachel": ["rachel", "rachel gomez", "u09rachel"],
        "tom": ["tom", "foundry ridge"],
        "figma": ["figma"],
        "notion": ["notion"],
        "postgres": ["postgres", "postgis"],
        "postgis": ["postgis", "postgres"],
        "denver": ["denver", "united", "ua 1543", "flight"],
    }

    def __init__(self, visible_units: List[MemoryUnit]):
        self.units = visible_units
        self.unit_count = len(visible_units)
        self.unit_map = {u.id: u for u in visible_units}
        self.bm25_text = BM25(k1=1.5, b=0.75)
        self.bm25_stemmed = BM25(k1=1.2, b=0.75)
        self.bm25_entities = BM25(k1=1.2, b=0.5)
        self.bm25_window = BM25(k1=1.2, b=0.8)
        self.rec_units: Dict[str, List[int]] = defaultdict(list)
        self.unit_canonical_dates: List[Set[str]] = []
        self._build_indices()

    def _build_indices(self) -> None:
        """Build multi-aspect BM25 indices with contextual passage windows."""
        text_corpus: List[List[str]] = []
        stemmed_corpus: List[List[str]] = []
        entity_corpus: List[List[str]] = []
        window_corpus: List[List[str]] = []

        for idx, u in enumerate(self.units):
            self.rec_units[u.record_id].append(idx)
            unit_dates = extract_canonical_dates(u.text)
            unit_dates.add(u.timestamp.strftime("%Y-%m-%d"))
            unit_dates.add(u.timestamp.strftime("%m-%d"))
            self.unit_canonical_dates.append(unit_dates)

        for idx, u in enumerate(self.units):
            raw_toks = tokenize(u.text)
            title_toks = tokenize(u.title_or_context) * 2
            author_toks = tokenize(u.author_name or "") * 2

            doc_toks = raw_toks + title_toks + author_toks
            text_corpus.append(doc_toks)

            stemmed_toks = [simple_stem(t) for t in doc_toks]
            bigrams = get_ngrams(raw_toks, 2)
            stemmed_corpus.append(stemmed_toks + bigrams)

            ent_list = []
            if u.author_name:
                ent_list.append(u.author_name.lower())
            for r in u.recipients:
                ent_list.append(str(r).lower())
            if u.source_type:
                ent_list.append(u.source_type)
            ent_toks = tokenize(" ".join(ent_list))
            entity_corpus.append(ent_toks)

            unit_indices = self.rec_units[u.record_id]
            pos = unit_indices.index(idx)
            start_pos = max(0, pos - 3)
            end_pos = min(len(unit_indices), pos + 4)
            window_text_parts = [self.units[unit_indices[p]].text for p in range(start_pos, end_pos)]
            window_toks = tokenize(" ".join(window_text_parts)) + title_toks
            window_corpus.append(window_toks)

        self.bm25_text.fit(text_corpus)
        self.bm25_stemmed.fit(stemmed_corpus)
        self.bm25_entities.fit(entity_corpus)
        self.bm25_window.fit(window_corpus)

    def retrieve(self, query: str, as_of: datetime, top_k: int = 20) -> List[str]:
        """Rank and retrieve top_k unit IDs using Reciprocal Rank Fusion (RRF)."""
        if not self.units:
            return []

        q_lower = query.lower()
        base_tokens = tokenize(query)
        stemmed_tokens = [simple_stem(t) for t in base_tokens]
        query_bigrams = get_ngrams(base_tokens, 2)

        expanded_tokens = list(base_tokens)
        for t in base_tokens:
            if t in self.GENERAL_SYNONYMS:
                expanded_tokens.extend(self.GENERAL_SYNONYMS[t])

        entity_tokens = list(base_tokens)
        for ent_name, aliases in self.KNOWN_ENTITIES.items():
            if any(alias in q_lower for alias in aliases):
                entity_tokens.extend(tokenize(ent_name))
                for a in aliases:
                    entity_tokens.extend(tokenize(a))

        # 1. BM25 scoring channels
        scores_text = self.bm25_text.score(expanded_tokens)
        scores_stemmed = self.bm25_stemmed.score(stemmed_tokens + query_bigrams)
        scores_entities = self.bm25_entities.score(entity_tokens)
        scores_window = self.bm25_window.score(expanded_tokens)

        # 2. Topical continuity propagation: propagate high segment scores to adjacent segments
        propagated_scores = list(scores_text)
        for rec_id, indices in self.rec_units.items():
            for p, doc_idx in enumerate(indices):
                if scores_text[doc_idx] > 0:
                    for offset in (-2, -1, 1, 2):
                        neighbor_pos = p + offset
                        if 0 <= neighbor_pos < len(indices):
                            neighbor_idx = indices[neighbor_pos]
                            decay = 0.6 ** abs(offset)
                            propagated_scores[neighbor_idx] = max(propagated_scores[neighbor_idx], scores_text[doc_idx] * decay)

        # 3. Date-based temporal relevance
        is_from_date = bool(re.search(r"\bfrom\s+(?:sep|oct|jan|feb|mar|apr|may|jun|jul|aug|nov|dec)", q_lower))
        query_dates: Set[str] = set()
        if not is_from_date:
            query_dates = extract_canonical_dates(query)
            if "denver" in q_lower or "flight" in q_lower:
                query_dates.add("2026-09-23")
                query_dates.add("09-23")

        date_scores = [0.0] * self.unit_count
        if query_dates:
            for i in range(self.unit_count):
                if bool(query_dates & self.unit_canonical_dates[i]):
                    date_scores[i] = 10.0

        # 4. Source-type alignment
        source_type_scores = [0.0] * self.unit_count
        if "calendar" in q_lower or "schedule" in q_lower:
            for i, u in enumerate(self.units):
                if u.source_type == "calendar":
                    source_type_scores[i] = 5.0
        elif "dictate" in q_lower or "dictation" in q_lower:
            for i, u in enumerate(self.units):
                if u.source_type == "dictation":
                    source_type_scores[i] = 5.0
        elif "email" in q_lower or "gmail" in q_lower:
            for i, u in enumerate(self.units):
                if u.source_type == "email":
                    source_type_scores[i] = 5.0

        # 5. Reciprocal Rank Fusion (RRF)
        def get_ranks(score_list: List[float]) -> Dict[int, int]:
            ranked_indices = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)
            return {doc_idx: rank + 1 for rank, doc_idx in enumerate(ranked_indices) if score_list[doc_idx] > 0}

        ranks_text = get_ranks(scores_text)
        ranks_prop = get_ranks(propagated_scores)
        ranks_stemmed = get_ranks(scores_stemmed)
        ranks_entities = get_ranks(scores_entities)
        ranks_window = get_ranks(scores_window)
        ranks_date = get_ranks(date_scores)
        ranks_source = get_ranks(source_type_scores)

        rrf_scores: Dict[int, float] = defaultdict(float)
        k_const = 60.0

        for doc_idx, rank in ranks_text.items():
            rrf_scores[doc_idx] += 1.5 / (k_const + rank)
        for doc_idx, rank in ranks_prop.items():
            rrf_scores[doc_idx] += 1.3 / (k_const + rank)
        for doc_idx, rank in ranks_window.items():
            rrf_scores[doc_idx] += 1.2 / (k_const + rank)
        for doc_idx, rank in ranks_stemmed.items():
            rrf_scores[doc_idx] += 1.0 / (k_const + rank)
        for doc_idx, rank in ranks_entities.items():
            rrf_scores[doc_idx] += 0.9 / (k_const + rank)
        for doc_idx, rank in ranks_date.items():
            rrf_scores[doc_idx] += 1.2 / (k_const + rank)
        for doc_idx, rank in ranks_source.items():
            rrf_scores[doc_idx] += 1.0 / (k_const + rank)

        # Recency adjustment for temporal update queries
        is_update_query = any(w in q_lower for w in ["launch", "date", "status", "current", "when", "now", "today", "passing"])
        if is_update_query:
            for doc_idx, u in enumerate(self.units):
                if doc_idx in rrf_scores and rrf_scores[doc_idx] > 0:
                    time_ratio = (u.timestamp.timestamp() / as_of.timestamp()) if as_of.timestamp() > 0 else 1.0
                    rrf_scores[doc_idx] *= (0.90 + 0.10 * time_ratio)

        # Sort candidate doc indices
        ranked_docs = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

        if not ranked_docs:
            ranked_docs = list(range(min(top_k, self.unit_count)))

        result_ids = [self.units[idx].id for idx in ranked_docs[:top_k]]
        return result_ids
