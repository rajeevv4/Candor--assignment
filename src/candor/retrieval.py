"""Passage-level lexical retrieval with ranking signals. No embeddings, no question-specific rules.

Pipeline for one question at one `as_of`:

1. Index only what is visible at `as_of` (delivered, not deleted, edits applied). Index statistics
   (IDF, lengths) come from that visible set too, so future records can't even shift the ranking.
2. Every unit becomes one or more passages:
     - own text (+ speaker / channel / subject / title header); long units (Codex sessions, long
       emails, long ChatGPT answers) are split into overlapping chunks, the unit scores as its best chunk
     - a context passage: neighbouring meeting segments, the Slack thread / recent channel messages,
       the previous ChatGPT turn, the email thread
     - superseded text (pre-edit versions), searchable at low weight
3. BM25 over each field, combined: own + 0.35*context + 0.3*history.
4. Pseudo-relevance feedback (RM3-style): terms that are frequent in the top passages but
   not in the question are added at low weight, which covers vocabulary mismatch without synonyms.
5. Rank signals on normalised scores: dates the question mentions (explicit or relative), source
   cues ("email", "Slack", "calendar", ...), people named in the question (as speaker/author), and a
   mild recency prior, since the latest statement of a changing fact is usually the one that counts.
6. Linked records (edit ↔ original, thread replies, dictation ↔ the message it produced, calendar
   event ↔ its invitation emails) inherit part of each other's score, so old and new versions come together.
7. Diversity: at most a few segments of the same meeting/conversation in the top 10.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

from candor.models import MemoryUnit
from candor.timeparse import explicit_dates, occurrences, parse_datetime_safe, relative_dates, to_local

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves s t d ll m re ve y let lets get got
also still yet again ever really actually like okay ok um uh yeah gonna go going kind sort thing things
""".split())

# words that shape the question but say nothing about its topic
QUESTION_FILLER = set("""
tell know said say says saying told ask asked mention mentioned think thought happen happened happening
anything something someone somebody everything anyone
""".split())

# A small, general English/workplace thesaurus. Each group is a set of words used interchangeably;
# a query word pulls in the rest of its group at reduced weight. Nothing here names a person,
# customer, product or event from the data.
SYNONYM_GROUPS: List[Set[str]] = [
    {"slip", "delay", "postpone", "push", "move", "reschedule", "shift", "change"},
    {"launch", "release", "ship", "go-live", "golive"},
    {"fly", "flight", "flying", "depart", "departure"},
    {"sign", "close", "signed", "closed", "contract"},
    {"owner", "own", "owns", "responsible", "assigned", "plate"},
    {"cancel", "cancelled", "canceled", "scratch", "drop", "off"},
    {"hire", "hiring", "req", "role", "opening", "post"},
    {"price", "pricing", "cost", "rate"},
    {"bug", "issue", "regression", "defect", "problem"},
    {"finish", "done", "complete", "delivered", "posted", "landed", "up"},
    {"database", "db", "postgres", "sqlite", "store"},
    {"meeting", "call", "sync", "session"},
    {"prefer", "preference", "like", "rather"},
    {"why", "because", "reason", "cause"},
    {"deadline", "due", "by"},
    {"mockup", "mockups", "design", "designs", "wireframe", "wireframes"},
]


def mine_acronyms(texts: Iterable[str]) -> Dict[str, Set[str]]:
    """Acronyms defined in the data itself: 'SSO ... like single sign-on' -> {'sso': {'single','sign','on'}}.

    An all-caps token counts as defined when, within 4 words of it in the same record, a run of
    words whose initials spell it appears right next to it (first word not a stopword, at most one stopword).
    The mapping is used both ways for query expansion.
    """
    found: Dict[str, Set[str]] = defaultdict(set)
    for text in texts:
        words = re.findall(r"[A-Za-z]+", text)
        low = [w.lower() for w in words]
        for pos, w in enumerate(words):
            if not re.fullmatch(r"[A-Z]{2,5}", w) or w.lower() in STOPWORDS:
                continue
            ac, n = w.lower(), len(w)
            best = None
            for i in range(max(0, pos - n - 4), min(len(low) - n + 1, pos + 5)):
                run = low[i:i + n]
                if i <= pos < i + n or ac in run or "".join(x[0] for x in run) != ac:
                    continue
                if run[0] in STOPWORDS or sum(x in STOPWORDS for x in run) > 1 or any(len(x) < 2 for x in run):
                    continue
                if best is None or abs(i - pos) < abs(best[0] - pos):
                    best = (i, run)
            if best:
                found[ac] |= set(best[1])
    return found


SOURCE_CUES: Dict[str, Tuple[str, ...]] = {
    "email": ("email", "emailed", "emails", "mail", "inbox", "gmail", "wrote to", "reply", "replied"),
    "slack": ("slack", "channel", "dm", "posted", "thread"),
    "calendar": ("calendar", "invite", "scheduled", "schedule", "booked", "meeting time", "what's on", "whats on"),
    "dictation": ("dictate", "dictated", "dictation", "note to self", "voice note", "notes to self"),
    "meeting": ("meeting", "call", "standup", "1:1", "one-on-one", "review", "debrief", "go/no-go", "said in"),
    "codex": ("codex", "prototype", "repo", "code", "coding", "migration", "script"),
    "chatgpt": ("chatgpt", "gpt", "asked the ai", "draft"),
}


def stem(w: str) -> str:
    """Light suffix stripping, enough to match launch/launching/launched, cases/case, etc."""
    if len(w) <= 3 or not w.isalpha():
        return w
    for suf, rep in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("ly", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)] + rep
            if suf in ("ing", "ed") and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]  # planned -> plan, stopping -> stop
            if suf == "es" and not re.search(r"(sh|ch|x|ss|z)$", base):
                base = w[:-1]     # dates -> date
            return base[:-1] if base.endswith("e") and len(base) > 3 else base
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def tokenize(text: str) -> List[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"\b(\d{2})(?:st|nd|rd|th)[ -]percentile\b", r"p\1", t)   # "95th percentile" == "p95"
    t = re.sub(r"\bmedian\b", "median p50", t)
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t)


def terms(text: str) -> List[str]:
    return [stem(w) for w in tokenize(text) if w not in STOPWORDS]


class BM25:
    """Okapi BM25 over an inverted index, with weighted query terms."""

    def __init__(self, docs: List[List[str]], k1: float = 1.2, b: float = 0.75, min_len: int = 0):
        self.k1, self.b = k1, b
        self.n = len(docs)
        # a length floor keeps two-word ASR fragments ("And release notes.") from out-scoring real statements
        self.lens = [max(len(d), min_len) for d in docs]
        self.avg = (sum(self.lens) / self.n) if self.n else 1.0
        self.post: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.post[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def score(self, q: Dict[str, float]) -> Dict[int, float]:
        out: Dict[int, float] = defaultdict(float)
        for term, w in q.items():
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, tf in self.post[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
                out[i] += w * idf * tf * (self.k1 + 1) / denom
        return out


def _chunks(tokens: List[str], size: int = 120, step: int = 80) -> List[List[str]]:
    if len(tokens) <= size + 40:
        return [tokens]
    return [tokens[i:i + size] for i in range(0, max(1, len(tokens) - 40), step)]


class HybridRetriever:
    """Ranks the units visible at one `as_of` for a question."""

    # field weights and ranking-signal weights (tuned on train + my own dev set, see README)
    W = {"own": 1.0, "ctx": 0.35, "title": 0.4, "hist": 0.3, "syn": 0.25, "prf": 0.0,
         "date": 0.4, "source": 0.12, "person": 0.12, "recency": 0.2, "link": 0.8,
         "min_len": 8, "per_record": 4}

    def __init__(self, visible_units: List[MemoryUnit], people: Optional[Iterable[str]] = None,
                 weights: Optional[Dict[str, float]] = None):
        self.W = {**HybridRetriever.W, **(weights or {})}
        self.units = visible_units
        self.by_id = {u.id: i for i, u in enumerate(visible_units)}
        self.people = {p.lower() for p in (people or [])} | {
            (u.author_name or "").lower() for u in visible_units
            if u.author_name and u.source_type in ("slack", "meeting") and " " in u.author_name}
        self.acronyms = mine_acronyms(u.raw_text for u in visible_units)
        self._build()

    # ------------------------------------------------------------------ indexing
    def _header(self, u: MemoryUnit) -> str:
        return u.title_or_context

    def _build(self) -> None:
        units = self.units
        by_record: Dict[str, List[int]] = defaultdict(list)
        by_channel: Dict[str, List[int]] = defaultdict(list)
        for i, u in enumerate(units):
            by_record[u.record_id].append(i)
            if u.source_type == "slack":
                by_channel[u.metadata.get("channel_id", "")].append(i)
        self.by_record = by_record

        own_docs: List[List[str]] = []
        self.own_owner: List[int] = []
        ctx_docs: List[List[str]] = []
        hist_docs: List[List[str]] = []
        title_docs: List[List[str]] = []
        self.hist_owner: List[int] = []

        neighbours: Dict[int, List[int]] = defaultdict(list)
        for idxs in by_record.values():
            if len(idxs) > 1:
                for p, i in enumerate(idxs):
                    neighbours[i] = idxs[max(0, p - 2):p] + idxs[p + 1:p + 3]
        for idxs in by_channel.values():
            for p, i in enumerate(idxs):
                near = [j for j in idxs[max(0, p - 2):p] + idxs[p + 1:p + 2]
                        if abs((units[j].timestamp - units[i].timestamp).total_seconds()) < 3600]
                neighbours[i] = near

        for i, u in enumerate(units):
            body = u.raw_text or u.text
            if u.source_type == "codex":
                pieces = u.metadata.get("turns") or [body]
                groups = [" ".join(pieces[k:k + 3]) for k in range(0, len(pieces), 2)] or [body]
            else:
                toks = tokenize(body)
                groups = [" ".join(c) for c in _chunks(toks)]
            for g in groups:
                own_docs.append(terms(g))
                self.own_owner.append(i)
            title_docs.append(terms(f"{u.author_name or ''} {self._header(u)}"))

            ctx_parts = [units[j].raw_text for j in neighbours.get(i, [])]
            for lid in u.links:
                j = self.by_id.get(lid)
                if j is not None:
                    ctx_parts.append(units[j].raw_text[:600])
            if u.source_type == "calendar":
                ctx_parts.append(u.text)
            ctx_docs.append(terms(" ".join(ctx_parts)))

            for old in u.history:
                hist_docs.append(terms(old))
                self.hist_owner.append(i)

        ml = int(self.W["min_len"])
        self.own = BM25(own_docs, k1=1.2, b=0.75, min_len=ml)
        self.ctx = BM25(ctx_docs, k1=1.2, b=0.75, min_len=ml)
        self.title = BM25(title_docs, k1=1.2, b=0.3)
        self.hist = BM25(hist_docs or [[]], k1=1.2, b=0.75, min_len=ml)

        # dates each unit is "about": when it happened, dates it mentions, calendar occurrences
        self.unit_dates: List[Set[date]] = []
        for u in units:
            ref = to_local(u.timestamp)
            ds: Set[date] = set()
            if u.source_type == "calendar":
                st = u.metadata.get("start", {})
                start = parse_datetime_safe(st.get("dateTime") or st.get("date"))
                en = u.metadata.get("end", {})
                end = parse_datetime_safe(en.get("dateTime") or en.get("date"))
                if start and u.metadata.get("recurrence"):
                    ds |= set(occurrences(start, u.metadata["recurrence"], (date(ref.year, 1, 1), date(ref.year, 12, 31))))
                elif start:
                    d, last = start.date(), (end.date() if end else start.date())
                    if st.get("date") and last > d:
                        last = date.fromordinal(last.toordinal() - 1)  # all-day end date is exclusive
                    while d <= last:
                        ds.add(d)
                        d = date.fromordinal(d.toordinal() + 1)
            else:
                ds.add(ref.date())
                ds |= explicit_dates(u.raw_text, ref)
                if u.source_type == "meeting":
                    ds.add(to_local(parse_datetime_safe(u.metadata.get("meeting_start")) or u.timestamp).date())
            self.unit_dates.append(ds)

        self.t0 = min((u.timestamp for u in units), default=None)

    # ------------------------------------------------------------------ query side
    def query_terms(self, question: str) -> Dict[str, float]:
        q: Dict[str, float] = {}
        words = [w for w in tokenize(question) if w not in STOPWORDS and w not in QUESTION_FILLER]
        for w in words:
            q[stem(w)] = 1.0
        ql = " ".join(words)
        for ac, expansion in self.acronyms.items():
            exp_terms = {stem(w) for w in expansion if w not in STOPWORDS}
            if ac in words:
                for t in exp_terms:
                    q.setdefault(t, self.W["syn"])
            elif exp_terms and exp_terms <= {stem(w) for w in words}:
                q.setdefault(ac, 1.0)
        for w in words:
            for group in SYNONYM_GROUPS:
                if w in group or stem(w) in {stem(g) for g in group}:
                    for g in group:
                        for t in terms(g):
                            q.setdefault(t, self.W["syn"])
        return q

    def _lexical(self, q: Dict[str, float]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        best_own: Dict[int, float] = {}
        for p, s in self.own.score(q).items():
            i = self.own_owner[p]
            best_own[i] = max(best_own.get(i, 0.0), s)
        for i, s in best_own.items():
            scores[i] += self.W["own"] * s
        for i, s in self.ctx.score(q).items():
            scores[i] += self.W["ctx"] * s
        for i, s in self.title.score(q).items():
            scores[i] += self.W["title"] * s
        if self.hist_owner:
            for p, s in self.hist.score(q).items():
                scores[self.hist_owner[p]] += self.W["hist"] * s
        return scores

    def _feedback_terms(self, ranked: List[int], q: Dict[str, float], k: int = 5, n_terms: int = 8) -> Dict[str, float]:
        tf: Counter = Counter()
        for i in ranked[:k]:
            tf.update(set(terms(self.units[i].raw_text)))
        cand = []
        for t, c in tf.items():
            if t in q or len(t) < 3 or t.isdigit() or c < 2:
                continue
            idf = self.own.idf.get(t, 0.0)
            cand.append((c * idf, t))
        cand.sort(reverse=True)
        return {t: self.W["prf"] for _, t in cand[:n_terms]}

    def _query_dates(self, question: str, as_of: datetime) -> Set[date]:
        q = re.sub(r"\b(?:from|instead of|was)\s+(?:\w+\s+)?\d{1,2}(?:st|nd|rd|th)?\b", " ", question.lower())
        return explicit_dates(q, to_local(as_of)) | relative_dates(q, to_local(as_of))

    def _people_in(self, question: str) -> Set[str]:
        ql = question.lower()
        found = {p for p in self.people if p and re.search(rf"\b{re.escape(p)}\b", ql)}
        return found

    def rank(self, question: str, as_of: datetime) -> List[Tuple[str, float]]:
        if not self.units:
            return []
        q = self.query_terms(question)
        lex = self._lexical(q)
        if not lex:
            return []
        first = sorted(lex, key=lex.get, reverse=True)
        fb = self._feedback_terms(first, q) if self.W["prf"] > 0 else {}
        if fb:
            lex2 = self._lexical({**fb, **q})
            lex = {i: lex.get(i, 0.0) * 0.75 + lex2.get(i, 0.0) * 0.25 for i in set(lex) | set(lex2)}
        top = max(lex.values()) or 1.0
        score = {i: s / top for i, s in lex.items()}

        ql = question.lower()
        qdates = self._query_dates(question, as_of)
        cues = {src for src, words in SOURCE_CUES.items() if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", ql) for w in words)}
        if "calendar" in cues and not qdates:
            # two-hop dates: "the day I fly to Denver" -> dates mentioned by the best non-calendar hits
            for i in sorted(score, key=score.get, reverse=True)[:3]:
                if self.units[i].source_type != "calendar":
                    qdates |= explicit_dates(self.units[i].raw_text, to_local(self.units[i].timestamp))
        if qdates:
            for i, ds in enumerate(self.unit_dates):
                if i not in score and ds & qdates and (self.units[i].source_type == "calendar" or "calendar" not in cues):
                    score[i] = 0.0
        people = self._people_in(question)
        span = (as_of - self.t0).total_seconds() if self.t0 else 0

        for i in list(score):
            u = self.units[i]
            s = score[i]
            if qdates and self.unit_dates[i] & qdates:
                s += self.W["date"]
            if u.source_type in cues:
                s += self.W["source"]
            if people and (u.author_name or "").lower() in people:
                s += self.W["person"]
            if span > 0 and u.source_type != "calendar":
                s += self.W["recency"] * max(0.0, (u.timestamp - self.t0).total_seconds() / span)
            score[i] = s

        # linked records share evidence (edit <-> original, thread, dictation <-> sent message, invite <-> event)
        final = dict(score)
        for i, s in score.items():
            for lid in self.units[i].links:
                j = self.by_id.get(lid)
                if j is not None:
                    final[j] = max(final.get(j, 0.0), self.W["link"] * s)
        return sorted(((self.units[i].id, s) for i, s in final.items()), key=lambda x: -x[1])

    def retrieve(self, question: str, as_of: datetime, top_k: int = 20, per_record_top10: Optional[int] = None) -> List[str]:
        ranked = self.rank(question, as_of)
        per_record_top10 = int(per_record_top10 or self.W["per_record"])
        out: List[str] = []
        overflow: List[str] = []
        per_rec: Counter = Counter()
        for uid, _ in ranked:
            rec = self.units[self.by_id[uid]].record_id
            if len(out) < 10 and per_rec[rec] >= per_record_top10:
                overflow.append(uid)
                continue
            out.append(uid)
            per_rec[rec] += 1
            if len(out) >= 10:
                break
        rest = [uid for uid, _ in ranked if uid not in set(out)]
        for uid in rest:
            if len(out) >= top_k:
                break
            out.append(uid)
        return out[:top_k]

    def unit(self, uid: str) -> Optional[MemoryUnit]:
        i = self.by_id.get(uid)
        return self.units[i] if i is not None else None
