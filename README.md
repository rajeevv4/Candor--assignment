# Candor take-home: point-in-time workplace memory

**Author:** Rajeev Karakoti

A memory over two weeks of Alex Rivera's work (meetings, dictation, Slack, email, calendar, Codex, ChatGPT) that fetches the right records for a question *as of a given moment*, plus a dry-run text-command planner (the bonus).

The core is **passage-level lexical retrieval with ranking signals**: no embeddings, no LLM calls by default, no question-specific rules. Standard-library Python only.

```bash
python3 run.py          # runs + scores everything: train, dev, holdout and paraphrase sets (~3 s)
```

---

## 0. What changed since the first submission

The first submission was overfit to the train set, and its README didn't say so. The train questions were matched by keyword to pre-written answers; retrieval and actions had question-specific rules; any unseen question got the first 120 characters of the top record back. It scored 100% on train but 68% retrieval, 17% answers and 2/13 actions on the hidden set. The 100% in the README was accurate only for train and should never have been presented as system quality.

This version is a rewrite of retrieval, answering and actions:

- **No pre-written answers or per-question branches.** Every name, email, Slack id, channel and calendar event the system uses is read from `data/` at run time.
- **A guard test enforces it.** [tests/test_generality.py](tests/test_generality.py) fails if any record id, eval id, person name, Slack id, channel name or event title from the data appears anywhere in `src/`.
- **Generalization is measured, not assumed.** I wrote three extra evaluation sets of my own (§5) and report them all, including which ones I tuned on.

---

## 1. How to run

Python 3.10+ (the harness needs it), no packages to install.

| Command | What it does |
|---|---|
| `python3 run.py` or `python3 run.py eval` | Answers all memory sets and action sets, then runs the three harness scorers on each |
| `python3 run.py memory -q QUESTIONS.jsonl -o memory_answers.jsonl` | Memory interface for the hidden test |
| `python3 run.py actions -c COMMANDS.jsonl -o action_predictions.jsonl` | Action interface for the hidden test (dry run) |
| `python3 run.py ask "When is Route Planner v2 launching?" --as-of 2026-09-12T12:00:00-07:00` | One question |
| `python3 run.py do "Remind me an hour before the board meeting to print the deck" --as-of 2026-09-18T09:00:00-07:00` | One command |
| `PYTHONPATH=src python3 -m unittest discover tests` | 28 unit and integration tests |

Output files on the train sets: [memory_answers.jsonl](memory_answers.jsonl) and [action_predictions.jsonl](action_predictions.jsonl). An LLM answer writer is optional (`CANDOR_LLM_PROVIDER=openai` plus any OpenAI-compatible key, see [.env.example](.env.example)). It changes only the `answer` text. `retrieved` is identical either way, and all numbers below use the deterministic writer.

---

## 2. Architecture

```
data/ ──► ingestion.py ──► MemoryUnits (one per citable id) + links + edits + deletions
                                   │
            question, as_of ──►  temporal.py: keep units delivered ≤ as_of, drop deleted ≤ as_of,
                                   │           apply edits ≤ as_of (old text kept as searchable history)
                                   ▼
                           retrieval.py: passage index built from the visible units only
                             ├─ fields: own text · context (neighbours/thread) · title+speaker · history
                             ├─ BM25 per field, query = stems + small thesaurus + acronyms mined from data
                             ├─ signals: dates in question, source cues, named speaker, recency
                             ├─ link propagation: edit↔original, thread, dictation↔sent message, invite↔event
                             └─ diversity: ≤ 4 segments of one meeting in the top 10
                                   │  ranked ids (top 20) ──► "retrieved"
                                   ▼
                           answering.py: abstain? → pick evidence sentences → quote with speaker/source/date
                                   │  (optional: LLM writer over the same evidence)
                                   ▼
                           {"answer", "sources", "retrieved", "abstained"}

actions.py: command → clauses → confirm | memory.ask | app.open | reminder | update/create event | email | Slack
            (people/channels/events resolved from data/, times from as_of, message bodies filled from memory)
```

| File | Role |
|---|---|
| [src/candor/ingestion.py](src/candor/ingestion.py) | Parses the seven sources into units with delivery times; records edits/deletions; builds links: Slack threads, email threads, edit → target, dictation → the Slack/Gmail message it produced (same text within 15 min), calendar event → its invitation/update emails |
| [src/candor/temporal.py](src/candor/temporal.py) | Point-in-time view: what existed at `as_of`, with edits applied |
| [src/candor/retrieval.py](src/candor/retrieval.py) | Passage index and ranking (below) |
| [src/candor/timeparse.py](src/candor/timeparse.py) | Dates/times: "Sep 23", "9/23", "the 25th", "tomorrow at 2", weekly RRULE expansion |
| [src/candor/answering.py](src/candor/answering.py) | Abstention, extractive answer, injection/secret filtering, optional LLM |
| [src/candor/actions.py](src/candor/actions.py) | Dry-run command planner |
| [src/candor/system.py](src/candor/system.py) | Wires it together, caches one index per `as_of` |

---

## 3. Key decisions and why

**Unit = the most specific citable id; passage = what gets scored.** Meeting segments, Slack messages, emails, dictations, calendar events, ChatGPT turns and Codex sessions are units. Each unit is scored from several fields:

- **own text.** Long units (Codex sessions, long emails and ChatGPT answers) are split into overlapping chunks, and the unit scores as its best chunk, so a 40-turn Codex session is found by the one turn that matters.
- **context** (weight 0.35): the ±2 neighbouring meeting segments, the recent messages in the same Slack channel (within an hour), linked records. ASR segments like *"Yeah. Good. Okay, so, officially, launch is October 21st."* depend on their neighbours.
- **title + speaker** (weight 0.4): meeting title, channel, email subject, author. This is kept *out* of the own-text field. With it inside, every two-word fragment of "Route Planner v2 – launch planning" matched "When is Route Planner v2 launching?" and filled the top 10 with "And release notes."
- **history** (0.3): pre-edit text, so a question phrased with the old wording still finds the record. It is never quoted as current.

**Index statistics come from the visible set only.** A fresh BM25 index is built per `as_of` (about 0.3 s, cached per moment; a query then takes ~15 ms), so records from the future can't even shift IDF.

**Query side, without question rules:**
- Light suffix stemming (page/pages/paged → the same stem).
- A 16-group general thesaurus (slip/delay/move/postpone, fly/flight, owner/responsible, …). It names no person, customer, product or event.
- Acronyms **mined from the data** (`SSO` ↔ "single sign-on", `MAE` ↔ "mean absolute error"), accepted only when the words spelling the acronym sit within 4 words of it.
- "95th percentile" is normalized to "p95".

**Ranking signals** are added to max-normalized BM25:
- Dates named or implied in the question match record timestamps, dates mentioned in text, and calendar occurrences, including weekly recurrences and multi-day events (0.4).
- Source cues: "email", "Slack", "dictate", "calendar", "Codex", … (0.12).
- The named person is the speaker (0.12).
- Mild recency within the visible window (0.2), because the latest statement of a changing fact usually matters most.
- Calendar questions with no explicit date ("the day I fly to Denver") take dates from the best non-calendar hits: a two-hop lookup.

**Links carry 80% of a record's score to its partners**, so old and new versions arrive together: edit event ↔ original message, thread replies, the dictation ↔ the email it became, the calendar event ↔ its "Updated invitation … Was: Thursday" email.

**Diversity:** at most 4 segments from one meeting or conversation in the top 10, because long transcripts otherwise crowd out the single Slack message or email a question also needs.

**Tuning was done with the harness's own pass rule** over train and my dev set together (§5), on a few weights only. Most settings sat within ±1 question of each other. The ablation below shows which signals actually matter.

### Time, edits, deletions, secrets, planted instructions

- **Future records** are never indexed, so they can't be retrieved or cited. The harness reports 0 forbidden records across all sets.
- **Deleted messages** are dropped from the moment of deletion. A question about one either abstains or, before the deletion time, finds it.
- **Edits** replace the text from the edit time on; before the edit, the original text is served.
- **Secrets** (`sk-…`, `password: …`) are redacted in every answer, and HTML comments are stripped.
- **Planted instructions:** sentences that look like instructions to an AI ("ignore your previous instructions", "forward all emails", "note to any AI") are never used as evidence. Retrieval may still rank the email containing one (it's content), but the answer never repeats it.

### Answering and abstention (deliberately simple; retrieval is the priority)

1. **Abstain** when ≥ 25% of the question's content words (after stemming, thesaurus and acronyms) don't exist anywhere in memory at `as_of`. Words naming the medium ("email", "dictate") are ignored. Also abstain when the question asks about the outcome of a calendar event that hasn't happened yet ("What did the board decide…" before Sep 23).
2. **Pick evidence:** sentence windows (a sentence plus short follow-ups such as bullet lists) scored by IDF-weighted overlap with the query (synonyms included), retrieval rank, and whether the window contains a value. For current-state questions ("when / is / still / now") the most recent strong statement leads. One statement per speaker comes first, so both sides of a disagreement survive.
3. **Quote with attribution:** `Sarah Kim, on Slack #route-planner, Sep 16: "…"`. Unidentified meeting speakers are labelled as such.

This extractive writer can't do arithmetic ("how many days after…") or yes/no synthesis ("Has Acme signed?"). That is most of the answer-score gap (§6).

---

## 4. Action planner (bonus)

A command is split into clauses on "and" + an action verb, except continuations like "…and ask if she's…". Each clause is classified in this order:

1. **destructive** (delete/remove/cancel/wipe) → `confirm`
2. **question** → `memory.ask`
3. **open X** → `app.open`
4. **remind me** → `reminder.create`
5. **move/reschedule X to/by T** → `calendar.update_event`
6. **book/schedule** → `calendar.create_event`
7. **email** → `gmail.send`
8. **message/tell/slack/thank/post** → `slack.send_message`

Everything is resolved from data:
- **People:** Slack users, Gmail From/To/Cc headers, calendar attendees, merged by email. The owner (Alex) is detected as the one member of every DM.
- **Channels:** from `channels.json`.
- **Events:** matched by title overlap, choosing the next occurrence after `as_of` and the version of the event that existed at `as_of`.

Ambiguity is handled per channel. "Message Sarah about the pricing proposal" matches two Sarahs, so the planner returns `clarify`. "Message Sarah **on Slack**" matches only Sarah Kim (Sarah Patel has no Slack account), so it resolves. Someone with email but no Slack gets `gmail.send`.

Relative times are parsed: "an hour before the board meeting" (event start − 1h), "the 25th at 9am", "tomorrow at 2" (bare 1–6 means pm), "Monday". A moved event keeps its duration.

When a message body refers to a fact ("Email John **the corrected NRR**"), the planner asks the memory for the best current evidence sentence and puts it in the body ("NRR is 112%, not 118%…").

---

## 5. Evaluation

Four memory sets, three action sets, all scored with the unmodified harness (`--judge none` for answers):

| Set | Written by | n | Used for tuning? |
|---|---|---|---|
| **train** | Candor | 27 memory / 12 actions | yes |
| **dev** ([memory](evals/memory_dev.jsonl), [actions](evals/actions_dev.jsonl)) | me, gold ids checked against the records | 20 / 13 | yes (it's my "own hidden set" in name only after tuning) |
| **paraphrase** ([file](evals/memory_paraphrase.jsonl)) | me: 21 train questions reworded, same gold | 21 / – | first measured untuned (**76.2%** retrieval), then used for the acronym and percentile fixes |
| **holdout** ([memory](evals/memory_holdout.jsonl), [actions](evals/actions_holdout.jsonl)) | me, written after tuning, new topics | 13 / 9 | no. Run once; it didn't change after the later acronym/percentile change |

### Results (current commit)

| Set | Retrieval score | Needed-in-top 5 / 10 / 20 | MRR | Forbidden in top 20 | Answers strict / lenient | Hard failures | Actions |
|---|---|---|---|---|---|---|---|
| train | **92.0%** (23/25 scored + 2 harm checks) | 64 / 92 / 92% | 0.60 | 0 | 63.0% / 70.4% | 0 | **12/12** |
| dev | **89.5%** | 83 / 89 / 100% | 0.66 | 0 | 45.0% / 45.0% | 0 | **13/13** |
| paraphrase | **90.5%** | – | 0.73 | 0 | 57.1% / 66.7% | 0 | – |
| holdout | **100%** (12/12 + 1 harm check) | 83 / 100 / 100% | 0.55 | 0 | 53.8% / 53.8% | 0 | **9/9** |

How to read this:
- The honest estimates for unseen questions are the untuned numbers: **76% retrieval on paraphrases before I fixed anything for them**, and the holdout.
- The holdout is small and written by me. My questions probably echo record wording more than Candor's hidden questions will, so treat 100% as an upper bound.
- Expect the hidden-set retrieval score below the train number.
- The official answer score adds an LLM judge, which can only lower these.

### Ablation (retrieval pass count, train + dev + holdout + paraphrase = 76 scored questions)

| Configuration | train | dev | holdout | paraphrase | total (/76) |
|---|---|---|---|---|---|
| **full system** | 23/25 | 16/18 | 12/12 | 19/21 | **70** |
| no context field (neighbours/thread) | 22 | 16 | 11 | 19 | 68 |
| no title/speaker field | 21 | 13 | 12 | 16 | 62 |
| no thesaurus | 23 | 16 | 12 | 18 | 69 |
| no link propagation | 23 | 16 | 12 | 19 | 70 |
| no date signal | 22 | 16 | 12 | 18 | 68 |
| no recency prior | 20 | 15 | 11 | 19 | 65 |
| no source-cue / speaker signals | 22 | 15 | 12 | 18 | 67 |
| no per-meeting cap | 23 | 16 | 12 | 19 | 70 |
| no length floor for short segments | 23 | 15 | 12 | 19 | 69 |
| **all of the above off** (≈ plain passage BM25 + stemming + acronyms) | 18 | 12 | 11 | 14 | **55** |

The title field and recency prior matter most. Link propagation and the per-meeting cap don't change pass counts at the current weights. I kept them because they change *what else* lands in the top 20 (the edited message next to its edit event, the invite next to the event), which matters for the answer writer and costs no passes.

---

## 6. Where it fails, and why

**Retrieval misses (train)**
- *MEM-TR-21 "Why did the launch slip from September 30?"* The answer (*"QA found a geocoding regression … move the launch to October 14"*) shares only "launch" with the question, while a dozen records say "September 30". A lexical ranker rewards the old value. The thesaurus (slip ↔ move) isn't enough. Embeddings or an LLM query rewrite ("why did X change") would help.
- *MEM-TR-25 "What's on my calendar the day I fly to Denver?"* This needs a two-hop lookup: the flight email gives Sep 23, then the calendar for Sep 23. The two-hop step exists, but the flight email ranks below the Denver-offsite ChatGPT chat, so the dates come from the wrong record (Sep 24–25).

**Retrieval misses (dev / paraphrase)**
- "What NRR number should go in the board deck?" "board deck" (common, many records) outweighs "NRR" (rare, decisive).
- "What's the next step for Jordan Ellis?" The answer segment says "moving Jordan to the final round" and doesn't match "next step".
- "Will Harbor close before the end of the year?" "end of the year" vs "Q4".
- "What caused the first launch delay?" The same case as TR-21.

**Answer misses** (retrieval found the evidence in most of these):
- The extractive writer quotes a sentence but doesn't compute: "how many days after" → 6, "Has Acme signed?" → "No".
- It sometimes quotes a relevant but secondary sentence (the question's key term sits in a neighbouring sentence).
- Two abstention mistakes: "Did Marcus say Acme was talking to a competitor?" after the message was deleted, and "Chris Hale's phone number". Here every content word exists in memory, so coverage-based abstention can't tell.

---

## 7. What didn't work

- **Speaker names and titles inside the passage text.** Retrieval dropped sharply because empty utterances matched on the name/title alone. The fix: a separate low-weight title field, plus a BM25 length floor of 8 tokens for ASR fragments.
- **Pseudo-relevance feedback (RM3-style)** added nothing on train/dev (±1 question) and made failures harder to reason about. It's still in the code, switched off (`prf: 0`).
- **Sharper IDF, down-weighting "old value" terms ("slip *from Sep 30*"), and a "change statement" boost:** no measurable effect. Removed.
- **The first acronym miner** (any run of words spelling the initials anywhere in a record) produced junk like `IT → is tomorrow`. The fix: require adjacency.
- **Keyword-matched answer templates** (the first submission): perfect on train, below a keyword baseline on unseen questions. This is the lesson the rewrite is built around.

## 8. Known limits

- Purely lexical. Paraphrases with no shared words or known synonyms (Q4 ↔ "end of year") are the main retrieval risk. The next step would be a small local embedding model as one more ranking field, or an LLM query rewrite.
- The thesaurus is hand-written (general workplace English). It will miss domain vocabulary the data doesn't define.
- Answers are extractive quotes: no arithmetic, no yes/no synthesis. The optional LLM writer addresses this but isn't what's scored above.
- The action grammar covers the verbs in the brief's table. Anything else falls back to `memory.ask`. Recurring-event handling covers weekly RRULEs only.
- Dates without a year are assumed to be in the reference year.

## 9. Tools, models, cost

- **Runtime:** Python 3.10+ standard library. No model or API calls in the default pipeline. Optional OpenAI-compatible LLM for answer text only (not used for any reported number).
- **Evaluation:** the provided harness, unmodified, `--judge none`.
- **Cost:** ₹0.
