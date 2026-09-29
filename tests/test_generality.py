"""Guard against question-specific logic: the source code must not mention anything from the data.

Fails if any record id, eval id, person name, Slack id/channel or event title from data/ appears
in src/. The system has to learn all of that from the data at run time.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestNoDataInCode(unittest.TestCase):

    def test_source_has_no_data_specific_strings(self):
        src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "candor").glob("*.py"))
        banned = set()
        # every citable id prefix pattern and eval ids
        for m in re.finditer(r"\b(?:MTG|SL|EM|DCT|CAL|CDX|CGPT|MEM|ACT)-[A-Z0-9][A-Z0-9#-]+", src):
            banned.add(m.group(0))
        users = json.loads((ROOT / "data/connectors/slack/users.json").read_text())
        chans = json.loads((ROOT / "data/connectors/slack/channels.json").read_text())
        names = {u["real_name"] for u in users if u.get("email")} | {u["id"] for u in users} | {c["id"] for c in chans}
        names |= {c["name"] for c in chans if not c["is_dm"] and "-" in c["name"]}
        events = [json.loads(l) for l in (ROOT / "data/connectors/google_calendar/events.jsonl").read_text().splitlines() if l.strip()]
        names |= {e["summary"] for e in events if len(e["summary"]) > 12}
        for n in names:
            if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", src):
                banned.add(n)
        self.assertEqual(sorted(banned), [], "data-specific strings found in src/")


if __name__ == "__main__":
    unittest.main()
