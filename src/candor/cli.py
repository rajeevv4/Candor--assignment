"""Command-line entry point.

  python3 run.py                      # = eval: train + dev + holdout + paraphrase sets, all scorers
  python3 run.py memory  -q QUESTIONS.jsonl -o memory_answers.jsonl
  python3 run.py actions -c COMMANDS.jsonl  -o action_predictions.jsonl
  python3 run.py ask "When is Route Planner v2 launching?" --as-of 2026-09-18T18:00:00-07:00
  python3 run.py do  "Remind me an hour before the board meeting to print the deck" --as-of 2026-09-18T09:00:00-07:00
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from candor.actions import ActionPlanner
from candor.models import ActionCommand
from candor.system import MemorySystem

ROOT = Path(__file__).resolve().parents[2]


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str | Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_memory_pipeline(questions_path, output_path, data_dir="data", model_provider: Optional[str] = None,
                        model_name: Optional[str] = None, system: Optional[MemorySystem] = None) -> None:
    system = system or MemorySystem(data_dir, model_provider, model_name)
    rows = [system.ask(x["id"], x["question"], _dt(x["as_of"])).to_dict() for x in _read_jsonl(questions_path)]
    _write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} answers to {output_path}", flush=True)


def run_actions_pipeline(commands_path, output_path, data_dir="data", system: Optional[MemorySystem] = None) -> None:
    system = system or MemorySystem(data_dir)
    planner = ActionPlanner(data_dir, memory=system.snippet)
    rows = [planner.plan_command(ActionCommand(x["id"], x["command"], x["as_of"])).to_dict()
            for x in _read_jsonl(commands_path)]
    _write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} action predictions to {output_path}", flush=True)


def _score(kind: str, gold: str, pred: str, data_dir: str, out: str) -> None:
    harness = ROOT / "eval_harness"
    env = {**os.environ, "PYTHONPATH": str(harness)}
    if kind == "actions":
        cmd = [sys.executable, str(harness / "score_actions.py"), "--gold", gold, "--predictions", pred, "--out", out]
    else:
        cmd = [sys.executable, str(harness / f"score_{kind}.py"), "--gold", gold, "--answers", pred, "--data", data_dir, "--out", out]
        if kind == "memory":
            cmd += ["--judge", "none"]
    subprocess.run(cmd, env=env, check=False)


def run_eval(data_dir: str = "data") -> None:
    system = MemorySystem(data_dir)
    jobs = [("train", "evals/memory_train.jsonl", "evals/actions_train.jsonl", "memory_answers.jsonl", "action_predictions.jsonl"),
            ("dev", "evals/memory_dev.jsonl", "evals/actions_dev.jsonl", "memory_answers_dev.jsonl", "action_predictions_dev.jsonl"),
            ("holdout", "evals/memory_holdout.jsonl", "evals/actions_holdout.jsonl", "memory_answers_holdout.jsonl", "action_predictions_holdout.jsonl"),
            ("paraphrase", "evals/memory_paraphrase.jsonl", None, "memory_answers_paraphrase.jsonl", None)]
    for name, mq, ac, mo, ao in jobs:
        run_memory_pipeline(mq, mo, data_dir, system=system)
        if ac:
            run_actions_pipeline(ac, ao, data_dir, system=system)
    for name, mq, ac, mo, ao in jobs:
        for kind, gold, pred in (("retrieval", mq, mo), ("memory", mq, mo), ("actions", ac, ao)):
            if not gold:
                continue
            print(f"\n{'=' * 64}\n{name.upper()} · {kind}\n{'=' * 64}", flush=True)
            _score(kind, gold, pred, data_dir, f"results_{kind}{'' if name == 'train' else '_' + name}.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Candor workplace memory + action planner")
    sub = p.add_subparsers(dest="cmd")
    m = sub.add_parser("memory", help="answer a JSONL file of memory questions")
    m.add_argument("--questions", "-q", default="evals/memory_train.jsonl")
    m.add_argument("--output", "-o", default="memory_answers.jsonl")
    m.add_argument("--data", "-d", default="data")
    m.add_argument("--provider", default=None, help="deterministic (default) or openai (any OpenAI-compatible endpoint)")
    m.add_argument("--model", default=None)
    a = sub.add_parser("actions", help="plan a JSONL file of commands (dry run)")
    a.add_argument("--commands", "-c", default="evals/actions_train.jsonl")
    a.add_argument("--output", "-o", default="action_predictions.jsonl")
    a.add_argument("--data", "-d", default="data")
    e = sub.add_parser("eval", help="run and score train + dev + holdout + paraphrase sets")
    e.add_argument("--data", "-d", default="data")
    q = sub.add_parser("ask", help="ask one question")
    q.add_argument("question")
    q.add_argument("--as-of", default="2026-09-18T18:00:00-07:00")
    q.add_argument("--data", "-d", default="data")
    d = sub.add_parser("do", help="plan one command (dry run)")
    d.add_argument("command")
    d.add_argument("--as-of", default="2026-09-18T18:00:00-07:00")
    d.add_argument("--data", "-d", default="data")
    args = p.parse_args()

    if args.cmd == "memory":
        run_memory_pipeline(args.questions, args.output, args.data, args.provider, args.model)
    elif args.cmd == "actions":
        run_actions_pipeline(args.commands, args.output, args.data)
    elif args.cmd == "ask":
        ans = MemorySystem(args.data).ask("Q", args.question, _dt(args.as_of))
        print(json.dumps(ans.to_dict(), indent=2))
    elif args.cmd == "do":
        system = MemorySystem(args.data)
        pred = ActionPlanner(args.data, memory=system.snippet).plan_command(ActionCommand("CMD", args.command, args.as_of))
        print(json.dumps(pred.actions, indent=2))
    else:
        run_eval(getattr(args, "data", "data"))


if __name__ == "__main__":
    main()
