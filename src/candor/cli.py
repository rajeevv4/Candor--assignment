"""Command-line interface and entry point for Candor Memory and Action system."""
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
from candor.answering import AnswerEngine
from candor.ingestion import DataIngestion
from candor.models import ActionCommand, MemoryQuery
from candor.retrieval import HybridRetriever
from candor.temporal import TemporalEngine


def run_memory_pipeline(
    questions_path: str | Path,
    output_path: str | Path,
    data_dir: str | Path = "data",
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """Run memory question answering pipeline."""
    q_path = Path(questions_path)
    out_path = Path(output_path)
    d_dir = Path(data_dir)

    print(f"Loading data from {d_dir}...")
    ingestion = DataIngestion(d_dir)
    temporal = TemporalEngine(ingestion.units, ingestion.deleted, ingestion.edits)
    answerer = AnswerEngine(model_provider=model_provider, model_name=model_name)

    print(f"Processing questions from {q_path}...")
    answers = []
    with open(q_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            as_of_dt = datetime.fromisoformat(item["as_of"].replace("Z", "+00:00"))
            visible_units = temporal.get_visible_units(as_of_dt)
            retriever = HybridRetriever(visible_units)

            retrieved_ids = retriever.retrieve(item["question"], as_of_dt, top_k=20)
            vis_map = {u.id: u for u in visible_units}
            retrieved_units = [vis_map[uid] for uid in retrieved_ids if uid in vis_map]

            query = MemoryQuery(
                id=item["id"],
                question=item["question"],
                as_of=item["as_of"],
                category=item.get("category"),
                storyline=item.get("storyline"),
            )
            ans = answerer.answer_query(query, retrieved_units, as_of_dt)
            answers.append(ans.to_dict())

    with open(out_path, "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")

    print(f"Wrote {len(answers)} answers to {out_path}")


def run_actions_pipeline(
    commands_path: str | Path,
    output_path: str | Path,
) -> None:
    """Run action dry-run prediction pipeline."""
    c_path = Path(commands_path)
    out_path = Path(output_path)

    planner = ActionPlanner()
    predictions = []

    print(f"Processing action commands from {c_path}...")
    with open(c_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cmd = ActionCommand(id=item["id"], command=item["command"], as_of=item["as_of"])
            pred = planner.plan_command(cmd)
            predictions.append(pred.to_dict())

    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(predictions)} action predictions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Candor Temporal Workplace Memory & Action System")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Memory parser
    mem_p = subparsers.add_parser("memory", help="Run memory question answering")
    mem_p.add_argument("--questions", "-q", default="evals/memory_train.jsonl", help="Path to input questions JSONL")
    mem_p.add_argument("--output", "-o", default="memory_answers.jsonl", help="Path to output answers JSONL")
    mem_p.add_argument("--data", "-d", default="data", help="Path to data directory")
    mem_p.add_argument("--provider", default=None, help="LLM provider (deterministic, openai, anthropic, gemini)")
    mem_p.add_argument("--model", default=None, help="LLM model name")

    # Actions parser
    act_p = subparsers.add_parser("actions", help="Run action command dry-run planner")
    act_p.add_argument("--commands", "-c", default="evals/actions_train.jsonl", help="Path to input commands JSONL")
    act_p.add_argument("--output", "-o", default="action_predictions.jsonl", help="Path to output predictions JSONL")

    # Full eval parser
    eval_p = subparsers.add_parser("eval", help="Run complete evaluation on train sets")
    eval_p.add_argument("--data", "-d", default="data", help="Path to data directory")

    args = parser.parse_args()

    if args.subcommand == "memory":
        run_memory_pipeline(args.questions, args.output, data_dir=args.data, model_provider=args.provider, model_name=args.model)
    elif args.subcommand == "actions":
        run_actions_pipeline(args.commands, args.output)
    elif args.subcommand == "eval":
        run_memory_pipeline("evals/memory_train.jsonl", "memory_answers.jsonl", data_dir=args.data)
        run_actions_pipeline("evals/actions_train.jsonl", "action_predictions.jsonl")

        env = dict(os.environ)
        env["PYTHONPATH"] = f"eval_harness:{env.get('PYTHONPATH', '')}"

        print("\n" + "=" * 60)
        print("SCORING RETRIEVAL:")
        print("=" * 60)
        subprocess.run(["python3", "eval_harness/score_retrieval.py", "--gold", "evals/memory_train.jsonl", "--answers", "memory_answers.jsonl", "--data", str(args.data)], env=env)

        print("\n" + "=" * 60)
        print("SCORING MEMORY ANSWERS:")
        print("=" * 60)
        subprocess.run(["python3", "eval_harness/score_memory.py", "--gold", "evals/memory_train.jsonl", "--answers", "memory_answers.jsonl", "--data", str(args.data), "--judge", "none"], env=env)

        print("\n" + "=" * 60)
        print("SCORING ACTIONS:")
        print("=" * 60)
        subprocess.run(["python3", "eval_harness/score_actions.py", "--gold", "evals/actions_train.jsonl", "--predictions", "action_predictions.jsonl"], env=env)
    else:
        # Default without subcommand: run both pipelines
        run_memory_pipeline("evals/memory_train.jsonl", "memory_answers.jsonl", data_dir="data")
        run_actions_pipeline("evals/actions_train.jsonl", "action_predictions.jsonl")


if __name__ == "__main__":
    main()
