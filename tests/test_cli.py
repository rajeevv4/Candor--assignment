"""Integration test for end-to-end command line execution."""
import json
import subprocess
import unittest
from pathlib import Path


class TestCLI(unittest.TestCase):

    def test_cli_end_to_end(self):
        tmp_mem_out = "tmp_test_mem.jsonl"
        tmp_act_out = "tmp_test_act.jsonl"

        try:
            # Run memory
            res_mem = subprocess.run(
                ["python3", "run.py", "memory", "-q", "evals/memory_train.jsonl", "-o", tmp_mem_out],
                capture_output=True,
                text=True
            )
            self.assertEqual(res_mem.returncode, 0)
            self.assertTrue(Path(tmp_mem_out).exists())

            with open(tmp_mem_out) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(lines), 27)

            # Run actions
            res_act = subprocess.run(
                ["python3", "run.py", "actions", "-c", "evals/actions_train.jsonl", "-o", tmp_act_out],
                capture_output=True,
                text=True
            )
            self.assertEqual(res_act.returncode, 0)
            self.assertTrue(Path(tmp_act_out).exists())

            with open(tmp_act_out) as f:
                act_lines = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(act_lines), 12)

        finally:
            if Path(tmp_mem_out).exists():
                Path(tmp_mem_out).unlink()
            if Path(tmp_act_out).exists():
                Path(tmp_act_out).unlink()


if __name__ == "__main__":
    unittest.main()
