"""Freeze the confirmatory run: hash the plan, code, analysis and chosen write config with a UTC time.

usage: lock.py <results dir>   (writes <results dir>/LOCK.json; refuses once the run has written results)
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out = Path(sys.argv[1])
for done in ("confirm.jsonl", "transfer.jsonl", "edit.jsonl", "scale.jsonl", "bench.jsonl", "confirm_para.jsonl",
             "confirm_para_bench.jsonl", "after_para.jsonl", "after_para_bench.jsonl"):
    if (out / done).exists():
        sys.exit(f"{done} already exists: too late to lock")
files = [ROOT / f for f in ("PLAN.md", "smem.py", "data.py", "run.py", "analyze.py", "analyze_transfer.py",
                               "analyze_edit.py", "analyze_scale.py", "analyze_bench.py",
                               "screen_para.py", "confirm_para.py", "after_para.py")]  # the paragraph confirmation and its follow-up
files += sorted((ROOT / "screen_tasks").glob("*"))  # the harm curve's questions (run 3)
lock = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sha256": {**{str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()},
                   "locked_config.json": hashlib.sha256((out / "locked_config.json").read_bytes()).hexdigest()},
        "locked_config": json.loads((out / "locked_config.json").read_text())}
(out / "LOCK.json").write_text(json.dumps(lock, indent=1))
print(json.dumps(lock, indent=1))
