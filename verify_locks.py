"""Check every file a pre-registration lock recorded against its sha256 in LOCK.json.

Each LOCK.json was written before the run it covers produced any data. The plan the locks also cover
(PLAN.md) is not published here yet, so its line reads "not included".

usage: python verify_locks.py
"""
import hashlib
import json
from pathlib import Path

ok, missing = True, 0
for lock_path in sorted(Path(".").glob("results_*/LOCK.json")):
    run = lock_path.parent
    lock = json.loads(lock_path.read_text())
    for name, want in lock["sha256"].items():
        path = run / name if name == "locked_config.json" else Path(name)
        if not path.exists():
            missing += 1
            print(f"{'not included':12s} {run.name} (locked {lock['utc']}): {path}")
            continue
        same = hashlib.sha256(path.read_bytes()).hexdigest() == want
        ok &= same
        print(f"{'ok' if same else 'DIFFERENT':12s} {run.name} (locked {lock['utc']}): {path}")
print(("every included locked file matches" if ok else "some locked files differ") + f"; {missing} not included")
