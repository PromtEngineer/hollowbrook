"""Run every lab in --quick mode (or --full) and report pass/fail and wall-clock.

    python3 labs/run_all_labs.py            # quick
    python3 labs/run_all_labs.py --full     # the real runs (an hour or so on a laptop)
    python3 labs/run_all_labs.py --only 10 19
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

LABS_DIR = os.path.dirname(os.path.abspath(__file__))
COURSE_DIR = os.path.dirname(LABS_DIR)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="lab numbers to run, e.g. --only 10 19")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    labs = sorted(glob.glob(os.path.join(LABS_DIR, "lab[0-9][0-9]_*.py")))
    if args.only:
        wanted = {f"lab{int(n):02d}_" for n in args.only}
        labs = [l for l in labs if any(os.path.basename(l).startswith(w) for w in wanted)]
    results = []
    for lab in labs:
        name = os.path.basename(lab)
        t0 = time.perf_counter()
        cmd = [sys.executable, lab] + (["--full"] if args.full else ["--quick"])
        print(f"\n>>> {' '.join(os.path.relpath(c, COURSE_DIR) if c.startswith('/') else c for c in cmd)}", flush=True)
        try:
            proc = subprocess.run(cmd, cwd=COURSE_DIR, capture_output=True, text=True, timeout=args.timeout)
            ok = proc.returncode == 0 and "✅ checks passed" in proc.stdout
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            ok, tail = False, ["TIMEOUT"]
        dt = time.perf_counter() - t0
        results.append((name, ok, dt))
        print("\n".join("    " + l for l in tail))
        print(f"    -> {'PASS' if ok else 'FAIL'} in {dt:.0f}s", flush=True)

    print("\n" + "=" * 60)
    for name, ok, dt in results:
        print(f"{'✅' if ok else '❌'} {name:<32} {dt:7.0f}s")
    n_ok = sum(ok for _, ok, _ in results)
    print(f"{n_ok}/{len(results)} labs passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
