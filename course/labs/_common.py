"""Shared helpers for every lab: argument parsing, checks, timing, figure saving.

Usage at the top of a lab:

    from _common import setup, check, banner, savefig, done
    args = setup("Lab 10: pretrain TinyLM")      # adds --quick / --full / --seed
"""
from __future__ import annotations

import argparse
import os
import sys
import time

LABS_DIR = os.path.dirname(os.path.abspath(__file__))
COURSE_DIR = os.path.dirname(LABS_DIR)
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

FIG_DIR = os.path.join(COURSE_DIR, "figures", "generated")
os.makedirs(FIG_DIR, exist_ok=True)

_checks: list[tuple[bool, str]] = []
_t0 = time.perf_counter()


def setup(title: str, extra: callable = None) -> argparse.Namespace:
    """Parse --quick/--full/--seed and print a banner."""
    p = argparse.ArgumentParser(description=title)
    p.add_argument("--quick", action="store_true", help="run in under ~1 minute")
    p.add_argument("--full", action="store_true", help="the real run (a few minutes)")
    p.add_argument("--seed", type=int, default=0)
    if extra is not None:
        extra(p)
    args = p.parse_args()
    if not args.full:
        args.quick = True                                  # quick is the default
    banner(title + ("  [quick]" if args.quick else "  [full]"))
    return args


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def section(text: str) -> None:
    print(f"\n--- {text} ---")


def check(cond: bool, msg: str) -> bool:
    """Record a check; prints ✅ or ❌ immediately."""
    _checks.append((bool(cond), msg))
    print(("✅ " if cond else "❌ ") + msg)
    return bool(cond)


def savefig(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"📊 saved {os.path.relpath(path, COURSE_DIR)}")
    return path


def elapsed() -> float:
    return time.perf_counter() - _t0


def done() -> None:
    """Summarise checks and exit non-zero if any failed."""
    n_ok = sum(ok for ok, _ in _checks)
    print(f"\n{n_ok}/{len(_checks)} checks passed in {elapsed():.1f}s")
    if n_ok == len(_checks):
        print("✅ checks passed")
    else:
        print("❌ some checks failed")
        sys.exit(1)


def plt():
    """matplotlib.pyplot with a non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    return _plt
