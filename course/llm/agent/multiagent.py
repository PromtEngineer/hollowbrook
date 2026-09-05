"""Multi-agent patterns (Chapter 28), each in a few lines so the shape is visible.

* ``orchestrate``            — orchestrator–workers: split, fan out in parallel, merge.
* ``generator_evaluator_loop`` — one agent proposes, another judges, repeat until ACCEPT.
* ``debate``                 — several models answer, then see each other's answers.

All three are built from ordinary ``Agent`` runs: a "multi-agent system" is just a
program that calls agents. The interesting decisions are *what each one sees*
(context isolation) and *how their outputs are combined* (merging, judging).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .backends import Backend
from .harness import Agent, AgentConfig
from .tools import ToolRegistry


def _run(backend: Backend, tools: ToolRegistry, task: str, system: str, max_turns: int = 8) -> str:
    cfg = AgentConfig(max_turns=max_turns, permission_policy="allow_all")
    return Agent(backend, tools, cfg, system_prompt=system).run(task).final_text


def parse_json_list(text: str) -> Optional[list[str]]:
    """Find a JSON list of strings anywhere in ``text`` (models wrap them in prose)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x) for x in items] if isinstance(items, list) else None


def orchestrate(orchestrator_backend: Backend, worker_backend: Backend, task: str,
                tools: ToolRegistry, n_workers: int = 3,
                planner_fn: Optional[Callable[[str], list[str]]] = None) -> str:
    """Orchestrator–workers.

    1. The orchestrator splits ``task`` into at most ``n_workers`` subtasks (a JSON list).
       ``planner_fn`` can replace the model here (a fixed split for tests or labs).
    2. Each subtask runs as an independent worker agent — in parallel threads, each
       with an empty context (the fan-out is where the wall-clock savings come from).
    3. The orchestrator merges the worker reports into one answer.
    """
    if planner_fn is not None:
        subtasks = planner_fn(task)
    else:
        plan_text = _run(orchestrator_backend, ToolRegistry(),
                         f"Split this task into at most {n_workers} independent subtasks. "
                         f"Reply with ONLY a JSON list of strings.\n\nTask: {task}",
                         "You are an orchestrator that plans work for other agents.", max_turns=1)
        subtasks = parse_json_list(plan_text) or [task]
    subtasks = subtasks[:n_workers]

    def work(sub: str) -> str:
        return _run(worker_backend, tools, sub, "You are a worker agent. Do only your subtask and report the result.")

    with ThreadPoolExecutor(max_workers=max(1, len(subtasks))) as pool:
        reports = list(pool.map(work, subtasks))

    merged_input = "\n\n".join(f"Subtask: {s}\nReport: {r}" for s, r in zip(subtasks, reports))
    return _run(orchestrator_backend, ToolRegistry(),
                f"Original task: {task}\n\nWorker reports:\n{merged_input}\n\nWrite the final combined answer.",
                "You are an orchestrator that merges worker reports.", max_turns=1)


def generator_evaluator_loop(generator_backend: Backend, evaluator_backend: Backend, task: str,
                             tools: ToolRegistry, max_rounds: int = 3,
                             accept_fn: Optional[Callable[[str], bool]] = None) -> tuple[str, int]:
    """Generate -> evaluate -> regenerate with the critique, until "ACCEPT".

    The evaluator sees only the task and the candidate, never the generator's
    reasoning: a fresh pair of eyes is the point. ``accept_fn`` lets a *program*
    (tests, a checker) be the judge instead of, or in addition to, a model.
    Returns (best candidate, rounds used).
    """
    critique = ""
    candidate = ""
    for round_ in range(1, max_rounds + 1):
        prompt = task if not critique else f"{task}\n\nYour previous attempt:\n{candidate}\n\nReviewer feedback:\n{critique}\n\nProduce an improved answer."
        candidate = _run(generator_backend, tools, prompt, "You are the generator. Produce the best answer you can.")
        if accept_fn is not None and accept_fn(candidate):
            return candidate, round_
        verdict = _run(evaluator_backend, ToolRegistry(),
                       f"Task: {task}\n\nCandidate answer:\n{candidate}\n\n"
                       "Reply with exactly ACCEPT if it fully solves the task, otherwise a short critique.",
                       "You are a strict evaluator.", max_turns=1)
        if verdict.strip().upper().startswith("ACCEPT"):
            return candidate, round_
        critique = verdict
    return candidate, max_rounds


def debate(backends: list[Backend], question: str, rounds: int = 2) -> list[str]:
    """Each backend answers; in later rounds each sees the others' answers and may revise.
    Returns the final answers (one per backend). Useful mostly for showing that
    agreement is not the same as correctness."""
    answers = [_run(b, ToolRegistry(), question, "Answer concisely.", max_turns=1) for b in backends]
    for _ in range(rounds - 1):
        new = []
        for i, b in enumerate(backends):
            others = "\n".join(f"- Agent {j}: {a}" for j, a in enumerate(answers) if j != i)
            new.append(_run(b, ToolRegistry(),
                            f"{question}\n\nYour answer: {answers[i]}\nOther agents said:\n{others}\n"
                            "Give your final answer.", "Answer concisely.", max_turns=1))
        answers = new
    return answers
