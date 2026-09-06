"""MiniHarness: a coding agent that survives across context windows (Chapter 27).

A long task will not fit in one context window, or even one process. The trick
production harnesses use is to move the *state* out of the model and into files:

    PLAN.md      what we intend to do (written once, read every session)
    PROGRESS.md  what has been done and verified (appended every session)
    MEMORY.md    free-form notes the agent leaves for itself
    harness.log  every tool call, for the human (observability)

Each ``run_session`` starts a *fresh* ``Agent`` (empty context) that reads those files,
works for a bounded number of turns, and stops. Then the harness — not the model —
runs the tests. Tests are the ground truth; the model's "I'm done" is just a claim.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from .backends import Backend, ToolCall
from .harness import Agent, AgentConfig, Hooks, Transcript
from .tools import make_builtin_tools

PLAN_PROMPT = ("Read the repository (list_dir, read_file, search) and write a short numbered plan "
               "for this task. Reply with the plan as plain text.\n\nTask: {task}")
SESSION_SYSTEM = ("You are a coding agent working in a sandboxed repository. Follow PLAN.md, "
                  "check PROGRESS.md for what is already done, edit files with write_file, "
                  "run run_tests to verify, and finish with a one-paragraph summary of what you did.")


class MiniHarness:
    def __init__(self, backend: Backend, workdir: str, config: Optional[AgentConfig] = None,
                 extra_hooks: Optional[Hooks] = None) -> None:
        self.backend = backend
        self.extra_hooks = extra_hooks              # caller-supplied hooks, merged into _hooks()
        self.workdir = os.path.realpath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        cfg = config or AgentConfig()
        cfg.permission_policy = "allow_all"          # the sandbox + hooks are the gate here
        cfg.memory_path = cfg.memory_path or self._path("MEMORY.md")
        self.config = cfg
        self.tools = make_builtin_tools(self.workdir)
        self.sessions: list[Transcript] = []

    # ------------------------------------------------------------- files
    def _path(self, name: str) -> str:
        return os.path.join(self.workdir, name)

    def _read(self, name: str) -> str:
        p = self._path(name)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

    def _append(self, name: str, text: str) -> None:
        with open(self._path(name), "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")

    def has_plan(self) -> bool:
        return os.path.exists(self._path("PLAN.md"))

    # ------------------------------------------------------------- hooks
    def _hooks(self) -> Hooks:
        hooks = Hooks()
        if self.extra_hooks is not None:
            hooks.pre_tool += list(self.extra_hooks.pre_tool)
            hooks.post_tool += list(self.extra_hooks.post_tool)
            hooks.on_event += list(self.extra_hooks.on_event)

        def guard_writes(call: ToolCall) -> Optional[str]:
            # Rule 1: the plan is the human's contract; the model may not rewrite it.
            # Rule 2 (paths outside workdir) is enforced by the sandbox, but a hook
            # is where a harness author would add more rules — so we show one here.
            if call.name == "write_file":
                target = os.path.normpath(call.arguments.get("path", ""))
                if target == "PLAN.md":
                    return "PLAN.md is read-only for the agent; append notes to PROGRESS.md instead"
                if target.startswith("..") or os.path.isabs(target):
                    return f"path '{target}' is outside the workdir"
            return None

        def log(call: ToolCall, result: str) -> None:
            self._append("harness.log", json.dumps({"t": round(time.time(), 2), "tool": call.name,
                                                    "args": call.arguments, "result": result[:200]}))
            return None

        hooks.pre_tool.append(guard_writes)
        hooks.post_tool.append(log)
        return hooks

    # ------------------------------------------------------------- phases
    def plan(self, task: str) -> str:
        """First run only: a read-only agent writes PLAN.md, and PROGRESS.md is started."""
        planner = Agent(self.backend, self.tools.subset(["list_dir", "read_file", "search"]),
                        AgentConfig(max_turns=6, permission_policy="allow_read_only"),
                        system_prompt="You are a planner. Do not edit anything.")
        plan_text = planner.run(PLAN_PROMPT.format(task=task)).final_text
        with open(self._path("PLAN.md"), "w", encoding="utf-8") as f:
            f.write(f"# Plan\n\nTask: {task}\n\n{plan_text}\n")
        with open(self._path("PROGRESS.md"), "w", encoding="utf-8") as f:
            f.write("# Progress\n\n(no sessions yet)\n")
        return plan_text

    def run_session(self, max_turns: Optional[int] = None, task: Optional[str] = None) -> Transcript:
        """One bounded work session followed by verification (the harness runs the tests)."""
        if not self.has_plan():
            self.plan(task or "Make the tests pass.")
        prompt = ("PLAN.md:\n" + self._read("PLAN.md") + "\n\nPROGRESS.md:\n" + self._read("PROGRESS.md")
                  + "\n\nContinue the plan from where PROGRESS.md leaves off.")
        cfg = AgentConfig(**{**self.config.__dict__, "max_turns": max_turns or self.config.max_turns})
        agent = Agent(self.backend, self.tools, cfg, hooks=self._hooks(), system_prompt=SESSION_SYSTEM)
        t = agent.run(prompt)
        self.sessions.append(t)
        ok, report = self.verify()
        # Number sessions from the file, not from this object: a *new process* resuming on
        # the same directory (the whole point of PROGRESS.md) must continue the count.
        n = self._read("PROGRESS.md").count("\n## Session ") + 1
        self._append("PROGRESS.md", f"\n## Session {n} ({t.stop_reason}, {t.turns} turns, {t.tool_calls_made} tool calls)\n"
                                    f"{t.final_text.strip() or '(no summary)'}\n\n"
                                    f"Verification: {'PASS' if ok else 'FAIL'}\n```\n{report.strip()[-800:]}\n```")
        return t

    def resume(self, max_turns: Optional[int] = None) -> Transcript:
        """Continue from PROGRESS.md — identical to a new session on purpose: the files
        *are* the checkpoint, so resuming after a crash needs nothing special."""
        assert self.has_plan(), "nothing to resume: call plan()/run_session() first"
        return self.run_session(max_turns)

    def verify(self, path: str = "tests") -> tuple[bool, str]:
        """Ground truth: run the test-suite. Returns (passed, report)."""
        report = self.tools.call("run_tests", {"path": path})
        return report.startswith("PASS"), report

    def loop(self, task: str, max_sessions: int = 3, max_turns: Optional[int] = None) -> bool:
        """Sessions until the tests pass or we run out of budget."""
        for _ in range(max_sessions):
            self.run_session(max_turns, task=task)
            if self.verify()[0]:
                return True
        return False
