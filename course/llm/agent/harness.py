"""The agent loop (Chapter 24) with the harness features of Chapters 25 and 27.

Read ``Agent.run`` top to bottom: it *is* the whole idea.

    messages = [task]
    repeat up to max_turns:
        reply = backend.complete(messages, tool schemas, system prompt)   # think
        if reply has no tool calls: stop                                   # done
        for each tool call:                                                # act
            pre-tool hooks  -> may block
            permission gate -> may deny
            registry.call   -> text (errors included)
            post-tool hooks -> may rewrite
            append the text as a tool_result message                       # observe
        if the context is nearly full: compact it

Everything else in this file is bookkeeping that makes the loop *safe* (permission
policies, hooks), *observable* (events, transcripts) and *long-running* (budget,
compaction, memory).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .backends import AssistantMessage, Backend, ToolCall
from .context import ContextBudget, MemoryFile, compact, truncate_tool_result
from .tools import Tool, ToolRegistry

DEFAULT_SYSTEM = (
    "You are a careful assistant that can call tools. Use a tool when it helps, read "
    "the result, and when you have the answer reply in plain text without calling a tool."
)

PermissionFn = Callable[[ToolCall, Tool], bool]


@dataclass
class AgentConfig:
    max_turns: int = 20
    permission_policy: str = "ask"          # "ask" | "allow_read_only" | "allow_all"
    context_budget_tokens: int = 8000
    compaction_threshold: float = 0.8
    compaction_keep_last: int = 6
    max_tool_result_chars: int = 2000
    memory_path: Optional[str] = None       # a MemoryFile the agent can read and append to
    verbose: bool = False
    summarizer: Optional[Callable[[list[dict]], str]] = None   # compaction: summarise old turns (Ch. 25)


@dataclass
class Event:
    """One line of the agent's observable history (what a tracer UI shows)."""
    kind: str        # assistant | tool_call | permission_denied | tool_result | compaction | hook | done | error | max_turns
    data: dict
    turn: int
    time: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"[turn {self.turn}] {self.kind}: {json.dumps(self.data, default=str)[:300]}"


class Hooks:
    """Points where the *harness owner* (not the model) injects behaviour.

    * ``pre_tool(call) -> None | str``: return a string to block the call with that reason.
    * ``post_tool(call, result) -> None | str``: return a string to replace the result
      (e.g. redact secrets, append a reminder).
    * ``on_event(event)``: logging, UIs, metrics.
    """

    def __init__(self) -> None:
        self.pre_tool: list[Callable[[ToolCall], Optional[str]]] = []
        self.post_tool: list[Callable[[ToolCall, str], Optional[str]]] = []
        self.on_event: list[Callable[[Event], None]] = []

    def run_pre(self, call: ToolCall) -> Optional[str]:
        for h in self.pre_tool:
            reason = h(call)
            if reason:
                return reason
        return None

    def run_post(self, call: ToolCall, result: str) -> str:
        for h in self.post_tool:
            new = h(call, result)
            if new is not None:
                result = new
        return result


@dataclass
class Transcript:
    """Everything that happened in one ``Agent.run``."""
    messages: list[dict] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    final_text: str = ""
    turns: int = 0
    tool_calls_made: int = 0
    stop_reason: str = ""      # "done" | "max_turns" | "error"

    def pretty(self) -> str:
        lines = []
        for m in self.messages:
            if m["role"] == "assistant":
                lines.append(f"ASSISTANT: {m['content']}")
                for c in m.get("tool_calls", []):
                    lines.append(f"  -> call {c['name']}({json.dumps(c['arguments'])})")
            elif m["role"] == "tool_result":
                lines.append(f"  <- result: {m['content'][:200]}")
            else:
                lines.append(f"{m['role'].upper()}: {m['content']}")
        lines.append(f"[{self.stop_reason} after {self.turns} turns, {self.tool_calls_made} tool calls]")
        return "\n".join(lines)


class Agent:
    """One model + one tool set + one policy = one agent."""

    def __init__(self, backend: Backend, tools: ToolRegistry, config: Optional[AgentConfig] = None,
                 hooks: Optional[Hooks] = None, system_prompt: str = DEFAULT_SYSTEM) -> None:
        self.backend = backend
        self.tools = tools
        self.config = config or AgentConfig()
        self.hooks = hooks or Hooks()
        self.system_prompt = system_prompt
        self.memory = MemoryFile(self.config.memory_path) if self.config.memory_path else None

    # ------------------------------------------------------------- helpers
    def _system(self) -> str:
        """System prompt + memory notes (memory is re-read every run: it may have grown)."""
        if self.memory is None:
            return self.system_prompt
        mem = self.memory.render_for_prompt()
        return self.system_prompt + ("\n\n" + mem if mem else "")

    def _emit(self, transcript: Transcript, kind: str, data: dict, turn: int) -> Event:
        ev = Event(kind, data, turn)
        transcript.events.append(ev)
        for h in self.hooks.on_event:
            h(ev)
        if self.config.verbose:
            print(ev)
        return ev

    def _permitted(self, call: ToolCall, tool: Tool, permission_fn: Optional[PermissionFn]) -> bool:
        """The permission gate. The policy answers first; "ask" defers to ``permission_fn``
        (a real harness shows a prompt here); with no function to ask, deny."""
        policy = self.config.permission_policy
        if policy == "allow_all":
            return True
        if policy == "allow_read_only" and tool.read_only:
            return True
        if permission_fn is not None:
            return bool(permission_fn(call, tool))
        return False

    # ---------------------------------------------------------------- loop
    def run(self, task: str, permission_fn: Optional[PermissionFn] = None) -> Transcript:
        t = Transcript(messages=[{"role": "user", "content": task}])
        system = self._system()
        schemas = self.tools.schemas()
        budget = ContextBudget(self.config.context_budget_tokens, system, json.dumps(schemas))

        for turn in range(1, self.config.max_turns + 1):
            t.turns = turn
            # ---- think: ask the model what to do next
            try:
                reply: AssistantMessage = self.backend.complete(t.messages, schemas, system)
            except Exception as e:  # noqa: BLE001
                self._emit(t, "error", {"error": f"{type(e).__name__}: {e}"}, turn)
                t.stop_reason = "error"
                return t
            t.messages.append(reply.to_message())
            self._emit(t, "assistant", {"text": reply.text, "n_tool_calls": len(reply.tool_calls)}, turn)

            # ---- stop condition: a reply with no tool calls is the final answer
            if not reply.tool_calls:
                t.final_text = reply.text
                t.stop_reason = "done"
                self._emit(t, "done", {"text": reply.text}, turn)
                return t

            # ---- act + observe: one tool_result message per call
            for call in reply.tool_calls:
                result = self._execute(t, call, permission_fn, turn)
                t.messages.append({"role": "tool_result", "tool_call_id": call.id, "content": result})

            # ---- housekeeping: keep the context under budget
            if budget.needs_compaction(t.messages, self.config.compaction_threshold):
                before = budget.used(t.messages)
                t.messages = compact(t.messages, keep_last=self.config.compaction_keep_last,
                                     summarizer=self.config.summarizer)
                after = budget.used(t.messages)
                self._emit(t, "compaction", {"tokens_before": before, "tokens_after": after}, turn)

        t.stop_reason = "max_turns"
        t.final_text = next((m["content"] for m in reversed(t.messages) if m["role"] == "assistant"), "")
        self._emit(t, "max_turns", {"max_turns": self.config.max_turns}, self.config.max_turns)
        return t

    def _execute(self, t: Transcript, call: ToolCall, permission_fn: Optional[PermissionFn], turn: int) -> str:
        """Hooks -> permission gate -> tool -> truncation -> hooks. Always returns text."""
        t.tool_calls_made += 1
        self._emit(t, "tool_call", {"id": call.id, "name": call.name, "arguments": call.arguments}, turn)

        reason = self.hooks.run_pre(call)
        if reason:
            self._emit(t, "hook", {"stage": "pre_tool", "blocked": True, "reason": reason}, turn)
            result = f"Blocked by hook: {reason}"
        else:
            tool = self.tools.get(call.name)
            if tool is None:
                result = self.tools.call(call.name, call.arguments)   # -> "Error: unknown tool"
            elif not self._permitted(call, tool, permission_fn):
                self._emit(t, "permission_denied", {"name": call.name, "read_only": tool.read_only}, turn)
                result = (f"Permission denied: '{call.name}' is not allowed under policy "
                          f"'{self.config.permission_policy}'. Try a different approach or ask the user.")
            else:
                result = self.tools.call(call.name, call.arguments)

        result = truncate_tool_result(result, self.config.max_tool_result_chars)
        result = self.hooks.run_post(call, result)
        self._emit(t, "tool_result", {"id": call.id, "name": call.name, "content": result[:500]}, turn)
        return result


def run_subagent(parent: Agent, task: str, tools_subset: Optional[list[str]] = None,
                 max_turns: int = 10, system_prompt: Optional[str] = None) -> str:
    """Run a fresh agent on ``task`` and return *only* its final text.

    This is "sub-agents as context isolation" (Chapter 25): the child does its
    reading and searching in its own window; the parent's window receives one
    paragraph, not fifty tool results. The child shares the backend, the hooks and
    the permission policy but starts with an empty context.
    """
    tools = parent.tools.subset(tools_subset) if tools_subset is not None else parent.tools
    cfg = AgentConfig(**{**parent.config.__dict__, "max_turns": max_turns})
    child = Agent(parent.backend, tools, cfg, hooks=parent.hooks,
                  system_prompt=system_prompt or parent.system_prompt)
    return child.run(task).final_text
