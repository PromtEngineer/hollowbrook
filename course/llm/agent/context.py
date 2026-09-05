"""Context engineering: the window is a budget, spend it deliberately (Chapter 25).

An agent that runs for 50 turns produces far more text than any context window
holds. Three tools keep it working:

* ``ContextBudget`` — *measure* how full the window is (approximately).
* ``compact`` — *shrink* the history: shorten old tool results, optionally replace
  old turns by one summary, always keep the system prompt and the recent tail.
* ``MemoryFile`` — *persist* what must survive compaction (and process restarts)
  as plain text the agent writes to itself: "notes to my future self".

Everything here is model-agnostic; the harness calls it between turns.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English text.

    This is an approximation (real tokenizers vary by 20–30 %), but a budget only
    needs to be roughly right — we compact well before the true limit.
    """
    return max(1, len(text) // 4)


def message_text(m: dict) -> str:
    """Everything in a message that costs tokens, as one string."""
    parts = [str(m.get("content", ""))]
    if m.get("tool_calls"):
        parts.append(json.dumps(m["tool_calls"]))
    return "\n".join(parts)


class ContextBudget:
    """How much of the window is used, and whether it is time to compact."""

    def __init__(self, max_tokens: int, system: str = "", tools_text: str = "") -> None:
        self.max_tokens = max_tokens
        # the fixed cost every turn pays: system prompt + tool schemas
        self.fixed = estimate_tokens(system) + estimate_tokens(tools_text)
        self._last_used = self.fixed

    def used(self, messages: list[dict]) -> int:
        self._last_used = self.fixed + sum(estimate_tokens(message_text(m)) for m in messages)
        return self._last_used

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._last_used)

    def needs_compaction(self, messages: Optional[list[dict]] = None, threshold: float = 0.8) -> bool:
        """True when usage crosses ``threshold`` of the budget (not 100 %: leave room
        for the next reply and the next tool result)."""
        used = self.used(messages) if messages is not None else self._last_used
        return used >= threshold * self.max_tokens


def truncate_tool_result(text: str, max_chars: int = 2000) -> str:
    """Keep the head and the tail of a long tool result; the middle is usually
    the least informative part (long listings, repeated log lines)."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    dropped = len(text) - 2 * half
    return text[:half] + f"\n... [{dropped} chars truncated] ...\n" + text[-half:]


def compact(messages: list[dict], keep_last: int = 6,
            summarizer: Optional[Callable[[list[dict]], str]] = None) -> list[dict]:
    """Shrink a message list while keeping what matters most.

    * The first message is kept verbatim if it is the system prompt (role "system")
      — the harness usually passes the system prompt separately, in which case the
      first message is the user's task and is *also* kept: the agent must never
      forget what it was asked to do.
    * The last ``keep_last`` messages are kept verbatim (recent context is the
      most relevant).
    * Older tool results are replaced by a one-line stub.
    * If a ``summarizer`` is given, all older turns collapse into one summary
      message; otherwise they stay (with stubbed tool results).
    """
    if len(messages) <= keep_last + 1:
        return list(messages)
    head, middle, tail = messages[:1], messages[1:-keep_last], messages[-keep_last:]

    stubbed = []
    for m in middle:
        if m.get("role") == "tool_result":
            m = dict(m, content=f"[tool result truncated: {len(m.get('content', ''))} chars]")
        stubbed.append(m)

    if summarizer is not None and stubbed:
        summary = summarizer(middle)   # summarize the *full* older turns, not the stubs
        stubbed = [{"role": "user", "content": "[Summary of earlier work]\n" + summary}]
    return head + stubbed + tail


class MemoryFile:
    """Append-only notes the agent keeps *outside* the context window.

    Compaction deletes; a memory file remembers. Real harnesses use the same trick
    (CLAUDE.md, NOTES.md, scratchpads): the model writes down decisions and facts
    so a fresh context — or a fresh process — can read them back.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def read(self) -> str:
        if not os.path.exists(self.path):
            return ""
        with open(self.path, encoding="utf-8") as f:
            return f.read()

    def append(self, note: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(note.rstrip("\n") + "\n")

    def render_for_prompt(self, max_chars: int = 4000) -> str:
        """The block that goes into the system prompt (tail-truncated: newest notes win)."""
        text = self.read().strip()
        if not text:
            return ""
        if len(text) > max_chars:
            text = "..." + text[-max_chars:]
        return "## Memory (notes from earlier sessions)\n" + text
