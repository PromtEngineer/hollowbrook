"""Backends: where the next assistant message comes from (Chapter 24).

The agent loop (harness.py) does not care *which* model answers. It only needs one
function::

    complete(messages, tools, system) -> AssistantMessage

We provide three implementations:

* ``ScriptedBackend`` — replays a fixed list of replies. Used by every test and by
  the labs that show the loop step by step: deterministic, instant, no GPU.
* ``TinyLMBackend``  — the model you trained in Parts I–III, using the chat template
  from ``llm/chat.py`` (``<|tool_call|>{json}<|end|>`` for tool use).
* ``AnthropicBackend`` — a real API model. Imported lazily so the course runs without
  the ``anthropic`` package or an API key.

Our internal message format is a list of dicts, kept as small as possible:

    {"role": "user",        "content": "..."}
    {"role": "assistant",   "content": "...", "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool_result", "content": "...", "tool_call_id": "..."}

Each backend translates this format to and from its own wire format.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, Union


@dataclass
class ToolCall:
    """One request from the model: "run tool ``name`` with these ``arguments``".

    The ``id`` lets us pair the eventual result with the request when several tools
    are called in one turn.
    """
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(name=d["name"], arguments=d.get("arguments", {}) or {},
                   id=d.get("id") or "call_" + uuid.uuid4().hex[:8])


@dataclass
class AssistantMessage:
    """What the model said this turn: text, zero or more tool calls, bookkeeping."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: str = ""                                   # the untouched model output, for debugging
    usage: dict = field(default_factory=dict)       # e.g. {"input_tokens": .., "output_tokens": ..}

    def to_message(self) -> dict:
        """Convert to our internal message dict (what goes back into the context)."""
        m: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            m["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        return m


class Backend(Protocol):
    """Anything with a ``complete`` method is a backend (structural typing)."""

    def complete(self, messages: list[dict], tools: list[dict], system: str) -> AssistantMessage: ...


# ------------------------------------------------------------------ scripted
ScriptItem = Union[AssistantMessage, str, dict]


class ScriptedBackend:
    """Replays ``script`` in order. Each item may be:

    * an ``AssistantMessage``;
    * a ``str`` — a plain text reply, no tool calls;
    * a ``dict`` — ``{"text": ..., "tool_calls": [{"name": ..., "arguments": {...}}]}``.

    Every call is recorded in ``self.calls`` so tests can assert what the model saw.
    When the script runs out it returns a final "(script exhausted)" text so the loop
    always terminates.
    """

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        self.script = [self._coerce(item) for item in script]
        self.calls: list[dict] = []
        self.index = 0

    @staticmethod
    def _coerce(item: ScriptItem) -> AssistantMessage:
        if isinstance(item, AssistantMessage):
            return item
        if isinstance(item, str):
            return AssistantMessage(text=item)
        calls = [c if isinstance(c, ToolCall) else ToolCall.from_dict(c) for c in item.get("tool_calls", [])]
        return AssistantMessage(text=item.get("text", ""), tool_calls=calls)

    def complete(self, messages: list[dict], tools: list[dict], system: str) -> AssistantMessage:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "system": system})
        if self.index >= len(self.script):
            return AssistantMessage(text="(script exhausted)")
        msg = self.script[self.index]
        self.index += 1
        return msg


# -------------------------------------------------------------------- TinyLM
class TinyLMBackend:
    """Drive the course's own ``TinyLM`` through the chat template.

    Tool results become ``<|tool_result|>`` turns and the assistant's tool calls
    become ``<|tool_call|>{json}`` turns — the same layout the SFT data used in
    Chapter 15, so a model fine-tuned on tool traces (Chapter 21) can drive this loop.
    """

    def __init__(self, model, tok, max_new_tokens: int = 64, temperature: float = 0.0) -> None:
        self.model, self.tok = model, tok
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    @staticmethod
    def to_chat_messages(messages: list[dict], tools: list[dict], system: str) -> list[dict]:
        """Our internal dicts -> the roles ``llm.chat.render`` understands."""
        sys_text = system
        if tools:
            sys_text += "\nTools (call with <|tool_call|>{\"name\":..,\"arguments\":{..}}<|end|>):\n"
            sys_text += json.dumps(tools)
        out = [{"role": "system", "content": sys_text}]
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                if m.get("content"):
                    out.append({"role": "assistant", "content": m["content"]})
                for c in m["tool_calls"]:
                    out.append({"role": "tool_call",
                                "content": json.dumps({"name": c["name"], "arguments": c["arguments"]})})
            elif m["role"] == "tool_result":
                out.append({"role": "tool_result", "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def complete(self, messages: list[dict], tools: list[dict], system: str) -> AssistantMessage:
        from ..chat import encode_chat, parse_tool_call
        from ..generate import generate
        # encode_chat tokenises content with allowed_special=False: tool results or user
        # text containing "<|user|>" can never be mistaken for a real role tag.
        ids = encode_chat(self.tok, self.to_chat_messages(messages, tools, system), add_generation_prompt=True)
        # Keep <|end|> in the text so parse_tool_call can find "<|tool_call|>...<|end|>".
        raw = generate(self.model, self.tok, "", max_new_tokens=self.max_new_tokens,
                       temperature=self.temperature, stop=("<|eos|>",), prompt_ids=ids)
        # A small model may keep going after its turn and hallucinate a <|user|> turn;
        # only the part before any foreign role tag counts as *its* reply.
        own = re.split(r"<\|(?:user|system|tool_result)\|>", raw)[0]
        call = parse_tool_call(own)
        text = own.split("<|tool_call|>")[0].split("<|end|>")[0].strip()
        tool_calls = [ToolCall(call["name"], call.get("arguments", {}) or {})] if call and "name" in call else []
        return AssistantMessage(text=text, tool_calls=tool_calls, raw=raw)


# ----------------------------------------------------------------- Anthropic
class AnthropicBackend:
    """A real frontier model through the Anthropic Messages API.

    The translation is the interesting part for a learner: the API keeps tool calls
    and results as typed *content blocks* inside ordinary user/assistant messages,
    where we keep them as separate roles.
    """

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1024) -> None:
        try:
            import anthropic  # imported lazily: the course must run without it
        except ImportError as e:
            raise ImportError("AnthropicBackend needs `pip install anthropic`") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable to use AnthropicBackend")
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    @staticmethod
    def to_api_messages(messages: list[dict]) -> list[dict]:
        """Our dicts -> Anthropic format. Consecutive tool results share one user message."""
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]})
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool_result":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)      # several results for one turn
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    @staticmethod
    def from_api_response(resp) -> AssistantMessage:
        """Anthropic response -> AssistantMessage (text blocks + tool_use blocks)."""
        text, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input), id=block.id))
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return AssistantMessage(text="\n".join(text), tool_calls=calls, raw=str(resp.stop_reason), usage=usage)

    def complete(self, messages: list[dict], tools: list[dict], system: str) -> AssistantMessage:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            tools=tools,                                  # same {name, description, input_schema} shape
            messages=self.to_api_messages(messages),
        )
        return self.from_api_response(resp)
