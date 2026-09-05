"""Chat template: how conversations become token sequences (Chapters 14–15).

TinyLM's format (one line per turn, shown with newlines for readability):

    <|bos|><|system|>You are helpful.<|end|>
    <|user|>What is 2 + 3?<|end|>
    <|assistant|>2 + 3 = 5<|end|>

Tool use adds two roles. A tool call is an *assistant* turn whose content starts with
the ``<|tool_call|>`` marker, so the model can learn to emit it right after
``<|assistant|>``; the harness replies with a ``<|tool_result|>...<|end|>`` turn:

    <|assistant|><|tool_call|>{"name": "calc", "arguments": {"expression": "2+3"}}<|end|>
    <|tool_result|>5<|end|>
    <|assistant|>2 + 3 = 5<|end|>

The important idea is the **loss mask**: during SFT we only train on the assistant's
tokens (and its ``<|end|>``), never on the prompt.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence

import torch

from .tokenizer import BPETokenizer, CHAT_SPECIAL_TOKENS

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>",
               "tool_call": "<|tool_call|>", "tool_result": "<|tool_result|>"}
END = "<|end|>"
BOS = "<|bos|>"


def turn_prefix(role: str) -> str:
    """The tokens that open a turn of the given role."""
    if role not in ROLE_TOKENS:
        raise ValueError(f"unknown role {role!r}; expected one of {list(ROLE_TOKENS)}")
    if role == "tool_call":                      # a tool call is an assistant turn
        return ROLE_TOKENS["assistant"] + ROLE_TOKENS["tool_call"]
    return ROLE_TOKENS[role]


def render(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    """messages: [{"role": "user", "content": "..."}, ...] -> template string."""
    out = [BOS]
    for m in messages:
        out.append(turn_prefix(m["role"]) + m["content"] + END)
    if add_generation_prompt:
        out.append(ROLE_TOKENS["assistant"])
    return "".join(out)


def build_sft_example(tok: BPETokenizer, messages: Sequence[dict], max_len: Optional[int] = None
                      ) -> tuple[list[int], list[int]]:
    """Tokenize a conversation and build the loss mask.

    Returns (ids, mask) where mask[i] = 1 iff ids[i] is an assistant (or tool_call)
    token the model should learn to produce. Everything the model is *given*
    (system, user, tool results, role tags) is masked out.
    """
    ids = [tok.special_tokens[BOS]]
    mask = [0]
    for m in messages:
        role_id = tok.special_tokens[ROLE_TOKENS["assistant" if m["role"] == "tool_call" else m["role"]]]
        content_ids = tok.encode(m["content"], allowed_special=False)
        if m["role"] == "tool_call":             # the marker is part of what the model must produce
            content_ids = [tok.special_tokens[ROLE_TOKENS["tool_call"]]] + content_ids
        end_id = tok.special_tokens[END]
        trainable = 1 if m["role"] in ("assistant", "tool_call") else 0
        ids += [role_id] + content_ids + [end_id]
        # the role tag is given by the template (the harness writes <|assistant|>), but the
        # model must learn to emit the content and <|end|> — so the tag is masked, the rest is not.
        mask += [0] + [trainable] * len(content_ids) + [trainable]
    if max_len is not None:
        ids, mask = ids[:max_len], mask[:max_len]
    return ids, mask


def collate(examples: Sequence[tuple[list[int], list[int]]], pad_id: int):
    """Right-pad a list of (ids, mask) to a (B, T) batch: inputs, targets, loss_mask."""
    T = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), T), pad_id, dtype=torch.long)
    m = torch.zeros((len(examples), T), dtype=torch.float)
    for i, (ids, mask) in enumerate(examples):
        x[i, :len(ids)] = torch.tensor(ids)
        m[i, :len(mask)] = torch.tensor(mask, dtype=torch.float)
    # next-token prediction: input = x[:, :-1], target = x[:, 1:], mask aligned to targets
    return x[:, :-1], x[:, 1:], m[:, 1:]


TOOL_CALL_RE = re.compile(r"<\|tool_call\|>\s*(\{.*?\})\s*(?:<\|end\|>|$)", re.S)


def parse_tool_call(text: str) -> Optional[dict]:
    """Extract {"name": ..., "arguments": {...}} from generated text, if present.

    Works whether or not the trailing ``<|end|>`` is present (``generate`` strips
    stop tokens from its output).
    """
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return call if isinstance(call, dict) and "name" in call else None


def ensure_chat_tokens(tok: BPETokenizer) -> BPETokenizer:
    for s in CHAT_SPECIAL_TOKENS:
        tok.add_special_token(s)
    return tok
