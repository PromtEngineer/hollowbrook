"""Chat template: how conversations become token sequences (Chapters 14–15).

TinyLM's format (one line per turn, shown with newlines for readability):

    <|bos|><|system|>You are helpful.<|end|>
    <|user|>What is 2 + 3?<|end|>
    <|assistant|>2 + 3 = 5<|end|>

Tool use adds two roles: the assistant emits ``<|tool_call|>{json}<|end|>`` and the
harness replies with ``<|tool_result|>...<|end|>``.

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


def render(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    """messages: [{"role": "user", "content": "..."}, ...] -> template string."""
    out = [BOS]
    for m in messages:
        out.append(ROLE_TOKENS[m["role"]] + m["content"] + END)
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
        role_id = tok.special_tokens[ROLE_TOKENS[m["role"]]]
        content_ids = tok.encode(m["content"], allowed_special=False)
        end_id = tok.special_tokens[END]
        trainable = 1 if m["role"] in ("assistant", "tool_call") else 0
        ids += [role_id] + content_ids + [end_id]
        # the role tag is given by the template for the first assistant turn, but the
        # model must learn to emit <|end|> — so the tag is masked and <|end|> is not.
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


TOOL_CALL_RE = re.compile(r"<\|tool_call\|>(.*?)<\|end\|>", re.S)


def parse_tool_call(text: str) -> Optional[dict]:
    """Extract {"name": ..., "arguments": {...}} from generated text, if present."""
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def ensure_chat_tokens(tok: BPETokenizer) -> BPETokenizer:
    for s in CHAT_SPECIAL_TOKENS:
        tok.add_special_token(s)
    return tok
