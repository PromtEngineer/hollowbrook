"""Turning a model's probabilities into text (Chapter 7).

Two layers of API:

* ``sample_next(logits, ...)`` — the sampling knobs (temperature, top-k, top-p, min-p).
* ``generate_ids(model, idx, ...)`` — the decode loop with a KV cache, batched.
* ``generate(model, tok, prompt, ...)`` — string in, string out.
* ``sample_group(model, prompt_ids, n, ...)`` — n samples from one prompt (used by RL).
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import TinyLM
from .tokenizer import BPETokenizer


# ------------------------------------------------------------------ sampling
@torch.no_grad()
def sample_next(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, min_p: Optional[float] = None,
                generator: Optional[torch.Generator] = None) -> Tensor:
    """Pick one next token per row of ``logits`` (B, V). Returns (B,) ids.

    temperature 0 -> greedy (argmax).
    top_k keeps only the k most likely tokens.
    top_p (nucleus) keeps the smallest set whose probability mass >= p.
    min_p drops tokens whose probability < min_p * (probability of the best token).
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k is not None and top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if min_p is not None and min_p > 0:
        probs = F.softmax(logits, dim=-1)
        limit = min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < limit, float("-inf"))
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # remove tokens after cumulative mass passes top_p (always keep the first)
        remove = cum - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def apply_repetition_penalty(logits: Tensor, idx: Tensor, penalty: float) -> Tensor:
    """Divide (positive) or multiply (negative) logits of tokens already in ``idx``."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for b in range(idx.shape[0]):
        seen = idx[b].unique()
        vals = logits[b, seen]
        logits[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


# --------------------------------------------------------------- decode loop
@torch.no_grad()
def generate_ids(model: TinyLM, idx: Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 min_p: Optional[float] = None, repetition_penalty: float = 1.0,
                 stop_ids: Sequence[int] = (), pad_id: Optional[int] = None,
                 use_cache: bool = True, seed: Optional[int] = None) -> Tensor:
    """Extend every row of ``idx`` (B, T0) by up to ``max_new_tokens`` tokens.

    Rows that emit a stop token are frozen (padded with ``pad_id``, or the stop token
    itself) while the rest of the batch continues. Returns (B, T0 + n_generated).
    """
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=idx.device).manual_seed(seed) if seed is not None else None
    B = idx.shape[0]
    stop = torch.tensor(list(stop_ids), device=idx.device, dtype=idx.dtype)
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
    cache = model.new_cache() if use_cache else None
    budget = model.cfg.max_seq_len - idx.shape[1]
    steps = min(max_new_tokens, budget)
    if steps <= 0:
        import warnings
        warnings.warn(f"prompt length {idx.shape[1]} leaves no room under max_seq_len={model.cfg.max_seq_len}; "
                      "nothing generated", stacklevel=2)
    cur = idx
    for _ in range(steps):
        if use_cache:
            logits, _ = model(cur, cache=cache)          # only the new tokens each step
        else:
            logits, _ = model(idx)                        # recompute everything (slow, for comparison)
        logits = logits[:, -1, :]                         # (B, V) — only the last position matters
        logits = apply_repetition_penalty(logits, idx, repetition_penalty)
        nxt = sample_next(logits, temperature, top_k, top_p, min_p, gen)
        if pad_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        cur = nxt[:, None]
        if stop.numel():
            finished |= torch.isin(nxt, stop)
            if bool(finished.all()):
                break
    if was_training:
        model.train()
    return idx


def generate(model: TinyLM, tok: BPETokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
             min_p: Optional[float] = None, repetition_penalty: float = 1.0,
             stop: Sequence[str] = ("<|eos|>", "<|end|>"), use_cache: bool = True,
             seed: Optional[int] = None, strip_prompt: bool = True,
             prompt_ids: Optional[Sequence[int]] = None) -> str:
    """String prompt -> string completion.

    Pass ``prompt_ids`` (e.g. from ``chat.encode_chat``) to bypass text encoding; chat
    code does this so user text can never smuggle in special tokens.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) if prompt_ids is not None else tok.encode(prompt)], device=device)
    stop_ids = [tok.special_tokens[s] for s in stop if s in tok.special_tokens]
    out = generate_ids(model, ids, max_new_tokens, temperature, top_k, top_p, min_p,
                       repetition_penalty, stop_ids=stop_ids, use_cache=use_cache, seed=seed)
    new_ids = out[0, ids.shape[1]:].tolist() if strip_prompt else out[0].tolist()
    # drop the stop token itself from the text
    new_ids = [i for i in new_ids if i not in stop_ids]
    return tok.decode(new_ids)


@torch.no_grad()
def sample_group(model: TinyLM, prompt_ids: Sequence[int], n: int, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
                 stop_ids: Sequence[int] = (), pad_id: Optional[int] = None,
                 seed: Optional[int] = None) -> Tensor:
    """``n`` independent samples for one prompt, as a (n, T0 + T_new) tensor.

    This is the rollout primitive for GRPO (Chapter 19): one prompt, a *group* of answers.
    Pass ``pad_id=tok.special_tokens["<|pad|>"]`` so finished rows are padded (without it,
    rows keep emitting tokens after their stop token).
    """
    device = next(model.parameters()).device
    idx = torch.tensor([list(prompt_ids)] * n, device=device)
    return generate_ids(model, idx, max_new_tokens, temperature, top_k, top_p,
                        stop_ids=stop_ids, pad_id=pad_id, seed=seed)


# ---------------------------------------------------------------- benchmark
def benchmark_decode(model: TinyLM, tok: BPETokenizer, prompt: str = "Once upon a time",
                     max_new_tokens: int = 64, repeats: int = 3) -> dict:
    """Tokens/second with and without the KV cache (Lab 7)."""
    out = {}
    for use_cache in (True, False):
        best = float("inf")
        for _ in range(repeats):                       # take the best of a few runs: less noise
            t0 = time.perf_counter()
            generate(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=1.0,
                     use_cache=use_cache, seed=0)
            best = min(best, time.perf_counter() - t0)
        out["cache" if use_cache else "no_cache"] = max_new_tokens / best
    return out
