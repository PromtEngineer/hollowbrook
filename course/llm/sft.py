"""Supervised fine-tuning, SFT (Chapter 15).

A pretrained model continues text. An *assistant* answers questions. SFT is the
bridge: we show the model a few thousand conversations in the chat template and
train it — with the same next-token loss as pretraining — but **only on the
assistant's tokens**. The prompt is context, not a target.

    <|bos|><|system|>You are TinyLM...<|end|><|user|>Write in capitals: kite<|end|><|assistant|>KITE<|end|>
    ─────────────────────── loss mask = 0 ────────────────────────────────────── ── mask = 1 ──

This file has three parts:

* ``build_sft_dataset`` / ``sft_train`` — the loop (≈ the pretraining loop with a mask);
* ``LoRALinear`` / ``apply_lora`` / ``merge_lora`` — parameter-efficient fine-tuning;
* ``respond`` — the "chat with the model" helper that evals and labs use.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .chat import END, build_sft_example, collate, render
from .generate import generate
from .model import TinyLM
from .optim import build_optimizer, lr_at, set_lr
from .tasks import SYSTEM_PROMPT, TASK_TYPES, TaskExample, make_examples
from .tokenizer import BPETokenizer
from .train import History


# ------------------------------------------------------------------- config
@dataclass
class SFTConfig:
    steps: int = 800                  # optimizer steps (ignored when ``epochs`` is set)
    batch_size: int = 16              # conversations per step
    lr: float = 1e-3                  # AdamW peak LR (tuned for nano/small; see note below)
    weight_decay: float = 0.0         # usually off for fine-tuning
    warmup_steps: int = 20
    schedule: str = "cosine"          # "cosine" | "wsd" | "constant" (see optim.lr_at)
    grad_clip: float = 1.0
    epochs: Optional[int] = None      # if set, steps = epochs * ceil(n_examples / batch_size)
    log_every: int = 25
    eval_every: int = 100
    eval_max_new_tokens: int = 16     # enough for every course task; keeps periodic evals fast
    system: Optional[str] = SYSTEM_PROMPT   # the system prompt rendered into every example (None = omit)
    seed: int = 0
    device: str = "cpu"
    max_len: int = 192                # truncate conversations longer than this many tokens
    lora_rank: int = 0                # 0 = full fine-tuning; > 0 = LoRA with this rank
    # Why these numbers: on the nano base model with 2000 examples of upper/reverse/add,
    # 300 steps at lr 1e-3 only learns the *format* (all-caps words, "a + b = c"), while
    # 800 steps reaches 100% on upper and reverse (add stays at 0 — carrying is too hard
    # for nano). Halving lr to 3e-4 learns nothing usable in 300 steps. Lab 15 measures this.


@dataclass
class SFTHistory(History):
    """The pretraining ``History`` plus exact-match accuracy on the validation examples."""
    val_acc: list[float] = field(default_factory=list)


# ------------------------------------------------------------------ dataset
def make_sft_examples(n: int, seed: int = 0, tasks: Sequence[str] = TASK_TYPES) -> list[TaskExample]:
    """``n`` instruction examples with verifiable answers (thin wrapper over ``tasks.make_examples``)."""
    return make_examples(n, seed=seed, tasks=tasks)


def build_sft_dataset(tok: BPETokenizer, examples: Sequence[TaskExample], max_len: Optional[int] = None,
                      system: Optional[str] = SYSTEM_PROMPT) -> list[tuple[list[int], list[int]]]:
    """Render every example through the chat template and tokenize it.

    Returns a list of ``(ids, mask)`` pairs. ``mask[i] == 1`` exactly when ``ids[i]``
    is a token the assistant produced (its answer and its closing ``<|end|>``);
    everything else — system prompt, user turn, role tags — is 0 and contributes
    nothing to the loss.
    """
    out = []
    n_cut = 0
    for ex in examples:
        # accept either a TaskExample or a raw conversation (list of {"role","content"} dicts),
        # so tool-use traces (Chapters 21, 27) can use the same training loop
        msgs = ex if isinstance(ex, (list, tuple)) else ex.messages(with_answer=True, system=system)
        ids, mask = build_sft_example(tok, msgs, max_len)
        if max_len is not None and sum(mask) == 0:
            n_cut += 1                                    # truncation removed every trainable token
        out.append((ids, mask))
    if n_cut:
        import warnings
        warnings.warn(f"{n_cut}/{len(out)} examples have no trainable tokens after truncation to "
                      f"max_len={max_len}; raise max_len (and the model's max_seq_len)", stacklevel=2)
    return out


def describe_mask(tok: BPETokenizer, ids: Sequence[int], mask: Sequence[int]) -> str:
    """Print-friendly view of one example: trained tokens in [brackets], the rest plain."""
    return "".join(f"[{tok.token_str(i)}]" if m else tok.token_str(i) for i, m in zip(ids, mask))


def _batches(data: list, batch_size: int, rng: random.Random):
    """Yield shuffled mini-batches forever (a new shuffle every epoch)."""
    order = list(range(len(data)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield [data[j] for j in order[i:i + batch_size]]


def sft_loss(model: TinyLM, batch: list[tuple[list[int], list[int]]], pad_id: int, device: str) -> Tensor:
    """The SFT loss for one batch: masked next-token cross-entropy.

    ``collate`` right-pads the batch and shifts it by one so that ``y[b, t]`` is the
    token after ``x[b, t]``. The mask ``m`` is aligned to ``y``, so a target is only
    counted when it is an assistant token (the model itself normalises by ``m.sum()``).
    """
    x, y, m = collate(batch, pad_id)                                 # each (B, T)
    _, loss = model(x.to(device), targets=y.to(device), loss_mask=m.to(device))
    return loss


@torch.no_grad()
def estimate_sft_loss(model: TinyLM, data: list, pad_id: int, batch_size: int = 16,
                      n_batches: int = 4, device: str = "cpu", seed: int = 123) -> float:
    """Masked loss over a fixed set of validation batches (same batches every call)."""
    was_training = model.training
    model.eval()
    gen = _batches(data, min(batch_size, len(data)), random.Random(seed))
    losses = [sft_loss(model, next(gen), pad_id, device).item() for _ in range(n_batches)]
    if was_training:
        model.train()
    return sum(losses) / len(losses)


# --------------------------------------------------------------- the loop
def sft_train(model: TinyLM, tok: BPETokenizer, examples: Sequence["TaskExample | list[dict]"], cfg: SFTConfig,
              val_examples: Optional[Sequence[TaskExample]] = None, verbose: bool = True) -> SFTHistory:
    """Fine-tune ``model`` in place on chat-formatted ``examples``; returns the history.

    Compared with ``train.train`` the only conceptual changes are (1) batches are whole
    conversations rather than random windows of a token stream, and (2) the loss is
    masked to assistant tokens. With ``cfg.lora_rank > 0`` the base weights are frozen
    and only small low-rank adapters train; they are merged back at the end so the
    returned model is an ordinary ``TinyLM``.
    """
    from .evals import eval_tasks   # local import: evals uses ``respond`` from this file

    torch.manual_seed(cfg.seed)
    device = cfg.device
    model.to(device).train()

    # Conversations longer than the RoPE tables would crash the forward pass, so
    # never build an example longer than the model can read.
    max_len = min(cfg.max_len, model.cfg.max_seq_len)
    data = build_sft_dataset(tok, examples, max_len, system=cfg.system)
    val_data = build_sft_dataset(tok, val_examples, max_len, system=cfg.system) if val_examples else None
    pad_id = tok.special_tokens["<|pad|>"]
    batch_size = min(cfg.batch_size, len(data))
    steps = cfg.steps if cfg.epochs is None else cfg.epochs * math.ceil(len(data) / batch_size)

    if cfg.lora_rank > 0:
        apply_lora(model, cfg.lora_rank, alpha=2 * cfg.lora_rank)
    optimizers = build_optimizer(model, "adamw", lr=cfg.lr, weight_decay=cfg.weight_decay)
    for opt in optimizers:                       # with LoRA, drop the frozen base weights
        for g in opt.param_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        print(f"[sft] {len(data)} examples | {steps} steps | batch {batch_size} | "
              f"{n_tr:,} trainable params" + (f" (LoRA rank {cfg.lora_rank})" if cfg.lora_rank else ""))

    history = SFTHistory()
    batches = _batches(data, batch_size, random.Random(cfg.seed))
    t0 = time.perf_counter()
    tokens_seen = 0
    for step in range(steps):
        scale = lr_at(step, steps, 1.0, cfg.warmup_steps, cfg.schedule)
        set_lr(optimizers, scale)

        batch = next(batches)
        loss = sft_loss(model, batch, pad_id, device)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        for opt in optimizers:
            opt.step()
        tokens_seen += sum(len(ids) for ids, _ in batch)

        if step % cfg.log_every == 0 or step == steps - 1:
            dt = time.perf_counter() - t0
            history.step.append(step)
            history.train_loss.append(loss.item())
            history.lr.append(scale * cfg.lr)
            history.tokens_per_sec.append(tokens_seen / max(dt, 1e-9))
            if verbose:
                print(f"step {step:5d} | loss {loss.item():.4f} | lr {scale * cfg.lr:.2e} | "
                      f"grad_norm {grad_norm:.2f} | {tokens_seen / max(dt, 1e-9):,.0f} tok/s")

        last = step == steps - 1
        if val_data is not None and ((step % cfg.eval_every == 0 and step > 0) or last):
            vl = estimate_sft_loss(model, val_data, pad_id, batch_size, device=device)
            res = eval_tasks(model, tok, val_examples, max_new_tokens=cfg.eval_max_new_tokens,
                             system=cfg.system)
            model.train()                        # generation switched the model to eval()
            history.val_step.append(step)
            history.val_loss.append(vl)
            history.val_acc.append(res.accuracy)
            if verbose:
                per_task = " ".join(f"{t}={a:.2f}" for t, a in sorted(res.per_task.items()))
                print(f"   val loss {vl:.4f} | accuracy {res.accuracy:.2f} ({per_task})")

    if cfg.lora_rank > 0:
        merge_lora(model)
    model.eval()
    return history


# --------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update:  y = W x + (B A x) · α/r.

    ``A`` is (r, d_in) and ``B`` is (d_out, r), so together they hold r·(d_in + d_out)
    numbers instead of d_in·d_out. ``B`` starts at zero, so the wrapped layer computes
    exactly what it did before — training moves it away from the base model gradually.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) / math.sqrt(d_in))   # (r, d_in)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))                    # (d_out, r)
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:                                     # x: (B, T, d_in)
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def merged_weight(self) -> Tensor:
        """W + (B A) · α/r — the single matrix the adapter is equivalent to."""
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)


LORA_TARGETS = ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def apply_lora(model: nn.Module, rank: int, alpha: Optional[float] = None,
               targets: Sequence[str] = LORA_TARGETS) -> nn.Module:
    """Freeze every parameter of ``model`` and wrap the named linear layers with LoRA.

    Returns the model (modified in place). Only the ``lora_A`` / ``lora_B`` tensors
    remain trainable afterwards. ``alpha`` defaults to ``2 * rank``.
    """
    alpha = 2 * rank if alpha is None else alpha
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        return model                                  # already adapted: idempotent, keep adapters trainable
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight and put the plain ``nn.Linear`` back.

    After this the model has the same parameter names as before ``apply_lora`` (so
    ``TinyLM.save`` / ``load`` work) and every parameter is trainable again.
    """
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight.copy_(child.merged_weight())
                setattr(module, name, child.base)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ respond
def respond(model: TinyLM, tok: BPETokenizer, user_text: str, system: Optional[str] = SYSTEM_PROMPT,
            max_new_tokens: int = 32, temperature: float = 0.0) -> str:
    """Ask the model one question and return the assistant's reply as text.

    The prompt is the chat template ending in ``<|assistant|>``; generation stops at
    ``<|end|>`` (or ``<|eos|>``, which a base model may still emit). Temperature 0 is
    greedy decoding — the right default for evaluation, where we want determinism.
    """
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": user_text})
    from .chat import encode_chat
    ids = encode_chat(tok, messages, add_generation_prompt=True)   # safe: no special-token injection
    return generate(model, tok, "", max_new_tokens=max_new_tokens, temperature=temperature,
                    stop=(END, "<|eos|>"), prompt_ids=ids).strip()
