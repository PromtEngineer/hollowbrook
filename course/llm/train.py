"""The pretraining loop (Chapter 10).

Everything a real training loop has, in ~150 lines: random-window batching, a warmup
+ decay learning-rate schedule, gradient accumulation, gradient clipping, AdamW or
Muon, periodic validation, checkpointing and resuming, and throughput logging.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import TinyLM
from .optim import build_optimizer, lr_at, set_lr


@dataclass
class TrainConfig:
    steps: int = 600
    batch_size: int = 32
    seq_len: int = 128
    grad_accum: int = 1               # effective batch = batch_size * grad_accum
    optimizer: str = "adamw"          # "adamw" | "muon"
    lr: float = 1e-3                  # AdamW peak LR (small models like a high LR)
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    warmup_steps: int = 50
    schedule: str = "cosine"          # "cosine" | "wsd" | "constant"
    min_lr_ratio: float = 0.1
    decay_frac: float = 0.2           # WSD only: fraction of training spent decaying (1.0 = pure anneal)
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 10
    log_every: int = 25
    seed: int = 0
    device: str = "cpu"
    ckpt_path: Optional[str] = None   # save here every eval_every steps if set
    dtype: str = "float32"            # "bfloat16" enables autocast on supported hardware

    def to_dict(self) -> dict:
        return asdict(self)


def get_batch(tokens: Tensor, batch_size: int, seq_len: int, generator: Optional[torch.Generator] = None,
              device: str = "cpu") -> tuple[Tensor, Tensor]:
    """Random windows from the packed token stream. y is x shifted one token left."""
    starts = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([tokens[s:s + seq_len] for s in starts])
    y = torch.stack([tokens[s + 1:s + 1 + seq_len] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: TinyLM, tokens: Tensor, batch_size: int, seq_len: int, n_batches: int = 10,
                  seed: int = 123, device: str = "cpu") -> float:
    """Average loss over a fixed set of random windows (same windows every call)."""
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(tokens, batch_size, seq_len, g, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


@dataclass
class History:
    step: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    tokens_per_sec: list[float] = field(default_factory=list)
    val_step: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def save_checkpoint(path: str, model: TinyLM, optimizers, step: int, cfg: TrainConfig, history: History) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(),
                "optimizers": [o.state_dict() for o in optimizers], "step": step,
                "train_config": cfg.to_dict(), "history": history.to_dict()}, path)


def load_checkpoint(path: str, model: TinyLM, optimizers=None) -> tuple[int, History]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    if optimizers is not None:
        for o, s in zip(optimizers, ckpt["optimizers"]):
            o.load_state_dict(s)
    hist = History(**ckpt["history"])
    return ckpt["step"], hist


def train(model: TinyLM, train_tokens: Tensor, val_tokens: Optional[Tensor], cfg: TrainConfig,
          on_step: Optional[Callable[[int, float], None]] = None, resume_from: Optional[str] = None,
          verbose: bool = True) -> History:
    """Train ``model`` in place and return the loss history."""
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model.to(device).train()
    optimizers = build_optimizer(model, cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                 muon_lr=cfg.muon_lr)
    history = History()
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step, history = load_checkpoint(resume_from, model, optimizers)
        if verbose:
            print(f"[train] resumed from {resume_from} at step {start_step}")
    g = torch.Generator().manual_seed(cfg.seed + start_step)
    use_autocast = cfg.dtype == "bfloat16"

    t0 = time.perf_counter()
    tokens_seen = 0
    for step in range(start_step, cfg.steps):
        # --- learning rate for this step ---------------------------------
        scale = lr_at(step, cfg.steps, 1.0, cfg.warmup_steps, cfg.schedule, cfg.min_lr_ratio, cfg.decay_frac)
        set_lr(optimizers, scale)

        # --- forward / backward (with gradient accumulation) --------------
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_tokens, cfg.batch_size, cfg.seq_len, g, device)
            with torch.autocast(device_type="cpu" if device == "cpu" else "cuda",
                                dtype=torch.bfloat16, enabled=use_autocast):
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_acc += loss.item() / cfg.grad_accum
            tokens_seen += x.numel()

        # --- clip, step ---------------------------------------------------
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()

        # --- logging ------------------------------------------------------
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            dt = time.perf_counter() - t0
            tps = tokens_seen / max(dt, 1e-9)
            history.step.append(step)
            history.train_loss.append(loss_acc)
            history.lr.append(scale * (cfg.lr if cfg.optimizer == "adamw" else cfg.muon_lr))
            history.tokens_per_sec.append(tps)
            if verbose:
                print(f"step {step:5d} | loss {loss_acc:.4f} | lr×{scale:.3f} | grad_norm {grad_norm:.2f} | {tps:,.0f} tok/s", flush=True)
        if on_step is not None:
            on_step(step, loss_acc)

        # --- validation & checkpoint --------------------------------------
        if val_tokens is not None and (step % cfg.eval_every == 0 or step == cfg.steps - 1) and step > 0:
            vl = estimate_loss(model, val_tokens, cfg.batch_size, cfg.seq_len, cfg.eval_batches, device=device)
            history.val_step.append(step)
            history.val_loss.append(vl)
            if verbose:
                print(f"   val loss {vl:.4f}  (perplexity {math.exp(vl):.1f})", flush=True)
            if cfg.ckpt_path:
                save_checkpoint(cfg.ckpt_path, model, optimizers, step + 1, cfg, history)
    model.eval()
    return history
