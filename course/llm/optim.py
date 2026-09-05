"""Optimizers and learning-rate schedules.

Chapter 10 covers AdamW and Muon; this file contains a from-scratch Muon
(MomentUm Orthogonalized by Newton-Schulz), the optimizer used for Kimi K2,
GLM-5 and DeepSeek-V4 pretraining in 2025–2026.

Muon in one sentence: take the momentum-averaged gradient of each weight *matrix*,
replace it by the nearest orthogonal matrix (all singular values -> 1), and step in
that direction. Vectors (norm gains, biases) and embeddings still use AdamW.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


# ---------------------------------------------------------------------- Muon
def newton_schulz_orthogonalize(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate U V^T for G = U S V^T using a quintic Newton-Schulz iteration.

    The polynomial coefficients (from Keller Jordan's implementation) are tuned so
    the singular values converge to ~1 quickly rather than exactly, which is all an
    optimizer needs. Works in bfloat16 on GPU; we keep float32 on CPU for clarity.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)            # scale so the top singular value <= 1
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D weight matrices. Pass *only* matrix parameters to it.

    Parameters
    ----------
    params : matrix parameters (weights of linear layers)
    lr : learning rate; 0.02 is a good default (much larger than Adam's ~3e-4)
    momentum : Nesterov momentum coefficient
    ns_steps : Newton-Schulz iterations
    weight_decay : decoupled weight decay, as in AdamW
    """

    def __init__(self, params: Iterable[Tensor], lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon only handles 2-D weight matrices"
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = newton_schulz_orthogonalize(update, steps=group["ns_steps"])
                # Scale so the update RMS matches Adam's, letting Muon share an LR scale
                # across matrix shapes (from the Moonshot "Muon is Scalable" recipe).
                update *= math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


# --------------------------------------------------------------- param groups
def split_params(model: torch.nn.Module):
    """Split parameters into (matrix params for Muon, everything else for AdamW).

    Embeddings and the output head are 2-D but are *not* given to Muon: their rows
    are lookup entries, not a linear map, and orthogonalising them hurts.
    """
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2
        is_embedding = "embed" in name or "lm_head" in name
        if is_matrix and not is_embedding:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def build_optimizer(model: torch.nn.Module, kind: str = "adamw", lr: float = 3e-4,
                    weight_decay: float = 0.1, betas=(0.9, 0.95), muon_lr: float = 0.02):
    """Create an optimizer (or a list of two, for Muon + AdamW)."""
    if kind == "adamw":
        params = [p for p in model.parameters() if p.requires_grad]    # skip frozen weights (LoRA)
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        return [torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, betas=betas)]
    if kind == "muon":
        muon_params, adam_params = split_params(model)
        return [
            Muon(muon_params, lr=muon_lr, weight_decay=weight_decay),
            torch.optim.AdamW(adam_params, lr=lr, betas=betas, weight_decay=0.0),
        ]
    raise ValueError(f"unknown optimizer kind {kind!r}")


# ----------------------------------------------------------------- schedules
def lr_at(step: int, total_steps: int, peak_lr: float, warmup_steps: int = 0,
          kind: str = "cosine", min_ratio: float = 0.1, decay_frac: float = 0.2) -> float:
    """Learning-rate multiplier schedule.

    kind = "cosine": linear warmup, then cosine decay to ``min_ratio * peak_lr``.
    kind = "wsd":    Warmup–Stable–Decay: warmup, flat at peak, then linear decay to 0
                     over the last ``decay_frac`` of training (used by many 2025–26 runs
                     because you can branch a decayed checkpoint from any point).
    kind = "constant": warmup then flat.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if kind == "constant":
        return peak_lr
    if kind == "cosine":
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    if kind == "wsd":
        decay_start = int(total_steps * (1 - decay_frac))
        if step < decay_start:
            return peak_lr
        return peak_lr * max(0.0, (total_steps - step) / max(1, total_steps - decay_start))
    raise ValueError(f"unknown schedule {kind!r}")


def set_lr(optimizers, scale: float) -> None:
    """Multiply every optimizer's base LR (stored as 'base_lr') by ``scale``."""
    for opt in optimizers:
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
