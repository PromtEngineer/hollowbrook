"""Distillation: teaching a small *student* model from a bigger *teacher* (Chapter 20).

Three recipes, from oldest to newest:

1. **Logit distillation** (Hinton, 2015) — ``kd_logit_loss``: on a fixed text, make the
   student's next-token distribution match the teacher's whole distribution (not just the
   one correct token). Off-policy: the text comes from a dataset.
2. **Sequence-level distillation / rejection-sampling SFT** — ``offline_distill``: sample
   answers from the teacher, keep the verified-correct ones, fine-tune the student on them.
   Still off-policy: the student is trained on text *it* would never have written.
3. **On-policy distillation** (OPD; Agarwal 2023 "GKD", Thinking Machines 2025, Qwen3) —
   ``on_policy_distill_step``: the *student* samples, the teacher grades every sampled token.
   The student is corrected on its own mistakes, exactly where it makes them.

Notation: π_s the student, π_t the teacher, y a sampled answer, t a token position.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from . import chat, tasks
from .generate import sample_group
from .model import TinyLM
from .rl import (_special, logprobs_from_logits, masked_mean, response_mask, split_completion,
                 token_logprobs)
from .tasks import TaskExample
from .tokenizer import BPETokenizer


# ----------------------------------------------------------- logit KD
def kd_logit_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor,
                  temperature: float = 1.0) -> Tensor:
    """Forward KL from the teacher to the student, averaged over masked positions.

        L = (1/Σ mask) Σ_t mask_t · τ² · Σ_v p_t(v) · [ log p_t(v) - log p_s(v) ]

    with p = softmax(logits / τ). Read: "for every position, the student must spread its
    probability over the vocabulary the same way the teacher does." A higher temperature τ
    softens both distributions so the student also learns which *wrong* tokens are nearly
    right ("dark knowledge"); τ² keeps the gradient scale independent of τ.

    student_logits, teacher_logits (B, T, V); mask (B, T). Zero iff the distributions match.
    """
    log_p_s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1)              # (B, T)
    return masked_mean(kl, mask) * temperature ** 2


# ---------------------------------------------------- on-policy distillation
@dataclass
class OPDConfig:
    steps: int = 40                 # on the nano model the reverse KL typically falls by ~2-3x within the first third of a run
    group_size: int = 4             # student samples per prompt
    prompts_per_step: int = 4
    max_new_tokens: int = 24
    temperature: float = 1.0
    lr: float = 2e-4
    grad_clip: float = 1.0
    adv_clip: float = 5.0           # clip |log π_t - log π_s| per token: a sharp teacher can otherwise
                                    # put a -13-nat "advantage" on one token and dominate the batch
    seed: int = 0
    log_every: int = 1


def on_policy_distill_step(student: TinyLM, teacher: TinyLM, tok: BPETokenizer,
                           examples: Sequence[TaskExample], cfg: OPDConfig,
                           optimizer: torch.optim.Optimizer) -> dict:
    """One OPD step: the student samples, the teacher grades each token, the student updates.

    The quantity we minimise is the *reverse* KL under the student's own samples,
        KL(π_s ‖ π_t) = E_{y~π_s} Σ_t [ log π_s(y_t) - log π_t(y_t) ],
    read as: "on the answers the student actually writes, how much more likely does the
    student find its own tokens than the teacher does?"

    We follow the Thinking Machines (2025) recipe and treat this like RL with a *per-token*
    reward: advantage A_t = log π_t(y_t) - log π_s(y_t) (positive where the teacher liked the
    token more than the student did), and the REINFORCE-style loss
        L = - (1/Σ mask) Σ_t mask_t · A_t · log π_s(y_t)          (A_t detached).
    Why not differentiate the sample of the KL directly? Because E_{y~π_s}[∇ log π_s(y)] = 0:
    the naive gradient of Σ_t (log π_s - log π_t) has zero mean and is pure noise. The
    score-function form above is the correct gradient of the reverse KL (Chapter 18's
    policy-gradient identity, with A_t in place of the reward).

    Compared with GRPO: no verifier is needed, every token gets a dense signal instead of one
    number per answer, and there is no group — but you need a teacher that is right.
    """
    pad_id, end_id, _ = _special(tok)
    ids_list, masks, texts = [], [], []
    student.eval()
    with torch.no_grad():
        for ex in examples:
            prompt_ids = chat.encode_chat(tok, ex.messages(with_answer=False))
            ids = sample_group(student, prompt_ids, cfg.group_size, cfg.max_new_tokens, cfg.temperature,
                               stop_ids=[end_id], pad_id=pad_id)                    # (G, T)
            P = len(prompt_ids)
            ids_list.append(ids)
            masks.append(response_mask(ids, P, pad_id, end_id))
            texts += [(ex, tok.decode(split_completion(row, P, pad_id, end_id))) for row in ids.tolist()]
    T = max(x.shape[1] for x in ids_list)
    ids = torch.cat([F.pad(x, (0, T - x.shape[1]), value=pad_id) for x in ids_list])    # (N, T)
    mask = torch.cat([F.pad(m, (0, T - 1 - m.shape[1]), value=0.0) for m in masks])     # (N, T-1)

    with torch.no_grad():
        logp_t = token_logprobs(teacher, ids)                     # teacher grades (N, T-1)
    student.train()
    logp_s = token_logprobs(student, ids)                         # student, with gradient
    raw_adv = (logp_t - logp_s).detach()                          # per-token advantage (unclipped)
    adv = raw_adv.clamp(-cfg.adv_clip, cfg.adv_clip) if cfg.adv_clip else raw_adv
    loss = -masked_mean(adv * logp_s, mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip).item()
    optimizer.step()
    student.eval()
    return {
        "loss": loss.item(),
        "reverse_kl": masked_mean(-raw_adv, mask).item(),         # true mean of log π_s - log π_t (unclipped)
        "accuracy": sum(tasks.verify(ex, txt) for ex, txt in texts) / len(texts),
        "resp_len": mask.sum(-1).mean().item(),
        "grad_norm": grad_norm,
        "n_tokens": int(mask.sum().item()),
    }


def opd_train(student: TinyLM, teacher: TinyLM, tok: BPETokenizer, examples: Sequence[TaskExample],
              cfg: OPDConfig, verbose: bool = True,
              on_step: Optional[Callable[[int, dict], None]] = None) -> list[dict]:
    """Run ``cfg.steps`` on-policy distillation steps, cycling through ``examples``."""
    torch.manual_seed(cfg.seed)
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    history = []
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        start = (step * cfg.prompts_per_step) % len(examples)
        batch = [examples[(start + i) % len(examples)] for i in range(cfg.prompts_per_step)]
        st = on_policy_distill_step(student, teacher, tok, batch, cfg, optimizer)
        st["step"] = step
        history.append(st)
        if verbose and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(f"opd step {step:4d} | reverse KL {st['reverse_kl']:.4f} | acc {st['accuracy']:.2f} | "
                  f"len {st['resp_len']:5.1f} | {time.perf_counter() - t0:.1f}s")
        if on_step is not None:
            on_step(step, st)
    return history


# ------------------------------------------------- sequence-level KD / RS-SFT
def sft_steps(model: TinyLM, tok: BPETokenizer, conversations: Sequence[list[dict]], steps: int,
              lr: float = 3e-4, batch_size: int = 8, seed: int = 0, grad_clip: float = 1.0) -> list[float]:
    """A 20-line masked-CE SFT loop (Chapter 15's ``llm.sft`` is the full version).

    Only assistant tokens are in the loss (``chat.build_sft_example`` builds that mask).
    """
    pad_id = tok.special_tokens["<|pad|>"]
    data = [chat.build_sft_example(tok, msgs, max_len=model.cfg.max_seq_len) for msgs in conversations]
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []
    model.train()
    from .optim import lr_at
    warmup = max(1, steps // 10)
    for step in range(steps):
        for grp in opt.param_groups:                       # warmup + cosine, like sft_train
            grp["lr"] = lr_at(step, steps, lr, warmup, "cosine")
        pick = torch.randint(0, len(data), (batch_size,), generator=g).tolist()
        x, y, m = chat.collate([data[i] for i in pick], pad_id)
        _, loss = model(x, y, loss_mask=m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


def offline_distill(student: TinyLM, teacher: TinyLM, tok: BPETokenizer, examples: Sequence[TaskExample],
                    n_samples: int = 4, sft_steps_n: int = 50, max_new_tokens: int = 24,
                    temperature: float = 1.0, lr: float = 3e-4, verbose: bool = True) -> dict:
    """Sequence-level KD, a.k.a. rejection-sampling SFT (Chapter 20, recipe 2).

    1. sample ``n_samples`` answers per prompt from the teacher,
    2. keep only the ones the verifier marks correct (the "rejection" step),
    3. SFT the student on the kept (prompt, answer) pairs.

    This is how most "distilled" open models (e.g. the DeepSeek-R1-Distill family) were made.
    Its weakness: the student learns from text it would never write itself, so it does not
    get corrected on *its own* failure modes — that is what OPD fixes.
    """
    pad_id, end_id, _ = _special(tok)
    teacher.eval()
    kept: list[list[dict]] = []
    n_total = 0
    with torch.no_grad():
        for ex in examples:
            prompt_ids = chat.encode_chat(tok, ex.messages(with_answer=False))
            ids = sample_group(teacher, prompt_ids, n_samples, max_new_tokens, temperature,
                               stop_ids=[end_id], pad_id=pad_id)
            for row in ids.tolist():
                text = tok.decode(split_completion(row, len(prompt_ids), pad_id, end_id))
                n_total += 1
                if tasks.verify(ex, text) >= 1.0:
                    msgs = ex.messages(with_answer=False) + [{"role": "assistant", "content": text}]
                    kept.append(msgs)
    if verbose:
        print(f"[offline_distill] kept {len(kept)}/{n_total} teacher samples")
    losses = sft_steps(student, tok, kept, sft_steps_n, lr=lr) if kept else []
    return {"n_kept": len(kept), "n_total": n_total, "keep_rate": len(kept) / max(1, n_total),
            "sft_losses": losses}
