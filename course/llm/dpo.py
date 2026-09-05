"""Direct Preference Optimization and its cousins (Chapter 17).

RLHF's classic recipe trains a reward model, then runs RL against it. **DPO**
(Rafailov et al., 2023) skips both steps: it turns a preference pair directly into a
loss on the policy's own log-probabilities. The key observation is that the RL
objective "maximise reward minus beta * KL(policy || reference)" has a closed-form
optimum, and rearranging it expresses the reward as

    r(x, y) = beta * [ log pi(y | x) - log pi_ref(y | x) ]  (+ a term that cancels in pairs)

Plugging that reward into the Bradley–Terry preference model (``reward.bradley_terry_loss``)
gives the DPO loss below. The whole method is: two forward passes per pair (policy and
frozen reference), a subtraction, a sigmoid.

This file also contains three popular variants (IPO, SimPO, ORPO), a shared training
loop, and the batching helpers. Shapes: B batch (pairs), T sequence length.
Notation: ``pi`` is the policy being trained, ``pi_ref`` the frozen reference copy.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import TinyLM
from .optim import lr_at
from .reward import PrefHistory, PreferencePair, encode_response
from .tokenizer import BPETokenizer


# ============================================================ sequence log-prob
def sequence_logprob(model: TinyLM, ids: Tensor, response_mask: Tensor, average: bool = False) -> Tensor:
    """log pi(response | prompt): the summed log-probability of the response tokens.

    ids:           (B, T) prompt + response, right-padded.
    response_mask: (B, T) 1.0 where ids[b, t] is a response token (see ``reward.encode_response``).

    A causal LM at position t predicts token t+1, so the prediction targets are
    ``ids[:, 1:]`` and the logits that score them are ``logits[:, :-1]``. The mask must
    shift the same way: target ids[b, t+1] counts iff response_mask[b, t+1] == 1.

    With ``average=True`` returns the per-token mean instead (used by SimPO / ORPO).
    Returns (B,).
    """
    logits, _ = model(ids)                                             # (B, T, V)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)               # (B, T-1, V)
    targets = ids[:, 1:]                                               # (B, T-1)
    token_logp = logp.gather(-1, targets[..., None]).squeeze(-1)       # (B, T-1) log p of each actual token
    mask = response_mask[:, 1:].float()                                # (B, T-1) aligned with targets
    total = (token_logp * mask).sum(-1)                                # (B,)
    if average:
        return total / mask.sum(-1).clamp(min=1.0)
    return total


def response_lengths(response_mask: Tensor) -> Tensor:
    """Number of response tokens scored per row: (B,)."""
    return response_mask[:, 1:].float().sum(-1)


# ================================================================== the losses
def _pair_stats(chosen_reward: Tensor, rejected_reward: Tensor) -> dict[str, float]:
    """Numbers to log for any pairwise loss. 'accuracy' counts an exact tie as half a point,
    so an untrained policy (policy == reference, every margin 0) reads 0.5 = coin flip."""
    margin = (chosen_reward - rejected_reward).detach()
    acc = (margin > 0).float() + 0.5 * (margin == 0).float()
    return {"chosen_reward": chosen_reward.detach().mean().item(),
            "rejected_reward": rejected_reward.detach().mean().item(),
            "margin": margin.mean().item(), "accuracy": acc.mean().item()}


def dpo_loss(policy_chosen_lp: Tensor, policy_rejected_lp: Tensor,
             ref_chosen_lp: Tensor, ref_rejected_lp: Tensor, beta: float = 0.1) -> tuple[Tensor, dict]:
    """The DPO loss for a batch of pairs.

        loss = -log sigmoid( beta * [ (log pi(c) - log pi_ref(c)) - (log pi(r) - log pi_ref(r)) ] )

    Read this as: the *implicit reward* of an answer is how much more likely the policy
    makes it than the reference does (scaled by beta). We want the chosen answer's
    implicit reward to beat the rejected answer's, exactly like Bradley–Terry, and the
    sigmoid turns the reward gap into a probability of "chosen wins".

    Why the reference? Without it the loss would push log pi(chosen) to 0 and
    log pi(rejected) to -inf, wrecking the model's language ability. Measuring
    everything *relative* to pi_ref keeps the policy near where it started; beta
    (typically 0.05–0.5) sets how far it may drift — small beta = more freedom.

    Inputs are (B,) summed log-probs. Returns (scalar loss, stats dict) where stats
    holds chosen/rejected implicit rewards, their margin and pairwise accuracy.
    """
    chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)          # (B,)
    rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)    # (B,)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, _pair_stats(chosen_reward, rejected_reward)


def ipo_loss(policy_chosen_lp: Tensor, policy_rejected_lp: Tensor,
             ref_chosen_lp: Tensor, ref_rejected_lp: Tensor, beta: float = 0.1) -> tuple[Tensor, dict]:
    """IPO (Azar et al., 2023): a squared loss instead of the sigmoid.

        loss = ( [log-ratio(c) - log-ratio(r)] - 1 / (2 beta) )^2

    DPO keeps rewarding an ever-larger gap (the sigmoid never quite saturates), which
    can over-fit noisy pairs. IPO asks the gap to reach a fixed target 1/(2 beta) and
    then stop, which is more robust when labels are unreliable.
    """
    gap = (policy_chosen_lp - ref_chosen_lp) - (policy_rejected_lp - ref_rejected_lp)   # (B,)
    loss = ((gap - 1.0 / (2 * beta)) ** 2).mean()
    return loss, _pair_stats(beta * (policy_chosen_lp - ref_chosen_lp),
                             beta * (policy_rejected_lp - ref_rejected_lp))


def simpo_loss(policy_chosen_avg_lp: Tensor, policy_rejected_avg_lp: Tensor,
               beta: float = 2.0, gamma: float = 0.5) -> tuple[Tensor, dict]:
    """SimPO (Meng et al., 2024): no reference model, length-normalised, with a margin.

        loss = -log sigmoid( beta * [ avg log pi(c) - avg log pi(r) ] - gamma )

    The reward is the policy's *average per-token* log-prob of the answer — the same
    quantity greedy decoding optimises, so training and generation agree — and the
    margin gamma demands the chosen answer win by at least that much. Dropping the
    reference halves the compute and memory. Inputs are per-token averages (B,).
    """
    chosen_reward = beta * policy_chosen_avg_lp
    rejected_reward = beta * policy_rejected_avg_lp
    loss = -F.logsigmoid(chosen_reward - rejected_reward - gamma).mean()
    return loss, _pair_stats(chosen_reward, rejected_reward)


def orpo_loss(policy_chosen_avg_lp: Tensor, policy_rejected_avg_lp: Tensor,
              lam: float = 0.1) -> tuple[Tensor, dict]:
    """ORPO (Hong et al., 2024): SFT and preference learning in one loss, no reference.

        loss = NLL(chosen) + lambda * ( -log sigmoid( log odds(c) - log odds(r) ) )
        odds(y) = p(y) / (1 - p(y)),   p(y) = exp(average per-token log-prob of y)

    The first term is ordinary supervised fine-tuning on the chosen answer; the second
    is a penalty that pushes the *odds* of the rejected answer below the chosen one.
    Because it includes SFT, ORPO can run directly on a base model.
    """
    def log_odds(avg_lp: Tensor) -> Tensor:
        # log(p / (1 - p)) = log p - log(1 - p); clamp keeps log(1 - p) finite when p -> 1
        return avg_lp - torch.log1p(-torch.exp(avg_lp).clamp(max=1 - 1e-6))

    nll = -policy_chosen_avg_lp.mean()
    ratio = log_odds(policy_chosen_avg_lp) - log_odds(policy_rejected_avg_lp)
    loss = nll + lam * (-F.logsigmoid(ratio).mean())
    return loss, _pair_stats(log_odds(policy_chosen_avg_lp), log_odds(policy_rejected_avg_lp))


# ==================================================================== batching
def build_pair_batch(tok: BPETokenizer, pairs: Sequence[PreferencePair], pad_id: int,
                     max_len: Optional[int] = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Encode pairs into (chosen_ids, chosen_mask, rejected_ids, rejected_mask), each (B, T).

    All four tensors share the same T (the longest sequence in the batch) so the chosen
    and rejected halves can be stacked into one (2B, T) forward pass. Padding sits on
    the right; causal attention means pad tokens never influence earlier positions.
    """
    chosen = [encode_response(tok, p.prompt_messages, p.chosen, max_len) for p in pairs]
    rejected = [encode_response(tok, p.prompt_messages, p.rejected, max_len) for p in pairs]
    T = max(len(ids) for ids, _ in chosen + rejected)

    def stack(encoded):
        ids = torch.full((len(encoded), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), T), dtype=torch.float)
        for i, (row, m) in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, :len(m)] = torch.tensor(m, dtype=torch.float)
        return ids, mask

    c_ids, c_mask = stack(chosen)
    r_ids, r_mask = stack(rejected)
    return c_ids, c_mask, r_ids, r_mask


# ==================================================================== training
@dataclass
class DPOConfig:
    steps: int = 150
    batch_size: int = 8          # pairs per step
    lr: float = 2e-5             # DPO moves the policy fast; keep the LR well below SFT's
    beta: float = 0.1            # KL strength (DPO/IPO) or reward scale (SimPO)
    gamma: float = 0.5           # SimPO target margin
    lam: float = 0.1             # ORPO weight of the odds-ratio term
    warmup: int = 10
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 10
    seed: int = 0
    loss: str = "dpo"            # "dpo" | "ipo" | "simpo" | "orpo"
    max_len: Optional[int] = None   # default: the policy's max_seq_len

    def needs_reference(self) -> bool:
        return self.loss in ("dpo", "ipo")


def make_reference(policy: TinyLM) -> TinyLM:
    """A frozen copy of the policy: pi_ref never changes during training."""
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def pair_loss(policy: TinyLM, ref: Optional[TinyLM], batch: tuple[Tensor, Tensor, Tensor, Tensor],
              cfg: DPOConfig) -> tuple[Tensor, dict]:
    """One batch: run policy (and reference) on chosen + rejected, apply the chosen loss."""
    c_ids, c_mask, r_ids, r_mask = batch
    B = c_ids.shape[0]
    ids = torch.cat([c_ids, r_ids])                                    # (2B, T)
    mask = torch.cat([c_mask, r_mask])                                 # (2B, T)
    if cfg.loss in ("simpo", "orpo"):
        avg = sequence_logprob(policy, ids, mask, average=True)        # (2B,)
        if cfg.loss == "simpo":
            return simpo_loss(avg[:B], avg[B:], cfg.beta, cfg.gamma)
        return orpo_loss(avg[:B], avg[B:], cfg.lam)
    policy_lp = sequence_logprob(policy, ids, mask)                    # (2B,)
    with torch.no_grad():
        ref_lp = sequence_logprob(ref, ids, mask)                      # (2B,)
    fn = dpo_loss if cfg.loss == "dpo" else ipo_loss
    return fn(policy_lp[:B], policy_lp[B:], ref_lp[:B], ref_lp[B:], cfg.beta)


def dpo_train(policy: TinyLM, ref: Optional[TinyLM], tok: BPETokenizer, pairs: Sequence[PreferencePair],
              cfg: DPOConfig = DPOConfig(), verbose: bool = True) -> PrefHistory:
    """Train ``policy`` in place on preference pairs. ``ref=None`` makes a frozen copy.

    Each step: sample a mini-batch of pairs, pad, compute log pi and log pi_ref for the
    chosen and rejected answers, apply ``cfg.loss``, AdamW step. Logs loss, the implicit
    reward margin and pairwise accuracy (both should climb; the margin shows how far
    the policy has moved from the reference).
    """
    if cfg.loss not in ("dpo", "ipo", "simpo", "orpo"):
        raise ValueError(f"unknown loss {cfg.loss!r}")
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    if ref is None and cfg.needs_reference():
        ref = make_reference(policy)
    device = next(policy.parameters()).device
    pad_id = tok.special_tokens["<|pad|>"]
    max_len = cfg.max_len or policy.cfg.max_seq_len
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    history = PrefHistory()
    order = list(range(len(pairs)))
    pos = len(order)                       # forces a shuffle on the first step
    policy.train()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        lr = lr_at(step, cfg.steps, cfg.lr, cfg.warmup, "cosine")
        for g in opt.param_groups:
            g["lr"] = lr
        if pos >= len(order):
            rng.shuffle(order)
            pos = 0
        batch_pairs = [pairs[i] for i in order[pos:pos + cfg.batch_size]]
        pos += cfg.batch_size
        batch = tuple(t.to(device) for t in build_pair_batch(tok, batch_pairs, pad_id, max_len))
        loss, stats = pair_loss(policy, ref, batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history.step.append(step)
            history.train_loss.append(loss.item())
            history.lr.append(lr)
            history.accuracy.append(stats["accuracy"])
            history.margin.append(stats["margin"])
            history.tokens_per_sec.append(0.0)
            if verbose:
                print(f"[{cfg.loss}] step {step:4d} | loss {loss.item():.4f} | margin {stats['margin']:+.3f} | "
                      f"acc {stats['accuracy']:.2f} | {time.perf_counter() - t0:.1f}s")
    policy.eval()
    return history


@torch.no_grad()
def dpo_eval(policy: TinyLM, ref: Optional[TinyLM], tok: BPETokenizer, pairs: Sequence[PreferencePair],
             cfg: DPOConfig = DPOConfig(), batch_size: int = 16) -> dict[str, float]:
    """Average loss / margin / accuracy of ``cfg.loss`` over a set of pairs (no training)."""
    policy.eval()
    if ref is None and cfg.needs_reference():
        ref = make_reference(policy)
    device = next(policy.parameters()).device
    pad_id = tok.special_tokens["<|pad|>"]
    max_len = cfg.max_len or policy.cfg.max_seq_len
    totals = {"loss": 0.0, "margin": 0.0, "accuracy": 0.0}
    n = 0
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        batch = tuple(t.to(device) for t in build_pair_batch(tok, chunk, pad_id, max_len))
        loss, stats = pair_loss(policy, ref, batch, cfg)
        totals["loss"] += loss.item() * len(chunk)
        totals["margin"] += stats["margin"] * len(chunk)
        totals["accuracy"] += stats["accuracy"] * len(chunk)
        n += len(chunk)
    return {k: v / max(1, n) for k, v in totals.items()}
